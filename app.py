from flask import Flask, request, jsonify, g, has_request_context
from flask_cors import CORS
import pandas as pd
import os
import requests
import io
import threading
import traceback
import time
import json
import math
import re
from datetime import datetime, timezone
from collections import defaultdict, deque

# OtoDeğer V10 decision-agent orchestration. The valuation engine remains
# isolated below; V10 only replaces the conversational assistant route.
from otodeger_v10_agent import V10_VERSION, ASSISTANT_BUILD, handle_v10_request
from otodeger_v10_state import get_state_service, StorageUnavailable
from otodeger_access_control import (
    AccessControlError,
    AuthenticationRequired,
    BusinessAccessRequired,
    ConcurrentRequestLimitReached,
    DeviceSeatLimitReached,
    SecurityConfigurationError,
    UsageLimitReached,
    get_access_manager,
)
from otodeger_fx import FXUnavailable, normalize_message_currency, conversion_note

# AI interpreter configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
COMMERCIAL_SHELL_VERSION = "13.0.5-premium-motion"

import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)

# Restrict browser origins and explicitly allow credentials/device identity.
# CORS is not an authentication boundary, but a wildcard origin is unnecessary
# exposure once paid account sessions exist.
_ALLOWED_ORIGINS = [
    value.strip()
    for value in str(
        os.environ.get(
            "ALLOWED_WEB_ORIGINS",
            "https://otodeger.online,https://www.otodeger.online,http://localhost:5173",
        )
    ).split(",")
    if value.strip()
]
CORS(
    app,
    origins=_ALLOWED_ORIGINS,
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Otodeger-Device"],
)

# Public API hardening. 128 KB is far above normal assistant requests while
# preventing oversized request bodies from consuming unnecessary resources.
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_REQUEST_BYTES", str(128 * 1024))
)

# =========================================================
# PUBLIC API SECURITY / COST CONTROLS
# =========================================================

ASSISTANT_MAX_MESSAGE_CHARS = int(
    os.environ.get("ASSISTANT_MAX_MESSAGE_CHARS", "1200")
)
ASSISTANT_RATE_10MIN = int(
    os.environ.get("ASSISTANT_RATE_10MIN", "20")
)
ASSISTANT_RATE_DAY = int(
    os.environ.get("ASSISTANT_RATE_DAY", "80")
)
SEARCH_RATE_10MIN = int(
    os.environ.get("SEARCH_RATE_10MIN", "40")
)
SEARCH_RATE_DAY = int(
    os.environ.get("SEARCH_RATE_DAY", "200")
)

# These caps count OpenAI HTTP calls, not user messages. Some complex Personal
# requests can make more than one model call, so the caps are intentionally
# separate from assistant-request limits.
OPENAI_IP_CALLS_HOUR = int(
    os.environ.get("OPENAI_IP_CALLS_HOUR", "40")
)
OPENAI_GLOBAL_CALLS_HOUR = int(
    os.environ.get("OPENAI_GLOBAL_CALLS_HOUR", "250")
)
OPENAI_GLOBAL_CALLS_DAY = int(
    os.environ.get("OPENAI_GLOBAL_CALLS_DAY", "1000")
)

PUBLIC_ASSISTANT_RESULT_CAP = int(
    os.environ.get("PUBLIC_ASSISTANT_RESULT_CAP", "20")
)
PUBLIC_SEARCH_RESULT_CAP = int(
    os.environ.get("PUBLIC_SEARCH_RESULT_CAP", "20")
)

# TEMPORARY TESTING SWITCH
# Set DISABLE_ASSISTANT_RATE_LIMITS=true on Render while only the owner is testing.
# IMPORTANT: set it back to false (or remove the env var) before public launch.
DISABLE_ASSISTANT_RATE_LIMITS = str(
    os.environ.get("DISABLE_ASSISTANT_RATE_LIMITS", "false")
).strip().casefold() in {"1", "true", "yes", "on"}

_RATE_LOCK = threading.Lock()
_RATE_BUCKETS = defaultdict(lambda: defaultdict(deque))


class AIUsageLimitExceeded(RuntimeError):
    pass


def _client_ip():
    """
    Render/proxy deployments normally provide X-Forwarded-For.
    Use the first address when present and fall back to Flask remote_addr.
    """
    forwarded = str(request.headers.get("X-Forwarded-For") or "").strip()
    if forwarded:
        candidate = forwarded.split(",")[0].strip()
        if candidate:
            return candidate[:128]
    return str(request.remote_addr or "unknown")[:128]


def _consume_rate(bucket, key, window_seconds, max_events):
    """
    Thread-safe in-memory limiter. This protects a single app process immediately.
    A shared Redis-backed limiter can replace it later if the deployment uses
    multiple independent workers/instances.
    """
    now = time.time()
    cutoff = now - float(window_seconds)
    key = str(key or "unknown")[:256]

    with _RATE_LOCK:
        dq = _RATE_BUCKETS[bucket][key]
        while dq and dq[0] <= cutoff:
            dq.popleft()

        if len(dq) >= int(max_events):
            retry_after = max(1, int(dq[0] + window_seconds - now))
            return False, retry_after

        dq.append(now)
        return True, 0


def _assistant_request_allowed():
    if DISABLE_ASSISTANT_RATE_LIMITS:
        return True, 0

    ip = _client_ip()

    ok, retry = _consume_rate(
        "assistant_10m", ip, 600, ASSISTANT_RATE_10MIN
    )
    if not ok:
        return False, retry

    ok, retry = _consume_rate(
        "assistant_day", ip, 86400, ASSISTANT_RATE_DAY
    )
    if not ok:
        return False, retry

    return True, 0


def _search_request_allowed():
    if DISABLE_ASSISTANT_RATE_LIMITS:
        return True, 0

    ip = _client_ip()

    ok, retry = _consume_rate(
        "search_10m", ip, 600, SEARCH_RATE_10MIN
    )
    if not ok:
        return False, retry

    ok, retry = _consume_rate(
        "search_day", ip, 86400, SEARCH_RATE_DAY
    )
    if not ok:
        return False, retry

    return True, 0


def _reserve_openai_call():
    """
    Hard server-side call ceilings. Client-supplied account tiers do not bypass
    these limits, so changing JSON to BUSINESS cannot defeat cost protection.

    During owner-only testing, DISABLE_ASSISTANT_RATE_LIMITS can temporarily
    bypass these ceilings. Re-enable before public launch.
    """
    if DISABLE_ASSISTANT_RATE_LIMITS:
        return

    ip = _client_ip()

    ok, retry = _consume_rate(
        "openai_ip_hour", ip, 3600, OPENAI_IP_CALLS_HOUR
    )
    if not ok:
        raise AIUsageLimitExceeded(
            f"OPENAI_IP_RATE_LIMIT:{retry}"
        )

    ok, retry = _consume_rate(
        "openai_global_hour", "GLOBAL", 3600, OPENAI_GLOBAL_CALLS_HOUR
    )
    if not ok:
        raise AIUsageLimitExceeded(
            f"OPENAI_GLOBAL_HOURLY_LIMIT:{retry}"
        )

    ok, retry = _consume_rate(
        "openai_global_day", "GLOBAL", 86400, OPENAI_GLOBAL_CALLS_DAY
    )
    if not ok:
        raise AIUsageLimitExceeded(
            f"OPENAI_GLOBAL_DAILY_LIMIT:{retry}"
        )

    # Shared commercial circuit breakers protect all Render workers and paid
    # accounts.  The legacy in-process ceilings above remain as an additional
    # emergency brake during migration.
    try:
        manager = get_access_manager()
        context = getattr(g, "otodeger_access_context", None) if has_request_context() else None
        if context is not None:
            manager.consume_ai_call_budget(context, client_ip=ip)
    except UsageLimitReached as exc:
        raise AIUsageLimitExceeded(
            f"COMMERCIAL_AI_LIMIT:{exc.retry_after}"
        ) from exc
    except SecurityConfigurationError as exc:
        raise AIUsageLimitExceeded("COMMERCIAL_SECURITY_UNAVAILABLE") from exc


def _openai_post(payload, timeout):
    """
    Single guarded path for every OpenAI Responses API call in this app.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    _reserve_openai_call()

    return requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )


def _looks_like_dataset_extraction_request(message):
    """
    Block clear bulk/export/reconstruction attempts before market search or
    OpenAI is called. Normal questions such as "show me more cars" are allowed.
    """
    low = str(message or "").strip().casefold()
    if not low:
        return False

    bulk_terms = re.search(
        r"\b(?:dump|export|download|extract|scrape|harvest|recreate|reconstruct|"
        r"entire|complete|full|raw|all rows?|every row|every listing|all listings|"
        r"database|dataset|data set|csv|json|sql|spreadsheet|table)\b",
        low,
        flags=re.IGNORECASE,
    )
    extraction_terms = re.search(
        r"\b(?:give|show|send|return|list|provide|output|print|enumerate|iterate|"
        r"copy|reveal|expose|recover|build|replicate|download|export)\b",
        low,
        flags=re.IGNORECASE,
    )
    systematic_terms = re.search(
        r"\b(?:every brand|every model|every year|all brands|all models|"
        r"all years|brand by brand|model by model|year by year|pagination|"
        r"next 100|next 1000|thousands of)\b",
        low,
        flags=re.IGNORECASE,
    )

    return bool(
        (bulk_terms and extraction_terms)
        or systematic_terms
    )


def _data_protection_answer(language):
    answers = {
        "TR": (
            "Toplu veri dışa aktarma veya veri tabanını yeniden oluşturabilecek "
            "istekleri desteklemiyorum. Belirli bir araç, bütçe, model, karşılaştırma "
            "veya piyasa kararı sorarsanız mevcut veriyi kullanarak yardımcı olabilirim."
        ),
        "EN": (
            "I can't provide bulk exports or information that could reconstruct the "
            "underlying dataset. Ask about a specific vehicle, budget, model, comparison "
            "or market decision and I can use the data to help with that."
        ),
        "RU": (
            "Я не предоставляю массовые выгрузки или данные, позволяющие восстановить "
            "исходную базу. Спросите о конкретном автомобиле, бюджете, модели, сравнении "
            "или рыночном решении — и я помогу на основе доступных данных."
        ),
    }
    return answers.get(language, answers["TR"])


def _fx_unavailable_answer(language):
    answers = {
        "TR": "Bütçenizi güncel kurla GBP'ye çevirmek için döviz verisine şu anda ulaşamıyorum. Eski bir kur kullanmak yerine biraz sonra tekrar denemenizi öneririm.",
        "EN": "I can't reach the current exchange-rate data needed to convert your budget to GBP right now. Rather than use a stale rate, please try again shortly.",
        "RU": "Сейчас не удаётся получить актуальный курс для пересчёта вашего бюджета в GBP. Я не буду использовать устаревший курс — попробуйте ещё раз немного позже.",
    }
    return answers.get(language, answers["TR"])


def _valuation_intent(message):
    """
    Route genuine own-car valuation questions to OtoDeğer's valuation flow.
    Do not intercept buyer questions such as "is this good value?".
    """
    low = str(message or "").strip().casefold()
    if not low:
        return False

    # A received offer is a conversational seller-evaluation task, even when the
    # same sentence also asks what the vehicle is worth. Do not intercept it into
    # the valuation handoff before the decision assistant can compare the offer.
    received_offer_patterns = [
        r"\b(?:someone|somebody|dealer|buyer).{0,20}offered\s+me\b",
        r"\bi(?:'ve| have)?\s+been\s+offered\b",
        r"\bi\s+(?:got|received|have)\s+(?:an?\s+)?offer\b",
        r"\boffer\s+on\s+my\s+(?:car|vehicle)\b",
        r"\bbana.{0,25}teklif\b", r"\bteklif\s+ald[ıi]m\b",
        r"\b(?:aracım|aracim|arabam).{0,25}teklif\b",
        r"\bмне\s+предложил", r"\bполучил[аи]?\s+предложение",
    ]
    if any(re.search(p, low, flags=re.IGNORECASE) for p in received_offer_patterns):
        return False

    patterns = [
        r"\b(?:aracım|arabam|aracim|arabamın|arabamin)\b.{0,35}\b(?:değer|deger|eder|kaç para|kac para|fiyat)\b",
        r"\b(?:aracımın|aracimin|arabamın|arabamin)\s+(?:değeri|degeri)\b",
        r"\b(?:araç|arac|araba)\s+değerleme\b",
        r"\b(?:how much is|what is)\s+my\s+(?:car|vehicle)\s+worth\b",
        r"\bhow much\s+(?:should|can|could)\s+i\s+(?:sell|list)\s+my\s+(?:car|vehicle)\s+for\b",
        # Natural owners usually name the vehicle instead of saying the word
        # "car": "How much should I sell my 2017 BMW 118i Sport for?".
        # Received-offer language has already been excluded above, so this is a
        # safe own-valuation handoff rather than an offer-evaluation intercept.
        r"\bhow much\s+(?:should|can|could)\s+i\s+(?:sell|list)\s+my\s+.{2,100}?\s+for\b",
        r"\bwhat\s+(?:should|could)\s+i\s+(?:sell|list)\s+my\s+(?:car|vehicle)\s+for\b",
        r"\bwhat\s+(?:should|could)\s+i\s+(?:sell|list)\s+my\s+.{2,100}?\s+for\b",
        r"\bwhat(?:'s| is)\s+a\s+good\s+(?:selling|listing)\s+price\s+for\s+my\s+(?:car|vehicle)\b",
        r"\bvalue\s+my\s+(?:car|vehicle)\b",
        r"\b(?:aracımı|aracimi|arabamı|arabami)\s+kaça\s+(?:satmalıyım|satmaliyim|satayım|satayim)\b",
        r"\b(?:aracımı|aracimi|arabamı|arabami)\s+ne\s+kadara\s+(?:satmalıyım|satmaliyim|satayım|satayim)\b",
        r"\bcar\s+valuation\b",
        r"\bvehicle\s+valuation\b",
        r"\bсколько\s+стоит\s+моя\s+машина\b",
        r"\bоцен(?:ить|ка)\s+(?:мою\s+)?машин",
    ]
    return any(re.search(p, low, flags=re.IGNORECASE) for p in patterns)


def _valuation_response(language):
    content = {
        "TR": (
            "Aracınızın güncel değerini özel değerleme aracımızla hesaplamak daha doğru olur. "
            "Yıl, marka, model ve versiyonu seçerek birkaç saniyede gerçek piyasa verilerine "
            "dayalı değer aralığını görebilirsiniz."
        ),
        "EN": (
            "For your car's current value, the best route is our dedicated valuation tool. "
            "Select the year, make, model and version and it will show a market-data-based "
            "value range in a few seconds."
        ),
        "RU": (
            "Для оценки текущей стоимости автомобиля лучше использовать наш специальный "
            "инструмент оценки. Выберите год, марку, модель и версию — и получите диапазон "
            "стоимости на основе рыночных данных за несколько секунд."
        ),
    }
    labels = {
        "TR": "Değerleme aracını aç",
        "EN": "Open valuation tool",
        "RU": "Открыть инструмент оценки",
    }

    return {
        "answer": content.get(language, content["TR"]),
        "actions": [{
            "type": "VALUATION",
            "label": labels.get(language, labels["TR"]),
            "url": "https://otodeger.online",
        }],
    }



def _looks_like_gibberish_message(message):
    """
    Catch obvious keyboard-smash / unusable input before the guided narrowing
    flow mistakes it for a broad vehicle-shopping request.

    This is deliberately conservative: normal short words, vehicle names,
    budgets and sentences are not blocked.
    """
    text = str(message or "").strip()
    if not text:
        return True

    # Numeric follow-ups need conversation context, including spaces and locale separators.
    if re.fullmatch(r"[\d.,\s'’]+", text):
        return False

    # Real market/search signals should always pass through.
    if re.search(r"£|\b\d{4}\b|\b\d+(?:[.,]\d+)?\s*(?:k|bin|gbp|pounds?|sterlin|km)\b", text, re.I):
        return False

    cleaned = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё]+", " ", text).strip()
    tokens = [t for t in cleaned.split() if t]

    if not tokens:
        return True

    # Only apply the keyboard-smash heuristic to very small inputs.
    if len(tokens) > 2:
        return False

    joined = "".join(tokens).casefold()
    if len(joined) < 6:
        return False

    vowels = set("aeiouyıöüâîûаеёиоуыэюя")
    vowel_ratio = sum(ch in vowels for ch in joined) / max(1, len(joined))

    # Examples: bgfbfgbfb, xzczxczxc. Keep the threshold conservative.
    return vowel_ratio < 0.16



def _expand_contextual_short_answer(message, conversation_history):
    """
    Resolve a bare numeric reply only when recent conversation makes its meaning
    unambiguous. This is a deterministic safety net around the semantic controller.

    Examples:
        EN: "What's your maximum budget?" -> "15.000" => £15,000
        TR: "Maksimum bütçeniz nedir?"     -> "15.000" => £15,000
        RU: "Какой максимальный бюджет?"   -> "15.000" => £15,000

    It also recognizes the immediately preceding user's budget-oriented request
    (e.g. "Bütçeme göre hangi araçları önerirsin?") if the assistant's wording
    was a broader narrowing prompt.
    """
    raw = str(message or "").strip()
    if not raw or not conversation_history:
        return raw

    # Currency-bearing replies already have enough semantic information.
    if re.search(
        r"£|\b(?:gbp|pounds?|sterlin|sterling|фунт(?:ов|а)?|фунт)\b",
        raw,
        re.I,
    ):
        return raw

    # Only reinterpret genuinely short numeric answers.
    if not re.fullmatch(r"\s*\d[\d.,\s]*\s*[kK]?\s*", raw):
        return raw

    previous_assistant = ""
    previous_user = ""
    for item in reversed(conversation_history or []):
        role = str(item.get("role") or "").casefold()
        value = str(item.get("text") or item.get("content") or "").strip()
        if not value:
            continue
        if role == "assistant" and not previous_assistant:
            previous_assistant = value
        elif role == "user" and not previous_user:
            previous_user = value
        if previous_assistant and previous_user:
            break

    # English, Turkish (including inflected forms), and Russian.
    assistant_asked_budget = bool(re.search(
        r"(?:maximum|max(?:imum)?)\s+budget|"
        r"(?:maksimum|azami)\s+bütçe\w*|"
        r"bütçe\w*.{0,25}(?:nedir|ne kadar|belirt|yaz)|"
        r"(?:максимальн\w*)\s+бюджет\w*|"
        r"бюджет\w*.{0,25}(?:какой|укажите|напишите)",
        previous_assistant,
        re.I,
    ))

    user_was_asking_budget_fit = bool(re.search(
        r"\b(?:my\s+budget|budget\b|"
        r"bütçe\w*|butce\w*|"
        r"бюджет\w*)",
        previous_user,
        re.I,
    ))

    if not assistant_asked_budget and not user_was_asking_budget_fit:
        return raw

    amount = _parse_human_number(raw)
    if amount is None:
        return raw

    # In a confirmed budget-answer context, "15" conventionally means £15k
    # for this market. 15.000/15,000 are already parsed as 15000.
    if re.fullmatch(r"\s*\d{1,3}\s*", raw) and amount < 1000:
        amount *= 1000

    if amount < 500 or amount > 500000:
        return raw

    return f"My maximum budget is £{int(round(amount)):,}"


def _unsupported_input_answer(language):
    answers = {
        "TR": (
            "Ne demek istediğinizi anlayamadım. Araç önerileri, model karşılaştırmaları, "
            "fiyat/piyasa analizi veya araç değerleme konusunda yardımcı olabilirim."
        ),
        "EN": (
            "I'm not sure what you mean. I can help with vehicle recommendations, model "
            "comparisons, pricing and market analysis, or valuing your car."
        ),
        "RU": (
            "Я не уверен, что понял запрос. Я могу помочь с подбором автомобиля, сравнением "
            "моделей, анализом цен и рынка или оценкой вашего автомобиля."
        ),
    }
    return answers.get(language, answers["TR"])


def _fallback_support_payload(language):
    suggestions = {
        "TR": [
            "Bütçeme göre hangi araçları önerirsin?",
            "İki modeli karşılaştırabilir misin?",
            "Aracımın değerini öğrenmek istiyorum",
        ],
        "EN": [
            "What cars fit my budget?",
            "Can you compare two models?",
            "I want to value my car",
        ],
        "RU": [
            "Какие автомобили подходят моему бюджету?",
            "Сравни две модели",
            "Хочу узнать стоимость моей машины",
        ],
    }
    instagram_labels = {
        "TR": "Aradığınız özellik yok mu? Instagram'dan bize yazın",
        "EN": "Can't find what you need? Message us on Instagram",
        "RU": "Не нашли нужную функцию? Напишите нам в Instagram",
    }
    return {
        "suggestions": suggestions.get(language, suggestions["TR"]),
        "actions": [{
            "type": "INSTAGRAM",
            "label": instagram_labels.get(language, instagram_labels["TR"]),
            # Frontend can replace this with the exact profile URL if desired.
            "url": "https://www.instagram.com/analist.kibris/",
        }],
    }


@app.after_request
def _security_response_headers(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


# -----------------------
# GOOGLE SHEETS CONNECTION
# -----------------------

GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

sheet = None


def initialize_lead_sheet():
    """Initialize Google Sheets only when credentials are configured.

    This keeps local/V10 development bootable without weakening production lead
    capture. Production should continue supplying GOOGLE_CREDENTIALS.
    """
    global sheet
    if not GOOGLE_CREDENTIALS:
        print("GOOGLE_CREDENTIALS not configured; lead capture is disabled.")
        sheet = None
        return
    try:
        credentials = Credentials.from_service_account_info(
            json.loads(GOOGLE_CREDENTIALS),
            scopes=SCOPES,
        )
        gc = gspread.authorize(credentials)
        sheet = gc.open("North Cyprus Vehicle Leads").sheet1
        print("Google Sheets lead capture ready")
    except Exception as exc:
        print("GOOGLE SHEETS INITIALIZATION FAILED:", exc)
        sheet = None


initialize_lead_sheet()

# -----------------------
# LOAD DATA (GITHUB CSV SOURCE - SAFE VERSION)
# -----------------------
_LEGACY_PUBLIC_DATA_BASE = "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
_DATA_SECURITY_ENFORCED = str(os.environ.get("COMMERCIAL_SECURITY_MODE", "legacy")).strip().casefold() == "enforced"

def _configured_data_url(env_name, legacy_filename):
    configured = str(os.environ.get(env_name) or "").strip()
    if configured:
        return configured
    # Paid launch must not silently publish/rely on the proprietary raw GitHub
    # source.  Legacy owner testing keeps the existing fallback until private
    # object storage/repo-local deployment is configured.
    if _DATA_SECURITY_ENFORCED:
        return ""
    return _LEGACY_PUBLIC_DATA_BASE + legacy_filename


CSV_URL = _configured_data_url("VALUATION_CSV_URL", "ads_base.csv")

df = pd.DataFrame()  # safe default

# 🔥 ADDED: readiness flag
DATA_READY = False

def load_data():
    global df, DATA_READY
    try:
        local_candidates = [
            str(os.environ.get("VALUATION_CSV_PATH") or "").strip(),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ads_base.csv"),
        ]
        local_path = next((path for path in local_candidates if path and os.path.exists(path)), None)
        if local_path:
            df = pd.read_csv(local_path)
        else:
            if not str(CSV_URL or "").strip():
                raise RuntimeError("PRIVATE_DATA_SOURCE_REQUIRED:ads_base.csv")
            r = requests.get(CSV_URL, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        DATA_READY = True
        print("CSV loaded successfully")
    except Exception as e:
        print("CSV LOAD FAILED:", e)
        df = pd.DataFrame()
        DATA_READY = False

load_data()

# -----------------------
# 🔥 ADDED: AUTO REFRESH EVERY 12 HOURS
# -----------------------
def refresh_data_loop():
    while True:
        time.sleep(12 * 60 * 60)  # 12 hours
        print("Refreshing CSV data from GitHub...")
        load_data()

threading.Thread(target=refresh_data_loop, daemon=True).start()

# -----------------------
# TYPE CLEANING
# -----------------------
def safe_prepare_dataframe():
    global df

    if df is None or df.empty:
        return

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["KM"] = pd.to_numeric(df["KM"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df["Brand"] = df["Brand"].astype(str).str.strip()
    df["Model"] = df["Model"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

safe_prepare_dataframe()

# =========================================================
# AI BUYING ASSISTANT - CURRENT MARKET DATA
# =========================================================

MARKET_CSV_URL = _configured_data_url("MARKET_CSV_URL", "market_base.csv")

market_df = pd.DataFrame()
MARKET_READY = False

# =========================================================
# AI BUYING ASSISTANT - BUYER INTELLIGENCE v1
# =========================================================
BUYER_MODEL_CSV_URL = _configured_data_url("BUYER_MODEL_CSV_URL", "buyer_model_intelligence.csv")
BUYER_CATEGORY_CSV_URL = _configured_data_url("BUYER_CATEGORY_CSV_URL", "buyer_category_intelligence.csv")

buyer_model_df = pd.DataFrame()
buyer_category_df = pd.DataFrame()
BUYER_INTELLIGENCE_READY = False

# Stable model-level buyer profiles keep DISCOVER deterministic and fast.
# These are generated offline by build_buyer_model_profiles_v1.py and committed
# beside the other production intelligence CSVs.
BUYER_MODEL_PROFILE_CSV_URL = _configured_data_url("BUYER_MODEL_PROFILE_CSV_URL", "buyer_model_profiles.csv")
model_profile_df = pd.DataFrame()
MODEL_PROFILE_READY = False
MODEL_PROFILE_LOOKUP = {}
ASSISTANT_PROFILE_VERSION = "1.0"


# =========================================================
# AI ASSISTANT - BUSINESS INTELLIGENCE v1.5
# =========================================================
# These files are generated offline by build_business_intelligence_v1.py
# and committed beside the other production intelligence CSVs.
BUSINESS_STOCK_CSV_URL = _configured_data_url("BUSINESS_STOCK_CSV_URL", "business_stock_intelligence.csv")
BUSINESS_COMPANY_CSV_URL = _configured_data_url("BUSINESS_COMPANY_CSV_URL", "business_company_intelligence.csv")
BUSINESS_MARKET_CSV_URL = _configured_data_url("BUSINESS_MARKET_CSV_URL", "business_market_intelligence.csv")
BUSINESS_ACTIVITY_CSV_URL = _configured_data_url("BUSINESS_ACTIVITY_CSV_URL", "business_company_activity_daily.csv")
# Detailed activity needs the actual daily observations, not the compact valuation
# snapshot.  This source is optional for backwards compatibility but should be
# configured in production so vehicle-level period reports can reconcile exactly.
BUSINESS_ACTIVITY_HISTORY_CSV_URL = _configured_data_url("BUSINESS_ACTIVITY_HISTORY_CSV_URL", "kktcarabam_daily.csv")

business_stock_df = pd.DataFrame()
business_company_df = pd.DataFrame()
business_market_df = pd.DataFrame()
business_activity_df = pd.DataFrame()
business_activity_history_df = pd.DataFrame()
BUSINESS_INTELLIGENCE_READY = False
BUSINESS_ACTIVITY_READY = False
BUSINESS_ACTIVITY_HISTORY_READY = False
BUSINESS_INTELLIGENCE_VERSION = "1.5"
BUSINESS_ACTIVITY_VERSION = "11.0"


def _load_assistant_csv_local_first(filename, url, timeout=25):
    """Prefer repo-local intelligence files; fall back to GitHub if absent.

    This removes an unnecessary network dependency on normal production boots while
    preserving the existing refresh/fallback behaviour.
    """
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(local_path):
        return pd.read_csv(local_path, low_memory=False)
    if not str(url or "").strip():
        raise RuntimeError(f"PRIVATE_DATA_SOURCE_REQUIRED:{filename}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), low_memory=False)


def _prepare_buyer_intelligence_frame(frame):
    frame = frame.copy()

    text_cols = [
        "VehicleType", "Brand", "Model", "CategoryDetail",
        "LiquidityEvidenceLevel", "PricePressureEvidenceLevel",
        "Buyer_LiquidityEvidenceConfidence",
        "Buyer_PricePressureEvidenceConfidence",
        "RecommendationGranularity",
    ]
    for col in text_cols:
        if col in frame.columns:
            frame[col] = (
                frame[col].fillna("").astype(str).str.strip()
            )

    numeric_cols = [
        "Year", "CurrentListings", "CurrentStartingPrice",
        "CurrentMedianPrice", "CurrentHighestPrice", "CurrentMedianKM",
        "GalleryListings", "PrivateListings",
        "Buyer_HistoricalDistinctListings",
        "Buyer_MedianObservedDaysToExit",
        "Buyer_Exit30EligibleListings",
        "Buyer_ObservedExitWithin30DaysRate",
        "Buyer_Exit60EligibleListings",
        "Buyer_ObservedExitWithin60DaysRate",
        "Buyer_Exit90EligibleListings",
        "Buyer_ObservedExitWithin90DaysRate",
        "Buyer_PricePressureEligibleListings",
        "Buyer_PriceReductionRate",
        "Buyer_MedianReductionPctAmongReduced",
        "Buyer_ListingVolumeRankWithinVehicleType",
        "Buyer_CategoryListingVolumeRankWithinModel",
        "Buyer_ModelListingVolumeRankWithinVehicleType",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


def load_buyer_intelligence():
    global buyer_model_df, buyer_category_df, BUYER_INTELLIGENCE_READY

    try:
        model_r = requests.get(BUYER_MODEL_CSV_URL, timeout=20)
        model_r.raise_for_status()
        category_r = requests.get(BUYER_CATEGORY_CSV_URL, timeout=20)
        category_r.raise_for_status()

        new_model = pd.read_csv(
            io.StringIO(model_r.text), low_memory=False
        )
        new_category = pd.read_csv(
            io.StringIO(category_r.text), low_memory=False
        )

        required_model = {
            "VehicleType", "Brand", "Model", "Year",
            "CurrentListings", "CurrentStartingPrice",
            "Buyer_ObservedExitWithin60DaysRate",
            "Buyer_LiquidityEvidenceConfidence",
            "LiquidityEvidenceLevel",
        }
        required_category = {
            "VehicleType", "Brand", "Model", "CategoryDetail", "Year",
            "CurrentListings", "CurrentStartingPrice",
            "Buyer_ObservedExitWithin60DaysRate",
            "Buyer_LiquidityEvidenceConfidence",
            "LiquidityEvidenceLevel",
        }

        missing_model = required_model - set(new_model.columns)
        missing_category = required_category - set(new_category.columns)
        if missing_model or missing_category:
            raise ValueError(
                "Buyer Intelligence schema mismatch. "
                f"model missing={sorted(missing_model)}, "
                f"category missing={sorted(missing_category)}"
            )

        buyer_model_df = _prepare_buyer_intelligence_frame(new_model)
        buyer_category_df = _prepare_buyer_intelligence_frame(new_category)
        BUYER_INTELLIGENCE_READY = True

        print(
            "Buyer Intelligence loaded successfully: "
            f"{len(buyer_model_df)} model-year rows, "
            f"{len(buyer_category_df)} category-year rows"
        )

    except Exception as e:
        print("BUYER INTELLIGENCE LOAD FAILED:", e)

        # Keep the last successful intelligence snapshot alive.
        if (
            buyer_model_df is None or buyer_model_df.empty
            or buyer_category_df is None or buyer_category_df.empty
        ):
            buyer_model_df = pd.DataFrame()
            buyer_category_df = pd.DataFrame()
            BUYER_INTELLIGENCE_READY = False



def _prepare_business_frame(frame):
    frame = frame.copy()

    text_cols = [
        "Link", "Company", "VehicleType", "Brand", "Model", "CategoryDetail",
        "Location", "Transmission", "Fuel", "Color", "Image",
        "PublicListingAgeDefinition", "BenchmarkSource",
        "ComparableEvidenceConfidence", "PricePositionBand",
        "StockAgeBand", "AttentionLevel", "AttentionReasons",
        "BusinessGranularity",
        "HistoricalBenchmarkSourceLiquidity",
        "HistoricalBenchmarkSourcePricePressure",
        "LiquidityEvidenceConfidence",
        "PricePressureEvidenceConfidence",
        "EvidenceQuality",
        "TurnoverSignal", "CurrentSupplySignal", "PricePressureSignal",
        "AcquisitionSignal", "AcquisitionReasons",
        "ObservedExitDefinition",
    ]
    for col in text_cols:
        if col in frame.columns:
            frame[col] = frame[col].fillna("").astype(str).str.strip()

    numeric_cols = [
        "Year", "KM", "CurrentAskingPrice", "StockAgeDays",
        "FirstObservedPrice", "HistoricalLatestObservedPrice",
        "PriceChangeAmount", "PriceChangePct",
        "HistoricalMedianObservedDaysToExit",
        "HistoricalP25ObservedDaysToExit",
        "HistoricalP75ObservedDaysToExit",
        "ObservedExit30Rate", "ObservedExit60Rate", "ObservedExit90Rate",
        "HistoricalPriceReductionRate",
        "HistoricalMedianReductionPctAmongReduced",
        "ComparableListings", "ComparableMedianPrice",
        "ComparableP25Price", "ComparableP75Price",
        "ComparableMinPrice", "ComparableMaxPrice",
        "ComparableMedianKM", "KMVsComparableMedian",
        "PriceVsMedianPct", "PricePercentile",
        "AgeVsMarketMedianDays", "AgeVsMarketMedianPct",
        "CurrentStockCount", "CurrentStockAskingValue",
        "MedianCurrentAskingPrice", "MedianPublicListingAgeDays",
        "P75PublicListingAgeDays",
        "FreshStockCount", "NormalStockCount", "AboveTypicalAgeCount",
        "AgedStockCount", "VeryAgedStockCount",
        "LowPricePositionCount", "MidMarketPriceCount",
        "HighPricePositionCount", "InsufficientPriceComparisonCount",
        "WatchStockCount", "AttentionStockCount", "HighAttentionStockCount",
        "DistinctBrands", "DistinctModels", "HistoricalDistinctListings",
        "HistoricalObservedMarketExits", "HistoricalPriceReductionEligibleListings",
        "HistoricalPriceReductionListings",
        "CurrentListings", "CurrentStartingPrice", "CurrentMedianPrice",
        "CurrentHighestPrice", "GalleryListings", "PrivateListings",
        "DistinctCompanies", "LiquidityHistoricalDistinctListings",
        "PricePressureHistoricalDistinctListings", "Exit60EligibleListings",
        "PricePressureEligibleListings", "MedianObservedDaysToExit",
        "ObservedExitWithin60DaysRate", "PriceReductionRate",
        "TurnoverPercentile", "RawSupplyScarcityPercentile",
        "CurrentMarketOpportunityPercentile",
        "PricePressureAttractivenessPercentile",
        "EvidencePriorityModifier", "OpportunityIndexInternal",
        "ConfidenceAdjustedOpportunityIndex", "OpportunityPercentile",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "PublicListingAgeIsLowerBound" in frame.columns:
        frame["PublicListingAgeIsLowerBound"] = (
            frame["PublicListingAgeIsLowerBound"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"true", "1", "yes"})
        )

    return frame


def load_business_intelligence():
    global business_stock_df, business_company_df, business_market_df
    global BUSINESS_INTELLIGENCE_READY

    try:
        new_stock = _load_assistant_csv_local_first(
            "business_stock_intelligence.csv", BUSINESS_STOCK_CSV_URL, timeout=25
        )
        new_company = _load_assistant_csv_local_first(
            "business_company_intelligence.csv", BUSINESS_COMPANY_CSV_URL, timeout=25
        )
        new_market = _load_assistant_csv_local_first(
            "business_market_intelligence.csv", BUSINESS_MARKET_CSV_URL, timeout=25
        )

        required_stock = {
            "Link", "Company", "Brand", "Model", "Year",
            "CurrentAskingPrice", "StockAgeDays",
            "PricePositionBand", "AttentionLevel",
        }
        required_company = {
            "Company", "CurrentStockCount", "HighAttentionStockCount",
            "VeryAgedStockCount", "HighPricePositionCount",
        }
        required_market = {
            "BusinessGranularity", "VehicleType", "Brand", "Model", "Year",
            "CurrentListings", "CurrentStartingPrice", "CurrentMedianPrice",
            "ObservedExitWithin60DaysRate", "PriceReductionRate",
            "EvidenceQuality", "OpportunityPercentile",
            "AcquisitionSignal", "AcquisitionReasons",
        }

        missing_stock = required_stock - set(new_stock.columns)
        missing_company = required_company - set(new_company.columns)
        missing_market = required_market - set(new_market.columns)

        if missing_stock or missing_company or missing_market:
            raise ValueError(
                "Business Intelligence schema mismatch. "
                f"stock missing={sorted(missing_stock)}, "
                f"company missing={sorted(missing_company)}, "
                f"market missing={sorted(missing_market)}"
            )

        business_stock_df = _prepare_business_frame(new_stock)
        # Business v1.5 stock output predates fuel segmentation. The live market
        # is already loaded before Business intelligence, so enrich by listing link
        # without changing any historical/price semantics. Future builders may
        # include Fuel directly; in that case we keep the existing value.
        if "Fuel" not in business_stock_df.columns and market_df is not None and not market_df.empty and {"Link","Fuel"}.issubset(market_df.columns):
            fuel_lookup=(market_df[["Link","Fuel"]].copy()
                         .drop_duplicates("Link",keep="last"))
            fuel_lookup["Link"]=fuel_lookup["Link"].fillna("").astype(str).str.strip()
            fuel_lookup["Fuel"]=fuel_lookup["Fuel"].fillna("").astype(str).str.strip()
            business_stock_df=business_stock_df.merge(fuel_lookup,on="Link",how="left")
        elif "Fuel" in business_stock_df.columns:
            business_stock_df["Fuel"]=business_stock_df["Fuel"].fillna("").astype(str).str.strip()
        business_company_df = _prepare_business_frame(new_company)
        business_market_df = _prepare_business_frame(new_market)
        BUSINESS_INTELLIGENCE_READY = True

        print(
            "Business Intelligence loaded successfully: "
            f"{len(business_stock_df)} stock rows, "
            f"{len(business_company_df)} companies, "
            f"{len(business_market_df)} market rows"
        )

    except Exception as e:
        print("BUSINESS INTELLIGENCE LOAD FAILED:", e)

        # Preserve the last successful snapshot if there is one.
        if (
            business_stock_df is None or business_stock_df.empty
            or business_company_df is None or business_company_df.empty
            or business_market_df is None or business_market_df.empty
        ):
            business_stock_df = pd.DataFrame()
            business_company_df = pd.DataFrame()
            business_market_df = pd.DataFrame()
            BUSINESS_INTELLIGENCE_READY = False



def load_business_activity():
    global business_activity_df, BUSINESS_ACTIVITY_READY
    try:
        frame = _load_assistant_csv_local_first(
            "business_company_activity_daily.csv", BUSINESS_ACTIVITY_CSV_URL, timeout=25
        )
        required = {
            "Date", "Company", "OpeningObservedStockCount",
            "ClosingObservedStockCount", "NetObservedStockChange",
            "NewlyObservedListings", "ObservedMarketExits",
            "AskingPriceReductions", "AskingPriceIncreases",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Business activity schema mismatch: {sorted(missing)}")
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Company"] = frame["Company"].fillna("").astype(str).str.strip()
        for col in required - {"Date", "Company"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        business_activity_df = frame[frame["Date"].notna() & (frame["Company"] != "")].copy()
        BUSINESS_ACTIVITY_READY = not business_activity_df.empty
        print(f"Business activity loaded successfully: {len(business_activity_df)} company-day rows")
    except Exception as exc:
        print("BUSINESS ACTIVITY LOAD FAILED:", exc)
        if business_activity_df is None or business_activity_df.empty:
            business_activity_df = pd.DataFrame()
            BUSINESS_ACTIVITY_READY = False



def load_business_activity_history():
    """Load raw daily listing observations for exact Business activity detail.

    The aggregate company-day file remains authoritative for counts.  This raw
    history supplies the pounds and vehicle identities behind those counts.
    """
    global business_activity_history_df, BUSINESS_ACTIVITY_HISTORY_READY
    try:
        frame = _load_assistant_csv_local_first(
            "kktcarabam_daily.csv", BUSINESS_ACTIVITY_HISTORY_CSV_URL, timeout=45
        )
        lower = {str(col).strip().casefold(): col for col in frame.columns}
        aliases = {
            "DATE": ["date", "tarih"],
            "Company": ["company", "seller", "dealer", "firma"],
            "Link": ["link", "url"],
            "Price": ["price", "fiyat"],
        }
        rename = {}
        for canonical, candidates in aliases.items():
            if canonical in frame.columns:
                continue
            found = next((lower.get(name.casefold()) for name in candidates if lower.get(name.casefold())), None)
            if found:
                rename[found] = canonical
        if rename:
            frame = frame.rename(columns=rename)
        required = {"DATE", "Company", "Link", "Price"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Business activity history schema mismatch: {sorted(missing)}")
        frame = frame.copy()
        frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
        frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
        frame["Company"] = frame["Company"].fillna("").astype(str).str.strip()
        frame["Link"] = frame["Link"].fillna("").astype(str).str.strip()
        for col in ["Year", "KM"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in ["Brand", "Model", "Category"]:
            if col in frame.columns:
                frame[col] = frame[col].fillna("").astype(str).str.strip()
        # Keep listing observations even when an asking price is temporarily
        # missing. They remain essential for exact stock/add/exit counts; the
        # activity report can estimate only the missing monetary component.
        frame = frame[
            frame["DATE"].notna()
            & frame["Company"].ne("")
            & frame["Link"].ne("")
        ].copy()
        business_activity_history_df = frame
        BUSINESS_ACTIVITY_HISTORY_READY = not frame.empty
        # Stock Purchase uses this same raw history for exact Year + Brand + Model +
        # Category turnover evidence.  Never retain a cache built from an older
        # history snapshot after the daily source refreshes.
        globals()["_GUIDED_STOCK_HISTORY_CACHE"] = None
        print(f"Business activity history loaded successfully: {len(frame)} daily listing rows")
    except Exception as exc:
        print("BUSINESS ACTIVITY HISTORY LOAD FAILED:", exc)
        if business_activity_history_df is None or business_activity_history_df.empty:
            business_activity_history_df = pd.DataFrame()
            BUSINESS_ACTIVITY_HISTORY_READY = False

def load_model_profiles():
    global model_profile_df, MODEL_PROFILE_READY, MODEL_PROFILE_LOOKUP

    try:
        r = requests.get(BUYER_MODEL_PROFILE_CSV_URL, timeout=15)
        r.raise_for_status()
        new_profiles = pd.read_csv(io.StringIO(r.text), low_memory=False).fillna("")
        required = {
            "Brand", "Model", "VehicleType", "BodyStyle", "SizeClass",
            "Economy", "Luxury", "Comfort", "Performance", "Practicality",
            "Family", "Commute", "Confidence"
        }
        missing = required - set(new_profiles.columns)
        if missing:
            raise ValueError(f"buyer_model_profiles.csv missing columns: {sorted(missing)}")

        for col in required:
            new_profiles[col] = new_profiles[col].fillna("").astype(str).str.strip()

        lookup = {}
        for row in new_profiles.to_dict("records"):
            brand = str(row.get("Brand") or "").strip()
            model = str(row.get("Model") or "").strip()
            if not brand or not model:
                continue
            lookup[(brand.casefold(), model.casefold())] = row

        model_profile_df = new_profiles
        MODEL_PROFILE_LOOKUP = lookup
        MODEL_PROFILE_READY = bool(lookup)
        print(f"Model Buyer Profiles loaded successfully: {len(lookup)} model families")
    except Exception as e:
        print("MODEL BUYER PROFILE LOAD FAILED:", e)
        # Preserve the last successful snapshot if one already exists.
        if not MODEL_PROFILE_LOOKUP:
            model_profile_df = pd.DataFrame()
            MODEL_PROFILE_READY = False


def _profile_level(value):
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}.get(str(value or "").strip().upper(), 0)


def _profile_matches_vehicle_type(profile, requested_type):
    requested = str(requested_type or "").strip().casefold()
    if not requested:
        return True
    vt = str(profile.get("VehicleType") or "").strip().upper()
    body = str(profile.get("BodyStyle") or "").strip().upper()
    size = str(profile.get("SizeClass") or "").strip().upper()

    if requested == "car":
        return vt == "CAR"
    if requested == "small_car":
        return vt == "CAR" and size in {"MICRO", "SMALL", "COMPACT"} and body not in {"SUV", "CROSSOVER", "PICKUP", "VAN", "MPV"}
    if requested in {"suv"}:
        return vt == "CAR" and body in {"SUV", "CROSSOVER"}
    if requested in {"crossover"}:
        return vt == "CAR" and body == "CROSSOVER"
    if requested in {"pickup", "pick-up"}:
        return vt == "PICKUP" or body == "PICKUP"
    if requested in {"motorcycle", "motosiklet"}:
        return vt in {"MOTORCYCLE", "SCOOTER"}
    if requested == "scooter":
        return vt == "SCOOTER" or body == "SCOOTER"
    return True


def _canonicalize_buyer_preferences(preferences):
    """Map multilingual/legacy preference labels to one internal taxonomy.

    Language is an input/output concern only. The market/profile engine should only
    ever see these canonical values, so EN/TR/RU requests behave identically.
    """
    aliases = {
        # vehicle profile
        "vehicle_type:car": "vehicle_type:car",
        "vehicle_type:automobile": "vehicle_type:car",
        "vehicle_type:otomobil": "vehicle_type:car",
        "vehicle_type:araba": "vehicle_type:car",
        "vehicle_type:автомобиль": "vehicle_type:car",
        "vehicle_type:машина": "vehicle_type:car",
        "vehicle_type:small_car": "vehicle_type:small_car",
        "vehicle_type:small car": "vehicle_type:small_car",
        "vehicle_type:küçük araç": "vehicle_type:small_car",
        "vehicle_type:kucuk arac": "vehicle_type:small_car",
        "vehicle_type:небольшая машина": "vehicle_type:small_car",
        "vehicle_type:небольшой автомобиль": "vehicle_type:small_car",
        "vehicle_type:suv": "vehicle_type:SUV",
        "vehicle_type:crossover": "vehicle_type:crossover",
        "vehicle_type:pickup": "vehicle_type:pickup",
        "vehicle_type:pick-up": "vehicle_type:pickup",
        "vehicle_type:motorcycle": "vehicle_type:motorcycle",
        "vehicle_type:motosiklet": "vehicle_type:motorcycle",
        "vehicle_type:мотоцикл": "vehicle_type:motorcycle",
        "vehicle_type:scooter": "vehicle_type:scooter",
        "vehicle_type:скутер": "vehicle_type:scooter",
        # buyer priorities
        "priority:economy": "priority:economy",
        "priority:economical": "priority:economy",
        "priority:ekonomik": "priority:economy",
        "priority:экономичная": "priority:economy",
        "priority:экономичный": "priority:economy",
        "priority:luxury": "priority:luxury",
        "priority:comfort": "priority:comfort",
        "priority:performance": "priority:performance",
        "priority:practicality": "priority:practicality",
        "use_case:family": "use_case:family",
        "use_case:commute": "use_case:commute",
    }
    out = []
    seen = set()
    for pref in preferences or []:
        raw = str(pref or "").strip()
        if not raw:
            continue
        canonical = aliases.get(raw.casefold(), raw)
        key = canonical.casefold()
        if key not in seen:
            out.append(canonical)
            seen.add(key)
    return out


def _profile_preference_score(profile, preferences):
    """Score buyer-profile fit without allowing soft traits to erase the market.

    Concrete vehicle classes (SUV/pickup/motorcycle/scooter) remain strict.
    Descriptors such as small-car, economy, comfort, practicality, family and
    commute are ranking preferences. This prevents a taxonomy/profile mismatch
    from turning valid hard-filtered inventory into an incorrect zero-result state.
    """
    preferences = _canonicalize_buyer_preferences(preferences)
    score = 0
    matched_soft = 0
    requested_soft = 0

    for pref in preferences or []:
        p = str(pref or "").strip().casefold()
        if p.startswith("vehicle_type:"):
            requested = p.split(":", 1)[1]
            # Explicit physical classes are genuine constraints. "small_car" is a
            # buyer profile/size preference and is therefore scored rather than fatal.
            if requested in {"car", "suv", "crossover", "pickup", "pick-up", "motorcycle", "motosiklet", "scooter"}:
                if not _profile_matches_vehicle_type(profile, requested):
                    return None
                score += 8
            elif requested == "small_car":
                requested_soft += 1
                if _profile_matches_vehicle_type(profile, requested):
                    matched_soft += 1
                    score += 8
                else:
                    score -= 5
        elif p == "priority:economy":
            requested_soft += 1
            level = _profile_level(profile.get("Economy"))
            if level >= 2:
                matched_soft += 1
                score += level * 4
            else:
                score -= 4
        elif p == "priority:luxury":
            requested_soft += 1
            level = _profile_level(profile.get("Luxury"))
            if level >= 2:
                matched_soft += 1
                score += level * 4
            else:
                score -= 4
        elif p == "priority:comfort":
            requested_soft += 1
            level = _profile_level(profile.get("Comfort"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "priority:performance":
            requested_soft += 1
            level = _profile_level(profile.get("Performance"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "priority:practicality":
            requested_soft += 1
            level = _profile_level(profile.get("Practicality"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "use_case:family":
            requested_soft += 1
            level = _profile_level(profile.get("Family"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3
        elif p == "use_case:commute":
            requested_soft += 1
            level = _profile_level(profile.get("Commute"))
            if level >= 2:
                matched_soft += 1
                score += level * 3
            else:
                score -= 3

    # At least one requested soft trait must be genuinely supported. This avoids
    # returning unrelated cars while still permitting graceful partial matches.
    if requested_soft and matched_soft == 0:
        return None

    confidence = str(profile.get("Confidence") or "").strip().upper()
    if confidence == "HIGH":
        score += 2
    elif confidence == "MEDIUM":
        score += 1
    return score


def _deterministic_profile_shortlist(model_market, preferences, max_models=30):
    if not MODEL_PROFILE_READY or not MODEL_PROFILE_LOOKUP:
        return []
    scored = []
    for summary in model_market:
        key = (str(summary.get("brand") or "").casefold(), str(summary.get("model") or "").casefold())
        profile = MODEL_PROFILE_LOOKUP.get(key)
        if not profile:
            continue
        score = _profile_preference_score(profile, preferences)
        if score is None:
            continue
        ranked_summary = dict(summary)
        ranked_summary["_profile_score"] = score
        scored.append((
            -score,
            -int(summary.get("newest_year") or 0),
            -min(int(summary.get("count") or 0), 100),
            str(summary.get("brand") or "").casefold(),
            str(summary.get("model") or "").casefold(),
            ranked_summary,
        ))
    scored.sort(key=lambda x: x[:-1])
    return [x[-1] for x in scored[:max_models]]


def load_market_data():
    global market_df, MARKET_READY

    try:
        r = requests.get(MARKET_CSV_URL, timeout=15)
        r.raise_for_status()

        new_market_df = pd.read_csv(
            io.StringIO(r.text),
            low_memory=False
        )

        # -----------------------
        # REQUIRED COLUMNS
        # -----------------------
        required_columns = [
            "Brand",
            "Model",
            "Category",
            "Year",
            "Price",
            "KM",
            "Company",
            "Location",
            "Transmission",
            "Color",
            "Image",
            "Link"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in new_market_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"market_base.csv missing columns: {missing_columns}"
            )

        # -----------------------
        # FUEL (ROLLOUT-SAFE)
        # -----------------------
        # New market builds persist Fuel directly. During deployment, an older
        # market_base.csv may still be live for a few minutes, so derive the same
        # canonical value from Link rather than failing the entire market loader.
        def fuel_from_link(link):
            text = str(link or "").strip().casefold()
            patterns = [
                (r"(?:^|-)mild-hibrit-benzin(?:-|$)", "Mild Hybrid Petrol"),
                (r"(?:^|-)mild-hibrit-dizel(?:-|$)", "Mild Hybrid Diesel"),
                (r"(?:^|-)plug-in-hibrit(?:-|$)", "Plug-in Hybrid"),
                (r"(?:^|-)elektrik(?:-|$)", "Electric"),
                (r"(?:^|-)hibrit(?:-|$)", "Hybrid"),
                (r"(?:^|-)dizel(?:-|$)", "Diesel"),
                (r"(?:^|-)benzin(?:-|$)", "Petrol"),
            ]
            for pattern, label in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return label
            return ""

        if "Fuel" not in new_market_df.columns:
            new_market_df["Fuel"] = new_market_df["Link"].apply(fuel_from_link)
            print("Market CSV Fuel column missing; derived Fuel from Link for compatibility")
        else:
            # Backfill any isolated blank Fuel rows from their listing URL.
            blank_fuel = new_market_df["Fuel"].fillna("").astype(str).str.strip().eq("")
            if blank_fuel.any():
                new_market_df.loc[blank_fuel, "Fuel"] = (
                    new_market_df.loc[blank_fuel, "Link"].apply(fuel_from_link)
                )

        # -----------------------
        # NUMERIC TYPES
        # -----------------------
        new_market_df["Year"] = pd.to_numeric(
            new_market_df["Year"],
            errors="coerce"
        )

        new_market_df["Price"] = pd.to_numeric(
            new_market_df["Price"],
            errors="coerce"
        )

        new_market_df["KM"] = pd.to_numeric(
            new_market_df["KM"],
            errors="coerce"
        )

        # -----------------------
        # TEXT TYPES
        # -----------------------
        text_columns = [
            "Brand",
            "Model",
            "Category",
            "Company",
            "Location",
            "Transmission",
            "Fuel",
            "Color",
            "Image",
            "Link"
        ]

        for col in text_columns:
            new_market_df[col] = (
                new_market_df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # Only replace the live dataframe after
        # the new file has loaded successfully.
        market_df = new_market_df
        MARKET_READY = True

        print(
            f"Market CSV loaded successfully: "
            f"{len(market_df)} listings"
        )

    except Exception as e:
        print("MARKET CSV LOAD FAILED:", e)

        # IMPORTANT:
        # If an old successful dataset already exists,
        # leave it running rather than wiping it.
        if market_df is None or market_df.empty:
            market_df = pd.DataFrame()
            MARKET_READY = False


load_market_data()
load_buyer_intelligence()
load_model_profiles()
load_business_intelligence()
load_business_activity()
load_business_activity_history()


# =========================================================
# MARKET DATA AUTO REFRESH EVERY 12 HOURS
# =========================================================

def refresh_market_data_loop():
    while True:
        time.sleep(12 * 60 * 60)

        print("Refreshing market CSV data from GitHub...")

        load_market_data()
        print("Refreshing Buyer Intelligence from GitHub...")
        load_buyer_intelligence()
        print("Refreshing Model Buyer Profiles from GitHub...")
        load_model_profiles()
        print("Refreshing Business Intelligence from GitHub...")
        load_business_intelligence()
        print("Refreshing Business activity intelligence...")
        load_business_activity()
        print("Refreshing Business activity raw history...")
        load_business_activity_history()


threading.Thread(
    target=refresh_market_data_loop,
    daemon=True
).start()

# =========================================================
# YEARS
# =========================================================
@app.route("/years", methods=["GET"])
def get_years():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())
    return jsonify(years)

# =========================================================
# BRANDS
# =========================================================
@app.route("/brands", methods=["GET"])
def get_brands():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

    year = request.args.get("year")
    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    return jsonify(sorted(filtered["Brand"].dropna().unique().tolist()))

# =========================================================
# MODELS
# =========================================================
@app.route("/models", methods=["GET"])
def get_models():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

    year = request.args.get("year")
    brand = request.args.get("brand")

    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    return jsonify(sorted(filtered["Model"].dropna().unique().tolist()))

# =========================================================
# CATEGORIES
# =========================================================
@app.route("/categories", methods=["GET"])
def get_categories():
    if not DATA_READY or df is None or df.empty:
        return jsonify([])

    year = request.args.get("year")
    brand = request.args.get("brand")
    model = request.args.get("model")

    # IMPORTANT: this endpoint belongs to the standalone otodeger.online
    # valuation flow. Keep legacy/raw category values here, including the
    # no-category sentinel produced by dataframe preparation (for example
    # "nan"). The valuation frontend relies on that response to advance when
    # a vehicle has no real category. AI-assistant category skipping is handled
    # separately by /api/guided/discovery/options and must not change this route.
    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[pd.to_numeric(filtered["Year"], errors="coerce") == year]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    if model not in [None, "", "null"]:
        filtered = filtered[
            filtered["Model"].astype(str).str.strip().str.lower()
            == str(model).strip().lower()
        ]

    return jsonify(sorted(filtered["Category"].dropna().unique().tolist()))


# =========================================================
# VALUATION ENGINE
# =========================================================
def get_valuation(df, year, brand, model, category):
    if not DATA_READY or df is None or df.empty:
        return {
            "median_price": None,
            "min_price": None,
            "max_price": None,
            "scatter": []
        }

    filtered = df.copy()

    if year not in [None, "", "null"]:
        try:
            year = int(float(year))
            filtered = filtered[
                pd.to_numeric(filtered["Year"], errors="coerce") == year
            ]
        except:
            pass

    if brand not in [None, "", "null"]:
        filtered = filtered[
            filtered["Brand"].astype(str).str.strip().str.lower()
            == str(brand).strip().lower()
        ]

    if model not in [None, "", "null"]:
        filtered = filtered[
            filtered["Model"].astype(str).str.strip().str.lower()
            == str(model).strip().lower()
        ]

    if category not in [None, "", "null"]:
        filtered = filtered[
            filtered["Category"].astype(str).str.strip().str.lower()
            == str(category).strip().lower()
        ]

    if filtered.empty:
        return {
            "median_price": None,
            "min_price": None,
            "max_price": None,
            "scatter": []
        }

    median_price = filtered["Price"].median()
    min_price = filtered["Price"].min()
    max_price = filtered["Price"].max()

    latest_date = filtered["DATE"].max()

    filtered["status"] = filtered["DATE"].apply(
        lambda x: "active" if x == latest_date else "removed"
    )

    scatter = filtered[["KM", "Price", "status"]].dropna().to_dict(orient="records")

    return {
        "median_price": float(median_price) if pd.notna(median_price) else None,
        "min_price": float(min_price) if pd.notna(min_price) else None,
        "max_price": float(max_price) if pd.notna(max_price) else None,
        "scatter": scatter
    }

# =========================================================
# API ENDPOINT
# =========================================================
@app.route("/get_valuation", methods=["POST"])
def valuation():
    data = request.json

    result = get_valuation(
        df,
        data.get("year"),
        data.get("brand"),
        data.get("model"),
        data.get("category")
    )

    return jsonify(result)


# =========================================================
# AI BUYING ASSISTANT - NATURAL LANGUAGE INTERPRETER
# =========================================================

AI_FILTER_KEYS = {
    "budget",
    "min_budget",
    "brands",
    "exclude_brands",
    "models",
    "exclude_models",
    "categories",
    "exclude_categories",
    "locations",
    "exclude_locations",
    "companies",
    "exclude_companies",
    "transmissions",
    "colors",
    "min_year",
    "max_year",
    "min_km",
    "max_km",
}

def sanitize_conversation_history(history, max_messages=16):
    """Keep only a compact, safe recent chat context for interpretation/response quality."""
    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        text = str(item.get("text") or item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        cleaned.append({
            "role": role,
            "text": text[:2500],
        })
    return cleaned


def _detect_message_language(text):
    text = str(text or "").strip()
    if not text:
        return None

    low = text.casefold()

    if re.search(r"[\u0400-\u04FF]", text):
        return "RU"

    if re.search(r"[çğıöşüÇĞİÖŞÜ]", text) or re.search(
        r"\b(istiyorum|olsun|bakıyorum|bütçe|araç|araba|motosiklet|hangisi|hepsine|"
        r"evet|hayır|yok|var mı|şart değil|istemiyorum|karşılaştır|göster)\b",
        low,
    ):
        return "TR"

    if re.search(
        r"\b(i|want|buy|suv|what|can|could|get|are|there|more|options|yes|no|all|them|"
        r"what about|show|compare|cheaper|newer|which|please|under|budget|available|"
        r"have|with|without|looking|find|tell|me|cars?|vehicles?)\b",
        low,
    ):
        return "EN"

    return None


def detect_conversation_language(message, requested_language="TR", conversation_history=None):
    """
    Follow the language the buyer is actually using.

    Short follow-ups such as "GLA", "yes" or a bare brand inherit the most recent
    detectable user language instead of snapping back to the UI default.
    """
    current = _detect_message_language(message)
    if current:
        return current

    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").lower() != "user":
            continue
        detected = _detect_message_language(item.get("text"))
        if detected:
            return detected

    requested = str(requested_language or "TR").upper()
    return requested if requested in {"TR", "EN", "RU"} else "TR"

def normalize_assistant_format(answer):
    """
    Enforce one consistent premium chat rhythm:
    intro paragraph
    model/listing lines with NO blank lines between them
    final paragraph

    The model is still responsible for wording; this only normalizes whitespace.
    """
    text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    raw = [line.strip() for line in text.split("\n")]
    nonempty = [line for line in raw if line]
    if len(nonempty) <= 2:
        return "\n\n".join(nonempty)

    def looks_like_option(line):
        # Vehicle/listing lines normally contain an em dash and a price/year fact.
        return "—" in line and (
            "£" in line
            or re.search(r"\b(19|20)\d{2}\b", line)
        )

    first_option = next((i for i, line in enumerate(nonempty) if looks_like_option(line)), None)
    if first_option is None:
        return "\n\n".join(nonempty)

    last_option = first_option
    while last_option + 1 < len(nonempty) and looks_like_option(nonempty[last_option + 1]):
        last_option += 1

    before = nonempty[:first_option]
    options = nonempty[first_option:last_option + 1]
    after = nonempty[last_option + 1:]

    parts = []
    if before:
        parts.append(" ".join(before))
    if options:
        parts.append("\n".join(options))
    if after:
        parts.append(" ".join(after))
    return "\n\n".join(parts)


def compact_market_context():
    """
    Give the model enough live vocabulary to map user language onto
    values that actually exist in market_base.csv without sending the
    entire dataset to the model.
    """
    if not MARKET_READY or market_df is None or market_df.empty:
        return {}

    def unique_values(column, limit=None):
        values = sorted(
            market_df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        return values[:limit] if limit else values

    return {
        "brands": unique_values("Brand"),
        "locations": unique_values("Location"),
        "transmissions": unique_values("Transmission"),
        "companies": unique_values("Company", 250),
    }


def _normalize_vehicle_phrase(value):
    """
    Normalize user/market vehicle names for deterministic mention matching.
    Hyphens and punctuation are treated as spaces so forms such as
    "e-Power", "e power" and "E-POWER" can resolve to the same market value.
    """
    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9çğıöşüа-яё]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def resolve_market_vehicle_mentions(message):
    """
    Resolve explicitly named vehicles against live Brand + Model + Category values.

    This version is intentionally linear-time. The previous implementation repeatedly
    filtered the market universe once for every model family, which could add ~20s to
    every assistant request on the production dataset.
    """
    if not MARKET_READY or market_df is None or market_df.empty:
        return []

    message_n = _normalize_vehicle_phrase(message)
    if not message_n:
        return []
    padded_message = f" {message_n} "

    universe = (
        market_df[["Brand", "Model", "Category"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
    )
    # A dirty-data row can occasionally have a brand name in the Model column
    # (for example Brand=Çekici, Model=Nissan). Never treat a model-only token as
    # a vehicle mention when that token is itself a real brand name.
    known_brand_norms = {
        _normalize_vehicle_phrase(x)
        for x in universe["Brand"].tolist()
        if str(x).strip()
    }

    # Build all lookup structures in one pass instead of rescanning the dataframe
    # for each Brand/Model candidate.
    families = {}
    model_to_brands = {}
    for row in universe.itertuples(index=False):
        brand = str(row.Brand).strip()
        model = str(row.Model).strip()
        category = str(row.Category).strip()
        if not brand or not model:
            continue

        brand_n = _normalize_vehicle_phrase(brand)
        model_n = _normalize_vehicle_phrase(model)
        key = (brand_n, model_n)
        family = families.setdefault(key, {
            "brand": brand,
            "model": model,
            "brand_n": brand_n,
            "model_n": model_n,
            "categories": {},
        })
        model_to_brands.setdefault(model_n, set()).add(brand_n)
        if category:
            category_n = _normalize_vehicle_phrase(category)
            if category_n:
                family["categories"][category_n] = category

    # Some real model names are also ordinary words. Model-only matching for
    # these names is dangerous because normal sentences can otherwise become a
    # vehicle selection (for example: "What cars fit my budget?" -> Honda Fit).
    #
    # We still allow them when the user writes the model with its normal title
    # casing (e.g. "Fit vs Yaris"), while full Brand + Model mentions always work.
    ambiguous_model_only_norms = {
        "fit", "one", "note", "up", "march", "focus", "golf"
    }
    raw_message = str(message or "")

    candidates = []
    for family in families.values():
        brand = family["brand"]
        model = family["model"]
        brand_n = family["brand_n"]
        model_n = family["model_n"]
        full_n = f"{brand_n} {model_n}".strip()

        match_strength = 0
        if full_n and f" {full_n} " in padded_message:
            match_strength = 3
        else:
            # For ambiguous word-like model names, require the model to appear
            # with its canonical casing when the brand is omitted. This keeps
            # natural language such as "cars fit my budget" from resolving to
            # Honda Fit, while "Compare Fit and Yaris" remains usable.
            ambiguous_model_only = model_n in ambiguous_model_only_norms
            canonical_model_only_mention = bool(
                re.search(
                    rf"(?<!\\w){re.escape(model)}(?!\\w)",
                    raw_message,
                )
            )

            if (
                model_n
                and len(model_n) >= 3
                and model_n not in known_brand_norms
                and f" {model_n} " in padded_message
                and len(model_to_brands.get(model_n, set())) == 1
                and (not ambiguous_model_only or canonical_model_only_mention)
            ):
                match_strength = 2

        if not match_strength:
            continue

        target = {
            "brand": brand,
            "model": model,
            "category": None,
            "_match_strength": match_strength,
            "_phrase_len": len(full_n),
        }

        category_matches = []
        for category_n, category in family["categories"].items():
            # Do not infer plain engine-size categories from incidental numbers.
            if not re.search(r"[a-zçğıöşüа-яё]", category_n, flags=re.IGNORECASE):
                continue
            model_category = f"{model_n} {category_n}".strip()
            full_category = f"{brand_n} {model_n} {category_n}".strip()
            if (
                (full_category and f" {full_category} " in padded_message)
                or (model_category and f" {model_category} " in padded_message)
            ):
                category_matches.append((len(category_n), category))

        if category_matches:
            category_matches.sort(reverse=True)
            target["category"] = category_matches[0][1]
            target["_match_strength"] = 4
            target["_phrase_len"] += category_matches[0][0]

        candidates.append(target)

    # Natural vehicle names do not always mirror the site's Brand -> Model ->
    # Category hierarchy. For example, users say "BMW 118i" while the market
    # stores BMW -> 1 Serisi -> 118i. When an explicitly mentioned brand plus a
    # non-numeric category token maps to exactly one model family for that brand,
    # resolve it deterministically to that canonical family.
    brand_category_hits = {}
    for family in families.values():
        brand_n = family["brand_n"]
        if not brand_n or f" {brand_n} " not in padded_message:
            continue
        for category_n, category in family["categories"].items():
            if not re.search(r"[a-zçğıöşüа-яё]", category_n, flags=re.IGNORECASE):
                continue
            if f" {category_n} " not in padded_message:
                continue
            key = (brand_n, category_n)
            brand_category_hits.setdefault(key, []).append((family, category))

    for (_brand_n, category_n), hits in brand_category_hits.items():
        model_keys = {(h[0]["brand_n"], h[0]["model_n"]) for h in hits}
        if len(model_keys) != 1:
            continue
        family, category = hits[0]
        candidates.append({
            "brand": family["brand"],
            "model": family["model"],
            "category": category,
            "_match_strength": 4,
            "_phrase_len": len(family["brand_n"]) + len(category_n),
        })

    # If a longer model phrase contains a shorter model phrase from the same brand,
    # keep the most specific family. This protects compound model names.
    candidates.sort(key=lambda x: (-x["_match_strength"], -x["_phrase_len"]))
    resolved = []
    seen = set()
    for item in candidates:
        key = (_normalize_vehicle_phrase(item["brand"]), _normalize_vehicle_phrase(item["model"]))
        if key in seen:
            continue
        # Skip a shorter same-brand model fully contained in an already-selected model.
        model_n = key[1]
        brand_n = key[0]
        if any(
            _normalize_vehicle_phrase(x["brand"]) == brand_n
            and model_n != _normalize_vehicle_phrase(x["model"])
            and f" {model_n} " in f" {_normalize_vehicle_phrase(x['model'])} "
            for x in resolved
        ):
            continue
        seen.add(key)
        item.pop("_match_strength", None)
        item.pop("_phrase_len", None)
        resolved.append(item)

    return resolved

def _attach_explicit_years_to_vehicle_targets(message, targets):
    """Attach a nearby explicit model year to each named comparison target.

    Years belong to the vehicle they are written next to (for example,
    "2022 Honda Fit and 2021 Toyota Yaris"). They must not become one global
    min/max-year filter across all comparison targets.
    """
    text = str(message or "")
    enriched = [dict(t) for t in (targets or [])]

    for target in enriched:
        brand = str(target.get("brand") or "").strip()
        model = str(target.get("model") or "").strip()
        if not brand or not model:
            continue

        vehicle = rf"{re.escape(brand)}\s+{re.escape(model)}"
        patterns = [
            rf"\b((?:19|20)\d{{2}})\s+(?:model\s+)?{vehicle}\b",
            rf"\b{vehicle}\s+(?:model\s+)?((?:19|20)\d{{2}})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                target["year"] = int(match.group(1))
                break

    return enriched


def _search_market_for_vehicle_targets(base_filters, targets):
    """
    Search each explicit vehicle target independently, then merge the results.

    This is essential for comparisons where each target can have its own category,
    e.g. Toyota Aqua versus Nissan Note e-Power. A single global Category filter
    would incorrectly apply e-Power to both models.
    """
    merged = []
    seen_links = set()
    total = 0

    neutral = dict(base_filters or {})
    for key in (
        "brands", "exclude_brands",
        "models", "exclude_models",
        "categories", "exclude_categories",
    ):
        neutral.pop(key, None)

    for target in targets or []:
        target_filters = dict(neutral)
        target_filters["brands"] = [target["brand"]]
        target_filters["models"] = [target["model"]]
        if target.get("category"):
            target_filters["categories"] = [target["category"]]
        if target.get("year") is not None:
            target_filters["min_year"] = int(target["year"])
            target_filters["max_year"] = int(target["year"])

        result = market_search(
            budget=target_filters.get("budget"),
            min_budget=target_filters.get("min_budget"),
            brands=target_filters.get("brands"),
            exclude_brands=target_filters.get("exclude_brands"),
            models=target_filters.get("models"),
            exclude_models=target_filters.get("exclude_models"),
            categories=target_filters.get("categories"),
            exclude_categories=target_filters.get("exclude_categories"),
            locations=target_filters.get("locations"),
            exclude_locations=target_filters.get("exclude_locations"),
            companies=target_filters.get("companies"),
            exclude_companies=target_filters.get("exclude_companies"),
            transmissions=target_filters.get("transmissions"),
            colors=target_filters.get("colors"),
            min_year=target_filters.get("min_year"),
            max_year=target_filters.get("max_year"),
            min_km=target_filters.get("min_km"),
            max_km=target_filters.get("max_km"),
            limit=5000,
            max_limit=5000,
            analysis_mode=True,
        )

        if not result.get("success"):
            return result

        # Explicit named comparison/shop targets must be exact Brand + Model
        # matches. market_search intentionally supports broader substring matching
        # for the public search API, but that would make "Honda Fit" also include
        # "Honda Fit Aria" inside a named comparison.
        target_brand_cf = str(target.get("brand") or "").strip().casefold()
        target_model_cf = str(target.get("model") or "").strip().casefold()
        target_category_cf = str(target.get("category") or "").strip().casefold()

        exact_items = []
        for item in result.get("results", []) or []:
            item_brand_cf = str(item.get("brand") or "").strip().casefold()
            item_model_cf = str(item.get("model") or "").strip().casefold()
            item_category_cf = str(item.get("category") or "").strip().casefold()

            if item_brand_cf != target_brand_cf or item_model_cf != target_model_cf:
                continue
            if target_category_cf and item_category_cf != target_category_cf:
                continue
            exact_items.append(item)

        total += len(exact_items)

        for item in exact_items:
            link = str(item.get("link") or "").strip()
            dedupe_key = link or json.dumps(item, ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            merged.append(item)

    merged.sort(
        key=lambda x: (
            float(x.get("price") or 10**12),
            -(int(x.get("year") or 0)),
        )
    )

    return {
        "success": True,
        "count": len(merged),
        "returned": len(merged),
        "results": merged,
        "target_raw_count": total,
    }


def extract_response_text(payload):
    """
    Extract the text returned by the Responses API without requiring
    the OpenAI Python package.
    """
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")

    return ""


def sanitize_ai_filters(raw_filters):
    """
    Never trust model/client filter values directly.

    Numeric filters must remain scalar numbers and list filters must remain
    arrays of strings. Malformed generative output such as {"max": 18000}
    is rejected here so it can never reach market_search / float().
    """
    if not isinstance(raw_filters, dict):
        return {}

    numeric_fields = {
        "budget", "min_budget", "min_year", "max_year", "min_km", "max_km",
    }
    integer_fields = {"min_year", "max_year"}
    list_fields = {
        "brands", "exclude_brands", "models", "exclude_models",
        "categories", "exclude_categories", "locations", "exclude_locations",
        "companies", "exclude_companies", "transmissions", "colors",
    }

    clean = {}

    for key, value in raw_filters.items():
        if key not in AI_FILTER_KEYS:
            continue

        if value in [None, "", [], {}]:
            continue

        if key in numeric_fields:
            if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
                print(f"IGNORING MALFORMED NUMERIC FILTER {key}: {value!r}", flush=True)
                continue

            try:
                number = float(value)
            except (TypeError, ValueError):
                print(f"IGNORING NON-NUMERIC FILTER {key}: {value!r}", flush=True)
                continue

            if not math.isfinite(number):
                continue

            clean[key] = int(number) if key in integer_fields else number
            continue

        if key in list_fields:
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                print(f"IGNORING MALFORMED LIST FILTER {key}: {value!r}", flush=True)
                continue

            normalized = []
            seen = set()
            for item in values:
                if isinstance(item, (dict, list, tuple, set)):
                    continue
                text = str(item or "").strip()
                key_text = text.casefold()
                if text and key_text not in seen:
                    normalized.append(text)
                    seen.add(key_text)

            if key == "transmissions" and normalized:
                # market_base.csv stores transmission values in the source-market
                # vocabulary. Normalize multilingual/user-facing variants to those
                # canonical stored values before deterministic filtering.
                transmission_aliases = {
                    "automatic": "Otomatik", "auto": "Otomatik", "otomatik": "Otomatik",
                    "автомат": "Otomatik", "акпп": "Otomatik",
                    "manual": "Düz", "manuel": "Düz", "düz": "Düz", "duz": "Düz",
                    "механика": "Düz", "мкпп": "Düz",
                    "semi automatic": "Yarı Otomatik", "semi-automatic": "Yarı Otomatik",
                    "yarı otomatik": "Yarı Otomatik", "yari otomatik": "Yarı Otomatik",
                }
                mapped = []
                mapped_seen = set()
                for value in normalized:
                    canonical = transmission_aliases.get(value.casefold(), value)
                    k = canonical.casefold()
                    if k not in mapped_seen:
                        mapped.append(canonical)
                        mapped_seen.add(k)
                normalized = mapped

            if normalized:
                clean[key] = normalized

    return clean


def _parse_human_number(token):
    token = str(token or "").strip().lower().replace(" ", "").rstrip(".,;:")
    if not token:
        return None
    multiplier = 1
    if token.endswith("k"):
        multiplier = 1000
        token = token[:-1]
    # 15,000 / 15.000 are thousands; 15.5 is decimal when small.
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif token.count(",") == 1:
        left, right = token.split(",")
        token = left + right if len(right) == 3 else left + "." + right
    elif token.count(".") == 1:
        left, right = token.split(".")
        token = left + right if len(right) == 3 and len(left) >= 1 else left + "." + right
    try:
        return float(token) * multiplier
    except (TypeError, ValueError):
        return None



# =========================================================
# AI ASSISTANT V8 - SEMANTIC CONVERSATION CONTROLLER
# =========================================================
#
# V8 separates "what did the user mean?" from "what does the market data say?".
# The model is used only to understand the conversational turn. All prices,
# counts, listings, liquidity, rankings and market evidence remain deterministic
# and are calculated from OtoDeğer data below.
#
# This deliberately replaces the old pattern of recovering arbitrary state from
# older prose. current_filters/current_preferences are the authoritative state;
# recent conversation is used only to understand the latest turn.

ASSISTANT_ORCHESTRATION_VERSION = "9.1"


def _v8_previous_assistant_text(conversation_history):
    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").casefold() != "assistant":
            continue
        value = str(item.get("text") or item.get("content") or "").strip()
        if value:
            return value
    return ""


def _v8_state_snapshot(current_filters, current_preferences, conversation_history):
    return {
        "filters": sanitize_ai_filters(current_filters or {}),
        "preferences": _canonicalize_buyer_preferences(current_preferences or []),
        "previous_assistant_message": _v8_previous_assistant_text(conversation_history)[:1200],
    }


def _v8_semantic_turn_plan(
    message,
    language,
    current_filters,
    current_preferences,
    conversation_history,
    explicit_targets,
):
    """
    Premium semantic controller for Personal buyer conversations.

    It returns an incremental state operation. It does NOT answer the market
    question and it never supplies market facts. That separation is important:
    language understanding can be probabilistic; OtoDeğer evidence cannot be.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    state = _v8_state_snapshot(
        current_filters,
        current_preferences,
        conversation_history,
    )

    instructions = """
You are the semantic conversation controller for OtoDeğer, a premium North Cyprus
vehicle-market assistant.

Your ONLY job is to understand the user's latest conversational turn and return
one compact JSON object describing how the application's structured state should
change. Never answer the user. Never invent vehicle-market facts.

Think like an excellent human sales/research assistant:
- Understand terse answers in context. If the assistant just asked for maximum
  budget and the user says "15.000", that is the budget answer, not nonsense.
- Preserve active constraints when the user is refining the SAME task:
  "I want something economical", "under 80,000 km", "automatic", "newer".
- Do not resurrect an older task after the user has started a newer one.
- A clearly new request starts a NEW_TASK. Examples:
  * after discussing two models, "What cars fit my budget?" starts a new discovery.
  * "Compare Honda Fit and Toyota Yaris" starts a new comparison.
- A follow-up such as "which one would you choose?", "show me those", "cheaper",
  "what about private sellers?", "under 80k km" normally CONTINUES the current task.
- CORRECT means the user is correcting/removing/replacing something in the active task.
- ANSWER means the user is directly answering the assistant's immediately preceding
  clarification/question.
- Never infer a Honda Fit from the ordinary English verb "fit".
- Do not translate brand/model names.
- resolved_explicit_vehicle_targets are deterministic application evidence. Use them
  to determine task/mode; do not invent additional vehicle names.
- Hard constraints go in filters. Subjective goals go in preferences.
- Vehicle classes use preferences, not Category:
  vehicle_type:car, vehicle_type:SUV, vehicle_type:crossover,
  vehicle_type:pickup, vehicle_type:small_car, vehicle_type:motorcycle,
  vehicle_type:scooter.
- Buyer goals use:
  priority:economy, priority:reliability, priority:performance, priority:luxury,
  priority:comfort, priority:practicality, use_case:family, use_case:commute.
- Listing ordering uses:
  listing_sort:lowest_km, listing_sort:cheapest, listing_sort:newest.
- If a new concrete vehicle_type is supplied, the application will replace the old type.
- Prices are GBP. "15k", "15 bin", "15 thousand" = 15000.
- "15.000" or "15,000" in an immediately requested budget context = 15000.
- Mileage numbers must not be confused with budgets.
- Exact model years attached to a named vehicle are handled by deterministic vehicle
  resolution, so do not turn two different comparison years into one global year filter.
- For a fresh explicit multi-vehicle comparison, use NEW_TASK and do not carry an old
  budget/body-type search unless the user repeats that constraint in the SAME message.
- If the user asks to see actual cars/listings/ads, mode is SHOP.
- If they compare two or more vehicles, mode is COMPARE.
- Otherwise mode is DISCOVER.
- needs_clarification=true only when the latest message is genuinely ambiguous and
  cannot safely be understood from state + previous assistant message.
- Do not ask generic questions merely because some optional buying dimensions are absent;
  the application has its own progressive narrowing flow.
- seller_mode: individual means private/Bireysel, gallery means dealers, both clears it.
- clear_filters contains only hard filter names that the user explicitly removes.
- clear_preferences contains canonical preference tags or prefixes the user explicitly
  removes. Use "vehicle_type:*" to clear an old vehicle class and "priority:*" only if
  the user explicitly removes all priorities.
- Return JSON only. No markdown.

JSON shape:
{
  "operation": "CONTINUE" | "NEW_TASK" | "ANSWER" | "CORRECT",
  "decision_mode": "DISCOVER" | "COMPARE" | "SHOP",
  "filters": {
    "budget": number|null,
    "min_budget": number|null,
    "brands": array|null,
    "exclude_brands": array|null,
    "models": array|null,
    "exclude_models": array|null,
    "categories": array|null,
    "exclude_categories": array|null,
    "locations": array|null,
    "exclude_locations": array|null,
    "companies": array|null,
    "exclude_companies": array|null,
    "transmissions": array|null,
    "colors": array|null,
    "min_year": integer|null,
    "max_year": integer|null,
    "min_km": number|null,
    "max_km": number|null
  },
  "clear_filters": [],
  "seller_mode": null | "individual" | "gallery" | "both",
  "preferences": [],
  "clear_preferences": [],
  "needs_clarification": false,
  "clarification_question": null,
  "awaiting": null | "budget" | "mileage" | "year" | "vehicle_type" | "model" | "seller"
}
"""

    payload = {
        "language": language,
        "latest_message": str(message or "")[:1200],
        "state": state,
        "recent_conversation": sanitize_conversation_history(
            conversation_history, max_messages=8
        ),
        "resolved_explicit_vehicle_targets": explicit_targets or [],
    }

    response = _openai_post(
        payload={
            "model": OPENAI_MODEL,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 850,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
        },
        timeout=(1.5, 7.0),
    )
    response.raise_for_status()
    response_text = extract_response_text(response.json())
    if not response_text:
        raise ValueError("V8_TURN_CONTROLLER_EMPTY_RESPONSE")

    plan = json.loads(response_text)

    operation = str(plan.get("operation") or "CONTINUE").upper()
    if operation not in {"CONTINUE", "NEW_TASK", "ANSWER", "CORRECT"}:
        operation = "CONTINUE"

    decision_mode = str(plan.get("decision_mode") or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"

    filters = sanitize_ai_filters(plan.get("filters") or {})
    filters = {k: v for k, v in filters.items() if v is not None}

    clear_filters = [
        str(k) for k in (plan.get("clear_filters") or [])
        if str(k) in AI_FILTER_KEYS
    ]

    seller_mode = plan.get("seller_mode")
    if seller_mode not in {None, "individual", "gallery", "both"}:
        seller_mode = None

    preferences = _canonicalize_buyer_preferences(plan.get("preferences") or [])
    clear_preferences = [
        str(v).strip() for v in (plan.get("clear_preferences") or [])
        if str(v).strip()
    ]

    awaiting = plan.get("awaiting")
    if awaiting not in {None, "budget", "mileage", "year", "vehicle_type", "model", "seller"}:
        awaiting = None

    clarification = str(plan.get("clarification_question") or "").strip() or None

    return {
        "operation": operation,
        "filters": filters,
        "clear_filters": clear_filters,
        "seller_mode": seller_mode,
        "preferences": preferences,
        "clear_preferences": clear_preferences,
        "needs_clarification": bool(plan.get("needs_clarification")),
        "clarification_question": clarification,
        "decision_mode": decision_mode,
        "awaiting": awaiting,
        "fast_path": False,
        "orchestration_version": ASSISTANT_ORCHESTRATION_VERSION,
    }



def _v8_apply_authoritative_numeric_constraints(message, resolved_targets, interpretation):
    """
    Numeric constraints stated explicitly in the latest user turn are deterministic
    application facts, not semantic guesses.

    The LLM controller decides conversational meaning/state, but explicit amounts
    such as £18k, £12k, 80,000 km, and year bounds are re-parsed by the deterministic
    parser and overwrite any malformed numeric value returned by the controller.

    This prevents failures such as £18k becoming £18,000,000 while preserving the
    semantic controller's operation/mode/preference understanding.
    """
    try:
        deterministic = fast_common_interpretation(
            message=message,
            resolved_targets=resolved_targets or [],
        ) or {}
    except Exception:
        return interpretation

    authoritative_keys = (
        "budget",
        "min_budget",
        "min_year",
        "max_year",
        "min_km",
        "max_km",
    )

    deterministic_filters = sanitize_ai_filters(
        deterministic.get("filters") or {}
    )
    if not deterministic_filters:
        return interpretation

    merged = dict(interpretation or {})
    semantic_filters = dict(merged.get("filters") or {})

    for key in authoritative_keys:
        value = deterministic_filters.get(key)
        if value is not None:
            semantic_filters[key] = value

    merged["filters"] = sanitize_ai_filters(semantic_filters)
    return merged

def _v8_apply_preference_changes(previous_preferences, interpretation):
    """Apply semantic preference removals, then normal replacement-aware merging."""
    previous = _canonicalize_buyer_preferences(previous_preferences or [])
    clear_preferences = [
        str(v).strip().casefold()
        for v in (interpretation.get("clear_preferences") or [])
        if str(v).strip()
    ]

    if clear_preferences:
        kept = []
        for pref in previous:
            p = str(pref).casefold()
            remove = False
            for clear in clear_preferences:
                if clear.endswith("*"):
                    if p.startswith(clear[:-1]):
                        remove = True
                        break
                elif p == clear:
                    remove = True
                    break
            if not remove:
                kept.append(pref)
        previous = kept

    return merge_preferences(previous, interpretation.get("preferences") or [])


def _v8_should_recover_compare_targets(message, interpretation):
    """
    Recover old comparison targets only when the semantic controller says the
    latest turn continues/answers/corrects the active comparison. DISCOVER turns
    can therefore never accidentally resurrect an older Fit/Yaris comparison.
    """
    if str(interpretation.get("decision_mode") or "").upper() != "COMPARE":
        return False
    return str(interpretation.get("operation") or "").upper() in {
        "CONTINUE", "ANSWER", "CORRECT"
    }


def _v8_should_recover_shop_target(message, interpretation):
    if str(interpretation.get("decision_mode") or "").upper() != "SHOP":
        return False
    return str(interpretation.get("operation") or "").upper() in {
        "CONTINUE", "ANSWER", "CORRECT"
    }

def fast_common_interpretation(message, resolved_targets=None):
    """
    Deterministic multilingual fast path for common buyer requests.
    Supports English, Turkish and Russian without a live LLM round-trip.
    Ambiguous conversational turns still fall back to the LLM interpreter.
    """
    raw = str(message or "").strip()
    low = raw.casefold()
    if not raw:
        return None

    # Terse contextual replies and explicit corrections/negations are intentionally
    # left to the conversational interpreter because they depend heavily on history.
    terse = {
        "yes", "no", "evet", "hayır", "да", "нет",
        "more", "more?", "daha", "daha?", "ещё", "еще",
        "all", "all of them", "hepsi", "все", "все варианты",
    }
    if low in terse:
        return None
    if re.search(r"\b(don't|do not|without|exclude|forget|istemiyorum|olmasın|hariç|istemem|unut|без|исключи|не хочу|забудь)\b", low):
        return None

    targets = list(resolved_targets or [])
    filters = {}
    preferences = []
    seller_mode = None

    # Decision mode — deliberately multilingual and conservative.
    shop_words = re.search(
        r"\b(listings?|ads?|advert(?:s|isements?)?|for sale|show me actual|"
        r"ilan(?:lar|ları|lari)?|satılık|satilik|göster|goster|"
        r"объявлен(?:ие|ия|ий)|покажи|показать|в продаже|прода(?:же|ются))\b",
        low,
    )
    compare_words = re.search(
        r"\b(compare|comparison|versus|vs\.?|karşılaştır|karsilastir|kıyasla|kiyasla|"
        r"сравни(?:ть|те)?|сравнение|против)\b",
        low,
    )
    listing_superlative = re.search(
        r"\b(lowest[- ]?mileage|lowest km|cheapest|lowest[- ]?priced|newest|"
        r"en düşük kilometreli|en dusuk kilometreli|en az kilometreli|en ucuz|en yeni|"
        r"с минимальным пробегом|сам(?:ый|ая|ое) дешев\w*|сам(?:ый|ая|ое) нов\w*)\b",
        low,
    )
    if shop_words or listing_superlative:
        decision_mode = "SHOP"
    elif compare_words or len(targets) >= 2:
        decision_mode = "COMPARE"
    else:
        decision_mode = "DISCOVER"

    # Budget forms: £15,000; 15k GBP; 15000 pounds; 15 bin; 15 тыс.
    budget_match = re.search(r"£\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)", raw)
    if not budget_match:
        budget_match = re.search(r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*£", raw)
    if not budget_match:
        budget_match = re.search(
            r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:gbp|pounds?|sterling|sterlin|sterlinlik|"
            r"фунт(?:ов|а)?|стерлинг(?:ов|а)?)\b",
            low,
        )
    if not budget_match:
        bin_match = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*bin\b", low)
        if bin_match:
            amount = _parse_human_number(bin_match.group(1))
            if amount is not None:
                filters["budget"] = amount * 1000
        else:
            ru_thousand = re.search(r"\b([0-9]+(?:[.,][0-9]+)?)\s*(?:тыс|тысяч)\b", low)
            if ru_thousand:
                amount = _parse_human_number(ru_thousand.group(1))
                if amount is not None:
                    filters["budget"] = amount * 1000
    else:
        amount = _parse_human_number(budget_match.group(1))
        if amount is not None:
            filters["budget"] = amount

    # Common currency-less shorthand: "15k budget", "around 15k max",
    # "under 15k". Guard it with price/budget language so mileage such as
    # "under 60k km" is never mistaken for a purchase budget.
    if "budget" not in filters:
        shorthand_budget = re.search(
            r"(?:(?:budget|price|spend|cost|around|max(?:imum)?|under|below|up to|"
            r"bütçe|butce|bütçem|butcem|fiyat|до|бюджет)\D{0,18})"
            r"([0-9]+(?:[.,][0-9]+)?)\s*[kK]\b",
            low,
        )
        if shorthand_budget:
            tail = low[shorthand_budget.end():shorthand_budget.end() + 12]
            if not re.match(r"\s*(?:km|kilomet|км)", tail):
                amount = _parse_human_number(shorthand_budget.group(1))
                if amount is not None:
                    filters["budget"] = amount * 1000

    if "budget" not in filters:
        shorthand_budget_after = re.search(
            r"\b([0-9]+(?:[.,][0-9]+)?)\s*[kK]\s*"
            r"(?:budget|max(?:imum)?|to spend|spend|price|"
            r"bütçe|butce|bütçem|butcem|бюджет)\b",
            low,
        )
        if shorthand_budget_after:
            amount = _parse_human_number(shorthand_budget_after.group(1))
            if amount is not None:
                filters["budget"] = amount * 1000

    # Common mileage restrictions, including both prefix and suffix forms.
    km_patterns = [
        r"(?:under|below|max(?:imum)?|less than|no more than|altında|en fazla|maksimum|до|максимум|не более)\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:km|kilomet(?:er|re)?|км)",
        r"([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*(?:km|kilomet(?:er|re)?|км)\s*(?:altında|altinda|ve altı|ve alti|or less|or below|maximum|максимум|или меньше)",
    ]
    for pattern in km_patterns:
        km_match = re.search(pattern, low)
        if km_match:
            amount = _parse_human_number(km_match.group(1))
            if amount is not None:
                filters["max_km"] = amount
            break

    # When the user has already established that the number is mileage,
    # accept shorthand such as "not crazy mileage, like max 60k".
    if "max_km" not in filters:
        mileage_context = re.search(
            r"\b(?:mileage|kilomet(?:er|re)s?|kilometre|kilometer|пробег)\b",
            low,
        )
        if mileage_context:
            # Only inspect text AFTER the mileage cue. This prevents a budget
            # such as "under 15k" earlier in the sentence from being reused as
            # max_km in "under 15k and not crazy mileage, like max 60k".
            mileage_tail = low[mileage_context.end():]
            implied_km = re.search(
                r"\b(?:max(?:imum)?|under|below|less than|up to|"
                r"en fazla|altında|altinda|до|максимум|не более)\s*"
                r"([0-9]+(?:[.,][0-9]+)?\s*[kK])\b",
                mileage_tail,
            )
            if implied_km:
                amount = _parse_human_number(implied_km.group(1))
                if amount is not None:
                    filters["max_km"] = amount

    # Common minimum-year wording in EN/TR/RU.
    year_patterns = [
        r"\bс\s*((?:19|20)\d{2})\s*(?:года|г\.?|и новее)?\b",
        r"\b((?:19|20)\d{2})\s*(?:onwards|or newer|and newer|ve üzeri|ve uzeri|ve sonrası|ve sonrasi|и новее|или новее)\b",
        r"(?:minimum|min(?:imum)? year|from|since|en az|minimum yıl|min yıl|от|не старше)\s*((?:19|20)\d{2})",
        r"((?:19|20)\d{2})\s*(?:model ve üstü|model ve ustu|modelden yeni|года и новее)",
    ]
    for pattern in year_patterns:
        year_match = re.search(pattern, low)
        if year_match:
            filters["min_year"] = int(year_match.group(1))
            break

    # A bare model year attached to a vehicle-class noun means that exact model
    # year (e.g. "2026 SUVs", "2024 cars"). Do not reinterpret this as a
    # minimum-year request.
    if "min_year" not in filters and "max_year" not in filters:
        exact_year = re.search(
            r"\b((?:19|20)\d{2})\s*(?:model\s*)?(?:cars?|automobiles?|suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
            r"arabalar?|otomobiller?|suv|motosikletler?|скутеры?|мотоциклы?|автомобили?)\b",
            low,
        )
        if not exact_year:
            exact_year = re.search(
                r"\b(?:cars?|automobiles?|suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
                r"arabalar?|otomobiller?|suv|motosikletler?|скутеры?|мотоциклы?|автомобили?)\s*(?:model\s*)?((?:19|20)\d{2})\b",
                low,
            )
        if exact_year:
            filters["min_year"] = int(exact_year.group(1))
            filters["max_year"] = int(exact_year.group(1))

    # Transmission.
    if re.search(r"\b(automatic|otomatik|автомат(?:ическ\w*)?|акпп)\b", low):
        filters["transmissions"] = ["Automatic"]
    elif re.search(r"\b(manual|manuel|механик(?:а|ическая)?|мкпп)\b", low):
        filters["transmissions"] = ["Manual"]

    # Seller type.
    if re.search(r"\b(private sellers?|private cars?|individual sellers?|bireysel|özel satıcı(?:lar)?|ozel satici(?:lar)?|частн(?:ый|ого|ые) продав(?:ец|цы)|частник(?:и)?)\b", low):
        seller_mode = "individual"
    elif re.search(r"\b(dealers?|dealerships?|galler(?:y|ies)|galeri(?:ler)?(?:den)?|дилер(?:ы)?|автосалон(?:ы)?)\b", low):
        seller_mode = "gallery"

    # Buyer-oriented soft preferences. These map to the precomputed profile layer,
    # so they remain deterministic and fast in all three supported languages.
    if re.search(
        r"\b(economical|economic|fuel efficient|fuel-efficient|economy|cheap to run|"
        r"ekonomik|az yakan|tasarruflu|düşük tüketim|dusuk tuketim|"
        r"экономич\w*|экономн\w*|низкий расход)\b",
        low,
    ):
        preferences.append("priority:economy")

    if re.search(r"\b(reliable|reliability|most reliable|güvenilir|guvenilir|dayanıklı|dayanikli|sorunsuz|надёжн\w*|надежн\w*)\b", low):
        preferences.append("priority:reliability")

    if re.search(r"\b(luxury|premium|luxurious|lüks|luks|премиальн\w*|роскошн\w*)\b", low):
        preferences.append("priority:luxury")

    if re.search(r"\b(comfortable|comfort|konforlu|konfor|комфортн\w*|комфорт)\b", low):
        preferences.append("priority:comfort")

    if re.search(r"\b(sporty|performance|sportif|performans|спортивн\w*|динамичн\w*|производительн\w*)\b", low):
        preferences.append("priority:performance")

    if re.search(r"\b(practical|practicality|pratik|kullanışlı|kullanisli|практичн\w*)\b", low):
        preferences.append("priority:practicality")

    if re.search(r"\b(family car|family vehicle|for my family|aile arabası|aile arabasi|aile için|aile icin|семейн\w* автомобил\w*|для семьи)\b", low):
        preferences.append("use_case:family")

    if re.search(r"\b(commute|commuting|daily commute|işe gidip gel|ise gidip gel|günlük kullanım|gunluk kullanim|для поездок на работу|на каждый день|ежедневн\w*)\b", low):
        preferences.append("use_case:commute")

    # Vehicle/body type. Note the natural Turkish/Russian variants that were
    # previously missed (e.g. "küçük bir araç", "небольшую машину").
    if re.search(
        r"\b(small(?:\s+[a-z-]+){0,2}\s+(?:cars?|vehicles?)|city cars?|compact cars?|"
        r"küçük(?:\s+bir)?(?:\s+[a-zçğıöşü-]+){0,2}\s+(?:araba|otomobil|araç)|şehir arabası|sehir arabasi|kompakt araba|"
        r"маленьк\w*(?:\s+[а-яё-]+){0,2}\s+(?:машин\w*|автомобил\w*)|небольш\w*(?:\s+[а-яё-]+){0,2}\s+(?:машин\w*|автомобил\w*)|"
        r"компактн\w*\s+(?:машин\w*|автомобил\w*)|городск\w*\s+автомобил\w*)\b",
        low,
    ):
        preferences.append("vehicle_type:small_car")
    elif re.search(r"\b(cars?|automobiles?|araba(?:lar)?|otomobil(?:ler)?|машин\w*|автомобил\w*)\b", low):
        preferences.append("vehicle_type:car")
    elif re.search(r"\b(suvs?|кроссовер(?:ы)?|внедорожник(?:и)?)\b", low):
        preferences.append("vehicle_type:SUV")
    elif re.search(r"\b(crossovers?|crossover cars?)\b", low):
        preferences.append("vehicle_type:crossover")
    elif re.search(r"\b(pick-?ups?|pickups?|kamyonet(?:ler)?|пикап(?:ы)?)\b", low):
        preferences.append("vehicle_type:pickup")
    elif re.search(r"\b(motorcycles?|motosiklet(?:ler)?|мотоцикл(?:ы)?)\b", low):
        preferences.append("vehicle_type:motorcycle")
    elif re.search(r"\b(scooters?|skuters?|скутер(?:ы)?)\b", low):
        preferences.append("vehicle_type:scooter")

    # Listing-level ordering requests are deterministic state, not subjective
    # recommendations. They are applied only after every hard market filter.
    if re.search(r"\b(lowest[- ]?mileage|lowest km|en düşük kilometreli|en dusuk kilometreli|en az kilometreli|с минимальным пробегом)\b", low):
        preferences.append("listing_sort:lowest_km")
    elif re.search(r"\b(cheapest|lowest[- ]?priced|en ucuz|сам(?:ый|ая|ое) дешев\w*)\b", low):
        preferences.append("listing_sort:cheapest")
    elif re.search(r"\b(newest|en yeni|сам(?:ый|ая|ое) нов\w*)\b", low):
        preferences.append("listing_sort:newest")

    # Named single targets can safely be canonicalized here; multi-target COMPARE
    # is handled independently by _search_market_for_vehicle_targets.
    if len(targets) == 1:
        filters["brands"] = [targets[0]["brand"]]
        filters["models"] = [targets[0]["model"]]
        if targets[0].get("category"):
            filters["categories"] = [targets[0]["category"]]

    # Stable de-duplication while preserving preference order.
    preferences = list(dict.fromkeys(preferences))

    recognized = bool(filters or preferences or targets or compare_words or shop_words or seller_mode)
    if not recognized:
        return None

    return {
        "filters": sanitize_ai_filters(filters),
        "clear_filters": [],
        "seller_mode": seller_mode,
        "preferences": _canonicalize_buyer_preferences(preferences),
        "needs_clarification": False,
        "clarification_question": None,
        "decision_mode": decision_mode,
        "fast_path": True,
    }

def interpret_market_query(message, current_filters=None, language="TR", conversation_history=None):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    current_filters = current_filters or {}
    market_context = compact_market_context()

    instructions = """
You are the query interpreter for a North Cyprus vehicle-market assistant.

Your job is NOT to answer the user and NOT to invent listings.
Convert the user's latest message into structured changes to a vehicle search.

The deterministic search engine supports ONLY these filter fields:
budget, min_budget, brands, exclude_brands, models, exclude_models,
categories, exclude_categories, locations, exclude_locations,
companies, exclude_companies, transmissions, colors,
min_year, max_year, min_km, max_km.

Important rules:
- Return JSON only.
- Set "decision_mode" to exactly one of DISCOVER, COMPARE or SHOP.
  DISCOVER = broad vehicle/model discovery, recommendations, narrowing, "what can I buy?", or "more options?".
  COMPARE = comparing two or more models/brands, evaluating a chosen model in depth, or questions about resale/liquidity/price-pressure/trade-offs for models under discussion.
  SHOP = the buyer explicitly asks to see/find/show actual individual listings/ads/vehicles for sale, or asks about a specific advertised vehicle.
- Do NOT use SHOP merely because the buyer wants to buy a vehicle; SHOP requires listing-level intent.
- A bare model name that deepens the conversation is COMPARE, not SHOP.
- "filters" must contain ONLY constraints expressed or clearly modified
  by the latest user message.
- Do not repeat old filters merely because they appear in current_filters.
- Use null/empty omission rather than guessing.
- Prices are GBP.
- Convert "15k", "15 bin", "15 thousand" to 15000.
- Convert mileage expressions similarly.
- "Bireysel" means a private/individual seller.
- If the user wants galleries/dealers, set exclude_companies to ["Bireysel"].
- If the user wants private sellers, set companies to ["Bireysel"].
- If the user says either/both seller types are fine, put
  "seller_mode": "both" so the application can clear the prior seller filter.
- If the user explicitly removes a previous constraint, put its field name
  in "clear_filters".
- Preserve real market spellings when they are supplied in market_context.
- resolved_vehicle_mentions is deterministic application evidence from the live market.
  When it contains a named vehicle, use its canonical Brand/Model spelling rather than
  inventing a compound model name. Example: a resolved target may be
  Brand=Nissan, Model=Note, Category=e-Power even if the buyer wrote "Nissan Note e-Power".
- For COMPARE turns with multiple resolved_vehicle_mentions, do NOT try to express
  per-vehicle categories as one global categories filter; the application searches
  those targets independently.
- Do not translate brand/model names.
- Do not turn subjective ideas such as reliable, sporty, economical,
  family-friendly, small, luxurious, or good value into unsupported hard
  filters. Put those concepts in "preferences".
- IMPORTANT: vehicle classes are represented as canonical vehicle_type tags in "preferences"
  because market_base Category is variant-level, not a trustworthy body-type field. The application
  enforces explicit physical classes (car/SUV/crossover/pick-up/motorcycle/scooter) as STRICT
  constraints using the validated model-profile VehicleType/BodyStyle layer. "small car" remains
  a softer size/profile preference. Never place vehicle classes into Category unless that exact
  value is explicitly confirmed as a real market Category value.
- Use short canonical preference tags whenever possible so they persist cleanly across turns:
  vehicle_type:car, vehicle_type:SUV, vehicle_type:crossover, vehicle_type:pickup, vehicle_type:small_car,
  vehicle_type:motorcycle, use_case:commute, use_case:family, priority:economy,
  priority:reliability, priority:performance, priority:luxury, priority:comfort,
  priority:practicality.
- Treat words such as economical/economic/ekonomik/fuel-efficient/az yakan as priority:economy.
- Treat luxury/premium/lüks as priority:luxury.
- Treat comfortable/konforlu as priority:comfort and practical/pratik as priority:practicality.
- Treat sporty/sportif/performance/performance-oriented/performans as priority:performance.
- When the latest message changes vehicle type (for example SUV -> motorcycle or SUV -> small car), return only the NEW vehicle_type tag; the application will replace the previous vehicle_type preference.
- When the user says a previous vehicle type is no longer required (for example "SUV olmasına gerek yok"), add any_vehicle_type unless they also specify a replacement vehicle type in the same message.
- If the user explicitly says they have no brand/model preference, add "any_brand_model".
- If they explicitly say they have no year or mileage restriction, add "any_year_km".
- If they explicitly say any vehicle type is fine, add "any_vehicle_type".
- "preferences" is for useful soft intent that the deterministic filters
  cannot represent yet.
- "needs_clarification" should only be true when the latest message itself is genuinely
  ambiguous (for example an unclear number/unit or an unclear brand/model reference).
- A broad request such as "recommend me a car" is NOT an ambiguity. Set
  needs_clarification=false and let the application's guided narrowing flow handle it.
- Do not use clarification_question merely to ask for budget, year, vehicle type, brand,
  model or mileage because the guided narrowing flow handles those choices.
- "clarification_question" should be short and in the user's language.
- Use recent_conversation ONLY to resolve terse follow-ups/corrections such as "yes", "all of them",
  "more?", "Mercedes?", "No Mercedes", "GLA", "cheaper" or "newer". Do not re-create old hard
  filters from prose history; current_filters is the authoritative persisted hard state.
- A short bare brand/model availability challenge such as "Mercedes?", "No Mercedes", "no BMW?",
  "Mercedes yok mu?" means "are there any?" unless the user clearly expresses exclusion intent.
  Treat it as an INCLUDE/query for that brand/model, preserving the other active criteria.
- Only exclude a brand when intent is explicit, e.g. "I don't want Mercedes", "exclude Mercedes",
  "without Mercedes", "Mercedes istemiyorum", "Mercedes olmasın", "Mercedes hariç".
- If the user corrects a misunderstanding ("No, I meant is there not Mercedes"), prefer the corrected
  availability intent and clear any contradictory brand exclusion introduced by the prior turn.
"""

    resolved_vehicle_mentions = resolve_market_vehicle_mentions(message)

    user_payload = {
        "language": language,
        "latest_message": message,
        "current_filters": current_filters,
        "recent_conversation": sanitize_conversation_history(conversation_history, max_messages=10),
        "market_context": market_context,
        "resolved_vehicle_mentions": resolved_vehicle_mentions,
    }

    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "budget": {"type": ["number", "null"]},
                    "min_budget": {"type": ["number", "null"]},
                    "brands": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_brands": {"type": ["array", "null"], "items": {"type": "string"}},
                    "models": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_models": {"type": ["array", "null"], "items": {"type": "string"}},
                    "categories": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_categories": {"type": ["array", "null"], "items": {"type": "string"}},
                    "locations": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_locations": {"type": ["array", "null"], "items": {"type": "string"}},
                    "companies": {"type": ["array", "null"], "items": {"type": "string"}},
                    "exclude_companies": {"type": ["array", "null"], "items": {"type": "string"}},
                    "transmissions": {"type": ["array", "null"], "items": {"type": "string"}},
                    "colors": {"type": ["array", "null"], "items": {"type": "string"}},
                    "min_year": {"type": ["integer", "null"]},
                    "max_year": {"type": ["integer", "null"]},
                    "min_km": {"type": ["number", "null"]},
                    "max_km": {"type": ["number", "null"]},
                },
                "required": [
                    "budget", "min_budget", "brands", "exclude_brands",
                    "models", "exclude_models", "categories", "exclude_categories",
                    "locations", "exclude_locations", "companies", "exclude_companies",
                    "transmissions", "colors", "min_year", "max_year",
                    "min_km", "max_km"
                ],
            },
            "clear_filters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "seller_mode": {
                "type": ["string", "null"],
                "enum": ["individual", "gallery", "both", None],
            },
            "preferences": {
                "type": "array",
                "items": {"type": "string"},
            },
            "needs_clarification": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "decision_mode": {
                "type": "string",
                "enum": ["DISCOVER", "COMPARE", "SHOP"],
            },
        },
        "required": [
            "filters",
            "clear_filters",
            "seller_mode",
            "preferences",
            "needs_clarification",
            "clarification_question",
            "decision_mode",
        ],
        "additionalProperties": False,
    }

    response = _openai_post(
        payload={
            "model": OPENAI_MODEL,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 700,
            "instructions": instructions,
            "input": json.dumps(user_payload, ensure_ascii=False),
        },
        timeout=(1.5, 8.0),
    )

    response.raise_for_status()

    response_payload = response.json()
    response_text = extract_response_text(response_payload)

    if not response_text:
        raise ValueError("AI_INTERPRETER_EMPTY_RESPONSE")

    interpreted = json.loads(response_text)
    interpreted["filters"] = sanitize_ai_filters(
        interpreted.get("filters", {})
    )

    # Remove null values emitted because the strict schema requires every
    # filter property to be present.
    interpreted["filters"] = {
        key: value
        for key, value in interpreted["filters"].items()
        if value is not None
    }

    decision_mode = str(interpreted.get("decision_mode") or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"
    interpreted["decision_mode"] = decision_mode

    return interpreted


@app.route("/api/interpret", methods=["POST"])
def api_interpret_market_query():
    try:
        data = request.json or {}

        message = str(data.get("message", "")).strip()

        if not message:
            return jsonify({
                "success": False,
                "error": "MESSAGE_REQUIRED"
            }), 400

        result = interpret_market_query(
            message=message,
            current_filters=data.get("current_filters") or {},
            language=str(data.get("language", "TR")).upper(),
        )

        return jsonify({
            "success": True,
            **result
        })

    except requests.HTTPError as e:
        status_code = (
            e.response.status_code
            if e.response is not None
            else None
        )

        response_text = (
            e.response.text[:2000]
            if e.response is not None
            else ""
        )

        print(
            "OPENAI INTERPRETER HTTP ERROR:",
            status_code,
            response_text,
            flush=True
        )

        return jsonify({
            "success": False,
            "error": "AI_INTERPRETER_FAILED",
            "openai_status": status_code,
            "openai_message": response_text
        }), 502

    except RuntimeError as e:
        print("AI INTERPRETER CONFIG ERROR:", e, flush=True)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 503

    except Exception as e:
        print("AI INTERPRETER FAILED:", repr(e), flush=True)

        return jsonify({
            "success": False,
            "error": "AI_INTERPRETER_FAILED"
        }), 500


# =========================================================
# AI BUYING ASSISTANT - MARKET SEARCH ENGINE
# =========================================================

def market_search(
    budget=None,
    min_budget=None,
    brands=None,
    exclude_brands=None,
    models=None,
    exclude_models=None,
    categories=None,
    exclude_categories=None,
    locations=None,
    exclude_locations=None,
    companies=None,
    exclude_companies=None,
    transmissions=None,
    fuels=None,
    exclude_fuels=None,
    colors=None,
    min_year=None,
    max_year=None,
    min_km=None,
    max_km=None,
    limit=20,
    max_limit=100,
    analysis_mode=False
):
    if not MARKET_READY or market_df is None or market_df.empty:
        return {
            "success": False,
            "error": "MARKET_DATA_NOT_READY",
            "count": 0,
            "returned": 0,
            "results": []
        }

    filtered = market_df.copy()

    # -----------------------
    # HELPERS
    # -----------------------

    def normalize(value):
        return str(value).strip().casefold()

    def normalize_list(values):
        if not values:
            return []

        if not isinstance(values, list):
            values = [values]

        return [
            normalize(value)
            for value in values
            if value not in [None, ""]
        ]

    def contains_any(series, values):
        values = normalize_list(values)

        if not values:
            return pd.Series(
                True,
                index=series.index
            )

        normalized_series = (
            series
            .fillna("")
            .astype(str)
            .map(normalize)
        )

        mask = pd.Series(
            False,
            index=series.index
        )

        for value in values:
            mask = mask | normalized_series.str.contains(
                value,
                regex=False,
                na=False
            )

        return mask

    # -----------------------
    # PRICE
    # -----------------------

    if budget not in [None, ""]:
        filtered = filtered[
            filtered["Price"] <= float(budget)
        ]

    if min_budget not in [None, ""]:
        filtered = filtered[
            filtered["Price"] >= float(min_budget)
        ]

    # -----------------------
    # BRAND
    # -----------------------

    if brands:
        wanted = set(
            normalize_list(brands)
        )

        filtered = filtered[
            filtered["Brand"]
            .map(normalize)
            .isin(wanted)
        ]

    if exclude_brands:
        unwanted = set(
            normalize_list(exclude_brands)
        )

        filtered = filtered[
            ~filtered["Brand"]
            .map(normalize)
            .isin(unwanted)
        ]

    # -----------------------
    # MODEL
    # -----------------------

    if models:
        filtered = filtered[
            contains_any(
                filtered["Model"],
                models
            )
        ]

    if exclude_models:
        filtered = filtered[
            ~contains_any(
                filtered["Model"],
                exclude_models
            )
        ]

    # -----------------------
    # CATEGORY / VARIANT
    # -----------------------

    if categories:
        filtered = filtered[
            contains_any(
                filtered["Category"],
                categories
            )
        ]

    if exclude_categories:
        filtered = filtered[
            ~contains_any(
                filtered["Category"],
                exclude_categories
            )
        ]
    
    # -----------------------
    # LOCATION
    # -----------------------

    if locations:
        filtered = filtered[
            contains_any(
                filtered["Location"],
                locations
            )
        ]

    if exclude_locations:
        filtered = filtered[
            ~contains_any(
                filtered["Location"],
                exclude_locations
            )
        ]

    # -----------------------
    # COMPANY / DEALERSHIP
    # -----------------------

    if companies:
        filtered = filtered[
            contains_any(
                filtered["Company"],
                companies
            )
        ]

    if exclude_companies:
        filtered = filtered[
            ~contains_any(
                filtered["Company"],
                exclude_companies
            )
        ]

    # -----------------------
    # TRANSMISSION
    # -----------------------

    if transmissions:
        def transmission_key(value):
            value = normalize(value)
            aliases = {
                "automatic": "otomatik", "auto": "otomatik", "автомат": "otomatik",
                "автоматическая": "otomatik", "manual": "düz", "manuel": "düz",
                "duz": "düz", "механика": "düz", "механическая": "düz",
                "semi automatic": "yarı otomatik", "semi-automatic": "yarı otomatik",
                "yari otomatik": "yarı otomatik", "полуавтомат": "yarı otomatik",
            }
            return aliases.get(value, value)
        wanted = {transmission_key(value) for value in normalize_list(transmissions)}
        filtered = filtered[
            filtered["Transmission"]
            .fillna("")
            .astype(str)
            .map(transmission_key)
            .isin(wanted)
        ]

    # -----------------------
    # FUEL
    # -----------------------
    # Fuel is derived from the KKTCarabam listing URL by market.py and stored
    # canonically in market_base.csv. Accept English and Turkish user/planner
    # forms so EN/TR/RU conversations resolve to the same underlying values.

    def normalize_fuel_query(value):
        text = normalize(value)
        aliases = {
            "benzin": "petrol",
            "benzinli": "petrol",
            "gasoline": "petrol",
            "petrol": "petrol",
            "бензин": "petrol",
            "dizel": "diesel",
            "diesel": "diesel",
            "дизель": "diesel",
            "hibrit": "hybrid",
            "hybrid": "hybrid",
            "гибрид": "hybrid",
            "mild hibrit": "mild hybrid",
            "mild-hibrit": "mild hybrid",
            "mild hybrid": "mild hybrid",
            "мягкий гибрид": "mild hybrid",
            "plug in hibrit": "plug-in hybrid",
            "plug-in hibrit": "plug-in hybrid",
            "plug in hybrid": "plug-in hybrid",
            "plug-in hybrid": "plug-in hybrid",
            "подключаемый гибрид": "plug-in hybrid",
            "elektrik": "electric",
            "elektrikli": "electric",
            "electric": "electric",
            "электро": "electric",
            "электрический": "electric",
            "электричество": "electric",
            "ev": "electric",
        }
        return aliases.get(text, text)

    def fuel_mask(series, values):
        if not values:
            return pd.Series(True, index=series.index)
        if not isinstance(values, list):
            values = [values]
        wanted = [normalize_fuel_query(v) for v in values if v not in [None, ""]]
        normalized_series = series.fillna("").astype(str).map(normalize)
        mask = pd.Series(False, index=series.index)
        for value in wanted:
            # Deliberately use contains: "hybrid" includes mild/plug-in hybrid,
            # and "diesel" includes mild-hybrid diesel.
            mask = mask | normalized_series.str.contains(value, regex=False, na=False)
        return mask

    if fuels and "Fuel" in filtered.columns:
        filtered = filtered[fuel_mask(filtered["Fuel"], fuels)]

    if exclude_fuels and "Fuel" in filtered.columns:
        filtered = filtered[~fuel_mask(filtered["Fuel"], exclude_fuels)]

    # -----------------------
    # COLOR
    # -----------------------

    if colors:
        filtered = filtered[
            contains_any(
                filtered["Color"],
                colors
            )
        ]

    # -----------------------
    # YEAR
    # -----------------------

    if min_year not in [None, ""]:
        filtered = filtered[
            filtered["Year"] >= int(min_year)
        ]

    if max_year not in [None, ""]:
        filtered = filtered[
            filtered["Year"] <= int(max_year)
        ]

    # -----------------------
    # MILEAGE
    # -----------------------
    # Cars with missing KM remain in the dataset normally.
    #
    # BUT if the user specifically requests a mileage
    # limit, a car with unknown KM cannot be said to meet it.

    if max_km not in [None, ""]:
        filtered = filtered[
            filtered["KM"].notna()
            & (filtered["KM"] <= float(max_km))
        ]

    if min_km not in [None, ""]:
        filtered = filtered[
            filtered["KM"].notna()
            & (filtered["KM"] >= float(min_km))
        ]

    # -----------------------
    # SORT
    # -----------------------

    filtered = filtered.sort_values(
        ["Price", "Year"],
        ascending=[True, False]
    )

    total_count = len(filtered)

    # -----------------------
    # LIMIT
    # -----------------------

    try:
        limit = int(limit)
    except:
        limit = 20

    try:
        max_limit = int(max_limit)
    except:
        max_limit = 100

    max_limit = max(1, min(max_limit, 5000))

    limit = max(
        1,
        min(limit, max_limit)
    )

    # Assistant market intelligence must be calculated from the COMPLETE matching
    # market, never from a display-sized/truncated sample. Public listing endpoints
    # still use the normal limit/max_limit behavior.
    results_df = filtered.copy() if analysis_mode else filtered.head(limit).copy()

    # -----------------------
    # JSON-SAFE RESULTS
    # -----------------------

    # Vectorized serialization is materially faster than iterrows() when the
    # assistant asks for thousands of internal candidates.
    raw_records = results_df[[
        "Brand", "Model", "Category", "Year", "Price", "KM",
        "Company", "Location", "Transmission", "Fuel", "Color", "Image", "Link"
    ]].to_dict(orient="records")

    results = [
        {
            "brand": row["Brand"],
            "model": row["Model"],
            "category": row["Category"],
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "price": float(row["Price"]) if pd.notna(row["Price"]) else None,
            "km": int(row["KM"]) if pd.notna(row["KM"]) else None,
            "company": row["Company"],
            "location": row["Location"],
            "transmission": row["Transmission"],
            "fuel": row.get("Fuel", ""),
            "color": row["Color"],
            "image": row["Image"],
            "link": row["Link"],
        }
        for row in raw_records
    ]

    return {
        "success": True,
        "count": total_count,
        "returned": len(results),
        "results": results
    }



# =========================================================
# AI BUYING ASSISTANT - CONVERSATION / GROUNDED RESPONSE
# =========================================================

def apply_interpretation_to_filters(previous_filters, interpretation):
    """
    Merge the AI's incremental filter changes into the current search state.
    This mirrors the frontend behavior, but keeps the full assistant endpoint
    self-contained and deterministic.
    """
    next_filters = dict(previous_filters or {})
    incoming = sanitize_ai_filters(interpretation.get("filters", {}))

    next_filters.update(incoming)

    seller_mode = interpretation.get("seller_mode")

    if seller_mode == "individual":
        next_filters["companies"] = ["Bireysel"]
        next_filters.pop("exclude_companies", None)

    elif seller_mode == "gallery":
        next_filters["exclude_companies"] = ["Bireysel"]
        next_filters.pop("companies", None)

    elif seller_mode == "both":
        next_filters.pop("companies", None)
        next_filters.pop("exclude_companies", None)

    for key in interpretation.get("clear_filters", []) or []:
        if key in AI_FILTER_KEYS:
            next_filters.pop(key, None)

    return next_filters


def merge_preferences(previous_preferences, new_preferences):
    """
    Keep soft preferences across turns, but treat vehicle type as a replaceable
    conversational state rather than an accumulating tag.
    """
    previous = [str(v).strip() for v in (previous_preferences or []) if str(v).strip()]
    incoming = [str(v).strip() for v in (new_preferences or []) if str(v).strip()]
    incoming_cf = [v.casefold() for v in incoming]

    incoming_vehicle_types = [
        v for v in incoming if v.casefold().startswith("vehicle_type:")
    ]
    incoming_any_vehicle = "any_vehicle_type" in incoming_cf
    incoming_listing_sorts = [
        v for v in incoming if v.casefold().startswith("listing_sort:")
    ]

    if incoming_listing_sorts:
        previous = [v for v in previous if not v.casefold().startswith("listing_sort:")]

    # A new explicit vehicle type replaces the old one. "Any vehicle type"
    # clears all old vehicle-type restrictions.
    if incoming_vehicle_types or incoming_any_vehicle:
        previous = [
            v for v in previous
            if not v.casefold().startswith("vehicle_type:")
            and v.casefold() != "any_vehicle_type"
        ]

    # If a concrete new type is supplied, do not retain any_vehicle_type.
    if incoming_vehicle_types:
        incoming = [v for v in incoming if v.casefold() != "any_vehicle_type"]

    merged = []
    seen = set()
    for value in previous + incoming:
        key = value.casefold()
        if key not in seen:
            merged.append(value)
            seen.add(key)

    return merged


def _preference_flags(preferences):
    prefs = {str(p).strip().casefold() for p in (preferences or []) if str(p).strip()}
    return {
        "has_vehicle_type": any(p.startswith("vehicle_type:") for p in prefs) or "any_vehicle_type" in prefs,
        "has_use_case": any(p.startswith("use_case:") for p in prefs),
        "any_brand_model": "any_brand_model" in prefs,
        "any_year_km": "any_year_km" in prefs,
    }


def _strict_vehicle_type_from_preferences(preferences):
    """Return the explicit physical vehicle class, if one is active."""
    for pref in reversed(_canonicalize_buyer_preferences(preferences)):
        p = str(pref or "").strip().casefold()
        if not p.startswith("vehicle_type:"):
            continue
        requested = p.split(":", 1)[1]
        if requested == "small_car":
            return "car"
        if requested in {"car", "suv", "crossover", "pickup", "pick-up", "motorcycle", "motosiklet", "scooter"}:
            return requested
    return None


def _apply_strict_vehicle_type_to_search_result(search_result, preferences):
    """Enforce explicit physical vehicle classes with the validated model profiles.

    This is deliberately separate from market_base Category: Category is variant-level
    and must never be repurposed/inferred as a body type.
    """
    requested = _strict_vehicle_type_from_preferences(preferences)
    if not requested or not search_result.get("success"):
        return search_result
    if not MODEL_PROFILE_READY or not MODEL_PROFILE_LOOKUP:
        # Fail closed for class-sensitive discovery rather than leaking boats/bikes
        # into a car request when the taxonomy layer is unavailable.
        result = dict(search_result)
        result["count"] = 0
        result["returned"] = 0
        result["results"] = []
        result["vehicle_type_filter_unavailable"] = True
        return result

    kept = []
    for item in search_result.get("results", []) or []:
        key = (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold())
        profile = MODEL_PROFILE_LOOKUP.get(key)
        if profile and _profile_matches_vehicle_type(profile, requested):
            kept.append(item)

    result = dict(search_result)
    result["count"] = len(kept)
    result["returned"] = len(kept)
    result["results"] = kept
    result["strict_vehicle_type"] = requested
    return result


def _asks_reliability_question(message):
    low = str(message or "").casefold()
    return bool(re.search(r"\b(reliable|reliability|most reliable|güvenilir|guvenilir|dayanıklı|dayanikli|sorunsuz|надёжн\w*|надежн\w*)\b", low))


def _reliability_scope_answer(language, filters):
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None
    if language == "TR":
        scope = f" {budget_text} bütçeniz içindeki" if budget_text else ""
        return (
            "Güvenilirliği yalnızca Kuzey Kıbrıs ilan verilerinden güvenilir biçimde belirleyemem; "
            "bunun için uzun dönem güvenilirlik, arıza/servis ve kullanıcı verileri gibi dış kaynaklar gerekir. "
            f"Yine de{scope} seçenekleri güncel fiyat, yaş, kilometre ve gözlenen yeniden satış piyasası davranışına göre karşılaştırabilirim; "
            "dış güvenilirlik kanıtı olmadan bir modeli ‘en güvenilir’ diye etiketlemem."
        )
    if language == "RU":
        scope = f" в пределах бюджета {budget_text}" if budget_text else ""
        return (
            "Надёжность нельзя достоверно определить только по объявлениям Северного Кипра: для этого нужны внешние данные "
            "о долгосрочной надёжности, ремонтах/сервисе и опыте владельцев. "
            f"Я могу сравнить варианты{scope} по текущей цене, возрасту, пробегу и наблюдаемому поведению на рынке перепродажи, "
            "но не буду называть модель «самой надёжной» без таких внешних доказательств."
        )
    scope = f" within your {budget_text} budget" if budget_text else ""
    return (
        "Reliability isn't something I can determine reliably from North Cyprus listing data alone; "
        "it requires external evidence such as long-term reliability, repair/service and owner data. "
        f"I can still compare the options{scope} by current price, age, mileage and observed resale-market behaviour, "
        "but I won't label one model ‘most reliable’ without that external evidence."
    )


def _listing_sort_mode(preferences):
    for pref in reversed(preferences or []):
        p = str(pref or "").strip().casefold()
        if p == "listing_sort:lowest_km":
            return "lowest_km"
        if p == "listing_sort:cheapest":
            return "cheapest"
        if p == "listing_sort:newest":
            return "newest"
    return None


def guided_narrowing_question(filters, preferences, count, language="TR", message=""):
    """Only guide when the request is genuinely too broad to be useful yet."""
    flags = _preference_flags(preferences)

    has_budget = filters.get("budget") not in [None, ""]
    has_year = bool(
        filters.get("min_year") not in [None, ""] or
        filters.get("max_year") not in [None, ""] or
        flags["any_year_km"]
    )
    has_vehicle_type = flags["has_vehicle_type"]
    has_brand_model = bool(
        filters.get("brands") or filters.get("models") or
        filters.get("exclude_brands") or filters.get("exclude_models") or
        flags["any_brand_model"]
    )
    has_other_hard_constraint = bool(
        filters.get("max_km") not in [None, ""] or
        filters.get("min_km") not in [None, ""] or
        filters.get("transmissions") or
        filters.get("locations") or
        filters.get("companies") or
        filters.get("exclude_companies")
    )

    # Once the buyer has provided two meaningful dimensions, start helping with
    # real market options instead of automatically asking another question.
    # A genuine buying preference (economy, luxury, practicality, family use, etc.)
    # counts as a dimension too: "£15k economical" is already useful enough.
    has_soft_priority = any(
        str(p).strip().casefold().startswith(("priority:", "use_case:"))
        for p in (preferences or [])
    )
    supplied = sum([
        bool(has_budget),
        bool(has_year),
        bool(has_vehicle_type),
        bool(has_brand_model),
        bool(has_other_hard_constraint),
        bool(has_soft_priority),
    ])
    if supplied >= 2:
        return None

    copy = {
        "TR": {
            "intro": "Size daha isabetli öneriler sunabilmem için aramanızı biraz daraltmanızı öneririm.",
            "budget": "maksimum bütçenizi",
            "year": "minimum model yılı beklentinizi",
            "type": "araç tipini (ör. SUV, pick-up, otomobil veya motosiklet)",
        },
        "EN": {
            "intro": "To give you more relevant recommendations, I'd suggest narrowing the search a little.",
            "budget": "your maximum budget",
            "year": "your minimum year requirement",
            "type": "vehicle type (e.g. SUV, pickup, car or motorcycle)",
        },
        "RU": {
            "intro": "Чтобы дать более точные рекомендации, я бы предложил немного сузить поиск.",
            "budget": "максимальный бюджет",
            "year": "минимальный год выпуска",
            "type": "тип транспорта (например SUV, пикап, автомобиль или мотоцикл)",
        },
    }
    t = copy.get(language, copy["TR"])
    low_message = str(message or "").strip().casefold()

    # Ask for the missing thing the buyer actually referred to.
    asks_budget_fit = bool(re.search(
        r"\b(?:what|which).{0,25}(?:cars?|vehicles?).{0,25}(?:fit|within|under).{0,12}(?:my\s+)?budget\b|"
        r"\bbudget\w*.{0,35}(?:cars?|vehicles?)\b|"
        r"\bbütçe\w*.{0,35}(?:hangi|araç|arac|araba|öner|oner)\w*|"
        r"\b(?:hangi|araç|arac|araba)\w*.{0,35}bütçe\w*|"
        r"\bбюджет\w*.{0,35}(?:машин|автомоб|подойд|вариант)\w*",
        low_message,
        re.IGNORECASE,
    ))
    if asks_budget_fit and not has_budget:
        return {
            "TR": "Tabii — maksimum bütçeniz nedir?",
            "RU": "Конечно — какой у вас максимальный бюджет?",
            "EN": "Sure — what's your maximum budget?",
        }.get(language, "Sure — what's your maximum budget?")

    missing = []
    if not has_budget:
        missing.append(t["budget"])
    if not has_year:
        missing.append(t["year"])
    if not has_vehicle_type:
        missing.append(t["type"])

    if not missing:
        return None

    if language == "TR":
        details = ", ".join(missing)
        details = details[:1].upper() + details[1:] if details else details
        return f'{t["intro"]} ' + details + " belirtebilirsiniz; bunlardan biri veya birkaçı yeterli olabilir."
    if language == "RU":
        return f'{t["intro"]} ' + ", ".join(missing) + ". Можно указать один или несколько из этих параметров."
    return f'{t["intro"]} You can add ' + ", ".join(missing) + "; one or more of these may be enough."


def _group_market_models(results, max_groups=350):
    """Aggregate real matching listings into factual brand/model market options."""
    grouped = {}

    for item in results or []:
        brand = str(item.get("brand") or "").strip()
        model = str(item.get("model") or "").strip()
        if not brand or not model:
            continue

        key = (brand.casefold(), model.casefold())
        bucket = grouped.setdefault(key, {
            "brand": brand,
            "model": model,
            "count": 0,
            "prices": [],
            "years": [],
            "year_prices": {},
            "_items": [],
            "kms": [],
            "transmissions": set(),
            "locations": set(),
        })
        bucket["count"] += 1
        bucket["_items"].append(item)

        parsed_price = None
        parsed_year = None

        try:
            if item.get("price") is not None:
                parsed_price = float(item["price"])
                bucket["prices"].append(parsed_price)
        except (TypeError, ValueError):
            parsed_price = None

        try:
            if item.get("year") is not None:
                parsed_year = int(item["year"])
                bucket["years"].append(parsed_year)
        except (TypeError, ValueError):
            parsed_year = None

        if parsed_price is not None and parsed_year is not None:
            bucket["year_prices"].setdefault(parsed_year, []).append(parsed_price)
        try:
            if item.get("km") is not None:
                bucket["kms"].append(int(item["km"]))
        except (TypeError, ValueError):
            pass

        transmission = str(item.get("transmission") or "").strip()
        location = str(item.get("location") or "").strip()
        if transmission:
            bucket["transmissions"].add(transmission)
        if location:
            bucket["locations"].add(location)

    summaries = []
    for bucket in grouped.values():
        prices = bucket.pop("prices")
        years = bucket.pop("years")
        year_prices = bucket.pop("year_prices")
        raw_items = bucket.pop("_items")
        kms = bucket.pop("kms")
        bucket["transmissions"] = sorted(bucket["transmissions"])
        bucket["locations"] = sorted(bucket["locations"])
        bucket["starting_price"] = min(prices) if prices else None
        bucket["highest_price"] = max(prices) if prices else None
        bucket["oldest_year"] = min(years) if years else None
        bucket["newest_year"] = max(years) if years else None

        newest_year = bucket["newest_year"]
        newest_prices = year_prices.get(newest_year, []) if newest_year is not None else []
        bucket["newest_year_starting_price"] = min(newest_prices) if newest_prices else None
        bucket["newest_year_highest_price"] = max(newest_prices) if newest_prices else None
        bucket["newest_year_count"] = len(newest_prices)

        if prices and years and year_prices:
            overall_start = bucket["starting_price"]
            cheapest_years = [
                year for year, vals in year_prices.items()
                if vals and min(vals) == overall_start
            ]
            bucket["starting_price_year"] = max(cheapest_years) if cheapest_years else None
        else:
            bucket["starting_price_year"] = None

        bucket["lowest_km"] = min(kms) if kms else None
        bucket["highest_km"] = max(kms) if kms else None

        # Conservative price-only opportunity signal. Prefer same-year + same
        # category peers when at least 3 exist; otherwise use same-year peers.
        #
        # The opportunity must also be relevant to the buyer's current search.
        # When newer matching years exist, do not surface an old cheap car merely
        # because it is far below the median for its own year.
        best_deal = None
        relevance_floor_year = None
        if years:
            max_matching_year = max(years)
            # Keep the signal close to the newest cars that actually satisfy the
            # buyer's current hard filters (budget/km/etc.).
            relevance_floor_year = max_matching_year - 2

        for candidate in raw_items:
            try:
                candidate_price = float(candidate.get("price"))
                candidate_year = int(candidate.get("year"))
            except (TypeError, ValueError):
                continue
            if candidate_price <= 0:
                continue
            if relevance_floor_year is not None and candidate_year < relevance_floor_year:
                continue

            candidate_category = str(candidate.get("category") or "").strip().casefold()
            same_year = []
            same_year_category = []
            for peer in raw_items:
                try:
                    peer_price = float(peer.get("price"))
                    peer_year = int(peer.get("year"))
                except (TypeError, ValueError):
                    continue
                if peer_price <= 0 or peer_year != candidate_year:
                    continue
                same_year.append(peer_price)
                peer_category = str(peer.get("category") or "").strip().casefold()
                if candidate_category and peer_category == candidate_category:
                    same_year_category.append(peer_price)

            benchmark_prices = same_year_category if len(same_year_category) >= 3 else same_year
            if len(benchmark_prices) < 3:
                continue

            median_ask = float(pd.Series(benchmark_prices).median())
            if median_ask <= 0 or candidate_price >= median_ask:
                continue

            below_median_pct = (median_ask - candidate_price) / median_ask
            if below_median_pct < 0.05:
                continue

            deal = {
                "year": candidate_year,
                "price": candidate_price,
                "median_asking_price": median_ask,
                "below_median_pct": below_median_pct,
                "comparison_count": len(benchmark_prices),
                "comparison_scope": "same_year_category" if len(same_year_category) >= 3 else "same_year",
                "km": candidate.get("km"),
                "category": candidate.get("category"),
                "company": candidate.get("company"),
                "link": candidate.get("link"),
            }
            if best_deal is None or deal["below_median_pct"] > best_deal["below_median_pct"]:
                best_deal = deal

        bucket["potential_value_listing"] = best_deal
        summaries.append(bucket)

    summaries.sort(
        key=lambda x: (
            -int(x.get("count") or 0),
            -(int(x.get("newest_year") or 0)),
            float(x.get("starting_price") or 10**12),
        )
    )
    return summaries[:max_groups]


_MODEL_QUALIFICATION_CACHE = {}
_MODEL_QUALIFICATION_CACHE_MAX = 200


def _qualification_cache_key(filters, preferences, model_market):
    # Includes the actual market families/aggregates, so a refreshed market naturally
    # produces a different key without coupling this cache to the valuation dataset.
    market_signature = tuple(
        (
            str(m.get("brand") or "").casefold(),
            str(m.get("model") or "").casefold(),
            int(m.get("count") or 0),
            int(m.get("newest_year") or 0),
            float(m.get("newest_year_starting_price") or 0),
            float(m.get("starting_price") or 0),
        )
        for m in model_market
    )
    relevant_prefs = tuple(sorted(
        str(p).strip().casefold()
        for p in (preferences or [])
        if str(p).strip().casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
    ))
    hard_signature = json.dumps(filters or {}, ensure_ascii=False, sort_keys=True, default=str)
    return (hard_signature, relevant_prefs, market_signature)


def shortlist_models_for_preferences(message, language, filters, preferences, results):
    """Qualify/rank real model families against soft intent.

    Production path is deterministic via buyer_model_profiles.csv. A bounded LLM
    fallback exists only while profiles are unavailable/incomplete and is never
    allowed to hold a live request for tens of seconds.
    """
    preferences = _canonicalize_buyer_preferences(preferences)
    relevant_preferences = [
        p for p in preferences
        if str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
    ]

    model_market = _group_market_models(results)
    if not model_market:
        return [], [], []
    if not relevant_preferences:
        return list(results or []), [], model_market

    deterministic = _deterministic_profile_shortlist(model_market, relevant_preferences, max_models=30)
    if deterministic:
        selected_keys = {(m["brand"].casefold(), m["model"].casefold()) for m in deterministic}
        qualified = [
            item for item in (results or [])
            if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_keys
        ]
        reasons = [{"brand": m["brand"], "model": m["model"], "reason": "profile_match"} for m in deterministic]
        return qualified, reasons, deterministic

    # If the production profile catalogue is loaded but nothing matches, respect that
    # result rather than asking a live generative model to override stable taxonomy.
    if MODEL_PROFILE_READY:
        return [], [], []

    # Temporary resilience path for deployments before buyer_model_profiles.csv exists.
    # Keep the latency budget tight; if the external model is slow, return the hard-filtered
    # market immediately rather than making the product appear broken.
    cache_key = _qualification_cache_key(filters, preferences, model_market)
    cached = _MODEL_QUALIFICATION_CACHE.get(cache_key)
    if cached is not None:
        selected_keys, reasons, selected_summaries = cached
        selected_set = set(selected_keys)
        qualified = [item for item in (results or []) if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_set]
        return qualified, list(reasons), list(selected_summaries)

    qualification_market = model_market[:120]
    instructions = """
Return JSON only: {"models":[{"brand":"...","model":"...","reason":"..."}]}.
Select only exact supplied model candidates matching the soft preferences.
Vehicle type is mandatory. Economy/luxury/comfort/performance/practicality/family/commute are broad model-level positioning only.
Never infer listing condition, reliability or value retention. Select up to 20.
"""
    payload = {
        "soft_preferences": relevant_preferences,
        "model_candidates": [{"brand": m["brand"], "model": m["model"]} for m in qualification_market],
    }
    try:
        response = _openai_post(
            payload={
                "model": OPENAI_MODEL, "reasoning": {"effort": "none"},
                "max_output_tokens": 450, "instructions": instructions,
                "input": json.dumps(payload, ensure_ascii=False),
            },
            timeout=(1.0, 3.0),
        )
        response.raise_for_status()
        parsed = json.loads(extract_response_text(response.json()).strip())
    except Exception as exc:
        print(f"MODEL_QUALIFICATION_FAST_FALLBACK: {exc}", flush=True)
        # Hard-filtered candidates are safer than a long timeout. The response remains
        # grounded; it is simply less preference-specific until profiles are deployed.
        return list(results or []), [], model_market[:20]

    market_by_key = {(m["brand"].casefold(), m["model"].casefold()): m for m in qualification_market}
    selected_keys, reasons, selected_summaries = [], [], []
    for item in parsed.get("models", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        brand, model = str(item.get("brand") or "").strip(), str(item.get("model") or "").strip()
        key = (brand.casefold(), model.casefold())
        if key in market_by_key and key not in selected_keys:
            selected_keys.append(key); selected_summaries.append(market_by_key[key])
            reasons.append({"brand": brand, "model": model, "reason": str(item.get("reason") or "").strip()})
        if len(selected_keys) >= 20:
            break

    if not selected_keys:
        return list(results or []), [], model_market[:20]
    selected_set = set(selected_keys)
    qualified = [item for item in (results or []) if (str(item.get("brand") or "").strip().casefold(), str(item.get("model") or "").strip().casefold()) in selected_set]
    if len(_MODEL_QUALIFICATION_CACHE) >= _MODEL_QUALIFICATION_CACHE_MAX:
        _MODEL_QUALIFICATION_CACHE.pop(next(iter(_MODEL_QUALIFICATION_CACHE)))
    _MODEL_QUALIFICATION_CACHE[cache_key] = (tuple(selected_keys), tuple(reasons), tuple(selected_summaries))
    return qualified, reasons, selected_summaries


def _select_model_options(model_summaries, reasons, filters, max_options=8):
    """Build a diverse factual option set; AI will normally surface only 3-5."""
    if not model_summaries:
        return []

    reason_map = {
        (str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold()): x.get("reason", "")
        for x in (reasons or [])
    }

    enriched = []
    for summary in model_summaries:
        item = dict(summary)
        item["reason"] = reason_map.get(
            (item["brand"].casefold(), item["model"].casefold()),
            "",
        )
        enriched.append(item)

    # AI qualifies soft concepts; deterministic Python ordering decides what is surfaced.
    # This makes identical market/filter state stable and favours newer options rather than
    # allowing model ordering to vary from one generation call to another.
    if reasons:
        budget = filters.get("budget")
        try:
            ceiling = float(budget) if budget not in [None, ""] else None
        except (TypeError, ValueError):
            ceiling = None

        def qualified_rank(x):
            profile_score = int(x.get("_profile_score") or 0)
            newest = int(x.get("newest_year") or 0)
            # For budgeted discovery, compare the price of the newest reachable year,
            # not the cheapest old example in the model family.
            anchor_price = x.get("newest_year_starting_price")
            if anchor_price in [None, ""]:
                anchor_price = x.get("starting_price")
            try:
                anchor_price = float(anchor_price)
            except (TypeError, ValueError):
                anchor_price = 10**12
            budget_distance = abs(ceiling - anchor_price) if ceiling is not None else anchor_price
            return (-profile_score, -newest, budget_distance, -min(int(x.get("count") or 0), 100), x["brand"].casefold(), x["model"].casefold())

        enriched.sort(key=qualified_rank)
        return enriched[:max_options]

    # Without a soft preference, provide a useful spread: represented, newest, and near budget.
    chosen, seen = [], set()

    def add(items):
        for item in items:
            key = (item["brand"].casefold(), item["model"].casefold())
            if key not in seen:
                chosen.append(item)
                seen.add(key)
            if len(chosen) >= max_options:
                return

    add(sorted(enriched, key=lambda x: (-int(x.get("count") or 0), -(int(x.get("newest_year") or 0))))[:3])
    add(sorted(enriched, key=lambda x: (-(int(x.get("newest_year") or 0)), float(x.get("starting_price") or 10**12)))[:3])

    budget = filters.get("budget")
    if budget not in [None, ""]:
        try:
            ceiling = float(budget)
            add(sorted(
                enriched,
                key=lambda x: abs(ceiling - float(x.get("starting_price") or 0))
            )[:4])
        except (TypeError, ValueError):
            pass

    add(enriched)
    return chosen[:max_options]


def select_assistant_candidates(results, filters, max_candidates=3):
    """Choose factual listing rows only for explicit listing-level follow-up."""
    if not results:
        return []

    clean = list(results)
    budget = filters.get("budget")
    chosen, seen = [], set()

    def add(items):
        for item in items:
            identity = item.get("link") or (
                item.get("brand"), item.get("model"), item.get("year"), item.get("price")
            )
            if identity not in seen:
                chosen.append(item)
                seen.add(identity)
            if len(chosen) >= max_candidates:
                return

    if budget not in [None, ""]:
        try:
            ceiling = float(budget)
            add(sorted(clean, key=lambda x: abs(ceiling - float(x.get("price") or 0)))[:8])
        except (TypeError, ValueError):
            pass

    add(sorted(clean, key=lambda x: (-(int(x.get("year") or 0)), float(x.get("price") or 0)))[:8])
    add(sorted(
        [x for x in clean if x.get("km") is not None],
        key=lambda x: (int(x.get("km") or 0), -int(x.get("year") or 0)),
    )[:8])
    add(sorted(clean, key=lambda x: float(x.get("price") or 0)))
    return chosen[:max_candidates]



def _json_number(value, integer=False):
    if value is None or pd.isna(value):
        return None
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


def _vehicle_type_preference(preferences):
    for pref in preferences or []:
        value = str(pref or "").strip()
        if value.casefold().startswith("vehicle_type:"):
            return value.split(":", 1)[1].strip()
    return None


def _buyer_vehicle_type_matches(series, requested_type):
    """
    Deterministic only where Category Master gives us a defensible mapping.
    Ambiguous concepts such as small_car/crossover remain AI qualifications.
    """
    if not requested_type:
        return pd.Series(True, index=series.index)

    requested = str(requested_type).strip().casefold()
    normalized = series.fillna("").astype(str).str.strip().str.casefold()

    exact_map = {
        # SUV / pickup / crossover / small-car are model-profile/body-class
        # qualifications. They have already been enforced before enrichment,
        # so do not re-filter Buyer Intelligence using Category Master's broader
        # VehicleType field here.
        "motorcycle": {"motosiklet"},
        "motosiklet": {"motosiklet"},
        "atv": {"atv & utv"},
        "utv": {"atv & utv"},
        "classic": {"klasik araçlar"},
        "boat": {"deniz araçları"},
    }

    allowed = exact_map.get(requested)
    if not allowed:
        return pd.Series(True, index=series.index)

    return normalized.isin(allowed)


def _matching_buyer_model_rows(model_option, filters, preferences):
    if (
        not BUYER_INTELLIGENCE_READY
        or buyer_model_df is None
        or buyer_model_df.empty
    ):
        return pd.DataFrame()

    brand = str(model_option.get("brand") or "").strip().casefold()
    model = str(model_option.get("model") or "").strip().casefold()

    rows = buyer_model_df[
        (buyer_model_df["Brand"].astype(str).str.casefold() == brand)
        & (buyer_model_df["Model"].astype(str).str.casefold() == model)
    ].copy()

    if rows.empty:
        return rows

    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    budget = filters.get("budget")
    min_budget = filters.get("min_budget")

    if min_year not in [None, ""]:
        rows = rows[rows["Year"] >= int(min_year)]
    if max_year not in [None, ""]:
        rows = rows[rows["Year"] <= int(max_year)]
    if budget not in [None, ""]:
        rows = rows[
            rows["CurrentStartingPrice"].notna()
            & (rows["CurrentStartingPrice"] <= float(budget))
        ]
    if min_budget not in [None, ""]:
        # Keep a year if its market range can reach the buyer's floor.
        rows = rows[
            rows["CurrentHighestPrice"].notna()
            & (rows["CurrentHighestPrice"] >= float(min_budget))
        ]

    requested_type = _vehicle_type_preference(preferences)
    if requested_type and not rows.empty:
        rows = rows[
            _buyer_vehicle_type_matches(
                rows["VehicleType"], requested_type
            )
        ]

    return rows


def _matching_buyer_category_rows(model_option, filters, preferences, hard_results):
    if (
        not BUYER_INTELLIGENCE_READY
        or buyer_category_df is None
        or buyer_category_df.empty
    ):
        return pd.DataFrame()

    brand = str(model_option.get("brand") or "").strip()
    model = str(model_option.get("model") or "").strip()
    brand_cf = brand.casefold()
    model_cf = model.casefold()

    # Restrict variants to CategoryDetail strings actually represented in the
    # current hard-filtered listing result set for this model.
    observed_categories = {
        str(x.get("category") or "").strip().casefold()
        for x in (hard_results or [])
        if str(x.get("brand") or "").strip().casefold() == brand_cf
        and str(x.get("model") or "").strip().casefold() == model_cf
        and str(x.get("category") or "").strip()
    }

    rows = buyer_category_df[
        (buyer_category_df["Brand"].astype(str).str.casefold() == brand_cf)
        & (buyer_category_df["Model"].astype(str).str.casefold() == model_cf)
    ].copy()

    if rows.empty:
        return rows

    if observed_categories:
        detail_cf = rows["CategoryDetail"].astype(str).str.casefold()
        mask = detail_cf.map(
            lambda detail: any(
                detail and detail in observed
                for observed in observed_categories
            )
        )
        rows = rows[mask]

    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    budget = filters.get("budget")

    if min_year not in [None, ""]:
        rows = rows[rows["Year"] >= int(min_year)]
    if max_year not in [None, ""]:
        rows = rows[rows["Year"] <= int(max_year)]
    if budget not in [None, ""]:
        rows = rows[
            rows["CurrentStartingPrice"].notna()
            & (rows["CurrentStartingPrice"] <= float(budget))
        ]

    requested_type = _vehicle_type_preference(preferences)
    if requested_type and not rows.empty:
        rows = rows[
            _buyer_vehicle_type_matches(
                rows["VehicleType"], requested_type
            )
        ]

    return rows


def enrich_model_options_with_buyer_intelligence(
    model_options,
    filters,
    preferences,
    hard_results,
):
    """
    Attach proprietary Cyprus-market evidence to the real model options.

    This does NOT calculate a universal score. It exposes transparent current
    market, liquidity and asking-price-pressure evidence for the response model.
    """
    if not model_options:
        return []

    enriched = []

    for option in model_options:
        item = dict(option)
        model_rows = _matching_buyer_model_rows(
            item, filters, preferences
        )

        if not model_rows.empty:
            model_rows = model_rows.sort_values(
                "Year", ascending=False
            )
            newest = model_rows.iloc[0]

            # Historical evidence repeats across model-year rows, so take the
            # newest matching row as the carrier of those model-level fields.
            item["buyer_intelligence"] = {
                "available_years": sorted(
                    [
                        int(x) for x in model_rows["Year"].dropna().unique()
                    ],
                    reverse=True,
                ),
                "newest_affordable_year": _json_number(
                    model_rows["Year"].max(), integer=True
                ),
                "newest_affordable_year_starting_price": _json_number(
                    newest.get("CurrentStartingPrice")
                ),
                "newest_affordable_year_median_price": _json_number(
                    newest.get("CurrentMedianPrice")
                ),
                "newest_affordable_year_median_km": _json_number(
                    newest.get("CurrentMedianKM")
                ),
                "historical_distinct_listings": _json_number(
                    newest.get("Buyer_HistoricalDistinctListings"),
                    integer=True,
                ),
                "median_observed_days_to_exit": _json_number(
                    newest.get("Buyer_MedianObservedDaysToExit")
                ),
                "exit_30_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin30DaysRate")
                ),
                "exit_30_eligible": _json_number(
                    newest.get("Buyer_Exit30EligibleListings"),
                    integer=True,
                ),
                "exit_60_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin60DaysRate")
                ),
                "exit_60_eligible": _json_number(
                    newest.get("Buyer_Exit60EligibleListings"),
                    integer=True,
                ),
                "exit_90_rate": _json_number(
                    newest.get("Buyer_ObservedExitWithin90DaysRate")
                ),
                "exit_90_eligible": _json_number(
                    newest.get("Buyer_Exit90EligibleListings"),
                    integer=True,
                ),
                "liquidity_confidence": str(
                    newest.get("Buyer_LiquidityEvidenceConfidence") or ""
                ),
                "liquidity_evidence_level": str(
                    newest.get("LiquidityEvidenceLevel") or ""
                ),
                "price_reduction_rate": _json_number(
                    newest.get("Buyer_PriceReductionRate")
                ),
                "price_pressure_eligible": _json_number(
                    newest.get("Buyer_PricePressureEligibleListings"),
                    integer=True,
                ),
                "median_reduction_pct_when_reduced": _json_number(
                    newest.get("Buyer_MedianReductionPctAmongReduced")
                ),
                "price_pressure_confidence": str(
                    newest.get("Buyer_PricePressureEvidenceConfidence") or ""
                ),
                "price_pressure_evidence_level": str(
                    newest.get("PricePressureEvidenceLevel") or ""
                ),
            }
        else:
            item["buyer_intelligence"] = None

        category_rows = _matching_buyer_category_rows(
            item, filters, preferences, hard_results
        )
        variant_context = []
        if not category_rows.empty:
            # Prefer variants with the most current supply, then newest year.
            category_rows = category_rows.sort_values(
                ["CurrentListings", "Year"],
                ascending=[False, False],
            )
            seen = set()
            for _, row in category_rows.iterrows():
                detail = str(row.get("CategoryDetail") or "").strip()
                key = detail.casefold()
                if not detail or key in seen:
                    continue
                seen.add(key)
                variant_context.append({
                    "category_detail": detail,
                    "year": _json_number(row.get("Year"), integer=True),
                    "current_listings": _json_number(
                        row.get("CurrentListings"), integer=True
                    ),
                    "starting_price": _json_number(
                        row.get("CurrentStartingPrice")
                    ),
                    "median_price": _json_number(
                        row.get("CurrentMedianPrice")
                    ),
                    "median_km": _json_number(
                        row.get("CurrentMedianKM")
                    ),
                    "exit_60_rate": _json_number(
                        row.get("Buyer_ObservedExitWithin60DaysRate")
                    ),
                    "exit_60_eligible": _json_number(
                        row.get("Buyer_Exit60EligibleListings"),
                        integer=True,
                    ),
                    "liquidity_confidence": str(
                        row.get("Buyer_LiquidityEvidenceConfidence") or ""
                    ),
                    "liquidity_evidence_level": str(
                        row.get("LiquidityEvidenceLevel") or ""
                    ),
                    "price_reduction_rate": _json_number(
                        row.get("Buyer_PriceReductionRate")
                    ),
                    "price_pressure_evidence_level": str(
                        row.get("PricePressureEvidenceLevel") or ""
                    ),
                })
                if len(variant_context) >= 4:
                    break

        item["variant_intelligence"] = variant_context
        enriched.append(item)

    return enriched





def _confidence_rank(value):
    text = str(value or "").strip().casefold()
    if text in {"high", "yüksek", "yuksek", "высокая", "высокий"}:
        return 3
    if text in {"medium", "moderate", "orta", "средняя", "средний"}:
        return 2
    if text in {"low", "düşük", "dusuk", "низкая", "низкий"}:
        return 1
    return 0


def _rerank_discover_options_with_buyer_intelligence(model_options, filters):
    """
    Evidence-aware DISCOVER ordering.

    This is deliberately NOT a universal vehicle score. It is a lexicographic
    ordering of already-qualified market options:
      1) preserve buyer-fit/profile qualification as the primary signal;
      2) prefer stronger historical evidence;
      3) among similarly qualified/evidenced models, prefer stronger observed
         market-turnover signals and deeper historical evidence;
      4) retain current-market relevance through newest affordable year,
         budget proximity and active choice.

    Reliability and value retention are intentionally absent because those are
    not established by OtoDeğer's proprietary market data.
    """
    options = [dict(x) for x in (model_options or [])]
    if len(options) < 2:
        return options

    budget = filters.get("budget")
    try:
        ceiling = float(budget) if budget not in [None, ""] else None
    except (TypeError, ValueError):
        ceiling = None

    def finite_float(value, default):
        try:
            value = float(value)
            if pd.isna(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def rank_key(item):
        bi = item.get("buyer_intelligence") or {}

        profile_score = int(item.get("_profile_score") or 0)

        confidence = _confidence_rank(
            bi.get("liquidity_confidence")
        )
        eligible_60 = int(bi.get("exit_60_eligible") or 0)
        exit_60 = finite_float(bi.get("exit_60_rate"), -1.0)
        median_days = finite_float(
            bi.get("median_observed_days_to_exit"), 10**9
        )
        historical_depth = int(
            bi.get("historical_distinct_listings") or 0
        )

        # Do not let tiny cohorts create a misleading liquidity advantage.
        # Below 10 mature 60-day observations, use the signal only after
        # stronger-evidence options have already ranked ahead.
        mature_liquidity = exit_60 if eligible_60 >= 10 else -1.0
        mature_days = median_days if eligible_60 >= 10 else 10**9

        newest = int(item.get("newest_year") or 0)
        active_count = int(item.get("count") or 0)

        anchor_price = item.get("newest_year_starting_price")
        if anchor_price in [None, ""]:
            anchor_price = item.get("starting_price")
        anchor_price = finite_float(anchor_price, 10**12)

        budget_distance = (
            abs(ceiling - anchor_price)
            if ceiling is not None else anchor_price
        )

        return (
            -profile_score,
            -confidence,
            -mature_liquidity,
            mature_days,
            -historical_depth,
            -newest,
            budget_distance,
            -min(active_count, 100),
            str(item.get("brand") or "").casefold(),
            str(item.get("model") or "").casefold(),
        )

    options.sort(key=rank_key)
    return options

def _format_gbp(value, language="EN"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if language == "TR":
        return "£" + f"{number:,.0f}".replace(",", ".")
    return "£" + f"{number:,.0f}"


def _discover_profile_reason(option, preferences, language):
    """Return only profile-catalogue-supported reasons for a DISCOVER recommendation."""
    key = (
        str(option.get("brand") or "").strip().casefold(),
        str(option.get("model") or "").strip().casefold(),
    )
    profile = MODEL_PROFILE_LOOKUP.get(key) if MODEL_PROFILE_READY else None
    if not profile:
        return None

    preferences = _canonicalize_buyer_preferences(preferences)
    reasons = []

    def strong(field):
        return _profile_level(profile.get(field)) >= 2

    for pref in preferences:
        p = str(pref or "").casefold()
        if p == "priority:economy" and strong("Economy"):
            reasons.append({"EN": "economy fit", "TR": "ekonomi önceliğine uygun", "RU": "подходит по экономичности"}[language])
        elif p == "priority:luxury" and strong("Luxury"):
            reasons.append({"EN": "luxury fit", "TR": "lüks önceliğine uygun", "RU": "подходит по уровню премиальности"}[language])
        elif p == "priority:comfort" and strong("Comfort"):
            reasons.append({"EN": "comfort fit", "TR": "konfor önceliğine uygun", "RU": "подходит по комфорту"}[language])
        elif p == "priority:performance" and strong("Performance"):
            reasons.append({"EN": "performance fit", "TR": "performans önceliğine uygun", "RU": "подходит по динамике"}[language])
        elif p == "priority:practicality" and strong("Practicality"):
            reasons.append({"EN": "practicality fit", "TR": "pratiklik önceliğine uygun", "RU": "подходит по практичности"}[language])
        elif p == "use_case:family" and strong("Family"):
            reasons.append({"EN": "family fit", "TR": "aile kullanımına uygun", "RU": "подходит для семьи"}[language])
        elif p == "use_case:commute" and strong("Commute"):
            reasons.append({"EN": "commute fit", "TR": "günlük kullanıma uygun", "RU": "подходит для ежедневных поездок"}[language])
        elif p == "vehicle_type:small_car" and _profile_matches_vehicle_type(profile, "small_car"):
            reasons.append({"EN": "small-car fit", "TR": "küçük araç tercihine uygun", "RU": "подходит как компактный автомобиль"}[language])

    return ", ".join(reasons[:2]) if reasons else None


def _build_discover_response_plan(language, filters, preferences, model_options):
    """
    Structured response planning layer for DISCOVER.

    It never invents vehicle facts. Every surfaced market fact comes from the
    deterministic option/evidence packet; soft-fit explanations come only from
    buyer_model_profiles.csv.
    """
    options = list(model_options or [])[:5]
    if not options:
        return None

    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    planned = []
    for index, item in enumerate(options):
        bi = item.get("buyer_intelligence") or {}
        planned.append({
            "rank": index + 1,
            "name": f"{item.get('brand','')} {item.get('model','')}".strip(),
            "newest_year": item.get("newest_year"),
            "newest_year_price": _format_gbp(item.get("newest_year_starting_price"), language),
            "count": int(item.get("count") or 0),
            "fit_reason": _discover_profile_reason(item, preferences, language),
            "liquidity_confidence": str(bi.get("liquidity_confidence") or "").strip().upper(),
        })

    confidences = [x["liquidity_confidence"] for x in planned]
    thin_evidence = bool(confidences) and all(c in {"LOW", "INSUFFICIENT", ""} for c in confidences)

    return {
        "goal": "DISCOVER",
        "budget_text": budget_text,
        "recommendation": planned[0],
        "alternatives": planned[1:],
        "thin_evidence": thin_evidence,
        "preferences": list(_canonicalize_buyer_preferences(preferences)),
    }


def _render_discover_response_plan(language, plan):
    """Premium, concise renderer for the deterministic DISCOVER response plan."""
    if not plan:
        return None

    top = plan["recommendation"]
    budget_text = plan.get("budget_text")

    if language == "TR":
        intro = (
            f"{budget_text} bütçeyle ilk bakacağım seçenek **{top['name']}**."
            if budget_text else
            f"İlk bakacağım seçenek **{top['name']}**."
        )
        if top.get("fit_reason"):
            intro += f" Profil verimizde {top['fit_reason']}."
        detail = []
        if top.get("newest_year") and top.get("newest_year_price"):
            detail.append(f"{top['newest_year']} modeller {top['newest_year_price']}'dan başlıyor")
        detail.append(f"mevcut filtrelerde {top['count']} ilan var")
        intro += " " + "; ".join(detail) + "."

        alt_lines = []
        for x in plan["alternatives"]:
            reason = f" · {x['fit_reason']}" if x.get("fit_reason") else ""
            if x.get("newest_year") and x.get("newest_year_price"):
                alt_lines.append(f"**{x['name']}** — {x['newest_year']} {x['newest_year_price']}'dan · {x['count']} ilan{reason}")
            else:
                alt_lines.append(f"**{x['name']}** — {x['count']} ilan{reason}")
        closing = "İsterseniz ilk 2–3 seçeneği doğrudan karşılaştırabilir veya gerçek ilanlara geçebiliriz."

    elif language == "RU":
        intro = (
            f"При бюджете {budget_text} я бы сначала посмотрел **{top['name']}**."
            if budget_text else
            f"Я бы сначала посмотрел **{top['name']}**."
        )
        if top.get("fit_reason"):
            intro += f" По профилю модели: {top['fit_reason']}."
        detail = []
        if top.get("newest_year") and top.get("newest_year_price"):
            detail.append(f"{top['newest_year']} год от {top['newest_year_price']}")
        detail.append(f"{top['count']} объявлений по текущим фильтрам")
        intro += " " + "; ".join(detail) + "."

        alt_lines = []
        for x in plan["alternatives"]:
            reason = f" · {x['fit_reason']}" if x.get("fit_reason") else ""
            if x.get("newest_year") and x.get("newest_year_price"):
                alt_lines.append(f"**{x['name']}** — {x['newest_year']} от {x['newest_year_price']} · {x['count']} объявлений{reason}")
            else:
                alt_lines.append(f"**{x['name']}** — {x['count']} объявлений{reason}")
        closing = "Дальше можно напрямую сравнить 2–3 лучших варианта или перейти к конкретным объявлениям."

    else:
        intro = (
            f"With a {budget_text} ceiling, my first look would be **{top['name']}**."
            if budget_text else
            f"My first look would be **{top['name']}**."
        )
        if top.get("fit_reason"):
            intro += f" Its model profile is a {top['fit_reason']}."
        detail = []
        if top.get("newest_year") and top.get("newest_year_price"):
            detail.append(f"{top['newest_year']} examples start at {top['newest_year_price']}")
        detail.append(f"{top['count']} listings match your current filters")
        intro += " " + "; ".join(detail) + "."

        alt_lines = []
        for x in plan["alternatives"]:
            reason = f" · {x['fit_reason']}" if x.get("fit_reason") else ""
            if x.get("newest_year") and x.get("newest_year_price"):
                alt_lines.append(f"**{x['name']}** — {x['newest_year']} from {x['newest_year_price']} · {x['count']} listings{reason}")
            else:
                alt_lines.append(f"**{x['name']}** — {x['count']} listings{reason}")
        closing = "From here, I can compare the strongest 2–3 directly or move into actual listings."

    parts = [intro]
    if alt_lines:
        parts.append("\n".join(alt_lines))

    if plan.get("thin_evidence"):
        if language == "TR":
            parts.append("Bu grupta geçmiş piyasa verisi daha sınırlı; sıralamayı daha düşük güvenle değerlendirmek doğru olur.")
        elif language == "RU":
            parts.append("По этой группе исторических данных меньше, поэтому порядок рекомендаций стоит считать менее уверенным.")
        else:
            parts.append("Market-history evidence is thinner for this group, so I’d treat the ordering as lower-confidence.")

    parts.append(closing)
    return "\n\n".join(parts)


def _fast_discover_answer(language, filters, preferences, model_options):
    plan = _build_discover_response_plan(
        language=language,
        filters=filters,
        preferences=preferences,
        model_options=model_options,
    )
    return _render_discover_response_plan(language, plan)


def _fast_compare_answer(message, language, filters, model_options):
    """Concise consumer comparison using only grounded market/Buyer Intelligence facts."""
    options = list(model_options or [])
    if len(options) < 2:
        return None

    targets = resolve_market_vehicle_mentions(message)
    targets = _attach_explicit_years_to_vehicle_targets(message, targets)
    target_labels = {}
    for t in targets:
        key = (str(t.get("brand") or "").casefold(), str(t.get("model") or "").casefold())
        label = f"{t.get('brand','')} {t.get('model','')}".strip()
        if t.get("year") is not None:
            label = f"{int(t['year'])} {label}"
        if t.get("category"):
            label += f" {t['category']}"
        target_labels[key] = label

    # Only compare explicitly named targets. This is also a defensive barrier
    # against malformed market rows ever surfacing as a third comparison vehicle.
    if target_labels:
        options = [
            o for o in options
            if (str(o.get("brand") or "").casefold(), str(o.get("model") or "").casefold()) in target_labels
        ]
    chosen = options[:4]
    if len(chosen) < 2:
        return None
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    def label(o):
        key = (str(o.get("brand") or "").casefold(), str(o.get("model") or "").casefold())
        return target_labels.get(key) or f"{o.get('brand','')} {o.get('model','')}".strip()

    def km_text(v):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None
        return f"{n:,}" if language != "TR" else f"{n:,}".replace(",", ".")

    def liquidity_status(option):
        bi = option.get("buyer_intelligence") or {}
        days = bi.get("median_observed_days_to_exit")
        confidence = str(bi.get("liquidity_confidence") or "").strip().upper()

        if days is None or confidence in {"LOW", "INSUFFICIENT", ""}:
            return "INSUFFICIENT", None

        try:
            days = float(days)
        except (TypeError, ValueError):
            return "INSUFFICIENT", None

        if days <= 25:
            return "FAST", days
        if days <= 40:
            return "MEDIUM", days
        return "SLOW", days

    def liquidity_label(option):
        status, _ = liquidity_status(option)
        labels = {
            "EN": {
                "FAST": "Fast",
                "MEDIUM": "Medium",
                "SLOW": "Slow",
                "INSUFFICIENT": "Not enough data",
            },
            "TR": {
                "FAST": "Hızlı",
                "MEDIUM": "Orta",
                "SLOW": "Yavaş",
                "INSUFFICIENT": "Yeterli veri yok",
            },
            "RU": {
                "FAST": "Высокая",
                "MEDIUM": "Средняя",
                "SLOW": "Низкая",
                "INSUFFICIENT": "Недостаточно данных",
            },
        }
        return labels.get(language, labels["EN"]).get(status, status)

    # Determine factual leaders.
    newest_winner = max(chosen, key=lambda o: (int(o.get("newest_year") or 0), int(o.get("count") or 0)))
    choice_winner = max(chosen, key=lambda o: int(o.get("count") or 0))

    liquidity_candidates=[]
    for o in chosen:
        bi=o.get("buyer_intelligence") or {}
        days=bi.get("median_observed_days_to_exit")
        rate=bi.get("exit_60_rate")
        if days is not None or rate is not None:
            liquidity_candidates.append((o, days, rate))
    liquidity_winner=None
    if liquidity_candidates:
        liquidity_winner=min(
            liquidity_candidates,
            key=lambda t: (float(t[1]) if t[1] is not None else 10**9, -(float(t[2]) if t[2] is not None else -1)),
        )[0]

    same_market_winner = label(choice_winner) == label(newest_winner)
    liquidity_winner_label = label(liquidity_winner) if liquidity_winner is not None else None

    # Natural follow-up: "Which one would you choose based on the North Cyprus market?"
    # Keep this deterministic and evidence-bounded. A vehicle is recommended only
    # when it leads a majority of the market signals we actually support here:
    # current choice, newest affordable year, and observed historical turnover.
    low_message = str(message or "").casefold()
    asks_market_choice = bool(re.search(
        r"\b(?:which (?:one )?(?:would you|do you) (?:choose|recommend|pick|prefer)|"
        r"which (?:one )?is (?:better|stronger) based on (?:the )?(?:north cyprus )?market|"
        r"hangisini (?:seçer|secer|önerir|onerir)sin(?:iz)?|"
        r"piyasaya göre hangisi|piyasaya gore hangisi|"
        r"какую (?:выбрать|порекомендуете)|какой (?:выбрать|порекомендуете))\b",
        low_message,
    ))

    if asks_market_choice:
        votes = {}
        def add_vote(option):
            if option is None:
                return
            key = (str(option.get("brand") or "").casefold(), str(option.get("model") or "").casefold())
            votes[key] = votes.get(key, 0) + 1

        add_vote(choice_winner)
        add_vote(newest_winner)
        add_vote(liquidity_winner)

        selected = max(
            chosen,
            key=lambda o: (
                votes.get((str(o.get("brand") or "").casefold(), str(o.get("model") or "").casefold()), 0),
                int(o.get("count") or 0),
                int(o.get("newest_year") or 0),
            ),
        )
        selected_key = (str(selected.get("brand") or "").casefold(), str(selected.get("model") or "").casefold())
        selected_votes = votes.get(selected_key, 0)
        selected_name = label(selected)

        reasons = []
        if selected is choice_winner:
            reasons.append("more current choice")
        if selected is newest_winner:
            reasons.append("access to a newer affordable model year")
        if liquidity_winner is not None and selected is liquidity_winner:
            reasons.append("the stronger observed historical turnover signal")

        if language == "TR":
            if selected_votes >= 2:
                reason_text = ", ".join(reasons)
                return (
                    f"Kuzey Kıbrıs piyasa verilerine göre {selected_name} tercih ederdim. "
                    f"Bu karşılaştırmada {reason_text} açısından daha güçlü görünüyor. "
                    "Bu, aracın doğası gereği daha iyi veya daha güvenilir olduğu anlamına gelmez; "
                    "tercih yalnızca elimizdeki yerel piyasa verilerine dayanıyor."
                )
            return (
                "Yalnızca mevcut Kuzey Kıbrıs piyasa verilerine dayanarak ikisinden birini net biçimde "
                "üstün ilan etmezdim; güçlü oldukları piyasa sinyalleri farklı."
            )
        if language == "RU":
            if selected_votes >= 2:
                reason_text = ", ".join(reasons)
                return (
                    f"По данным рынка Северного Кипра я бы выбрал {selected_name}. "
                    f"В этом сравнении он сильнее по следующим сигналам: {reason_text}. "
                    "Это не означает, что автомобиль сам по себе лучше или надёжнее; "
                    "выбор основан только на доступных локальных рыночных данных."
                )
            return (
                "Только по текущим данным рынка Северного Кипра я бы не называл один из этих вариантов "
                "однозначно лучшим: они сильнее по разным рыночным сигналам."
            )
        if selected_votes >= 2:
            reason_text = ", ".join(reasons)
            return (
                f"Based on the North Cyprus market data, I'd choose {selected_name}. "
                f"In this comparison it has {reason_text}. "
                "That doesn't mean it is inherently the better or more reliable vehicle; "
                "the choice is based only on the local market evidence we have."
            )
        return (
            "Based only on the current North Cyprus market data, I wouldn't call either one the clear winner; "
            "they lead on different market signals."
        )

    if language == "TR":
        prefix = f"{budget_text} bütçenizde " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} hem daha fazla seçenek hem de daha yeni araçlara erişim sunuyor"
        else:
            intro = prefix + f"{label(choice_winner)} daha fazla seçenek sunarken {label(newest_winner)} daha yeni araçlara erişim sağlıyor"
        if liquidity_winner_label and liquidity_winner_label != label(choice_winner):
            intro += f"; {liquidity_winner_label} ise tarihsel yeniden satış kolaylığı sinyalinde daha güçlü."
        else:
            intro += "."
    elif language == "RU":
        prefix = f"При бюджете {budget_text} " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} предлагает и больший выбор, и доступ к более новым автомобилям"
        else:
            intro = prefix + f"у {label(choice_winner)} больше выбора, а {label(newest_winner)} даёт доступ к более новым автомобилям"
        if liquidity_winner_label and liquidity_winner_label != label(choice_winner):
            intro += f"; при этом исторический сигнал по лёгкости перепродажи сильнее у {liquidity_winner_label}."
        else:
            intro += "."
    else:
        prefix = f"With your {budget_text} ceiling, " if budget_text else ""
        if same_market_winner:
            intro = prefix + f"{label(choice_winner)} offers considerably more choice and access to newer cars"
        else:
            intro = prefix + f"{label(choice_winner)} offers more choice, while {label(newest_winner)} gives you access to newer cars"
        if liquidity_winner_label and liquidity_winner_label != label(choice_winner):
            intro += f", while {liquidity_winner_label} has the stronger historical resale-ease signal."
        else:
            intro += "."

    sections=[]
    for o in chosen:
        name=label(o)
        count=int(o.get("count") or 0)
        newest_year=o.get("newest_year")
        newest_price=_format_gbp(o.get("newest_year_starting_price"), language)
        overall_price=_format_gbp(o.get("starting_price"), language)
        overall_price_year=o.get("starting_price_year")
        bi=o.get("buyer_intelligence") or {}
        days=bi.get("median_observed_days_to_exit")
        deal=o.get("potential_value_listing") or {}

        deal_text = ""
        if deal:
            deal_price = _format_gbp(deal.get("price"), language)
            deal_median = _format_gbp(deal.get("median_asking_price"), language)
            deal_year = deal.get("year")
            deal_link = str(deal.get("link") or "").strip()
            try:
                pct_number = float(deal.get("below_median_pct")) * 100
                pct_text = f"{pct_number:.0f}%"
            except (TypeError, ValueError):
                pct_text = None

            if language == "TR":
                deal_text = (
                    f" Potansiyel fırsat: {deal_year} model bir ilan {deal_price}; "
                    f"karşılaştırılabilir güncel ilanların medyanı {deal_median}"
                    + (f" — yaklaşık {pct_text} daha düşük" if pct_text else "")
                    + "."
                )
                deal_text += " Bu yalnızca ilan fiyatına dayalı bir sinyaldir; kilometre, donanım ve kondisyon farkı açıklayabilir."
            elif language == "RU":
                deal_text = (
                    f" Потенциально интересное предложение: {deal_year} за {deal_price}; "
                    f"медианная цена сопоставимых текущих объявлений — {deal_median}"
                    + (f" — примерно на {pct_text} ниже" if pct_text else "")
                    + "."
                )
                deal_text += " Это сигнал только по цене объявления; пробег, комплектация и состояние могут объяснять разницу."
            else:
                deal_text = (
                    f" Potential value listing: a {deal_year} is advertised at {deal_price} versus "
                    f"a {deal_median} median asking price for comparable current listings"
                    + (f" — about {pct_text} lower" if pct_text else "")
                    + "."
                )
                deal_text += " That's a price-only signal, so mileage, trim and condition still need checking."

        if language == "TR":
            facts=f"{count} aktif seçenek"
            if newest_year and newest_price:
                facts += f" · {newest_year} model {newest_price}'dan başlıyor"
            elif overall_price:
                facts += f" · başlangıç fiyatı {overall_price}"
                if overall_price_year:
                    facts += f" ({overall_price_year})"
            resale=f" Likidite: {liquidity_label(o)}."
            sections.append(f"{name}\n{facts}.{deal_text}{resale}")
        elif language == "RU":
            facts=f"{count} активных вариантов"
            if newest_year and newest_price:
                facts += f" · {newest_year} год от {newest_price}"
            elif overall_price:
                facts += f" · цены от {overall_price}"
                if overall_price_year:
                    facts += f" ({overall_price_year})"
            resale=f" Ликвидность: {liquidity_label(o)}."
            sections.append(f"{name}\n{facts}.{deal_text}{resale}")
        else:
            facts=f"{count} currently available"
            if newest_year and newest_price:
                facts += f" · {newest_year} starts from {newest_price}"
            elif overall_price:
                facts += f" · prices start from {overall_price}"
                if overall_price_year:
                    facts += f" for a {overall_price_year}"
            resale=f" Liquidity: {liquidity_label(o)}."
            sections.append(f"{name}\n{facts}.{deal_text}{resale}")

    comparable_liquidity = []
    for o in chosen:
        status, days_value = liquidity_status(o)
        if days_value is not None:
            comparable_liquidity.append((o, days_value))

    relative_turnover = None
    if len(comparable_liquidity) >= 2:
        fastest_o, fastest_days = min(comparable_liquidity, key=lambda x: x[1])
        slowest_o, slowest_days = max(comparable_liquidity, key=lambda x: x[1])
        if slowest_days > 0 and fastest_days < slowest_days:
            relative_turnover = (
                fastest_o,
                max(1, round((slowest_days - fastest_days) / slowest_days * 100)),
            )

    if language == "TR":
        close_parts=[f"Seçenek sayısı ve daha yeni araçlara erişim açısından {label(choice_winner)} daha güçlü."]
        if relative_turnover:
            faster_o, faster_pct = relative_turnover
            close_parts.append(
                f"Likidite tarafında {label(faster_o)} gözlenen piyasa verilerinde yaklaşık %{faster_pct} daha hızlı hareket ediyor."
            )
        elif liquidity_winner is not None:
            close_parts.append(f"Likidite sinyali {label(liquidity_winner)} için daha güçlü.")
        close_parts.append(
            "Bu likidite sinyali ilanların gözlemden çıkış hızına dayanır; doğrulanmış satış anlamına gelmez. "
            "Maksimum kilometrenizi söylerseniz karşılaştırmayı doğrudan o sınırdaki araçlara indirebilirim."
        )
        closing=" ".join(close_parts)
    elif language == "RU":
        close_parts=[f"По выбору и доступу к более новым машинам сильнее {label(choice_winner)}."]
        if relative_turnover:
            faster_o, faster_pct = relative_turnover
            close_parts.append(
                f"По ликвидности {label(faster_o)} в наблюдаемых рыночных данных движется примерно на {faster_pct}% быстрее."
            )
        elif liquidity_winner is not None:
            close_parts.append(f"Сигнал ликвидности сильнее у {label(liquidity_winner)}.")
        close_parts.append(
            "Сигнал ликвидности основан на скорости исчезновения объявлений из наблюдаемого рынка, а не на подтверждённых продажах. "
            "Укажите максимальный пробег — и я сравню только подходящие варианты."
        )
        closing=" ".join(close_parts)
    else:
        close_parts=[f"For choice and access to newer cars, {label(choice_winner)} is stronger."]
        if relative_turnover:
            faster_o, faster_pct = relative_turnover
            close_parts.append(
                f"For liquidity, {label(faster_o)} moves about {faster_pct}% quicker in the observed market data."
            )
        elif liquidity_winner is not None:
            close_parts.append(f"The liquidity signal is stronger for {label(liquidity_winner)}.")
        close_parts.append(
            "Liquidity is based on how quickly listings leave the observed market, not confirmed sales. "
            "Give me your maximum mileage and I can compare only the cars that actually meet it."
        )
        closing=" ".join(close_parts)

    return intro + "\n\n" + "\n\n".join(sections) + "\n\n" + closing


def _localize_listing_value(value, field, language):
    """Localize common structured market values without altering seller names."""
    text = str(value or "").strip()
    if not text or language == "TR":
        return text
    key = _normalize_vehicle_phrase(text)
    if language == "EN":
        maps = {
            "transmission": {"otomatik": "Automatic", "manuel": "Manual"},
            "company": {"bireysel": "Individual seller"},
            "color": {
                "siyah": "Black", "beyaz": "White", "gumus": "Silver",
                "gri": "Grey", "fume": "Dark grey", "mavi": "Blue",
                "kirmizi": "Red", "yesil": "Green", "sari": "Yellow",
                "bej": "Beige", "kahverengi": "Brown", "turuncu": "Orange",
                "lacivert": "Navy", "bordo": "Burgundy",
                "mavi okyanus": "Ocean Blue", "mavi parlement": "Parliament Blue",
                "inci beyaz": "Pearl White", "metalik gri": "Metallic Grey",
                "koyu gri": "Dark Grey", "acik gri": "Light Grey"
            },
        }
        return maps.get(field, {}).get(key, text)
    if language == "RU":
        maps = {
            "transmission": {"otomatik": "автомат", "manuel": "механика"},
            "company": {"bireysel": "частный продавец"},
            "color": {
                "siyah": "чёрный", "beyaz": "белый", "gumus": "серебристый",
                "gri": "серый", "fume": "тёмно-серый", "mavi": "синий",
                "kirmizi": "красный", "yesil": "зелёный", "sari": "жёлтый"
            },
        }
        return maps.get(field, {}).get(key, text)
    return text


def _listing_mileage_anomaly(item):
    """Flag implausibly low advertised mileage without asserting the listing is wrong."""
    try:
        year = int(item.get("year"))
        km = float(item.get("km"))
    except (TypeError, ValueError):
        return False
    current_year = datetime.now(timezone.utc).year
    age = max(0, current_year - year)
    if km < 0:
        return True
    if age >= 2 and km < 500:
        return True
    if age >= 5 and km < 1500:
        return True
    return False


def _select_shop_representatives(results, max_candidates=3, sort_mode=None):
    """Pick a small useful SHOP set, or honor an explicit listing-level sort.

    All rows have already passed the deterministic active filters. Selection is
    presentation-only and never relaxes budget/model/year/KM/location/seller rules.
    """
    clean = list(results or [])
    if not clean:
        return []

    if sort_mode == "lowest_km":
        ranked = [x for x in clean if x.get("km") is not None]
        ranked.sort(key=lambda x: (int(x.get("km")), -int(x.get("year") or 0), float(x.get("price") or 10**12)))
        return ranked[:max_candidates]
    if sort_mode == "cheapest":
        ranked = sorted(clean, key=lambda x: (float(x.get("price") or 10**12), -int(x.get("year") or 0), int(x.get("km")) if x.get("km") is not None else 10**12))
        return ranked[:max_candidates]
    if sort_mode == "newest":
        ranked = sorted(clean, key=lambda x: (-int(x.get("year") or 0), int(x.get("km")) if x.get("km") is not None else 10**12, float(x.get("price") or 10**12)))
        return ranked[:max_candidates]

    chosen, seen = [], set()

    def identity(item):
        return item.get("link") or (
            item.get("brand"), item.get("model"), item.get("category"),
            item.get("year"), item.get("price"), item.get("km"), item.get("company")
        )

    def take(item):
        if item is None:
            return
        key = identity(item)
        if key not in seen:
            chosen.append(item)
            seen.add(key)

    # Default "best matching" presentation:
    # - every row already satisfies the buyer's hard filters, including budget;
    # - prefer recent model years;
    # - within the recent band, prefer plausible/lower advertised mileage;
    # - use asking price only as a final factual tie-break.
    #
    # This deliberately avoids using "cheapest remaining" as the third default
    # representative, which could surface an ancient bargain beside two modern
    # cars even though the buyer asked for the best matches rather than cheapest.
    valid_years = [
        int(x.get("year"))
        for x in clean
        if x.get("year") not in [None, ""]
    ]
    newest_year = max(valid_years) if valid_years else None
    recent_floor = (newest_year - 5) if newest_year is not None else None

    def default_rank_key(item):
        year = int(item.get("year") or 0)
        km_missing = item.get("km") is None
        km_value = int(item.get("km")) if item.get("km") is not None else 10**12
        price = float(item.get("price") or 10**12)
        return (
            -year,
            bool(_listing_mileage_anomaly(item)),
            km_missing,
            km_value,
            price,
        )

    recent_pool = [
        x for x in clean
        if recent_floor is None
        or (
            x.get("year") not in [None, ""]
            and int(x.get("year")) >= recent_floor
        )
    ]
    recent_ranked = sorted(recent_pool, key=default_rank_key)

    for item in recent_ranked:
        if len(chosen) >= max_candidates:
            break
        take(item)

    # Defensive fill only when the market genuinely has fewer than the requested
    # number of recent examples. Older rows remain eligible rather than being
    # silently excluded from SHOP.
    if len(chosen) < max_candidates:
        all_ranked = sorted(clean, key=default_rank_key)
        for item in all_ranked:
            if len(chosen) >= max_candidates:
                break
            take(item)

    return chosen[:max_candidates]


def _fast_shop_answer(language, filters, search_result, listing_candidates, preferences=None):
    """Render listing-level results locally and always use progressive disclosure."""
    candidates = list(listing_candidates or [])[:3]
    if not candidates:
        return None
    total = int(search_result.get("count", 0) or 0)
    budget = filters.get("budget")
    budget_text = _format_gbp(budget, language) if budget not in [None, ""] else None

    sort_mode = _listing_sort_mode(preferences)
    first = candidates[0]
    vehicle_name = f"{first.get('brand','')} {first.get('model','')}".strip()
    vehicle_name = re.sub(r"\\s+", " ", vehicle_name)

    if language == "TR":
        qualifier = {"lowest_km": "en düşük kilometreli", "cheapest": "en düşük fiyatlı", "newest": "en yeni"}.get(sort_mode, "güncel")
        intro = f"İşte" + (f" {budget_text} bütçeniz içinde" if budget_text else "") + f" {qualifier} {vehicle_name} ilanları:"
    elif language == "RU":
        qualifier = {"lowest_km": "с минимальным пробегом", "cheapest": "с самой низкой ценой", "newest": "самые новые"}.get(sort_mode, "актуальные")
        intro = f"Вот {qualifier} объявления {vehicle_name}" + (f" в рамках бюджета {budget_text}:" if budget_text else ":")
    else:
        qualifier = {"lowest_km": "lowest-mileage", "cheapest": "lowest-priced", "newest": "newest"}.get(sort_mode, "current")
        intro = f"Here are the {qualifier} {vehicle_name} listings" + (f" within your {budget_text} budget:" if budget_text else ":")

    lines=[]
    for x in candidates:
        name=f"{x.get('brand','')} {x.get('model','')}".strip()
        name=re.sub(r"\\s+", " ", name)
        year=x.get('year') or '—'
        price=_format_gbp(x.get('price'), language) or '—'
        details=[]
        if x.get('km') is not None:
            km=int(x['km'])
            km_txt=f"{km:,}" if language != 'TR' else f"{km:,}".replace(',', '.')
            if _listing_mileage_anomaly(x):
                if language == "TR":
                    details.append(f"ilan km: {km_txt} (doğrulayın)")
                elif language == "RU":
                    details.append(f"заявленный пробег: {km_txt} км (проверьте)")
                else:
                    details.append(f"advertised {km_txt} km (verify)")
            else:
                details.append(f"{km_txt} km")
        # Colour stays available as a filter, but is omitted from default replies.
        for key in ('transmission','company','location'):
            val = _localize_listing_value(x.get(key), key, language)
            if val:
                details.append(val)
        lines.append(f"{name}, {year} — {price}" + (" · " + " · ".join(details) if details else ""))

    if total > 10:
        if language == "TR":
            closing=f"Toplam {total} eşleşme var; yüzlerce ilan sıralamak yerine aramayı daraltmak daha faydalı olur. Maksimum kilometre, minimum model yılı, konum veya galeri/bireysel satıcı tercihinizi yazabilirsiniz."
        elif language == "RU":
            closing=f"Всего найдено {total} вариантов. Вместо длинного списка лучше сузить поиск: укажите максимальный пробег, минимальный год, район или дилер/частный продавец."
        else:
            closing=f"There are {total} matches, so a long list would not be very useful. Give me a maximum mileage, minimum year, location, or dealer/private-seller preference and I'll narrow it down."
    elif total > len(candidates):
        closing = {
            'TR': f"Toplam {total} eşleşme var. Daha fazlasını gösterebilir veya kilometre/yıl gibi bir kriterle daraltabilirim.",
            'RU': f"Всего найдено {total} вариантов. Можно показать ещё или сузить поиск по пробегу/году.",
        }.get(language, f"There are {total} matches. I can show more or narrow them by mileage, year or another preference.")
    else:
        closing = {
            'TR': f"Mevcut kriterlerinize uyan {total} ilan bunlar.",
            'RU': f"Это все {total} объявлений, соответствующих текущим критериям.",
        }.get(language, f"These are the {total} current listings matching your criteria.")

    return intro + "\n\n" + "\n".join(lines) + "\n\n" + closing




def _extract_explicit_market_brands(message):
    """
    Deterministically resolve brand names that the user explicitly typed.

    The semantic controller is allowed to understand intent, but it must not be
    allowed to silently drop one side of a concrete request such as
    "BMW or Mercedes". Canonical spellings come from the live market data.
    """
    raw = str(message or "").strip()
    if not raw or market_df is None or market_df.empty or "Brand" not in market_df.columns:
        return []

    def norm(value):
        value = str(value or "").casefold()
        value = value.replace("&", " and ")
        value = re.sub(r"[^a-z0-9çğıöşü]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    message_norm = f" {norm(raw)} "
    brands = [
        str(v).strip()
        for v in market_df["Brand"].dropna().astype(str).unique().tolist()
        if str(v).strip()
    ]

    # Explicit human aliases that differ from the canonical market spelling.
    aliases = {
        "mercedes": "Mercedes-Benz",
        "mercedes benz": "Mercedes-Benz",
        "mercedes-benz": "Mercedes-Benz",
    }

    found = []
    seen = set()

    # Match canonical live-market brand names first.
    for brand in brands:
        bnorm = norm(brand)
        if bnorm and f" {bnorm} " in message_norm:
            key = brand.casefold()
            if key not in seen:
                found.append(brand)
                seen.add(key)

    # Then resolve common aliases only if that canonical brand exists live.
    live_by_fold = {b.casefold(): b for b in brands}
    for alias, canonical in aliases.items():
        if f" {norm(alias)} " in message_norm:
            live = live_by_fold.get(canonical.casefold())
            if live and live.casefold() not in seen:
                found.append(live)
                seen.add(live.casefold())

    return found


def _apply_authoritative_explicit_brands(message, interpretation):
    """
    Concrete brand names typed by the user are authoritative constraints.

    This runs after the semantic controller. It prevents an LLM parse such as
    ["BMW"] from losing "Mercedes" in "BMW or Mercedes".
    """
    explicit_brands = _extract_explicit_market_brands(message)
    if not explicit_brands:
        return interpretation

    updated = dict(interpretation or {})
    filters = dict(updated.get("filters") or {})
    filters["brands"] = explicit_brands
    updated["filters"] = sanitize_ai_filters(filters)
    return updated


def _requested_brand_presence_guard(message, model_options):
    """
    Cross-check requested brands against the actual option packet.

    This is generic for every live market brand, not hard-coded to BMW/Mercedes.
    """
    requested = _extract_explicit_market_brands(message)
    if not requested:
        return []

    option_brands = {
        str(opt.get("brand") or "").strip().casefold()
        for opt in (model_options or [])
        if str(opt.get("brand") or "").strip()
    }
    return [brand for brand in requested if brand.casefold() in option_brands]


def generate_grounded_market_answer(message, language, filters, preferences, search_result, conversation_history=None, decision_mode="DISCOVER"):
    """Progressive-disclosure buying advice grounded in deterministic market data."""
    hard_count = int(search_result.get("count", 0) or 0)
    hard_results = search_result.get("results", []) or []

    if _asks_reliability_question(message):
        return _reliability_scope_answer(language, filters), hard_results, hard_count, []

    if hard_count == 0:
        fallback = {
            "TR": "Bu kriterlere uyan aktif ilan bulamadım. İsterseniz bütçe, yıl, kilometre veya diğer kriterlerden birini esnetebiliriz.",
            "EN": "I couldn't find an active listing matching those criteria. We can loosen the budget, year, mileage or another filter.",
            "RU": "Я не нашёл активных объявлений по этим критериям. Можно немного ослабить бюджет, год, пробег или другой фильтр.",
        }
        return fallback.get(language, fallback["TR"]), [], 0, []

    decision_mode = str(decision_mode or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"

    # Model qualification is useful during DISCOVER because soft ideas such as
    # economical/small/luxury need model-level automotive knowledge. Once the buyer
    # explicitly moves to COMPARE or SHOP, the named market targets are already known.
    # Re-running that extra LLM step adds latency and can only narrow relevant evidence.
    if decision_mode == "DISCOVER":
        qualified_results, model_reasons, qualified_summaries = shortlist_models_for_preferences(
            message=message,
            language=language,
            filters=filters,
            preferences=preferences,
            results=hard_results,
        )
    else:
        qualified_results = list(hard_results)
        model_reasons = []
        qualified_summaries = _group_market_models(hard_results)

    has_soft_pref = any(
        str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
        for p in (preferences or [])
    )

    if decision_mode == "DISCOVER" and has_soft_pref and not qualified_results:
        # Soft buyer traits are advisory ranking signals. Never report an empty market
        # when the user's hard constraints actually have inventory. Fall back to that
        # inventory and explain/surface the closest options deterministically.
        qualified_results = list(hard_results)
        model_reasons = []
        qualified_summaries = _group_market_models(hard_results)

    advisory_results = qualified_results if qualified_results else hard_results
    advisory_count = len(advisory_results)
    model_summaries = qualified_summaries if qualified_summaries else _group_market_models(advisory_results)
    model_options = _select_model_options(model_summaries, model_reasons, filters, max_options=20)
    model_options = enrich_model_options_with_buyer_intelligence(
        model_options=model_options,
        filters=filters,
        preferences=preferences,
        hard_results=advisory_results,
    )

    if decision_mode == "DISCOVER":
        model_options = _rerank_discover_options_with_buyer_intelligence(
            model_options=model_options,
            filters=filters,
        )

    # Make buyer relevance explicit for the response model. `count`, `newest_year`
    # and the newest-year price above are calculated from the ACTIVE deterministic
    # search result, so when a budget exists they describe what that budget can buy.
    budget_ceiling = filters.get("budget")
    for option in model_options:
        option["active_filter_context"] = {
            "budget_ceiling": budget_ceiling,
            "matching_listing_count": option.get("count"),
            "newest_matching_year": option.get("newest_year"),
            "newest_matching_year_starting_price": option.get("newest_year_starting_price"),
            "lowest_matching_asking_price": option.get("starting_price"),
        }
    if decision_mode == "SHOP":
        listing_candidates = _select_shop_representatives(
            advisory_results, max_candidates=3, sort_mode=_listing_sort_mode(preferences)
        )
    else:
        listing_candidates = select_assistant_candidates(
            advisory_results, filters, max_candidates=3
        )

    # V9: DISCOVER and COMPARE deliberately continue to the grounded conversational
    # renderer below. Deterministic Python owns the facts and state; the language
    # model owns how those facts are communicated in context.
    #
    # SHOP remains deterministic because concrete listing presentation benefits
    # from a stable structure and structured LISTING actions.
    if decision_mode == "SHOP":
        fast_answer = _fast_shop_answer(language, filters, search_result, listing_candidates, preferences)
        if fast_answer:
            return fast_answer, advisory_results, advisory_count, model_options

    instructions = """
You are OtoDeğer AI, a goal-driven vehicle-market copilot for North Cyprus.

The user is not here to receive a market report. They are trying to accomplish something: choose a
car, find a car to buy, compare alternatives, value a car, or make a commercial vehicle decision.
Your job is to move them toward that outcome with the least amount of information needed to make
the next useful decision.

GOAL FIRST:
- Infer the user's current goal from latest_message + recent_conversation + application state.
- Preserve that goal and all still-active constraints across turns. A short refinement such as
  "SUV", "only BMW or Mercedes", "2018+", "automatic", or "yes" continues the existing journey.
- Treat each reply as the NEXT STEP in one ongoing task, not as a fresh market report.
- Before writing, silently decide:
  1) What is the user trying to achieve?
  2) What do we already know?
  3) What decision or action is immediately in front of them?
  4) What is the smallest amount of evidence needed to help with that decision?
- Do not display this internal reasoning or these labels.

DECISION COMPRESSION:
- Lead with the conclusion. Do not lead with a database overview.
- If there is a clear best route, recommend ONE route and explain it with 1-2 decisive facts.
- If there is a genuine tradeoff, show at most 2-3 options and explain the difference in plain language.
- When the user explicitly gives a small set of acceptable brands, prefer one strong candidate per requested
  brand before adding secondary candidates from the same brand. This prevents a higher-volume brand from
  crowding a valid alternative brand out of the answer.
- Do not list alternatives merely because they exist in model_options.
- Do not mention counts, mileage ranges, liquidity, price pressure, confidence, oldest/newest years,
  or other statistics unless that fact materially changes the user's decision.
- NEVER say that a requested brand/model has no matching options unless the authoritative supplied
  model_options/listing evidence actually contains zero matching candidates for that requested brand/model.
- If the evidence packet contains a requested brand/model, acknowledge it even when another option ranks higher.
- When the user explicitly narrows to multiple brands (for example BMW or Mercedes), inspect EACH requested
  brand represented in model_options before concluding that one brand has no qualifying vehicles.
- Keep useful evidence in reserve. OtoDeğer should know more than it says.
- If the user asks for detail, more options, evidence, or "why?", then expand.
- Never repeat information the user already knows unless it is necessary to explain a consequence.

CONVERSATION:
- Respond to the PURPOSE of the latest turn in the context of recent_conversation.
- If the user narrows the search, explain what that means for their current goal; do not restart the shortlist.
- If the user changes direction, adapt the existing journey rather than treating it as a new session.
- If a direct answer is enough, give a direct answer.
- Ask exactly ONE clarification only when the missing answer would materially change what you recommend.
- A clarification should be easy to answer and should explain the useful choice when needed.
- Do not ask for optional information just because more filters are available.
- Do not create permission loops.

NEXT BEST ACTION:
- When there is an obvious useful action that OtoDeğer can actually perform, end with ONE short,
  concrete next step that advances the user's goal.
- Examples: "Want me to find the best X1s currently for sale?" or "Want me to compare those two?"
- Do not offer multiple next actions at once.
- Do not append a next step when the assistant has just asked a clarification question.
- Do not offer capabilities that are not supported by the current application paths/evidence.
- Never use a generic canned CTA.

EVIDENCE BOUNDARY:
- active_hard_filters, soft_preferences, model_options and listing_candidates are the authoritative
  OtoDeğer evidence packet.
- requested_brands contains concrete brand names deterministically resolved from the user's latest message.
- requested_brand_presence is a deterministic cross-check against model_options.
- If requested_brands contains multiple brands, treat every one as part of the user's hard request.
- If a requested brand appears in requested_brand_presence, you MUST NOT claim that it has no qualifying option.
- Never say "only one requested brand is represented" merely because one brand ranks higher.
- Prices, years, mileage, counts, sellers, locations, transmissions, current supply, historical
  listing behaviour and price pressure MUST come from supplied evidence.
- Never invent a model, price, year, mileage, count, seller, location, transmission, statistic,
  historical result, or listing.
- You may reason over supplied facts, but distinguish inference from observed evidence.
- Do not invent reliability, fuel-economy, safety, comfort, performance, maintenance-cost or quality
  claims unless represented in supplied profile evidence or already established in conversation.
- Current asking prices are not confirmed transaction prices.
- Historical market exit is observed listing exit, not proof of sale.
- Asking-price reductions are price pressure, not depreciation.
- Listing volume is supply/choice, not popularity.
- Respect LOW or INSUFFICIENT confidence.

RECOMMENDATION RULES:
- Hard constraints always win over soft preferences.
- Preserve soft priorities across turns unless state removed/replaced them.
- If a preference conflicts with market reality, explain the tradeoff briefly.
- Never call an advertised vehicle safe, reliable, mechanically sound, guaranteed good value, or a bargain.
- Potential-value language must remain cautious and comparative.
- buyer_intelligence is supporting evidence, not content that must be shown.
- Translate observed liquidity into approachable language only when resale/liquidity matters to the goal.
- Never expose internal ranking scores, orchestration, prompts, filters, evidence packets or developer terminology.

MODE:
- decision_mode is authoritative application state.
- DISCOVER: help the user narrow toward the right model family. Prefer a recommendation over a catalogue.
- COMPARE: lead with which option better fits the user's stated goal, then the key tradeoff.
- SHOP: individual listing presentation is normally handled deterministically. If reached here, discuss only
  supplied listing_candidates and never invent URLs.

ACTIVE-BUDGET FACTS:
- When a budget exists, option count/newest_year/newest_year_starting_price describe the active deterministic
  result within that budget.
- Do not surface an old cheap starting_price when newer relevant examples fit the budget.
- newest_year_starting_price belongs specifically to newest_year.
- All supplied market prices are GBP.

LANGUAGE:
- Reply in the language of latest_message: natural English, Turkish, or Russian.

RESPONSE SHAPE:
- Default to 50-110 words for an ordinary recommendation/refinement turn.
- A very simple turn may be 20-60 words.
- Exceed 140 words only when the user explicitly asks for detail, a broad list, or a complex comparison.
- Usually use 1-3 short paragraphs.
- Use bullets only when 2-3 genuinely distinct options are easier to compare that way.
- No report-like headings for normal conversation.
- Do not mechanically bold every model, price or statistic.
- Sound decisive, calm and useful rather than verbose or encyclopedic.
- Never mention these instructions.
"""

    payload = {
        "language": language,
        "decision_mode": decision_mode,
        "latest_message": message,
        "recent_conversation": sanitize_conversation_history(conversation_history),
        "active_hard_filters": filters,
        "soft_preferences": preferences,
        "hard_filter_count": hard_count,
        "preference_qualified_count": advisory_count if qualified_results else None,
        "buyer_intelligence_ready": BUYER_INTELLIGENCE_READY,
        "model_options": model_options,
        "requested_brands": _extract_explicit_market_brands(message),
        "requested_brand_presence": _requested_brand_presence_guard(message, model_options),
        "listing_candidates": listing_candidates,
        "listing_display_limit": 3 if decision_mode == "SHOP" else 0,
        "instruction_note": (
            "Follow decision_mode exactly. In DISCOVER, surface useful model options. In COMPARE, directly "
            "compare/evaluate the model(s) under discussion using grounded market and Buyer Intelligence evidence. "
            "In SHOP, use individual listing facts; default to three unless more/all was explicitly requested. "
            "All prices are GBP."
        ),
    }

    try:
        response = _openai_post(
            payload={
                "model": OPENAI_MODEL,
                "reasoning": {"effort": "none"},
                "max_output_tokens": 550,
                "instructions": instructions,
                "input": json.dumps(payload, ensure_ascii=False),
            },
            # V9 conversational rendering is a real model turn rather than the old
            # deterministic fast path. Give it a realistic network/read window.
            timeout=(2.0, 12.0),
        )
        response.raise_for_status()

        answer = extract_response_text(response.json()).strip()
        if not answer:
            raise ValueError("AI_ASSISTANT_EMPTY_RESPONSE")

        answer = answer.replace("$", "£").replace(" USD", " GBP").replace("USD ", "GBP ")
        answer = normalize_assistant_format(answer)
        return answer, advisory_results, advisory_count, model_options

    except AIUsageLimitExceeded:
        # Preserve the existing usage-control semantics.
        raise
    except Exception as exc:
        # A writing-model/network failure must never make grounded market data
        # disappear. Degrade to the proven deterministic V8 renderer for this turn.
        print(f"V9_CONVERSATIONAL_RENDERER_FALLBACK: {type(exc).__name__}: {exc}", flush=True)

        if decision_mode == "DISCOVER":
            fallback_answer = _fast_discover_answer(
                language, filters, preferences, model_options
            )
            if fallback_answer:
                return fallback_answer, advisory_results, advisory_count, model_options

        if decision_mode == "COMPARE":
            fallback_answer = _fast_compare_answer(
                message, language, filters, model_options
            )
            if fallback_answer:
                return fallback_answer, advisory_results, advisory_count, model_options

        # If no safe deterministic answer exists, allow the endpoint's existing
        # error handling to report the failure rather than fabricating an answer.
        raise




def _recover_latest_assistant_compare_offer(message, conversation_history):
    """
    Recover the exact pair from the immediately preceding assistant offer.

    This handles natural follow-ups such as:
        Assistant: "...I'd focus on BMW X1 and Mercedes-Benz GLA.
                    Want me to compare those two directly?"
        User:      "Yes, compare them"

    It is intentionally narrow:
    - only runs for explicit compare/acceptance language;
    - only inspects the most recent assistant turn before the current user turn;
    - prefers the sentence(s) immediately preceding the assistant's compare offer;
    - requires at least two live-market vehicle targets.

    This prevents stale comparisons from being resurrected from older conversation history.
    """
    low = str(message or "").strip().casefold()
    if not low or not conversation_history:
        return []

    compare_followup = bool(re.search(
        r"\b(?:yes(?:,)?\s*)?(?:compare|comparison|compare them|compare those|"
        r"compare the two|compare those two|yes|yeah|yep|sure|okay|ok)\b",
        low,
        re.IGNORECASE,
    ))
    if not compare_followup:
        return []

    # Only the immediately preceding assistant turn is eligible.
    previous_assistant = None
    for item in reversed(conversation_history or []):
        role = str(item.get("role") or "").casefold()
        content = str(item.get("text") or item.get("content") or "").strip()
        if not content:
            continue

        if role == "assistant":
            previous_assistant = content
            break

        # If we encounter another substantive user turn before an assistant turn,
        # do not search farther back.
        if role == "user":
            return []

    if not previous_assistant:
        return []

    assistant_low = previous_assistant.casefold()
    if not re.search(
        r"\b(?:compare|comparison|karşılaştır|karsilastir|сравн)\w*\b",
        assistant_low,
        re.IGNORECASE,
    ):
        return []

    # Split into conversational sentences/lines. Find the final compare-offer sentence.
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", previous_assistant)
        if part.strip()
    ]

    compare_idx = None
    for i in range(len(sentences) - 1, -1, -1):
        if re.search(
            r"\b(?:compare|comparison|karşılaştır|karsilastir|сравн)\w*\b",
            sentences[i].casefold(),
            re.IGNORECASE,
        ):
            compare_idx = i
            break

    if compare_idx is None:
        return []

    # If the compare sentence itself names both vehicles, use those.
    direct_targets = resolve_market_vehicle_mentions(sentences[compare_idx])
    if len(direct_targets) >= 2:
        return direct_targets[:2]

    # Otherwise "those two/them" normally refers to the closest named pair just before it.
    # Search a very small local window so older alternatives in the same answer do not leak in.
    for window_size in (1, 2, 3):
        start = max(0, compare_idx - window_size)
        context = " ".join(sentences[start:compare_idx])
        targets = resolve_market_vehicle_mentions(context)
        if len(targets) >= 2:
            # Prefer the last two targets mentioned closest to the compare offer.
            return targets[-2:]

    return []


def _recover_recent_compare_targets(message, conversation_history):
    """
    Recover the most recent explicit multi-model comparison only for a genuine
    constraint-only continuation. This preserves COMPARE state across turns such
    as "Only consider cars below 80,000 km" without carrying stale targets into
    unrelated new searches.
    """
    low = str(message or "").strip().casefold()
    if not low or not conversation_history:
        return []

    continuation_cue = re.search(
        r"\b(?:only consider|consider only|only include|within|under|below|"
        r"maximum|max\b|up to|from\s+(?:19|20)\d{2}|"
        r"(?:19|20)\d{2}\s+(?:or newer|onwards)|"
        r"automatic only|manual only|only automatic|only manual|"
        r"increase (?:my |the )?budget|raise (?:my |the )?budget|"
        r"decrease (?:my |the )?budget|lower (?:my |the )?budget|"
        r"change (?:my |the )?budget|set (?:my |the )?budget|"
        r"which (?:one )?(?:has|shows) (?:the )?(?:stronger|better) "
        r"(?:resale(?:-market)? activity|resale|market activity|turnover)|"
        r"which (?:one )?is (?:stronger|better) for resale|"
        r"sadece|yalnızca|yalnizca|altında|altinda|"
        r"только|до\s+[0-9]|не более)\b",
        low,
    )
    recommendation_cue = re.search(
        r"\b(?:which (?:one )?(?:would you|do you) (?:choose|recommend|pick|prefer)|"
        r"which (?:one )?is (?:better|stronger) based on (?:the )?(?:north cyprus )?market|"
        r"hangisini (?:seçer|secer|önerir|onerir)sin(?:iz)?|"
        r"piyasaya göre hangisi|piyasaya gore hangisi|"
        r"какую (?:выбрать|порекомендуете)|какой (?:выбрать|порекомендуете))\b",
        low,
    )
    if not (continuation_cue or recommendation_cue):
        return []

    # Clear scope changes should start a new search rather than revive old models.
    scope_switch = re.search(
        r"\b(?:make it|instead|actually i want|i want|"
        r"suvs?|crossovers?|pick-?ups?|motorcycles?|scooters?|"
        r"small cars?|small vehicles?|"
        r"motosiklet(?:ler)?|мотоцикл(?:ы)?|скутер(?:ы)?)\b",
        low,
    )
    if scope_switch:
        return []

    # Do not search arbitrarily far back for an old comparison. The immediately
    # preceding substantive USER request defines the active conversational task.
    # This prevents:
    #   compare Fit/Yaris -> new budget search -> economical -> "under 80,000 km"
    # from resurrecting the old Fit/Yaris comparison.
    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").casefold() != "user":
            continue

        content = str(item.get("text") or item.get("content") or "").strip()
        if not content:
            continue

        content_low = content.casefold()
        if not re.search(
            r"\b(?:compare|comparison|versus|vs\.?|karşılaştır|karsilastir|kıyasla|kiyasla|"
            r"сравни(?:ть|те)?|сравнение|против)\b",
            content_low,
        ):
            return []

        targets = resolve_market_vehicle_mentions(content)
        return targets if len(targets) >= 2 else []

    return []


def _recover_recent_recommendation_target(message, conversation_history):
    """
    Resolve contextual references such as "your first recommendation" or
    "the first option" to the first vehicle named in the most recent assistant
    recommendation. This is intentionally narrow so stale models are not carried
    into unrelated turns.
    """
    low = str(message or "").strip().casefold()
    if not low or not conversation_history:
        return []

    first_ref = re.search(
        r"\b(?:first recommendation|first option|first one|top recommendation|"
        r"(?:the )?(?:stronger|better|strongest|best) (?:one|option)|"
        r"(?:the )?one you (?:recommend|recommended|chose|choose|picked|pick|prefer)|"
        r"your recommendation|your choice|"
        r"önerdiğin(?:iz)?|onerdigin(?:iz)?|seçtiğin(?:iz)?|sectigin(?:iz)?|"
        r"рекомендованн\w+|ваш выбор|котор\w+ вы (?:рекомендуете|выбрали))\b",
        low,
    )
    listing_ref = re.search(
        r"\b(?:listings?|ads?|show me actual|for sale|"
        r"ilan(?:lar|ları|lari)?|göster|goster|"
        r"объявлен(?:ие|ия|ий)|покажи|показать)\b",
        low,
    )
    if not (first_ref and listing_ref):
        return []

    contextual_choice_ref = re.search(
        r"\b(?:(?:the )?(?:stronger|better|strongest|best) (?:one|option)|"
        r"(?:the )?one you (?:recommend|recommended|chose|choose|picked|pick|prefer)|"
        r"your recommendation|your choice|"
        r"önerdiğin(?:iz)?|onerdigin(?:iz)?|seçtiğin(?:iz)?|sectigin(?:iz)?|"
        r"рекомендованн\w+|ваш выбор|котор\w+ вы (?:рекомендуете|выбрали))\b",
        low,
    )

    for item in reversed(conversation_history or []):
        if str(item.get("role") or "").casefold() != "assistant":
            continue
        content = str(item.get("text") or item.get("content") or "")

        if contextual_choice_ref:
            # If the preceding comparison explicitly states which model has the
            # stronger resale/turnover signal, resolve that winner first. This is
            # more reliable than taking the first model named in a multi-model
            # comparison paragraph.
            resale_sentences = re.split(r"(?<=[.!?])\s+|\n+", content.strip())
            for sentence in resale_sentences:
                sentence_low = sentence.casefold()
                if not (
                    re.search(r"\b(?:stronger|better|faster)\b", sentence_low)
                    and re.search(
                        r"\b(?:resale|turnover|market activity|historical market signal|"
                        r"observed historical turnover)\b",
                        sentence_low,
                    )
                ):
                    continue
                sentence_targets = resolve_market_vehicle_mentions(sentence)
                if len(sentence_targets) == 1:
                    return [sentence_targets[0]]

            # Recommendation answers deliberately name the selected vehicle in
            # the opening paragraph. Resolve that paragraph next so other compared
            # vehicles later in the answer cannot steal the reference.
            opening = re.split(r"\n\s*\n|(?<=[.!?])\s+", content.strip(), maxsplit=1)[0]
            opening_targets = resolve_market_vehicle_mentions(opening)
            if len(opening_targets) == 1:
                return [opening_targets[0]]
            if len(opening_targets) >= 2:
                opening_cf = opening.casefold()
                ranked_opening_targets = []
                for target in opening_targets:
                    brand = str(target.get("brand") or "").strip()
                    model = str(target.get("model") or "").strip()
                    full_name = f"{brand} {model}".strip().casefold()
                    model_name = model.casefold()
                    full_count = opening_cf.count(full_name) if full_name else 0
                    model_count = opening_cf.count(model_name) if model_name else 0
                    ranked_opening_targets.append(
                        (max(full_count, model_count), target)
                    )
                ranked_opening_targets.sort(key=lambda x: x[0], reverse=True)
                if (
                    ranked_opening_targets
                    and ranked_opening_targets[0][0] > ranked_opening_targets[1][0]
                ):
                    return [ranked_opening_targets[0][1]]

        targets = resolve_market_vehicle_mentions(content)
        if targets:
            return [targets[0]]

    return []


def _looks_like_unknown_explicit_vehicle_shop_request(message, resolved_targets, current_filters, conversation_history):
    """
    Detect a fresh SHOP request that appears to name a specific vehicle, but that
    name cannot be resolved against the live Brand+Model universe.

    This prevents e.g. "Show me listings for Zorblax Hypercar 9000" from silently
    falling back to the entire market. Broad requests such as "show me SUV listings"
    and contextual follow-ups are deliberately excluded.
    """
    if resolved_targets:
        return False
    if current_filters or conversation_history:
        return False

    raw = str(message or "").strip()
    if not raw:
        return False

    patterns = [
        r"(?i)\b(?:listings?|ads?|adverts?|advertisements?)\s+(?:for|of)\s+(.+?)\s*[?.!]*$",
        r"(?i)\bshow\s+me\s+(?:actual\s+)?(.+?)\s+(?:listings?|ads?|adverts?|advertisements?)\s*[?.!]*$",
    ]
    candidate = None
    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            candidate = m.group(1).strip(" \t\r\n.,!?")
            break

    if not candidate:
        return False

    candidate_n = _normalize_vehicle_phrase(candidate)
    generic = {
        "car", "cars", "vehicle", "vehicles", "suv", "suvs", "crossover", "crossovers",
        "motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes",
        "pickup", "pickups", "pick up", "pick ups", "4x4", "4x4s",
        "family car", "family cars", "small car", "small cars", "city car", "city cars",
        "economical car", "economical cars", "automatic car", "automatic cars",
    }
    if candidate_n in generic:
        return False

    # If the phrase contains a real market brand, it can still be a legitimate
    # broad brand request even when no particular model was named.
    if MARKET_READY and market_df is not None and not market_df.empty:
        known_brands = {
            _normalize_vehicle_phrase(x)
            for x in market_df["Brand"].dropna().astype(str).unique().tolist()
            if str(x).strip()
        }
        padded = f" {candidate_n} "
        if any(f" {brand_n} " in padded for brand_n in known_brands if brand_n):
            return False

    # Treat it as an attempted specific name only when it looks name-like:
    # multiple title-cased words and/or a model-number token.
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё0-9-]+", candidate)
    alpha_words = [w for w in words if re.search(r"[A-Za-zÇĞİÖŞÜçğıöşüА-Яа-яЁё]", w)]
    titleish = sum(1 for w in alpha_words if w[:1].isupper())
    has_number = any(re.search(r"\d", w) for w in words)

    return (titleish >= 2 and len(alpha_words) >= 2) or (titleish >= 1 and has_number)



# =========================================================
# BUSINESS ASSISTANT v6 - UNIFIED ACCOUNT / COMPANY CONTEXT
# =========================================================

def _normalize_access_tier(value):
    """
    Normalise future account entitlement without enforcing a paywall yet.

    Current behaviour remains backward-compatible:
    - missing tier -> PERSONAL
    - BUSINESS enables Business entitlement metadata
    - FREE/PERSONAL remain available for the existing buyer assistant

    Enforcement will be added only after the account/payment layer exists.
    """
    value = str(value or "").strip().upper()
    aliases = {
        "FREE": "FREE",
        "PERSONAL": "PERSONAL",
        "BUYER": "PERSONAL",
        "BUSINESS": "BUSINESS",
        "DEALER": "BUSINESS",
        "GALLERY": "BUSINESS",
    }
    return aliases.get(value, "PERSONAL")


def _conversation_text(conversation_history, max_messages=8):
    """
    Safely flatten recent conversation history for lightweight context recovery.
    """
    if not isinstance(conversation_history, list):
        return ""

    parts = []
    for item in conversation_history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            content = item.get("message")
        if content is None:
            content = item.get("text")
        if content:
            parts.append(str(content))

    return "\n".join(parts)


def _resolve_business_company_context(
    message,
    conversation_history=None,
    requested_company=None,
):
    """
    Resolve dealership/company context in the following order:

    1. exact account-linked `business_company`
    2. explicit company mention in the current message
    3. explicit company mention in recent conversation history

    This lets a dealer say:
        "Asal Oto Galeri stokları nasıl?"
        "Hangilerinin fiyatı yüksek?"
    without repeating the company name on every turn.

    The resolution remains exact against Business Company values; fuzzy account
    identity is deliberately avoided.
    """
    if requested_company:
        resolved = _resolve_business_company(
            message="",
            requested_company=requested_company,
        )
        if resolved:
            return resolved

    resolved = _resolve_business_company(message)
    if resolved:
        return resolved

    history_text = _conversation_text(conversation_history)
    if history_text:
        resolved = _resolve_business_company(history_text)
        if resolved:
            return resolved

    return None


def _business_capabilities_payload(access_tier, company):
    """
    Small machine-readable capability block for the frontend/account layer.
    No access is blocked yet; this is integration metadata only.
    """
    return {
        "access_tier": access_tier,
        "business_entitled": access_tier == "BUSINESS",
        "business_company": company,
        "business_company_linked": bool(company),
        "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
        "business_modes": [
            "BUSINESS_ACQUIRE",
            "BUSINESS_STOCK",
            "BUSINESS_PRICE",
            "BUSINESS_MANAGE",
            "BUSINESS_MARKET",
        ],
    }


# =========================================================
# BUSINESS ASSISTANT v5 - MARKET UNDERSTANDING
# =========================================================

def _business_market_intent(message):
    """
    Detect explicit commercial-market questions without intercepting ordinary
    Personal vehicle-shopping questions.
    """
    raw = str(message or "").strip()
    if not raw:
        return False

    low = raw.casefold()

    commercial_cues = [
        r"\bgaleri\b", r"\bgalerici\b", r"\bstok\b", r"\benvanter\b",
        r"\bdealer(?:ship)?\b", r"\binventory\b", r"\bmarket\s+for\s+dealers\b",
        r"\bmy\s+dealership\b", r"\bour\s+dealership\b",
        r"\bавтосалон\b", r"\bдилер\b", r"\bсклад\b",
    ]

    market_cues = [
        r"\bpiyasa\b", r"\bmarket\b", r"\bsegment\b", r"\bkategori\b",
        r"\byavaş\b", r"\byavas\b", r"\bhızlı\b", r"\bhizli\b",
        r"\bhareket\b", r"\bdaral\b", r"\bgeniş\b", r"\bgenis\b",
        r"\bfiyat\s+indir\b", r"\bfiyat\s+düş\b", r"\bfiyat\s+dus\b",
        r"\bprice\s+cut\b", r"\bprice\s+reduction\b",
        r"\bturnover\b", r"\bliquidity\b", r"\bsupply\b",
        r"\bwhat(?:'s|\s+is)\s+happening\b",
        r"\bwhat\s+is\s+moving\b", r"\bwhat\s+is\s+slow\b",
        r"\bрынок\b", r"\bсегмент\b", r"\bликвид\b",
        r"\bснижени.*цен\b", r"\bчто\s+происходит\b",
    ]

    has_commercial = any(re.search(p, low, flags=re.IGNORECASE) for p in commercial_cues)
    has_market = any(re.search(p, low, flags=re.IGNORECASE) for p in market_cues)

    # Also allow clearly dealer-oriented broad market questions even if "dealer"
    # is implied via wording such as "what should galleries watch?"
    dealer_market_phrase = bool(re.search(
        r"\b(?:galeriler|dealers?)\b.{0,40}\b(?:piyasa|market|segment|kategori|stok|supply|turnover)",
        low,
        flags=re.IGNORECASE,
    ))

    return bool((has_commercial and has_market) or dealer_market_phrase)


def _business_market_target_mask(message, frame):
    """
    Narrow Business market analysis to explicit Brand/Model/Category/Year mentions
    where possible. Uses broad textual matching so the function remains robust to
    category wording differences.
    """
    work = frame.copy()
    low = str(message or "").casefold()

    # Year
    years = [int(x) for x in re.findall(r"\b((?:19|20)\d{2})\b", str(message or ""))]
    if years and "Year" in work.columns:
        numeric_year = pd.to_numeric(work["Year"], errors="coerce")
        mask = numeric_year.isin(years)
        if mask.any():
            work = work[mask].copy()

    # Vehicle mention resolver for brand/model/category.
    targets = resolve_market_vehicle_mentions(message)
    if targets:
        mask = pd.Series(False, index=work.index)
        for target in targets:
            this = pd.Series(True, index=work.index)
            if target.get("brand"):
                this &= (
                    work["Brand"].fillna("").astype(str).str.casefold()
                    == str(target.get("brand") or "").casefold()
                )
            if target.get("model"):
                this &= (
                    work["Model"].fillna("").astype(str).str.casefold()
                    == str(target.get("model") or "").casefold()
                )
            if target.get("category") and "CategoryDetail" in work.columns:
                this &= (
                    work["CategoryDetail"].fillna("").astype(str).str.casefold()
                    == str(target.get("category") or "").casefold()
                )
            mask |= this
        if mask.any():
            work = work[mask].copy()

    # Common broad category words that may not resolve through model mentions.
    category_terms = {
        "suv": ["suv", "arazi/suv/pick-up", "arazi"],
        "pickup": ["pick-up", "pickup"],
        "pick-up": ["pick-up", "pickup"],
        "otomobil": ["otomobil"],
        "car": ["otomobil"],
        "cars": ["otomobil"],
    }

    for term, values in category_terms.items():
        if re.search(rf"\b{re.escape(term)}\b", low, flags=re.IGNORECASE):
            if "VehicleType" in work.columns:
                m = work["VehicleType"].fillna("").astype(str).str.casefold().apply(
                    lambda x: any(v.casefold() in x for v in values)
                )
                if m.any():
                    work = work[m].copy()
            break

    return work


def _business_market_rank_rows(frame, mode="balanced", limit=8):
    if frame is None or frame.empty:
        return frame

    work = frame.copy()

    for col in [
        "CurrentListings",
        "ObservedExitWithin60DaysRate",
        "PriceReductionRate",
        "OpportunityPercentile",
        "TurnoverPercentile",
        "CurrentMarketOpportunityPercentile",
        "PricePressureAttractivenessPercentile",
        "CurrentMedianPrice",
        "HistoricalDistinctListings",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    evidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INSUFFICIENT": 9}
    signal_rank = {
        "VERY_STRONG": 0,
        "STRONG": 1,
        "MODERATE": 2,
        "WEAK": 3,
        "CAUTION": 4,
        "INSUFFICIENT_EVIDENCE": 9,
    }

    work["_evidence_rank"] = work["EvidenceQuality"].map(evidence_rank).fillna(9)
    work["_signal_rank"] = work["AcquisitionSignal"].map(signal_rank).fillna(9)

    if mode == "slow":
        work = work.sort_values(
            ["ObservedExitWithin60DaysRate", "PriceReductionRate", "_evidence_rank", "CurrentListings"],
            ascending=[True, False, True, False],
            na_position="last",
        )
    elif mode == "price_pressure":
        work = work.sort_values(
            ["PriceReductionRate", "_evidence_rank", "CurrentListings"],
            ascending=[False, True, False],
            na_position="last",
        )
    elif mode == "supply":
        work = work.sort_values(
            ["CurrentListings", "_evidence_rank", "ObservedExitWithin60DaysRate"],
            ascending=[False, True, False],
            na_position="last",
        )
    else:
        work = work.sort_values(
            ["_signal_rank", "_evidence_rank", "OpportunityPercentile", "CurrentListings"],
            ascending=[True, True, False, False],
            na_position="last",
        )

    # Avoid returning many variants from the same family.
    seen = set()
    selected = []
    for row in work.to_dict("records"):
        key = (
            str(row.get("Brand") or "").casefold(),
            str(row.get("Model") or "").casefold(),
        )
        if key in seen and key != ("", ""):
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break

    return pd.DataFrame(selected)


def _business_market_public(row):
    return {
        "granularity": str(row.get("BusinessGranularity") or "").strip(),
        "vehicle_type": str(row.get("VehicleType") or "").strip(),
        "brand": str(row.get("Brand") or "").strip(),
        "model": str(row.get("Model") or "").strip(),
        "category": str(row.get("CategoryDetail") or "").strip(),
        "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
        "current_listings": int(row["CurrentListings"]) if pd.notna(row.get("CurrentListings")) else None,
        "current_starting_price": float(row["CurrentStartingPrice"]) if pd.notna(row.get("CurrentStartingPrice")) else None,
        "current_median_price": float(row["CurrentMedianPrice"]) if pd.notna(row.get("CurrentMedianPrice")) else None,
        "current_highest_price": float(row["CurrentHighestPrice"]) if pd.notna(row.get("CurrentHighestPrice")) else None,
        "gallery_listings": int(row["GalleryListings"]) if pd.notna(row.get("GalleryListings")) else None,
        "private_listings": int(row["PrivateListings"]) if pd.notna(row.get("PrivateListings")) else None,
        "distinct_companies": int(row["DistinctCompanies"]) if pd.notna(row.get("DistinctCompanies")) else None,
        "historical_distinct_listings": int(row["HistoricalDistinctListings"]) if pd.notna(row.get("HistoricalDistinctListings")) else None,
        "median_observed_days_to_exit": float(row["MedianObservedDaysToExit"]) if pd.notna(row.get("MedianObservedDaysToExit")) else None,
        "observed_exit60_rate": float(row["ObservedExitWithin60DaysRate"]) if pd.notna(row.get("ObservedExitWithin60DaysRate")) else None,
        "price_reduction_rate": float(row["PriceReductionRate"]) if pd.notna(row.get("PriceReductionRate")) else None,
        "evidence_quality": str(row.get("EvidenceQuality") or "").strip(),
        "acquisition_signal": str(row.get("AcquisitionSignal") or "").strip(),
        "opportunity_percentile": float(row["OpportunityPercentile"]) if pd.notna(row.get("OpportunityPercentile")) else None,
        "acquisition_reasons": str(row.get("AcquisitionReasons") or "").strip(),
    }


def _business_market_answer(message, language):
    if not BUSINESS_INTELLIGENCE_READY or business_market_df is None or business_market_df.empty:
        text = {
            "TR": "Business piyasa verisi şu anda hazır değil. Lütfen biraz sonra tekrar deneyin.",
            "EN": "The Business market data is not ready right now. Please try again shortly.",
            "RU": "Business-данные по рынку сейчас недоступны. Попробуйте чуть позже.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_INTELLIGENCE_NOT_READY",
            "rows": [],
        }

    work = _business_market_target_mask(message, business_market_df)

    # Default dealer-market universe: normal dealership vehicles only.
    if "VehicleType" in work.columns:
        normal_mask = work["VehicleType"].fillna("").astype(str).str.casefold().apply(
            lambda x: any(k in x for k in ["otomobil", "suv", "pick-up", "pickup", "arazi"])
        )
        if normal_mask.any():
            work = work[normal_mask].copy()

    low = str(message or "").casefold()
    if re.search(r"\b(yavaş|yavas|slow|slowest|zayıf|zayif|weak|stagn|долго|медлен)\b", low):
        mode = "slow"
    elif re.search(r"\b(price\s+cut|price\s+reduction|fiyat\s+indir|fiyat\s+düş|fiyat\s+dus|снижени.*цен)\b", low):
        mode = "price_pressure"
    elif re.search(r"\b(supply|arz|stok\s+çok|stok\s+cok|çok\s+ilan|cok\s+ilan|предложени)\b", low):
        mode = "supply"
    else:
        mode = "balanced"

    ranked = _business_market_rank_rows(work, mode=mode, limit=8)

    if ranked is None or ranked.empty:
        text = {
            "TR": "Bu piyasa sorusu için yeterli Business verisi bulamadım.",
            "EN": "I couldn't find enough Business-market data for that question.",
            "RU": "Для этого рыночного вопроса недостаточно Business-данных.",
        }
        return text.get(language, text["TR"]), {
            "success": True,
            "rows": [],
            "mode": mode,
        }

    rows = [_business_market_public(r) for r in ranked.to_dict("records")]

    lines = []
    for idx, r in enumerate(rows, 1):
        name = " ".join(
            x for x in [
                str(r.get("year") or "").strip(),
                r.get("brand") or "",
                r.get("model") or "",
                r.get("category") or "",
            ] if x
        ).strip() or r.get("vehicle_type") or "Market segment"

        listings = r.get("current_listings")
        exit60 = r.get("observed_exit60_rate")
        cuts = r.get("price_reduction_rate")
        med_days = r.get("median_observed_days_to_exit")
        median_price = _business_money(r.get("current_median_price"))
        evidence = r.get("evidence_quality") or "—"
        signal = r.get("acquisition_signal") or "—"

        exit_text = f"{exit60*100:.0f}%" if exit60 is not None else "—"
        cuts_text = f"{cuts*100:.0f}%" if cuts is not None else "—"

        if language == "EN":
            parts = [
                f"{listings} current listings" if listings is not None else None,
                f"median ask {median_price}" if median_price else None,
                f"{exit_text} observed 60-day exit",
                f"{cuts_text} historical price-reduction rate",
                f"median observed exit {med_days:.0f}d" if med_days is not None else None,
                f"{evidence.lower()} evidence",
                f"signal {signal.replace('_', ' ').lower()}",
            ]
        elif language == "RU":
            parts = [
                f"{listings} текущих объявлений" if listings is not None else None,
                f"медианная цена {median_price}" if median_price else None,
                f"{exit_text} наблюдаемый выход за 60 дней",
                f"{cuts_text} историческая доля снижения цены",
                f"медианный выход {med_days:.0f} дн." if med_days is not None else None,
                f"достоверность {evidence}",
                f"сигнал {signal}",
            ]
        else:
            parts = [
                f"{listings} güncel ilan" if listings is not None else None,
                f"medyan ilan {median_price}" if median_price else None,
                f"%{exit60*100:.0f} gözlenen 60 günlük çıkış" if exit60 is not None else None,
                f"%{cuts*100:.0f} tarihsel fiyat indirimi" if cuts is not None else None,
                f"medyan gözlenen çıkış {med_days:.0f} gün" if med_days is not None else None,
                f"{evidence.lower()} kanıt",
                f"sinyal {signal.replace('_', ' ').lower()}",
            ]

        lines.append(f"{idx}. {name}\n   " + " · ".join(x for x in parts if x))

    if language == "EN":
        if mode == "slow":
            intro = "These are the weaker/slower-moving areas in the current dealer-market evidence:"
        elif mode == "price_pressure":
            intro = "These areas show the strongest historical asking-price reduction pressure:"
        elif mode == "supply":
            intro = "These areas currently have the heaviest advertised supply:"
        else:
            intro = "These are the strongest current dealer-market opportunities in the available evidence:"
        note = (
            "This is a cross-sectional market snapshot combined with historical listing behaviour. "
            "It does not prove that the overall market is accelerating or slowing over time unless a time-series trend is explicitly available. "
            "Observed market exit means a listing disappeared from observation, not a confirmed sale."
        )
    elif language == "RU":
        if mode == "slow":
            intro = "По текущим данным это более слабые/медленные зоны дилерского рынка:"
        elif mode == "price_pressure":
            intro = "Здесь исторически наблюдается самое сильное давление снижения цен объявлений:"
        elif mode == "supply":
            intro = "Здесь сейчас самая высокая видимая плотность предложений:"
        else:
            intro = "По доступным данным это самые сильные текущие рыночные возможности для дилера:"
        note = (
            "Это текущий срез рынка, объединённый с историческим поведением объявлений. "
            "Он сам по себе не доказывает, что весь рынок ускоряется или замедляется во времени, если нет отдельного временного ряда. "
            "Наблюдаемый выход с рынка означает исчезновение объявления из наблюдения, а не подтверждённую продажу."
        )
    else:
        if mode == "slow":
            intro = "Mevcut dealer piyasa verisinde daha zayıf/yavaş görünen alanlar şunlar:"
        elif mode == "price_pressure":
            intro = "Tarihsel ilan fiyatı indirimi baskısının en yüksek olduğu alanlar şunlar:"
        elif mode == "supply":
            intro = "Şu anda ilan arzının en yoğun olduğu alanlar şunlar:"
        else:
            intro = "Mevcut veriye göre dealer açısından en güçlü piyasa fırsatları şunlar:"
        note = (
            "Bu analiz güncel piyasa kesitini tarihsel ilan davranışıyla birleştirir. "
            "Ayrı bir zaman serisi olmadan piyasanın genel olarak hızlandığını veya yavaşladığını kesin biçimde göstermez. "
            "Gözlenen piyasa çıkışı da doğrulanmış satış anlamına gelmez."
        )

    return intro + "\n\n" + "\n\n".join(lines) + "\n\n" + note, {
        "success": True,
        "rows": rows,
        "mode": mode,
    }


# =========================================================
# BUSINESS ASSISTANT v4 - AGING STOCK / INVENTORY ACTIONS
# =========================================================

def _business_manage_intent(message):
    """
    Detect dealer questions about aging stock and what action to take.
    Kept conservative so ordinary Personal ownership/buying questions are untouched.
    """
    raw = str(message or "").strip()
    if not raw:
        return False

    low = raw.casefold()

    commercial_cues = [
        r"\bstok(?:um|umdaki|larım|larim|ta|taki)?\b",
        r"\bgaleri(?:m|mde|mdeki|min|ye)?\b",
        r"\benvanter(?:im|imde|de)?\b",
        r"\bdealer(?:ship)?\b",
        r"\binventory\b",
        r"\bmy\s+stock\b",
        r"\bour\s+stock\b",
        r"\bmy\s+dealership\b",
        r"\bавтосалон\b",
        r"\bсклад\b",
    ]

    aging_cues = [
        r"\bgündür\b", r"\bgund[üu]r\b",
        r"\bgün\b", r"\bgun\b",
        r"\bhaftadır\b", r"\bhaftadir\b",
        r"\baydır\b", r"\baydir\b",
        r"\bbekliyor\b", r"\bbekleyen\b",
        r"\buzun\s+süredir\b", r"\buzun\s+suredir\b",
        r"\beskiyen\s+stok\b", r"\byaşlanan\s+stok\b", r"\byaslanan\s+stok\b",
        r"\bilanda\s+uzun\b",
        r"\bdays?\b", r"\bweeks?\b", r"\bmonths?\b",
        r"\baging\b", r"\baged\b", r"\bstale\b",
        r"\bsitting\b", r"\btoo\s+long\b",
        r"\blongest\b", r"\boldest\s+stock\b",
        r"\bдн(?:я|ей)?\b", r"\bнедел", r"\bмесяц",
        r"\bдолго\b", r"\bзалежал",
    ]

    action_cues = [
        r"\bne\s+yap\b", r"\bne\s+yapmalıyım\b", r"\bne\s+yapmaliyim\b",
        r"\bne\s+öner\b", r"\bne\s+oner\b",
        r"\bindir\b", r"\bdüşür\b", r"\bdusur\b",
        r"\bfiyatı\s+düşür\b", r"\bfiyati\s+dusur\b",
        r"\bbeklet\b", r"\btut\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bshould\s+i\b", r"\bwhat\s+do\s+i\s+do\b",
        r"\breduce\b", r"\bcut\s+(?:the\s+)?price\b",
        r"\bhold\b", r"\bkeep\s+(?:the\s+)?price\b",
        r"\baction\b", r"\bmanage\b",
        r"\bчто\s+делать\b", r"\bснизить\b", r"\bдержать\b",
    ]

    has_commercial = any(re.search(p, low, flags=re.IGNORECASE) for p in commercial_cues)
    has_aging = any(re.search(p, low, flags=re.IGNORECASE) for p in aging_cues)
    has_action = any(re.search(p, low, flags=re.IGNORECASE) for p in action_cues)

    # Dealer + aging is sufficient for "which of my cars have been sitting too long?"
    # Dealer + explicit action is sufficient for "should I reduce this stock car?"
    return bool(has_commercial and (has_aging or has_action))


def _business_manage_action(row):
    """
    Transparent deterministic action logic based on:
    observed listing age vs historical median/P75,
    current advertised price position, evidence quality,
    and whether the advert has already shown a price reduction.

    No sale probability or guaranteed time-to-sell claims.
    """
    age = pd.to_numeric(pd.Series([row.get("StockAgeDays")]), errors="coerce").iloc[0]
    med = pd.to_numeric(pd.Series([row.get("HistoricalMedianObservedDaysToExit")]), errors="coerce").iloc[0]
    p75 = pd.to_numeric(pd.Series([row.get("HistoricalP75ObservedDaysToExit")]), errors="coerce").iloc[0]

    price_position = str(row.get("PricePositionBand") or "").strip().upper()
    liquidity_conf = str(row.get("LiquidityEvidenceConfidence") or "").strip().upper()
    comp_conf = str(row.get("ComparableEvidenceConfidence") or "").strip().upper()
    reduced = bool(row.get("HasReducedPrice", False))

    reliable_liquidity = liquidity_conf in {"HIGH", "MEDIUM"}
    reliable_price = comp_conf in {"HIGH", "MEDIUM"}

    beyond_p75 = bool(
        pd.notna(age) and pd.notna(p75) and reliable_liquidity and age > p75
    )
    beyond_median = bool(
        pd.notna(age) and pd.notna(med) and reliable_liquidity and age > med
    )
    high_price = price_position in {"HIGH", "HIGH_MID"}
    low_price = price_position in {"LOW", "LOW_MID"}
    mid_price = price_position == "MID_MARKET"

    reasons = []

    if not reliable_liquidity and not reliable_price:
        action = "REVIEW_MANUALLY"
        reasons.append("historical age and comparable-price evidence are limited")
    elif beyond_p75 and high_price and reliable_price:
        action = "REPRICE_REVIEW"
        reasons.append("listing age is beyond the historical upper-quartile exit benchmark")
        reasons.append("current asking price is above the middle of comparable advertised prices")
        if reduced:
            reasons.append("the advert has already shown at least one asking-price reduction")
    elif beyond_p75 and (mid_price or low_price):
        action = "NON_PRICE_REVIEW"
        reasons.append("listing age is beyond the historical upper-quartile exit benchmark")
        reasons.append("price position is not obviously high versus current comparables")
        if reduced:
            reasons.append("the advert has already shown at least one asking-price reduction")
    elif beyond_median and high_price and reliable_price:
        action = "WATCH_REPRICE"
        reasons.append("listing age is above the historical median observed exit benchmark")
        reasons.append("current asking price is toward the upper end of comparable advertised prices")
    elif beyond_median:
        action = "WATCH"
        reasons.append("listing age is above the historical median observed exit benchmark")
        if low_price:
            reasons.append("price is already below the middle of comparable advertised prices")
    else:
        action = "HOLD_MONITOR"
        if pd.notna(age) and pd.notna(med) and reliable_liquidity:
            reasons.append("listing age is still within the historical median observed exit benchmark")
        else:
            reasons.append("there is not enough age pressure in the available evidence to justify an automatic price action")
        if high_price and reliable_price:
            reasons.append("price is relatively high, but age evidence does not yet point to urgent action")

    return action, reasons


def _business_manage_row_public(row):
    action, action_reasons = _business_manage_action(row)

    return {
        "link": str(row.get("Link") or "").strip(),
        "company": str(row.get("Company") or "").strip(),
        "brand": str(row.get("Brand") or "").strip(),
        "model": str(row.get("Model") or "").strip(),
        "category": str(row.get("CategoryDetail") or "").strip(),
        "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
        "km": float(row["KM"]) if pd.notna(row.get("KM")) else None,
        "asking_price": float(row["CurrentAskingPrice"]) if pd.notna(row.get("CurrentAskingPrice")) else None,
        "listing_age_days": int(row["StockAgeDays"]) if pd.notna(row.get("StockAgeDays")) else None,
        "listing_age_is_lower_bound": bool(row.get("PublicListingAgeIsLowerBound", False)),
        "historical_median_days_to_exit": float(row["HistoricalMedianObservedDaysToExit"]) if pd.notna(row.get("HistoricalMedianObservedDaysToExit")) else None,
        "historical_p75_days_to_exit": float(row["HistoricalP75ObservedDaysToExit"]) if pd.notna(row.get("HistoricalP75ObservedDaysToExit")) else None,
        "liquidity_confidence": str(row.get("LiquidityEvidenceConfidence") or "").strip(),
        "historical_exit60_rate": float(row["ObservedExit60Rate"]) if pd.notna(row.get("ObservedExit60Rate")) else None,
        "historical_price_reduction_rate": float(row["HistoricalPriceReductionRate"]) if pd.notna(row.get("HistoricalPriceReductionRate")) else None,
        "price_position": str(row.get("PricePositionBand") or "").strip(),
        "comparable_confidence": str(row.get("ComparableEvidenceConfidence") or "").strip(),
        "comparable_count": int(row["ComparableListings"]) if pd.notna(row.get("ComparableListings")) else None,
        "comparable_median_price": float(row["ComparableMedianPrice"]) if pd.notna(row.get("ComparableMedianPrice")) else None,
        "comparable_p25_price": float(row["ComparableP25Price"]) if pd.notna(row.get("ComparableP25Price")) else None,
        "comparable_p75_price": float(row["ComparableP75Price"]) if pd.notna(row.get("ComparableP75Price")) else None,
        # Do not trust the upstream PriceVsMedianPct scale here.  Some rows in
        # historical exports encode it as a ratio while others encode a percent,
        # which can make a £17k ask vs a £6.25k median appear as ~2% above.
        # The asking price and comparable median are the authoritative inputs, so
        # derive the percentage directly from those values whenever possible.
        "price_vs_median_pct": (
            ((float(row["CurrentAskingPrice"]) - float(row["ComparableMedianPrice"]))
             / float(row["ComparableMedianPrice"]) * 100.0)
            if pd.notna(row.get("CurrentAskingPrice"))
            and pd.notna(row.get("ComparableMedianPrice"))
            and float(row["ComparableMedianPrice"]) > 0
            else (float(row["PriceVsMedianPct"]) if pd.notna(row.get("PriceVsMedianPct")) else None)
        ),
        "has_reduced_price": bool(row.get("HasReducedPrice", False)),
        "price_change_pct": float(row["PriceChangePct"]) if pd.notna(row.get("PriceChangePct")) else None,
        "attention_level": str(row.get("AttentionLevel") or "").strip(),
        "recommended_action": action,
        "action_reasons": action_reasons,
    }


def _business_manage_action_label(action, language):
    labels = {
        "TR": {
            "REPRICE_REVIEW": "fiyatı yeniden değerlendir",
            "NON_PRICE_REVIEW": "fiyat dışı nedenleri incele",
            "WATCH_REPRICE": "yakından izle; fiyatı gözden geçir",
            "WATCH": "izle",
            "HOLD_MONITOR": "şimdilik koru ve izle",
            "REVIEW_MANUALLY": "manuel inceleme gerekli",
        },
        "EN": {
            "REPRICE_REVIEW": "review the price",
            "NON_PRICE_REVIEW": "review non-price factors",
            "WATCH_REPRICE": "watch closely; review pricing",
            "WATCH": "monitor",
            "HOLD_MONITOR": "hold for now and monitor",
            "REVIEW_MANUALLY": "manual review needed",
        },
        "RU": {
            "REPRICE_REVIEW": "пересмотреть цену",
            "NON_PRICE_REVIEW": "проверить неценовые факторы",
            "WATCH_REPRICE": "внимательно следить и проверить цену",
            "WATCH": "наблюдать",
            "HOLD_MONITOR": "пока оставить и наблюдать",
            "REVIEW_MANUALLY": "нужна ручная проверка",
        },
    }
    return labels.get(language, labels["TR"]).get(action, action)


def _business_manage_answer(message, language, requested_company=None):
    if not BUSINESS_INTELLIGENCE_READY or business_stock_df is None or business_stock_df.empty:
        text = {
            "TR": "Business stok yönetimi verisi şu anda hazır değil. Lütfen biraz sonra tekrar deneyin.",
            "EN": "The Business stock-management data is not ready right now. Please try again shortly.",
            "RU": "Данные Business для управления складом сейчас недоступны. Попробуйте чуть позже.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_INTELLIGENCE_NOT_READY",
            "company": None,
            "vehicles": [],
        }

    company, work = _business_stock_target_rows(
        message,
        requested_company=requested_company,
    )

    explicit_targets = resolve_market_vehicle_mentions(message)

    if not company and not explicit_targets:
        text = {
            "TR": "Stok yönetimi analizini hangi galeri için yapacağımı bilmem gerekiyor. Şimdilik mesajınıza galeri adını ekleyin; hesap bağlantısı geldiğinde bu otomatik olacak.",
            "EN": "I need to know which dealership's stock to manage. For now, include the dealership name in your message; account linking will make this automatic later.",
            "RU": "Мне нужно знать, склад какого автосалона анализировать. Пока укажите название в сообщении; позже это будет определяться по аккаунту автоматически.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_COMPANY_REQUIRED",
            "company": None,
            "vehicles": [],
        }

    if work.empty:
        text = {
            "TR": f"{company or 'Bu galeri'} stoklarında mesajınızdaki aracı bulamadım.",
            "EN": f"I couldn't find the referenced vehicle in {(company or 'this dealership')}'s current stock.",
            "RU": f"Я не нашёл указанный автомобиль в текущем складе {company or 'этого автосалона'}.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_STOCK_VEHICLE_NOT_FOUND",
            "company": company,
            "vehicles": [],
        }

    # For portfolio questions, prioritise the most aged/high-attention stock.
    # For an explicitly named vehicle, retain all matching rows but keep the most
    # decision-relevant variants at the top.
    age = pd.to_numeric(work["StockAgeDays"], errors="coerce")
    p75 = pd.to_numeric(work["HistoricalP75ObservedDaysToExit"], errors="coerce")
    median = pd.to_numeric(work["HistoricalMedianObservedDaysToExit"], errors="coerce")
    price_vs = pd.to_numeric(work["PriceVsMedianPct"], errors="coerce")

    work["_beyond_p75"] = ((age > p75) & p75.notna()).astype(int)
    work["_beyond_median"] = ((age > median) & median.notna()).astype(int)
    work["_age"] = age
    work["_price_vs"] = price_vs

    attention_rank = {"HIGH": 0, "ATTENTION": 1, "WATCH": 2, "NONE": 3}
    work["_attention_rank"] = work["AttentionLevel"].map(attention_rank).fillna(9)

    work = work.sort_values(
        ["_beyond_p75", "_beyond_median", "_attention_rank", "_age", "_price_vs"],
        ascending=[False, False, True, False, False],
        na_position="last",
    )

    limit = 5 if explicit_targets else 8
    vehicles = [
        _business_manage_row_public(r)
        for r in work.head(limit).to_dict("records")
    ]

    lines = []
    for idx, v in enumerate(vehicles, 1):
        name = " ".join(
            x for x in [
                str(v.get("year") or "").strip(),
                v.get("brand") or "",
                v.get("model") or "",
                v.get("category") or "",
            ] if x
        ).strip()

        ask = _business_money(v.get("asking_price"))
        med_price = _business_money(v.get("comparable_median_price"))
        p75_price = _business_money(v.get("comparable_p75_price"))
        age_days = v.get("listing_age_days")
        med_days = v.get("historical_median_days_to_exit")
        p75_days = v.get("historical_p75_days_to_exit")
        action = _business_manage_action_label(v.get("recommended_action"), language)

        if language == "EN":
            age_text = (
                f"observed for at least {age_days} days"
                if v.get("listing_age_is_lower_bound")
                else f"observed advertised for {age_days} days"
            ) if age_days is not None else "listing age unavailable"
            benchmarks = []
            if med_days is not None:
                benchmarks.append(f"historical median exit benchmark {med_days:.0f}d")
            if p75_days is not None:
                benchmarks.append(f"P75 {p75_days:.0f}d")
            price_bits = [
                f"ask {ask}" if ask else None,
                f"price position {_business_price_position_label(v.get('price_position'), 'EN')}",
                f"comparable median {med_price}" if med_price else None,
                f"P75 price {p75_price}" if p75_price else None,
            ]
            lines.append(
                f"{idx}. {name} — {action}\n   "
                + " · ".join([age_text] + benchmarks + [x for x in price_bits if x])
            )

        elif language == "RU":
            age_text = (
                f"наблюдается как минимум {age_days} дн."
                if v.get("listing_age_is_lower_bound")
                else f"наблюдается в объявлениях {age_days} дн."
            ) if age_days is not None else "возраст объявления неизвестен"
            benchmarks = []
            if med_days is not None:
                benchmarks.append(f"историческая медиана выхода {med_days:.0f} дн.")
            if p75_days is not None:
                benchmarks.append(f"P75 {p75_days:.0f} дн.")
            price_bits = [
                f"цена {ask}" if ask else None,
                f"позиция {_business_price_position_label(v.get('price_position'), 'RU')}",
                f"медиана сопоставимых {med_price}" if med_price else None,
                f"P75 цены {p75_price}" if p75_price else None,
            ]
            lines.append(
                f"{idx}. {name} — {action}\n   "
                + " · ".join([age_text] + benchmarks + [x for x in price_bits if x])
            )

        else:
            age_text = (
                f"en az {age_days} gündür gözlemleniyor"
                if v.get("listing_age_is_lower_bound")
                else f"{age_days} gündür ilanda gözlemleniyor"
            ) if age_days is not None else "ilan yaşı bilinmiyor"
            benchmarks = []
            if med_days is not None:
                benchmarks.append(f"tarihsel medyan çıkış {med_days:.0f} gün")
            if p75_days is not None:
                benchmarks.append(f"P75 {p75_days:.0f} gün")
            price_bits = [
                f"ilan {ask}" if ask else None,
                f"fiyat konumu {_business_price_position_label(v.get('price_position'), 'TR')}",
                f"benzer ilan medyanı {med_price}" if med_price else None,
                f"fiyat P75 {p75_price}" if p75_price else None,
            ]
            lines.append(
                f"{idx}. {name} — {action}\n   "
                + " · ".join([age_text] + benchmarks + [x for x in price_bits if x])
            )

    if language == "EN":
        if explicit_targets:
            intro = "For the referenced stock, this is the action the current evidence supports:"
        else:
            intro = f"For {company}, these are the vehicles I would review first based on observed listing age and current price position:"
        note = (
            "This is inventory-management evidence, not a sale prediction. "
            "Listing age starts when OtoDeğer first observed the advert, not when the dealer acquired the vehicle. "
            "Historical market exit means the listing disappeared from observation; it is not a confirmed sale. "
            "A price reduction should only be considered after checking condition, specification, preparation costs and your margin."
        )
    elif language == "RU":
        if explicit_targets:
            intro = "По указанному автомобилю текущие данные поддерживают следующее действие:"
        else:
            intro = f"Для {company} в первую очередь я бы проверил эти автомобили по наблюдаемому возрасту объявления и текущей ценовой позиции:"
        note = (
            "Это данные для управления складом, а не прогноз продажи. "
            "Возраст считается с момента, когда OtoDeğer впервые увидел объявление, а не с даты покупки автомобиля дилером. "
            "Исторический выход с рынка означает исчезновение объявления из наблюдения, а не подтверждённую продажу. "
            "Перед снижением цены нужно учитывать состояние, комплектацию, затраты на подготовку и вашу маржу."
        )
    else:
        if explicit_targets:
            intro = "Mesajınızdaki stok aracı için mevcut verinin desteklediği aksiyon şu:"
        else:
            intro = f"{company} için gözlenen ilan yaşı ve mevcut fiyat konumuna göre önce şu araçları gözden geçirirdim:"
        note = (
            "Bu bir satış tahmini değil, stok yönetimi göstergesidir. "
            "İlan yaşı galerinin aracı aldığı tarihten değil, OtoDeğer'in ilanı ilk gördüğü tarihten başlar. "
            "Tarihsel piyasa çıkışı da doğrulanmış satış anlamına gelmez. "
            "Fiyat indirimi düşünmeden önce kondisyon, donanım, hazırlık maliyeti ve marjınızı da kontrol etmek gerekir."
        )

    return intro + "\n\n" + "\n\n".join(lines) + "\n\n" + note, {
        "success": True,
        "company": company,
        "vehicles": vehicles,
    }


# =========================================================
# BUSINESS ASSISTANT v3 - PRICE / COMPETITIVE POSITIONING
# =========================================================

def _business_price_intent(message):
    """
    Detect explicit dealer/inventory pricing questions without intercepting
    ordinary Personal buyer price questions.
    """
    raw = str(message or "").strip()
    if not raw:
        return False

    low = raw.casefold()

    commercial_cues = [
        r"\bstok(?:um|umdaki|larım|larim|ta|taki)?\b",
        r"\bgaleri(?:m|mde|mdeki|min|ye)?\b",
        r"\benvanter(?:im|imde|de)?\b",
        r"\bdealer(?:ship)?\b",
        r"\binventory\b",
        r"\bmy\s+stock\b",
        r"\bour\s+stock\b",
        r"\bmy\s+car\b",
        r"\bour\s+car\b",
        r"\bдилер\b",
        r"\bавтосалон\b",
        r"\bсклад\b",
    ]

    price_cues = [
        r"\bfiyat\b", r"\bfiyatı\b", r"\bfiyati\b", r"\bfiyatlar\b",
        r"\bpahalı\b", r"\bpahali\b", r"\bucuz\b",
        r"\byüksek\b", r"\byuksek\b",
        r"\bprice\b", r"\bpricing\b", r"\bpriced\b",
        r"\boverpriced\b", r"\bunderpriced\b",
        r"\bexpensive\b", r"\bcheap\b",
        r"\bcompetitive\b", r"\bcompetitively\b",
        r"\bhow\s+high\b", r"\bhow\s+much\s+should\s+i\s+(?:ask|price)\b",
        r"\basking\s+price\b",
        r"\bцена\b", r"\bцену\b", r"\bдорог\b", r"\bдешев\b",
        r"\bконкурент\b",
    ]

    has_commercial = any(re.search(p, low, flags=re.IGNORECASE) for p in commercial_cues)
    has_price = any(re.search(p, low, flags=re.IGNORECASE) for p in price_cues)

    return bool(has_commercial and has_price)


def _business_stock_target_rows(message, requested_company=None):
    """
    Resolve Business-stock rows referenced by the user.

    Company scope is applied when known. Vehicle mentions use the existing
    deterministic live-market mention resolver so Personal and Business
    understand Brand/Model/Category names consistently.
    """
    company = _resolve_business_company(message, requested_company=requested_company)

    if business_stock_df is None or business_stock_df.empty:
        return company, pd.DataFrame()

    work = business_stock_df.copy()

    if company:
        work = work[
            work["Company"].fillna("").astype(str).str.casefold()
            == str(company).casefold()
        ].copy()

    targets = resolve_market_vehicle_mentions(message)

    # Explicit year, when present, is useful for disambiguating stock rows.
    years = [
        int(x)
        for x in re.findall(r"\b((?:19|20)\d{2})\b", str(message or ""))
    ]

    if targets:
        mask = pd.Series(False, index=work.index)
        for target in targets:
            this_mask = (
                work["Brand"].fillna("").astype(str).str.casefold()
                == str(target.get("brand") or "").casefold()
            ) & (
                work["Model"].fillna("").astype(str).str.casefold()
                == str(target.get("model") or "").casefold()
            )

            if target.get("category"):
                this_mask &= (
                    work["CategoryDetail"].fillna("").astype(str).str.casefold()
                    == str(target.get("category") or "").casefold()
                )

            mask |= this_mask

        work = work[mask].copy()

    if years and not work.empty:
        numeric_year = pd.to_numeric(work["Year"], errors="coerce")
        year_mask = numeric_year.isin(years)
        if year_mask.any():
            work = work[year_mask].copy()

    return company, work


def _business_price_position_label(value, language):
    value = str(value or "").strip().upper()

    labels = {
        "TR": {
            "HIGH": "yüksek",
            "HIGH_MID": "yüksek-orta",
            "MID_MARKET": "piyasa ortası",
            "LOW_MID": "düşük-orta",
            "LOW": "düşük",
            "INSUFFICIENT_EVIDENCE": "yetersiz veri",
        },
        "EN": {
            "HIGH": "high",
            "HIGH_MID": "upper-middle",
            "MID_MARKET": "mid-market",
            "LOW_MID": "lower-middle",
            "LOW": "low",
            "INSUFFICIENT_EVIDENCE": "insufficient evidence",
        },
        "RU": {
            "HIGH": "высокая",
            "HIGH_MID": "выше средней",
            "MID_MARKET": "середина рынка",
            "LOW_MID": "ниже средней",
            "LOW": "низкая",
            "INSUFFICIENT_EVIDENCE": "недостаточно данных",
        },
    }
    return labels.get(language, labels["TR"]).get(value, value or "—")


def _business_price_row_public(row):
    return {
        "link": str(row.get("Link") or "").strip(),
        "company": str(row.get("Company") or "").strip(),
        "brand": str(row.get("Brand") or "").strip(),
        "model": str(row.get("Model") or "").strip(),
        "category": str(row.get("CategoryDetail") or "").strip(),
        "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
        "km": float(row["KM"]) if pd.notna(row.get("KM")) else None,
        "asking_price": float(row["CurrentAskingPrice"]) if pd.notna(row.get("CurrentAskingPrice")) else None,
        "comparable_count": int(row["ComparableListings"]) if pd.notna(row.get("ComparableListings")) else None,
        "comparable_confidence": str(row.get("ComparableEvidenceConfidence") or "").strip(),
        "comparable_median_price": float(row["ComparableMedianPrice"]) if pd.notna(row.get("ComparableMedianPrice")) else None,
        "comparable_p25_price": float(row["ComparableP25Price"]) if pd.notna(row.get("ComparableP25Price")) else None,
        "comparable_p75_price": float(row["ComparableP75Price"]) if pd.notna(row.get("ComparableP75Price")) else None,
        "comparable_min_price": float(row["ComparableMinPrice"]) if pd.notna(row.get("ComparableMinPrice")) else None,
        "comparable_max_price": float(row["ComparableMaxPrice"]) if pd.notna(row.get("ComparableMaxPrice")) else None,
        "price_vs_median_pct": float(row["PriceVsMedianPct"]) if pd.notna(row.get("PriceVsMedianPct")) else None,
        "price_percentile": float(row["PricePercentile"]) if pd.notna(row.get("PricePercentile")) else None,
        "price_position": str(row.get("PricePositionBand") or "").strip(),
        "benchmark_source": str(row.get("BenchmarkSource") or "").strip(),
        "listing_age_days": int(row["StockAgeDays"]) if pd.notna(row.get("StockAgeDays")) else None,
        "listing_age_is_lower_bound": bool(row.get("PublicListingAgeIsLowerBound", False)),
    }


def _business_price_answer(message, language, requested_company=None):
    if not BUSINESS_INTELLIGENCE_READY or business_stock_df is None or business_stock_df.empty:
        text = {
            "TR": "Business fiyat verisi şu anda hazır değil. Lütfen biraz sonra tekrar deneyin.",
            "EN": "The Business pricing data is not ready right now. Please try again shortly.",
            "RU": "Данные Business по ценам сейчас недоступны. Попробуйте чуть позже.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_INTELLIGENCE_NOT_READY",
            "company": None,
            "vehicles": [],
        }

    company, work = _business_stock_target_rows(
        message,
        requested_company=requested_company,
    )

    low = str(message or "").casefold()

    wants_portfolio_review = bool(re.search(
        r"\b(?:hangi|which|show|göster|goster|list|tüm|tum|all)\b.{0,40}"
        r"\b(?:pahalı|pahali|yüksek|yuksek|overpriced|high[- ]priced|price|fiyat|дорог|цена)",
        low,
        flags=re.IGNORECASE,
    ))

    # If no explicit vehicle was resolved, a dealer-wide pricing question
    # requires a known company. Otherwise we cannot know whose stock to inspect.
    if work.empty and not company:
        text = {
            "TR": "Fiyat analizini hangi galeri için yapacağımı bilmem gerekiyor. Şimdilik mesajınıza galeri adını ekleyin; hesap bağlantısı geldiğinde bu otomatik olacak.",
            "EN": "I need to know which dealership's stock to price-check. For now, include the dealership name in your message; account linking will make this automatic later.",
            "RU": "Мне нужно знать, для какого автосалона проверять цены. Пока укажите название в сообщении; позже это будет определяться по аккаунту автоматически.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_COMPANY_REQUIRED",
            "company": None,
            "vehicles": [],
        }

    if company and work.empty:
        # Company exists but no target rows matched.
        text = {
            "TR": f"{company} stoklarında mesajınızdaki aracı bulamadım.",
            "EN": f"I couldn't find the vehicle you referenced in {company}'s current stock.",
            "RU": f"Я не нашёл указанный автомобиль в текущем складе {company}.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_STOCK_VEHICLE_NOT_FOUND",
            "company": company,
            "vehicles": [],
        }

    # Portfolio-level price review: show highest price-position stock first.
    if wants_portfolio_review or not resolve_market_vehicle_mentions(message):
        position_rank = {
            "HIGH": 0,
            "HIGH_MID": 1,
            "MID_MARKET": 2,
            "LOW_MID": 3,
            "LOW": 4,
            "INSUFFICIENT_EVIDENCE": 9,
        }
        work["_position_rank"] = work["PricePositionBand"].map(position_rank).fillna(9)
        work["_price_vs_median"] = pd.to_numeric(work["PriceVsMedianPct"], errors="coerce")
        work["_asking"] = pd.to_numeric(work["CurrentAskingPrice"], errors="coerce")

        review = work.sort_values(
            ["_position_rank", "_price_vs_median", "_asking"],
            ascending=[True, False, False],
            na_position="last",
        ).head(8)

        vehicles = [_business_price_row_public(r) for r in review.to_dict("records")]

        if not vehicles:
            text = {
                "TR": f"{company} için karşılaştırılabilir fiyat kanıtı olan stok bulamadım.",
                "EN": f"I couldn't find stock with usable comparable-price evidence for {company}.",
                "RU": f"Для {company} я не нашёл склад с достаточными сопоставимыми ценовыми данными.",
            }
            return text.get(language, text["TR"]), {
                "success": True,
                "company": company,
                "vehicles": [],
            }

        lines = []
        for idx, v in enumerate(vehicles, 1):
            name = " ".join(
                x for x in [
                    str(v.get("year") or "").strip(),
                    v.get("brand") or "",
                    v.get("model") or "",
                    v.get("category") or "",
                ] if x
            ).strip()

            asking = _business_money(v.get("asking_price"))
            median = _business_money(v.get("comparable_median_price"))
            p75 = _business_money(v.get("comparable_p75_price"))
            diff = v.get("price_vs_median_pct")
            diff_pct = round(diff * 100) if diff is not None else None
            position = _business_price_position_label(v.get("price_position"), language)
            confidence = v.get("comparable_confidence") or "—"

            if language == "EN":
                parts = [
                    f"asking {asking}" if asking else None,
                    f"{position} price position",
                    f"comparable median {median}" if median else None,
                    f"upper quartile {p75}" if p75 else None,
                    f"{diff_pct:+d}% vs median" if diff_pct is not None else None,
                    f"{confidence.lower()} evidence",
                ]
            elif language == "RU":
                parts = [
                    f"цена {asking}" if asking else None,
                    f"позиция: {position}",
                    f"медиана сопоставимых {median}" if median else None,
                    f"верхний квартиль {p75}" if p75 else None,
                    f"{diff_pct:+d}% к медиане" if diff_pct is not None else None,
                    f"достоверность: {confidence}",
                ]
            else:
                confidence_tr = {
                    "HIGH": "yüksek",
                    "MEDIUM": "orta",
                    "LOW": "düşük",
                }.get(confidence, confidence.casefold())
                parts = [
                    f"ilan {asking}" if asking else None,
                    f"fiyat konumu: {position}",
                    f"benzer ilan medyanı {median}" if median else None,
                    f"üst çeyrek {p75}" if p75 else None,
                    f"medyana göre %{diff_pct:+d}" if diff_pct is not None else None,
                    f"{confidence_tr} kanıt",
                ]

            lines.append(
                f"{idx}. {name}\n   " + " · ".join(x for x in parts if x)
            )

        if language == "EN":
            intro = (
                f"The first {company} vehicles I would price-check are the ones sitting highest relative to comparable current asking prices:"
            )
            note = (
                "These comparisons use advertised prices, not confirmed transaction prices. "
                "A high price position does not automatically mean the vehicle should be reduced; specification, condition and acquisition economics can justify a premium."
            )
        elif language == "RU":
            intro = (
                f"В первую очередь у {company} я бы проверил автомобили, которые стоят выше всего относительно сопоставимых текущих цен объявлений:"
            )
            note = (
                "Сравнение основано на ценах объявлений, а не на подтверждённых ценах сделок. "
                "Высокая ценовая позиция не означает автоматически, что цену нужно снижать: комплектация, состояние и закупочная экономика могут оправдывать премию."
            )
        else:
            intro = (
                f"{company} stoklarında önce, benzer güncel ilanlara göre en yüksek fiyat konumunda duran araçları kontrol ederdim:"
            )
            note = (
                "Bu karşılaştırma gerçekleşmiş satış fiyatlarına değil, ilan fiyatlarına dayanır. "
                "Yüksek fiyat konumu tek başına indirim gerektiği anlamına gelmez; donanım, kondisyon ve alış maliyeti primi haklı çıkarabilir."
            )

        return intro + "\n\n" + "\n\n".join(lines) + "\n\n" + note, {
            "success": True,
            "company": company,
            "vehicles": vehicles,
        }

    # Specific-vehicle pricing analysis.
    position_rank = {
        "HIGH": 0,
        "HIGH_MID": 1,
        "MID_MARKET": 2,
        "LOW_MID": 3,
        "LOW": 4,
        "INSUFFICIENT_EVIDENCE": 9,
    }
    work["_position_rank"] = work["PricePositionBand"].map(position_rank).fillna(9)
    work["_comparables"] = pd.to_numeric(work["ComparableListings"], errors="coerce")
    work = work.sort_values(
        ["_position_rank", "_comparables"],
        ascending=[True, False],
        na_position="last",
    ).head(5)

    vehicles = [_business_price_row_public(r) for r in work.to_dict("records")]

    if not vehicles:
        text = {
            "TR": "Bu araç için yeterli stok/fiyat verisi bulamadım.",
            "EN": "I couldn't find enough stock/pricing data for that vehicle.",
            "RU": "По этому автомобилю недостаточно данных о складе и ценах.",
        }
        return text.get(language, text["TR"]), {
            "success": True,
            "company": company,
            "vehicles": [],
        }

    lines = []
    for idx, v in enumerate(vehicles, 1):
        name = " ".join(
            x for x in [
                str(v.get("year") or "").strip(),
                v.get("brand") or "",
                v.get("model") or "",
                v.get("category") or "",
            ] if x
        ).strip()

        asking = _business_money(v.get("asking_price"))
        median = _business_money(v.get("comparable_median_price"))
        p25 = _business_money(v.get("comparable_p25_price"))
        p75 = _business_money(v.get("comparable_p75_price"))
        diff = v.get("price_vs_median_pct")
        diff_pct = round(diff * 100) if diff is not None else None
        position = _business_price_position_label(v.get("price_position"), language)
        confidence = v.get("comparable_confidence") or "—"
        comp_n = v.get("comparable_count")

        if language == "EN":
            competitive = (
                f"A defensible advertised-market range is roughly {p25}–{p75}"
                if p25 and p75 else
                f"The comparable median is about {median}" if median else
                "Comparable-price evidence is limited"
            )
            parts = [
                f"current ask {asking}" if asking else None,
                f"{position} position",
                f"{diff_pct:+d}% vs comparable median" if diff_pct is not None else None,
                competitive,
                f"{comp_n} comparables" if comp_n is not None else None,
                f"{confidence.lower()} evidence",
            ]
        elif language == "RU":
            competitive = (
                f"Ориентир конкурентного диапазона объявлений: примерно {p25}–{p75}"
                if p25 and p75 else
                f"Медиана сопоставимых объявлений: около {median}" if median else
                "Сопоставимых ценовых данных мало"
            )
            parts = [
                f"текущая цена {asking}" if asking else None,
                f"позиция: {position}",
                f"{diff_pct:+d}% к медиане" if diff_pct is not None else None,
                competitive,
                f"сопоставимых объявлений: {comp_n}" if comp_n is not None else None,
                f"достоверность: {confidence}",
            ]
        else:
            competitive = (
                f"İlan piyasasında savunulabilir rekabetçi aralık yaklaşık {p25}–{p75}"
                if p25 and p75 else
                f"Benzer ilan medyanı yaklaşık {median}" if median else
                "Karşılaştırılabilir fiyat kanıtı sınırlı"
            )
            confidence_tr = {
                "HIGH": "yüksek",
                "MEDIUM": "orta",
                "LOW": "düşük",
            }.get(confidence, confidence.casefold())
            parts = [
                f"mevcut ilan {asking}" if asking else None,
                f"fiyat konumu: {position}",
                f"benzer ilan medyanına göre %{diff_pct:+d}" if diff_pct is not None else None,
                competitive,
                f"{comp_n} benzer ilan" if comp_n is not None else None,
                f"{confidence_tr} kanıt",
            ]

        lines.append(f"{idx}. {name}\n   " + " · ".join(x for x in parts if x))

    if language == "EN":
        intro = (
            f"For {company}, this is how the referenced stock sits against comparable current asking prices:"
            if company else
            "This is how the referenced listing sits against comparable current asking prices:"
        )
        note = (
            "Treat the P25–P75 range as competitive advertised-market positioning, not a guaranteed sale-price range. "
            "The evidence does not include the vehicle's condition, exact optional equipment, preparation cost or your acquisition cost."
        )
    elif language == "RU":
        intro = (
            f"Для {company} указанный автомобиль выглядит так относительно сопоставимых текущих цен объявлений:"
            if company else
            "Указанный автомобиль выглядит так относительно сопоставимых текущих цен объявлений:"
        )
        note = (
            "Диапазон P25–P75 — это ориентир конкурентного позиционирования по объявлениям, а не гарантированный диапазон цены продажи. "
            "Здесь не учтены состояние, точная комплектация, затраты на подготовку и ваша закупочная цена."
        )
    else:
        intro = (
            f"{company} için mesajınızdaki aracın benzer güncel ilanlara göre fiyat konumu şöyle:"
            if company else
            "Mesajınızdaki aracın benzer güncel ilanlara göre fiyat konumu şöyle:"
        )
        note = (
            "P25–P75 aralığını gerçekleşecek satış fiyatı değil, rekabetçi ilan fiyatı konumu olarak düşünün. "
            "Kondisyon, tam donanım, hazırlık maliyeti ve sizin alış maliyetiniz bu veride yok."
        )

    return intro + "\n\n" + "\n\n".join(lines) + "\n\n" + note, {
        "success": True,
        "company": company,
        "vehicles": vehicles,
    }


# =========================================================
# BUSINESS ASSISTANT v2 - MY STOCK / INVENTORY HEALTH
# =========================================================

def _normalize_company_name(value):
    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9çğıöşüа-яё]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _resolve_business_company(message, requested_company=None):
    """
    Resolve the gallery/company either from an explicit request field
    (future account association) or from a clear company mention in the message.
    """
    if not BUSINESS_INTELLIGENCE_READY or business_company_df is None or business_company_df.empty:
        return None

    companies = [
        str(x).strip()
        for x in business_company_df["Company"].dropna().astype(str).tolist()
        if str(x).strip()
    ]

    if requested_company:
        requested_n = _normalize_company_name(requested_company)
        exact = [c for c in companies if _normalize_company_name(c) == requested_n]
        if len(exact) == 1:
            return exact[0]

    message_n = _normalize_company_name(message)
    if not message_n:
        return None

    # Prefer exact full company-name mentions and then sufficiently specific
    # containment matches. Very short/placeholder company values are ignored.
    matches = []
    padded = f" {message_n} "
    for company in companies:
        cn = _normalize_company_name(company)
        if len(cn) < 4 or cn in {".", "bireysel"}:
            continue
        if f" {cn} " in padded:
            matches.append((len(cn), company))

    if matches:
        matches.sort(reverse=True)
        return matches[0][1]

    return None


def _business_stock_intent(message):
    raw = str(message or "").strip()
    if not raw:
        return False

    low = raw.casefold()

    stock_cues = [
        r"\bstoklarım\b", r"\bstoklarim\b",
        r"\bstokum\b", r"\bstokuma\b", r"\bstokumdaki\b",
        r"\benvanterim\b", r"\benvanter\b",
        r"\bgalerimdeki\b", r"\bgalerimin\b",
        r"\bmy\s+stock\b", r"\bmy\s+inventory\b",
        r"\bour\s+stock\b", r"\bour\s+inventory\b",
        r"\bmy\s+dealership\b", r"\bour\s+dealership\b",
        r"\bмой\s+склад\b", r"\bмои\s+машины\b",
        r"\bмой\s+автосалон\b",
    ]
    health_cues = [
        r"\bnasıl\b", r"\bnasil\b", r"\bdurum\b",
        r"\bkontrol\b", r"\bbak\b", r"\bincele\b",
        r"\bsorun\b", r"\bdikkat\b",
        r"\bhealth\b", r"\bdoing\b", r"\bperform",
        r"\bproblem\b", r"\battention\b", r"\breview\b",
        r"\bкак\b", r"\bсостояни", r"\bпроверь\b",
    ]

    has_stock = any(re.search(p, low, flags=re.IGNORECASE) for p in stock_cues)
    has_health = any(re.search(p, low, flags=re.IGNORECASE) for p in health_cues)

    # "Which of my stock..." and similar questions are also clearly stock-health.
    specific_stock_question = bool(re.search(
        r"\b(?:hangi|which|what)\b.{0,30}\b(?:stok|stock|inventory|araç|arac|vehicle|car)",
        low,
        flags=re.IGNORECASE,
    ))

    return bool(has_stock and (has_health or specific_stock_question))


def _business_company_snapshot(company):
    if not company:
        return None

    rows = business_company_df[
        business_company_df["Company"].fillna("").astype(str).str.casefold()
        == str(company).casefold()
    ]
    if rows.empty:
        return None

    row = rows.iloc[0].to_dict()

    stock = business_stock_df[
        business_stock_df["Company"].fillna("").astype(str).str.casefold()
        == str(company).casefold()
    ].copy()

    if stock.empty:
        priority = []
    else:
        attention_rank = {"HIGH": 0, "ATTENTION": 1, "WATCH": 2, "NONE": 3}
        price_rank = {"HIGH": 0, "HIGH_MID": 1, "MID_MARKET": 2, "LOW_MID": 3, "LOW": 4}

        stock["_attention_rank"] = stock["AttentionLevel"].map(attention_rank).fillna(9)
        stock["_price_rank"] = stock["PricePositionBand"].map(price_rank).fillna(9)
        stock["_age"] = pd.to_numeric(stock["StockAgeDays"], errors="coerce")

        priority_df = stock.sort_values(
            ["_attention_rank", "_age", "_price_rank"],
            ascending=[True, False, True],
            na_position="last",
        ).head(6)

        priority = []
        for item in priority_df.to_dict("records"):
            priority.append({
                "link": str(item.get("Link") or "").strip(),
                "brand": str(item.get("Brand") or "").strip(),
                "model": str(item.get("Model") or "").strip(),
                "category": str(item.get("CategoryDetail") or "").strip(),
                "year": int(item["Year"]) if pd.notna(item.get("Year")) else None,
                "km": float(item["KM"]) if pd.notna(item.get("KM")) else None,
                "asking_price": float(item["CurrentAskingPrice"]) if pd.notna(item.get("CurrentAskingPrice")) else None,
                "listing_age_days": int(item["StockAgeDays"]) if pd.notna(item.get("StockAgeDays")) else None,
                "listing_age_is_lower_bound": bool(item.get("PublicListingAgeIsLowerBound", False)),
                "price_position": str(item.get("PricePositionBand") or "").strip(),
                "attention_level": str(item.get("AttentionLevel") or "").strip(),
                "attention_reasons": [
                    x for x in str(item.get("AttentionReasons") or "").split("|") if x
                ],
                "comparable_count": int(item["ComparableListings"]) if pd.notna(item.get("ComparableListings")) else None,
                "comparable_confidence": str(item.get("ComparableEvidenceConfidence") or "").strip(),
                "comparable_median_price": float(item["ComparableMedianPrice"]) if pd.notna(item.get("ComparableMedianPrice")) else None,
                "historical_median_days_to_exit": float(item["HistoricalMedianObservedDaysToExit"]) if pd.notna(item.get("HistoricalMedianObservedDaysToExit")) else None,
            })

    return {
        "company": company,
        "summary": {
            "current_stock_count": int(row["CurrentStockCount"]) if pd.notna(row.get("CurrentStockCount")) else 0,
            "stock_asking_value": float(row["CurrentStockAskingValue"]) if pd.notna(row.get("CurrentStockAskingValue")) else None,
            "median_asking_price": float(row["MedianCurrentAskingPrice"]) if pd.notna(row.get("MedianCurrentAskingPrice")) else None,
            "median_listing_age_days": float(row["MedianPublicListingAgeDays"]) if pd.notna(row.get("MedianPublicListingAgeDays")) else None,
            "fresh_count": int(row["FreshStockCount"]) if pd.notna(row.get("FreshStockCount")) else 0,
            "normal_count": int(row["NormalStockCount"]) if pd.notna(row.get("NormalStockCount")) else 0,
            "above_typical_count": int(row["AboveTypicalAgeCount"]) if pd.notna(row.get("AboveTypicalAgeCount")) else 0,
            "aged_count": int(row["AgedStockCount"]) if pd.notna(row.get("AgedStockCount")) else 0,
            "very_aged_count": int(row["VeryAgedStockCount"]) if pd.notna(row.get("VeryAgedStockCount")) else 0,
            "high_price_position_count": int(row["HighPricePositionCount"]) if pd.notna(row.get("HighPricePositionCount")) else 0,
            "watch_count": int(row["WatchStockCount"]) if pd.notna(row.get("WatchStockCount")) else 0,
            "attention_count": int(row["AttentionStockCount"]) if pd.notna(row.get("AttentionStockCount")) else 0,
            "high_attention_count": int(row["HighAttentionStockCount"]) if pd.notna(row.get("HighAttentionStockCount")) else 0,
            "historical_distinct_listings": int(row["HistoricalDistinctListings"]) if pd.notna(row.get("HistoricalDistinctListings")) else None,
            "historical_observed_exits": int(row["HistoricalObservedMarketExits"]) if pd.notna(row.get("HistoricalObservedMarketExits")) else None,
            "historical_median_observed_days_to_exit": float(row["HistoricalMedianObservedDaysToExit"]) if pd.notna(row.get("HistoricalMedianObservedDaysToExit")) else None,
            "historical_price_reduction_rate": float(row["HistoricalPriceReductionRate"]) if pd.notna(row.get("HistoricalPriceReductionRate")) else None,
        },
        "priority_vehicles": priority,
    }


def _business_stock_answer(message, language, requested_company=None):
    company = _resolve_business_company(message, requested_company=requested_company)

    if not company:
        text = {
            "TR": "Stok analizini yapabilmem için hangi galeriye ait olduğunuzu bilmem gerekiyor. Şimdilik mesajınızda galeri adını yazabilirsiniz; hesap sistemi geldiğinde bu otomatik olacak.",
            "EN": "I need to know which dealership is yours before I can analyse your stock. For now, include the dealership name in your message; once account linking is added, this will be automatic.",
            "RU": "Чтобы проанализировать ваш склад, мне нужно знать название автосалона. Пока укажите его в сообщении; после подключения аккаунтов это будет определяться автоматически.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_COMPANY_REQUIRED",
            "company": None,
            "priority_vehicles": [],
        }

    snapshot = _business_company_snapshot(company)
    if not snapshot:
        text = {
            "TR": f"{company} için Business stok verisi bulamadım.",
            "EN": f"I couldn't find Business stock data for {company}.",
            "RU": f"Я не нашёл Business-данные по складу для {company}.",
        }
        return text.get(language, text["TR"]), {
            "success": False,
            "error": "BUSINESS_COMPANY_NOT_FOUND",
            "company": company,
            "priority_vehicles": [],
        }

    s = snapshot["summary"]
    vehicles = snapshot["priority_vehicles"]

    total = max(int(s.get("current_stock_count") or 0), 1)
    aged_total = (
        int(s.get("above_typical_count") or 0)
        + int(s.get("aged_count") or 0)
        + int(s.get("very_aged_count") or 0)
    )
    aged_pct = round(aged_total / total * 100)
    high_attention_pct = round(int(s.get("high_attention_count") or 0) / total * 100)

    lines = []
    for idx, vehicle in enumerate(vehicles, 1):
        name = " ".join(
            x for x in [
                str(vehicle.get("year") or "").strip(),
                vehicle.get("brand") or "",
                vehicle.get("model") or "",
                vehicle.get("category") or "",
            ] if x
        ).strip()

        asking = _business_money(vehicle.get("asking_price"))
        age = vehicle.get("listing_age_days")
        price_pos = vehicle.get("price_position") or "—"
        level = vehicle.get("attention_level") or "—"
        comp_median = _business_money(vehicle.get("comparable_median_price"))

        if language == "EN":
            age_text = (
                f"observed for at least {age} days"
                if vehicle.get("listing_age_is_lower_bound")
                else f"observed advertised for {age} days"
            ) if age is not None else "listing age unavailable"
            details = [
                asking,
                age_text,
                f"price position: {price_pos.replace('_', ' ').lower()}",
            ]
            if comp_median:
                details.append(f"comparable median {comp_median}")
            lines.append(f"{idx}. {name} — {level}\n   " + " · ".join(x for x in details if x))

        elif language == "RU":
            age_text = (
                f"наблюдается как минимум {age} дн."
                if vehicle.get("listing_age_is_lower_bound")
                else f"наблюдается в объявлениях {age} дн."
            ) if age is not None else "возраст объявления неизвестен"
            details = [
                asking,
                age_text,
                f"ценовая позиция: {price_pos}",
            ]
            if comp_median:
                details.append(f"медиана сопоставимых объявлений {comp_median}")
            lines.append(f"{idx}. {name} — {level}\n   " + " · ".join(x for x in details if x))

        else:
            age_text = (
                f"en az {age} gündür gözlemleniyor"
                if vehicle.get("listing_age_is_lower_bound")
                else f"{age} gündür ilanda gözlemleniyor"
            ) if age is not None else "ilan yaşı bilinmiyor"
            price_pos_tr = {
                "HIGH": "yüksek",
                "HIGH_MID": "yüksek-orta",
                "MID_MARKET": "piyasa ortası",
                "LOW_MID": "düşük-orta",
                "LOW": "düşük",
                "INSUFFICIENT_EVIDENCE": "yetersiz veri",
            }.get(price_pos, price_pos)
            level_tr = {
                "HIGH": "yüksek dikkat",
                "ATTENTION": "dikkat",
                "WATCH": "izle",
                "NONE": "normal",
            }.get(level, level)
            details = [
                asking,
                age_text,
                f"fiyat konumu: {price_pos_tr}",
            ]
            if comp_median:
                details.append(f"benzer ilan medyanı {comp_median}")
            lines.append(f"{idx}. {name} — {level_tr}\n   " + " · ".join(x for x in details if x))

    stock_value = _business_money(s.get("stock_asking_value"))
    median_price = _business_money(s.get("median_asking_price"))

    if language == "EN":
        intro = (
            f"{company} currently has {s['current_stock_count']} advertised vehicles"
            + (f" with a combined asking value of about {stock_value}" if stock_value else "")
            + ". "
            f"{aged_pct}% of stock is above its typical observed market-age range, "
            f"and {s['high_attention_count']} vehicles ({high_attention_pct}%) are in the highest-attention group."
        )
        overview = (
            f"Median advertised age is {s['median_listing_age_days']:.0f} days"
            + (f" and median asking price is {median_price}" if median_price else "")
            + ". The first vehicles I would review are:"
        )
        note = (
            "Listing age means days since OtoDeğer first observed the vehicle advertised; it is not the dealership's acquisition age. "
            "Observed market exit is historical listing behaviour, not a confirmed sale."
        )

    elif language == "RU":
        intro = (
            f"У {company} сейчас {s['current_stock_count']} активных объявлений"
            + (f" с общей заявленной стоимостью около {stock_value}" if stock_value else "")
            + ". "
            f"{aged_pct}% склада находится выше типичного наблюдаемого рыночного возраста, "
            f"а {s['high_attention_count']} автомобилей ({high_attention_pct}%) относятся к группе повышенного внимания."
        )
        overview = (
            f"Медианный наблюдаемый возраст объявления — {s['median_listing_age_days']:.0f} дн."
            + (f", медианная цена — {median_price}" if median_price else "")
            + ". В первую очередь я бы проверил:"
        )
        note = (
            "Возраст объявления — это дни с момента, когда OtoDeğer впервые увидел автомобиль в рекламе, а не возраст нахождения автомобиля у дилера. "
            "Исторический выход с рынка не означает подтверждённую продажу."
        )

    else:
        intro = (
            f"{company} için şu anda {s['current_stock_count']} aktif ilan görüyorum"
            + (f"; toplam ilan değeri yaklaşık {stock_value}" if stock_value else "")
            + ". "
            f"Stokun %{aged_pct}'i kendi piyasa davranışına göre tipik yaş aralığının üzerinde, "
            f"{s['high_attention_count']} araç (%{high_attention_pct}) ise en yüksek dikkat grubunda."
        )
        overview = (
            f"Medyan gözlenen ilan yaşı {s['median_listing_age_days']:.0f} gün"
            + (f", medyan ilan fiyatı {median_price}" if median_price else "")
            + ". İlk olarak şu araçları gözden geçirirdim:"
        )
        note = (
            "Buradaki ilan yaşı, OtoDeğer'in aracı ilk kez ilanda gördüğü tarihten itibaren geçen süredir; galerinin aracı satın aldığı tarih değildir. "
            "Tarihsel piyasa çıkışı da doğrulanmış satış anlamına gelmez."
        )

    answer = intro + "\n\n" + overview
    if lines:
        answer += "\n\n" + "\n\n".join(lines)
    answer += "\n\n" + note

    return answer, {
        "success": True,
        "company": company,
        "summary": s,
        "priority_vehicles": vehicles,
    }


# =========================================================
# BUSINESS ASSISTANT v1 - ACQUISITION / STOCKING OPPORTUNITIES
# =========================================================

def _business_acquire_intent(message, conversation_history=None):
    """
    Conservative first Business router.

    BUSINESS_ACQUIRE activates only when the message clearly combines:
      - dealer/gallery/stock/inventory commercial context, and
      - an acquisition/stocking decision.

    This intentionally does NOT intercept ordinary buyer questions such as
    "What should I buy for £15k?".
    """
    raw = str(message or "").strip()
    if not raw:
        return False

    low = raw.casefold()

    commercial_cues = [
        r"\bgaleri(?:m|me|mi|ler|lerim)?\b",
        r"\bstok(?:um|a|ta|lamak|layayım|layalim|layalım)?\b",
        r"\benvanter(?:im|e|de)?\b",
        r"\bdealer(?:ship)?\b",
        r"\binventory\b",
        r"\bstock\b",
        r"\bforecourt\b",
        r"\bresale\s+stock\b",
        r"\bдилер\b",
        r"\bавтосалон\b",
        r"\bсклад\b",
    ]
    acquire_cues = [
        r"\bne\s+al(?:ayım|alim|malıyım|maliyim)\b",
        r"\bhangi\s+araç(?:ları|lari)?\s+(?:al|stokla)",
        r"\bstok(?:a|uma)?\s+(?:ne|hangi)",
        r"\bstokla(?:mak|yacağım|yacagim|malıyım|maliyim)",
        r"\bwhat\s+should\s+i\s+(?:buy|stock)\b",
        r"\bwhat\s+(?:cars?|vehicles?)\s+should\s+i\s+(?:buy|stock)\b",
        r"\bwhat\s+should\s+we\s+(?:buy|stock)\b",
        r"\bwhich\s+(?:cars?|vehicles?|models?)\s+should\s+(?:i|we)\s+(?:buy|stock)\b",
        r"\bstocking\s+opportunit",
        r"\bbuy\s+for\s+(?:my|our)\s+(?:stock|inventory|dealership)\b",
        r"\bчто\s+(?:купить|закупить)\b",
        r"\bкакие\s+машины\s+(?:купить|закупить)\b",
    ]

    has_commercial = any(re.search(p, low, flags=re.IGNORECASE) for p in commercial_cues)
    has_acquire = any(re.search(p, low, flags=re.IGNORECASE) for p in acquire_cues)

    return bool(has_commercial and has_acquire)


def _business_parse_budget(message):
    """Parse dealer budgets with the same locale-safe number logic as Personal."""
    raw = str(message or "")
    patterns = [
        r"£\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)",
        r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*£",
        r"\b([0-9]+(?:[.,][0-9]+)?\s*[kK])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        value = _parse_human_number(match.group(1))
        if value is not None and 500 <= value <= 500000:
            return float(value)
    return None


def _business_vehicle_type_filter(message):
    low = str(message or "").casefold()

    if re.search(r"\b(?:suv|crossover|4x4)\b", low):
        return "SUV"
    if re.search(r"\b(?:pickup|pick-up|pick up)\b", low):
        return "PICKUP"
    if re.search(r"\b(?:otomobil|car|cars|araba|arabalar|автомобил|машин)\b", low):
        return "CAR"

    return None


def _business_market_type_mask(frame, requested_type=None):
    vt = frame["VehicleType"].fillna("").astype(str).str.casefold()

    # Business V1 focuses on normal dealership vehicle stock. Other vehicle
    # classes remain in the source data but are not proactively recommended.
    normal_market = (
        vt.str.contains("otomobil", regex=False)
        | vt.str.contains("suv", regex=False)
        | vt.str.contains("pick", regex=False)
        | vt.str.contains("arazi", regex=False)
    )

    if requested_type == "SUV":
        return normal_market & (
            vt.str.contains("suv", regex=False)
            | vt.str.contains("arazi", regex=False)
        )
    if requested_type == "PICKUP":
        return normal_market & vt.str.contains("pick", regex=False)
    if requested_type == "CAR":
        return normal_market & vt.str.contains("otomobil", regex=False)

    return normal_market


def _business_acquisition_candidates(message, limit=8):
    if not BUSINESS_INTELLIGENCE_READY or business_market_df is None or business_market_df.empty:
        return {
            "success": False,
            "error": "BUSINESS_INTELLIGENCE_NOT_READY",
            "budget": _business_parse_budget(message),
            "results": [],
        }

    work = business_market_df.copy()
    requested_type = _business_vehicle_type_filter(message)
    budget = _business_parse_budget(message)

    work = work[_business_market_type_mask(work, requested_type)].copy()

    # Prefer the more specific category/variant-year rows. Model-year rows remain
    # available as fallback when the precise layer cannot produce enough options.
    specific = work[work["BusinessGranularity"].str.upper() == "CATEGORY_YEAR"].copy()
    fallback = work[work["BusinessGranularity"].str.upper() == "MODEL_YEAR"].copy()

    def eligible(frame):
        x = frame.copy()

        x = x[
            x["AcquisitionSignal"].isin(
                ["VERY_STRONG", "STRONG", "MODERATE", "WEAK", "CAUTION"]
            )
        ].copy()

        signal_rank = {
            "VERY_STRONG": 0,
            "STRONG": 1,
            "MODERATE": 2,
            "WEAK": 3,
            "CAUTION": 4,
        }
        evidence_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

        x["_signal_rank"] = x["AcquisitionSignal"].map(signal_rank).fillna(9)
        x["_evidence_rank"] = x["EvidenceQuality"].map(evidence_rank).fillna(9)
        x["_confidence_index"] = pd.to_numeric(
            x.get("ConfidenceAdjustedOpportunityIndex"),
            errors="coerce",
        )
        x["_opp_pct"] = pd.to_numeric(
            x.get("OpportunityPercentile"),
            errors="coerce",
        )

        return x.sort_values(
            ["_signal_rank", "_evidence_rank", "_confidence_index", "_opp_pct"],
            ascending=[True, True, False, False],
            na_position="last",
        )

    ordered = pd.concat(
        [eligible(specific), eligible(fallback)],
        ignore_index=True,
    )

    if ordered.empty:
        return {
            "success": True,
            "budget": budget,
            "requested_type": requested_type,
            "results": [],
        }

    # Avoid returning a page dominated by many variants of one model family.
    selected = []
    model_counts = {}
    seen_variant_keys = set()

    for row in ordered.to_dict("records"):
        brand = str(row.get("Brand") or "").strip()
        model = str(row.get("Model") or "").strip()
        category = str(row.get("CategoryDetail") or "").strip()
        year = row.get("Year")

        family_key = (brand.casefold(), model.casefold())
        variant_key = (
            brand.casefold(),
            model.casefold(),
            category.casefold(),
            int(year) if pd.notna(year) else None,
        )

        if variant_key in seen_variant_keys:
            continue
        if model_counts.get(family_key, 0) >= 1:
            continue

        selected.append(row)
        seen_variant_keys.add(variant_key)
        model_counts[family_key] = model_counts.get(family_key, 0) + 1

        if len(selected) >= int(limit):
            break

    public = []
    for row in selected:
        public.append({
            "vehicle_type": str(row.get("VehicleType") or "").strip(),
            "brand": str(row.get("Brand") or "").strip(),
            "model": str(row.get("Model") or "").strip(),
            "category": str(row.get("CategoryDetail") or "").strip(),
            "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
            "current_listings": (
                int(row["CurrentListings"])
                if pd.notna(row.get("CurrentListings")) else None
            ),
            "starting_price": (
                float(row["CurrentStartingPrice"])
                if pd.notna(row.get("CurrentStartingPrice")) else None
            ),
            "median_price": (
                float(row["CurrentMedianPrice"])
                if pd.notna(row.get("CurrentMedianPrice")) else None
            ),
            "historical_median_observed_days_to_exit": (
                float(row["MedianObservedDaysToExit"])
                if pd.notna(row.get("MedianObservedDaysToExit")) else None
            ),
            "observed_exit_within_60_days_rate": (
                float(row["ObservedExitWithin60DaysRate"])
                if pd.notna(row.get("ObservedExitWithin60DaysRate")) else None
            ),
            "historical_price_reduction_rate": (
                float(row["PriceReductionRate"])
                if pd.notna(row.get("PriceReductionRate")) else None
            ),
            "evidence_quality": str(row.get("EvidenceQuality") or "").strip(),
            "liquidity_benchmark_source": str(
                row.get("HistoricalBenchmarkSourceLiquidity") or ""
            ).strip(),
            "price_pressure_benchmark_source": str(
                row.get("HistoricalBenchmarkSourcePricePressure") or ""
            ).strip(),
            "opportunity_percentile": (
                float(row["OpportunityPercentile"])
                if pd.notna(row.get("OpportunityPercentile")) else None
            ),
            "acquisition_signal": str(row.get("AcquisitionSignal") or "").strip(),
            "acquisition_reasons": [
                x for x in str(row.get("AcquisitionReasons") or "").split("|") if x
            ],
        })

    return {
        "success": True,
        "budget": budget,
        "requested_type": requested_type,
        "results": public,
    }


def _business_pct(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value) * 100.0)


def _business_money(value):
    if value is None or pd.isna(value):
        return None
    return f"£{float(value):,.0f}"


def _business_acquisition_answer(message, language):
    result = _business_acquisition_candidates(message, limit=8)

    if not result.get("success"):
        fallback = {
            "TR": "Business piyasa verisi şu anda hazır değil. Lütfen biraz sonra tekrar deneyin.",
            "EN": "The Business market data is not ready right now. Please try again shortly.",
            "RU": "Данные Business по рынку сейчас недоступны. Попробуйте ещё раз чуть позже.",
        }
        return fallback.get(language, fallback["TR"]), result

    candidates = result.get("results") or []
    budget = result.get("budget")

    if not candidates:
        fallback = {
            "TR": "Bu araç tipi için yeterli kanıta sahip bir stok fırsatı bulamadım.",
            "EN": "I couldn't find a stocking opportunity with sufficient evidence for that vehicle type.",
            "RU": "Я не нашёл достаточно подтверждённых вариантов для закупки по этому типу автомобиля.",
        }
        return fallback.get(language, fallback["TR"]), result

    lines = []

    for idx, item in enumerate(candidates, 1):
        name_bits = [
            str(item.get("year") or "").strip(),
            item.get("brand") or "",
            item.get("model") or "",
        ]
        if item.get("category"):
            name_bits.append(item["category"])
        vehicle_name = " ".join(x for x in name_bits if x).strip()

        starting = _business_money(item.get("starting_price"))
        median = _business_money(item.get("median_price"))
        exit60 = _business_pct(item.get("observed_exit_within_60_days_rate"))
        reduction = _business_pct(item.get("historical_price_reduction_rate"))

        evidence = item.get("evidence_quality") or "—"
        signal = item.get("acquisition_signal") or "—"

        if language == "EN":
            evidence_bits = []
            if exit60 is not None:
                evidence_bits.append(f"{exit60}% observed 60-day exit rate")
            if reduction is not None:
                evidence_bits.append(f"{reduction}% historical asking-price reduction rate")
            if item.get("current_listings") is not None:
                evidence_bits.append(f"{item['current_listings']} current listings")
            market_part = (
                f"Current asking range starts around {starting}"
                + (f", median {median}" if median else "")
                + "."
                if starting else ""
            )
            lines.append(
                f"{idx}. {vehicle_name} — {signal.replace('_', ' ').title()} "
                f"({evidence} evidence)\n"
                f"   {market_part} "
                + (", ".join(evidence_bits) + "." if evidence_bits else "")
            )
        elif language == "RU":
            evidence_bits = []
            if exit60 is not None:
                evidence_bits.append(f"наблюдаемый выход с рынка за 60 дней: {exit60}%")
            if reduction is not None:
                evidence_bits.append(f"историческая доля снижения цены: {reduction}%")
            if item.get("current_listings") is not None:
                evidence_bits.append(f"активных объявлений: {item['current_listings']}")
            market_part = (
                f"Текущие цены начинаются примерно от {starting}"
                + (f", медиана {median}" if median else "")
                + "."
                if starting else ""
            )
            lines.append(
                f"{idx}. {vehicle_name} — {signal.replace('_', ' ')} "
                f"(достоверность: {evidence})\n"
                f"   {market_part} "
                + (", ".join(evidence_bits) + "." if evidence_bits else "")
            )
        else:
            evidence_bits = []
            if exit60 is not None:
                evidence_bits.append(f"60 günde gözlenen piyasa çıkış oranı %{exit60}")
            if reduction is not None:
                evidence_bits.append(f"tarihsel fiyat indirimi oranı %{reduction}")
            if item.get("current_listings") is not None:
                evidence_bits.append(f"güncel {item['current_listings']} ilan")
            market_part = (
                f"Güncel ilanlar yaklaşık {starting}'dan başlıyor"
                + (f", medyan {median}" if median else "")
                + "."
                if starting else ""
            )
            signal_tr = {
                "VERY_STRONG": "Çok güçlü",
                "STRONG": "Güçlü",
                "MODERATE": "Orta",
                "WEAK": "Zayıf",
                "CAUTION": "Dikkat",
            }.get(signal, signal)
            evidence_tr = {
                "HIGH": "yüksek",
                "MEDIUM": "orta",
                "LOW": "düşük",
            }.get(evidence, evidence.casefold())
            lines.append(
                f"{idx}. {vehicle_name} — {signal_tr} "
                f"({evidence_tr} kanıt)\n"
                f"   {market_part} "
                + (", ".join(evidence_bits) + "." if evidence_bits else "")
            )

    if language == "EN":
        intro = "Using current advertised-market evidence, these are the strongest evidence-backed stocking opportunities I can identify:"
        note = (
            "This is a market-opportunity ranking, not a wholesale purchase-price or profit prediction. "
            "Actual dealer acquisition cost is unknown, so advertised asking prices are used only as resale-market and competition context and are not compared with a stock-purchase budget."
        )
    elif language == "RU":
        intro = "По текущим рыночным объявлениям наиболее сильные подтверждённые варианты для закупки выглядят так:"
        note = (
            "Это рейтинг рыночной привлекательности, а не прогноз закупочной цены или прибыли. "
            "Фактическая закупочная стоимость неизвестна, поэтому цены объявлений используются только как контекст розничного рынка и конкуренции и не сравниваются с бюджетом на закупку."
        )
    else:
        intro = "Güncel ilan piyasası verilerine göre, verinin en güçlü desteklediği stok seçenekleri şunlar:"
        note = (
            "Bu bir piyasa fırsatı sıralamasıdır; alış maliyeti veya kâr tahmini değildir. "
            "Gerçek galeri alış maliyeti bilinmediği için ilan fiyatları yalnızca perakende piyasa ve rekabet bağlamı olarak kullanılır; stok alım bütçesiyle karşılaştırılmaz."
        )

    return intro + "\n\n" + "\n\n".join(lines) + "\n\n" + note, result


@app.route("/api/assistant_legacy", methods=["POST"])
def api_ai_buying_assistant_legacy():
    request_started = time.perf_counter()
    try:
        data = request.json or {}

        message = str(data.get("message", "")).strip()
        conversation_history = sanitize_conversation_history(
            data.get("conversation_history") or []
        )
        requested_language = str(data.get("language", "TR")).upper()
        language = detect_conversation_language(
            message,
            requested_language=requested_language,
            conversation_history=conversation_history,
        )
        current_filters = sanitize_ai_filters(
            data.get("current_filters") or {}
        )
        current_preferences = data.get("current_preferences") or []

        if not message:
            return jsonify({
                "success": False,
                "error": "MESSAGE_REQUIRED"
            }), 400

        if len(message) > ASSISTANT_MAX_MESSAGE_CHARS:
            return jsonify({
                "success": False,
                "error": "MESSAGE_TOO_LONG",
                "max_chars": ASSISTANT_MAX_MESSAGE_CHARS,
            }), 413

        allowed, retry_after = _assistant_request_allowed()
        if not allowed:
            response = jsonify({
                "success": False,
                "error": "RATE_LIMITED",
                "retry_after_seconds": retry_after,
            })
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        # Unified account / Business context.
        # The frontend may already send these fields now; later authentication
        # and payments can populate them automatically.
        access_tier = _normalize_access_tier(
            data.get("access_tier") or data.get("tier")
        )
        requested_business_company = str(
            data.get("business_company") or ""
        ).strip() or None

        resolved_business_company = _resolve_business_company_context(
            message=message,
            conversation_history=conversation_history,
            requested_company=requested_business_company,
        )

        business_capabilities = _business_capabilities_payload(
            access_tier=access_tier,
            company=resolved_business_company,
        )

        # Resolve narrow conversational short answers before fallback detection.
        # In particular, a bare "15.000" immediately after we asked for a maximum
        # budget should be understood as the answer to that question.
        message = _expand_contextual_short_answer(message, conversation_history)

        # Obvious unusable / keyboard-smash input should never be treated as a
        # broad shopping request and sent into guided narrowing.
        if _looks_like_gibberish_message(message):
            fallback = _fallback_support_payload(language)
            return jsonify({
                "success": True,
                "answer": _unsupported_input_answer(language),
                "filters": current_filters,
                "preferences": current_preferences,
                "count": 0,
                "returned": 0,
                "results": [],
                "model_options": [],
                "suggestions": fallback["suggestions"],
                "actions": fallback["actions"],
                "decision_mode": "FALLBACK",
                "stage": "unsupported_input",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "access_tier": access_tier,
                "business_capabilities": business_capabilities,
            })

        # Protect proprietary data from bulk reconstruction attempts before
        # any search or model call is made.
        if _looks_like_dataset_extraction_request(message):
            fallback = _fallback_support_payload(language)
            return jsonify({
                "success": True,
                "answer": _data_protection_answer(language),
                "filters": current_filters,
                "preferences": current_preferences,
                "count": 0,
                "returned": 0,
                "results": [],
                "model_options": [],
                "suggestions": fallback["suggestions"],
                "actions": fallback["actions"],
                "decision_mode": "PROTECTED_DATA",
                "stage": "protected_data",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "access_tier": access_tier,
                "business_capabilities": business_capabilities,
            })

        # Own-car valuation is a first-class OtoDeğer flow, not a generative
        # approximation inside chat.
        if _valuation_intent(message):
            valuation = _valuation_response(language)
            return jsonify({
                "success": True,
                "answer": valuation["answer"],
                "filters": current_filters,
                "preferences": current_preferences,
                "count": 0,
                "returned": 0,
                "results": [],
                "model_options": [],
                "actions": valuation["actions"],
                "decision_mode": "VALUATION",
                "stage": "valuation_handoff",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "access_tier": access_tier,
                "business_capabilities": business_capabilities,
            })

        # Business V5: dealer-market understanding.
        # Broad commercial market questions are handled here before stock-specific
        # actions, while ordinary Personal shopping questions remain untouched.
        if _business_market_intent(message):
            business_started = time.perf_counter()
            business_answer, business_result = _business_market_answer(
                message=message,
                language=language,
            )
            business_seconds = time.perf_counter() - business_started
            total_seconds = time.perf_counter() - request_started

            print(
                f"ASSISTANT_TIMING mode=BUSINESS_MARKET "
                f"business={business_seconds:.2f}s total={total_seconds:.2f}s",
                flush=True,
            )

            return jsonify({
                "success": True,
                "answer": business_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": len(business_result.get("rows") or []),
                "returned": 0,
                "results": [],
                "model_options": [],
                "business_options": business_result.get("rows") or [],
                "business_market_mode": business_result.get("mode"),
                "business_company": resolved_business_company,
                "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
                "business_capabilities": business_capabilities,
                "access_tier": access_tier,
                "decision_mode": "BUSINESS_MARKET",
                "stage": "business_market_understanding",
            })

        # Business V4: aging-stock / inventory-action questions.
        # This runs before pricing because a question such as
        # "This has been sitting 73 days — should I reduce it?" is primarily
        # an inventory-management decision that uses pricing as one input.
        if _business_manage_intent(message):
            business_started = time.perf_counter()
            business_answer, business_result = _business_manage_answer(
                message=message,
                language=language,
                requested_company=resolved_business_company,
            )
            business_seconds = time.perf_counter() - business_started
            total_seconds = time.perf_counter() - request_started

            print(
                f"ASSISTANT_TIMING mode=BUSINESS_MANAGE "
                f"business={business_seconds:.2f}s total={total_seconds:.2f}s",
                flush=True,
            )

            return jsonify({
                "success": True,
                "answer": business_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": len(business_result.get("vehicles") or []),
                "returned": 0,
                "results": [],
                "model_options": [],
                "business_options": business_result.get("vehicles") or [],
                "business_company": business_result.get("company"),
                "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
                "business_capabilities": business_capabilities,
                "access_tier": access_tier,
                "decision_mode": "BUSINESS_MANAGE",
                "stage": "business_inventory_action",
            })

        # Business V3: competitive pricing / price-position questions.
        # This runs before generic stock-health routing so a request such as
        # "Which cars in my stock are overpriced?" is treated as pricing.
        if _business_price_intent(message):
            business_started = time.perf_counter()
            business_answer, business_result = _business_price_answer(
                message=message,
                language=language,
                requested_company=resolved_business_company,
            )
            business_seconds = time.perf_counter() - business_started
            total_seconds = time.perf_counter() - request_started

            print(
                f"ASSISTANT_TIMING mode=BUSINESS_PRICE "
                f"business={business_seconds:.2f}s total={total_seconds:.2f}s",
                flush=True,
            )

            return jsonify({
                "success": True,
                "answer": business_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": len(business_result.get("vehicles") or []),
                "returned": 0,
                "results": [],
                "model_options": [],
                "business_options": business_result.get("vehicles") or [],
                "business_company": business_result.get("company"),
                "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
                "business_capabilities": business_capabilities,
                "access_tier": access_tier,
                "decision_mode": "BUSINESS_PRICE",
                "stage": "business_pricing",
            })

        # Business V2: own-stock / inventory-health questions.
        if _business_stock_intent(message):
            business_started = time.perf_counter()
            business_answer, business_result = _business_stock_answer(
                message=message,
                language=language,
                requested_company=resolved_business_company,
            )
            business_seconds = time.perf_counter() - business_started
            total_seconds = time.perf_counter() - request_started

            print(
                f"ASSISTANT_TIMING mode=BUSINESS_STOCK "
                f"business={business_seconds:.2f}s total={total_seconds:.2f}s",
                flush=True,
            )

            return jsonify({
                "success": True,
                "answer": business_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": len(business_result.get("priority_vehicles") or []),
                "returned": 0,
                "results": [],
                "model_options": [],
                "business_options": business_result.get("priority_vehicles") or [],
                "business_company": business_result.get("company"),
                "business_summary": business_result.get("summary"),
                "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
                "business_capabilities": business_capabilities,
                "access_tier": access_tier,
                "decision_mode": "BUSINESS_STOCK",
                "stage": "business_stock_health",
            })

        # Business V1 acquisition/stocking questions.
        # This branch is deliberately conservative and only activates when the
        # user explicitly frames the request as a gallery/dealer/stock decision.
        # Ordinary Personal DISCOVER / COMPARE / SHOP requests continue through
        # the existing buyer pipeline unchanged below.
        if _business_acquire_intent(message, conversation_history):
            business_started = time.perf_counter()
            business_answer, business_result = _business_acquisition_answer(
                message=message,
                language=language,
            )
            business_seconds = time.perf_counter() - business_started
            total_seconds = time.perf_counter() - request_started

            print(
                f"ASSISTANT_TIMING mode=BUSINESS_ACQUIRE "
                f"business={business_seconds:.2f}s total={total_seconds:.2f}s",
                flush=True,
            )

            return jsonify({
                "success": True,
                "answer": business_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": len(business_result.get("results") or []),
                "returned": 0,
                "results": [],
                "model_options": [],
                "business_options": business_result.get("results") or [],
                "business_budget": business_result.get("budget"),
                "business_vehicle_type": business_result.get("requested_type"),
                "business_intelligence_version": BUSINESS_INTELLIGENCE_VERSION,
                "business_capabilities": business_capabilities,
                "access_tier": access_tier,
                "decision_mode": "BUSINESS_ACQUIRE",
                "stage": "business_recommendation",
            })

        # =====================================================
        # PERSONAL ASSISTANT V8 — semantic turn orchestration
        # =====================================================
        #
        # Resolve ONLY vehicles explicitly named in the latest message first.
        # History recovery happens later and only when the semantic controller
        # confirms that this turn continues the same COMPARE/SHOP task.
        resolve_started = time.perf_counter()
        resolved_targets = resolve_market_vehicle_mentions(message)
        resolved_targets = _attach_explicit_years_to_vehicle_targets(
            message, resolved_targets
        )

        low_message = str(message or "").casefold()
        contextual_one = bool(
            conversation_history
            and re.search(
                r"\b(?:which one|the one|one you|your recommendation|your choice|"
                r"(?:the )?(?:stronger|better|strongest|best) (?:one|option))\b",
                low_message,
            )
        )
        if contextual_one and resolved_targets:
            resolved_targets = [
                target for target in resolved_targets
                if not (
                    str(target.get("brand") or "").casefold() == "mini"
                    and str(target.get("model") or "").casefold() == "one"
                )
            ]

        explicit_current_message_targets = [dict(t) for t in resolved_targets]
        resolve_seconds = time.perf_counter() - resolve_started

        interpret_started = time.perf_counter()
        try:
            interpretation = _v8_semantic_turn_plan(
                message=message,
                language=language,
                current_filters=current_filters,
                current_preferences=current_preferences,
                conversation_history=conversation_history,
                explicit_targets=explicit_current_message_targets,
            )
        except Exception as exc:
            # Graceful degradation: the deterministic parser remains a fallback,
            # not the primary conversation brain.
            print(f"V8_CONTROLLER_DEGRADED_FALLBACK: {exc}", flush=True)
            interpretation = fast_common_interpretation(
                message=message,
                resolved_targets=explicit_current_message_targets,
            )
            if interpretation is None:
                try:
                    interpretation = interpret_market_query(
                        message=message,
                        current_filters=current_filters,
                        language=language,
                        conversation_history=conversation_history,
                    )
                except Exception as legacy_exc:
                    print(
                        f"LEGACY_INTERPRETER_DEGRADED_FALLBACK: {legacy_exc}",
                        flush=True,
                    )
                    question = {
                        "TR": "Bunu doğru anlayabilmem için neyi değiştirmek istediğinizi biraz daha açık yazar mısınız?",
                        "RU": "Уточните, пожалуйста, что именно вы хотите изменить, чтобы я понял вас правильно.",
                        "EN": "Could you clarify what you'd like me to change so I can apply it correctly?",
                    }.get(language, "Could you clarify what you'd like me to change?")
                    interpretation = {
                        "operation": "CONTINUE",
                        "filters": {},
                        "clear_filters": [],
                        "seller_mode": None,
                        "preferences": [],
                        "clear_preferences": [],
                        "needs_clarification": True,
                        "clarification_question": question,
                        "decision_mode": "COMPARE" if explicit_current_message_targets else "DISCOVER",
                        "fast_path": False,
                        "degraded": True,
                        "orchestration_version": ASSISTANT_ORCHESTRATION_VERSION,
                    }

        # Explicit numeric constraints are deterministic evidence. The semantic
        # controller remains responsible for task meaning, but it cannot redefine
        # £18k as £18m or otherwise distort a number the user directly supplied.
        interpretation = _v8_apply_authoritative_numeric_constraints(
            message,
            explicit_current_message_targets,
            interpretation,
        )

        # Concrete brands typed by the user are deterministic evidence too.
        # Example: "BMW or Mercedes" must become both canonical brands even if
        # the semantic controller happens to emit only one of them.
        interpretation = _apply_authoritative_explicit_brands(
            message,
            interpretation,
        )

        interpretation.setdefault("operation", "CONTINUE")
        interpretation.setdefault(
            "orchestration_version", ASSISTANT_ORCHESTRATION_VERSION
        )
        interpret_seconds = time.perf_counter() - interpret_started

        decision_mode = str(
            interpretation.get("decision_mode") or "DISCOVER"
        ).upper()
        if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
            decision_mode = "DISCOVER"
            interpretation["decision_mode"] = decision_mode

        if len(explicit_current_message_targets) >= 2:
            decision_mode = "COMPARE"
            interpretation["decision_mode"] = "COMPARE"
            interpretation["operation"] = "NEW_TASK"

        # First resolve explicit follow-ups to the immediately preceding assistant
        # offer, e.g. "Want me to compare the X1 and GLA?" -> "Yes, compare them".
        # This is safer and more precise than searching older user turns.
        if not resolved_targets and str(decision_mode).upper() == "COMPARE":
            offered_compare_targets = _recover_latest_assistant_compare_offer(
                message,
                conversation_history,
            )
            if len(offered_compare_targets) >= 2:
                resolved_targets = offered_compare_targets
                interpretation["decision_mode"] = "COMPARE"
                interpretation["operation"] = "CONTINUE"

        # Older history recovery remains subordinate to semantic task state and
        # only runs when the immediate assistant offer did not resolve the pair.
        if not resolved_targets and _v8_should_recover_compare_targets(
            message, interpretation
        ):
            resolved_targets = _recover_recent_compare_targets(
                message,
                conversation_history,
            )

        if not resolved_targets and _v8_should_recover_shop_target(
            message, interpretation
        ):
            resolved_targets = _recover_recent_recommendation_target(
                message,
                conversation_history,
            )

        # A genuine new task starts from a clean Personal search state.
        # Only constraints/preferences explicitly supplied in the new turn survive.
        operation = str(interpretation.get("operation") or "CONTINUE").upper()
        state_base_filters = {} if operation == "NEW_TASK" else current_filters
        state_base_preferences = [] if operation == "NEW_TASK" else current_preferences

        next_filters = apply_interpretation_to_filters(
            state_base_filters,
            interpretation,
        )
        next_preferences = _v8_apply_preference_changes(
            state_base_preferences,
            interpretation,
        )

        # A fresh explicit multi-vehicle comparison is always self-contained.
        # Per-vehicle year/category is applied target-by-target, never as one
        # impossible global cross-product.
        explicit_named_multi_compare = (
            decision_mode == "COMPARE"
            and len(explicit_current_message_targets) >= 2
        )
        if explicit_named_multi_compare:
            fresh_filters = sanitize_ai_filters(
                interpretation.get("filters") or {}
            )
            for key in (
                "brands", "exclude_brands",
                "models", "exclude_models",
                "categories", "exclude_categories",
            ):
                fresh_filters.pop(key, None)
            next_filters = fresh_filters
            next_preferences = _canonicalize_buyer_preferences(
                interpretation.get("preferences", []) or []
            )

        current_brands = list(current_filters.get("brands") or [])
        current_models = list(current_filters.get("models") or [])

        # Carry the canonical single vehicle forward for listing refinements only
        # when V8 says this is a continuation of the existing SHOP task.
        if (
            decision_mode == "SHOP"
            and operation != "NEW_TASK"
            and not resolved_targets
            and len(current_brands) == 1
            and len(current_models) == 1
        ):
            carried_target = {
                "brand": current_brands[0],
                "model": current_models[0],
            }
            current_categories = list(current_filters.get("categories") or [])
            if len(current_categories) == 1:
                carried_target["category"] = current_categories[0]
            resolved_targets = [carried_target]

        if (
            decision_mode == "SHOP"
            and _looks_like_unknown_explicit_vehicle_shop_request(
                message,
                resolved_targets,
                current_filters,
                conversation_history,
            )
        ):
            unknown_answer = {
                "TR": "Bu araç adını güncel Kıbrıs piyasa verilerimde eşleştiremedim. Marka ve modeli kontrol edip tekrar yazar mısınız?",
                "RU": "Я не смог сопоставить это название автомобиля с актуальными данными рынка Кипра. Проверьте, пожалуйста, марку и модель.",
                "EN": "I couldn't match that vehicle name to the current Cyprus market data. Please check the make and model and try again.",
            }.get(language, "I couldn't match that vehicle name to the current Cyprus market data. Please check the make and model and try again.")

            return jsonify({
                "success": True,
                "answer": unknown_answer,
                "filters": current_filters,
                "preferences": current_preferences,
                "count": 0,
                "returned": 0,
                "results": [],
                "model_options": [],
                "interpretation": interpretation,
                "decision_mode": "SHOP",
                "stage": "clarification",
                "resolved_vehicle_targets": [],
            })

        # Deterministic explicit-constraint clearing.
        # Conversational phrases such as "mileage doesn't matter anymore" are
        # unambiguous state changes and should not depend on the LLM returning
        # clear_filters correctly.
        low_message_for_clear = message.casefold()

        clear_mileage = bool(
            re.search(
                r"\b(?:mileage|kilomet(?:er|re)s?|km)\s+"
                r"(?:doesn['’]?t|does not|doesnt)\s+matter(?:\s+anymore)?\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:any|whatever)\s+(?:mileage|kilomet(?:er|re)s?|km)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:forget|remove|clear|ignore)\s+(?:the\s+)?"
                r"(?:mileage|kilomet(?:er|re)s?|km)(?:\s+(?:limit|restriction))?\b",
                low_message_for_clear,
            )
            or re.search(
                r"\bkilometre\s+(?:önemli\s+değil|onemli\s+degil)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:пробег\s+не\s+важен|любой\s+пробег)\b",
                low_message_for_clear,
            )
        )

        if clear_mileage:
            next_filters.pop("min_km", None)
            next_filters.pop("max_km", None)

        # Deterministic year-floor clearing for natural corrections such as
        # "older cars are fine too". This means remove the existing minimum-year
        # restriction only; keep the selected vehicle, budget, and other filters.
        clear_min_year = bool(
            re.search(
                r"\b(?:older (?:cars?|vehicles?) (?:are|is) fine(?: too)?|"
                r"older (?:cars?|vehicles?) (?:are|is) okay(?: too)?|"
                r"older (?:cars?|vehicles?) (?:are|is) ok(?: too)?|"
                r"older is fine(?: too)?|older are fine(?: too)?|"
                r"any year is fine|year doesn['’]?t matter(?: anymore)?|"
                r"forget (?:the )?(?:minimum|min) year|"
                r"remove (?:the )?(?:minimum|min) year(?: limit| restriction)?)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:eski araçlar da olur|eski araclar da olur|"
                r"yıl önemli değil|yil onemli degil)\b",
                low_message_for_clear,
            )
            or re.search(
                r"\b(?:старые машины тоже подойдут|год не важен|любой год)\b",
                low_message_for_clear,
            )
        )

        if clear_min_year:
            next_filters.pop("min_year", None)

        # Deterministic scope-reset guard for explicit physical-class changes.
        # The interpreter remains responsible for soft use-cases, but phrases such
        # as "forget small cars, I want an SUV" must not preserve the old class or
        # an old budget simply because the conversational interpreter missed part
        # of a compound correction.
        low_message = message.casefold()
        explicit_scope_reset = bool(re.search(
            r"\b(?:forget|actually|instead|rather|unut|aslında|aslinda|yerine|"
            r"забудь|на самом деле|вместо)\b",
            low_message,
        ))

        if explicit_scope_reset:
            explicit_type = None
            if re.search(r"\b(?:suv|4x4|crossover)\b", low_message):
                explicit_type = "vehicle_type:SUV"
            elif re.search(r"\b(?:motorcycle|motorbike|motosiklet|мотоцикл)\b", low_message):
                explicit_type = "vehicle_type:motorcycle"
            elif re.search(r"\b(?:pickup|pick-up|pick up)\b", low_message):
                explicit_type = "vehicle_type:pickup"

            if explicit_type:
                next_preferences = [
                    p for p in next_preferences
                    if not str(p).casefold().startswith("vehicle_type:")
                    and str(p).casefold() != "any_vehicle_type"
                ]
                next_preferences.append(explicit_type)

                # "Forget <old vehicle class>, I want <new class>" is an explicit
                # replacement of that earlier recommendation brief. Soft buyer-fit
                # preferences attached to the abandoned brief must not silently
                # survive unless the user states them again in the correction turn.
                #
                # Keep this deliberately narrower than generic "actually make it
                # an SUV": that wording can legitimately retain preferences such
                # as economical/family-friendly.
                explicit_forget_old_class = bool(re.search(
                    r"\b(?:forget|unut|забудь)\b.*\b"
                    r"(?:small\s+cars?|city\s+cars?|cars?|suvs?|crossovers?|"
                    r"motorcycles?|motorbikes?|motosiklet(?:ler)?|мотоцикл(?:ы|ов)?|"
                    r"pick(?:-|\s*)ups?)\b",
                    low_message,
                ))

                if explicit_forget_old_class:
                    incoming_pref_keys = {
                        str(p).casefold()
                        for p in (interpretation.get("preferences", []) or [])
                        if str(p).strip()
                        and not str(p).casefold().startswith("vehicle_type:")
                        and str(p).casefold() != "any_vehicle_type"
                    }

                    next_preferences = [
                        p for p in next_preferences
                        if str(p).casefold().startswith("vehicle_type:")
                        or str(p).casefold() == "any_vehicle_type"
                        or str(p).casefold() in incoming_pref_keys
                    ]

            # Preserve the already-validated budget parser as the source of truth,
            # but ensure a replacement budget in a compound scope-change sentence
            # is applied even if the conversational interpreter omitted it.
            # Do not call fast_common_interpretation() here: explicit correction
            # turns are intentionally excluded from that fast path. Extract only
            # the replacement budget deterministically from the correction text.
            reset_budget = None

            budget_match = re.search(
                r"£\s*([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)",
                message,
            )
            if not budget_match:
                budget_match = re.search(
                    r"\b([0-9](?:[0-9.,]|\s(?=\d))*\s*[kK]?)\s*£",
                    message,
                )
            if not budget_match:
                budget_match = re.search(
                    r"\b([0-9]+(?:[.,][0-9]+)?)\s*[kK]\b",
                    low_message,
                )

            if budget_match:
                # _parse_human_number() already expands a trailing k exactly once
                # (e.g. "18k" -> 18000). Do not multiply a second time here.
                reset_budget = _parse_human_number(budget_match.group(1))

            if reset_budget is not None:
                next_filters["budget"] = reset_budget

        # Canonicalize explicitly named single vehicles. This fixes natural compound
        # names such as "Nissan Note e-Power" without hard-coding any vehicle.
        if len(resolved_targets) == 1 and decision_mode in {"DISCOVER", "COMPARE", "SHOP"}:
            target = resolved_targets[0]
            next_filters["brands"] = [target["brand"]]
            next_filters["models"] = [target["model"]]
            if target.get("category"):
                next_filters["categories"] = [target["category"]]
            else:
                next_filters.pop("categories", None)

            next_filters.pop("exclude_brands", None)
            next_filters.pop("exclude_models", None)
            next_filters.pop("exclude_categories", None)

        if interpretation.get("degraded") and interpretation.get("clarification_question"):
            return jsonify({
                "success": True,
                "answer": interpretation["clarification_question"],
                "filters": next_filters,
                "preferences": next_preferences,
                "count": None,
                "returned": 0,
                "results": [],
                "interpretation": interpretation,
                "decision_mode": decision_mode,
                "stage": "clarification",
            })

        if (
            interpretation.get("needs_clarification")
            and interpretation.get("clarification_question")
        ):
            return jsonify({
                "success": True,
                "answer": interpretation["clarification_question"],
                "filters": next_filters,
                "preferences": next_preferences,
                "count": None,
                "returned": 0,
                "results": [],
                "interpretation": interpretation,
                "decision_mode": decision_mode,
                "stage": "clarification",
                "orchestration_version": ASSISTANT_ORCHESTRATION_VERSION,
                "conversation_operation": interpretation.get("operation"),
                "awaiting": interpretation.get("awaiting"),
            })

        # Pull the full filtered result set for candidate selection.
        #
        # Explicit multi-vehicle comparisons are searched target-by-target so each
        # named vehicle can carry its own category/variant. This avoids the invalid
        # cross-product created by global filters such as:
        #   models=[Aqua, Note], categories=[e-Power]
        # which would wrongly require Aqua itself to be e-Power.
        search_started = time.perf_counter()
        if decision_mode in {"COMPARE", "SHOP"} and len(resolved_targets) >= 2:
            compare_base_filters = dict(next_filters)
            for key in (
                "brands", "exclude_brands",
                "models", "exclude_models",
                "categories", "exclude_categories",
            ):
                compare_base_filters.pop(key, None)

            next_filters = compare_base_filters
            search_result = _search_market_for_vehicle_targets(
                base_filters=compare_base_filters,
                targets=resolved_targets,
            )
        else:
            search_result = market_search(
                budget=next_filters.get("budget"),
                min_budget=next_filters.get("min_budget"),
                brands=next_filters.get("brands"),
                exclude_brands=next_filters.get("exclude_brands"),
                models=next_filters.get("models"),
                exclude_models=next_filters.get("exclude_models"),
                categories=next_filters.get("categories"),
                exclude_categories=next_filters.get("exclude_categories"),
                locations=next_filters.get("locations"),
                exclude_locations=next_filters.get("exclude_locations"),
                companies=next_filters.get("companies"),
                exclude_companies=next_filters.get("exclude_companies"),
                transmissions=next_filters.get("transmissions"),
                colors=next_filters.get("colors"),
                min_year=next_filters.get("min_year"),
                max_year=next_filters.get("max_year"),
                min_km=next_filters.get("min_km"),
                max_km=next_filters.get("max_km"),
                limit=5000,
                max_limit=5000,
                analysis_mode=True,
            )

        search_seconds = time.perf_counter() - search_started

        if not search_result.get("success"):
            return jsonify(search_result), 503

        # Explicit physical vehicle classes are authoritative. Apply them after
        # the normal market filters using the validated model-profile taxonomy,
        # never by guessing from listing titles or CategoryDetail.
        search_result = _apply_strict_vehicle_type_to_search_result(
            search_result, next_preferences
        )

        # A single explicitly resolved COMPARE/SHOP target is a canonical vehicle,
        # not a fuzzy model search. market_search intentionally keeps substring
        # matching for the public search API, so "Honda Fit" can also match
        # "Honda Fit Aria". Remove those fuzzy neighbours here before counts,
        # model options, SHOP representatives, and the public assistant response
        # are produced.
        if decision_mode in {"COMPARE", "SHOP"} and len(resolved_targets) == 1:
            target = resolved_targets[0]
            target_brand_cf = str(target.get("brand") or "").strip().casefold()
            target_model_cf = str(target.get("model") or "").strip().casefold()
            target_category_cf = str(target.get("category") or "").strip().casefold()

            exact_results = []
            for item in search_result.get("results", []) or []:
                item_brand_cf = str(item.get("brand") or "").strip().casefold()
                item_model_cf = str(item.get("model") or "").strip().casefold()
                item_category_cf = str(item.get("category") or "").strip().casefold()

                if item_brand_cf != target_brand_cf or item_model_cf != target_model_cf:
                    continue
                if target_category_cf and item_category_cf != target_category_cf:
                    continue
                exact_results.append(item)

            search_result = dict(search_result)
            search_result["results"] = exact_results
            search_result["count"] = len(exact_results)
            search_result["returned"] = len(exact_results)

        # Guided buying flow: broad searches get a compact group of useful
        # narrowing dimensions; only later do we offer secondary refinements.
        guide_question = None
        if decision_mode == "DISCOVER" and not _asks_reliability_question(message):
            guide_question = guided_narrowing_question(
                filters=next_filters,
                preferences=next_preferences,
                count=search_result.get("count", 0),
                language=language,
                message=message,
            )

        if guide_question:
            public_results = (search_result.get("results") or [])[:PUBLIC_ASSISTANT_RESULT_CAP]
            return jsonify({
                "success": True,
                "answer": guide_question,
                "filters": next_filters,
                "preferences": next_preferences,
                "count": search_result.get("count", 0),
                "returned": len(public_results),
                "results": public_results,
                "interpretation": interpretation,
                "decision_mode": "DISCOVER",
                "stage": "narrowing",
                "orchestration_version": ASSISTANT_ORCHESTRATION_VERSION,
                "conversation_operation": interpretation.get("operation"),
                "awaiting": interpretation.get("awaiting"),
            })

        answer_started = time.perf_counter()
        answer, advisory_results, advisory_count, model_options = generate_grounded_market_answer(
            message=message,
            language=language,
            filters=next_filters,
            preferences=next_preferences,
            search_result=search_result,
            conversation_history=conversation_history,
            decision_mode=decision_mode,
        )
        answer_seconds = time.perf_counter() - answer_started
        total_seconds = time.perf_counter() - request_started
        print(
            f"ASSISTANT_TIMING mode={decision_mode} "
            f"resolve={resolve_seconds:.2f}s interpret={interpret_seconds:.2f}s "
            f"search={search_seconds:.2f}s answer={answer_seconds:.2f}s "
            f"fast_interpret={bool(interpretation.get('fast_path'))} total={total_seconds:.2f}s",
            flush=True,
        )
        if total_seconds > 5.0:
            print(
                f"ASSISTANT_SLO_WARN mode={decision_mode} total={total_seconds:.2f}s "
                f"profile_ready={MODEL_PROFILE_READY}",
                flush=True,
            )

        if decision_mode == "SHOP":
            public_results = _select_shop_representatives(
                advisory_results or search_result.get("results") or [],
                max_candidates=3,
                sort_mode=_listing_sort_mode(next_preferences),
            )
        else:
            public_results = (
                advisory_results or search_result.get("results") or []
            )[:100]
        fallback_support = (
            _fallback_support_payload(language)
            if int(search_result.get("count", 0) or 0) == 0
            else {"suggestions": [], "actions": []}
        )

        response_actions = list(fallback_support["actions"])
        if decision_mode == "COMPARE":
            for option in (model_options or [])[:4]:
                deal = option.get("potential_value_listing") or {}
                deal_url = str(deal.get("link") or "").strip()
                if not deal_url:
                    continue
                vehicle_name = f"{option.get('brand','')} {option.get('model','')}".strip()
                if language == "TR":
                    action_label = f"{vehicle_name} ilanını aç"
                elif language == "RU":
                    action_label = f"Открыть объявление {vehicle_name}"
                else:
                    action_label = f"Open {vehicle_name} listing"
                response_actions.append({
                    "type": "LISTING",
                    "label": action_label,
                    "url": deal_url,
                })

        return jsonify({
            "success": True,
            "answer": answer,
            "filters": next_filters,
            "preferences": next_preferences,
            "count": search_result.get("count", 0),
            "advisory_count": advisory_count,
            "model_options": model_options,
            "returned": len(public_results),
            "results": public_results,
            "interpretation": interpretation,
            "resolved_vehicle_targets": resolved_targets,
            "decision_mode": decision_mode,
            "stage": "recommendation",
            "suggestions": fallback_support["suggestions"],
            "actions": response_actions,
            "access_tier": access_tier,
            "business_capabilities": business_capabilities,
            "orchestration_version": ASSISTANT_ORCHESTRATION_VERSION,
            "conversation_operation": interpretation.get("operation"),
            "awaiting": interpretation.get("awaiting"),
        })

    except AIUsageLimitExceeded as e:
        print("AI USAGE LIMIT:", str(e), flush=True)
        return jsonify({
            "success": False,
            "error": "AI_USAGE_LIMIT_REACHED",
        }), 429

    except requests.HTTPError as e:
        status_code = (
            e.response.status_code
            if e.response is not None
            else None
        )

        response_text = (
            e.response.text[:2000]
            if e.response is not None
            else ""
        )

        print(
            "OPENAI ASSISTANT HTTP ERROR:",
            status_code,
            response_text,
            flush=True
        )

        return jsonify({
            "success": False,
            "error": "AI_ASSISTANT_TEMPORARILY_UNAVAILABLE"
        }), 502

    except RuntimeError as e:
        print("AI ASSISTANT CONFIG ERROR:", e, flush=True)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 503

    except Exception as e:
        print("AI ASSISTANT FAILED:", repr(e), flush=True)
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": "AI_ASSISTANT_FAILED"
        }), 500



# =========================================================
# OTODEĞER V10 DECISION AGENT
# =========================================================
def _access_control_error_response(exc):
    payload = {
        "success": False,
        "error": getattr(exc, "code", "ACCESS_DENIED"),
    }
    retry_after = int(getattr(exc, "retry_after", 0) or 0)
    metadata = getattr(exc, "metadata", None)
    if retry_after:
        payload["retry_after_seconds"] = retry_after
    if isinstance(metadata, dict):
        payload.update({k: v for k, v in metadata.items() if k not in payload})
    response = jsonify(payload)
    response.status_code = int(getattr(exc, "status_code", 403) or 403)
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
    return response


def _request_access_context(data=None):
    data = data if isinstance(data, dict) else {}
    manager = get_access_manager()
    return manager.resolve_context(
        request,
        client_tier=data.get("access_tier") or data.get("tier"),
        client_company=data.get("business_company"),
        client_ip=_client_ip(),
    )



def _normalise_fraction(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if abs(numeric) > 1 and abs(numeric) <= 100:
        numeric = numeric / 100.0
    return numeric



def _round_business_benchmark(value, step=50):
    """Round derived market statistics to a sensible vehicle-pricing increment.

    Actual advertised asking prices remain exact. Derived medians/targets should
    not expose false precision such as £14,262.50 in a commercial vehicle report.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    step = max(int(step or 50), 1)
    return int(math.floor((numeric / step) + 0.5) * step)


def _business_inventory_price_history(company, links):
    """Return observed asking-price history for active inventory links.

    This is the vehicle's own advertised-price history, not a market benchmark.
    It is used to make Inventory Pricing explanations specific and operational:
    when a price moved, by how much, and whether earlier repricing has already
    changed the vehicle's position.  Missing raw daily history is represented as
    unavailable rather than inferred from aggregate flags.
    """
    link_set = {str(value or "").strip() for value in (links or []) if str(value or "").strip()}
    if not link_set:
        return {}

    source = business_activity_history_df if BUSINESS_ACTIVITY_HISTORY_READY and business_activity_history_df is not None else pd.DataFrame()
    if source is None or source.empty:
        return {}

    lower = {str(col).strip().casefold(): col for col in source.columns}
    date_col = lower.get("date") or lower.get("tarih")
    company_col = lower.get("company") or lower.get("seller") or lower.get("dealer") or lower.get("firma")
    link_col = lower.get("link") or lower.get("url")
    price_col = lower.get("price") or lower.get("currentaskingprice") or lower.get("fiyat")
    if not all([date_col, company_col, link_col, price_col]):
        return {}

    work = source[[date_col, company_col, link_col, price_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work[company_col] = work[company_col].fillna("").astype(str).str.strip()
    work[link_col] = work[link_col].fillna("").astype(str).str.strip()
    target_company = _normalize_company_name(company)
    work = work[
        work[date_col].notna()
        & work[price_col].notna()
        & work[link_col].isin(link_set)
        & (work[company_col].map(_normalize_company_name) == target_company)
    ].copy()
    if work.empty:
        return {}

    history = {}
    for link, group in work.groupby(link_col, sort=False):
        ordered = group.sort_values(date_col, kind="stable").copy()
        ordered["_day"] = ordered[date_col].dt.normalize()
        ordered = ordered.drop_duplicates(subset=["_day"], keep="last")
        if ordered.empty:
            continue

        first_price = float(ordered.iloc[0][price_col])
        latest_price = float(ordered.iloc[-1][price_col])
        changes = []
        previous_price = first_price
        for _, row in ordered.iloc[1:].iterrows():
            current_price = float(row[price_col])
            if abs(current_price - previous_price) < 0.5:
                previous_price = current_price
                continue
            amount = current_price - previous_price
            changes.append({
                "date": pd.Timestamp(row[date_col]).date().isoformat(),
                "previous_price": previous_price,
                "new_price": current_price,
                "change_amount": amount,
                "change_pct": round((amount / previous_price) * 100.0, 2) if previous_price else None,
                "direction": "INCREASE" if amount > 0 else "REDUCTION",
            })
            previous_price = current_price

        reductions = sum(1 for item in changes if item["direction"] == "REDUCTION")
        increases = sum(1 for item in changes if item["direction"] == "INCREASE")
        net_amount = latest_price - first_price
        history[str(link)] = {
            "price_history_available": True,
            "first_observed_price": first_price,
            "first_observed_date": pd.Timestamp(ordered.iloc[0][date_col]).date().isoformat(),
            "latest_observed_price": latest_price,
            "latest_observed_date": pd.Timestamp(ordered.iloc[-1][date_col]).date().isoformat(),
            "price_change_count": len(changes),
            "price_reduction_count": reductions,
            "price_increase_count": increases,
            "price_history_net_change": net_amount,
            "price_history_net_change_pct": round((net_amount / first_price) * 100.0, 2) if first_price else None,
            # Most recent changes first for the UI; retain enough context without
            # turning each inventory card into a transaction log.
            "price_history": list(reversed(changes[-6:])),
        }
    return history


def _business_inventory_pricing_recommendations(company, limit=500):
    """Return a complete, price-focused view of a gallery's active inventory.

    Every active vehicle is assigned to exactly one pricing bucket:
      * PRICE_REVIEW               — asking position is high vs current comparables
      * COMPETITIVELY_POSITIONED   — broadly market-aligned
      * POTENTIALLY_UNDER_PRICED   — asking position is low vs current comparables
      * NO_MARKET_COMPARABLES      — no defensible current benchmark can be formed

    Stock age is deliberately excluded.  A vehicle without a usable market-price
    benchmark is never silently placed in a conclusion-bearing bucket merely to
    make the category counts add up.
    """
    if not company or business_stock_df is None or business_stock_df.empty:
        return {"recommendations": [], "summary": {}}

    work = business_stock_df[
        business_stock_df["Company"].fillna("").astype(str).str.casefold()
        == str(company).casefold()
    ].copy()
    if work.empty:
        return {"recommendations": [], "summary": {}}

    links = work["Link"].dropna().astype(str).str.strip().tolist() if "Link" in work.columns else []
    price_history_by_link = _business_inventory_price_history(company, links)

    bucket_rank = {
        "PRICE_REVIEW": 0,
        "POTENTIALLY_UNDER_PRICED": 1,
        "COMPETITIVELY_POSITIONED": 2,
        "NO_MARKET_COMPARABLES": 3,
    }
    counts = {
        "PRICE_REVIEW": 0,
        "COMPETITIVELY_POSITIONED": 0,
        "POTENTIALLY_UNDER_PRICED": 0,
        "NO_MARKET_COMPARABLES": 0,
    }
    rows = []

    for raw in work.to_dict("records"):
        public = _business_manage_row_public(raw)
        link = str(public.get("link") or "").strip()
        if link in price_history_by_link:
            public.update(price_history_by_link[link])
        else:
            public["price_history_available"] = False
            public["price_history"] = []
            public["price_change_count"] = None
            public["price_reduction_count"] = None
            public["price_increase_count"] = None

        price_position = str(public.get("price_position") or "").strip().upper()
        comp_conf = str(public.get("comparable_confidence") or "").strip().upper()
        comp_count_raw = public.get("comparable_count")
        try:
            comp_count = int(comp_count_raw) if comp_count_raw is not None else 0
        except (TypeError, ValueError):
            comp_count = 0

        median_raw = public.get("comparable_median_price")
        try:
            median_numeric = float(median_raw) if median_raw is not None else None
            if median_numeric is not None and not math.isfinite(median_numeric):
                median_numeric = None
        except (TypeError, ValueError):
            median_numeric = None

        # Pricing conclusions require an actual current-market benchmark.  A row
        # with no median / no comparable set belongs in its own explicit bucket;
        # the vehicle's own price history can add context but cannot substitute
        # for a market comparison.
        has_market_benchmark = (
            median_numeric is not None
            and median_numeric > 0
            and comp_count > 0
            and price_position not in {"", "UNKNOWN", "UNAVAILABLE", "NONE"}
        )

        p25_raw = public.get("comparable_p25_price")
        p75_raw = public.get("comparable_p75_price")
        public["display_comparable_median_price"] = _round_business_benchmark(median_numeric, 50)
        public["display_comparable_p25_price"] = _round_business_benchmark(p25_raw, 50)
        public["display_comparable_p75_price"] = _round_business_benchmark(p75_raw, 50)

        asking = public.get("asking_price")
        try:
            asking_numeric = float(asking) if asking is not None else None
            if asking_numeric is not None and not math.isfinite(asking_numeric):
                asking_numeric = None
        except (TypeError, ValueError):
            asking_numeric = None

        # Treat small deviations from the comparable median as normal market
        # noise, not as a pricing opportunity.  In particular, a vehicle should
        # not be labelled "potentially under-priced" merely because it is £1 or
        # £50 below the median.  The gap must be commercially material in both
        # percentage and cash terms.
        materiality_threshold = None
        price_gap_to_median = None
        if has_market_benchmark and asking_numeric is not None:
            price_gap_to_median = asking_numeric - median_numeric
            materiality_threshold = max(500.0, median_numeric * 0.05)

        if not has_market_benchmark or asking_numeric is None:
            bucket = "NO_MARKET_COMPARABLES"
        elif price_gap_to_median is not None and price_gap_to_median >= materiality_threshold:
            bucket = "PRICE_REVIEW"
        elif price_gap_to_median is not None and price_gap_to_median <= -materiality_threshold:
            bucket = "POTENTIALLY_UNDER_PRICED"
        else:
            bucket = "COMPETITIVELY_POSITIONED"

        counts[bucket] += 1
        public["pricing_bucket"] = bucket
        public["pricing_evidence_limited"] = (
            bucket == "NO_MARKET_COMPARABLES"
            or comp_conf not in {"HIGH", "MEDIUM"}
            or comp_count < 2
        )
        public["pricing_materiality_threshold"] = _round_business_benchmark(materiality_threshold, 50) if materiality_threshold is not None else None
        public["price_gap_to_median"] = price_gap_to_median

        median_target = _round_business_benchmark(median_numeric, 250)
        lower_target = _round_business_benchmark(p25_raw, 250)
        if (
            bucket == "PRICE_REVIEW"
            and asking_numeric is not None
            and median_target is not None
            and abs(float(median_target)) < asking_numeric
        ):
            high_target = abs(int(median_target))
            low_target = abs(int(lower_target)) if lower_target is not None else None
            public["suggested_market_position_high"] = high_target
            public["suggested_market_position_low"] = low_target if low_target is not None and low_target < high_target else None
        elif bucket == "POTENTIALLY_UNDER_PRICED" and median_target is not None:
            # For a low-priced vehicle, show the user where the current market
            # sits rather than merely flagging the gap.  Use the lower quartile
            # to median as a cautious market-alignment range when available.
            high_target = abs(int(median_target))
            low_target = abs(int(lower_target)) if lower_target is not None else None
            if low_target is not None and asking_numeric is not None and low_target <= asking_numeric:
                low_target = None
            public["suggested_market_position_high"] = high_target
            public["suggested_market_position_low"] = low_target if low_target is not None and low_target < high_target else None
        else:
            public["suggested_market_position_low"] = None
            public["suggested_market_position_high"] = None

        raw_delta = public.get("price_vs_median_pct")
        try:
            delta = float(raw_delta) if raw_delta is not None else 0.0
            if abs(delta) <= 1:
                delta *= 100.0
        except (TypeError, ValueError):
            delta = 0.0
        if bucket == "PRICE_REVIEW":
            sort_pressure = delta
        elif bucket == "POTENTIALLY_UNDER_PRICED":
            sort_pressure = abs(min(delta, 0.0))
        elif bucket == "NO_MARKET_COMPARABLES":
            sort_pressure = 0.0
        else:
            sort_pressure = -abs(delta)

        rows.append({
            **public,
            "_bucket_rank": bucket_rank[bucket],
            "_sort_pressure": float(sort_pressure),
            "_comp_count": int(comp_count or 0),
        })

    rows.sort(
        key=lambda item: (
            item.get("_bucket_rank", 9),
            -float(item.get("_sort_pressure") or 0),
            -int(item.get("_comp_count") or 0),
            str(item.get("brand") or ""),
            str(item.get("model") or ""),
        )
    )

    try:
        requested_limit = int(limit or len(rows))
    except (TypeError, ValueError):
        requested_limit = len(rows)
    requested_limit = max(1, min(requested_limit, 500))

    public_rows = []
    for item in rows[:requested_limit]:
        item = dict(item)
        item.pop("_bucket_rank", None)
        item.pop("_sort_pressure", None)
        item.pop("_comp_count", None)
        public_rows.append(item)

    summary = {
        "active_stock_count": int(len(work)),
        "price_review_count": int(counts["PRICE_REVIEW"]),
        "competitively_positioned_count": int(counts["COMPETITIVELY_POSITIONED"]),
        "potentially_underpriced_count": int(counts["POTENTIALLY_UNDER_PRICED"]),
        "no_market_comparables_count": int(counts["NO_MARKET_COMPARABLES"]),
    }
    return {"recommendations": public_rows, "summary": summary}

def _business_advertising_recommendations(company, limit=5):
    """Rank current dealership stock by where paid visibility is most defensible.

    Advertising is treated as a visibility lever, not a substitute for pricing.
    The score therefore combines stock-age pressure, historical market turnover,
    competitive price position and evidence quality. Vehicles that are clearly
    high-priced versus current comparables are penalised because repricing may be
    the more appropriate first action.
    """
    if not company or business_stock_df is None or business_stock_df.empty:
        return []

    work = business_stock_df[
        business_stock_df["Company"].fillna("").astype(str).str.casefold()
        == str(company).casefold()
    ].copy()
    if work.empty:
        return []

    price_scores = {
        "LOW": 1.00,
        "LOW_MID": 0.92,
        "MID_MARKET": 0.82,
        "HIGH_MID": 0.48,
        "HIGH": 0.22,
    }
    confidence_scores = {"HIGH": 1.0, "MEDIUM": 0.78, "LOW": 0.48}

    ranked = []
    for row in work.to_dict("records"):
        public = _business_manage_row_public(row)
        age = public.get("listing_age_days")
        median_days = public.get("historical_median_days_to_exit")
        exit60 = _normalise_fraction(public.get("historical_exit60_rate"))
        price_position = str(public.get("price_position") or "").strip().upper()
        liquidity_conf = str(public.get("liquidity_confidence") or "").strip().upper()
        comparable_conf = str(public.get("comparable_confidence") or "").strip().upper()

        if age is not None and median_days not in [None, 0]:
            ratio = max(float(age), 0.0) / max(float(median_days), 1.0)
            age_score = min(max((ratio - 0.45) / 1.55, 0.0), 1.0)
        elif age is not None:
            age_score = min(max(float(age), 0.0) / 180.0, 1.0) * 0.72
        else:
            age_score = 0.20

        if exit60 is not None:
            liquidity_score = min(max(exit60, 0.0), 1.0)
        elif median_days is not None:
            liquidity_score = min(max(1.0 - (float(median_days) / 180.0), 0.08), 0.92)
        else:
            liquidity_score = 0.35

        price_score = price_scores.get(price_position, 0.62)
        evidence_score = (
            confidence_scores.get(liquidity_conf, 0.55)
            + confidence_scores.get(comparable_conf, 0.55)
        ) / 2.0

        score = (
            0.36 * age_score
            + 0.34 * liquidity_score
            + 0.22 * price_score
            + 0.08 * evidence_score
        )
        # Advertising should not be presented as the first fix for a vehicle whose
        # asking price is already materially high versus the market.
        if price_position == "HIGH":
            score -= 0.11
        elif price_position == "HIGH_MID":
            score -= 0.05
        score = max(0.0, min(score, 1.0))

        public["promotion_score"] = round(score, 4)
        public["historical_exit60_rate"] = exit60
        public["historical_distinct_listings"] = int(row["HistoricalDistinctListings"]) if pd.notna(row.get("HistoricalDistinctListings")) else None
        ranked.append(public)

    ranked.sort(
        key=lambda item: (
            float(item.get("promotion_score") or 0),
            float(item.get("listing_age_days") or 0),
            -abs(float(item.get("price_vs_median_pct") or 0)),
        ),
        reverse=True,
    )
    return ranked[:max(1, min(int(limit or 5), 8))]



def _business_activity_company_rows(frame, company, company_column="Company"):
    """Return rows for a gallery using the same normalization as Business access.

    Activity aggregates and stock intelligence can originate from separate offline
    builders.  Small punctuation/spacing differences must not make asking-value
    totals silently disappear from the report.
    """
    if frame is None or frame.empty or not company or company_column not in frame.columns:
        return pd.DataFrame()
    target = _normalize_company_name(company)
    if not target:
        return pd.DataFrame()
    normalized = frame[company_column].fillna("").astype(str).map(_normalize_company_name)
    return frame[normalized == target].copy()


def _business_activity_current_stock_value(company):
    """Return the selected gallery's current asking-value exposure when available."""
    if not company:
        return None

    try:
        if business_company_df is not None and not business_company_df.empty and {"Company", "CurrentStockAskingValue"}.issubset(business_company_df.columns):
            rows = _business_activity_company_rows(business_company_df, company)
            if not rows.empty:
                value = pd.to_numeric(rows.iloc[0].get("CurrentStockAskingValue"), errors="coerce")
                if pd.notna(value):
                    return float(value)
    except Exception:
        pass

    try:
        if business_stock_df is not None and not business_stock_df.empty and {"Company", "CurrentAskingPrice"}.issubset(business_stock_df.columns):
            rows = _business_activity_company_rows(business_stock_df, company)
            prices = pd.to_numeric(rows["CurrentAskingPrice"], errors="coerce").dropna()
            if not prices.empty:
                return float(prices.sum())
    except Exception:
        pass

    return None


def _business_activity_event_details(company, start, end, expected_metrics=None, daily_expected=None, available_end=None):
    """Reconstruct exact vehicle activity plus useful asking-value movement.

    Aggregate company-day counts remain authoritative. Raw daily observations supply
    identities and listed prices. Exact prices are used whenever present; if an
    individual monetary value is absent, only that value is estimated from the
    closest observed gallery asking-price level instead of blanking the whole report.
    Vehicle identities are never invented.
    """
    empty = {
        "available": False,
        "details_complete": False,
        "coverage": {},
        "new_listings": [],
        "observed_exits": [],
        "price_reductions": [],
        "price_increases": [],
        "daily_events": {},
        "daily_values": {},
        "opening_stock_value": None,
        "closing_stock_value": None,
        "net_stock_value_change": None,
        "net_stock_value_change_pct": None,
        "new_listing_asking_value": None,
        "observed_exit_asking_value": None,
        "price_reduction_value": None,
        "price_increase_value": None,
        "net_price_change_value": None,
        "value_movement_complete": False,
        "value_estimates_used": False,
    }

    if not company:
        return empty

    def _prepare_activity_source(candidate):
        if candidate is None or not isinstance(candidate, pd.DataFrame) or candidate.empty:
            return None
        prepared = candidate.copy()
        lower = {str(col).strip().casefold(): col for col in prepared.columns}

        def _find(*names):
            for name in names:
                if name in prepared.columns:
                    return name
                found = lower.get(str(name).strip().casefold())
                if found:
                    return found
            return None

        date_col_local = _find("DATE", "Date")
        company_col_local = _find("Company", "Seller", "Dealer")
        link_col_local = _find("Link", "URL", "Url")
        price_col_local = _find("Price", "CurrentAskingPrice")
        if not all([date_col_local, company_col_local, link_col_local, price_col_local]):
            return None
        prepared[date_col_local] = pd.to_datetime(prepared[date_col_local], errors="coerce")
        prepared[price_col_local] = pd.to_numeric(prepared[price_col_local], errors="coerce")
        prepared[company_col_local] = prepared[company_col_local].fillna("").astype(str).str.strip()
        prepared[link_col_local] = prepared[link_col_local].fillna("").astype(str).str.strip()
        target_company = _normalize_company_name(company)
        prepared["_company_key"] = prepared[company_col_local].map(_normalize_company_name)
        prepared = prepared[
            prepared[date_col_local].notna()
            & prepared[link_col_local].ne("")
            & (prepared["_company_key"] == target_company)
        ].copy()
        if prepared.empty or prepared[date_col_local].dt.normalize().nunique() < 2:
            return None
        return {
            "frame": prepared,
            "date": date_col_local,
            "company": company_col_local,
            "link": link_col_local,
            "price": price_col_local,
            "brand": _find("Brand"),
            "model": _find("Model"),
            "category": _find("Category", "CategoryDetail"),
            "year": _find("Year"),
            "km": _find("KM", "Mileage"),
        }

    source_candidates = []
    if BUSINESS_ACTIVITY_HISTORY_READY and business_activity_history_df is not None and not business_activity_history_df.empty:
        source_candidates.append(business_activity_history_df)
    # `ads_base.csv` is not the preferred activity source, but older deployments
    # can still contain repeated daily observations there.  Use it as a defensive
    # identity/detail fallback rather than returning an empty vehicle-level report.
    if isinstance(df, pd.DataFrame) and not df.empty:
        source_candidates.append(df)

    prepared_source = None
    for candidate in source_candidates:
        prepared_source = _prepare_activity_source(candidate)
        if prepared_source is not None:
            break
    if prepared_source is None:
        return empty

    source = prepared_source["frame"]
    date_col = prepared_source["date"]
    company_col = prepared_source["company"]
    link_col = prepared_source["link"]
    price_col = prepared_source["price"]
    brand_col = prepared_source["brand"]
    model_col = prepared_source["model"]
    category_col = prepared_source["category"]
    year_col = prepared_source["year"]
    km_col = prepared_source["km"]

    source["_activity_date"] = source[date_col].dt.normalize()
    source = source.sort_values(["_activity_date", link_col, date_col], kind="stable")
    source = source.drop_duplicates(subset=["_activity_date", link_col], keep="last")

    def _plain_number(value, integer=False):
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            return None
        return int(round(float(number))) if integer else float(number)

    def _plain_text(value):
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value or "").strip()

    def _market_estimate_for_row(row):
        """Best available current-market asking estimate for a missing listing price."""
        try:
            if business_market_df is None or business_market_df.empty:
                return None
            work = business_market_df.copy()
            year = _plain_number(row.get(year_col), integer=True) if year_col else None
            brand = _plain_text(row.get(brand_col)) if brand_col else ""
            model = _plain_text(row.get(model_col)) if model_col else ""
            category = _plain_text(row.get(category_col)) if category_col else ""
            if year is not None and "Year" in work.columns:
                work = work[pd.to_numeric(work["Year"], errors="coerce") == int(year)]
            if brand and "Brand" in work.columns:
                work = work[work["Brand"].fillna("").astype(str).str.casefold() == brand.casefold()]
            if model and "Model" in work.columns:
                work = work[work["Model"].fillna("").astype(str).str.casefold() == model.casefold()]
            category_col_market = "CategoryDetail" if "CategoryDetail" in work.columns else ("Category" if "Category" in work.columns else None)
            if category and category_col_market:
                exact = work[work[category_col_market].fillna("").astype(str).str.casefold() == category.casefold()]
                if not exact.empty:
                    work = exact
            for value_col in ["CurrentMedianPrice", "CurrentStartingPrice"]:
                if value_col in work.columns:
                    values = pd.to_numeric(work[value_col], errors="coerce").dropna()
                    values = values[values > 0]
                    if not values.empty:
                        return float(values.median())
        except Exception:
            return None
        return None

    def _event_base(row, event_date, link):
        listed_price = _plain_number(row.get(price_col))
        estimated_price = False
        if listed_price is None:
            listed_price = _market_estimate_for_row(row)
            estimated_price = listed_price is not None
        return {
            "date": pd.Timestamp(event_date).date().isoformat(),
            "link": _plain_text(link),
            "year": _plain_number(row.get(year_col), integer=True) if year_col else None,
            "brand": _plain_text(row.get(brand_col)) if brand_col else "",
            "model": _plain_text(row.get(model_col)) if model_col else "",
            "category": _plain_text(row.get(category_col)) if category_col else "",
            "km": _plain_number(row.get(km_col), integer=True) if km_col else None,
            "price": listed_price,
            "price_estimated": bool(estimated_price),
        }

    snapshots = {}
    for activity_date, group in source.groupby("_activity_date", sort=True):
        snapshot = {}
        for _, row in group.iterrows():
            link = _plain_text(row.get(link_col))
            if link:
                snapshot[link] = row
        if snapshot:
            snapshots[pd.Timestamp(activity_date).normalize()] = snapshot
    snapshot_dates = sorted(snapshots)
    if not snapshot_dates:
        return empty

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    expected_metrics = expected_metrics or {}

    def _snapshot_on_or_before(target):
        target = pd.Timestamp(target).normalize()
        eligible = [day for day in snapshot_dates if day <= target]
        if not eligible:
            return None, {}
        day = eligible[-1]
        return day, snapshots.get(day, {})

    def _previous_snapshot(day):
        if day is None:
            return None, {}
        day = pd.Timestamp(day).normalize()
        eligible = [candidate for candidate in snapshot_dates if candidate < day]
        if not eligible:
            return None, {}
        prev_day = eligible[-1]
        return prev_day, snapshots.get(prev_day, {})

    def _snapshot_price_stats(snapshot):
        prices = []
        for row in snapshot.values():
            price = _plain_number(row.get(price_col))
            if price is not None and price > 0:
                prices.append(float(price))
        return {
            "listed_count": len(snapshot),
            "priced_count": len(prices),
            "sum": float(sum(prices)) if prices else None,
            "median": float(pd.Series(prices).median()) if prices else None,
        }

    company_price_fallback = None
    try:
        current_rows = _business_activity_company_rows(business_stock_df, company)
        if not current_rows.empty and "CurrentAskingPrice" in current_rows.columns:
            current_prices = pd.to_numeric(current_rows["CurrentAskingPrice"], errors="coerce").dropna()
            current_prices = current_prices[current_prices > 0]
            if not current_prices.empty:
                company_price_fallback = float(current_prices.median())
    except Exception:
        company_price_fallback = None

    price_change_fallback = None
    try:
        history_map = _business_inventory_price_history(company)
        observed_changes = []
        for payload in (history_map or {}).values():
            for change in payload.get("price_history") or []:
                amount = pd.to_numeric(change.get("change_amount"), errors="coerce")
                if pd.notna(amount) and abs(float(amount)) >= 0.5:
                    observed_changes.append(abs(float(amount)))
        if observed_changes:
            price_change_fallback = float(pd.Series(observed_changes).median())
    except Exception:
        price_change_fallback = None
    if price_change_fallback is None and company_price_fallback is not None:
        price_change_fallback = max(250.0, company_price_fallback * 0.03)

    expected_daily = {str(item.get("date")): item for item in (daily_expected or []) if item.get("date")}
    daily_events = defaultdict(lambda: {
        "new_listings": [], "observed_exits": [],
        "price_reductions": [], "price_increases": [],
    })
    daily_values = {}
    all_new, all_exits, all_reductions, all_increases = [], [], [], []
    estimates_used = False

    def _estimate_total(known_values, expected_count, fallback):
        clean = []
        for value in known_values:
            try:
                number = float(value)
                if math.isfinite(number):
                    clean.append(number)
            except (TypeError, ValueError):
                pass
        expected_count = max(0, int(expected_count or 0))
        if expected_count == 0:
            return 0.0, False
        if len(clean) == expected_count:
            return float(sum(clean)), False
        basis = float(pd.Series(clean).median()) if clean else (float(fallback) if fallback is not None else None)
        if basis is None or not math.isfinite(basis):
            return None, False
        if len(clean) > expected_count:
            return float(basis * expected_count), True
        return float(sum(clean) + basis * (expected_count - len(clean))), True

    for date_key, expected in expected_daily.items():
        report_day = pd.Timestamp(date_key).normalize()
        raw_day, current_snapshot = _snapshot_on_or_before(report_day)
        prev_day, previous_snapshot = _previous_snapshot(raw_day)
        current_links = set(current_snapshot)
        previous_links = set(previous_snapshot)
        if raw_day == report_day:
            new_links = sorted(current_links - previous_links)
            exit_links = sorted(previous_links - current_links)
            common_links = sorted(current_links & previous_links)
        else:
            new_links, exit_links, common_links = [], [], []

        new_events = []
        for link in new_links:
            event = _event_base(current_snapshot[link], report_day, link)
            event["type"] = "NEW_LISTING"
            new_events.append(event)
        exit_events = []
        for link in exit_links:
            event = _event_base(previous_snapshot[link], report_day, link)
            event["type"] = "OBSERVED_EXIT"
            exit_events.append(event)

        reductions, increases = [], []
        for link in common_links:
            prev_row, curr_row = previous_snapshot[link], current_snapshot[link]
            prev_price = _plain_number(prev_row.get(price_col))
            curr_price = _plain_number(curr_row.get(price_col))
            if prev_price is None or curr_price is None or abs(curr_price - prev_price) < 0.5:
                continue
            event = _event_base(curr_row, report_day, link)
            event["previous_price"] = float(prev_price)
            event["new_price"] = float(curr_price)
            event["change_amount"] = float(curr_price) - float(prev_price)
            event["change_pct"] = round(((curr_price - prev_price) / prev_price) * 100.0, 2) if prev_price else None
            if curr_price < prev_price:
                event["type"] = "PRICE_REDUCTION"; reductions.append(event)
            else:
                event["type"] = "PRICE_INCREASE"; increases.append(event)

        daily_events[date_key] = {
            "new_listings": new_events, "observed_exits": exit_events,
            "price_reductions": reductions, "price_increases": increases,
        }
        all_new.extend(new_events); all_exits.extend(exit_events)
        all_reductions.extend(reductions); all_increases.extend(increases)

        stock_stats = _snapshot_price_stats(current_snapshot)
        prev_stats = _snapshot_price_stats(previous_snapshot)
        expected_closing = int(expected.get("closing_stock") or 0)
        expected_new = int(expected.get("new_listings") or 0)
        expected_exits = int(expected.get("observed_exits") or 0)
        expected_reductions = int(expected.get("price_reductions") or 0)
        expected_increases = int(expected.get("price_increases") or 0)

        stock_value = stock_stats["sum"]
        stock_est = False
        missing_priced = max(0, expected_closing - stock_stats["priced_count"])
        if expected_closing > 0 and (stock_stats["listed_count"] != expected_closing or missing_priced > 0):
            basis = stock_stats["median"] or company_price_fallback
            if basis is not None:
                known_sum = stock_stats["sum"] or 0.0
                if stock_stats["listed_count"] == expected_closing:
                    stock_value = known_sum + basis * missing_priced
                else:
                    stock_value = basis * expected_closing
                stock_est = True

        new_value, new_est = _estimate_total([e.get("price") for e in new_events], expected_new, stock_stats["median"] or company_price_fallback)
        exit_value, exit_est = _estimate_total([e.get("price") for e in exit_events], expected_exits, prev_stats["median"] or company_price_fallback)
        reduction_value, reduction_est = _estimate_total([abs(float(e.get("change_amount") or 0)) for e in reductions], expected_reductions, price_change_fallback)
        increase_value, increase_est = _estimate_total([abs(float(e.get("change_amount") or 0)) for e in increases], expected_increases, price_change_fallback)
        estimates_used = estimates_used or stock_est or new_est or exit_est or reduction_est or increase_est
        daily_values[date_key] = {
            "stock_asking_value": stock_value,
            "new_listing_asking_value": new_value,
            "observed_exit_asking_value": exit_value,
            "price_reduction_value": reduction_value,
            "price_increase_value": increase_value,
        }

    # Supplement the flat vehicle-detail feed directly from every raw snapshot
    # transition in the requested window.  Aggregate company-day builders and the
    # raw scraper can occasionally have slightly different date coverage; that
    # should not make otherwise observable vehicle identities disappear from the
    # interactive report.  Aggregate counts remain authoritative.
    seen_event_keys = {
        (str(item.get("type") or ""), str(item.get("date") or ""), str(item.get("link") or ""))
        for item in (all_new + all_exits + all_reductions + all_increases)
    }
    for raw_day in snapshot_dates:
        raw_day = pd.Timestamp(raw_day).normalize()
        if raw_day < start or raw_day > end:
            continue
        prev_day, previous_snapshot = _previous_snapshot(raw_day)
        current_snapshot = snapshots.get(raw_day, {})
        if prev_day is None:
            continue
        current_links = set(current_snapshot)
        previous_links = set(previous_snapshot)
        for link in sorted(current_links - previous_links):
            event = _event_base(current_snapshot[link], raw_day, link)
            event["type"] = "NEW_LISTING"
            key = (event["type"], event["date"], event["link"])
            if key not in seen_event_keys:
                all_new.append(event); seen_event_keys.add(key)
        for link in sorted(previous_links - current_links):
            event = _event_base(previous_snapshot[link], raw_day, link)
            event["type"] = "OBSERVED_EXIT"
            key = (event["type"], event["date"], event["link"])
            if key not in seen_event_keys:
                all_exits.append(event); seen_event_keys.add(key)
        for link in sorted(current_links & previous_links):
            prev_row, curr_row = previous_snapshot[link], current_snapshot[link]
            prev_price = _plain_number(prev_row.get(price_col))
            curr_price = _plain_number(curr_row.get(price_col))
            if prev_price is None or curr_price is None or abs(curr_price - prev_price) < 0.5:
                continue
            event = _event_base(curr_row, raw_day, link)
            event["previous_price"] = float(prev_price)
            event["new_price"] = float(curr_price)
            event["change_amount"] = float(curr_price) - float(prev_price)
            event["change_pct"] = round(((curr_price - prev_price) / prev_price) * 100.0, 2) if prev_price else None
            if curr_price < prev_price:
                event["type"] = "PRICE_REDUCTION"; target_events = all_reductions
            else:
                event["type"] = "PRICE_INCREASE"; target_events = all_increases
            key = (event["type"], event["date"], event["link"])
            if key not in seen_event_keys:
                target_events.append(event); seen_event_keys.add(key)

    event_map = {
        "new_listings": all_new, "observed_exits": all_exits,
        "price_reductions": all_reductions, "price_increases": all_increases,
    }
    coverage = {key: len(events) == int(expected_metrics.get(key) or 0) for key, events in event_map.items()}
    details_complete = bool(coverage) and all(coverage.values())

    first_report_day = min([pd.Timestamp(k).normalize() for k in expected_daily], default=start)
    first_raw_day, _ = _snapshot_on_or_before(first_report_day)
    _, opening_snapshot = _previous_snapshot(first_raw_day)
    _, closing_snapshot = _snapshot_on_or_before(end)
    opening_stats = _snapshot_price_stats(opening_snapshot)
    closing_stats = _snapshot_price_stats(closing_snapshot)
    expected_opening = int(expected_metrics.get("opening_stock") or 0)
    expected_closing = int(expected_metrics.get("closing_stock") or 0)

    def _snapshot_total(stats, expected_count):
        nonlocal estimates_used
        if expected_count <= 0:
            return 0.0
        basis = stats.get("median") or company_price_fallback
        if stats.get("listed_count") == expected_count and stats.get("priced_count") == expected_count:
            return stats.get("sum")
        if basis is None:
            return stats.get("sum")
        estimates_used = True
        if stats.get("listed_count") == expected_count:
            return float(stats.get("sum") or 0) + basis * max(0, expected_count - int(stats.get("priced_count") or 0))
        return basis * expected_count

    opening_value = _snapshot_total(opening_stats, expected_opening)
    closing_value = _snapshot_total(closing_stats, expected_closing)
    if available_end is not None and end == pd.Timestamp(available_end).normalize():
        current_value = _business_activity_current_stock_value(company)
        if current_value is not None:
            closing_value = float(current_value)

    def _sum_daily_metric(value_key, count_key):
        values = []
        for date_key, expected in expected_daily.items():
            if int(expected.get(count_key) or 0) <= 0:
                continue
            value = (daily_values.get(date_key) or {}).get(value_key)
            if value is None:
                return None
            values.append(float(value))
        return float(sum(values)) if values else 0.0

    new_value = _sum_daily_metric("new_listing_asking_value", "new_listings")
    exit_value = _sum_daily_metric("observed_exit_asking_value", "observed_exits")
    reduction_value = _sum_daily_metric("price_reduction_value", "price_reductions")
    increase_value = _sum_daily_metric("price_increase_value", "price_increases")
    net_price_change = (increase_value - reduction_value) if increase_value is not None and reduction_value is not None else None
    event_bridge = (new_value - exit_value + net_price_change) if new_value is not None and exit_value is not None and net_price_change is not None else None
    if opening_value is None and closing_value is not None and event_bridge is not None:
        opening_value = closing_value - event_bridge; estimates_used = True
    elif closing_value is None and opening_value is not None and event_bridge is not None:
        closing_value = opening_value + event_bridge; estimates_used = True

    value_complete = opening_value is not None and closing_value is not None
    net_value_change = (closing_value - opening_value) if value_complete else event_bridge
    net_value_pct = round((net_value_change / opening_value) * 100.0, 1) if value_complete and opening_value else None

    def _event_sort_key(event):
        return (str(event.get("date") or ""), abs(float(event.get("change_amount") or event.get("price") or 0)))
    for events in event_map.values():
        events.sort(key=_event_sort_key, reverse=True)

    return {
        "available": bool(daily_values or any(event_map.values())),
        "details_complete": details_complete,
        "coverage": coverage,
        "new_listings": all_new,
        "observed_exits": all_exits,
        "price_reductions": all_reductions,
        "price_increases": all_increases,
        "daily_events": dict(daily_events),
        "daily_values": daily_values,
        "opening_stock_value": opening_value,
        "closing_stock_value": closing_value,
        "net_stock_value_change": net_value_change,
        "net_stock_value_change_pct": net_value_pct,
        "new_listing_asking_value": new_value,
        "observed_exit_asking_value": exit_value,
        "price_reduction_value": reduction_value,
        "price_increase_value": increase_value,
        "net_price_change_value": net_price_change,
        "value_movement_complete": value_complete,
        "value_estimates_used": bool(estimates_used),
    }


def _business_activity_ageing_report(company):
    """Current-stock age bands with exact days and observed pricing timeline."""
    if not company or business_stock_df is None or business_stock_df.empty:
        return {"total": 0, "buckets": []}
    work = _business_activity_company_rows(business_stock_df, company)
    if work.empty:
        return {"total": 0, "buckets": []}

    try:
        pricing_rows = (_business_inventory_pricing_recommendations(company, limit=500) or {}).get("recommendations") or []
    except Exception:
        pricing_rows = []
    pricing_by_link = {str(x.get("link") or "").strip(): x for x in pricing_rows if str(x.get("link") or "").strip()}
    try:
        ageing_links = work["Link"].dropna().astype(str).str.strip().tolist() if "Link" in work.columns else []
        history_by_link = _business_inventory_price_history(company, ageing_links)
    except Exception:
        history_by_link = {}

    reference_date = None
    try:
        activity_rows = _business_activity_company_rows(business_activity_df, company)
        if not activity_rows.empty:
            reference_date = pd.to_datetime(activity_rows["Date"], errors="coerce").max()
    except Exception:
        pass
    if reference_date is None or pd.isna(reference_date):
        reference_date = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    reference_date = pd.Timestamp(reference_date).normalize()

    # Four operational ageing bands.  Boundaries are half-open in practice:
    # 0–59, 60–119, 120–179 and 180+ days, while the customer-facing labels
    # remain the simpler 0–60 / 60–120 / 120–180 / 180+ convention.
    defs = [
        ("AGE_0_60", "0–60 days", 0, 59),
        ("AGE_60_120", "60–120 days", 60, 119),
        ("AGE_120_180", "120–180 days", 120, 179),
        ("AGE_180_PLUS", "180+ days", 180, None),
    ]
    buckets = {key: {"key": key, "label": label, "min_days": low, "max_days": high, "vehicles": []} for key,label,low,high in defs}

    for raw in work.to_dict("records"):
        public = _business_manage_row_public(raw)
        link = str(public.get("link") or "").strip()
        pricing = pricing_by_link.get(link) or public
        hist = history_by_link.get(link) or {}
        try:
            age_days = max(0, int(public.get("listing_age_days")))
        except (TypeError, ValueError):
            continue
        first_date = hist.get("first_observed_date")
        first_date_estimated = not bool(first_date)
        if not first_date:
            first_date = (reference_date - pd.Timedelta(days=age_days)).date().isoformat()
        first_price = hist.get("first_observed_price")
        if first_price is None:
            first_price = public.get("asking_price")
        item = {
            "link": link, "year": public.get("year"), "brand": public.get("brand"),
            "model": public.get("model"), "category": public.get("category"), "km": public.get("km"),
            "age_days": age_days, "age_is_lower_bound": bool(public.get("listing_age_is_lower_bound")),
            "first_observed_date": first_date, "first_observed_is_estimated": first_date_estimated,
            "first_observed_price": first_price, "current_price": public.get("asking_price"),
            "price_history": hist.get("price_history") or [],
            "price_change_count": hist.get("price_change_count") or 0,
            "price_reduction_count": hist.get("price_reduction_count") or 0,
            "price_increase_count": hist.get("price_increase_count") or 0,
            "pricing_bucket": pricing.get("pricing_bucket") or "NO_MARKET_COMPARABLES",
            "comparable_median_price": pricing.get("display_comparable_median_price") or pricing.get("comparable_median_price"),
            "comparable_count": pricing.get("comparable_count"),
            "price_vs_median_pct": pricing.get("price_vs_median_pct"),
        }
        for key,_label,low,high in defs:
            if age_days >= low and (high is None or age_days <= high):
                buckets[key]["vehicles"].append(item); break

    total = sum(len(v["vehicles"]) for v in buckets.values())
    out=[]
    for key,label,low,high in defs:
        bucket=buckets[key]
        bucket["vehicles"].sort(key=lambda x:(-int(x.get("age_days") or 0),str(x.get("brand") or ""),str(x.get("model") or "")))
        bucket["count"]=len(bucket["vehicles"])
        bucket["pct"]=round((bucket["count"]/total)*100.0,1) if total else 0.0
        out.append(bucket)
    return {"total": total, "buckets": out}

def _business_activity_period_summary(company, period="30D", custom_from=None, custom_to=None):
    """Build a deterministic, decision-oriented activity report for one gallery.

    Aggregate counts come from business_company_activity_daily.csv. Vehicle-level
    movement and asking-value changes are added only when historical observations
    can reproduce those authoritative counts.
    """
    if not company or business_activity_df is None or business_activity_df.empty:
        return {"data_available": False, "summary": {}, "daily": [], "top_days": [], "events": {}}

    work = _business_activity_company_rows(business_activity_df, company)
    if work.empty:
        return {"data_available": False, "summary": {}, "daily": [], "top_days": [], "events": {}}

    work = work.sort_values("Date").copy()
    available_start = pd.Timestamp(work["Date"].min()).normalize()
    available_end = pd.Timestamp(work["Date"].max()).normalize()
    period_key = str(period or "30D").strip().upper()

    if period_key == "CUSTOM":
        start = pd.to_datetime(custom_from, errors="coerce")
        end = pd.to_datetime(custom_to, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return {"data_available": False, "summary": {}, "daily": [], "top_days": [], "events": {}, "error": "INVALID_CUSTOM_PERIOD"}
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if start > end:
            start, end = end, start
        start = max(start, available_start)
        end = min(end, available_end)
    else:
        days = {"7D": 7, "30D": 30, "90D": 90}.get(period_key, 30)
        end = available_end
        start = max(available_start, end - pd.Timedelta(days=days - 1))

    if start > end:
        return {
            "data_available": False,
            "summary": {},
            "daily": [],
            "top_days": [],
            "events": {},
            "available_from": available_start.date().isoformat(),
            "available_to": available_end.date().isoformat(),
        }

    period_days = max(1, int((end - start).days) + 1)
    previous_end = start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=period_days - 1)

    current = work[(work["Date"] >= start) & (work["Date"] <= end)].copy()
    previous = work[(work["Date"] >= previous_start) & (work["Date"] <= previous_end)].copy()

    if current.empty:
        return {
            "data_available": False,
            "summary": {},
            "daily": [],
            "top_days": [],
            "events": {},
            "period_start": start.date().isoformat(),
            "period_end": end.date().isoformat(),
            "available_from": available_start.date().isoformat(),
            "available_to": available_end.date().isoformat(),
        }

    metric_columns = {
        "new_listings": "NewlyObservedListings",
        "observed_exits": "ObservedMarketExits",
        "price_reductions": "AskingPriceReductions",
        "price_increases": "AskingPriceIncreases",
    }

    def _sum(frame, column):
        if frame is None or frame.empty or column not in frame.columns:
            return 0
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return int(round(float(values.sum())))

    def _first_number(frame, column):
        if frame is None or frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return int(round(float(values.iloc[0]))) if not values.empty else None

    def _last_number(frame, column):
        if frame is None or frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return int(round(float(values.iloc[-1]))) if not values.empty else None

    def _aggregate(frame):
        result = {name: _sum(frame, column) for name, column in metric_columns.items()}
        result["price_changes"] = result["price_reductions"] + result["price_increases"]
        result["total_activity"] = result["new_listings"] + result["observed_exits"] + result["price_changes"]
        opening = _first_number(frame, "OpeningObservedStockCount")
        closing = _last_number(frame, "ClosingObservedStockCount")
        if opening is not None and closing is not None:
            net = closing - opening
        else:
            net = _sum(frame, "NetObservedStockChange")
        result["opening_stock"] = opening
        result["closing_stock"] = closing
        result["net_stock_change"] = int(net)
        if frame is None or frame.empty:
            result["active_days"] = 0
        else:
            event_total = sum(
                pd.to_numeric(frame[column], errors="coerce").fillna(0)
                for column in metric_columns.values()
            )
            result["active_days"] = int((event_total > 0).sum())
        return result

    current_metrics = _aggregate(current)
    previous_metrics = _aggregate(previous)

    comparisons = {}
    for key in ["new_listings", "observed_exits", "price_reductions", "price_increases", "price_changes", "total_activity"]:
        now = int(current_metrics.get(key) or 0)
        before = int(previous_metrics.get(key) or 0)
        delta = now - before
        if before > 0:
            delta_pct = round((delta / before) * 100.0, 1)
        else:
            delta_pct = None
        comparisons[key] = {
            "current": now,
            "previous": before,
            "delta": delta,
            "delta_pct": delta_pct,
            "trend": "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT",
        }

    def _optional_row_value(row, *names):
        for name in names:
            if name not in row:
                continue
            value = pd.to_numeric(row.get(name), errors="coerce")
            if pd.notna(value):
                return float(value)
        return None

    daily = []
    for row in current.sort_values("Date").to_dict("records"):
        new_listings = int(round(float(row.get("NewlyObservedListings") or 0))) if pd.notna(row.get("NewlyObservedListings")) else 0
        exits = int(round(float(row.get("ObservedMarketExits") or 0))) if pd.notna(row.get("ObservedMarketExits")) else 0
        reductions = int(round(float(row.get("AskingPriceReductions") or 0))) if pd.notna(row.get("AskingPriceReductions")) else 0
        increases = int(round(float(row.get("AskingPriceIncreases") or 0))) if pd.notna(row.get("AskingPriceIncreases")) else 0
        daily.append({
            "date": pd.Timestamp(row.get("Date")).date().isoformat(),
            "new_listings": new_listings,
            "observed_exits": exits,
            "price_reductions": reductions,
            "price_increases": increases,
            "net_stock_change": int(round(float(row.get("NetObservedStockChange") or 0))) if pd.notna(row.get("NetObservedStockChange")) else 0,
            "closing_stock": int(round(float(row.get("ClosingObservedStockCount")))) if pd.notna(row.get("ClosingObservedStockCount")) else None,
            "stock_asking_value": _optional_row_value(
                row,
                "ClosingObservedStockAskingValue", "ClosingStockAskingValue",
                "ClosingAskingValue", "ObservedStockAskingValue", "CurrentStockAskingValue",
            ),
            "opening_stock_asking_value": _optional_row_value(
                row,
                "OpeningObservedStockAskingValue", "OpeningStockAskingValue", "OpeningAskingValue",
            ),
            "new_listing_asking_value": _optional_row_value(
                row,
                "NewlyObservedListingAskingValue", "NewlyObservedListingsAskingValue",
                "NewListingAskingValue", "NewListingsAskingValue",
            ),
            "observed_exit_asking_value": _optional_row_value(
                row,
                "ObservedMarketExitAskingValue", "ObservedMarketExitsAskingValue",
                "ObservedExitAskingValue", "ObservedExitsAskingValue",
            ),
            "price_reduction_value": _optional_row_value(
                row, "AskingPriceReductionValue", "AskingPriceReductionsValue",
            ),
            "price_increase_value": _optional_row_value(
                row, "AskingPriceIncreaseValue", "AskingPriceIncreasesValue",
            ),
            "total_activity": new_listings + exits + reductions + increases,
        })

    detail = _business_activity_event_details(
        company,
        start,
        end,
        expected_metrics=current_metrics,
        daily_expected=daily,
        available_end=available_end,
    )

    # Prefer value fields produced directly by the offline daily builder.  When
    # an older daily file contains counts only, fill individual dates from raw
    # vehicle histories only when those histories reconcile to the daily count.
    reconstructed_daily_values = detail.get("daily_values") or {}
    for item in daily:
        reconstructed = reconstructed_daily_values.get(str(item.get("date") or "")) or {}
        for key in [
            "stock_asking_value", "new_listing_asking_value", "observed_exit_asking_value",
            "price_reduction_value", "price_increase_value",
        ]:
            if item.get(key) is None and reconstructed.get(key) is not None:
                item[key] = float(reconstructed.get(key))

    top_days = sorted(
        [item.copy() for item in daily if item.get("total_activity", 0) > 0],
        key=lambda item: (item.get("total_activity", 0), item.get("date", "")),
        reverse=True,
    )[:3]

    coverage = detail.get("coverage") or {}
    daily_events = detail.get("daily_events") or {}
    for item in top_days:
        date_key = str(item.get("date") or "")
        day_events = daily_events.get(date_key) or {}
        detail_payload = {}
        complete_for_day = True
        for key in ["new_listings", "observed_exits", "price_reductions", "price_increases"]:
            expected_count = int(item.get(key) or 0)
            events = list(day_events.get(key) or [])
            exact = bool(coverage.get(key)) and len(events) == expected_count
            detail_payload[key] = events if exact else []
            if expected_count > 0 and not exact:
                complete_for_day = False
        item["event_details"] = detail_payload
        item["event_details_complete"] = complete_for_day

    reductions = int(current_metrics.get("price_reductions") or 0)
    increases = int(current_metrics.get("price_increases") or 0)
    price_changes = reductions + increases
    reduction_share = round((reductions / price_changes) * 100.0, 1) if price_changes else None
    exits = int(current_metrics.get("observed_exits") or 0)
    new_listings = int(current_metrics.get("new_listings") or 0)
    inflow_per_exit = round(new_listings / exits, 2) if exits > 0 else None

    def _first_daily_value(key):
        for item in daily:
            value = pd.to_numeric(item.get(key), errors="coerce")
            if pd.notna(value):
                return float(value)
        return None

    def _last_daily_value(key):
        for item in reversed(daily):
            value = pd.to_numeric(item.get(key), errors="coerce")
            if pd.notna(value):
                return float(value)
        return None

    def _sum_daily_value(key, count_key):
        relevant = [item for item in daily if int(item.get(count_key) or 0) > 0]
        if not relevant:
            return 0.0
        values = []
        for item in relevant:
            value = pd.to_numeric(item.get(key), errors="coerce")
            if pd.isna(value):
                return None
            values.append(float(value))
        return float(sum(values))

    opening_stock_value = _first_daily_value("opening_stock_asking_value")
    if opening_stock_value is None:
        opening_stock_value = detail.get("opening_stock_value")
    closing_stock_value = _last_daily_value("stock_asking_value")
    if closing_stock_value is None:
        closing_stock_value = detail.get("closing_stock_value")
    if closing_stock_value is None and end == available_end:
        closing_stock_value = _business_activity_current_stock_value(company)

    new_listing_value = _sum_daily_value("new_listing_asking_value", "new_listings")
    if new_listing_value is None:
        new_listing_value = detail.get("new_listing_asking_value")
    exit_value = _sum_daily_value("observed_exit_asking_value", "observed_exits")
    if exit_value is None:
        exit_value = detail.get("observed_exit_asking_value")
    reduction_value = _sum_daily_value("price_reduction_value", "price_reductions")
    if reduction_value is None:
        reduction_value = detail.get("price_reduction_value")
    increase_value = _sum_daily_value("price_increase_value", "price_increases")
    if increase_value is None:
        increase_value = detail.get("price_increase_value")

    net_price_value = None
    if reduction_value is not None and increase_value is not None:
        net_price_value = float(increase_value) - float(reduction_value)
    elif detail.get("net_price_change_value") is not None:
        net_price_value = float(detail.get("net_price_change_value"))

    activity_value_estimates_used = bool(detail.get("value_estimates_used"))
    closing_count_est = int(current_metrics.get("closing_stock") or 0)
    ref_average = (float(closing_stock_value) / closing_count_est) if closing_stock_value is not None and closing_count_est > 0 else None
    if ref_average is None:
        try:
            stock_rows = _business_activity_company_rows(business_stock_df, company)
            prices = pd.to_numeric(stock_rows.get("CurrentAskingPrice"), errors="coerce").dropna() if not stock_rows.empty and "CurrentAskingPrice" in stock_rows.columns else pd.Series(dtype=float)
            prices = prices[prices > 0]
            ref_average = float(prices.median()) if not prices.empty else None
        except Exception:
            ref_average = None
    typical_change = max(250.0, ref_average * 0.03) if ref_average is not None else None
    if new_listing_value is None and new_listings > 0 and ref_average is not None:
        new_listing_value = new_listings * ref_average; activity_value_estimates_used = True
    if exit_value is None and exits > 0 and ref_average is not None:
        exit_value = exits * ref_average; activity_value_estimates_used = True
    if reduction_value is None and reductions > 0 and typical_change is not None:
        reduction_value = reductions * typical_change; activity_value_estimates_used = True
    if increase_value is None and increases > 0 and typical_change is not None:
        increase_value = increases * typical_change; activity_value_estimates_used = True
    if reduction_value is not None and increase_value is not None:
        net_price_value = float(increase_value) - float(reduction_value)
    bridge = (float(new_listing_value) - float(exit_value) + float(net_price_value)) if new_listing_value is not None and exit_value is not None and net_price_value is not None else None
    if opening_stock_value is None and closing_stock_value is not None and bridge is not None:
        opening_stock_value = float(closing_stock_value) - bridge; activity_value_estimates_used = True
    elif closing_stock_value is None and opening_stock_value is not None and bridge is not None:
        closing_stock_value = float(opening_stock_value) + bridge; activity_value_estimates_used = True

    if ref_average is not None:
        running = float(opening_stock_value) if opening_stock_value is not None else None
        for item in daily:
            if item.get("new_listing_asking_value") is None:
                item["new_listing_asking_value"] = int(item.get("new_listings") or 0) * ref_average
                activity_value_estimates_used = activity_value_estimates_used or int(item.get("new_listings") or 0) > 0
            if item.get("observed_exit_asking_value") is None:
                item["observed_exit_asking_value"] = int(item.get("observed_exits") or 0) * ref_average
                activity_value_estimates_used = activity_value_estimates_used or int(item.get("observed_exits") or 0) > 0
            if item.get("price_reduction_value") is None and typical_change is not None:
                item["price_reduction_value"] = int(item.get("price_reductions") or 0) * typical_change
                activity_value_estimates_used = activity_value_estimates_used or int(item.get("price_reductions") or 0) > 0
            if item.get("price_increase_value") is None and typical_change is not None:
                item["price_increase_value"] = int(item.get("price_increases") or 0) * typical_change
                activity_value_estimates_used = activity_value_estimates_used or int(item.get("price_increases") or 0) > 0
            if running is not None:
                running += float(item.get("new_listing_asking_value") or 0) - float(item.get("observed_exit_asking_value") or 0) - float(item.get("price_reduction_value") or 0) + float(item.get("price_increase_value") or 0)
                if item.get("stock_asking_value") is None:
                    item["stock_asking_value"] = running; activity_value_estimates_used = True
                else:
                    running = float(item.get("stock_asking_value"))

    value_movement_complete = opening_stock_value is not None and closing_stock_value is not None
    net_stock_value_change = (
        float(closing_stock_value) - float(opening_stock_value)
        if value_movement_complete else detail.get("net_stock_value_change")
    )
    net_stock_value_change_pct = None
    if value_movement_complete and opening_stock_value and abs(float(opening_stock_value)) > 0.01:
        net_stock_value_change_pct = round((float(net_stock_value_change) / float(opening_stock_value)) * 100.0, 1)

    opening_stock_count = current_metrics.get("opening_stock")
    closing_stock_count = current_metrics.get("closing_stock")
    opening_average_asking = (
        float(opening_stock_value) / int(opening_stock_count)
        if opening_stock_value is not None and opening_stock_count else None
    )
    closing_average_asking = (
        float(closing_stock_value) / int(closing_stock_count)
        if closing_stock_value is not None and closing_stock_count else None
    )
    incoming_average_asking = (
        float(new_listing_value) / new_listings if new_listing_value is not None and new_listings > 0 else None
    )
    exiting_average_asking = (
        float(exit_value) / exits if exit_value is not None and exits > 0 else None
    )

    summary = {
        **current_metrics,
        "period_key": period_key,
        "period_days": period_days,
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
        "previous_period_start": previous_start.date().isoformat(),
        "previous_period_end": previous_end.date().isoformat(),
        "previous_period_available": bool(previous_start >= available_start and not previous.empty),
        "comparisons": comparisons,
        "reduction_share_pct": reduction_share,
        "inflow_per_exit": inflow_per_exit,
        "average_daily_activity": round((current_metrics.get("total_activity") or 0) / period_days, 2),
        "available_from": available_start.date().isoformat(),
        "available_to": available_end.date().isoformat(),
        "vehicle_detail_available": bool(detail.get("available")),
        "vehicle_detail_complete": bool(detail.get("details_complete")),
        "detail_coverage": coverage,
        "opening_stock_value": opening_stock_value,
        "closing_stock_value": closing_stock_value,
        "net_stock_value_change": net_stock_value_change,
        "net_stock_value_change_pct": net_stock_value_change_pct,
        "new_listing_asking_value": new_listing_value,
        "observed_exit_asking_value": exit_value,
        "price_reduction_value": reduction_value,
        "price_increase_value": increase_value,
        "net_price_change_value": net_price_value,
        "opening_average_asking_value": opening_average_asking,
        "closing_average_asking_value": closing_average_asking,
        "new_listing_average_asking_value": incoming_average_asking,
        "observed_exit_average_asking_value": exiting_average_asking,
        "value_movement_complete": bool(value_movement_complete),
        "stock_value_series_points": int(sum(1 for item in daily if item.get("stock_asking_value") is not None)),
        "incoming_value_series_points": int(sum(1 for item in daily if item.get("new_listing_asking_value") is not None and int(item.get("new_listings") or 0) > 0)),
        "exit_value_series_points": int(sum(1 for item in daily if item.get("observed_exit_asking_value") is not None and int(item.get("observed_exits") or 0) > 0)),
        "value_estimates_used": bool(activity_value_estimates_used),
    }

    return {
        "data_available": True,
        "summary": summary,
        "daily": daily,
        "top_days": top_days,
        "events": {
            "new_listings": detail.get("new_listings") or [],
            "observed_exits": detail.get("observed_exits") or [],
            "price_reductions": detail.get("price_reductions") or [],
            "price_increases": detail.get("price_increases") or [],
        },
        "ageing": _business_activity_ageing_report(company),
    }



@app.route("/api/guided/business/activity-summary", methods=["POST"])
def api_guided_business_activity_summary():
    """Return a period-over-period operating pulse for the selected gallery."""
    try:
        data = request.get_json(silent=True) or {}
        manager = get_access_manager()
        context = _request_access_context(data)
        g.otodeger_access_context = context
        requested_business = _normalize_access_tier(data.get("access_tier") or data.get("tier")) == "BUSINESS"

        if manager.enforcement_enabled:
            if not requested_business or not context.business_entitled:
                raise BusinessAccessRequired("An authorised Business account is required")
            manager.activate_business_device(context)
            company = str(context.org_name or "").strip()
        elif context.business_entitled and requested_business and context.org_name:
            company = str(context.org_name).strip()
        else:
            company = _resolve_business_company(
                "",
                requested_company=str(data.get("business_company") or "").strip() or None,
            )

        if not company:
            return jsonify({"success": False, "error": "BUSINESS_COMPANY_REQUIRED"}), 400

        payload = _business_activity_period_summary(
            company,
            period=data.get("period") or "30D",
            custom_from=data.get("custom_from"),
            custom_to=data.get("custom_to"),
        )
        return jsonify({
            "success": True,
            "company": company,
            **payload,
        })
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("BUSINESS ACTIVITY SUMMARY FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "BUSINESS_ACTIVITY_SUMMARY_UNAVAILABLE",
            "data_available": False,
            "summary": {},
            "daily": [],
            "top_days": [],
        }), 503



@app.route("/api/guided/business/inventory-pricing", methods=["POST"])
def api_guided_business_inventory_pricing():
    """Return structured pricing priorities for the selected gallery's own stock."""
    try:
        data = request.get_json(silent=True) or {}
        manager = get_access_manager()
        context = _request_access_context(data)
        g.otodeger_access_context = context
        requested_business = _normalize_access_tier(data.get("access_tier") or data.get("tier")) == "BUSINESS"

        if manager.enforcement_enabled:
            if not requested_business or not context.business_entitled:
                raise BusinessAccessRequired("An authorised Business account is required")
            manager.activate_business_device(context)
            company = str(context.org_name or "").strip()
        elif context.business_entitled and requested_business and context.org_name:
            company = str(context.org_name).strip()
        else:
            company = _resolve_business_company(
                "",
                requested_company=str(data.get("business_company") or "").strip() or None,
            )

        if not company:
            return jsonify({
                "success": False,
                "error": "BUSINESS_COMPANY_REQUIRED",
                "recommendations": [],
                "summary": {},
            }), 400

        payload = _business_inventory_pricing_recommendations(company, limit=data.get("limit") or 500)
        recommendations = payload.get("recommendations") or []
        return jsonify({
            "success": True,
            "company": company,
            "recommendations": recommendations,
            "summary": payload.get("summary") or {},
            "count": len(recommendations),
        })
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("BUSINESS INVENTORY PRICING RECOMMENDATIONS FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "BUSINESS_INVENTORY_PRICING_UNAVAILABLE",
            "recommendations": [],
            "summary": {},
        }), 503


@app.route("/api/guided/business/ad-recommendations", methods=["POST"])
def api_guided_business_ad_recommendations():
    """Rank a gallery's own active stock for advertising allocation.

    In enforced commercial mode the organisation comes only from the verified
    entitlement. During owner-preview/shadow mode the existing gallery selector
    may supply a recognised company name for product testing.
    """
    try:
        data = request.get_json(silent=True) or {}
        manager = get_access_manager()
        context = _request_access_context(data)
        g.otodeger_access_context = context
        requested_business = _normalize_access_tier(data.get("access_tier") or data.get("tier")) == "BUSINESS"

        if manager.enforcement_enabled:
            if not requested_business or not context.business_entitled:
                raise BusinessAccessRequired("An authorised Business account is required")
            manager.activate_business_device(context)
            company = str(context.org_name or "").strip()
        elif context.business_entitled and requested_business and context.org_name:
            company = str(context.org_name).strip()
        else:
            company = _resolve_business_company(
                "",
                requested_company=str(data.get("business_company") or "").strip() or None,
            )

        if not company:
            return jsonify({"success": False, "error": "BUSINESS_COMPANY_REQUIRED", "recommendations": []}), 400

        recommendations = _business_advertising_recommendations(company, limit=data.get("limit") or 5)
        return jsonify({
            "success": True,
            "company": company,
            "recommendations": recommendations,
            "count": len(recommendations),
        })
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("BUSINESS AD RECOMMENDATIONS FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "error": "BUSINESS_AD_RECOMMENDATIONS_UNAVAILABLE", "recommendations": []}), 503


@app.route("/api/business/galleries", methods=["GET"])
def api_business_gallery_options():
    """Return gallery names that have Business intelligence coverage.

    This is used only to bind the Business UI to a real gallery dataset so
    inventory-specific reports cannot accidentally run without company context.
    It does not modify or expose the standalone valuation engine.
    """
    try:
        names = set()
        for frame in (business_company_df, business_stock_df):
            if frame is None or frame.empty or "Company" not in frame.columns:
                continue
            for value in frame["Company"].dropna().astype(str):
                name = value.strip()
                if name:
                    names.add(name)
        return jsonify({"success": True, "galleries": sorted(names, key=lambda value: value.casefold())})
    except Exception as exc:
        print("BUSINESS GALLERY OPTIONS FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "galleries": [], "error": "BUSINESS_GALLERIES_UNAVAILABLE"}), 503


@app.route("/api/access/status", methods=["GET"])
def api_access_status():
    """Safe account/entitlement snapshot for the frontend.

    No secret/session token is ever returned by this endpoint. Clerk session JWTs
    are verified from the request Authorization header, and Business access is
    derived only from server-owned entitlements.
    """
    try:
        manager = get_access_manager()
        context = _request_access_context({})
        return jsonify({"success": True, **manager.status_payload(context)})
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("ACCESS STATUS FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "ACCESS_STATUS_UNAVAILABLE"}), 503


@app.route("/api/access/business/activate", methods=["POST"])
def api_activate_business_access():
    """Register/use the current device as one Business seat.

    In paid-launch enforcement mode the company and seat allowance come only
    from the signed session.  The browser cannot choose another gallery.
    """
    try:
        data = request.get_json(silent=True) or {}
        manager = get_access_manager()
        context = _request_access_context(data)
        if manager.enforcement_enabled and not context.business_entitled:
            raise BusinessAccessRequired("An authorised Business account is required")
        if context.business_entitled:
            seat = manager.activate_business_device(context)
        else:
            # Legacy owner-testing compatibility only.
            seat = {"seat_limit": 0, "seats_used": 0, "device_registered": False}
        return jsonify({"success": True, **manager.status_payload(context), **seat})
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("BUSINESS ACTIVATE FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "BUSINESS_ACCESS_UNAVAILABLE"}), 503


@app.route("/api/access/devices", methods=["GET", "DELETE"])
def api_business_devices():
    """Organisation-admin device seat management."""
    try:
        manager = get_access_manager()
        context = _request_access_context({})
        if request.method == "GET":
            result = manager.list_business_devices(context)
        else:
            data = request.get_json(silent=True) or {}
            result = manager.deactivate_business_device(context, data.get("device_id"))
        return jsonify({"success": True, **result})
    except AccessControlError as exc:
        return _access_control_error_response(exc)
    except Exception as exc:
        print("BUSINESS DEVICES FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "BUSINESS_DEVICES_UNAVAILABLE"}), 503


@app.route("/api/assistant", methods=["POST"])
def api_v10_decision_agent():
    """Production conversational endpoint with commercial access controls.

    The 11.14 Gold decision engine remains unchanged.  This route owns the
    security boundary around it: account entitlement, Business organisation,
    device seats, usage quota, concurrency, language and data-extraction guards.
    """
    lease = None
    manager = None
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("message"), str):
            return jsonify({"success":False,"error":"INVALID_REQUEST"}),400
        message = data["message"].strip()
        requested_language = str(data.get("language") or "TR").strip().upper()
        if requested_language not in {"EN","TR","RU"}:
            requested_language = "TR"
        # Follow the language the user is actually writing when detectable;
        # browser preference remains the fallback for short/ambiguous turns.
        language = detect_conversation_language(message, requested_language, None)
        data["language"] = language

        if not message:
            return jsonify({"success": False, "error": "MESSAGE_REQUIRED"}), 400
        if len(message) > ASSISTANT_MAX_MESSAGE_CHARS:
            return jsonify({
                "success": False,
                "error": "MESSAGE_TOO_LONG",
                "max_chars": ASSISTANT_MAX_MESSAGE_CHARS,
            }), 413

        manager = get_access_manager()
        access_context = _request_access_context(data)
        g.otodeger_access_context = access_context
        requested_mode = _normalize_access_tier(data.get("access_tier") or data.get("tier"))
        requested_business = requested_mode == "BUSINESS"

        # Paid-launch invariant: Business is an entitlement, not a client mode
        # string.  Company identity comes from the verified session and cannot be
        # switched by editing JSON/localStorage.
        if manager.enforcement_enabled and requested_business:
            if not access_context.business_entitled:
                raise BusinessAccessRequired("An authorised Business account is required")
            manager.activate_business_device(access_context)
            data["access_tier"] = "BUSINESS"
            data["business_company"] = access_context.org_name
        elif access_context.business_entitled and requested_business:
            # Signed sessions take precedence even during migration/shadow mode.
            data["access_tier"] = "BUSINESS"
            data["business_company"] = access_context.org_name
        else:
            # Personal+ is a billing entitlement, not a third assistant mode.
            data["access_tier"] = requested_mode if not manager.enforcement_enabled else "PERSONAL"
            if data["access_tier"] != "BUSINESS":
                data.pop("business_company", None)

        # Shared paid-tier quotas and concurrency are enforced before any OpenAI
        # or market-analysis work.  In legacy/shadow mode these are no-ops.
        usage = manager.consume_assistant_quota(access_context, client_ip=_client_ip())
        lease = manager.acquire_concurrency(access_context, client_ip=_client_ip())

        # Keep a coarse IP limiter as an additional abuse shield.  Paid launch
        # should configure these values above any legitimate per-account tier.
        allowed, retry_after = _assistant_request_allowed()
        if not allowed:
            response = jsonify({
                "success": False,
                "error": "RATE_LIMITED",
                "retry_after_seconds": retry_after,
            })
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        # Proprietary dataset protection remains ahead of every model/tool call.
        if _looks_like_dataset_extraction_request(message):
            return jsonify({
                "success": True,
                "answer": _data_protection_answer(language),
                "conversation_id": data.get("conversation_id"),
                "state_revision": data.get("state_revision"),
                "decision_mode": "PROTECTED_DATA",
                "stage": "protected_data",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "language": language,
                "access": manager.status_payload(access_context),
                "usage": usage,
                "filters": {}, "preferences": [], "results": [],
                "model_options": [], "actions": [], "suggestions": [],
            })

        if _looks_like_gibberish_message(message):
            fallback = _fallback_support_payload(language)
            return jsonify({
                "success": True,
                "answer": _unsupported_input_answer(language),
                "conversation_id": data.get("conversation_id"),
                "state_revision": data.get("state_revision"),
                "decision_mode": "FALLBACK",
                "stage": "unsupported_input",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "language": language,
                "access": manager.status_payload(access_context),
                "usage": usage,
                "filters": {}, "preferences": [], "results": [],
                "model_options": [], "actions": fallback.get("actions") or [],
                "suggestions": fallback.get("suggestions") or [],
            })

        original_message = message
        fx_conversions = []
        try:
            message, fx_conversions = normalize_message_currency(message)
            data["message"] = message
        except FXUnavailable:
            return jsonify({
                "success": False,
                "error": "FX_RATE_TEMPORARILY_UNAVAILABLE",
                "answer": _fx_unavailable_answer(language),
                "language": language,
                "access": manager.status_payload(access_context),
                "usage": usage,
            }), 503

        # Legacy/shadow compatibility: owner testing can still provide a gallery
        # until external authentication is wired.  Enforced mode never enters
        # this branch with a client-selected company.
        if data.get("access_tier") == "BUSINESS" and not manager.enforcement_enabled:
            resolved_company = _resolve_business_company_context(
                message, requested_company=str(data.get("business_company") or "").strip() or None
            )
            if resolved_company:
                data["business_company"] = resolved_company

        # Personal own-car valuation intentionally hands off to the dedicated
        # valuation product. Offer evaluation stays in chat.
        if data.get("access_tier") != "BUSINESS" and _valuation_intent(message):
            valuation = _valuation_response(language)
            return jsonify({
                "success": True,
                "answer": valuation["answer"],
                "conversation_id": data.get("conversation_id"),
                "state_revision": data.get("state_revision"),
                "decision_mode": "VALUATION",
                "stage": "valuation_handoff",
                "assistant_build": ASSISTANT_BUILD,
                "v10_version": V10_VERSION,
                "language": language,
                "access": manager.status_payload(access_context),
                "usage": usage,
                "filters": {}, "preferences": [], "results": [],
                "model_options": [], "actions": valuation["actions"], "suggestions": [],
                "fx_conversions": [item.public_payload() for item in fx_conversions],
            })

        host = globals()
        payload, status = handle_v10_request(data, host)
        if isinstance(payload, dict):
            payload.setdefault("language", language)
            payload.setdefault("access", manager.status_payload(access_context))
            payload.setdefault("usage", usage)
            if fx_conversions:
                note = conversion_note(fx_conversions, language)
                if payload.get("success") and payload.get("answer") and note:
                    payload["answer"] = f"{note}\n\n{payload['answer']}"
                payload["fx_conversions"] = [item.public_payload() for item in fx_conversions]
        return jsonify(payload), status

    except AccessControlError as exc:
        print("COMMERCIAL ACCESS BLOCK:", getattr(exc, "code", type(exc).__name__), flush=True)
        return _access_control_error_response(exc)
    except StorageUnavailable as exc:
        print("V10 STATE STORAGE UNAVAILABLE:", exc, flush=True)
        return jsonify({
            "success": False,
            "error": "ASSISTANT_STATE_TEMPORARILY_UNAVAILABLE",
        }), 503
    except AIUsageLimitExceeded as exc:
        print("V10 AI USAGE LIMIT:", exc, flush=True)
        return jsonify({"success": False, "error": "AI_USAGE_LIMIT_REACHED"}), 429
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        print("V10 OPENAI HTTP ERROR:", status_code, flush=True)
        return jsonify({"success": False, "error": "AI_ASSISTANT_TEMPORARILY_UNAVAILABLE"}), 502
    except Exception as exc:
        print("V10 ASSISTANT FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "error": "AI_ASSISTANT_FAILED"}), 500
    finally:
        if manager is not None:
            manager.release_concurrency(lease)


# =========================================================
# GUIDED DECISION ENGINE - DISCOVERY OPTIONS (13.0 FOUNDATION)
# =========================================================

IMPORT_MAX_AGE_YEARS = int(os.environ.get("IMPORT_MAX_AGE_YEARS", "5"))


def _guided_list_arg(name):
    """Read repeated query args while tolerating comma-separated fallback."""
    values = []
    for raw in request.args.getlist(name):
        values.extend(str(raw or "").split(","))
    return [value.strip() for value in values if value and value.strip()]


# Curated overrides for brands whose marketplace taxonomy is known to be
# unreliable. These only affect the AI assistant guided-discovery index.
# Mixed-use brands (for example Honda) are intentionally NOT listed here.
_GUIDED_MOTORCYCLE_ONLY_BRAND_KEYS = {
    "aprilia", "yamaha", "rks", "triumph", "ducati", "kawasaki",
    "harleydavidson", "ktm", "husqvarna", "benelli", "vespa", "piaggio",
    "kymco", "sym", "bajaj", "royalenfield", "motoguzzi", "mvagusta",
    "cfmoto", "keeway",
}
_GUIDED_ATV_ONLY_BRAND_KEYS = {"yuki"}
_GUIDED_UNSUPPORTED_BRAND_KEYS = {"jet"}


def _guided_taxonomy_key(value):
    key = (
        str(value or "").strip().casefold().replace("i̇", "i")
        .replace("ı", "i").replace("ş", "s").replace("ç", "c")
        .replace("ğ", "g").replace("ü", "u").replace("ö", "o")
    )
    return re.sub(r"[^a-z0-9]+", "", key)


def _guided_classify_row(row):
    """Return exactly one supported vehicle class for a guided-market row.

    The source marketplace's VehicleType is authoritative for unmistakable
    classes such as motorcycles and ATVs.  Model profiles are used only to
    refine broad car/4x4 rows (for example CAR -> SUV, or ARAZI -> PICKUP).

    Keeping the classes mutually exclusive is important: selecting CAR must
    never leak MOTORCYCLE/ATV rows into subsequent Brand/Model/Category facets.
    This helper is used only by the AI assistant guided-market index.
    """
    brand = str(row.get("Brand") or "").strip()
    model = str(row.get("Model") or "").strip()
    brand_key = _guided_taxonomy_key(brand)
    profile = MODEL_PROFILE_LOOKUP.get((brand.casefold(), model.casefold()), {})

    # Brand-level overrides are evaluated before the source VehicleType because
    # several motorcycle/ATV brands have been observed with generic automobile
    # taxonomy in the source feed. Unknown/unsupported brands are excluded rather
    # than silently assumed to be cars.
    if brand_key in _GUIDED_UNSUPPORTED_BRAND_KEYS:
        return set()
    if brand_key in _GUIDED_MOTORCYCLE_ONLY_BRAND_KEYS:
        return {"MOTORCYCLE"}
    if brand_key in _GUIDED_ATV_ONLY_BRAND_KEYS:
        return {"ATV"}

    raw_vt = str(row.get("VehicleType") or "").strip()
    raw_cf = (
        raw_vt.casefold().replace("i̇", "i")
        .replace("ı", "i").replace("ş", "s").replace("ç", "c")
        .replace("ğ", "g").replace("ü", "u").replace("ö", "o")
    )
    raw_words = re.sub(r"[^a-z0-9]+", " ", raw_cf).strip()
    raw_key = raw_words.replace(" ", "")

    # Non-supported source-taxonomy rows should never fall through to CAR.
    if raw_key in {"gezi", "is", "cekici", "surat"}:
        return set()

    profile_vt = str(profile.get("VehicleType") or "").strip().upper()
    body = str(profile.get("BodyStyle") or "").strip().upper().replace("-", "_")

    # 1) Explicit source classes take precedence over model-profile defaults.
    # This prevents motorcycle/ATV rows from being labelled CAR when a generic
    # profile happens to exist for the same brand/model text.
    if re.search(r"(?:^| )(?:motosiklet|motorcycle|scooter|vespa)(?: |$)", raw_words):
        return {"MOTORCYCLE"}
    if re.search(r"(?:^| )(?:atv|utv|quad)(?: |$)", raw_words):
        return {"ATV"}
    if re.search(r"(?:^| )(?:pick ?up|pickup|kamyonet)(?: |$)", raw_words):
        return {"PICKUP"}

    # ARAZI/4x4 is a broad marketplace bucket.  Let the profile distinguish a
    # true pick-up; otherwise it belongs to SUV.  Never emit both classes.
    if re.search(r"(?:^| )(?:arazi|suv|crossover|4x4)(?: |$)", raw_words):
        if profile_vt == "PICKUP" or body == "PICKUP":
            return {"PICKUP"}
        return {"SUV"}

    # Some source rows carry an ordinary-car VehicleType even though their
    # category/link text explicitly identifies a motorcycle or ATV. Catch those
    # signals before accepting the broad automobile label.
    special_haystack = " ".join([
        str(row.get("Brand") or ""), str(row.get("Model") or ""),
        str(row.get("Category") or ""), str(row.get("CategoryDetail") or ""),
        str(row.get("Link") or ""),
    ]).casefold()
    if re.search(r"\b(?:motosiklet|motorcycle|scooter|vespa)\b", special_haystack):
        return {"MOTORCYCLE"}
    if re.search(r"\b(?:atv|utv|quad)\b", special_haystack):
        return {"ATV"}

    # 2) Refine ordinary car rows using the validated model/body profile.
    if raw_key in {"otomobil", "car", "automobile", "araba"}:
        if profile_vt == "PICKUP" or body == "PICKUP":
            return {"PICKUP"}
        if profile_vt in {"SUV", "CROSSOVER"} or body in {"SUV", "CROSSOVER"}:
            return {"SUV"}
        return {"CAR"}

    # 3) If the source type is blank/unknown, use the profile as the next-best
    # deterministic source.  Again, classification is intentionally exclusive.
    if profile_vt in {"MOTORCYCLE", "SCOOTER"} or body in {"MOTORCYCLE", "SCOOTER"}:
        return {"MOTORCYCLE"}
    if profile_vt in {"ATV", "UTV", "ATV/UTV", "ATV_UTV", "QUAD"} or body in {"ATV", "UTV", "ATV/UTV", "ATV_UTV", "QUAD"}:
        return {"ATV"}
    if profile_vt == "PICKUP" or body == "PICKUP":
        return {"PICKUP"}
    if profile_vt in {"SUV", "CROSSOVER"} or body in {"SUV", "CROSSOVER"}:
        return {"SUV"}
    if profile_vt == "CAR":
        return {"CAR"}

    # 4) Conservative text fallback for genuinely unprofiled rows.  Inspecting
    # Brand/Model/Category is useful, but special classes always win over CAR.
    haystack = " ".join([
        str(row.get("Brand") or ""), str(row.get("Model") or ""),
        str(row.get("Category") or ""), str(row.get("CategoryDetail") or ""),
        str(row.get("Link") or ""),
    ]).casefold()
    if re.search(r"\b(?:motosiklet|motorcycle|scooter|vespa)\b", haystack):
        return {"MOTORCYCLE"}
    if re.search(r"\b(?:atv|utv|quad)\b", haystack):
        return {"ATV"}
    if re.search(r"\b(?:pick[- ]?up|pickup|kamyonet)\b", haystack):
        return {"PICKUP"}
    if re.search(r"\b(?:suv|crossover|4x4|arazi)\b", haystack):
        return {"SUV"}

    # Do not guess CAR for an unknown source taxonomy. A false positive here
    # pollutes every later Brand/Model/Category facet; omission is safer until
    # the row has an explicit source type or a validated model profile.
    return set()


def _guided_type_matches(row, requested_types):
    if not requested_types or "ALL" in {str(v).upper() for v in requested_types}:
        return True
    requested = {str(v).strip().upper().replace("-", "_") for v in requested_types}
    return bool(_guided_classify_row(row) & requested)


_GUIDED_EXCLUDED_BRAND_KEYS = {"is", "cekici", "surat", "gezi"} | _GUIDED_UNSUPPORTED_BRAND_KEYS
_GUIDED_SUPPORTED_TYPES = ("CAR", "SUV", "PICKUP", "MOTORCYCLE", "ATV")
_GUIDED_INDEX_LOCK = threading.Lock()
_GUIDED_INDEX_SOURCE = None
_GUIDED_INDEX = pd.DataFrame()
_GUIDED_OPTIONS_CACHE = {}
_GUIDED_OPTIONS_CACHE_MAX = 256


def _guided_brand_key(value):
    return _guided_taxonomy_key(value)


def _guided_brand_allowed(value):
    return _guided_brand_key(value) not in _GUIDED_EXCLUDED_BRAND_KEYS


def _guided_market_index():
    """Small, pre-normalised listing index used only by the AI guided forms.

    The old selector copied and rescanned the complete market dataframe several
    times per click, including row-wise regex classification. This index performs
    that work once per market refresh; subsequent selections are vectorised.
    """
    global _GUIDED_INDEX_SOURCE, _GUIDED_INDEX, _GUIDED_OPTIONS_CACHE

    if not MARKET_READY or market_df is None or market_df.empty:
        return pd.DataFrame()

    source_key = (id(market_df), len(market_df), id(MODEL_PROFILE_LOOKUP), len(MODEL_PROFILE_LOOKUP))
    if _GUIDED_INDEX_SOURCE == source_key and _GUIDED_INDEX is not None and not _GUIDED_INDEX.empty:
        return _GUIDED_INDEX

    with _GUIDED_INDEX_LOCK:
        if _GUIDED_INDEX_SOURCE == source_key and _GUIDED_INDEX is not None and not _GUIDED_INDEX.empty:
            return _GUIDED_INDEX

        cols = [c for c in (
            "Brand", "Model", "Category", "CategoryDetail", "Year", "Price", "KM",
            "VehicleType", "Company", "Location", "Transmission", "Fuel", "Color",
            "Image", "Link"
        ) if c in market_df.columns]
        idx = market_df[cols].copy()
        if "CategoryDetail" not in idx.columns:
            idx["CategoryDetail"] = idx["Category"] if "Category" in idx.columns else ""
        if "VehicleType" not in idx.columns:
            idx["VehicleType"] = ""

        idx["Brand"] = idx.get("Brand", "").fillna("").astype(str).str.strip()
        idx["Model"] = idx.get("Model", "").fillna("").astype(str).str.strip()
        idx["CategoryDetail"] = idx.get("CategoryDetail", "").fillna("").astype(str).str.strip()
        idx["Year"] = pd.to_numeric(idx.get("Year"), errors="coerce")
        idx["Price"] = pd.to_numeric(idx.get("Price"), errors="coerce")
        if "KM" in idx.columns:
            idx["KM"] = pd.to_numeric(idx["KM"], errors="coerce")

        # Remove malformed/non-vehicle source-taxonomy brands once, not per click.
        brand_keys = idx["Brand"].map(_guided_brand_key)
        idx = idx[~brand_keys.isin(_GUIDED_EXCLUDED_BRAND_KEYS)]
        idx = idx[idx["Brand"].ne("") & idx["Model"].ne("") & idx["Year"].notna() & idx["Price"].notna()]

        # Classification is the only row-wise operation and is paid once per
        # dataset refresh. Every request after this uses boolean columns.
        type_sets = idx.apply(_guided_classify_row, axis=1)
        for vehicle_type in _GUIDED_SUPPORTED_TYPES:
            idx[f"_type_{vehicle_type}"] = type_sets.map(lambda values, t=vehicle_type: t in values)

        idx["_brand_cf"] = idx["Brand"].str.casefold()
        idx["_model_cf"] = idx["Model"].str.casefold()
        idx["_model_key"] = idx["Brand"] + "||" + idx["Model"]

        _GUIDED_INDEX = idx.reset_index(drop=True)
        _GUIDED_INDEX_SOURCE = source_key
        _GUIDED_OPTIONS_CACHE = {}
        print(f"Guided market index rebuilt: {len(_GUIDED_INDEX)} listings", flush=True)
        return _GUIDED_INDEX


_GUIDED_LISTING_PRICE_RANGES = {
    "0_10000": (0.0, 10000.0),
    "10000_20000": (10000.0, 20000.0),
    "20000_30000": (20000.0, 30000.0),
    "30000_40000": (30000.0, 40000.0),
    "40000_50000": (40000.0, 50000.0),
    "50000_60000": (50000.0, 60000.0),
    "60000_70000": (60000.0, 70000.0),
    "70000_80000": (70000.0, 80000.0),
    "80000_90000": (80000.0, 90000.0),
    "90000_100000": (90000.0, 100000.0),
    "100000_PLUS": (100000.0, None),
}


def _guided_safe_text(value):
    """Return a JSON-safe display string for scalar marketplace values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "<na>"} else text


def _guided_normalize_listing_price_ranges(values):
    if values in [None, ""]:
        return tuple()
    if isinstance(values, str):
        values = [values]
    normalized = []
    seen = set()
    for raw in values or []:
        key = str(raw or "").strip().upper()
        if not key or key in {"__ALL__", "ALL"}:
            continue
        if key not in _GUIDED_LISTING_PRICE_RANGES or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return tuple(normalized)


def _guided_listing_price_range_label(key):
    bounds = _GUIDED_LISTING_PRICE_RANGES.get(str(key or "").strip().upper())
    if not bounds:
        return None
    low, high = bounds
    if high is None:
        return f"£{int(low / 1000)}K+"
    return f"£{int(low / 1000)}–{int(high / 1000)}K"


def _guided_listing_price_mask(index, price_ranges):
    normalized = _guided_normalize_listing_price_ranges(price_ranges)
    if not normalized:
        return pd.Series(True, index=index.index)
    mask = pd.Series(False, index=index.index)
    prices = pd.to_numeric(index["Price"], errors="coerce")
    for key in normalized:
        low, high = _GUIDED_LISTING_PRICE_RANGES[key]
        band = prices.ge(float(low))
        if high is not None:
            band &= prices.lt(float(high))
        mask |= band
    return mask


def _guided_index_filter(index, *, min_year=None, vehicle_types=None, brands=None, model_keys=None, categories=None, max_budget=None, price_ranges=None):
    if index is None or index.empty:
        return index.iloc[0:0] if isinstance(index, pd.DataFrame) else pd.DataFrame()

    mask = pd.Series(True, index=index.index)
    if min_year is not None:
        mask &= index["Year"].ge(int(min_year))
    if max_budget is not None:
        mask &= index["Price"].le(float(max_budget))
    if price_ranges:
        mask &= _guided_listing_price_mask(index, price_ranges)
    if vehicle_types:
        type_mask = pd.Series(False, index=index.index)
        for vehicle_type in vehicle_types:
            col = f"_type_{str(vehicle_type).strip().upper().replace('-', '_')}"
            if col in index.columns:
                type_mask |= index[col]
        mask &= type_mask
    if brands:
        wanted = {str(value).strip().casefold() for value in brands if str(value).strip()}
        mask &= index["_brand_cf"].isin(wanted)
    if model_keys:
        wanted_models = {str(value).strip().casefold() for value in model_keys if str(value).strip()}
        mask &= index["_model_key"].str.casefold().isin(wanted_models)
    if categories:
        wanted_categories = {str(value).strip().casefold() for value in categories if str(value).strip()}
        mask &= index["CategoryDetail"].str.casefold().isin(wanted_categories)
    return index.loc[mask]


def _guided_discovery_sources():
    """Compatibility helper for the guided assistant only."""
    idx = _guided_market_index()
    return idx, idx


def _guided_cache_get(key):
    with _GUIDED_INDEX_LOCK:
        value = _GUIDED_OPTIONS_CACHE.get(key)
        return dict(value) if isinstance(value, dict) else None


def _guided_cache_put(key, value):
    with _GUIDED_INDEX_LOCK:
        if len(_GUIDED_OPTIONS_CACHE) >= _GUIDED_OPTIONS_CACHE_MAX:
            # Query combinations are tiny; FIFO eviction is sufficient here.
            try:
                _GUIDED_OPTIONS_CACHE.pop(next(iter(_GUIDED_OPTIONS_CACHE)))
            except StopIteration:
                pass
        _GUIDED_OPTIONS_CACHE[key] = dict(value)


# Build the small guided index during application start rather than making the
# first user click pay the preprocessing cost. This touches only market_df used by
# the AI assistant and does not mutate the valuation dataframe/routes.
try:
    _guided_market_index()
except Exception as _guided_warm_error:
    print("Guided market index warmup deferred:", repr(_guided_warm_error), flush=True)


@app.route("/api/guided/discovery/options", methods=["GET"])
def api_guided_discovery_options():
    """Fast deterministic option facets for the AI assistant guided forms only."""
    index = _guided_market_index()
    if index is None or index.empty:
        return jsonify({"success": False, "error": "GUIDED_OPTIONS_NOT_READY"}), 503

    try:
        requested_types = tuple(_guided_list_arg("vehicle_type"))
        selected_brands = tuple(_guided_list_arg("brand"))
        selected_model_keys = tuple(_guided_list_arg("model_key"))
        categories_only = str(request.args.get("categories_only") or "false").strip().casefold() in {
            "1", "true", "yes", "on"
        }
        all_models = str(request.args.get("all_models") or "false").strip().casefold() in {
            "1", "true", "yes", "on"
        }

        min_year_raw = str(request.args.get("min_year") or "").strip()
        min_year = None
        if min_year_raw and min_year_raw.upper() != "ALL":
            min_year = int(float(min_year_raw))

        max_budget_raw = str(request.args.get("max_budget") or "").strip()
        max_budget = None
        if max_budget_raw:
            max_budget = float(max_budget_raw)
            if not math.isfinite(max_budget) or max_budget <= 0:
                raise ValueError("INVALID_MAX_BUDGET")

        listing_price_ranges = _guided_normalize_listing_price_ranges(_guided_list_arg("price_range"))

        import_only = str(request.args.get("import_only") or "false").strip().casefold() in {
            "1", "true", "yes", "on"
        }
        import_floor = datetime.now(timezone.utc).year - max(0, IMPORT_MAX_AGE_YEARS)
        if import_only:
            min_year = max(min_year or import_floor, import_floor)

        cache_key = (
            _GUIDED_INDEX_SOURCE, categories_only, all_models,
            requested_types, selected_brands, selected_model_keys,
            min_year, max_budget, listing_price_ranges, import_only,
        )
        cached = _guided_cache_get(cache_key)
        if cached is not None:
            return jsonify(cached)

        # First apply the hard constraints that precede Brand/Model selection.
        base = _guided_index_filter(
            index,
            min_year=min_year,
            vehicle_types=requested_types,
            max_budget=max_budget,
            price_ranges=listing_price_ranges,
        )

        if categories_only:
            effective_model_keys = set(selected_model_keys)
            if all_models:
                all_model_rows = _guided_index_filter(base, brands=selected_brands)
                effective_model_keys.update(all_model_rows["_model_key"].dropna().astype(str).unique().tolist())

            model_rows = _guided_index_filter(base, brands=selected_brands, model_keys=effective_model_keys)
            categories_by_model = {}
            match_count_by_model = {}
            for model_key in sorted(effective_model_keys):
                rows = model_rows[model_rows["_model_key"].str.casefold() == str(model_key).casefold()]
                match_count_by_model[model_key] = int(len(rows))
                if rows.empty:
                    categories_by_model[model_key] = []
                    continue
                values = rows["CategoryDetail"].dropna().astype(str).str.strip()
                values = values[(values != "") & (values.str.casefold() != "nan") & (values != "-")]
                categories_by_model[model_key] = sorted(values.unique().tolist())

            payload = {
                "success": True,
                "effective_min_year": min_year,
                "effective_max_budget": max_budget,
                "effective_listing_price_ranges": list(listing_price_ranges),
                "import_only": import_only,
                "import_max_age_years": IMPORT_MAX_AGE_YEARS,
                "categories_by_model": categories_by_model,
                "match_count_by_model": match_count_by_model,
                "total_matches": int(len(model_rows)),
            }
            _guided_cache_put(cache_key, payload)
            return jsonify(payload)

        # Vehicle-type facets come from the already budget/year-constrained rows.
        type_scope = _guided_index_filter(index, min_year=min_year, brands=selected_brands, max_budget=max_budget, price_ranges=listing_price_ranges)
        available_vehicle_types = [
            vehicle_type for vehicle_type in _GUIDED_SUPPORTED_TYPES
            if f"_type_{vehicle_type}" in type_scope.columns and bool(type_scope[f"_type_{vehicle_type}"].any())
        ]

        viable_years = sorted(base["Year"].dropna().astype(int).unique().tolist(), reverse=True)
        brands = sorted(base["Brand"].dropna().astype(str).str.strip().loc[lambda x: x != ""].unique().tolist())

        model_frame = _guided_index_filter(base, brands=selected_brands)
        model_rows = model_frame[["Brand", "Model", "_model_key"]].drop_duplicates().sort_values(["Brand", "Model"])
        models = []
        for row in model_rows.to_dict("records"):
            brand = str(row.get("Brand") or "").strip()
            model = str(row.get("Model") or "").strip()
            key = str(row.get("_model_key") or "").strip()
            if not brand or not model or not key:
                continue
            models.append({"key": key, "brand": brand, "model": model, "label": f"{brand} {model}"})

        payload = {
            "success": True,
            "min_year": int(index["Year"].min()) if index["Year"].notna().any() else None,
            "max_year": int(index["Year"].max()) if index["Year"].notna().any() else None,
            "effective_min_year": min_year,
            "effective_max_budget": max_budget,
            "effective_listing_price_ranges": list(listing_price_ranges),
            "import_only": import_only,
            "import_max_age_years": IMPORT_MAX_AGE_YEARS,
            "vehicle_types": available_vehicle_types,
            "years": viable_years,
            "brands": brands,
            "models": models,
            "categories_by_model": {},
            "total_matches": int(len(model_frame if selected_brands else base)),
        }
        _guided_cache_put(cache_key, payload)
        return jsonify(payload)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "INVALID_GUIDED_OPTIONS"}), 400
    except Exception as exc:
        print("GUIDED DISCOVERY OPTIONS FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "GUIDED_OPTIONS_UNAVAILABLE"}), 503


def _guided_apply_category_keys(rows, category_keys):
    """Apply per-model category selections produced by the guided UI."""
    if rows is None or rows.empty or not category_keys:
        return rows
    keys = [str(value) for value in category_keys if value not in [None, "", "__ALL__"]]
    if not keys:
        return rows

    mask = pd.Series(False, index=rows.index)
    for key in keys:
        parts = key.split("||")
        if len(parts) < 3:
            continue
        brand, model = parts[0].strip(), parts[1].strip()
        category = "||".join(parts[2:]).strip()
        model_mask = (
            rows["_brand_cf"].eq(brand.casefold())
            & rows["_model_cf"].eq(model.casefold())
        )
        if category and category != "__ALL__":
            model_mask &= rows["CategoryDetail"].str.casefold().eq(category.casefold())
        mask |= model_mask
    return rows.loc[mask]



def _guided_stock_history_frame():
    """Prepare normalized daily listing history for stock-purchase analysis.

    The stock recommender must measure turnover at the exact visible
    Year + Brand + Model + Category grain.  `ads_base.csv` is a compact valuation
    snapshot and cannot reliably reconstruct listing lifetimes, so prefer the raw
    daily listing history already loaded for Business Activity.  The valuation
    frame remains only a backwards-compatible fallback for deployments where it
    genuinely contains repeated daily observations.
    """
    global _GUIDED_STOCK_HISTORY_CACHE

    raw_history = globals().get("business_activity_history_df")
    raw_ready = bool(globals().get("BUSINESS_ACTIVITY_HISTORY_READY"))
    source_name = "daily_listing_history"
    source = raw_history if raw_ready and isinstance(raw_history, pd.DataFrame) and not raw_history.empty else None

    if source is None:
        source_name = "valuation_snapshot_fallback"
        source = df if isinstance(df, pd.DataFrame) and not df.empty else None

    if source is None or source.empty:
        return None

    lower = {str(col).strip().casefold(): col for col in source.columns}
    def col(*names):
        for name in names:
            if name in source.columns:
                return name
            found = lower.get(str(name).strip().casefold())
            if found:
                return found
        return None

    date_col = col("DATE", "Date")
    link_col = col("Link", "URL", "Url")
    brand_col = col("Brand")
    model_col = col("Model")
    category_col = col("Category", "CategoryDetail")
    year_col = col("Year")
    price_col = col("Price", "CurrentAskingPrice")
    if not all([date_col, link_col, brand_col, model_col, category_col, year_col]):
        return None

    dates = pd.to_datetime(source[date_col], errors="coerce")
    latest = dates.max()
    earliest = dates.min()
    if pd.isna(latest) or pd.isna(earliest):
        return None

    stamp = (source_name, len(source), str(pd.Timestamp(latest)), str(pd.Timestamp(earliest)))
    cached = globals().get("_GUIDED_STOCK_HISTORY_CACHE")
    if isinstance(cached, dict) and cached.get("stamp") == stamp:
        return cached.get("payload")

    columns = [date_col, link_col, brand_col, model_col, category_col, year_col]
    if price_col:
        columns.append(price_col)
    work = source[columns].copy()
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
    work["_link"] = work[link_col].fillna("").astype(str).str.strip()
    work["_brand"] = work[brand_col].fillna("").astype(str).str.strip().str.casefold()
    work["_model"] = work[model_col].fillna("").astype(str).str.strip().str.casefold()
    work["_category"] = work[category_col].fillna("").astype(str).str.strip().str.casefold()
    work["_year"] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work["_price"] = pd.to_numeric(work[price_col], errors="coerce") if price_col else float("nan")
    work = work[
        work["_date"].notna()
        & work["_link"].ne("")
        & work["_brand"].ne("")
        & work["_model"].ne("")
        & work["_year"].notna()
    ][["_date", "_link", "_brand", "_model", "_category", "_year", "_price"]].copy()

    # A compact one-row-per-listing snapshot cannot support observed exit-time or
    # 60-day turnover statistics.  Refuse to manufacture historical metrics from it.
    if source_name == "valuation_snapshot_fallback" and not work.empty:
        observations_per_link = work.groupby("_link", sort=False)["_date"].nunique()
        if observations_per_link.empty or int(observations_per_link.max()) <= 1:
            return None

    payload = {
        "frame": work,
        "earliest": pd.Timestamp(earliest).normalize(),
        "latest": pd.Timestamp(latest).normalize(),
        "metrics": {},
    }
    _GUIDED_STOCK_HISTORY_CACHE = {"stamp": stamp, "payload": payload}
    return payload



def _guided_stock_history_metrics_bulk(current_groups):
    """Precompute turnover metrics for all current stock candidates in one pass.

    The older implementation filtered and regrouped the full daily listing-history
    frame once for every candidate (and often a second time for the same-year
    model fallback). Broad dealer selections can produce dozens of exact
    Year+Brand+Model+Category candidates, turning one report request into many
    repeated scans of the same large dataframe.

    This routine narrows the history once to the model-years actually present in
    the current shortlist universe, reduces it to one row per distinct listing,
    and fills the existing metrics cache for both exact category-year and
    same-year model scopes. Subsequent `_guided_stock_history_metrics` lookups are
    therefore O(1) for this request.
    """
    prepared = _guided_stock_history_frame()
    if not prepared or not current_groups:
        return

    work = prepared.get("frame")
    if work is None or work.empty:
        return

    wanted_model_year = set()
    wanted_exact = set()
    for item in current_groups:
        brand = str(item.get("brand") or "").strip().casefold()
        model = str(item.get("model") or "").strip().casefold()
        category = str(item.get("category") or "").strip().casefold()
        try:
            year = int(float(item.get("year")))
        except (TypeError, ValueError):
            continue
        if not brand or not model:
            continue
        wanted_model_year.add((brand, model, year))
        if category:
            wanted_exact.add((brand, model, category, year))

    if not wanted_model_year:
        return

    # Filter once. Constructing a temporary MultiIndex is materially faster than
    # N dataframe scans for broad selections and does not mutate the cached frame.
    model_index = pd.MultiIndex.from_arrays(
        [work["_brand"], work["_model"], work["_year"]],
        names=["brand", "model", "year"],
    )
    wanted_index = pd.MultiIndex.from_tuples(
        list(wanted_model_year), names=["brand", "model", "year"]
    )
    subset = work.loc[model_index.isin(wanted_index)].copy()
    if subset.empty:
        return

    listing_cols = ["_brand", "_model", "_year", "_category", "_link"]
    subset = (
        subset.sort_values(listing_cols + ["_date"], kind="stable")
        .drop_duplicates(subset=listing_cols + ["_date"], keep="last")
    )
    subset["_price_numeric"] = pd.to_numeric(subset["_price"], errors="coerce")
    subset["_price_diff"] = subset.groupby(listing_cols, sort=False)["_price_numeric"].diff()
    subset["_price_reduced_step"] = subset["_price_diff"].lt(-0.5)

    # Reduce the daily history to one row per exact listing before computing any
    # candidate statistics. This is usually orders of magnitude smaller.
    listing_summary = (
        subset.groupby(listing_cols, sort=False, dropna=False)
        .agg(
            first_seen=("_date", "min"),
            last_seen=("_date", "max"),
            valid_price_count=("_price_numeric", "count"),
            had_reduction=("_price_reduced_step", "max"),
        )
        .reset_index()
    )
    if listing_summary.empty:
        return

    dataset_first = prepared["earliest"]
    dataset_last = prepared["latest"]
    listing_summary["left_censored"] = listing_summary["first_seen"].le(dataset_first)
    listing_summary["exited"] = listing_summary["last_seen"].lt(dataset_last)
    listing_summary["observed_days"] = (
        (listing_summary["last_seen"] - listing_summary["first_seen"]).dt.days + 1
    ).clip(lower=1)
    listing_summary["current_observed_age"] = (
        (dataset_last - listing_summary["first_seen"]).dt.days + 1
    ).clip(lower=1)
    listing_summary["turnover_eligible"] = (
        ~listing_summary["left_censored"]
        & (
            listing_summary["exited"]
            | listing_summary["current_observed_age"].ge(60)
        )
    )
    listing_summary["exit60_success"] = (
        listing_summary["turnover_eligible"]
        & listing_summary["exited"]
        & listing_summary["observed_days"].le(60)
    )
    listing_summary["exit_duration_eligible"] = (
        ~listing_summary["left_censored"] & listing_summary["exited"]
    )
    listing_summary["price_pressure_eligible"] = listing_summary["valid_price_count"].ge(2)
    listing_summary["had_reduction"] = listing_summary["had_reduction"].fillna(False).astype(bool)

    def summarize(group):
        distinct = int(len(group))
        turnover_sample = int(group["turnover_eligible"].sum())
        exits = group.loc[group["exit_duration_eligible"], "observed_days"]
        exit_count = int(len(exits))
        median_exit = float(exits.median()) if exit_count else None
        success_count = int(group["exit60_success"].sum())
        exit60_rate = (float(success_count) / float(turnover_sample)) if turnover_sample else None
        price_eligible = int(group["price_pressure_eligible"].sum())
        reduced = int((group["price_pressure_eligible"] & group["had_reduction"]).sum())
        reduction_rate = (float(reduced) / float(price_eligible)) if price_eligible else None
        turnover_sufficient = bool(
            turnover_sample >= 5
            and exit_count >= 3
            and exit60_rate is not None
            and median_exit is not None
        )
        quality = (
            "HIGH" if turnover_sample >= 30 and exit_count >= 20
            else "MEDIUM" if turnover_sample >= 15 and exit_count >= 10
            else "LOW" if turnover_sufficient
            else "INSUFFICIENT"
        )
        return {
            "historical_distinct_listings": distinct,
            "turnover_sample_size": turnover_sample,
            "observed_exit_count": exit_count,
            "median_observed_days_to_exit": median_exit,
            "observed_exit_within_60_days_rate": exit60_rate,
            "historical_price_reduction_rate": reduction_rate,
            "price_pressure_sample_size": price_eligible,
            "evidence_quality": quality,
            "turnover_sufficient": turnover_sufficient,
            "history_start_date": dataset_first.date().isoformat(),
            "history_end_date": dataset_last.date().isoformat(),
            "evidence_source": "RAW_DAILY_HISTORY",
        }

    # Exact Year + Brand + Model + Category cache.
    for key_values, group in listing_summary.groupby(
        ["_brand", "_model", "_category", "_year"], sort=False, dropna=False
    ):
        brand, model, category, year = key_values
        exact_key = (str(brand), str(model), str(category), int(year))
        if exact_key not in wanted_exact:
            continue
        prepared["metrics"][(str(brand), str(model), str(category), int(year))] = summarize(group)

    # Same-year Brand + Model fallback cache. Collapse category splits so a listing
    # is counted once even if its category label changed between observations.
    model_listing_summary = (
        listing_summary.groupby(["_brand", "_model", "_year", "_link"], sort=False, dropna=False)
        .agg(
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
            valid_price_count=("valid_price_count", "sum"),
            had_reduction=("had_reduction", "max"),
        )
        .reset_index()
    )
    model_listing_summary["left_censored"] = model_listing_summary["first_seen"].le(dataset_first)
    model_listing_summary["exited"] = model_listing_summary["last_seen"].lt(dataset_last)
    model_listing_summary["observed_days"] = (
        (model_listing_summary["last_seen"] - model_listing_summary["first_seen"]).dt.days + 1
    ).clip(lower=1)
    model_listing_summary["current_observed_age"] = (
        (dataset_last - model_listing_summary["first_seen"]).dt.days + 1
    ).clip(lower=1)
    model_listing_summary["turnover_eligible"] = (
        ~model_listing_summary["left_censored"]
        & (model_listing_summary["exited"] | model_listing_summary["current_observed_age"].ge(60))
    )
    model_listing_summary["exit60_success"] = (
        model_listing_summary["turnover_eligible"]
        & model_listing_summary["exited"]
        & model_listing_summary["observed_days"].le(60)
    )
    model_listing_summary["exit_duration_eligible"] = (
        ~model_listing_summary["left_censored"] & model_listing_summary["exited"]
    )
    model_listing_summary["price_pressure_eligible"] = model_listing_summary["valid_price_count"].ge(2)
    model_listing_summary["had_reduction"] = model_listing_summary["had_reduction"].fillna(False).astype(bool)

    for key_values, group in model_listing_summary.groupby(
        ["_brand", "_model", "_year"], sort=False, dropna=False
    ):
        brand, model, year = key_values
        model_key = (str(brand), str(model), int(year))
        if model_key not in wanted_model_year:
            continue
        prepared["metrics"][(str(brand), str(model), "__MODEL_YEAR__", int(year))] = summarize(group)


def _guided_stock_history_metrics(brand, model, year, category=None):
    """Compute observed turnover statistics from distinct listing histories.

    When category is supplied the sample is exact Year + Brand + Model + Category.
    With category=None it is the same-year Brand + Model fallback across categories.
    No cross-year fallback is used: model behaviour can change materially by year.
    """
    prepared = _guided_stock_history_frame()
    if not prepared:
        return None

    brand_key = str(brand or "").strip().casefold()
    model_key = str(model or "").strip().casefold()
    category_key = str(category or "").strip().casefold() if category not in [None, ""] else None
    try:
        year_key = int(float(year))
    except (TypeError, ValueError):
        return None

    cache_key = (brand_key, model_key, category_key or "__MODEL_YEAR__", year_key)
    if cache_key in prepared["metrics"]:
        return prepared["metrics"][cache_key]

    work = prepared["frame"]
    subset = work[
        work["_brand"].eq(brand_key)
        & work["_model"].eq(model_key)
        & work["_year"].eq(year_key)
    ]
    if category_key is not None:
        subset = subset[subset["_category"].eq(category_key)]
    if subset.empty:
        prepared["metrics"][cache_key] = None
        return None

    dataset_first = prepared["earliest"]
    dataset_last = prepared["latest"]
    distinct_listings = 0
    exited_durations = []
    exit60_eligible = 0
    exit60_success = 0
    price_pressure_eligible = 0
    reduced_listings = 0

    for _link, group in subset.groupby("_link", sort=False):
        ordered = group.sort_values("_date", kind="stable").drop_duplicates(subset=["_date"], keep="last")
        if ordered.empty:
            continue
        distinct_listings += 1
        first_seen = pd.Timestamp(ordered.iloc[0]["_date"]).normalize()
        last_seen = pd.Timestamp(ordered.iloc[-1]["_date"]).normalize()
        left_censored = first_seen <= dataset_first
        exited = last_seen < dataset_last
        observed_days = max(1, int((last_seen - first_seen).days) + 1)

        # Turnover statistics exclude listings already present on the first day of
        # the tracking window because their true starting age is unknown.
        if not left_censored:
            if exited:
                exited_durations.append(observed_days)
                exit60_eligible += 1
                if observed_days <= 60:
                    exit60_success += 1
            else:
                current_observed_age = max(1, int((dataset_last - first_seen).days) + 1)
                if current_observed_age >= 60:
                    exit60_eligible += 1

        prices = pd.to_numeric(ordered["_price"], errors="coerce").dropna().tolist()
        if len(prices) >= 2:
            price_pressure_eligible += 1
            if any(float(prices[i]) < float(prices[i - 1]) - 0.5 for i in range(1, len(prices))):
                reduced_listings += 1

    median_exit = float(pd.Series(exited_durations, dtype="float64").median()) if exited_durations else None
    exit60_rate = (float(exit60_success) / float(exit60_eligible)) if exit60_eligible else None
    reduction_rate = (float(reduced_listings) / float(price_pressure_eligible)) if price_pressure_eligible else None

    # A fallback is needed only when the exact category-year turnover sample is
    # genuinely too small to support a useful liquidity read. Keep the threshold
    # deliberately modest so exact year/variant evidence is preferred whenever it
    # can reasonably stand on its own.
    turnover_sufficient = bool(
        exit60_eligible >= 5
        and len(exited_durations) >= 3
        and exit60_rate is not None
        and median_exit is not None
    )
    evidence_quality = (
        "HIGH" if exit60_eligible >= 30 and len(exited_durations) >= 20
        else "MEDIUM" if exit60_eligible >= 15 and len(exited_durations) >= 10
        else "LOW" if turnover_sufficient
        else "INSUFFICIENT"
    )

    result = {
        "historical_distinct_listings": int(distinct_listings),
        "turnover_sample_size": int(exit60_eligible),
        "observed_exit_count": int(len(exited_durations)),
        "median_observed_days_to_exit": median_exit,
        "observed_exit_within_60_days_rate": exit60_rate,
        "historical_price_reduction_rate": reduction_rate,
        "price_pressure_sample_size": int(price_pressure_eligible),
        "evidence_quality": evidence_quality,
        "turnover_sufficient": turnover_sufficient,
        "history_start_date": dataset_first.date().isoformat(),
        "history_end_date": dataset_last.date().isoformat(),
    }
    prepared["metrics"][cache_key] = result
    return result


def _guided_precomputed_stock_history_metrics(brand, model, year, category=None):
    """Fallback turnover evidence from the precomputed Business market table.

    This is used only when the raw daily listing history is unavailable or cannot
    support the requested same-year scope.  Granularity remains year-specific:
    CATEGORY_YEAR for an exact variant, or MODEL_YEAR across categories.  We never
    pool different years for guided stock recommendations.
    """
    if not BUSINESS_INTELLIGENCE_READY or business_market_df is None or business_market_df.empty:
        return None

    try:
        year_key = int(float(year))
    except (TypeError, ValueError):
        return None

    work = business_market_df.copy()
    if not {"BusinessGranularity", "Brand", "Model", "Year"}.issubset(work.columns):
        return None

    brand_key = str(brand or "").strip().casefold()
    model_key = str(model or "").strip().casefold()
    work = work[
        work["Brand"].fillna("").astype(str).str.strip().str.casefold().eq(brand_key)
        & work["Model"].fillna("").astype(str).str.strip().str.casefold().eq(model_key)
        & pd.to_numeric(work["Year"], errors="coerce").eq(year_key)
    ].copy()
    if work.empty:
        return None

    if category not in [None, ""]:
        category_key = str(category).strip().casefold()
        work = work[work["BusinessGranularity"].fillna("").astype(str).str.upper().eq("CATEGORY_YEAR")].copy()
        if "CategoryDetail" not in work.columns:
            return None
        work = work[work["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold().eq(category_key)].copy()
    else:
        work = work[work["BusinessGranularity"].fillna("").astype(str).str.upper().eq("MODEL_YEAR")].copy()

    if work.empty:
        return None

    evidence_rank = {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "INSUFFICIENT": 0}
    work["_evidence_rank"] = work.get("EvidenceQuality", pd.Series(index=work.index, dtype=object)).fillna("").astype(str).str.upper().map(evidence_rank).fillna(1)
    if "Exit60EligibleListings" in work.columns:
        work["_sample_rank"] = pd.to_numeric(work["Exit60EligibleListings"], errors="coerce").fillna(0)
    elif "HistoricalDistinctListings" in work.columns:
        work["_sample_rank"] = pd.to_numeric(work["HistoricalDistinctListings"], errors="coerce").fillna(0)
    else:
        work["_sample_rank"] = 0
    row = work.sort_values(["_evidence_rank", "_sample_rank"], ascending=[False, False]).iloc[0]

    def num(name):
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    exit60_rate = num("ObservedExitWithin60DaysRate")
    if exit60_rate is not None and exit60_rate > 1.0 and exit60_rate <= 100.0:
        exit60_rate /= 100.0
    reduction_rate = num("PriceReductionRate")
    if reduction_rate is not None and reduction_rate > 1.0 and reduction_rate <= 100.0:
        reduction_rate /= 100.0

    sample = num("Exit60EligibleListings")
    if sample is None:
        sample = num("HistoricalDistinctListings")
    sample = int(sample or 0)
    exits = num("HistoricalObservedMarketExits")
    median_exit = num("MedianObservedDaysToExit")
    hist_distinct = num("HistoricalDistinctListings")
    quality = str(row.get("EvidenceQuality") or "").strip().upper() or "LOW"
    usable = bool(sample >= 3 and exit60_rate is not None and median_exit is not None)

    return {
        "historical_distinct_listings": int(hist_distinct) if hist_distinct is not None else None,
        "turnover_sample_size": sample,
        "observed_exit_count": int(exits) if exits is not None else None,
        "median_observed_days_to_exit": median_exit,
        "observed_exit_within_60_days_rate": exit60_rate,
        "historical_price_reduction_rate": reduction_rate,
        "price_pressure_sample_size": int(num("PricePressureEligibleListings") or 0),
        "evidence_quality": quality if quality != "INSUFFICIENT" or usable else "INSUFFICIENT",
        "turnover_sufficient": usable and quality != "INSUFFICIENT",
        "history_start_date": None,
        "history_end_date": None,
        "evidence_source": "PRECOMPUTED_BUSINESS_MARKET",
    }


def _guided_stock_metrics_usable(metrics, minimum_sample=3):
    if not isinstance(metrics, dict):
        return False
    sample = pd.to_numeric(pd.Series([metrics.get("turnover_sample_size")]), errors="coerce").iloc[0]
    rate = pd.to_numeric(pd.Series([metrics.get("observed_exit_within_60_days_rate")]), errors="coerce").iloc[0]
    days = pd.to_numeric(pd.Series([metrics.get("median_observed_days_to_exit")]), errors="coerce").iloc[0]
    return bool(pd.notna(sample) and float(sample) >= float(minimum_sample) and pd.notna(rate) and pd.notna(days))



def _guided_stock_current_market_fallback(rows, limit=5, listing_price_ranges=None, max_listing_price=None):
    """Return exact current-market stock candidates when turnover evidence fails.

    This is a resilience path, not a substitute for historical ranking.  The
    the dealer's maximum desired advertised listing price determines which exact
    Year+Brand+Model+Category combinations qualify. Once qualified, current
    competition/pricing is measured against ALL live listings for that exact
    combination. This ceiling is a retail-market filter, not acquisition spend.
    No turnover, demand or liquidity claim is manufactured.
    """
    if rows is None or rows.empty:
        return []

    work = rows.copy()
    required = {"Brand", "Model", "CategoryDetail", "Year", "Price"}
    if not required.issubset(work.columns):
        return []

    work["_brand_key"] = work["Brand"].fillna("").astype(str).str.strip().str.casefold()
    work["_model_key2"] = work["Model"].fillna("").astype(str).str.strip().str.casefold()
    work["_category_key"] = work["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold()
    work["_year_key"] = pd.to_numeric(work["Year"], errors="coerce").astype("Int64")
    work = work[
        work["_brand_key"].ne("")
        & work["_model_key2"].ne("")
        & work["_category_key"].ne("")
        & ~work["_category_key"].isin({"-", "nan", "none"})
        & work["_year_key"].notna()
    ].copy()
    if work.empty:
        return []

    full_market = _guided_market_index()
    full = full_market.copy() if isinstance(full_market, pd.DataFrame) and not full_market.empty else work.copy()
    full["_brand_key"] = full["Brand"].fillna("").astype(str).str.strip().str.casefold()
    full["_model_key2"] = full["Model"].fillna("").astype(str).str.strip().str.casefold()
    full["_category_key"] = full["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold()
    full["_year_key"] = pd.to_numeric(full["Year"], errors="coerce").astype("Int64")
    if "Link" in full.columns:
        full["_link_key"] = full["Link"].fillna("").astype(str).str.strip()

    candidates = []
    for (brand_key, model_key, category_key, year_key), qualifying in work.groupby(
        ["_brand_key", "_model_key2", "_category_key", "_year_key"], dropna=False
    ):
        if qualifying.empty:
            continue
        first = qualifying.iloc[0]
        market = full[
            full["_brand_key"].eq(brand_key)
            & full["_model_key2"].eq(model_key)
            & full["_category_key"].eq(category_key)
            & full["_year_key"].eq(int(year_key))
        ].copy()
        if "_link_key" in market.columns:
            market = market[market["_link_key"].ne("")].drop_duplicates("_link_key", keep="last")
        if market.empty:
            market = qualifying.copy()
            if "Link" in market.columns:
                market["_link_key"] = market["Link"].fillna("").astype(str).str.strip()
                market = market[market["_link_key"].ne("")].drop_duplicates("_link_key", keep="last")
        prices = pd.to_numeric(market["Price"], errors="coerce").dropna()
        qualifying_prices = pd.to_numeric(qualifying["Price"], errors="coerce").dropna()
        candidates.append({
            "vehicle_type": str(first.get("VehicleType") if pd.notna(first.get("VehicleType")) else "").strip(),
            "brand": str(first.get("Brand") if pd.notna(first.get("Brand")) else "").strip(),
            "model": str(first.get("Model") if pd.notna(first.get("Model")) else "").strip(),
            "category": str(first.get("CategoryDetail") if pd.notna(first.get("CategoryDetail")) else "").strip(),
            "year": int(year_key),
            "current_listings": int(len(market)),
            "starting_price": float(prices.min()) if not prices.empty else None,
            "median_price": float(prices.median()) if not prices.empty else None,
            "highest_price": float(prices.max()) if not prices.empty else None,
            "selected_price_ranges": list(_guided_normalize_listing_price_ranges(listing_price_ranges)),
            "selected_price_range_listings": int(len(qualifying_prices)) if not qualifying_prices.empty else 0,
            "selected_price_range_starting_price": float(qualifying_prices.min()) if not qualifying_prices.empty else None,
            "selected_price_range_median_price": float(qualifying_prices.median()) if not qualifying_prices.empty else None,
            "selected_price_range_highest_price": float(qualifying_prices.max()) if not qualifying_prices.empty else None,
            "desired_listing_range_listings": int(len(qualifying_prices)) if not qualifying_prices.empty else 0,
            "desired_listing_range_starting_price": float(qualifying_prices.min()) if not qualifying_prices.empty else None,
            "maximum_desired_listing_price": float(max_listing_price) if max_listing_price is not None else None,
            "historical_distinct_listings": None,
            "turnover_sample_size": None,
            "median_observed_days_to_exit": None,
            "observed_exit_within_60_days_rate": None,
            "historical_price_reduction_rate": None,
            "evidence_quality": "INSUFFICIENT",
            "evidence_scope": "CURRENT_MARKET_ONLY",
            "fallback_context": {"scope": "CURRENT_MARKET_ONLY", "reason": "HISTORICAL_EVIDENCE_TEMPORARILY_UNAVAILABLE"},
            "history_start_date": None,
            "history_end_date": None,
            # Deterministic resilience ranking: lower live competition first;
            # within that, prefer combinations with more evidence inside the
            # selected retail segment.  This does NOT imply stronger demand.
            "_fallback_current_supply": int(len(market)),
            "_fallback_qualifying_supply": int(len(qualifying_prices)),
        })

    candidates.sort(key=lambda item: (
        int(item.get("_fallback_current_supply") or 0),
        -int(item.get("_fallback_qualifying_supply") or 0),
        float(item.get("median_price")) if item.get("median_price") is not None else float("inf"),
    ))

    selected = []
    seen_families = set()
    for item in candidates:
        family = (str(item.get("brand") or "").casefold(), str(item.get("model") or "").casefold())
        if family in seen_families:
            continue
        seen_families.add(family)
        item.pop("_fallback_current_supply", None)
        item.pop("_fallback_qualifying_supply", None)
        selected.append(item)
        if len(selected) >= int(limit):
            break

    supplies = [int(item.get("current_listings") or 0) for item in selected]
    if len(supplies) >= 3:
        series = pd.Series(supplies, dtype="float64")
        low, high = float(series.quantile(0.33)), float(series.quantile(0.67))
    else:
        low = high = None
    for item in selected:
        value = float(item.get("current_listings") or 0)
        if low is not None and high is not None:
            item["competition_context"] = "LOW" if value <= low else "HIGH" if value >= high else "MODERATE"
        else:
            item["competition_context"] = "LOW" if value <= 2 else "MODERATE" if value <= 6 else "HIGH"
        item["stock_rank_score"] = None
    return selected

def _guided_stock_opportunities(rows, limit=5, listing_price_ranges=None, max_listing_price=None):
    """Rank exact Year + Brand + Model + Category stock opportunities quickly.

    IMPORTANT ARCHITECTURE:
    - `rows` is the exact live candidate universe produced by the guided selector.
      The maximum advertised listing price is only an eligibility ceiling.
    - CURRENT competition/pricing is always recalculated from the full live market
      for the exact Year + Brand + Model + Category, across all asking prices.
    - HISTORICAL turnover comes from the precomputed Business Market Intelligence
      table.  That table is generated offline from the daily history and already
      contains CATEGORY_YEAR and MODEL_YEAR statistics, so a customer request never
      needs to duplicate/normalise the full raw daily-history dataframe in memory.
    - Exact CATEGORY_YEAR evidence is preferred.  Only if it is too thin do we use
      same-year Brand + Model evidence across categories.  Different years are never
      pooled.

    This restores the fast, deterministic architecture the stock task originally
    used while retaining the stricter evidence-grain rules added later.
    """
    if rows is None or rows.empty:
        return []

    required = {"Brand", "Model", "CategoryDetail", "Year", "Price"}
    if not required.issubset(rows.columns):
        return []

    qualifying = rows.copy()
    qualifying["_brand_key"] = qualifying["Brand"].fillna("").astype(str).str.strip().str.casefold()
    qualifying["_model_key2"] = qualifying["Model"].fillna("").astype(str).str.strip().str.casefold()
    qualifying["_category_key"] = qualifying["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold()
    qualifying["_year_key"] = pd.to_numeric(qualifying["Year"], errors="coerce").astype("Int64")
    qualifying = qualifying[
        qualifying["_brand_key"].ne("")
        & qualifying["_model_key2"].ne("")
        & qualifying["_category_key"].ne("")
        & ~qualifying["_category_key"].isin({"-", "nan", "none"})
        & qualifying["_year_key"].notna()
    ].copy()
    if qualifying.empty:
        return []

    # Full live market for competition/pricing.  The user's listing-price ceiling
    # must not change the meaning of these statistics after a vehicle qualifies.
    live = _guided_market_index()
    live = live.copy() if isinstance(live, pd.DataFrame) and not live.empty else qualifying.copy()
    live["_brand_key"] = live["Brand"].fillna("").astype(str).str.strip().str.casefold()
    live["_model_key2"] = live["Model"].fillna("").astype(str).str.strip().str.casefold()
    live["_category_key"] = live["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold()
    live["_year_key"] = pd.to_numeric(live["Year"], errors="coerce").astype("Int64")
    if "Link" in live.columns:
        live["_link_key"] = live["Link"].fillna("").astype(str).str.strip()

    candidate_keys = []
    candidate_meta = {}
    for key, group in qualifying.groupby(["_brand_key", "_model_key2", "_category_key", "_year_key"], dropna=False, sort=False):
        brand_key, model_key, category_key, year_key = key
        if pd.isna(year_key) or group.empty:
            continue
        year_int = int(year_key)
        first = group.iloc[0]
        q = group.copy()
        if "Link" in q.columns:
            q["_link_key"] = q["Link"].fillna("").astype(str).str.strip()
            q = q[q["_link_key"].ne("")].drop_duplicates("_link_key", keep="last")
        if q.empty:
            continue
        q_prices = pd.to_numeric(q["Price"], errors="coerce").dropna()
        exact_key = (str(brand_key), str(model_key), str(category_key), year_int)
        candidate_keys.append(exact_key)
        candidate_meta[exact_key] = {
            "brand": _guided_safe_text(first.get("Brand")),
            "model": _guided_safe_text(first.get("Model")),
            "category": _guided_safe_text(first.get("CategoryDetail")),
            "vehicle_type": _guided_safe_text(first.get("VehicleType")),
            "desired_listing_range_listings": int(len(q_prices)),
            "desired_listing_range_starting_price": float(q_prices.min()) if not q_prices.empty else None,
        }
    if not candidate_keys:
        return []

    # Recalculate all current exact-market statistics in one grouped pass.
    candidate_model_year = {(b, m, y) for b, m, _c, y in candidate_keys}
    live_model_index = pd.MultiIndex.from_arrays(
        [live["_brand_key"], live["_model_key2"], live["_year_key"]],
        names=["brand", "model", "year"],
    )
    wanted_model_index = pd.MultiIndex.from_tuples(list(candidate_model_year), names=["brand", "model", "year"])
    live_subset = live.loc[live_model_index.isin(wanted_model_index)].copy()
    if "_link_key" in live_subset.columns:
        live_subset = live_subset[live_subset["_link_key"].ne("")].drop_duplicates("_link_key", keep="last")
    live_subset["_price_numeric"] = pd.to_numeric(live_subset.get("Price"), errors="coerce")

    current_lookup = {}
    for key, group in live_subset.groupby(["_brand_key", "_model_key2", "_category_key", "_year_key"], dropna=False, sort=False):
        b, m, c, y = key
        if pd.isna(y):
            continue
        exact_key = (str(b), str(m), str(c), int(y))
        if exact_key not in candidate_meta:
            continue
        prices = group["_price_numeric"].dropna()
        current_lookup[exact_key] = {
            "current_listings": int(len(group)),
            "starting_price": float(prices.min()) if not prices.empty else None,
            "median_price": float(prices.median()) if not prices.empty else None,
            "highest_price": float(prices.max()) if not prices.empty else None,
        }

    # Build a compact lookup from the precomputed Business market table ONCE.
    exact_history = {}
    model_year_history = {}
    if BUSINESS_INTELLIGENCE_READY and isinstance(business_market_df, pd.DataFrame) and not business_market_df.empty:
        hist = business_market_df.copy()
        needed = {"BusinessGranularity", "Brand", "Model", "Year"}
        if needed.issubset(hist.columns):
            hist["_brand_key"] = hist["Brand"].fillna("").astype(str).str.strip().str.casefold()
            hist["_model_key"] = hist["Model"].fillna("").astype(str).str.strip().str.casefold()
            hist["_category_key"] = hist.get("CategoryDetail", pd.Series("", index=hist.index)).fillna("").astype(str).str.strip().str.casefold()
            hist["_year_key"] = pd.to_numeric(hist["Year"], errors="coerce").astype("Int64")
            hist["_granularity"] = hist["BusinessGranularity"].fillna("").astype(str).str.strip().str.upper()

            hist_model_index = pd.MultiIndex.from_arrays(
                [hist["_brand_key"], hist["_model_key"], hist["_year_key"]],
                names=["brand", "model", "year"],
            )
            hist = hist.loc[hist_model_index.isin(wanted_model_index)].copy()

            def _num(row, name):
                value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
                return None if pd.isna(value) else float(value)

            def _row_metrics(row):
                rate = _num(row, "ObservedExitWithin60DaysRate")
                if rate is not None and 1.0 < rate <= 100.0:
                    rate /= 100.0
                reduction = _num(row, "PriceReductionRate")
                if reduction is not None and 1.0 < reduction <= 100.0:
                    reduction /= 100.0
                hist_distinct = _num(row, "HistoricalDistinctListings")
                sample = _num(row, "Exit60EligibleListings")
                if sample is None:
                    sample = hist_distinct
                exits = _num(row, "HistoricalObservedMarketExits")
                median_exit = _num(row, "MedianObservedDaysToExit")
                quality = _guided_safe_text(row.get("EvidenceQuality")).upper() or "LOW"
                sample_i = int(sample or 0)
                exits_i = int(exits or 0)
                sufficient = bool(sample_i >= 5 and exits_i >= 3 and rate is not None and median_exit is not None)
                return {
                    "historical_distinct_listings": int(hist_distinct) if hist_distinct is not None else None,
                    "turnover_sample_size": sample_i,
                    "observed_exit_count": exits_i if exits is not None else None,
                    "median_observed_days_to_exit": median_exit,
                    "observed_exit_within_60_days_rate": rate,
                    "historical_price_reduction_rate": reduction,
                    "evidence_quality": quality,
                    "turnover_sufficient": sufficient,
                    "evidence_source": "PRECOMPUTED_BUSINESS_MARKET",
                }

            # Prefer the row with the deepest usable sample if a builder emitted duplicates.
            if "Exit60EligibleListings" in hist.columns:
                sample_rank = pd.to_numeric(hist["Exit60EligibleListings"], errors="coerce")
            else:
                sample_rank = pd.Series(float("nan"), index=hist.index, dtype="float64")
            if "HistoricalDistinctListings" in hist.columns:
                distinct_rank = pd.to_numeric(hist["HistoricalDistinctListings"], errors="coerce")
            else:
                distinct_rank = pd.Series(0.0, index=hist.index, dtype="float64")
            hist["_sample_rank"] = sample_rank.fillna(distinct_rank).fillna(0)
            quality_rank = {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "INSUFFICIENT": 0}
            hist["_quality_rank"] = hist.get("EvidenceQuality", pd.Series("", index=hist.index)).fillna("").astype(str).str.upper().map(quality_rank).fillna(1)
            hist = hist.sort_values(["_quality_rank", "_sample_rank"], ascending=[False, False])

            for row in hist.to_dict("records"):
                b = str(row.get("_brand_key") or "")
                m = str(row.get("_model_key") or "")
                y = row.get("_year_key")
                if y is None or pd.isna(y):
                    continue
                y = int(y)
                granularity = str(row.get("_granularity") or "").upper()
                if granularity == "CATEGORY_YEAR":
                    c = str(row.get("_category_key") or "")
                    key = (b, m, c, y)
                    if key in candidate_meta and key not in exact_history:
                        exact_history[key] = _row_metrics(row)
                elif granularity == "MODEL_YEAR":
                    key = (b, m, y)
                    if key in candidate_model_year and key not in model_year_history:
                        model_year_history[key] = _row_metrics(row)

    candidate_records = []
    for key in candidate_keys:
        b, m, c, y = key
        meta = candidate_meta[key]
        current = current_lookup.get(key)
        if current is None:
            # A candidate existed under the selector ceiling, so this should be rare;
            # retain it from qualifying rows rather than fail the whole report.
            q = qualifying[
                qualifying["_brand_key"].eq(b)
                & qualifying["_model_key2"].eq(m)
                & qualifying["_category_key"].eq(c)
                & qualifying["_year_key"].eq(y)
            ]
            prices = pd.to_numeric(q.get("Price"), errors="coerce").dropna()
            current = {
                "current_listings": int(len(q)),
                "starting_price": float(prices.min()) if not prices.empty else None,
                "median_price": float(prices.median()) if not prices.empty else None,
                "highest_price": float(prices.max()) if not prices.empty else None,
            }

        exact = exact_history.get(key)
        broader = model_year_history.get((b, m, y))
        evidence = exact
        scope = "EXACT_CATEGORY_YEAR"
        fallback_context = None

        if not (exact and exact.get("turnover_sufficient")):
            if _guided_stock_metrics_usable(broader, minimum_sample=3):
                evidence = broader
                scope = "MODEL_YEAR_FALLBACK"
                fallback_context = {
                    "scope": scope,
                    "exact_historical_distinct_listings": exact.get("historical_distinct_listings") if exact else None,
                    "exact_turnover_sample_size": exact.get("turnover_sample_size") if exact else None,
                    "fallback_historical_distinct_listings": broader.get("historical_distinct_listings"),
                    "fallback_turnover_sample_size": broader.get("turnover_sample_size"),
                    "fallback_source": "PRECOMPUTED_BUSINESS_MARKET",
                }
            elif _guided_stock_metrics_usable(exact, minimum_sample=2):
                evidence = exact
                scope = "EXACT_THIN"
                fallback_context = {
                    "scope": scope,
                    "exact_historical_distinct_listings": exact.get("historical_distinct_listings"),
                    "exact_turnover_sample_size": exact.get("turnover_sample_size"),
                }
            else:
                evidence = exact or {}
                scope = "CURRENT_MARKET_ONLY"
                fallback_context = {
                    "scope": scope,
                    "exact_historical_distinct_listings": exact.get("historical_distinct_listings") if exact else None,
                    "exact_turnover_sample_size": exact.get("turnover_sample_size") if exact else None,
                }

        candidate_records.append({
            "brand_key": b,
            "model_key": m,
            "category_key": c,
            "year": y,
            **meta,
            **current,
            "historical_distinct_listings": exact.get("historical_distinct_listings") if exact else None,
            "turnover_sample_size": evidence.get("turnover_sample_size") if evidence else None,
            "median_observed_days_to_exit": evidence.get("median_observed_days_to_exit") if evidence else None,
            "observed_exit_within_60_days_rate": evidence.get("observed_exit_within_60_days_rate") if evidence else None,
            "historical_price_reduction_rate": evidence.get("historical_price_reduction_rate") if evidence else None,
            "evidence_quality": (evidence.get("evidence_quality") if evidence else None) or ("INSUFFICIENT" if scope == "CURRENT_MARKET_ONLY" else "LOW"),
            "evidence_scope": scope,
            "fallback_context": fallback_context,
        })

    if not candidate_records:
        return []

    ranked = pd.DataFrame(candidate_records)

    def _pct_rank(series, higher_is_better=True):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() <= 1:
            return pd.Series(0.5, index=numeric.index, dtype="float64")
        return numeric.rank(ascending=higher_is_better, pct=True, method="average")

    ranked["_turnover"] = pd.to_numeric(ranked["observed_exit_within_60_days_rate"], errors="coerce")
    ranked["_median_exit_days"] = pd.to_numeric(ranked["median_observed_days_to_exit"], errors="coerce")
    ranked["_current_supply"] = pd.to_numeric(ranked["current_listings"], errors="coerce")
    ranked["_history_depth"] = pd.to_numeric(ranked["turnover_sample_size"], errors="coerce")
    ranked["_price_reduction"] = pd.to_numeric(ranked["historical_price_reduction_rate"], errors="coerce")
    ranked["_scope_score"] = ranked["evidence_scope"].map({
        "EXACT_CATEGORY_YEAR": 1.0,
        "MODEL_YEAR_FALLBACK": 0.62,
        "EXACT_THIN": 0.56,
        "CURRENT_MARKET_ONLY": 0.28,
    }).fillna(0.4)
    ranked["_evidence_score"] = ranked["evidence_quality"].map({"HIGH": 1.0, "MEDIUM": 0.70, "LOW": 0.38}).fillna(0.25)
    ranked["_turnover_score"] = _pct_rank(ranked["_turnover"], True).fillna(0.25)
    ranked["_speed_score"] = _pct_rank(ranked["_median_exit_days"], False).fillna(0.25)
    ranked["_competition_score"] = _pct_rank(ranked["_current_supply"], False).fillna(0.35)
    ranked["_history_score"] = _pct_rank(ranked["_history_depth"], True).fillna(0.20)
    ranked["_price_pressure_score"] = _pct_rank(ranked["_price_reduction"], False).fillna(0.50)
    ranked["_stock_score"] = (
        0.31 * ranked["_turnover_score"]
        + 0.22 * ranked["_speed_score"]
        + 0.18 * ranked["_competition_score"]
        + 0.10 * ranked["_history_score"]
        + 0.07 * ranked["_evidence_score"]
        + 0.04 * ranked["_price_pressure_score"]
        + 0.08 * ranked["_scope_score"]
    )
    ranked = ranked.sort_values(
        ["_stock_score", "_scope_score", "_turnover", "_median_exit_days", "_current_supply", "_history_depth"],
        ascending=[False, False, False, True, True, False],
        na_position="last",
    )

    selected = []
    seen_families = set()
    for row in ranked.to_dict("records"):
        family = (str(row.get("brand_key") or ""), str(row.get("model_key") or ""))
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(row)
        if len(selected) >= int(limit):
            break

    valid_supply = pd.Series([item.get("current_listings") for item in selected], dtype="float64").dropna()
    supply_low = float(valid_supply.quantile(0.33)) if len(valid_supply) >= 3 else None
    supply_high = float(valid_supply.quantile(0.67)) if len(valid_supply) >= 3 else None

    def _competition_context(value):
        if value is None or pd.isna(value):
            return "UNKNOWN"
        value = float(value)
        if supply_low is not None and supply_high is not None:
            if value <= supply_low:
                return "LOW"
            if value >= supply_high:
                return "HIGH"
            return "MODERATE"
        if value <= 2:
            return "LOW"
        if value <= 6:
            return "MODERATE"
        return "HIGH"

    result = []
    for row in selected:
        result.append({
            "vehicle_type": str(row.get("vehicle_type") or "").strip(),
            "brand": str(row.get("brand") or "").strip(),
            "model": str(row.get("model") or "").strip(),
            "category": str(row.get("category") or "").strip(),
            "year": int(row.get("year")),
            "current_listings": int(row.get("current_listings") or 0),
            "competition_context": _competition_context(row.get("current_listings")),
            "stock_rank_score": float(row.get("_stock_score")) if row.get("_stock_score") is not None and not pd.isna(row.get("_stock_score")) else None,
            "starting_price": float(row.get("starting_price")) if row.get("starting_price") is not None and not pd.isna(row.get("starting_price")) else None,
            "median_price": float(row.get("median_price")) if row.get("median_price") is not None and not pd.isna(row.get("median_price")) else None,
            "highest_price": float(row.get("highest_price")) if row.get("highest_price") is not None and not pd.isna(row.get("highest_price")) else None,
            "selected_price_ranges": [],
            "selected_price_range_listings": int(row.get("desired_listing_range_listings") or 0),
            "selected_price_range_starting_price": row.get("desired_listing_range_starting_price"),
            "selected_price_range_median_price": None,
            "selected_price_range_highest_price": None,
            "desired_listing_range_listings": int(row.get("desired_listing_range_listings") or 0),
            "desired_listing_range_starting_price": row.get("desired_listing_range_starting_price"),
            "maximum_desired_listing_price": float(max_listing_price) if max_listing_price is not None else None,
            "historical_distinct_listings": int(row.get("historical_distinct_listings")) if row.get("historical_distinct_listings") is not None and not pd.isna(row.get("historical_distinct_listings")) else None,
            "turnover_sample_size": int(row.get("turnover_sample_size")) if row.get("turnover_sample_size") is not None and not pd.isna(row.get("turnover_sample_size")) else None,
            "median_observed_days_to_exit": float(row.get("median_observed_days_to_exit")) if row.get("median_observed_days_to_exit") is not None and not pd.isna(row.get("median_observed_days_to_exit")) else None,
            "observed_exit_within_60_days_rate": float(row.get("observed_exit_within_60_days_rate")) if row.get("observed_exit_within_60_days_rate") is not None and not pd.isna(row.get("observed_exit_within_60_days_rate")) else None,
            "historical_price_reduction_rate": float(row.get("historical_price_reduction_rate")) if row.get("historical_price_reduction_rate") is not None and not pd.isna(row.get("historical_price_reduction_rate")) else None,
            "evidence_quality": str(row.get("evidence_quality") or "").strip(),
            "evidence_scope": str(row.get("evidence_scope") or "").strip(),
            "fallback_context": row.get("fallback_context"),
            "history_start_date": None,
            "history_end_date": None,
        })
    return result


@app.route("/api/guided/discovery/stock-result", methods=["POST"])
def api_guided_stock_result():
    """Fast, isolated evidence endpoint for the paid Stock Purchase task.

    Stock Purchase used to share the generic discovery-result path and later began
    rebuilding raw daily history during that request.  That made one of the most
    important Business workflows depend on unrelated report work and on a large
    historical dataframe.  This endpoint intentionally does only four things:

    1. reproduce the selector's exact live-market filters;
    2. rank exact Year + Brand + Model + Category opportunities using the small
       precomputed Business market-intelligence table;
    3. return one qualifying live example for every recommendation; and
    4. never call OpenAI or load/rebuild the raw daily listing history.

    The maximum budget field here means maximum desired *advertised listing price*.
    It is an eligibility ceiling only; once a combination qualifies, competition,
    pricing and turnover statistics use the full exact market for that vehicle.
    """
    started = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        task = str(data.get("task") or "STOCK_PURCHASE").strip().upper()
        if task != "STOCK_PURCHASE":
            return jsonify({"success": False, "error": "INVALID_STOCK_TASK"}), 400

        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        vehicle_types = [str(v) for v in (answers.get("vehicleTypes") or []) if str(v) != "__ALL__"]
        brands = [str(v) for v in (answers.get("brands") or []) if str(v) != "__ALL__"]
        model_keys = [str(v) for v in (answers.get("models") or []) if str(v) != "__ALL__"]
        category_keys = [str(v) for v in (answers.get("categories") or []) if str(v)]

        min_year = answers.get("minYear")
        if min_year in [None, "", "__ALL__", "ALL"]:
            min_year = None
        else:
            min_year = int(float(min_year))

        no_maximum = bool(answers.get("stockBudgetNoMaximum"))
        max_listing_price = None if no_maximum else answers.get("budget")
        if max_listing_price in [None, ""]:
            max_listing_price = None
        else:
            max_listing_price = float(max_listing_price)
            if not math.isfinite(max_listing_price) or max_listing_price <= 0:
                return jsonify({"success": False, "error": "INVALID_MAX_LISTING_PRICE"}), 400

        index = _guided_market_index()
        if index is None or index.empty:
            return jsonify({"success": False, "error": "GUIDED_OPTIONS_NOT_READY"}), 503

        rows = _guided_index_filter(
            index,
            min_year=min_year,
            vehicle_types=vehicle_types,
            brands=brands,
            model_keys=model_keys,
            max_budget=max_listing_price,
            price_ranges=None,
        )
        rows = _guided_apply_category_keys(rows, category_keys)
        filter_elapsed = time.perf_counter() - started

        if rows is None or rows.empty:
            print(
                f"GUIDED STOCK RESULT: 0 exact qualifying live rows after filters in {filter_elapsed:.3f}s",
                flush=True,
            )
            return jsonify({
                "success": True,
                "count": 0,
                "groups": [],
                "year_options": [],
                "results": [],
                "stock_opportunities": [],
                "no_matches": True,
                "timing_ms": {"filter": int(filter_elapsed * 1000)},
            })

        rank_started = time.perf_counter()
        try:
            opportunities = _guided_stock_opportunities(
                rows,
                limit=5,
                listing_price_ranges=None,
                max_listing_price=max_listing_price,
            )
        except Exception as exc:
            print("GUIDED STOCK FAST RANKING FAILED:", repr(exc), flush=True)
            traceback.print_exc()
            opportunities = _guided_stock_current_market_fallback(
                rows,
                limit=5,
                listing_price_ranges=None,
                max_listing_price=max_listing_price,
            )
        if not opportunities:
            opportunities = _guided_stock_current_market_fallback(
                rows,
                limit=5,
                listing_price_ranges=None,
                max_listing_price=max_listing_price,
            )
        rank_elapsed = time.perf_counter() - rank_started

        # One live example for each exact recommendation.  Prefer a listing inside
        # the user's desired listing-price ceiling because that is what made the
        # opportunity eligible.  All statistics on the recommendation itself still
        # describe the full exact market.
        result_cols = [
            c for c in (
                "Brand", "Model", "CategoryDetail", "Year", "Price", "KM",
                "Company", "Color", "Transmission", "Location", "Image", "Link"
            ) if c in rows.columns
        ]
        examples = []
        if opportunities and result_cols:
            candidate_rows = rows.copy()
            if "Link" in candidate_rows.columns:
                candidate_rows = candidate_rows[
                    candidate_rows["Link"].fillna("").astype(str).str.strip().ne("")
                ]
            for opportunity in opportunities:
                mask = (
                    candidate_rows["Brand"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("brand") or "").strip().casefold())
                    & candidate_rows["Model"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("model") or "").strip().casefold())
                    & candidate_rows["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("category") or "").strip().casefold())
                    & pd.to_numeric(candidate_rows["Year"], errors="coerce").eq(opportunity.get("year"))
                )
                matches = candidate_rows.loc[mask].sort_values("Price", ascending=True).head(1)
                if matches.empty:
                    continue
                row = matches.iloc[0]
                examples.append({
                    "brand": _guided_safe_text(row.get("Brand")),
                    "model": _guided_safe_text(row.get("Model")),
                    "category": _guided_safe_text(row.get("CategoryDetail")),
                    "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
                    "price": float(row["Price"]) if pd.notna(row.get("Price")) else None,
                    "km": float(row["KM"]) if "KM" in matches.columns and pd.notna(row.get("KM")) else None,
                    "company": _guided_safe_text(row.get("Company")),
                    "color": _guided_safe_text(row.get("Color")),
                    "transmission": _guided_safe_text(row.get("Transmission")),
                    "location": _guided_safe_text(row.get("Location")),
                    "image": _guided_safe_text(row.get("Image")),
                    "link": _guided_safe_text(row.get("Link")),
                })

        total_elapsed = time.perf_counter() - started
        print(
            f"GUIDED STOCK RESULT: {len(opportunities)} recommendations from {len(rows)} qualifying live rows "
            f"(filter={filter_elapsed:.3f}s rank={rank_elapsed:.3f}s total={total_elapsed:.3f}s)",
            flush=True,
        )
        return jsonify({
            "success": True,
            "count": int(len(rows)),
            "groups": [],
            "year_options": [],
            "results": examples,
            "stock_opportunities": opportunities,
            "no_matches": False,
            "timing_ms": {
                "filter": int(filter_elapsed * 1000),
                "rank": int(rank_elapsed * 1000),
                "total": int(total_elapsed * 1000),
            },
        })
    except (TypeError, ValueError) as exc:
        print("GUIDED STOCK RESULT INVALID REQUEST:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "INVALID_GUIDED_STOCK_RESULT"}), 400
    except Exception as exc:
        print("GUIDED STOCK RESULT FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "error": "GUIDED_STOCK_RESULT_UNAVAILABLE"}), 503


@app.route("/api/guided/discovery/result", methods=["POST"])
def api_guided_discovery_result():
    """Exact current-listing evidence for a completed guided AI-assistant task.

    This deliberately uses the same pre-normalised listing index as the chips, so
    the completed report cannot disagree with a path the selector just allowed.
    """
    index = _guided_market_index()
    if index is None or index.empty:
        return jsonify({"success": False, "error": "GUIDED_OPTIONS_NOT_READY"}), 503

    try:
        data = request.get_json(silent=True) or {}
        task = str(data.get("task") or "").strip().upper()

        # Backwards-compatible fast path: older cached frontends still POST Stock
        # Purchase to the generic discovery endpoint. Route those requests through
        # the isolated stock-result implementation so they never fall back onto the
        # expensive historical/generic report path.
        if task == "STOCK_PURCHASE":
            return api_guided_stock_result()

        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        vehicle_types = [str(v) for v in (answers.get("vehicleTypes") or []) if str(v) != "__ALL__"]
        brands = [str(v) for v in (answers.get("brands") or []) if str(v) != "__ALL__"]
        model_keys = [str(v) for v in (answers.get("models") or []) if str(v) != "__ALL__"]
        category_keys = [str(v) for v in (answers.get("categories") or []) if str(v)]

        min_year = answers.get("minYear")
        if min_year in [None, "", "__ALL__", "ALL"]:
            min_year = None
        else:
            min_year = int(float(min_year))

        max_budget = answers.get("budget")
        if max_budget in [None, ""]:
            max_budget = None
        else:
            max_budget = float(max_budget)

        # Legacy listing-price bands are ignored for new Stock Purchase journeys.
        # The Stock Purchase `budget` is deliberately a MAXIMUM ADVERTISED LISTING
        # PRICE used to define the target retail segment. It is not acquisition cost.
        listing_price_ranges = tuple() if task == "STOCK_PURCHASE" else _guided_normalize_listing_price_ranges(answers.get("priceRanges") or [])
        listing_budget_ceiling = max_budget

        rows = _guided_index_filter(
            index,
            min_year=min_year,
            vehicle_types=vehicle_types,
            brands=brands,
            model_keys=model_keys,
            max_budget=listing_budget_ceiling,
            price_ranges=None,
        )
        rows = _guided_apply_category_keys(rows, category_keys)

        if rows.empty:
            return jsonify({"success": True, "count": 0, "groups": [], "results": []})

        # Guided-assistant-only data quality guard. Very small odometer values on
        # older used vehicles are almost always placeholder/data-entry artefacts
        # (for example 1 km on a 2016 car). Keep genuine near-new current/recent
        # vehicles intact, but exclude implausible values from median evidence.
        rows = rows.copy()
        if "KM" in rows.columns:
            rows["_guided_km"] = pd.to_numeric(rows["KM"], errors="coerce")
            years_numeric = pd.to_numeric(rows["Year"], errors="coerce") if "Year" in rows.columns else pd.Series(index=rows.index, dtype=float)
            current_year = datetime.now().year
            # A 2+ year-old used vehicle showing only a few hundred kilometres is
            # possible, but unusual enough that it should not drive medians/ranking
            # without verification. Preserve the raw listing value for display,
            # mark it as unverified, and exclude it only from statistical evidence.
            implausibly_low = (years_numeric <= current_year - 2) & rows["_guided_km"].notna() & (rows["_guided_km"] < 1_000)
            implausibly_high = rows["_guided_km"].notna() & (rows["_guided_km"] > 1_000_000)
            rows["_guided_km_reliable"] = ~(implausibly_low | implausibly_high)
            rows.loc[~rows["_guided_km_reliable"], "_guided_km"] = float("nan")
        else:
            rows["_guided_km_reliable"] = True

        grouped = (
            rows.groupby(["Brand", "Model", "CategoryDetail"], dropna=False)
            .agg(
                listing_count=("Price", "size"),
                min_price=("Price", "min"),
                median_price=("Price", "median"),
                max_price=("Price", "max"),
                newest_year=("Year", "max"),
                median_km=("_guided_km", "median") if "_guided_km" in rows.columns else ("Price", "size"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values(["listing_count", "median_price"], ascending=[False, True])

        # Preserve explicit model selections before applying the compact 20-group
        # report cap. Otherwise one selected model with many high-volume variants
        # can crowd a second explicitly selected model out of the recommendation
        # evidence on the first request.
        if model_keys and not grouped.empty:
            grouped = grouped.copy()
            grouped["_guided_group_model_key"] = (
                grouped["Brand"].fillna("").astype(str).str.strip()
                + "||"
                + grouped["Model"].fillna("").astype(str).str.strip()
            ).str.casefold()
            wanted_model_keys = []
            seen_model_keys = set()
            for value in model_keys:
                key = str(value or "").strip().casefold()
                if key and key not in seen_model_keys:
                    wanted_model_keys.append(key)
                    seen_model_keys.add(key)

            coverage_indices = []
            for key in wanted_model_keys:
                matches = grouped.index[grouped["_guided_group_model_key"].eq(key)].tolist()
                if matches:
                    coverage_indices.append(matches[0])

            coverage = grouped.loc[coverage_indices] if coverage_indices else grouped.iloc[0:0]
            remainder = grouped.loc[~grouped.index.isin(coverage_indices)]
            grouped = pd.concat([coverage, remainder], axis=0).head(20).drop(columns=["_guided_group_model_key"], errors="ignore")
        else:
            grouped = grouped.head(20)
        groups = []
        for row in grouped.to_dict("records"):
            groups.append({
                "brand": _guided_safe_text(row.get("Brand")),
                "model": _guided_safe_text(row.get("Model")),
                "category": _guided_safe_text(row.get("CategoryDetail")),
                "listing_count": int(row.get("listing_count") or 0),
                "min_price": float(row["min_price"]) if pd.notna(row.get("min_price")) else None,
                "median_price": float(row["median_price"]) if pd.notna(row.get("median_price")) else None,
                "max_price": float(row["max_price"]) if pd.notna(row.get("max_price")) else None,
                "newest_year": int(row["newest_year"]) if pd.notna(row.get("newest_year")) else None,
                "median_km": float(row["median_km"]) if pd.notna(row.get("median_km")) else None,
            })

        # Enrich the guided buyer groups with the existing historical buyer-intelligence
        # evidence so Personal recommendations can rank resale/demand preferences without
        # changing any standalone valuation routes or calculations.
        if groups and task != "STOCK_PURCHASE":
            hard_results = [
                {
                    "brand": str(row.get("Brand") or ""),
                    "model": str(row.get("Model") or ""),
                    "category": str(row.get("CategoryDetail") or ""),
                }
                for row in rows[["Brand", "Model", "CategoryDetail"]].drop_duplicates().to_dict("records")
            ]
            enrich_inputs = [
                {
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "category": item.get("category"),
                    "newest_year": item.get("newest_year"),
                    "count": item.get("listing_count"),
                    "starting_price": item.get("min_price"),
                }
                for item in groups
            ]
            try:
                enriched = enrich_model_options_with_buyer_intelligence(
                    enrich_inputs,
                    {"budget": max_budget, "min_year": min_year},
                    [],
                    hard_results,
                )
                for item, enriched_item in zip(groups, enriched):
                    item["buyer_intelligence"] = enriched_item.get("buyer_intelligence")
            except Exception as exc:
                print("GUIDED BUYER INTELLIGENCE ENRICHMENT FAILED:", repr(exc), flush=True)
                for item in groups:
                    item["buyer_intelligence"] = None

        year_grouped = (
            rows.groupby(["Brand", "Model", "CategoryDetail", "Year"], dropna=False)
            .agg(
                listing_count=("Price", "size"),
                min_price=("Price", "min"),
                median_price=("Price", "median"),
                max_price=("Price", "max"),
                median_km=("_guided_km", "median") if "_guided_km" in rows.columns else ("Price", "size"),
            )
            .reset_index()
            .sort_values(["Brand", "Model", "CategoryDetail", "Year"], ascending=[True, True, True, False])
        )
        year_options = []
        for row in year_grouped.to_dict("records"):
            year_options.append({
                "brand": _guided_safe_text(row.get("Brand")),
                "model": _guided_safe_text(row.get("Model")),
                "category": _guided_safe_text(row.get("CategoryDetail")),
                "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
                "listing_count": int(row.get("listing_count") or 0),
                "min_price": float(row["min_price"]) if pd.notna(row.get("min_price")) else None,
                "median_price": float(row["median_price"]) if pd.notna(row.get("median_price")) else None,
                "max_price": float(row["max_price"]) if pd.notna(row.get("max_price")) else None,
                "median_km": float(row["median_km"]) if pd.notna(row.get("median_km")) else None,
            })

        # Recommendation cards are titled with the newest qualifying year, so every
        # visible current-market statistic on those cards must describe that SAME
        # year. Keep the broader group evidence for market depth/history, but attach
        # an exact-year evidence slice for ranking, writing and display.
        year_lookup = {}
        for item in year_options:
            key = (
                str(item.get("brand") or "").casefold(),
                str(item.get("model") or "").casefold(),
                str(item.get("category") or "").casefold(),
                int(item.get("year")) if item.get("year") is not None else None,
            )
            year_lookup[key] = item
        for item in groups:
            newest_year = item.get("newest_year")
            key = (
                str(item.get("brand") or "").casefold(),
                str(item.get("model") or "").casefold(),
                str(item.get("category") or "").casefold(),
                int(newest_year) if newest_year is not None else None,
            )
            exact = year_lookup.get(key) or {}
            item["newest_year_listing_count"] = int(exact.get("listing_count") or 0)
            item["newest_year_min_price"] = exact.get("min_price")
            item["newest_year_median_price"] = exact.get("median_price")
            item["newest_year_max_price"] = exact.get("max_price")
            item["newest_year_median_km"] = exact.get("median_km")

        stock_opportunities = []
        if task == "STOCK_PURCHASE":
            _stock_rank_started = time.perf_counter()
            try:
                stock_opportunities = _guided_stock_opportunities(
                    rows, limit=5, listing_price_ranges=None, max_listing_price=max_budget
                )
            except Exception as stock_exc:
                # A history-source/schema problem must never take down the whole
                # paid Stock Purchase report. Preserve exact current-market
                # candidates and make the missing turnover evidence explicit.
                print("GUIDED STOCK OPPORTUNITY RANKING FAILED:", repr(stock_exc), flush=True)
                traceback.print_exc()
                stock_opportunities = _guided_stock_current_market_fallback(
                    rows, limit=5, listing_price_ranges=None, max_listing_price=max_budget
                )

            if not stock_opportunities:
                stock_opportunities = _guided_stock_current_market_fallback(
                    rows, limit=5, listing_price_ranges=None, max_listing_price=max_budget
                )
            print(
                f"GUIDED STOCK RANKING TOTAL: {len(stock_opportunities)} selected from "
                f"{len(rows)} qualifying live rows in {time.perf_counter() - _stock_rank_started:.3f}s",
                flush=True,
            )

        result_cols = [c for c in ("Brand", "Model", "CategoryDetail", "Year", "Price", "KM", "_guided_km_reliable", "Company", "Color", "Transmission", "Location", "Image", "Link") if c in rows.columns]
        posted_rows = rows
        if "Link" in posted_rows.columns:
            posted_rows = posted_rows[posted_rows["Link"].fillna("").astype(str).str.strip().ne("")]

        # Return a compact but diverse current-listing pool. Put one listing from
        # every Brand + Model + Category group first, then fill with the strongest
        # remaining current listings. The frontend can therefore guarantee that
        # Top Matches represents every shortlisted option before repeating one.
        ordered_posted = posted_rows.sort_values(["Year", "Price"], ascending=[False, True])
        identity_cols = [c for c in ("Brand", "Model", "CategoryDetail") if c in ordered_posted.columns]
        if identity_cols:
            # First reserve one live listing for each explicitly selected model,
            # then one for every Brand + Model + Category group, then fill the pool.
            # This keeps all user-selected models visible in downstream Top Matches
            # whenever a live linked listing exists for them.
            selected_model_indices = []
            if model_keys and {"Brand", "Model"}.issubset(ordered_posted.columns):
                model_series = (
                    ordered_posted["Brand"].fillna("").astype(str).str.strip()
                    + "||"
                    + ordered_posted["Model"].fillna("").astype(str).str.strip()
                ).str.casefold()
                seen = set()
                for value in model_keys:
                    key = str(value or "").strip().casefold()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    matches = ordered_posted.index[model_series.eq(key)].tolist()
                    if matches:
                        selected_model_indices.append(matches[0])

            # For Stock Purchase, reserve one linked live example for every exact
            # Year + Brand + Model + Category recommendation before any generic fill.
            stock_example_indices = []
            if stock_opportunities and {"Brand", "Model", "CategoryDetail", "Year"}.issubset(ordered_posted.columns):
                for opportunity in stock_opportunities:
                    mask = (
                        ordered_posted["Brand"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("brand") or "").strip().casefold())
                        & ordered_posted["Model"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("model") or "").strip().casefold())
                        & ordered_posted["CategoryDetail"].fillna("").astype(str).str.strip().str.casefold().eq(str(opportunity.get("category") or "").strip().casefold())
                        & pd.to_numeric(ordered_posted["Year"], errors="coerce").eq(opportunity.get("year"))
                    )
                    matches = ordered_posted.index[mask].tolist()
                    if matches:
                        stock_example_indices.append(matches[0])

            selected_model_rows = ordered_posted.loc[selected_model_indices] if selected_model_indices else ordered_posted.iloc[0:0]
            diverse = ordered_posted.drop_duplicates(subset=identity_cols, keep="first")
            combined_indices = list(dict.fromkeys(stock_example_indices + list(selected_model_rows.index) + list(diverse.index)))
            covered = ordered_posted.loc[combined_indices] if combined_indices else ordered_posted.iloc[0:0]
            remainder = ordered_posted.loc[~ordered_posted.index.isin(combined_indices)]
            result_rows = pd.concat([covered, remainder], axis=0)[result_cols].head(80)
        else:
            result_rows = ordered_posted[result_cols].head(80)
        results = []
        for row in result_rows.to_dict("records"):
            results.append({
                "brand": _guided_safe_text(row.get("Brand")),
                "model": _guided_safe_text(row.get("Model")),
                "category": _guided_safe_text(row.get("CategoryDetail")),
                "year": int(row["Year"]) if pd.notna(row.get("Year")) else None,
                "price": float(row["Price"]) if pd.notna(row.get("Price")) else None,
                "km": float(row["KM"]) if pd.notna(row.get("KM")) else None,
                "km_reliable": bool(row.get("_guided_km_reliable", True)),
                "company": _guided_safe_text(row.get("Company")),
                "color": _guided_safe_text(row.get("Color")),
                "transmission": _guided_safe_text(row.get("Transmission")),
                "location": _guided_safe_text(row.get("Location")),
                "image": _guided_safe_text(row.get("Image")),
                "link": _guided_safe_text(row.get("Link")),
            })

        return jsonify({
            "success": True,
            "count": int(len(rows)),
            "groups": groups,
            "year_options": year_options,
            "results": results,
            "stock_opportunities": stock_opportunities,
        })
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "INVALID_GUIDED_RESULT"}), 400
    except Exception as exc:
        print("GUIDED DISCOVERY RESULT FAILED:", repr(exc), flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "error": "GUIDED_RESULT_UNAVAILABLE"}), 503


@app.route("/api/guided/discovery/recommendations", methods=["POST"])
def api_guided_discovery_recommendations():
    """Write the Personal buyer shortlist like one experienced adviser.

    Ranking and market facts remain deterministic in the guided-discovery flow.
    This endpoint receives at most five already-ranked candidates and performs one
    writing/synthesis model call across the whole shortlist so the explanations
    can compare options naturally instead of repeating sentence templates.

    This is AI-assistant-only. It does not call or modify the standalone
    OtoDeğer valuation routes or valuation calculations.
    """
    try:
        allowed, retry_after = _assistant_request_allowed()
        if not allowed:
            response = jsonify({
                "success": False,
                "error": "RATE_LIMITED",
                "retry_after_seconds": retry_after,
            })
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response

        data = request.get_json(silent=True) or {}
        language = str(data.get("language") or "TR").strip().upper()
        if language not in {"EN", "TR", "RU"}:
            language = "TR"

        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list):
            return jsonify({"success": False, "error": "INVALID_CANDIDATES"}), 400

        candidates = []
        seen_keys = set()
        for raw in raw_candidates[:5]:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key") or "").strip()[:240]
            brand = str(raw.get("brand") or "").strip()[:80]
            model = str(raw.get("model") or "").strip()[:120]
            category = str(raw.get("category") or "").strip()[:120]
            if not key or not brand or not model or key in seen_keys:
                continue
            seen_keys.add(key)

            def safe_number(name, *, integer=False):
                value = raw.get(name)
                if value in [None, ""]:
                    return None
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        return None
                    return int(round(number)) if integer else round(number, 4)
                except (TypeError, ValueError):
                    return None

            powertrain_type = str(raw.get("powertrain_type") or "").strip().lower()
            if powertrain_type not in {"electric", "hybrid", "diesel", "petrol"}:
                powertrain_type = None

            candidates.append({
                "key": key,
                "rank": safe_number("rank", integer=True),
                "brand": brand,
                "model": model,
                "category": category,
                "newest_year": safe_number("newest_year", integer=True),
                # These are exact statistics for the displayed newest qualifying
                # year, not the cheapest/median values pooled across older years.
                "min_price_gbp": safe_number("min_price", integer=True),
                "median_price_gbp": safe_number("median_price", integer=True),
                "max_price_gbp": safe_number("max_price", integer=True),
                "current_listings": safe_number("current_listings", integer=True),
                "current_listings_all_years": safe_number("current_listings_all_years", integer=True),
                "median_km": safe_number("median_km", integer=True),
                "historical_listings": safe_number("historical_listings", integer=True),
                "exit_60_rate": safe_number("exit_60_rate"),
                "exit_60_eligible": safe_number("exit_60_eligible", integer=True),
                "premium_brand": bool(raw.get("premium_brand")),
                "powertrain_type": powertrain_type,
                "engine_litres": safe_number("engine_litres"),
            })

        if not candidates:
            return jsonify({"success": True, "recommendations": []})

        allowed_goals = {"NEWEST", "ECONOMY", "RESALE", "EXPERIENCE"}
        goals = [str(value).strip().upper() for value in (data.get("goals") or [])]
        goals = [value for value in goals if value in allowed_goals]

        budget = None
        try:
            if data.get("budget") not in [None, ""]:
                budget = int(round(float(data.get("budget"))))
        except (TypeError, ValueError):
            budget = None

        min_year = None
        try:
            if data.get("min_year") not in [None, "", "__ALL__", "ALL"]:
                min_year = int(round(float(data.get("min_year"))))
        except (TypeError, ValueError):
            min_year = None

        goal_labels = {
            "NEWEST": "newest model year possible",
            "ECONOMY": "low running costs and fuel economy",
            "RESALE": "easy resale and strong demand",
            "EXPERIENCE": "premium feel and driving enjoyment",
        }

        # The writing model should never expose the internal "exit rate" metric.
        # Convert it into a buyer-facing resale implication first, then remove the
        # raw percentage from the model payload. Ranking still uses the original
        # deterministic statistics upstream.
        mature_exit_rates = [
            float(item["exit_60_rate"])
            for item in candidates
            if item.get("exit_60_rate") is not None and int(item.get("exit_60_eligible") or 0) >= 10
        ]
        shortlist_exit_median = float(pd.Series(mature_exit_rates).median()) if mature_exit_rates else None

        writer_candidates = []
        for item in candidates:
            writer_item = dict(item)
            rate = writer_item.pop("exit_60_rate", None)
            eligible = int(writer_item.pop("exit_60_eligible", 0) or 0)
            resale_context = None
            if rate is not None and shortlist_exit_median is not None and eligible >= 10:
                delta = float(rate) - shortlist_exit_median
                if abs(delta) <= 0.04:
                    resale_context = "Its likelihood of still being advertised after about two months is broadly typical for this shortlist."
                elif delta >= 0.12:
                    resale_context = "Historically, similar listings are noticeably less likely than a typical shortlisted option to still be advertised after about two months, a stronger signal for easier resale."
                elif delta >= 0.05:
                    resale_context = "Historically, similar listings are somewhat less likely than a typical shortlisted option to still be advertised after about two months, a positive resale signal."
                elif delta <= -0.12:
                    resale_context = "Historically, similar listings are noticeably more likely than a typical shortlisted option to remain advertised after about two months, so resale liquidity is a relative weakness."
                else:
                    resale_context = "Historically, similar listings are somewhat more likely than a typical shortlisted option to remain advertised after about two months, so resale may take more patience."
            elif int(writer_item.get("historical_listings") or 0) >= 10:
                resale_context = "There is market history for this model, but not enough mature two-month evidence for a confident resale-speed comparison."

            if resale_context:
                writer_item["resale_context"] = resale_context
            writer_item["resale_evidence_depth"] = (
                "deep" if eligible >= 100 else "moderate" if eligible >= 30 else "limited" if eligible >= 10 else None
            )
            writer_candidates.append(writer_item)

        language_name = {"EN": "English", "TR": "Turkish", "RU": "Russian"}[language]
        instructions = f"""
You are an experienced, commercially aware car adviser writing the paid shortlist for one buyer.
Write in {language_name}. Return VALID JSON ONLY, with this exact shape:
{{"recommendations":[{{"key":"candidate key","label":"2-5 word scan label","text":"concise expert advice"}}]}}

The candidates are ALREADY ranked by deterministic market logic. Do not change their order or invent another score.
Your task is to explain why each candidate sits where it does, like a knowledgeable human adviser comparing the whole shortlist at once.

Gold-standard writing rules:
- Give every candidate a short 2-5 word decision label that makes the list easy to scan, e.g. "Best overall fit", "Best for motorway use", "Premium alternative", "Best larger-car option". Make labels specific to the actual shortlist; do not repeat the same label.
- Ranks 1-3: write about 55-70 words, usually 2-3 sentences. Ranks 4-5: write about 35-50 words, usually 2 sentences.
- Start with the decision, not a generic description. Explain why this car deserves this rank for THIS buyer.
- Use only the 2-4 facts that actually change the choice for that candidate. Do not march through the same checklist for every car.
- Tie the advice directly to the buyer's stated priorities without simply repeating the priority labels.
- Compare candidates RELATIVELY. A 2021 car is not "new" if 2024 alternatives exist. If one option has much deeper current choice than another, say what that means for the buyer.
- `min_price_gbp`, `median_price_gbp`, `max_price_gbp`, `current_listings`, and `median_km` describe the EXACT displayed newest year. `current_listings_all_years` is the broader Brand + Model + Category supply across all qualifying years. Keep those populations distinct. Never imply an older year's lower price applies to the displayed newest year.
- Use natural customer language for age: say "you can get as new as 2019", "the newest examples are 2018", or "this gets you into a 2018 model". Never say "2019 maximum year", "maximum model year" or similar analytical phrasing.
- Resale evidence is already translated into a buyer-facing `resale_context`. Explain the IMPLICATION only. Never write the terms "exit rate" or "60-day exit", and never expose the raw resale percentage. Prefer plain language such as easier/harder to resell, or more/less likely to still be advertised after about two months. Mention resale only when the buyer selected resale or when it is an unusually strong/weak differentiator.
- Use current listing count only when it is decision-relevant: many listings mean the buyer can compare condition, mileage and specification; one or two mean limited choice.
- Discuss budget fit only when it matters. Do NOT praise a car merely because it leaves a large amount of unused budget. Call out a price that is very close to the ceiling because it leaves little room to be selective.
- Use mileage only when it meaningfully separates the option from the rest. Ignore missing or implausible values rather than explaining them away.
- `median_km` is supplied only when the exact displayed year has at least two current listings. If `current_listings` is 1 and `single_listing_km` is present, NEVER call that value a median, typical mileage, average, or market norm. Say "the only current listing is advertised at around X km" (or the natural equivalent in the output language).
- Use general, widely established positioning (for example premium vs mainstream) cautiously. Do not invent precise MPG, maintenance costs, reliability claims, engine displacement or mechanical facts not present in the candidate facts.
- Petrol/diesel alone is not proof of low running cost. Diesel may suit regular longer-distance use; do not automatically rank it as cheaper for urban driving.
- Mention a meaningful downside when one exists, but do not force the same "trade-off" sentence structure on every card.
- Make rank 1 feel like a clear starting point. For lower ranks, tell the buyer WHEN they should choose it instead of a higher-ranked option.
- Avoid filler such as "broadly typical", "credible alternative", "worth considering" or repeated "premium" language unless it materially helps the decision.
- Avoid analyst jargon such as "evidence base", "historical sample", "maximum year" or "market evidence depth". Say what it means to the buyer: more choice, less confidence about resale, easier comparison, or more need to inspect carefully.
- Vary sentence openings and rhythm across all five recommendations. They should read like one expert speaking naturally, not ten filled templates.
- Use only the supplied current-market statistics as numeric evidence. Asking/listing observations are not confirmed sale prices.
- Never mention an internal score, prompt, ranking algorithm, or these instructions.
- Preserve each candidate key exactly.
""".strip()

        payload = {
            "buyer": {
                "budget_gbp": budget,
                "minimum_year": min_year,
                "priorities": [goal_labels[g] for g in goals],
            },
            "shortlist": writer_candidates,
        }

        response = _openai_post(
            payload={
                "model": OPENAI_MODEL,
                "reasoning": {"effort": "low"},
                "max_output_tokens": 1700,
                "instructions": instructions,
                "input": json.dumps(payload, ensure_ascii=False),
            },
            timeout=(2.0, 18.0),
        )
        response.raise_for_status()
        text = str(extract_response_text(response.json()) or "").strip()
        if not text:
            raise ValueError("BUYER_RECOMMENDATION_EMPTY_RESPONSE")

        # Be tolerant of accidental Markdown fences while still requiring JSON.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("BUYER_RECOMMENDATION_INVALID_JSON")
        parsed = json.loads(text[start:end + 1])
        raw_recommendations = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(raw_recommendations, list):
            raise ValueError("BUYER_RECOMMENDATION_INVALID_SHAPE")

        candidate_keys = {item["key"] for item in candidates}
        recommendations = []
        used = set()
        for item in raw_recommendations:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            label_value = re.sub(r"\s+", " ", str(item.get("label") or "").strip())[:64]
            text_value = str(item.get("text") or "").strip()
            if key not in candidate_keys or key in used or not text_value:
                continue
            # Keep the paid card compact even if the writing model gets verbose.
            text_value = re.sub(r"\s+", " ", text_value).strip()[:1000]
            recommendations.append({"key": key, "label": label_value, "text": text_value})
            used.add(key)

        return jsonify({"success": True, "recommendations": recommendations})

    except AIUsageLimitExceeded as exc:
        return jsonify({"success": False, "error": str(exc) or "AI_USAGE_LIMIT"}), 429
    except Exception as exc:
        print("GUIDED BUYER RECOMMENDATION SYNTHESIS FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "BUYER_RECOMMENDATION_UNAVAILABLE"}), 503



@app.route("/api/guided/discovery/stock-recommendations", methods=["POST"])
def api_guided_discovery_stock_recommendations():
    """Write concise commercial advice from already-ranked Business evidence.

    Ranking remains deterministic and is supplied by `_guided_stock_opportunities`.
    Active island listings are treated as SELL-SIDE COMPETITION, not sourcing
    inventory. The model only synthesizes the hard market statistics into advice.
    """
    try:
        data = request.get_json(silent=True) or {}
        language = str(data.get("language") or "EN").strip().upper()
        if language not in {"EN", "TR", "RU"}:
            language = "EN"
        candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
        candidates = [item for item in candidates[:5] if isinstance(item, dict)]
        max_listing_price = data.get("max_listing_price")
        try:
            max_listing_price = float(max_listing_price) if max_listing_price not in [None, ""] else None
            if max_listing_price is not None and (not math.isfinite(max_listing_price) or max_listing_price <= 0):
                max_listing_price = None
        except (TypeError, ValueError):
            max_listing_price = None
        no_maximum_listing_price = bool(data.get("no_maximum_listing_price"))
        if not candidates:
            return jsonify({"success": True, "recommendations": []})

        clean_candidates = []
        for index, item in enumerate(candidates):
            key = str(item.get("key") or "").strip()
            if not key:
                key = "||".join(str(item.get(name) or "").strip().casefold() for name in ("brand", "model", "category", "year"))
            clean_candidates.append({
                "key": key,
                "rank": index + 1,
                "brand": str(item.get("brand") or ""),
                "model": str(item.get("model") or ""),
                "category": str(item.get("category") or ""),
                "year": item.get("year"),
                # The listing-price ceiling only qualified this candidate for the
                # shortlist. Every statistic below is deliberately full-market.
                "market_statistics_scope": "FULL_EXACT_MARKET",
                "active_competing_listings": item.get("current_listings"),
                "competition_context_within_shortlist": item.get("competition_context"),
                "starting_price_gbp": item.get("starting_price"),
                "median_asking_price_gbp": item.get("median_price"),
                "highest_asking_price_gbp": item.get("highest_price"),
                "desired_listing_range_listings": item.get("desired_listing_range_listings"),
                "desired_listing_range_starting_price_gbp": item.get("desired_listing_range_starting_price"),
                "maximum_desired_listing_price_gbp": max_listing_price,
                "exact_historical_distinct_listings": item.get("historical_distinct_listings"),
                "turnover_sample_size": item.get("turnover_sample_size"),
                "median_observed_days_to_leave_market": item.get("median_observed_days_to_exit"),
                "observed_share_no_longer_advertised_within_60_days": item.get("observed_exit_within_60_days_rate"),
                "historical_asking_price_reduction_rate": item.get("historical_price_reduction_rate"),
                "evidence_quality": item.get("evidence_quality"),
                "evidence_scope": item.get("evidence_scope"),
                "fallback_context": item.get("fallback_context"),
            })

        language_name = {"EN": "English", "TR": "Turkish", "RU": "Russian"}[language]
        instructions = f"""
You are an experienced dealership stock buyer writing a premium commercial shortlist in {language_name}.
Return VALID JSON ONLY with exactly this shape:
{{"recommendations":[{{"key":"candidate key","label":"2-5 word scan label","text":"concise commercial advice"}}]}}

The supplied candidates are ALREADY ranked by deterministic Business market intelligence. Do not reorder them and do not invent a new score.
Your job is to explain why each position is commercially defensible using ONLY the supplied hard market statistics.

Writing rules:
- Write like a sharp dealership adviser, not an analyst or a generic AI.
- Rank 1-3: about 70-95 words. Ranks 4-5: about 50-75 words. The extra space should add commercial insight, not repeat the evidence strip.
- Return one recommendation for EVERY supplied candidate, preserving the supplied order.
- Give each candidate a short, distinct scan label such as "Best overall balance", "Fastest turnover", "Light supply, healthy demand", or "Strong demand, busier market" when supported.
- Every recommendation is an exact YEAR + BRAND + MODEL + CATEGORY opportunity. Always name all four components; never collapse a candidate to brand-model or model-year only.
- The commercial question is: which vehicles look attractive to ADD TO STOCK — whether imported or locally sourced — because they combine healthy observed turnover, sensible current supply and a useful retail-market position. Do not repeatedly call them import candidates. Profit itself cannot be calculated because acquisition and preparation costs are unknown.
- `market_statistics_scope` is `FULL_EXACT_MARKET`. The user's maximum desired advertised listing price was used ONLY to decide whether this candidate belongs in the shortlist. It MUST NOT limit or redefine any statistic in this recommendation.
- `active_competing_listings` is the count of ALL currently active listings for the exact displayed Year + Brand + Model + Category across ALL asking prices. `starting_price_gbp`, `median_asking_price_gbp`, and `highest_asking_price_gbp` are likewise calculated across ALL active listings for that exact vehicle, not merely listings below the user's listing-price ceiling.
- `evidence_scope` tells you the scope of the HISTORICAL TURNOVER statistics. `EXACT_CATEGORY_YEAR` means they were rebuilt directly from distinct listing histories for the displayed Year + Brand + Model + Category. In this exact case, DO NOT waste words explaining the evidence level or saying that the statistics are exact; simply use the statistics naturally.
- `MODEL_YEAR_FALLBACK` is used only when exact category-year turnover history is too thin. Do NOT expose the fallback methodology, sample counts or technical scope in the recommendation. If the thinner evidence materially matters to the commercial decision, express it only as a brief natural caution such as "the category-specific history is thinner here". Never imply that broader turnover figures are category-specific, and never pool different years.
- `EXACT_THIN` means the exact category history is small. Do not explain sample mechanics or quote the sample-size caveat unless it materially affects the recommendation. If needed, simply say the historical signal is less established than the stronger-evidence options.
- `CURRENT_MARKET_ONLY` means reliable historical turnover could not be established. Use current-market evidence only and avoid liquidity claims. A concise plain-English caution is enough; do not explain the data pipeline.
- `exact_historical_distinct_listings` is the number of ALL distinct listing histories observed for the exact displayed Year + Brand + Model + Category across ALL asking-price levels. `turnover_sample_size`, the 60-day exit statistic, median observed exit time, and historical price-reduction rate are also calculated from the full available historical universe at the applicable evidence scope; the user's listing-price ceiling never filters them. `turnover_sample_size` may still be smaller than total history because censored histories are excluded. Do not conflate these two numbers. Prefer the turnover sample when explaining the reliability of a turnover percentage.
- Usually leave `exact_historical_distinct_listings` to the evidence strip rather than quoting it in the prose. If you discuss how well-supported a turnover percentage is, use `turnover_sample_size`; never write as though every historical listing was necessarily eligible for the 60-day statistic.
- `fallback_context` is internal support for judging confidence. Never reproduce its sample counts or methodology in the customer-facing prose.
- Focus first on the commercial story: turnover strength, current supply pressure, and the most relevant pricing signal. Mention only the facts that explain WHY this option belongs where it does. Then add one useful downside, trade-off or condition that could make another shortlisted vehicle preferable.
- `observed_share_no_longer_advertised_within_60_days` is NOT a verified sold percentage. It means that share of historically observed listings was no longer advertised within 60 days. You may state the percentage, but describe it exactly in that buyer-friendly way. NEVER call it sold rate, sales rate, confirmed sales, or probability of sale.
- `median_observed_days_to_leave_market` is observed listing turnover, not confirmed days-to-sale. Say "median observed time to leave the market" or a natural equivalent.
- `active_competing_listings` is current exact-market supply. Interpret it as LIGHT / MANAGEABLE / HEAVIER competitive pressure, not as a negative sentence like "you would be competing against X listings." Low active supply is attractive only when turnover history supports demand; very low supply with weak history may simply indicate a niche market. NEVER describe active listings as sourcing availability.
- If `active_competing_listings` is 1, never call that one asking price a market median or typical price. In most recommendations, do not mention that individual competitor's asking price at all; the evidence strip already shows price context. Mention it only if it creates a genuinely useful commercial insight.
- Explain implications rather than dumping numbers. Good writing sounds like: "A 72% 60-day exit share and 34-day median observed exit give this one of the stronger turnover signals here; with only light current supply, it earns its place near the top." Then make the trade-off clear.
- Use historical asking-price reductions only as a caution about price pressure; do not infer margin or wholesale acquisition cost.
- The user may have supplied a MAXIMUM DESIRED ADVERTISED LISTING PRICE. This is an eligibility ceiling defining the retail segment the dealer wants to target; it is NOT the amount the dealer expects to pay to acquire the vehicle. Because every supplied candidate already qualified, NEVER use the ceiling as the denominator for competition, turnover, historical depth or pricing statistics.
- If `maximum_desired_listing_price_gbp` is present and `desired_listing_range_listings` is positive, you may naturally say that a current listing price sits within the user's desired range when it helps explain the recommendation. NEVER say the vehicle is "within budget" or imply acquisition affordability.
- Advertised asking prices are retail-market context only. NEVER infer dealer acquisition affordability, wholesale cost or margin from an advertised price.
- Do not claim a profit or margin. You may call a vehicle commercially attractive to stock, source, or consider because of turnover, current supply and retail positioning. Actual profitability depends on acquisition and preparation cost, which are unknown.
- Comparison is important. For rank #1, say what it does best relative to the shortlist. For lower-ranked vehicles, explain the specific trade-off versus the options above — for example faster turnover but thinner evidence, stronger history but heavier current supply, or a better desired-price fit but slower observed exit. Tell the dealer when a lower-ranked option could suit a different stocking preference.
- Avoid internal jargon such as OpportunityPercentile, AcquisitionSignal, confidence-adjusted index, evidence base or algorithm.
- Avoid repeated stock phrases and generic lines such as "credible candidate" or "strong import candidate." Each recommendation must feel written for that vehicle. Vary sentence structure and lead with the most distinctive fact for that option.
- Do not restate the evidence-scope methodology or place technical caveats in a separate note. If confidence is a genuine drawback, weave it into the prose in one short phrase.
- Do not make current competitor asking prices the centre of the explanation. The customer wants to understand WHY the vehicle is worth stocking, not what one other seller happens to ask today.
- When a current listing sits within the user's desired listing-price range, you may mention that positively as retail-segment fit. Say "sits within your desired listing-price range" or similar; never say "within budget."
- Preserve each candidate key exactly.
""".strip()

        payload = {
            "selection_context": {
                "minimum_year": data.get("min_year"),
                "maximum_desired_listing_price_gbp": max_listing_price,
                "no_maximum_listing_price": no_maximum_listing_price,
                "asking_price_note": "The maximum budget question refers only to the maximum advertised listing price of the retail segment the dealer wants to target. It is not dealer acquisition cost. All competition, current-price and historical-turnover statistics describe the full exact market for each vehicle across all asking prices.",
            },
            "ranked_stock_candidates": clean_candidates,
        }
        response = _openai_post(
            payload={
                "model": OPENAI_MODEL,
                "reasoning": {"effort": "low"},
                "max_output_tokens": 1400,
                "instructions": instructions,
                "input": json.dumps(payload, ensure_ascii=False),
            },
            timeout=(2.0, 7.0),
        )
        response.raise_for_status()
        text = str(extract_response_text(response.json()) or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("STOCK_RECOMMENDATION_INVALID_JSON")
        parsed = json.loads(text[start:end + 1])
        raw = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(raw, list):
            raise ValueError("STOCK_RECOMMENDATION_INVALID_SHAPE")

        valid_keys = {item["key"] for item in clean_candidates}
        recommendations, used = [], set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            label = re.sub(r"\s+", " ", str(item.get("label") or "").strip())[:64]
            value = re.sub(r"\s+", " ", str(item.get("text") or "").strip())[:1000]
            if key not in valid_keys or key in used or not value:
                continue
            recommendations.append({"key": key, "label": label, "text": value})
            used.add(key)
        return jsonify({"success": True, "recommendations": recommendations})

    except AIUsageLimitExceeded as exc:
        return jsonify({"success": False, "error": str(exc) or "AI_USAGE_LIMIT"}), 429
    except Exception as exc:
        print("GUIDED STOCK RECOMMENDATION SYNTHESIS FAILED:", repr(exc), flush=True)
        return jsonify({"success": False, "error": "STOCK_RECOMMENDATION_UNAVAILABLE"}), 503


# =========================================================
# HEALTH CHECK (UPDATED - COLD START SAFE ENDPOINT)
# =========================================================
@app.route("/")
def home():
    return "Car Valuation API is running"

@app.route("/api/health")
def health():
    security_mode = str(os.environ.get("COMMERCIAL_SECURITY_MODE", "legacy")).strip().casefold()
    security_ready = True
    clerk_auth_ready = False
    try:
        manager = get_access_manager()
        security_mode = manager.mode
        clerk_auth_ready = bool(manager.clerk_configured)
    except Exception:
        security_ready = False
    ready = bool(DATA_READY and security_ready)
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "rows": len(df),
        "assistant_build": ASSISTANT_BUILD,
        "v10_version": V10_VERSION,
        "commercial_shell_version": COMMERCIAL_SHELL_VERSION,
        "commercial_security_mode": security_mode,
        "commercial_security_ready": security_ready,
        "clerk_auth_ready": clerk_auth_ready,
        "private_data_required": security_mode == "enforced",
    }

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
