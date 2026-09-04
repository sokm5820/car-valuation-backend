from flask import Flask, request, jsonify
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

# AI interpreter configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

import gspread
from google.oauth2.service_account import Credentials

app = Flask(__name__)
CORS(app)

# -----------------------
# GOOGLE SHEETS CONNECTION
# -----------------------

GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

credentials = Credentials.from_service_account_info(
    json.loads(GOOGLE_CREDENTIALS),
    scopes=SCOPES
)

gc = gspread.authorize(credentials)

sheet = gc.open("North Cyprus Vehicle Leads").sheet1

# -----------------------
# LOAD DATA (GITHUB CSV SOURCE - SAFE VERSION)
# -----------------------
CSV_URL = "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/ads_base.csv"

df = pd.DataFrame()  # safe default

# 🔥 ADDED: readiness flag
DATA_READY = False

def load_data():
    global df, DATA_READY
    try:
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

MARKET_CSV_URL = "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/market_base.csv"

market_df = pd.DataFrame()
MARKET_READY = False

# =========================================================
# AI BUYING ASSISTANT - BUYER INTELLIGENCE v1
# =========================================================
BUYER_MODEL_CSV_URL = (
    "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
    "buyer_model_intelligence.csv"
)
BUYER_CATEGORY_CSV_URL = (
    "https://raw.githubusercontent.com/sokm5820/car-valuation-backend/main/"
    "buyer_category_intelligence.csv"
)

buyer_model_df = pd.DataFrame()
buyer_category_df = pd.DataFrame()
BUYER_INTELLIGENCE_READY = False


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

            if normalized:
                clean[key] = normalized

    return clean


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
- Do not translate brand/model names.
- Do not turn subjective ideas such as reliable, sporty, economical,
  family-friendly, small, luxurious, or good value into unsupported hard
  filters. Put those concepts in "preferences".
- IMPORTANT: vehicle body/type requests such as SUV, crossover, pick-up, small car,
  normal car/automobile, motorcycle or scooter are SOFT preferences, not Category filters.
  Put them in "preferences" using the canonical vehicle_type tags below. Never place
  SUV/pick-up/motorcycle/etc. into categories unless that exact value is explicitly
  confirmed as a real market Category value.
- Use short canonical preference tags whenever possible so they persist cleanly across turns:
  vehicle_type:SUV, vehicle_type:crossover, vehicle_type:pickup, vehicle_type:small_car,
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

    user_payload = {
        "language": language,
        "latest_message": message,
        "current_filters": current_filters,
        "recent_conversation": sanitize_conversation_history(conversation_history),
        "market_context": market_context,
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

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": instructions,
            "input": json.dumps(user_payload, ensure_ascii=False),
        },
        timeout=30,
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
    colors=None,
    min_year=None,
    max_year=None,
    min_km=None,
    max_km=None,
    limit=20,
    max_limit=100
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
        filtered = filtered[
            contains_any(
                filtered["Transmission"],
                transmissions
            )
        ]

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

    results_df = filtered.head(limit).copy()

    # -----------------------
    # JSON-SAFE RESULTS
    # -----------------------

    results = []

    for _, row in results_df.iterrows():

        results.append({
            "brand": row["Brand"],
            "model": row["Model"],
            "category": row["Category"],

            "year": (
                int(row["Year"])
                if pd.notna(row["Year"])
                else None
            ),

            "price": (
                float(row["Price"])
                if pd.notna(row["Price"])
                else None
            ),

            "km": (
                int(row["KM"])
                if pd.notna(row["KM"])
                else None
            ),

            "company": row["Company"],
            "location": row["Location"],
            "transmission": row["Transmission"],
            "color": row["Color"],
            "image": row["Image"],
            "link": row["Link"]
        })

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


def guided_narrowing_question(filters, preferences, count, language="TR"):
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
            "kms": [],
            "transmissions": set(),
            "locations": set(),
        })
        bucket["count"] += 1

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
    """Qualify/rank only real model families against soft intent and vehicle type."""
    preferences = list(preferences or [])
    relevant_preferences = [
        p for p in preferences
        if str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
    ]

    model_market = _group_market_models(results)
    if not model_market:
        return [], [], []

    cache_key = _qualification_cache_key(filters, preferences, model_market)
    cached = _MODEL_QUALIFICATION_CACHE.get(cache_key)
    if cached is not None:
        selected_keys, reasons, selected_summaries = cached
        selected_set = set(selected_keys)
        qualified = [
            item for item in (results or [])
            if (
                str(item.get("brand") or "").strip().casefold(),
                str(item.get("model") or "").strip().casefold(),
            ) in selected_set
        ]
        return qualified, list(reasons), list(selected_summaries)

    # If there is no soft preference, every hard-filtered model family remains eligible.
    if not relevant_preferences:
        return list(results or []), [], model_market

    instructions = """
You are the model-qualification layer for a North Cyprus vehicle buying assistant.

The supplied model_market contains ONLY model families that already satisfy the buyer's
hard filters such as budget, year, mileage, brand, transmission and location.
Use general automotive knowledge only to qualify and rank these REAL model families
against the buyer's soft intent.

Return JSON only:
{"models":[{"brand":"...","model":"...","reason":"..."}]}

Rules:
- Select ONLY exact brand/model pairs present in model_market.
- vehicle_type:SUV / crossover / pickup / motorcycle / small_car is a REQUIRED qualification.
  Never include the wrong body/vehicle type.
- priority:economy: favour model families generally associated with economical use/ownership.
- priority:luxury: favour premium/luxury-positioned brands or model families.
- priority:reliability, performance, comfort and practicality are model-level positioning only.
- use_case:family and use_case:commute may be used as broad model-level reasoning.
- Never make claims about the condition or quality of an individual advertised vehicle.
- Select up to 20 qualifying model families, ordered by relevance.
- Keep reason extremely short and neutral. It is internal context, not advertising copy.
"""

    payload = {
        "language": language,
        "latest_message": message,
        "hard_filters": filters,
        "soft_preferences": relevant_preferences,
        "model_market": model_market,
    }

    response = None
    last_error = None
    for attempt in range(2):
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "instructions": instructions,
                    "input": json.dumps(payload, ensure_ascii=False),
                },
                timeout=45,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.6)

    if response is None or not response.ok:
        print(f"Model qualification request failed: {last_error}")
        return [], [], []

    response_text = extract_response_text(response.json()).strip()
    if not response_text:
        return [], [], []

    try:
        parsed = json.loads(response_text)
        selected = parsed.get("models", []) if isinstance(parsed, dict) else []
    except Exception:
        return [], [], []

    market_by_key = {
        (m["brand"].casefold(), m["model"].casefold()): m
        for m in model_market
    }
    selected_keys = []
    reasons = []
    selected_summaries = []

    for item in selected:
        if not isinstance(item, dict):
            continue
        brand = str(item.get("brand") or "").strip()
        model = str(item.get("model") or "").strip()
        key = (brand.casefold(), model.casefold())
        if brand and model and key in market_by_key and key not in selected_keys:
            selected_keys.append(key)
            selected_summaries.append(market_by_key[key])
            reasons.append({
                "brand": brand,
                "model": model,
                "reason": str(item.get("reason") or "").strip(),
            })
        if len(selected_keys) >= 20:
            break

    if not selected_keys:
        return [], [], []

    selected_set = set(selected_keys)
    qualified = [
        item for item in (results or [])
        if (
            str(item.get("brand") or "").strip().casefold(),
            str(item.get("model") or "").strip().casefold(),
        ) in selected_set
    ]

    # Cache the qualification for identical market/filter/preference state so the same
    # buyer request does not produce a different eligible model universe on every turn.
    if len(_MODEL_QUALIFICATION_CACHE) >= _MODEL_QUALIFICATION_CACHE_MAX:
        _MODEL_QUALIFICATION_CACHE.pop(next(iter(_MODEL_QUALIFICATION_CACHE)))
    _MODEL_QUALIFICATION_CACHE[cache_key] = (
        tuple(selected_keys),
        tuple(reasons),
        tuple(selected_summaries),
    )

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
            newest = int(x.get("newest_year") or 0)
            start = float(x.get("starting_price") or 10**12)
            budget_distance = abs(ceiling - start) if ceiling is not None else start
            return (-newest, budget_distance, -int(x.get("count") or 0), x["brand"].casefold(), x["model"].casefold())

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
        "suv": {"suv, pick-up"},
        "pickup": {"suv, pick-up"},
        "pick-up": {"suv, pick-up"},
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



def generate_grounded_market_answer(message, language, filters, preferences, search_result, conversation_history=None, decision_mode="DISCOVER"):
    """Progressive-disclosure buying advice grounded in deterministic market data."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY_NOT_CONFIGURED")

    hard_count = int(search_result.get("count", 0) or 0)
    hard_results = search_result.get("results", []) or []

    if hard_count == 0:
        fallback = {
            "TR": "Bu kriterlere uyan aktif ilan bulamadım. İsterseniz bütçe, yıl, kilometre veya diğer kriterlerden birini esnetebiliriz.",
            "EN": "I couldn't find an active listing matching those criteria. We can loosen the budget, year, mileage or another filter.",
            "RU": "Я не нашёл активных объявлений по этим критериям. Можно немного ослабить бюджет, год, пробег или другой фильтр.",
        }
        return fallback.get(language, fallback["TR"]), [], 0, []

    qualified_results, model_reasons, qualified_summaries = shortlist_models_for_preferences(
        message=message,
        language=language,
        filters=filters,
        preferences=preferences,
        results=hard_results,
    )

    has_soft_pref = any(
        str(p).casefold().startswith(("vehicle_type:", "priority:", "use_case:"))
        for p in (preferences or [])
    )

    if has_soft_pref and not qualified_results:
        fallback = {
            "TR": "Belirttiğiniz kriterlerle bu tercihi karşılayan bir model bulamadım. İsterseniz bütçe, yıl, kilometre veya araç tipi kriterlerinden birini esnetebiliriz.",
            "EN": "I couldn't find a model that matches both those filters and that preference. We can loosen the budget, year, mileage or vehicle type.",
            "RU": "Я не нашёл модель, которая одновременно соответствует этим фильтрам и предпочтению. Можно ослабить бюджет, год, пробег или тип автомобиля.",
        }
        return fallback.get(language, fallback["TR"]), [], 0, []

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
    decision_mode = str(decision_mode or "DISCOVER").upper()
    if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
        decision_mode = "DISCOVER"

    listing_candidate_limit = 12 if decision_mode == "SHOP" else 3
    listing_candidates = select_assistant_candidates(
        advisory_results, filters, max_candidates=listing_candidate_limit
    )

    instructions = """
You are a premium, neutral vehicle-buying decision assistant for North Cyprus.
Think like a helpful expert conversation, not a search form and not a salesperson.

CORE BEHAVIOUR — PREMIUM DECISION ASSISTANT:
1. Decide whether the buyer has supplied enough information to make a useful answer.
   - Too broad: "I have £18k, what should I buy?" -> ask ONE high-value follow-up question.
   - Specific enough: "I have £18k and want an SUV" -> answer now with a substantial market overview.
   Never ask a follow-up merely to reduce an already-useful answer to an arbitrary 3 models.
2. When specific enough, surface a BROAD, useful set of model families from model_options.
   - Normally show 8-12 options when that many exist.
   - If the buyer explicitly asks for ALL options, show all supplied model_options (up to 20).
   - If the buyer asks "more?", use recent_conversation to avoid repeating models already shown and
     show additional supplied options. If none remain, say so plainly.
3. Answer the user's actual question FIRST. Follow-up questions are optional, not mandatory.
   Do not create permission loops ("shall I...?", "would you like...?") when the user has already
   asked for the action.
4. Short replies inherit conversational meaning from recent_conversation:
   - "yes" executes the action just offered.
   - "all of them" means all currently relevant options.
   - "Mercedes?" / "No Mercedes?" means check Mercedes within active criteria, not exclude it.
   - a model name such as "GLA" means deepen into that model's current market.
5. Progressive depth:
   broad insufficient request -> one clarifying question
   sufficient request -> broad model-family overview
   chosen brand/model -> useful model-market summary
   explicit listing request -> concrete listings
   specific listing question -> factual listing discussion
6. For a chosen model, give genuinely useful market context from the supplied facts: matching count,
   newest matching year, price of the cheapest listing IN THAT NEWEST YEAR, overall asking-price
   floor/range, mileage range if available, and transmissions if available. Do not merely repeat one
   shallow "newest year + overall starting price" line.
7. Individual listing mode is for explicit listing requests. Show up to 3 by default; show more/all
   only when the user explicitly asks. Never print raw URLs unless requested.

DECISION MODE — AUTHORITATIVE:
- decision_mode is application state, not a suggestion. Follow it.
- DISCOVER: help the buyer understand the model-family opportunity set. Do not drift into individual listings.
- COMPARE: directly evaluate the model(s) under discussion. Use Buyer Intelligence when relevant to explain transparent trade-offs such as current supply, observed turnover and asking-price pressure. Do not collapse the comparison into a universal score.
- SHOP: work at individual-listing level using listing_candidates. Default to 3 listings unless the latest message explicitly asks for more/all. Keep every listing statement factual and grounded.
- Never change modes yourself. The application has already classified the turn.

NEUTRALITY:
- Never tell the buyer to buy a specific advertised vehicle.
- Never call a listing good, safe, reliable, problem-free, high-quality, best value or mechanically sound.
- Do not rank individual listings as "best".
- If asked which vehicle/model is definitely reliable, problem-free, safe, guaranteed, or which one
  they should definitely buy, DO NOT substitute different models based on general reputation and do
  not endorse one. State plainly that you cannot determine a definite purchase choice or guaranteed
  condition from these data. Then offer an objective comparison of ONLY the models already under
  discussion. Mention inspection/service history only as a sensible verification step, not as proof.
- Model-level positioning may be discussed cautiously when the buyer asks for comparison or a
  soft preference such as economical, premium/luxury, practical, sporty or family-oriented.
- During initial discovery, keep model lines factual: model name, newest matching year and starting
  asking price. Do not append sales-like adjectives such as balanced, quality, comfortable, best,
  low ownership cost, good choice or driving-focused unless the buyer explicitly asks to compare
  characteristics and the statement is clearly model-level general context.
- If the buyer says "economic/economical/ekonomik", help identify suitable real model families.
- If they say "luxury/premium/lüks", help identify premium-positioned real model families.
- Hard facts about the current market always override general automotive knowledge.

BUYER INTELLIGENCE:
- model_options may contain buyer_intelligence and variant_intelligence calculated from OtoDeğer's
  validated current + historical Cyprus-market datasets.
- Treat these fields as proprietary factual evidence, not as general automotive knowledge.
- Use historical evidence to improve recommendations when it is relevant to the buyer's decision.
- "exit_60_rate" means the observed share of a mature historical listing cohort that left the observed
  market within 60 days. It does NOT prove the vehicles were sold.
- "median_observed_days_to_exit" is observed listing duration before market exit; do not call it
  guaranteed "days to sell".
- "price_reduction_rate" is the share of eligible historical listings whose asking price was reduced.
  It is evidence of asking-price pressure, NOT depreciation or value retention.
- Respect liquidity_confidence / price_pressure_confidence. Avoid strong conclusions from LOW or
  INSUFFICIENT evidence. Mention limited evidence when it materially affects a comparison.
- liquidity_evidence_level / price_pressure_evidence_level tells you whether CATEGORY or MODEL
  evidence is being used. Category evidence is more specific; MODEL means the category sample was
  too thin and the system deliberately fell back upward.
- Never convert listing volume into a claim of popularity.
- Never invent a universal resale score, reliability score or recommendation score.
- variant_intelligence may be used to distinguish variants within the same model when the data
  supports it. Do not invent a variant that is absent from variant_intelligence.
- Prefer useful relative conclusions ("historically faster observed turnover", "more current supply",
  "less asking-price reduction pressure") backed by supplied evidence over dumping statistics.
- When the buyer asks about resale/liquidity, use the historical evidence directly and describe it as
  observed market behaviour, not a guarantee of future resale.
- When historical intelligence is unavailable, simply omit that claim rather than filling it with
  general knowledge.

GROUNDING:
- model_options and listing_candidates come from the live deterministic market search.
- Never invent a model, price, year, mileage, seller, location, transmission or count.
- Never mention a model that is not in model_options during discovery.
- "starting_price" is the lowest asking price across ALL matching listings for that model.
- "starting_price_year" is the model year attached to that overall cheapest asking-price level.
- "newest_year" is the newest matching model year.
- "newest_year_starting_price" is the cheapest asking price specifically among listings of newest_year.
- These facts are intentionally separate. NEVER write "up to 2021, starting from £11,000" when
  £11,000 belongs to an older year. That wording falsely implies a 2021 can be bought for £11,000.
- Preferred concise model line in English:
  Mazda CX-3 — newest: 2021 from £X · overall from £Y · N listings
  Omit any component whose supplied value is null.
- Equivalent natural phrasing should be used in Turkish/Russian.
- If a requested vehicle type is present, all surfaced models must satisfy it.

MONEY:
- Every supplied market price is GBP.
- Always format as £18.000 / £17.500 style in Turkish, and appropriate thousands formatting in English/Russian.
- Never output $, USD, EUR, €, TL or another currency for supplied market prices.

LANGUAGE — IMPORTANT:
- Reply in the language of the user's latest message. Do not keep replying in Turkish merely because
  earlier UI state or conversation content was Turkish.
- If the latest message is English, answer in natural English. If Turkish, answer in natural Turkish.
  If Russian, answer in natural Russian.

FORMAT — IMPORTANT:
- Clean, premium chat formatting. No markdown bullets, numbering, asterisks or bold markers.
- For an option overview use exactly:
  one concise intro paragraph
  BLANK LINE
  one model per line, with NO blank lines between model lines
  BLANK LINE
  one concise synthesis/next-step paragraph only when it adds value
- Keep terminology stable inside one conversation. Prefer "newest" + "overall from" consistently in
  English rather than alternating among "up to", "newest year" and "starting from".
- Do not hide useful options merely to make the answer shorter.
- Do not explain the system or database to the buyer.
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
        "listing_candidates": listing_candidates,
        "listing_display_limit": 3 if decision_mode == "SHOP" else 0,
        "instruction_note": (
            "Follow decision_mode exactly. In DISCOVER, surface useful model options. In COMPARE, directly "
            "compare/evaluate the model(s) under discussion using grounded market and Buyer Intelligence evidence. "
            "In SHOP, use individual listing facts; default to three unless more/all was explicitly requested. "
            "All prices are GBP."
        ),
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False),
        },
        timeout=45,
    )
    response.raise_for_status()

    answer = extract_response_text(response.json()).strip()
    if not answer:
        raise ValueError("AI_ASSISTANT_EMPTY_RESPONSE")

    answer = answer.replace("$", "£").replace(" USD", " GBP").replace("USD ", "GBP ")
    answer = normalize_assistant_format(answer)
    return answer, advisory_results, advisory_count, model_options


@app.route("/api/assistant", methods=["POST"])
def api_ai_buying_assistant():
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

        interpretation = interpret_market_query(
            message=message,
            current_filters=current_filters,
            language=language,
            conversation_history=conversation_history,
        )

        decision_mode = str(
            interpretation.get("decision_mode") or "DISCOVER"
        ).upper()
        if decision_mode not in {"DISCOVER", "COMPARE", "SHOP"}:
            decision_mode = "DISCOVER"

        next_filters = apply_interpretation_to_filters(
            current_filters,
            interpretation,
        )

        next_preferences = merge_preferences(
            current_preferences,
            interpretation.get("preferences", []),
        )

        if (
            interpretation.get("needs_clarification")
            and interpretation.get("clarification_question")
            and (interpretation.get("filters") or interpretation.get("clear_filters"))
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
            })

        # Pull the full filtered result set for candidate selection. The public
        # response is still capped below so the frontend is not flooded with rows.
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
        )

        if not search_result.get("success"):
            return jsonify(search_result), 503

        # Guided buying flow: broad searches get a compact group of useful
        # narrowing dimensions; only later do we offer secondary refinements.
        guide_question = guided_narrowing_question(
            filters=next_filters,
            preferences=next_preferences,
            count=search_result.get("count", 0),
            language=language,
        )

        if guide_question:
            public_results = (search_result.get("results") or [])[:100]
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
            })

        answer, advisory_results, advisory_count, model_options = generate_grounded_market_answer(
            message=message,
            language=language,
            filters=next_filters,
            preferences=next_preferences,
            search_result=search_result,
            conversation_history=conversation_history,
            decision_mode=decision_mode,
        )

        public_results = (advisory_results or search_result.get("results") or [])[:100]
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
            "decision_mode": decision_mode,
            "stage": "recommendation",
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
            "OPENAI ASSISTANT HTTP ERROR:",
            status_code,
            response_text,
            flush=True
        )

        return jsonify({
            "success": False,
            "error": "AI_ASSISTANT_FAILED",
            "openai_status": status_code,
            "openai_message": response_text
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
# AI BUYING ASSISTANT - MARKET SEARCH API
# =========================================================

@app.route("/api/search", methods=["POST"])
def api_market_search():

    try:
        data = request.json or {}

        result = market_search(
            budget=data.get("budget"),
            min_budget=data.get("min_budget"),

            brands=data.get("brands"),
            exclude_brands=data.get("exclude_brands"),

            models=data.get("models"),
            exclude_models=data.get("exclude_models"),

            categories=data.get("categories"),
            exclude_categories=data.get("exclude_categories"),

            locations=data.get("locations"),
            exclude_locations=data.get("exclude_locations"),

            companies=data.get("companies"),
            exclude_companies=data.get("exclude_companies"),

            transmissions=data.get("transmissions"),
            colors=data.get("colors"),

            min_year=data.get("min_year"),
            max_year=data.get("max_year"),

            min_km=data.get("min_km"),
            max_km=data.get("max_km"),

            limit=data.get("limit", 20)
        )

        return jsonify(result)

    except (TypeError, ValueError) as e:
        print("INVALID MARKET SEARCH:", e)

        return jsonify({
            "success": False,
            "error": "INVALID_SEARCH_PARAMETERS",
            "count": 0,
            "returned": 0,
            "results": []
        }), 400

    except Exception as e:
        print("MARKET SEARCH FAILED:", e)

        return jsonify({
            "success": False,
            "error": "SEARCH_FAILED",
            "count": 0,
            "returned": 0,
            "results": []
        }), 500

# =========================================================
# AI BUYING ASSISTANT - MARKET OPTIONS
# =========================================================

@app.route("/api/market_options", methods=["GET"])
def market_options():
    if not MARKET_READY or market_df is None or market_df.empty:
        return jsonify({
            "success": False,
            "error": "MARKET_DATA_NOT_READY"
        }), 503

    try:
        brands = sorted(
            market_df["Brand"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        models = sorted(
            market_df["Model"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        locations = sorted(
            market_df["Location"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        transmissions = sorted(
            market_df["Transmission"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        companies = sorted(
            market_df["Company"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )

        return jsonify({
            "success": True,
            "brands": brands,
            "models": models,
            "locations": locations,
            "transmissions": transmissions,
            "companies": companies
        })

    except Exception as e:
        print("MARKET OPTIONS FAILED:", e)

        return jsonify({
            "success": False,
            "error": "MARKET_OPTIONS_FAILED"
        }), 500

# =========================================================
# AI BUYING ASSISTANT - HEALTH CHECK
# =========================================================

@app.route("/api/market_health", methods=["GET"])
def market_health():

    return jsonify({
        "status": "ok" if MARKET_READY else "loading",
        "ready": MARKET_READY,
        "rows": len(market_df),
        "buyer_intelligence_ready": BUYER_INTELLIGENCE_READY,
        "buyer_model_rows": len(buyer_model_df),
        "buyer_category_rows": len(buyer_category_df),
    })

# =========================================================
# LEAD SUBMISSION
# =========================================================
@app.route("/submit_lead", methods=["POST"])
def submit_lead():
    try:
        data = request.json or {}

        # -----------------------
        # GET SUBMITTED DATA
        # -----------------------
        year = data.get("year")
        brand = data.get("brand")
        model = data.get("model")
        category = data.get("category")

        valuation = data.get("valuation")
        min_price = data.get("min_price")
        max_price = data.get("max_price")

        name = str(data.get("name", "")).strip()
        phone = str(data.get("phone", "")).strip()
        consent = data.get("consent")

        # -----------------------
        # VALIDATION
        # -----------------------

        if not name:
            return jsonify({
                "success": False,
                "error": "NAME_REQUIRED"
            }), 400

        if not phone:
            return jsonify({
                "success": False,
                "error": "PHONE_REQUIRED"
            }), 400

        if consent is not True:
            return jsonify({
                "success": False,
                "error": "CONSENT_REQUIRED"
            }), 400

        if not year or not brand or not model or not category:
            return jsonify({
                "success": False,
                "error": "VEHICLE_INFO_INCOMPLETE"
            }), 400

        # -----------------------
        # DATES
        # -----------------------

        submitted_at = pd.Timestamp.now()

        expires_at = submitted_at + pd.Timedelta(days=90)

        # -----------------------
        # ADD LEAD TO GOOGLE SHEET
        # -----------------------

        sheet.append_row([
            submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
            year,
            brand,
            model,
            category,
            valuation,
            min_price,
            max_price,
            name,
            phone,
            "TRUE",
            expires_at.strftime("%Y-%m-%d %H:%M:%S")
        ])

        # -----------------------
        # SUCCESS
        # -----------------------

        return jsonify({
            "success": True
        })

    except Exception as e:
        print("LEAD SUBMISSION FAILED:", e)

        return jsonify({
            "success": False,
            "error": "SUBMISSION_FAILED"
        }), 500

# =========================================================
# HEALTH CHECK (UPDATED - COLD START SAFE ENDPOINT)
# =========================================================
@app.route("/")
def home():
    return "Car Valuation API is running"

@app.route("/api/health")
def health():
    return {
        "status": "ok" if DATA_READY else "loading",
        "ready": DATA_READY,
        "rows": len(df)
    }

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)