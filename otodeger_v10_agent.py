"""OtoDeğer V10 decision-agent orchestration.

Principle: AI understands intent and writes naturally; Python owns state,
retrieval, calculations, evidence and guardrails.

This module deliberately contains no Flask routes and no valuation mutations.
`app.py` injects its existing deterministic market/business helpers through a
plain host dictionary so V10 can be tested in isolation.
"""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
import time
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from otodeger_v11_contract import ContractError, NUMBER_RE, parse_number, clarification

from otodeger_v10_state import (
    Action, Audience, Job, ObjectType, Transition,
    InvalidTurnPlan, StateConflict, StateNotFound, StorageUnavailable,
    apply_turn_plan, get_state_service, register_object,
)

V10_VERSION = "11.0-conversation-contract"
ASSISTANT_BUILD = "11.1-boundary-audit"
SUPPORTED_LANGUAGES = {"TR", "EN", "RU"}


class AssistantDataUnavailable(RuntimeError):
    """Search did not establish a trustworthy result set."""


def _checked_search(result):
    if not isinstance(result, Mapping) or result.get('success') is False or result.get('error') or result.get('vehicle_type_filter_unavailable'):
        raise AssistantDataUnavailable('Search unavailable')
    rows = result.get('results')
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise AssistantDataUnavailable('Search returned invalid rows')
    return dict(result)


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if hasattr(value, 'item'):
        return _json_safe(value.item())
    if isinstance(value, datetime):
        return value.isoformat()
    raise ContractError('Unsupported response value')


# ---------------------------------------------------------------------------
# Small safe helpers
# ---------------------------------------------------------------------------

def _lang(value: Any) -> str:
    value = str(value or "EN").upper().strip()
    return value if value in SUPPORTED_LANGUAGES else "EN"


def _text(value: Any, max_len: int = 500) -> str:
    return str(value or "").strip()[:max_len]


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(value: Any) -> Optional[int]:
    x = _finite(value)
    return int(round(x)) if x is not None else None


def _money(value: Any) -> Optional[str]:
    x = _finite(value)
    if x is None:
        return None
    return f"£{x:,.0f}"


def _pct(value: Any) -> Optional[float]:
    x = _finite(value)
    if x is None:
        return None
    return round(x * 100 if abs(x) <= 1.5 else x, 1)


def _json_from_text(raw: str) -> Dict[str, Any]:
    raw = str(raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start:end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass
    return {}


def _response_text(payload: Mapping[str, Any]) -> str:
    for item in payload.get("output", []) or []:
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                return str(content.get("text") or "")
    return ""


def _host(host: Mapping[str, Any], name: str, default=None):
    return host.get(name, default)


def _df_records(frame, limit: int = 100) -> List[Dict[str, Any]]:
    if frame is None:
        return []
    try:
        if frame.empty:
            return []
        records = frame.head(limit).where(frame.head(limit).notna(), None).to_dict("records")
        return [{str(k): v for k, v in r.items()} for r in records]
    except Exception:
        return []


def _state_objects_for_prompt(state: Mapping[str, Any], max_items: int = 24) -> List[Dict[str, Any]]:
    objects = state.get("objects") or {}
    ids: List[str] = []
    for source in [
        ((state.get("focus") or {}).get("object_ids") or []),
        (state.get("shortlist") or []),
        ((state.get("last_result") or {}).get("target_ids") or []),
        ((state.get("offered_action") or {}).get("target_ids") or []),
    ]:
        for oid in source:
            if oid in objects and oid not in ids:
                ids.append(oid)
    for oid in objects:
        if oid not in ids:
            ids.append(oid)
        if len(ids) >= max_items:
            break
    out = []
    for oid in ids[:max_items]:
        obj = objects.get(oid) or {}
        payload = {k: v for k, v in obj.items() if k not in {"id", "type"}}
        # Never send unnecessary full image blobs or verbose metadata to semantic model.
        payload.pop("image", None)
        out.append({"id": oid, "type": obj.get("type"), "payload": payload})
    return out


def _focus_objects(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    objects = state.get("objects") or {}
    out = []
    for oid in ((state.get("focus") or {}).get("object_ids") or []):
        if oid in objects:
            out.append(objects[oid])
    return out


def _model_targets_from_objects(state: Mapping[str, Any], ids: Sequence[str]) -> List[Dict[str, Any]]:
    objects = state.get("objects") or {}
    out = []
    for oid in ids:
        obj = objects.get(oid) or {}
        payload = {k: v for k, v in obj.items() if k not in {"id", "type"}}
        if obj.get("type") in {ObjectType.MODEL.value, ObjectType.LISTING.value, ObjectType.OWNED_VEHICLE.value, ObjectType.STOCK_ITEM.value}:
            brand, model = _text(payload.get("brand"), 100), _text(payload.get("model"), 120)
            if brand and model:
                target = {"brand": brand, "model": model}
                for key in ("year", "category"):
                    if payload.get(key) not in (None, ""):
                        target[key] = payload.get(key)
                out.append(target)
    # stable dedupe
    seen, clean = set(), []
    for item in out:
        key = (item.get("brand", "").casefold(), item.get("model", "").casefold(), item.get("year"))
        if key not in seen:
            seen.add(key); clean.append(item)
    return clean


# ---------------------------------------------------------------------------
# State / semantic interpretation
# ---------------------------------------------------------------------------

def _audience_from_access(access_tier: str) -> str:
    return Audience.BUSINESS.value if str(access_tier or "").upper() == "BUSINESS" else Audience.PERSONAL.value


def _numeric_updates(message, state):
    """Bind numeric spans to units/fields; never use a mileage token as money."""
    low = message.casefold().replace('\u0307', '')
    updates = {}
    money_candidates = []
    budget_words = r"budget|bütçe\w*|butce\w*|бюджет\w*"
    for match in NUMBER_RE.finditer(low):
        before, after = low[max(0,match.start()-55):match.start()], low[match.end():match.end()+45]
        value = parse_number(match.group('number'))
        field = None
        if re.match(r"\s*(?:km\b|kilomet\w*|км\b)", after) or re.search(r"(?:mileage|пробег|kilometre|km)\s*(?:limit|sınırı|siniri|maximum|max|of|to|is|:)?\s*$", before):
            field = 'min_km' if re.search(r"(?:over|above|at least|minimum)\s*$", before) else 'max_km'
        elif 1900 <= value <= 2100 and (re.search(r"(?:from|since|min(?:imum)? year|newer than|не старше|en az)\s*$", before) or re.match(r"\s*(?:onwards|or newer|and newer|ve sonrası|ve sonrasi|или новее)", after)):
            field = 'min_year'
        elif 1900 <= value <= 2100 and re.search(r"(?:max(?:imum)? year|older than|up to year)\s*$", before):
            field = 'max_year'
        elif re.search(r"£\s*$", before) or re.match(r"\s*(?:gbp|pounds?\b|sterlin\w*|фунт\w*)", after) or re.search(rf"(?:{budget_words}|under|below|up to|ceiling|spend|afford|stretch to)\s*(?:is|of|to|at|by|:)?\s*$", before) or re.match(rf"\s*(?:to (?:the )?)?(?:{budget_words})", after):
            money_candidates.append((match, value, before, after))
        if field:
            if not value.is_integer():
                raise ContractError('Year and mileage must be whole numbers')
            updates[field] = int(value)
    for match, value, before, after in money_candidates:
        # Choose the monetary role nearest this token. Unrelated numbers keep their own units.
        if re.search(r"offer(?:ed)?|teklif|предлож", before):
            field = 'offer_price'
        elif re.search(r"asking|advertised|listed|for sale at|ilan fiyat|satış fiyat|satis fiyat|price is", before):
            field = 'asking_price'
        elif state.get('audience') == Audience.BUSINESS.value and re.search(r"buy in|bring in|acquir|stock|offered|tedarik|stok", before):
            field = 'acquisition_price'
        elif re.search(budget_words + r"|under|below|up to|ceiling|spend|afford|stretch|can i buy|what can i buy", low):
            field = 'budget_max'
        else:
            continue
        if field == 'budget_max':
            current = _finite((state.get('constraints') or {}).get(field))
            absolute = re.search(r"(?:increase|raise|reduce|decrease)\s+(?:the\s+)?budget\s+to\s*$", before)
            relative_plus = re.search(r"\badd\b|(?:increase|raise).*\bby\b|ekle|artır|artir|добав|увелич.*\bна\b", low)
            relative_minus = re.search(r"\bsubtract\b|(?:reduce|decrease).*\bby\b|azalt|уменьш.*\bна\b", low)
            if (relative_plus or relative_minus) and not absolute:
                if current is None:
                    raise ContractError('An existing budget is needed for a relative edit')
                if re.match(r"\s*%", after):
                    value = current * value / 100
                value = current + value if relative_plus else current - value
        updates[field] = value
    return updates


def _numeric_followup_plan(message, state):
    """Handle complete numeric replies without a model call; ambiguity asks once."""
    low = message.casefold().strip().rstrip('!?')
    awaiting = clarification(state.get('awaiting')) or {}
    field = awaiting.get('field')
    job = state.get('job')
    values = _numeric_updates(message, state)
    try:
        bare_value = parse_number(low)
    except ContractError:
        bare_value = None
    numeric_fields = {'budget_max','budget_min','max_km','min_km','min_year','max_year'}
    if bare_value is not None:
        if field in numeric_fields:
            values = {field: bare_value}
        elif job == Job.FIND_A_CAR.value and (state.get('constraints') or {}).get('budget_max') is not None and bare_value > 2100:
            values = {'budget_max': bare_value}
        else:
            return {'transition':'CONTINUE','action':'ASK_CLARIFICATION','job':job,
                    'awaiting':{'question':{'EN':'Does that number refer to your budget, model year, or mileage?',
                    'TR':'Bu sayı bütçeniz, model yılı veya kilometre ile mi ilgili?',
                    'RU':'Это число означает бюджет, год выпуска или пробег?'}[_lang(state.get('language'))]}}
    # Consume only complete simple edits. Mixed requests still go through semantic planning.
    remainder = NUMBER_RE.sub(' ', low)
    remainder = re.sub(r"£|[,.:;!?]", ' ', remainder)
    allowed_words = {'make','the','budget','set','to','add','increase','raise','reduce','decrease','subtract','by','from',
                     'under','below','up','maximum','max','minimum','min','year','years','mileage','limit','km','kilometres','kilometers',
                     'gbp','pound','pounds','sterlin','bütçe','bütçem','butce','butcem','olsun','ekle','artır','artir','azalt',
                     'бюджет','до','на','увеличить','уменьшить','добавить','пробег','км'}
    simple = all(word in allowed_words for word in remainder.split())
    if values and simple and job in {Job.FIND_A_CAR.value, Job.COMPARE_CARS.value}:
        return {'transition':'REFINE','action':'SEARCH_VEHICLES','job':Job.FIND_A_CAR.value,
                'constraints_delta':values, 'awaiting':None}
    return None


def _deterministic_followup_plan(message: str, state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute explicit offered actions and guided filter prompts deterministically."""
    raw = _text(message, 1200)
    low = raw.casefold().strip()
    offered = state.get("offered_action") or {}
    offered_type = str(offered.get("type") or "").upper()
    offered_targets = list(offered.get("target_ids") or [])
    focus_ids = list(((state.get("focus") or {}).get("object_ids") or []))
    lang = _lang(state.get("language") or "EN")

    # If the UI has asked the user for a bound, a bare numeric reply is enough.
    awaiting = clarification(state.get("awaiting")) or {}
    awaiting_field = str(awaiting.get("field") or "") if isinstance(awaiting, Mapping) else ""
    if awaiting_field == "category":
        options = awaiting.get("options") or []
        selected = next((x for x in options if str(x).casefold() == low), None)
        if selected:
            return {"transition":"REFINE", "action":_default_action_for_job(state.get("job")),
                    "job":state.get("job"), "constraints_delta":{"category":selected}, "awaiting":None}
    numeric = _numeric_followup_plan(raw, state)
    if numeric:
        return numeric

    # Generic UI controls intentionally ask for the user's own bound instead of
    # silently applying an arbitrary 2018/100k rule.
    if low in {"set minimum year", "set a minimum year", "set year minimum", "minimum year filter",
               "minimum yılı belirle", "minimum yili belirle", "minimum model yılı", "minimum model yili",
               "задать минимальный год", "минимальный год"}:
        q = {
            "EN":"What is the oldest model year you would still consider?",
            "TR":"Değerlendireceğiniz en eski model yılı kaç olsun?",
            "RU":"Какой самый ранний год выпуска вы готовы рассматривать?",
        }[lang]
        return {"transition":Transition.CONTINUE.value,"action":Action.ASK_CLARIFICATION.value,
                "job":state.get("job") or Job.FIND_A_CAR.value,"awaiting":{"field":"min_year","question":q}}

    if low in {"set mileage limit", "set a mileage limit", "set km limit", "set a km limit", "maximum mileage",
               "km sınırı belirle", "km siniri belirle", "kilometre sınırı belirle", "kilometre siniri belirle",
               "задать лимит пробега", "максимальный пробег"}:
        q = {
            "EN":"What maximum mileage would you be comfortable with?",
            "TR":"En fazla kaç kilometre sizin için uygun olur?",
            "RU":"Какой максимальный пробег для вас приемлем?",
        }[lang]
        return {"transition":Transition.CONTINUE.value,"action":Action.ASK_CLARIFICATION.value,
                "job":state.get("job") or Job.FIND_A_CAR.value,"awaiting":{"field":"max_km","question":q}}

    if low in {"economy or luxury?", "economy or luxury", "choose economy or luxury",
               "ekonomik mi lüks mü?", "ekonomik mi luks mu?", "ekonomik mi lüks mü", "экономичность или премиум?"}:
        q = {
            "EN":"Would you rather prioritise economical ownership or a more premium/luxury feel?",
            "TR":"Daha ekonomik kullanım mı, yoksa daha premium/lüks bir araç mı önceliğiniz?",
            "RU":"Что важнее: экономичность владения или более премиальное ощущение?",
        }[lang]
        return {"transition":Transition.CONTINUE.value,"action":Action.ASK_CLARIFICATION.value,
                "job":state.get("job") or Job.FIND_A_CAR.value,"awaiting":{"field":"preference_direction","question":q}}

    if low in {"broaden year range", "broaden the year range", "older cars too", "consider older years",
               "yıl aralığını genişlet", "yil araligini genislet", "daha eski araçlar da", "расширить диапазон лет"}:
        q = {
            "EN":"What is the oldest model year you would be comfortable considering?",
            "TR":"En eski hangi model yılına kadar inmeyi düşünürsünüz?",
            "RU":"До какого минимального года выпуска вы готовы расширить поиск?",
        }[lang]
        return {"transition":Transition.CONTINUE.value,"action":Action.ASK_CLARIFICATION.value,
                "job":state.get("job") or Job.FIND_A_CAR.value,"awaiting":{"field":"min_year","question":q}}

    # Ordinal references are state-resolution, not an LLM judgment. Resolve
    # common "the second one"-style selections against the structured
    # shortlist/focus so the user can narrow a comparison without a needless
    # clarification turn.
    ordinal_patterns = [
        (1, r"^(?:the\s+)?(?:first|1st)(?:\s+one)?[.!]?$|^(?:ilk(?:i| olan)?)?[.!]?$|^(?:перв(?:ый|ая|ое))(?:\s+вариант)?[.!]?$"),
        (2, r"^(?:the\s+)?(?:second|2nd)(?:\s+one)?[.!]?$|^(?:ikinci(?:si| olan)?)[.!]?$|^(?:втор(?:ой|ая|ое))(?:\s+вариант)?[.!]?$"),
        (3, r"^(?:the\s+)?(?:third|3rd)(?:\s+one)?[.!]?$|^(?:üçüncü(?:sü| olan)?|ucuncu(?:su| olan)?)[.!]?$|^(?:трет(?:ий|ья|ье))(?:\s+вариант)?[.!]?$"),
        (4, r"^(?:the\s+)?(?:fourth|4th)(?:\s+one)?[.!]?$|^(?:dördüncü(?:sü| olan)?|dorduncu(?:su| olan)?)[.!]?$|^(?:четв[её]рт(?:ый|ая|ое))(?:\s+вариант)?[.!]?$"),
    ]
    candidates = list(state.get("shortlist") or []) or focus_ids
    # Python casefold represents Turkish capital İ as ``i`` + combining dot.
    # Remove that combining mark for these narrow ordinal phrase checks.
    ordinal_text = low.replace("\u0307", "")
    for ordinal, pattern in ordinal_patterns:
        if re.fullmatch(pattern, ordinal_text, flags=re.I) and len(candidates) >= ordinal:
            selected_id = candidates[ordinal - 1]
            selected = (state.get("objects") or {}).get(selected_id) or {}
            selected_type = str(selected.get("type") or "").upper()
            delta: Dict[str, Any] = {}
            if selected_type == ObjectType.MODEL.value:
                brand = _text(selected.get("brand"), 100)
                model_name = _text(selected.get("model"), 120)
                if brand:
                    delta["brands"] = [brand]
                if model_name:
                    delta["models"] = [model_name]
                return {
                    "transition": Transition.SWITCH_SUBTASK.value,
                    "action": Action.SEARCH_VEHICLES.value,
                    "job": Job.FIND_A_CAR.value if state.get("audience") == Audience.PERSONAL.value else state.get("job"),
                    "constraints_delta": delta,
                    "target_ids": [selected_id],
                    "shortlist_ids": [selected_id],
                    "awaiting": None,
                }

    acceptance = bool(re.fullmatch(r"\s*(?:yes|yeah|yep|sure|ok|okay|do it|go ahead|please|evet|tamam|olur|göster|goster|да|хорошо|давай)\s*[.!]?\s*", low, flags=re.I))
    # Only pure commands use this shortcut. A request that adds criteria or names
    # new cars must reach semantic planning instead of silently reusing old state.
    command = re.sub(r"^(?:yes|okay|ok|sure)[, ]+", "", low).strip(' .!?“”"')
    show_links = bool(re.fullmatch(r"(?:(?:please |can you |can i )?(?:show|see|send|give|open)(?: me)? (?:the |current |these |those )?)?(?:links?|listings?|ads?)(?: please)?|(?:ilanları|ilanlari|linkleri)(?: göster| goster)?|(?:покажи|показать)(?: мне)? (?:объявления|ссылки)", command, re.I))
    compare = bool(re.fullmatch(r"(?:please |can you )?compare(?: (?:them|these|those|the options|these options|the two|the cars))?|comparison|(?:bunları |bunlari |bu seçenekleri )?(?:karşılaştır|karsilastir)|(?:сравни|сравнить)(?: их| варианты)?", command, re.I))

    if acceptance and offered_type:
        return {
            "transition": Transition.CONTINUE.value,
            "action": offered_type,
            "job": state.get("job"),
            "target_ids": offered_targets,
        }
    if show_links and (offered_type == Action.SHOW_LISTINGS.value or focus_ids):
        return {
            "transition": Transition.CONTINUE.value,
            "action": Action.SHOW_LISTINGS.value,
            "job": state.get("job") or Job.FIND_A_CAR.value,
            "target_ids": offered_targets or focus_ids,
        }
    if compare and len(offered_targets or focus_ids) >= 2:
        return {
            "transition": Transition.SWITCH_SUBTASK.value,
            "action": Action.COMPARE_VEHICLES.value,
            "job": Job.COMPARE_CARS.value if state.get("audience") == Audience.PERSONAL.value else state.get("job"),
            "target_ids": (offered_targets or focus_ids)[:4],
        }
    return None


def _authoritative_message_evidence(message: str, host: Mapping[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"vehicle_targets": [], "brands": []}
    resolver = _host(host, "resolve_market_vehicle_mentions")
    attach_years = _host(host, "_attach_explicit_years_to_vehicle_targets")
    brand_resolver = _host(host, "_extract_explicit_market_brands")
    try:
        targets = resolver(message) if callable(resolver) else []
        if callable(attach_years):
            targets = attach_years(message, targets)
        evidence["vehicle_targets"] = [dict(t) for t in (targets or [])[:10]]
    except Exception:
        pass
    try:
        evidence["brands"] = list(brand_resolver(message) or []) if callable(brand_resolver) else []
    except Exception:
        pass
    # Direct numbers are evidence, but interpretation determines their meaning.
    evidence["numbers"] = re.findall(r"(?:£\s*)?\d[\d.,]*\s*(?:k|K|bin|GBP|gbp|pounds?|sterlin|km)?", message)[:10]
    return evidence


def _semantic_plan(message: str, language: str, state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    deterministic = _deterministic_followup_plan(message, state)
    if deterministic:
        return deterministic

    post = _host(host, "_openai_post")
    model = _host(host, "OPENAI_MODEL", "gpt-5.6-luna")
    if not callable(post):
        raise RuntimeError("V10_OPENAI_POST_UNAVAILABLE")

    authoritative = _authoritative_message_evidence(message, host)
    state_context = {
        "audience": state.get("audience"),
        "job": state.get("job"),
        "goal": state.get("goal"),
        "constraints": state.get("constraints") or {},
        "preferences": state.get("preferences") or {},
        "objects": _state_objects_for_prompt(state),
        "shortlist_ids": state.get("shortlist") or [],
        "focus_ids": ((state.get("focus") or {}).get("object_ids") or []),
        "last_result": state.get("last_result"),
        "offered_action": state.get("offered_action"),
        "awaiting": state.get("awaiting"),
    }

    instructions = f"""
You are the semantic planner for OtoDeğer V10, a decision agent for the North Cyprus vehicle market.
Return ONE JSON object only. Never answer the user and never invent market facts.
The UI language is {language}; language does not change brand/model names.

The product helps users accomplish decisions, not merely search a database.
PERSONAL jobs: FIND_A_CAR, EVALUATE_PURCHASE, EVALUATE_SALE, COMPARE_CARS, UNDERSTAND_MARKET.
BUSINESS jobs: ACQUIRE_STOCK, PRICE_STOCK, MOVE_AGING_STOCK, ANALYZE_BUSINESS, UNDERSTAND_MARKET.
Actions: ASK_CLARIFICATION, SEARCH_VEHICLES, SHOW_LISTINGS, COMPARE_VEHICLES,
EVALUATE_PURCHASE, EVALUATE_SALE, VALUE_VEHICLE, ANALYZE_MARKET,
ANALYZE_STOCK_PRICES, ANALYZE_AGING_STOCK, RECOMMEND_ACQUISITIONS,
ANALYZE_BUSINESS_PERIOD, EXPLAIN_RESULT.
Transitions: CONTINUE, REFINE, SWITCH_SUBTASK, START_NEW_GOAL, CORRECT.
Object types: MODEL, LISTING, OWNED_VEHICLE, STOCK_ITEM, COMPANY, MARKET_SEGMENT.

State rules:
- Preserve compatible constraints/preferences on REFINE and CONTINUE.
- START_NEW_GOAL only when the user's actual objective changed, not merely because the action changed.
- A comparison/listing/evaluation can be a subtask of an existing buying goal.
- Explicit correction replaces/clears only the corrected field unless user says to forget/reset the task.
- "it/them/those/the Mercedes/the second one/the cheaper one" must resolve to provided object IDs when possible.
- If the immediately offered action is accepted, execute that action and its target IDs.
- Never resurrect objects from an older goal after START_NEW_GOAL.
- Ask at most one clarification and only if the missing information materially prevents a useful answer.
- Help as soon as there is enough information; £15k + SUV is enough to SEARCH_VEHICLES.
- If user asks for links/listings after a recommendation/comparison, SHOW_LISTINGS for the current focus/shortlist, not a broad unrelated search.
- Personal user asking whether a specific advertised car is worth its asking price: EVALUATE_PURCHASE.
- Personal owner asking if an offer is fair / what to list for / whether to reduce: EVALUATE_SALE.
- Business user asking what to bring in / whether to acquire a specific car: ACQUIRE_STOCK + RECOMMEND_ACQUISITIONS.
- Business pricing question: PRICE_STOCK + ANALYZE_STOCK_PRICES.
- Business aging/sitting/slow-stock question: MOVE_AGING_STOCK + ANALYZE_AGING_STOCK.
- Business period/change/recap question: ANALYZE_BUSINESS + ANALYZE_BUSINESS_PERIOD.
- In a focused PRICE_STOCK conversation, wording such as "sell it quickly", "get it gone", or "not maximum margin" is a pricing-strategy refinement: stay PRICE_STOCK + ANALYZE_STOCK_PRICES. Do not switch to MOVE_AGING_STOCK unless the user is asking which stock is old/slow/stuck.
- Historical questions about asking-price reductions/cuts (for example "which price cuts were biggest?") are ANALYZE_BUSINESS + ANALYZE_BUSINESS_PERIOD, not PRICE_STOCK. PRICE_STOCK is for what price to set now.
- A business question contrasting the company with the wider market (for example "was it us or the whole market?") is UNDERSTAND_MARKET + ANALYZE_MARKET. Do not ask for sales/revenue/margin metrics first.
- Month names without a year mean the most recent occurrence of that month relative to current_date supplied in the input. Do not ask which year when that interpretation is unambiguous.
- For North Cyprus vehicle discovery, location normally does not block a useful first answer. If a new broad vehicle goal has a body type but no numeric budget, ask only for maximum budget rather than also asking for a preferred location.

Hard constraints keys allowed: budget_min,budget_max,vehicle_type,brands,models,min_year,max_year,max_km,min_km,
transmission,seller_type,location,fuel_type,exclude_locations,exclude_fuels,category,company,period_start,period_end,asking_price,offer_price,
acquisition_price,desired_sale_price,currency.
Preference keys: economy,reliability,performance,luxury,comfort,practicality,family,commute,size,resale,low_mileage,newer,avoid_fuel.
Use avoid_fuel for negative fuel preferences such as "I hate diesels"; preserve it across turns until explicitly changed or cleared.
Use exclude_locations when the user says a place is unacceptable (for example "no Güzelyurt"); do not convert an exclusion into a positive location filter.

When the user explicitly names a model, add an object spec and target it. Use authoritative vehicle targets exactly; do not alter their canonical brand/model.
When the user clearly describes their own vehicle, use OWNED_VEHICLE. When they describe a specific advertised/purchase candidate (for example year + model + asking price and/or mileage), use LISTING even if no URL is supplied; LISTING identity can use brand/model/year/price/km. Use MODEL only for a model family rather than one specific advertised car.
Do not invent object IDs. Existing objects must use their exact IDs. New objects should use aliases such as new_vehicle_1 and target_ids may refer to those aliases.

Required JSON shape:
{{
 "transition":"REFINE",
 "action":"SEARCH_VEHICLES",
 "job":"FIND_A_CAR",
 "goal_summary":"Find an SUV to buy within £15,000",
 "constraints_delta":{{}},
 "clear_constraints":[],
 "preferences_delta":{{}},
 "clear_preferences":[],
 "objects":[{{"type":"MODEL","alias":"new_vehicle_1","payload":{{"brand":"BMW","model":"X1","year":2019,"price":15000,"km":140000}}}}],
 "target_ids":[],
 "shortlist_ids":[],
 "awaiting":null
}}
Do not return offered_action/result; those are produced by the application after execution.
Use exclude_fuels for an explicit prohibition such as "no diesel". A softer preference
uses preferences_delta.avoid_fuel. Clear exclude_fuels when the user explicitly allows those fuels again.
""".strip()

    try:
        current_date = datetime.now(ZoneInfo("Europe/Nicosia")).date().isoformat()
    except Exception:
        current_date = datetime.utcnow().date().isoformat()

    payload = {
        "latest_message": message,
        "current_date": current_date,
        "authoritative_message_evidence": authoritative,
        "current_state": state_context,
    }
    response = post(
        payload={
            "model": model,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 1000,
            "instructions": instructions,
            "input": json.dumps(payload, ensure_ascii=False, default=str),
        },
        timeout=(2.0, 10.0),
    )
    response.raise_for_status()
    parsed = _json_from_text(_response_text(response.json()))
    if not parsed:
        raise RuntimeError("V10_SEMANTIC_PLAN_INVALID")

    # Canonical explicit market evidence is authoritative. If the model omitted a
    # directly named vehicle, add it deterministically rather than repairing prose later.
    existing_specs = list(parsed.get("objects") or []) if isinstance(parsed.get("objects"), list) else []
    existing_pairs = {
        (_text((x.get("payload") or {}).get("brand"),100).casefold(), _text((x.get("payload") or {}).get("model"),120).casefold())
        for x in existing_specs if isinstance(x, Mapping)
    }
    added_aliases = []
    for idx, target in enumerate(authoritative.get("vehicle_targets") or [], 1):
        pair = (_text(target.get("brand"),100).casefold(), _text(target.get("model"),120).casefold())
        if all(pair) and pair not in existing_pairs:
            alias = f"explicit_vehicle_{idx}"
            existing_specs.append({"type": ObjectType.MODEL.value, "alias": alias, "payload": dict(target)})
            added_aliases.append(alias)
            existing_pairs.add(pair)
    if existing_specs:
        parsed["objects"] = existing_specs
    if len(authoritative.get("vehicle_targets") or []) >= 2 and str(parsed.get("action") or "").upper() in {Action.COMPARE_VEHICLES.value, ""}:
        existing_targets = list(parsed.get("target_ids") or [])
        parsed["target_ids"] = (existing_targets + added_aliases)[:10]
        parsed["action"] = Action.COMPARE_VEHICLES.value
        if state.get("audience") == Audience.PERSONAL.value:
            parsed["job"] = Job.COMPARE_CARS.value

    # Explicit brands typed this turn cannot be silently dropped.
    explicit_brands = authoritative.get("brands") or []
    if explicit_brands:
        delta = dict(parsed.get("constraints_delta") or {})
        delta["brands"] = explicit_brands
        parsed["constraints_delta"] = delta
    parsed = _authoritative_numeric_constraints(message, parsed, state, host)
    parsed = _normalize_semantic_plan(parsed, message, state, language)
    return parsed


def _most_recent_named_month_period(message: str) -> Optional[Tuple[str, str]]:
    """Resolve a named month without a year to its most recent occurrence in Cyprus."""
    low = str(message or "").casefold()
    month_aliases = {
        1: ("january", "jan", "ocak", "январь", "января"),
        2: ("february", "feb", "şubat", "subat", "февраль", "февраля"),
        3: ("march", "mar", "mart", "март", "марта"),
        4: ("april", "apr", "nisan", "апрель", "апреля"),
        5: ("may", "mayıs", "mayis", "май", "мая"),
        6: ("june", "jun", "haziran", "июнь", "июня"),
        7: ("july", "jul", "temmuz", "июль", "июля"),
        8: ("august", "aug", "ağustos", "agustos", "август", "августа"),
        9: ("september", "sep", "sept", "eylül", "eylul", "сентябрь", "сентября"),
        10: ("october", "oct", "ekim", "октябрь", "октября"),
        11: ("november", "nov", "kasım", "kasim", "ноябрь", "ноября"),
        12: ("december", "dec", "aralık", "aralik", "декабрь", "декабря"),
    }
    # If the user supplied a year, leave the exact interpretation to the semantic plan.
    if re.search(r"\b(?:19|20)\d{2}\b", low):
        return None
    month = None
    for number, aliases in month_aliases.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", low, re.I) for alias in aliases):
            month = number
            break
    if month is None:
        return None
    try:
        today = datetime.now(ZoneInfo("Europe/Nicosia")).date()
    except Exception:
        today = datetime.utcnow().date()
    year = today.year if month <= today.month else today.year - 1
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def _normalize_semantic_plan(parsed: Mapping[str, Any], message: str, state: Mapping[str, Any], language: str) -> Dict[str, Any]:
    """Deterministic repair for state-safe, high-value conversational semantics.

    This does not infer market facts. It canonicalizes object identity and resolves
    a few product-contract decisions that should not drift between model calls.
    """
    out = copy.deepcopy(dict(parsed))
    low = str(message or "").casefold()
    audience = str(state.get("audience") or Audience.PERSONAL.value).upper()
    current_job = str(state.get("job") or "").upper()

    # COMPANY object identity must use canonical `name`/`company_id`. Models may
    # naturally emit {"company": "Demo Motors"}; repair that before state hashing.
    specs = []
    for raw_spec in list(out.get("objects") or []):
        if not isinstance(raw_spec, Mapping):
            specs.append(raw_spec)
            continue
        spec = copy.deepcopy(dict(raw_spec))
        if str(spec.get("type") or "").upper() == ObjectType.COMPANY.value:
            payload = dict(spec.get("payload") or {})
            if not payload.get("name") and not payload.get("company_id"):
                candidate = payload.get("company") or payload.get("company_name") or payload.get("business_name")
                if not candidate:
                    candidate = (state.get("constraints") or {}).get("company")
                if candidate:
                    payload["name"] = candidate
            spec["payload"] = payload
        specs.append(spec)
    if specs or "objects" in out:
        out["objects"] = specs

    if audience == Audience.BUSINESS.value:
        # Pricing-strategy refinement on one focused stock item stays in PRICE_STOCK.
        quick_strategy = bool(re.search(
            r"\b(?:gone quickly|sell (?:it )?quickly|sell (?:it )?fast|quick sale|faster sale|not maximum margin|not max(?:imum)? margin|"
            r"hızlı sat|hizli sat|çabuk sat|cabuk sat|marjı? maks|marji? maks|"
            r"продат.{0,10}быстр|быстр.{0,10}продаж|не максимальн.{0,10}марж)",
            low, re.I))
        if current_job == Job.PRICE_STOCK.value and quick_strategy:
            out["job"] = Job.PRICE_STOCK.value
            out["action"] = Action.ANALYZE_STOCK_PRICES.value
            out["transition"] = Transition.REFINE.value
            out["awaiting"] = None

        # Historical asking-price-cut analysis belongs to business performance,
        # not the forward-looking stock pricing job.
        price_cut_history = bool(re.search(
            r"(?:which|what|biggest|largest).{0,30}(?:price cuts?|price reductions?)|"
            r"(?:price cuts?|price reductions?).{0,30}(?:biggest|largest)|"
            r"(?:en büyük|en buyuk).{0,25}(?:fiyat indir|indirim)|"
            r"(?:сам(?:ые|ый)).{0,25}(?:снижени.{0,10}цен|уценк)",
            low, re.I))
        if price_cut_history and current_job == Job.ANALYZE_BUSINESS.value:
            out["job"] = Job.ANALYZE_BUSINESS.value
            out["action"] = Action.ANALYZE_BUSINESS_PERIOD.value
            out["transition"] = Transition.CONTINUE.value
            out["awaiting"] = None

        # Company-vs-market attribution is explicitly an UNDERSTAND_MARKET task.
        market_contrast = bool(re.search(
            r"\b(?:us|our|we).{0,35}(?:whole market|market weak|market down)|"
            r"(?:whole market|market weak|market down).{0,35}(?:us|our|we)|"
            r"(?:bizde|bizim).{0,35}(?:piyasa|pazar)|(?:piyasa|pazar).{0,35}(?:bizde|bizim)|"
            r"(?:у нас).{0,35}(?:рынок)|(?:рынок).{0,35}(?:у нас)",
            low, re.I))
        if market_contrast:
            out["job"] = Job.UNDERSTAND_MARKET.value
            out["action"] = Action.ANALYZE_MARKET.value
            out["transition"] = Transition.SWITCH_SUBTASK.value if current_job and current_job != Job.UNDERSTAND_MARKET.value else Transition.CONTINUE.value
            out["awaiting"] = None

        # A named month without a year uses the most recent occurrence. This
        # prevents needless "which year?" turns for normal business reporting.
        month_period = _most_recent_named_month_period(message)
        if month_period and (str(out.get("job") or "").upper() == Job.ANALYZE_BUSINESS.value or current_job == Job.ANALYZE_BUSINESS.value):
            delta = dict(out.get("constraints_delta") or {})
            delta["period_start"], delta["period_end"] = month_period
            out["constraints_delta"] = delta
            out["job"] = Job.ANALYZE_BUSINESS.value
            out["action"] = Action.ANALYZE_BUSINESS_PERIOD.value
            out["awaiting"] = None
            if str(out.get("transition") or "").upper() == Transition.START_NEW_GOAL.value and current_job:
                out["transition"] = Transition.CONTINUE.value

    return out


def _authoritative_numeric_constraints(message: str, parsed: Mapping[str, Any], state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the shared, unit-aware numeric interpretation to a semantic plan."""
    out = copy.deepcopy(dict(parsed))
    raw_delta = out.get("constraints_delta")
    if raw_delta is not None and not isinstance(raw_delta, Mapping):
        raise InvalidTurnPlan("constraints_delta must be an object")
    delta = dict(raw_delta or {})
    delta.update(_numeric_updates(message, state))
    low = message.casefold().replace("\u0307", "")
    exclusions = []
    for family, words in {'diesel':r'diesel|dizel|дизель\w*', 'petrol':r'petrol|gasoline|benzin|бензин\w*',
                          'electric':r'electric|elektrikli|электрическ\w*', 'hybrid':r'hybrid|hibrit|гибрид\w*'}.items():
        if re.search(rf"\b(?:no|not|without|exclude|без|не хочу)\s+(?:any\s+)?(?:{words})\b|\b(?:{words})\s+(?:istemiyorum|olmasın|olmasin)\b", low):
            exclusions.append(family)
    if exclusions:
        delta['exclude_fuels'] = list(dict.fromkeys(list((state.get('constraints') or {}).get('exclude_fuels') or []) + exclusions))

    # Deterministic interpretation of the exact contextual suggestion language.
    # These are hard filters once clicked/typed; do not leave them to model drift.
    m=re.search(r"\b(20\d{2}|19\d{2})\s*(?:or newer|and newer|ve sonrası|ve sonrasi|или новее)\b", low, re.I)
    if m:
        delta["min_year"] = int(m.group(1))
    negated = bool(re.search(r"\b(?:no|not|without|don't|dont|istemiyorum|olmasın|olmasin|не|без)\b", low))
    if not negated and re.search(r"\b(?:automatic only|only automatic|sadece otomatik|otomatik sadece|только автомат)\b", low, re.I):
        delta["transmission"] = "Automatic"
    if not negated and re.search(r"\b(?:gallery sellers? only|dealers? only|galleries? only|sadece galeriler?|только дилер)\b", low, re.I):
        delta["seller_type"] = "gallery"
    elif not negated and re.search(r"\b(?:private sellers? only|individual sellers? only|sadece bireysel|частн(?:ый|ые) продав)\b", low, re.I):
        delta["seller_type"] = "private"

    # Protect the most important physical-class terms as authoritative constraints.
    # This works alongside the semantic parser rather than replacing it.
    if not negated and re.search(r"\bSUVs?\b", str(message), re.I):
        delta["vehicle_type"] = "SUV"
    elif not negated and re.search(r"\b(?:crossover|crossovers)\b", low, re.I):
        delta["vehicle_type"] = "crossover"
    elif not negated and re.search(r"\b(?:pickup|pick-up|pickups|pick-ups)\b", low, re.I):
        delta["vehicle_type"] = "pickup"

    out["constraints_delta"]=delta
    return out


# ---------------------------------------------------------------------------
# Canonical state -> legacy market filters / profile preferences
# ---------------------------------------------------------------------------

def _legacy_filters(state: Mapping[str, Any]) -> Dict[str, Any]:
    c = dict(state.get("constraints") or {})
    out: Dict[str, Any] = {}
    mapping = {
        "budget_max": "budget", "budget_min": "min_budget", "brands": "brands",
        "models": "models", "min_year": "min_year", "max_year": "max_year",
        "min_km": "min_km", "max_km": "max_km", "location": "locations",
        "exclude_locations": "exclude_locations", "transmission": "transmissions", "fuel_type": "fuels", "category": "categories", "company": "companies",
    }
    for src, dest in mapping.items():
        val = c.get(src)
        if val in (None, "", [], {}):
            continue
        if dest in {"locations", "transmissions", "fuels", "categories", "companies"} and not isinstance(val, list):
            val = [val]
        out[dest] = val

    # Canonical V10 seller_type maps onto the legacy market's Company field.
    # "Bireysel" is the private-seller marker in the source dataset.
    seller_type = str(c.get("seller_type") or "").strip().casefold()
    if seller_type in {"private", "individual", "bireysel"}:
        out["companies"] = ["Bireysel"]
        out.pop("exclude_companies", None)
    elif seller_type in {"gallery", "dealer", "dealership", "galeri"}:
        out["exclude_companies"] = ["Bireysel"]
        # Do not let a stale company filter override the requested seller class.
        if c.get("company") in (None, "", [], {}):
            out.pop("companies", None)
    return out


def _legacy_preferences(state: Mapping[str, Any]) -> List[str]:
    c, p = state.get("constraints") or {}, state.get("preferences") or {}
    out: List[str] = []
    if c.get("vehicle_type"):
        out.append(f"vehicle_type:{c['vehicle_type']}")
    for key in ("economy","reliability","performance","luxury","comfort","practicality"):
        if p.get(key) not in (None, False, 0, ""):
            out.append(f"priority:{key}")
    for key in ("family","commute"):
        if p.get(key) not in (None, False, 0, ""):
            out.append(f"use_case:{key}")
    if p.get("low_mileage") not in (None, False, 0, ""):
        out.append("listing_sort:lowest_km")
    if p.get("newer") not in (None, False, 0, ""):
        out.append("listing_sort:newest")
    return out


def _search_all(state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    fn = _host(host, "market_search")
    if not callable(fn):
        raise AssistantDataUnavailable("Market search unavailable")
    f = _legacy_filters(state)
    search_kwargs = {
        "budget": f.get("budget"), "min_budget": f.get("min_budget"), "brands": f.get("brands"),
        "models": f.get("models"), "categories": f.get("categories"), "locations": f.get("locations"), "exclude_locations": f.get("exclude_locations"),
        "companies": f.get("companies"), "exclude_companies": f.get("exclude_companies"), "transmissions": f.get("transmissions"),
        "min_year": f.get("min_year"), "max_year": f.get("max_year"), "min_km": f.get("min_km"), "max_km": f.get("max_km"),
        "limit": 5000, "max_limit": 5000, "analysis_mode": True,
    }
    # Compatibility: older/mock host search functions do not know the new Fuel
    # keyword. Only send it when the user actually has a hard fuel constraint.
    if f.get("fuels") not in (None, "", [], {}):
        search_kwargs["fuels"] = f.get("fuels")
    result = _checked_search(fn(**search_kwargs))
    excluded = (state.get('constraints') or {}).get('exclude_fuels') or []
    if excluded:
        def fuel_family_text(row):
            fuel = str(row.get('fuel') or '').casefold()
            for source, target in [('dizel','diesel'),('дизель','diesel'),('benzin','petrol'),('бензин','petrol'),
                                   ('gasoline','petrol'),('hibrit','hybrid'),('гибрид','hybrid'),('elektrik','electric'),('электр','electric')]:
                fuel = fuel.replace(source, target)
            return fuel
        # An unknown fuel cannot be verified as meeting an explicit exclusion.
        rows = [r for r in result['results'] if fuel_family_text(r) and not any(str(x).casefold() in fuel_family_text(r) for x in excluded)]
        result = dict(result, results=rows, count=len(rows), returned=len(rows))

    # Negative fuel preference is soft: avoid it when alternatives exist, but do
    # not turn a preference into a silent hard zero-result constraint.
    avoid_fuel = str((state.get("preferences") or {}).get("avoid_fuel") or "").strip().casefold()
    fuel_aliases = {
        "benzin": "petrol", "benzinli": "petrol", "gasoline": "petrol", "бензин": "petrol",
        "dizel": "diesel", "дизель": "diesel",
        "hibrit": "hybrid", "гибрид": "hybrid",
        "mild hibrit": "mild hybrid", "mild-hibrit": "mild hybrid", "мягкий гибрид": "mild hybrid",
        "plug in hibrit": "plug-in hybrid", "plug-in hibrit": "plug-in hybrid", "подключаемый гибрид": "plug-in hybrid",
        "elektrik": "electric", "elektrikli": "electric", "электро": "electric", "электрический": "electric", "ev": "electric",
    }
    avoid_fuel = fuel_aliases.get(avoid_fuel, avoid_fuel)
    if avoid_fuel and isinstance(result, Mapping):
        source_rows = [dict(x) for x in (result.get("results") or []) if isinstance(x, Mapping)]
        if source_rows:
            def fuel_text(row):
                text = str(row.get("fuel") or row.get("fuel_type") or row.get("Fuel") or "").strip().casefold()
                replacements = [
                    ("подключаемый гибрид", "plug-in hybrid"),
                    ("plug-in hibrit", "plug-in hybrid"),
                    ("plug in hibrit", "plug-in hybrid"),
                    ("мягкий гибрид", "mild hybrid"),
                    ("mild-hibrit", "mild hybrid"),
                    ("mild hibrit", "mild hybrid"),
                    ("электрический", "electric"),
                    ("электричество", "electric"),
                    ("elektrikli", "electric"),
                    ("elektrik", "electric"),
                    ("электро", "electric"),
                    ("бензин", "petrol"),
                    ("benzinli", "petrol"),
                    ("benzin", "petrol"),
                    ("gasoline", "petrol"),
                    ("дизель", "diesel"),
                    ("dizel", "diesel"),
                    ("гибрид", "hybrid"),
                    ("hibrit", "hybrid"),
                ]
                for src, dest in replacements:
                    text = text.replace(src, dest)
                return text
            preferred = [row for row in source_rows if avoid_fuel not in fuel_text(row)]
            if preferred:
                result = dict(result)
                result["results"] = preferred
                result["count"] = len(preferred)
                result["returned"] = len(preferred)

    # Vehicle type is a physical-class constraint, not a text preference. The
    # legacy market_search does not own that taxonomy, so enforce it with the
    # validated model-profile layer before V10 ranks or registers any result.
    # This prevents a request for an SUV from ever drifting into Fit/Swift/etc.
    strict = _host(host, "_apply_strict_vehicle_type_to_search_result")
    if callable(strict) and (state.get("constraints") or {}).get("vehicle_type"):
        try:
            result = strict(result, _legacy_preferences(state))
        except Exception as exc:
            raise AssistantDataUnavailable("Vehicle classification failed") from exc
        result = _checked_search(result)
    return result


def _normal_vehicle_default_filter(results: Sequence[Mapping[str, Any]], state: Mapping[str, Any], host: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Exclude motorcycles/ATVs by default unless explicitly requested.

    We use the stable model profile catalogue instead of model-name heuristics.
    """
    vehicle_type = str((state.get("constraints") or {}).get("vehicle_type") or "").casefold()
    if vehicle_type in {"motorcycle", "scooter", "atv", "quad"}:
        return [dict(x) for x in results]
    lookup = _host(host, "MODEL_PROFILE_LOOKUP", {}) or {}
    out = []
    for row in results:
        key = (_text(row.get("brand"),100).casefold(), _text(row.get("model"),120).casefold())
        prof = lookup.get(key) or lookup.get((_text(row.get("brand"),100), _text(row.get("model"),120)))
        vt = _text((prof or {}).get("VehicleType") or (prof or {}).get("vehicle_type"), 40).casefold()
        if vt in {"motorcycle", "scooter", "atv", "quad"}:
            continue
        out.append(dict(row))
    return out


def _model_summary(rows: Sequence[Mapping[str, Any]], max_groups: int = 80) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str,str], List[Mapping[str, Any]]] = {}
    for r in rows:
        brand, model = _text(r.get("brand"),100), _text(r.get("model"),120)
        if not brand or not model:
            continue
        buckets.setdefault((brand, model), []).append(r)
    out = []
    for (brand, model), items in buckets.items():
        prices = [float(x["price"]) for x in items if _finite(x.get("price")) is not None]
        years = [int(x["year"]) for x in items if _int(x.get("year")) is not None]
        kms = [int(x["km"]) for x in items if _int(x.get("km")) is not None]
        if not prices:
            continue
        newest = max(years) if years else None
        newest_items = [x for x in items if _int(x.get("year")) == newest] if newest else list(items)
        newest_prices = [float(x["price"]) for x in newest_items if _finite(x.get("price")) is not None]
        out.append({
            "brand": brand, "model": model, "count": len(items),
            "starting_price": min(prices), "median_price": statistics.median(prices),
            "newest_year": newest, "newest_year_starting_price": min(newest_prices) if newest_prices else min(prices),
            "median_km": statistics.median(kms) if kms else None,
        })
    out.sort(key=lambda x: (-int(x.get("newest_year") or 0), float(x.get("newest_year_starting_price") or 1e18), -int(x.get("count") or 0)))
    return out[:max_groups]


def _historical_model_activity(host: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate buyer-intelligence history to model-family activity signals.

    This is an internal ranking signal only. Historical listing observations and
    observed market exits are not confirmed transactions/sales.
    """
    df = _host(host, "buyer_model_df")
    if df is None or getattr(df, "empty", True):
        return {}
    required = {"Brand", "Model"}
    if not required.issubset(set(getattr(df, "columns", []))):
        return {}
    try:
        work = df.copy()
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for (brand, model), grp in work.groupby(["Brand", "Model"], dropna=False):
            b, m = _text(brand, 100), _text(model, 120)
            if not b or not m:
                continue
            hist = 0.0
            if "Buyer_HistoricalDistinctListings" in grp.columns:
                vals = grp["Buyer_HistoricalDistinctListings"].dropna()
                hist = float(vals.sum()) if len(vals) else 0.0
            eligible = 0.0
            exit_proxy = 0.0
            if {"Buyer_Exit60EligibleListings", "Buyer_ObservedExitWithin60DaysRate"}.issubset(grp.columns):
                for _, r in grp.iterrows():
                    e = _finite(r.get("Buyer_Exit60EligibleListings")) or 0.0
                    rate = _finite(r.get("Buyer_ObservedExitWithin60DaysRate"))
                    if rate is None:
                        continue
                    if rate > 1.5:
                        rate = rate / 100.0
                    eligible += max(0.0, e)
                    exit_proxy += max(0.0, e) * max(0.0, min(1.0, rate))
            weighted_exit_rate = (exit_proxy / eligible) if eligible > 0 else 0.0
            # Volume says how often this family appears in the observed market;
            # the exit component rewards evidence of listings leaving the market.
            activity_score = hist * (0.55 + weighted_exit_rate)
            out[(b.casefold(), m.casefold())] = {
                "historical_distinct_listings": int(round(hist)),
                "exit60_eligible": int(round(eligible)),
                "observed_exit60_proxy": round(exit_proxy, 1),
                "observed_exit60_rate": round(weighted_exit_rate, 4),
                "activity_score": round(activity_score, 3),
            }
        return out
    except Exception:
        return {}


def _search_vehicle_models(state: Mapping[str, Any], message: str, host: Mapping[str, Any]) -> Dict[str, Any]:
    search = _search_all(state, host)
    rows = _normal_vehicle_default_filter(search.get("results") or [], state, host)
    if not rows:
        return {"kind":"vehicle_search", "count":0, "models":[], "listings":[], "alternative_models":[]}

    # IMPORTANT: every displayed metric must be calculated from the exact current
    # filtered listing set. Do not borrow counts/medians from broader intelligence
    # tables because those may include older years, different mileage bands or
    # vehicles outside the active budget.
    summaries = _model_summary(rows, max_groups=120)

    constraints = state.get("constraints") or {}
    preferences = state.get("preferences") or {}
    has_named_vehicle = bool(constraints.get("brands") or constraints.get("models"))
    has_decision_priority = any(
        preferences.get(k) not in (None, False, 0, "", [], {})
        for k in ("economy","reliability","performance","luxury","comfort","practicality","family","commute","size","resale","low_mileage","newer")
    )

    ranked: List[Dict[str, Any]] = []
    if not has_named_vehicle and not has_decision_priority:
        # Broad discovery: rank the already-filtered candidates by historical
        # observed market activity, then by current choice/newness. This helps
        # surface the parts of the market where listings are most active without
        # treating removals as confirmed sales.
        activity = _historical_model_activity(host)
        for item in summaries:
            x = dict(item)
            hist = activity.get((str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold()), {})
            x.update(hist)
            ranked.append(x)
        ranked.sort(key=lambda x: (
            -float(x.get("activity_score") or 0.0),
            -int(x.get("historical_distinct_listings") or 0),
            -int(x.get("count") or 0),
            -int(x.get("newest_year") or 0),
            float(x.get("newest_year_starting_price") or 1e18),
        ))
    else:
        ranked = [dict(x) for x in summaries]
        # Named-brand refinement should primarily answer what the budget buys.
        ranked.sort(key=lambda x: (
            -int(x.get("newest_year") or 0),
            float(x.get("newest_year_starting_price") or 1e18),
            -int(x.get("count") or 0),
        ))

    # Do not impose an arbitrary five-model catalogue. Keep the useful active
    # set, but suppress the long tail of niche/low-evidence models. The activity
    # threshold is relative to the strongest matching model so it adapts to the
    # query rather than hard-coding a brand list.
    if not has_named_vehicle and not has_decision_priority and ranked:
        top_activity = float(ranked[0].get("activity_score") or 0.0)
        useful = [x for x in ranked if int(x.get("count") or 0) >= 2 and (top_activity <= 0 or float(x.get("activity_score") or 0.0) >= top_activity * 0.12)]
        shown = (useful or ranked)[:8]
    else:
        shown = ranked[:8]

    shown_pairs = {(str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold()) for x in shown}
    alternative_models: List[str] = []
    for x in ranked:
        pair=(str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold())
        if pair in shown_pairs:
            continue
        label=" ".join(v for v in [_text(x.get("brand"),80), _text(x.get("model"),100)] if v)
        if label and label not in alternative_models:
            alternative_models.append(label)
        if len(alternative_models) >= 4:
            break

    return {
        "kind":"vehicle_search",
        "count":len(rows),
        "models":shown,
        "listings":[],
        "alternative_models":alternative_models,
        "discovery_style":"OPTIONS" if not has_named_vehicle else "BRAND_OPTIONS",
        "ranking_basis":"observed_market_activity" if not has_named_vehicle and not has_decision_priority else "current_fit",
    }

def _target_listing_search(state: Mapping[str, Any], target_ids: Sequence[str], host: Mapping[str, Any]) -> Dict[str, Any]:
    # Apply every active constraint/preference through the same search as discovery.
    # Model selection must not bypass fuel exclusions, seller class or vehicle type.
    targets = _model_targets_from_objects(state, target_ids)
    rows = _search_all(state, host).get("results") or []
    if targets:
        def matches(row, target):
            return (str(row.get("brand") or "").casefold() == str(target.get("brand") or "").casefold()
                    and str(row.get("model") or "").casefold() == str(target.get("model") or "").casefold()
                    and (not target.get("category") or str(row.get("category") or "").casefold() == str(target["category"]).casefold()))
        rows = [row for row in rows if any(matches(row, target) for target in targets)]
    def rank(r):
        year = _int(r.get("year")) or 0
        km = _int(r.get("km"))
        price = _finite(r.get("price"))
        return (-year, km if km is not None else 10**9, price if price is not None else 1e18)
    rows = sorted([dict(x) for x in rows], key=rank)
    return {"kind":"listings", "count":len(rows), "listings":rows[:12], "models":[]}


def _comparable_rows(state: Mapping[str, Any], target: Mapping[str, Any], host: Mapping[str, Any], *, year_window: int = 2) -> List[Dict[str, Any]]:
    brand, model = _text(target.get("brand"),100), _text(target.get("model"),120)
    if not brand or not model:
        return []
    fn = _host(host, "market_search")
    if not callable(fn):
        return []
    try:
        f = _legacy_filters(state)
        result = fn(
            budget=f.get("budget"), min_budget=f.get("min_budget"),
            brands=[brand], models=[model], categories=[target.get("category")] if target.get("category") else f.get("categories"),
            locations=f.get("locations"), transmissions=f.get("transmissions"),
            min_year=f.get("min_year"), max_year=f.get("max_year"),
            min_km=f.get("min_km"), max_km=f.get("max_km"),
            limit=5000, max_limit=5000, analysis_mode=True,
        )
        rows = [dict(x) for x in (result.get("results") or []) if _text(x.get("brand"),100).casefold()==brand.casefold() and _text(x.get("model"),120).casefold()==model.casefold()]
    except Exception:
        return []
    year = _int(target.get("year"))
    if year is not None:
        tight = [x for x in rows if _int(x.get("year")) is not None and abs((_int(x.get("year")) or 0)-year) <= year_window]
        rows = tight
    category = target.get("category") or (state.get("constraints") or {}).get("category")
    if category:
        rows = [x for x in rows if str(x.get("category") or "").strip().casefold() == str(category).strip().casefold()]
    return rows


def _evaluate_price_subject(state: Mapping[str, Any], target_ids: Sequence[str], host: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    objects = state.get("objects") or {}
    candidates = target_ids or ((state.get("focus") or {}).get("object_ids") or [])
    if not candidates:
        return {"kind":mode, "status":"needs_subject", "evidence":{}}
    obj = objects.get(candidates[0]) or {}
    subject = {k: v for k, v in obj.items() if k not in {"id", "type"}}
    c = state.get("constraints") or {}
    asking = _finite(subject.get("price") or subject.get("asking_price") or c.get("asking_price"))
    offer = _finite(c.get("offer_price") or subject.get("offer_price"))
    price_to_evaluate = asking if mode == "purchase_evaluation" else (offer if offer is not None else asking)
    rows = _comparable_rows(state, subject, host)
    categories = sorted({str(x.get("category") or "").strip() for x in rows if str(x.get("category") or "").strip()})
    if not (subject.get("category") or c.get("category")) and len(categories) > 1:
        question = {"EN":"Which version is your car? These variants have different values: ",
                    "TR":"Aracınız hangi versiyon? Bu versiyonların değerleri farklı: ",
                    "RU":"Какая у вас версия? Стоимость этих версий различается: "}[_lang(state.get("language"))]
        return {"kind":"clarification", "awaiting":{"field":"category", "question":question + "; ".join(categories), "options":categories}}
    prices = [float(x["price"]) for x in rows if _finite(x.get("price")) is not None]
    if not prices:
        return {"kind":mode, "status":"insufficient", "subject":subject, "price":price_to_evaluate, "comparables":[]}
    median = statistics.median(prices)
    p25 = statistics.quantiles(prices, n=4, method="inclusive")[0] if len(prices)>=4 else min(prices)
    p75 = statistics.quantiles(prices, n=4, method="inclusive")[2] if len(prices)>=4 else max(prices)
    pos = None
    vs_med = None
    if price_to_evaluate is not None and median > 0:
        vs_med = (price_to_evaluate - median)/median
        if price_to_evaluate <= p25: pos = "LOW"
        elif price_to_evaluate >= p75: pos = "HIGH"
        else: pos = "MID"
    closest = sorted(rows, key=lambda x: abs((_finite(x.get("price")) or 1e18) - (price_to_evaluate if price_to_evaluate is not None else median)))[:5]
    return {
        "kind":mode, "status":"ok", "subject":subject, "price":price_to_evaluate,
        "comparables_count":len(prices), "median_asking_price":median, "p25_asking_price":p25,
        "p75_asking_price":p75, "position":pos, "vs_median_pct":round(vs_med*100,1) if vs_med is not None else None,
        "comparables":closest,
        "basis":"current asking prices, not confirmed transaction prices",
    }


def _compare_models(state: Mapping[str, Any], target_ids: Sequence[str], host: Mapping[str, Any]) -> Dict[str, Any]:
    objects = state.get("objects") or {}
    ids = list(target_ids or ((state.get("focus") or {}).get("object_ids") or []))[:4]
    comps = []
    for oid in ids:
        obj = objects.get(oid) or {}
        payload = {k: v for k, v in obj.items() if k not in {"id", "type"}}
        rows = _comparable_rows(state, payload, host)
        if not rows:
            continue
        prices=[float(x["price"]) for x in rows if _finite(x.get("price")) is not None]
        years=[_int(x.get("year")) for x in rows if _int(x.get("year")) is not None]
        if not years:
            continue
        newest_year=max(years)
        newest_rows=[x for x in rows if _int(x.get("year")) == newest_year]
        newest_prices=[float(x["price"]) for x in newest_rows if _finite(x.get("price")) is not None]
        best_newest = sorted(newest_rows, key=lambda x: _finite(x.get("price")) or 1e18)[0] if newest_rows else None
        comps.append({
            "object_id": oid,
            "brand":payload.get("brand"),
            "model":payload.get("model"),
            "count":len(rows),
            "newest_year":newest_year,
            "newest_year_count":len(newest_rows),
            "newest_year_starting_price":min(newest_prices) if newest_prices else None,
            "newest_year_median_price":statistics.median(newest_prices) if newest_prices else None,
            "newest_year_highest_price":max(newest_prices) if newest_prices else None,
            "representative_listing":best_newest,
        })
    return {"kind":"comparison", "vehicles":comps}

def _business_company_rows(state: Mapping[str, Any], host: Mapping[str, Any]):
    company = _text((state.get("constraints") or {}).get("company"), 160)
    stock = _host(host, "business_stock_df")
    if not company or stock is None:
        return company, None
    try:
        rows = stock[stock["Company"].fillna("").astype(str).str.casefold()==company.casefold()].copy()
        return company, rows
    except Exception:
        return company, None



def _filter_business_rows_to_targets(state: Mapping[str, Any], rows, target_ids: Sequence[str]):
    if rows is None or not target_ids:
        return rows
    objects=state.get("objects") or {}
    masks=[]
    try:
        for oid in target_ids:
            obj=objects.get(oid) or {}
            if obj.get("type") not in {ObjectType.STOCK_ITEM.value,ObjectType.LISTING.value,ObjectType.MODEL.value}:
                continue
            link=_text(obj.get("link"),500)
            if link and "Link" in rows.columns:
                masks.append(rows["Link"].fillna("").astype(str)==link)
                continue
            brand=_text(obj.get("brand"),100); model=_text(obj.get("model"),120); year=_int(obj.get("year"))
            mask=(rows["Brand"].fillna("").astype(str).str.casefold()==brand.casefold()) & (rows["Model"].fillna("").astype(str).str.casefold()==model.casefold())
            if year is not None and "Year" in rows.columns:
                mask=mask & (rows["Year"]==year)
            masks.append(mask)
        if masks:
            combined=masks[0]
            for m in masks[1:]: combined=combined|m
            matched=rows[combined].copy()
            if not matched.empty: return matched
    except Exception:
        pass
    return rows

def _business_price_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    company, rows = _business_company_rows(state, host)
    if rows is None or getattr(rows,"empty",True):
        return {"kind":"business_pricing", "company":company, "vehicles":[], "status":"company_or_stock_unavailable"}
    rows=_filter_business_rows_to_targets(state,rows,target_ids)
    def score(row):
        band=str(row.get("PricePositionBand") or "").upper()
        attention=str(row.get("AttentionLevel") or "").upper()
        pct=_finite(row.get("PriceVsMedianPct")) or 0
        return (2 if "HIGH" in band else 0)+(2 if "HIGH" in attention else 1 if "ATTENTION" in attention else 0)+max(0,pct)
    records=_df_records(rows,500)
    records.sort(key=score, reverse=True)
    return {"kind":"business_pricing","company":company,"vehicles":records[:12],"status":"ok"}


def _business_aging_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    company, rows = _business_company_rows(state, host)
    if rows is None or getattr(rows,"empty",True):
        return {"kind":"business_aging", "company":company, "vehicles":[], "status":"company_or_stock_unavailable"}
    rows=_filter_business_rows_to_targets(state,rows,target_ids)
    records=_df_records(rows,500)
    records.sort(key=lambda x:(_finite(x.get("StockAgeDays")) or -1, _finite(x.get("PriceVsMedianPct")) or -999), reverse=True)
    return {"kind":"business_aging","company":company,"vehicles":records[:12],"status":"ok"}


def _business_acquire_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    df=_host(host,"business_market_df")
    if df is None or getattr(df,"empty",True):
        return {"kind":"business_acquisition","options":[],"status":"unavailable"}
    work=df.copy()
    c=state.get("constraints") or {}
    objects=state.get("objects") or {}
    target_obj=(objects.get(target_ids[0]) or {}) if target_ids else {}
    specific_brand=_text(target_obj.get("brand"),100)
    specific_model=_text(target_obj.get("model"),120)
    specific_year=_int(target_obj.get("year"))
    try:
        if specific_brand and "Brand" in work:
            work=work[work["Brand"].fillna("").astype(str).str.casefold()==specific_brand.casefold()]
        if specific_model and "Model" in work:
            work=work[work["Model"].fillna("").astype(str).str.casefold()==specific_model.casefold()]
        if specific_year is not None and "Year" in work:
            exact=work[work["Year"]==specific_year]
            if not exact.empty: work=exact
        if c.get("budget_max") is not None and "CurrentStartingPrice" in work:
            work=work[work["CurrentStartingPrice"].notna() & (work["CurrentStartingPrice"]<=float(c["budget_max"]))]
        if c.get("vehicle_type") and "VehicleType" in work:
            vt=str(c["vehicle_type"]).casefold(); work=work[work["VehicleType"].fillna("").astype(str).str.casefold().str.contains(vt,regex=False)]
        if c.get("brands") and "Brand" in work:
            wanted={str(x).casefold() for x in c["brands"]}; work=work[work["Brand"].fillna("").astype(str).str.casefold().isin(wanted)]
        # Prefer model/year granularity and normal road cars unless explicitly non-car.
        if "BusinessGranularity" in work:
            preferred=work[work["BusinessGranularity"].fillna("").astype(str).str.upper().isin({"MODEL_YEAR","CATEGORY_YEAR","CATEGORY_NEAR_YEAR"})]
            if not preferred.empty: work=preferred
        sort_cols=[]; ascending=[]
        for col in ("OpportunityPercentile","ConfidenceAdjustedOpportunityIndex","ObservedExitWithin60DaysRate"):
            if col in work.columns: sort_cols.append(col); ascending.append(False)
        if sort_cols: work=work.sort_values(sort_cols,ascending=ascending)
        records=_df_records(work,100)
    except Exception:
        records=[]
    # Deduplicate model/year and avoid opaque scores in user-facing packet.
    out=[]; seen=set()
    for r in records:
        key=(str(r.get("Brand") or "").casefold(),str(r.get("Model") or "").casefold(),_int(r.get("Year")))
        if key in seen: continue
        seen.add(key)
        out.append(r)
        if len(out)>=12: break
    if specific_brand and specific_model:
        company=_text(c.get("company"),160)
        existing_count=0
        stock=_host(host,"business_stock_df")
        try:
            if company and stock is not None and not stock.empty:
                m=(stock["Company"].fillna("").astype(str).str.casefold()==company.casefold()) & (stock["Brand"].fillna("").astype(str).str.casefold()==specific_brand.casefold()) & (stock["Model"].fillna("").astype(str).str.casefold()==specific_model.casefold())
                existing_count=int(m.sum())
        except Exception:
            pass
        acquisition_price=_finite(c.get("acquisition_price") or target_obj.get("price") or target_obj.get("asking_price"))
        return {"kind":"business_acquisition_specific","subject":{"brand":specific_brand,"model":specific_model,"year":specific_year,"acquisition_price":acquisition_price},"existing_stock_count":existing_count,"market_evidence":out[:5],"status":"ok" if out else "insufficient"}
    return {"kind":"business_acquisition","options":out,"status":"ok" if out else "insufficient"}


def _business_period_tool(state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    company=_text((state.get("constraints") or {}).get("company"),160)
    c=state.get("constraints") or {}
    activity=_host(host,"business_activity_df")
    if activity is not None and not getattr(activity,"empty",True) and company:
        try:
            work=activity[activity["Company"].fillna("").astype(str).str.casefold()==company.casefold()].copy()
            if c.get("period_start") and "Date" in work:
                import pandas as pd
                work=work[pd.to_datetime(work["Date"],errors="coerce")>=pd.Timestamp(c["period_start"])]
            if c.get("period_end") and "Date" in work:
                import pandas as pd
                work=work[pd.to_datetime(work["Date"],errors="coerce")<=pd.Timestamp(c["period_end"])]
            work = work.sort_values("Date")
            daily_records = _df_records(work, 90)
            events={"new":[],"exits":[],"reductions":[],"increases":[]}
            def parse_event_json(value):
                try:
                    parsed=json.loads(str(value or "[]"))
                    return parsed if isinstance(parsed,list) else []
                except Exception:
                    return []
            for row in daily_records:
                events["new"].extend(parse_event_json(row.get("NewListingsJson")))
                events["exits"].extend(parse_event_json(row.get("ObservedExitListingsJson")))
                events["reductions"].extend(parse_event_json(row.get("PriceReductionListingsJson")))
                events["increases"].extend(parse_event_json(row.get("PriceIncreaseListingsJson")))
            for key in events:
                events[key]=events[key][:100]
            if daily_records:
                opening = _int(daily_records[0].get("OpeningObservedStockCount")) or 0
                closing = _int(daily_records[-1].get("ClosingObservedStockCount")) or 0
                summary = {
                    "period_start": str(daily_records[0].get("Date")),
                    "period_end": str(daily_records[-1].get("Date")),
                    "opening_observed_stock": opening,
                    "closing_observed_stock": closing,
                    "net_observed_stock_change": closing - opening,
                    "newly_observed_listings": sum(_int(x.get("NewlyObservedListings")) or 0 for x in daily_records),
                    "observed_market_exits": sum(_int(x.get("ObservedMarketExits")) or 0 for x in daily_records),
                    "asking_price_reductions": sum(_int(x.get("AskingPriceReductions")) or 0 for x in daily_records),
                    "asking_price_increases": sum(_int(x.get("AskingPriceIncreases")) or 0 for x in daily_records),
                    "observed_stock_definition": "Observed advertised stock in OtoDeğer data; not confirmed physical inventory",
                    "observed_exit_definition": "Observed market exit; not a confirmed sale",
                }
            else:
                summary = {}
            return {"kind":"business_period","company":company,"status":"ok","summary":summary,"events":events,"daily":daily_records}
        except Exception:
            pass
    # Current snapshot fallback is useful, but explicitly not a historical recap.
    comp=_host(host,"business_company_df")
    if comp is not None and not getattr(comp,"empty",True) and company:
        try:
            rows=comp[comp["Company"].fillna("").astype(str).str.casefold()==company.casefold()]
            return {"kind":"business_period","company":company,"status":"current_snapshot_only","snapshot":_df_records(rows,1)}
        except Exception:
            pass
    return {"kind":"business_period","company":company,"status":"unavailable"}


def _market_understanding_tool(state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    df=_host(host,"business_market_df")
    if df is None or getattr(df,"empty",True):
        search=_search_vehicle_models(state,"",host)
        return {"kind":"market_understanding","status":"current_market_only","models":search.get("models") or []}
    work=df.copy(); c=state.get("constraints") or {}
    try:
        if c.get("brands") and "Brand" in work:
            wanted={str(x).casefold() for x in c["brands"]}; work=work[work["Brand"].fillna("").astype(str).str.casefold().isin(wanted)]
        if c.get("models") and "Model" in work:
            wanted={str(x).casefold() for x in c["models"]}; work=work[work["Model"].fillna("").astype(str).str.casefold().isin(wanted)]
        if c.get("vehicle_type") and "VehicleType" in work:
            vt=str(c["vehicle_type"]).casefold(); work=work[work["VehicleType"].fillna("").astype(str).str.casefold().str.contains(vt,regex=False)]
        if "EvidenceQuality" in work:
            work=work[work["EvidenceQuality"].fillna("").astype(str).str.upper()!="INSUFFICIENT"]
        sort_cols=[x for x in ["ObservedExitWithin60DaysRate","CurrentListings"] if x in work.columns]
        if sort_cols: work=work.sort_values(sort_cols,ascending=[False]*len(sort_cols))
        rows=_df_records(work,20)
    except Exception:
        rows=[]
    return {"kind":"market_understanding","status":"ok" if rows else "insufficient","rows":rows}


def _execute_tool(action: str, state: Mapping[str, Any], resolved_plan: Mapping[str, Any], message: str, host: Mapping[str, Any]) -> Dict[str, Any]:
    targets=list(resolved_plan.get("target_ids") or [])
    if action == Action.SEARCH_VEHICLES.value:
        return _search_vehicle_models(state,message,host)
    if action == Action.SHOW_LISTINGS.value:
        return _target_listing_search(state,targets,host)
    if action == Action.COMPARE_VEHICLES.value:
        return _compare_models(state,targets,host)
    if action == Action.EVALUATE_PURCHASE.value:
        return _evaluate_price_subject(state,targets,host,"purchase_evaluation")
    if action in {Action.EVALUATE_SALE.value, Action.VALUE_VEHICLE.value}:
        return _evaluate_price_subject(state,targets,host,"sale_evaluation")
    if action == Action.ANALYZE_STOCK_PRICES.value:
        return _business_price_tool(state,host,targets)
    if action == Action.ANALYZE_AGING_STOCK.value:
        return _business_aging_tool(state,host,targets)
    if action == Action.RECOMMEND_ACQUISITIONS.value:
        return _business_acquire_tool(state,host,targets)
    if action == Action.ANALYZE_BUSINESS_PERIOD.value:
        return _business_period_tool(state,host)
    if action == Action.ANALYZE_MARKET.value:
        return _market_understanding_tool(state,host)
    if action == Action.EXPLAIN_RESULT.value:
        return {"kind":"explain_result","last_result":state.get("last_result")}
    if action == Action.ASK_CLARIFICATION.value:
        return {"kind":"clarification","awaiting":resolved_plan.get("awaiting")}
    if action in {Action.CHANGE_CONSTRAINTS.value, Action.CHANGE_FOCUS.value, Action.CONTINUE_CURRENT_GOAL.value}:
        # After a pure state change, choose the natural tool for current job.
        return _execute_tool(_default_action_for_job(state.get("job")),state,resolved_plan,message,host)
    return {"kind":"unsupported_action","action":action}


def _default_action_for_job(job: Optional[str]) -> str:
    return {
        Job.FIND_A_CAR.value: Action.SEARCH_VEHICLES.value,
        Job.EVALUATE_PURCHASE.value: Action.EVALUATE_PURCHASE.value,
        Job.EVALUATE_SALE.value: Action.EVALUATE_SALE.value,
        Job.COMPARE_CARS.value: Action.COMPARE_VEHICLES.value,
        Job.UNDERSTAND_MARKET.value: Action.ANALYZE_MARKET.value,
        Job.ACQUIRE_STOCK.value: Action.RECOMMEND_ACQUISITIONS.value,
        Job.PRICE_STOCK.value: Action.ANALYZE_STOCK_PRICES.value,
        Job.MOVE_AGING_STOCK.value: Action.ANALYZE_AGING_STOCK.value,
        Job.ANALYZE_BUSINESS.value: Action.ANALYZE_BUSINESS_PERIOD.value,
    }.get(job, Action.ASK_CLARIFICATION.value)


# ---------------------------------------------------------------------------
# Evidence -> state objects, decision policy, response
# ---------------------------------------------------------------------------

def _register_evidence_objects(state: Dict[str, Any], evidence: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    focus: List[str] = []
    shortlist: List[str] = []
    kind=evidence.get("kind")
    if kind == "vehicle_search":
        for item in (evidence.get("models") or [])[:12]:
            oid=register_object(state,ObjectType.MODEL.value,{"brand":item.get("brand"),"model":item.get("model")})
            shortlist.append(oid)
        # Search results may contain many viable models, but conversational focus
        # should represent the two primary options the assistant is presenting.
        # The wider set remains in shortlist for discovery; pronouns such as
        # "them" therefore resolve to the two active options rather than a
        # silent third model.
        focus=shortlist[:2]
    elif kind == "listings":
        for item in (evidence.get("listings") or [])[:12]:
            oid=register_object(state,ObjectType.LISTING.value,dict(item))
            shortlist.append(oid)
        focus=shortlist[:5]
    elif kind == "comparison":
        focus=[str(x.get("object_id")) for x in (evidence.get("vehicles") or []) if x.get("object_id")]
    elif kind in {"purchase_evaluation","sale_evaluation"}:
        # Keep existing subject focus; register comparable listings for follow-up links.
        for item in (evidence.get("comparables") or [])[:8]:
            try:
                shortlist.append(register_object(state,ObjectType.LISTING.value,dict(item)))
            except Exception:
                pass
    elif kind in {"business_pricing","business_aging"}:
        for row in (evidence.get("vehicles") or [])[:12]:
            payload={
                "link":row.get("Link"),"company":row.get("Company"),"brand":row.get("Brand"),"model":row.get("Model"),
                "year":_int(row.get("Year")),"price":_finite(row.get("CurrentAskingPrice")),"km":_int(row.get("KM")),
                "stock_age_days":_int(row.get("StockAgeDays")),"price_position":row.get("PricePositionBand"),
            }
            try: shortlist.append(register_object(state,ObjectType.STOCK_ITEM.value,payload))
            except Exception: pass
        focus=shortlist[:5]
    elif kind == "business_period":
        events=evidence.get("events") or {}
        for event_key in ("exits","reductions","new","increases"):
            for row in (events.get(event_key) or [])[:20]:
                payload={
                    "link":row.get("Link") or row.get("link") or row.get("listing_id"),
                    "company":evidence.get("company"),
                    "brand":row.get("Brand") or row.get("brand"),
                    "model":row.get("Model") or row.get("model"),
                    "year":_int(row.get("Year") or row.get("year")),
                    "event_type":event_key,
                    "old_price":_finite(row.get("old_price")),
                    "new_price":_finite(row.get("new_price")),
                }
                try:
                    shortlist.append(register_object(state,ObjectType.STOCK_ITEM.value,payload))
                except Exception:
                    pass
        focus=shortlist[:10]
    elif kind in {"business_acquisition","business_acquisition_specific"}:
        source_rows = (evidence.get("options") or []) if kind == "business_acquisition" else (evidence.get("market_evidence") or [])
        for row in source_rows[:12]:
            payload={"brand":row.get("Brand"),"model":row.get("Model"),"year":_int(row.get("Year")),"category":row.get("CategoryDetail")}
            try: shortlist.append(register_object(state,ObjectType.MODEL.value,payload))
            except Exception: pass
        focus=shortlist[:3]
    return focus, shortlist


def _decision_policy(state: Mapping[str, Any], action: str, evidence: Mapping[str, Any], message: str = "") -> Dict[str, Any]:
    kind=evidence.get("kind")
    if kind == "purchase_evaluation":
        if evidence.get("status") != "ok" or evidence.get("price") is None:
            return {"verdict":"INSUFFICIENT_EVIDENCE","reason":"Need a specific asking price and comparable market evidence."}
        vs=_finite(evidence.get("vs_median_pct")) or 0
        if vs >= 10: verdict="EXPENSIVE"
        elif vs <= -8: verdict="ATTRACTIVE_ASKING_PRICE"
        else: verdict="REASONABLE"
        return {"verdict":verdict,"reason":f"Asking price is {abs(vs):.1f}% {'above' if vs>0 else 'below'} the comparable median." if abs(vs)>=0.1 else "Asking price is close to the comparable median."}
    if kind == "sale_evaluation":
        if evidence.get("status") != "ok" or evidence.get("price") is None:
            return {"verdict":"INSUFFICIENT_EVIDENCE","reason":"Need the vehicle/offer details and comparable asking-price evidence."}
        vs=_finite(evidence.get("vs_median_pct")) or 0
        if vs <= -10: verdict="COUNTER_OR_REVIEW"
        elif vs >= 5: verdict="STRONG_RELATIVE_OFFER"
        else: verdict="REASONABLE_RELATIVE_OFFER"
        return {"verdict":verdict,"reason":f"Offer is {abs(vs):.1f}% {'above' if vs>0 else 'below'} the comparable asking-price median."}
    if kind == "vehicle_search":
        models=evidence.get("models") or []
        # Discovery is an option-mapping step unless the buyer has supplied a
        # decision preference strong enough to justify a recommendation.
        preferences=state.get("preferences") or {}
        priority_text=" ".join(str(v).casefold() for v in preferences.values() if v not in (None, "", [], {}))
        has_decision_priority=any(term in priority_text for term in ("econom", "reliab", "performance", "luxury", "comfort", "practical", "family", "commute"))
        return {
            "verdict": ("RECOMMEND_TOP_OPTION" if models and has_decision_priority else "PRESENT_OPTIONS") if models else "NO_MATCHES",
            "preferred": models[0] if models and has_decision_priority else None,
        }
    if kind == "comparison":
        vehicles=evidence.get("vehicles") or []
        if not vehicles: return {"verdict":"INSUFFICIENT_EVIDENCE"}
        low=_text(message,500).casefold()
        asks_for_choice=bool(re.search(r"\b(?:which|choose|pick|prefer|better|recommend|would you)\b|hangisi|seç|sec|tercih|öner|oner|какой|выбрать|лучше|предпоч", low, re.I))
        if not asks_for_choice:
            return {"verdict":"COMPARE_OPTIONS"}
        # When the buyer explicitly asks for a choice, compare what the budget
        # actually reaches: newest affordable year, its asking price, then breadth.
        ranked=sorted(vehicles,key=lambda x:(
            -(_int(x.get("newest_year")) or 0),
            _finite(x.get("newest_year_starting_price")) or 1e18,
            -int(x.get("count") or 0),
            _finite(x.get("median_price")) or 1e18,
        ))
        return {"verdict":"PREFER", "preferred_object_id":ranked[0].get("object_id")}
    if kind == "business_pricing":
        return {"verdict":"REVIEW_PRIORITY_PRICES" if evidence.get("vehicles") else "NO_PRIORITY_ISSUES_FOUND"}
    if kind == "business_aging":
        return {"verdict":"ACT_ON_OLDEST_FIRST" if evidence.get("vehicles") else "NO_AGING_STOCK_FOUND"}
    if kind == "business_acquisition":
        return {"verdict":"ACQUIRE_FROM_TOP_CANDIDATES" if evidence.get("options") else "INSUFFICIENT_EVIDENCE"}
    if kind == "business_acquisition_specific":
        rows=evidence.get("market_evidence") or []
        if not rows: return {"verdict":"INSUFFICIENT_EVIDENCE"}
        top=rows[0]
        signal=str(top.get("AcquisitionSignal") or "").upper()
        acquisition=_finite((evidence.get("subject") or {}).get("acquisition_price"))
        current_start=_finite(top.get("CurrentStartingPrice"))
        existing=int(evidence.get("existing_stock_count") or 0)
        if signal in {"VERY_STRONG","STRONG"} and acquisition is not None and current_start is not None and acquisition < current_start and existing < 3:
            verdict="STRONG_CANDIDATE"
        elif existing >= 3:
            verdict="REVIEW_EXISTING_EXPOSURE"
        else:
            verdict="CONSIDER_WITH_CAUTION"
        return {"verdict":verdict,"existing_stock_count":existing}
    if kind == "business_period":
        return {"verdict":"PERIOD_RECAP" if evidence.get("status")=="ok" else "HISTORICAL_ACTIVITY_NOT_READY"}
    return {"verdict":"INFORM"}


def _actions_from_evidence(state: Mapping[str, Any], action: str, evidence: Mapping[str, Any], language: str) -> Tuple[List[Dict[str,Any]], Optional[Dict[str,Any]]]:
    labels={
        "EN":{"listing":"Open listing","valuation":"Open valuation tool"},
        "TR":{"listing":"İlanı aç","valuation":"Değerleme aracını aç"},
        "RU":{"listing":"Открыть объявление","valuation":"Открыть оценку"},
    }[_lang(language)]
    actions=[]
    if evidence.get("kind") == "listings":
        # Listing actions are the actual shortlist presentation. Make each clickable
        # row information-rich enough for the buyer to decide whether it is worth
        # opening; do not reduce it to a generic year/brand/model card.
        currency = str(((state.get("constraints") or {}).get("currency") or "GBP")).upper()
        symbol = {"GBP":"£", "EUR":"€", "USD":"$"}.get(currency, currency + " ")
        for row in (evidence.get("listings") or [])[:10]:
            url=_text(row.get("link"),500)
            if not url: continue
            vehicle=" ".join(str(x) for x in [row.get("year"),row.get("brand"),row.get("model"),row.get("category")] if x not in (None,""))
            details=[]
            price=_finite(row.get("price"))
            if price is not None:
                details.append(f"{symbol}{price:,.0f}")
            km=_int(row.get("km"))
            if km is not None:
                details.append(f"{km:,} km")
            transmission=_text(row.get("transmission"),80)
            if transmission:
                tcf=transmission.casefold()
                if "otomatik" in tcf or "automatic" in tcf:
                    transmission={"EN":"Automatic","TR":"Otomatik","RU":"Автомат"}[_lang(language)]
                elif "manuel" in tcf or "manual" in tcf:
                    transmission={"EN":"Manual","TR":"Manuel","RU":"Механика"}[_lang(language)]
                details.append(transmission)
            fuel=_text(row.get("fuel"),80) or _text(row.get("fuel_type"),80)
            if fuel:
                fuel_key=fuel.casefold()
                fuel_labels={
                    "petrol":{"EN":"Petrol","TR":"Benzin","RU":"Бензин"},
                    "diesel":{"EN":"Diesel","TR":"Dizel","RU":"Дизель"},
                    "hybrid":{"EN":"Hybrid","TR":"Hibrit","RU":"Гибрид"},
                    "mild hybrid petrol":{"EN":"Mild hybrid petrol","TR":"Mild hibrit benzin","RU":"Мягкий гибрид, бензин"},
                    "mild hybrid diesel":{"EN":"Mild hybrid diesel","TR":"Mild hibrit dizel","RU":"Мягкий гибрид, дизель"},
                    "plug-in hybrid":{"EN":"Plug-in hybrid","TR":"Plug-in hibrit","RU":"Подключаемый гибрид"},
                    "electric":{"EN":"Electric","TR":"Elektrik","RU":"Электро"},
                }
                fuel=(fuel_labels.get(fuel_key) or {}).get(_lang(language), fuel)
                details.append(fuel)
            location=_text(row.get("location"),100)
            if location:
                details.append(location)
            company=_text(row.get("company"),160)
            if company:
                seller = ({"EN":"Private seller","TR":"Bireysel","RU":"Частный продавец"}[_lang(language)]
                          if company.casefold()=="bireysel" else company)
                details.append(seller)
            label=vehicle
            if details:
                label += " — " + " · ".join(details)
            actions.append({"type":"LISTING","label":label,"url":url})
        return actions, None
    if evidence.get("kind") in {"vehicle_search","comparison"}:
        target_ids=list(((state.get("focus") or {}).get("object_ids") or []))
        if target_ids:
            return actions,{"type":Action.SHOW_LISTINGS.value,"target_ids":target_ids,"metadata":{}}
    if evidence.get("kind") in {"purchase_evaluation","sale_evaluation"}:
        shortlist=list(state.get("shortlist") or [])
        if shortlist:
            return actions,{"type":Action.SHOW_LISTINGS.value,"target_ids":shortlist[:10],"metadata":{}}
    return actions,None


def _suggestions_from_context(state: Mapping[str, Any], evidence: Mapping[str, Any], language: str) -> List[str]:
    """Decision-guiding prompts. Ask for the user's bounds instead of inventing them."""
    lang=_lang(language)
    kind=evidence.get("kind")
    c=state.get("constraints") or {}
    models=evidence.get("models") or evidence.get("vehicles") or []
    if kind == "clarification":
        return (clarification(evidence.get("awaiting")) or {}).get("options", [])[:6]
    if (kind == "vehicle_search" and not models) or (kind == "listings" and not evidence.get("listings")) or (kind == "comparison" and len(models) < 2):
        # The answer asks what to relax. Do not stack more restrictions onto an empty set.
        return []

    base={
        "EN": {"year":"Set minimum year","km":"Set mileage limit","auto":"Automatic only","gallery":"Gallery sellers only",
               "links":"Show listings","compare":"Compare these options","other":"Show other brands",
               "econlux":"Economy or luxury?","broaden":"Broaden year range"},
        "TR": {"year":"Minimum yılı belirle","km":"KM sınırı belirle","auto":"Sadece otomatik","gallery":"Sadece galeriler",
               "links":"İlanları göster","compare":"Bu seçenekleri karşılaştır","other":"Diğer markaları göster",
               "econlux":"Ekonomik mi lüks mü?","broaden":"Yıl aralığını genişlet"},
        "RU": {"year":"Задать минимальный год","km":"Задать лимит пробега","auto":"Только автомат","gallery":"Только автосалоны",
               "links":"Показать объявления","compare":"Сравнить эти варианты","other":"Показать другие марки",
               "econlux":"Экономичность или премиум?","broaden":"Расширить диапазон лет"},
    }[lang]

    out=[]
    if kind=="vehicle_search":
        # Broad discovery should learn taste before stacking arbitrary hard filters.
        named = bool(c.get("brands") or c.get("models"))
        if not named:
            if not any((state.get("preferences") or {}).get(k) for k in ("economy", "luxury", "reliability", "performance", "comfort", "practicality")):
                out.append(base["econlux"])
            if not c.get("min_year"):
                out.append(base["year"])
            if not c.get("max_km"):
                out.append(base["km"])
        else:
            if len(models)>=2:
                out.append(base["compare"])
            if not c.get("min_year"):
                out.append(base["year"])
            if not c.get("max_km") and len(out)<3:
                out.append(base["km"])
            if evidence.get("alternative_models") and len(out)<3:
                out.append(base["other"])
    elif kind=="comparison":
        vehicles=evidence.get("vehicles") or []
        # If each newest-year route is represented by only one or two cars, the
        # best next step is to choose a model or broaden the year range—not add
        # another arbitrary filter.
        thin = bool(vehicles) and max(int(x.get("newest_year_count") or 0) for x in vehicles) <= 2
        if thin and vehicles:
            for x in vehicles[:2]:
                label=" ".join(v for v in [_text(x.get("brand"),80),_text(x.get("model"),100)] if v)
                if label:
                    out.append(("Focus on " + label) if lang=="EN" else ((label + "'e odaklan") if lang=="TR" else ("Сфокусироваться на " + label)))
            if len(out)<3:
                out.append(base["broaden"])
        else:
            out.append(base["links"])
            if not c.get("min_year"):
                out.append(base["year"])
            if not c.get("max_km") and len(out)<3:
                out.append(base["km"])
    elif kind=="listings":
        if not c.get("min_year"):
            out.append(base["year"])
        if not c.get("max_km"):
            out.append(base["km"])
        if not c.get("seller_type"):
            out.append(base["gallery"])
    return out[:3]


def _compact_evidence(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    e=copy.deepcopy(dict(evidence))
    # Keep model context rich but bounded.
    for key, limit in (("models",8),("listings",10),("vehicles",8),("options",8),("comparables",8),("rows",12),("daily",35)):
        if isinstance(e.get(key),list): e[key]=e[key][:limit]
    return e


def _fallback_answer(language: str, state: Mapping[str, Any], decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    lang=_lang(language); kind=evidence.get("kind")
    if kind=="vehicle_search":
        models=evidence.get("models") or []
        if not models:
            return {"EN":"I couldn't find a current match for those constraints. The most useful next step is to loosen one constraint.","TR":"Bu kriterlere uyan güncel bir seçenek bulamadım. En faydalı sonraki adım kriterlerden birini gevşetmek.","RU":"Я не нашёл актуальных вариантов по этим условиям. Лучше всего немного ослабить одно из ограничений."}[lang]
        shown=models[:8]
        def line(m):
            label=f"{m.get('brand')} {m.get('model')}"
            year=m.get('newest_year'); price=_money(m.get('newest_year_starting_price'))
            count=int(m.get('count') or 0)
            if lang=="TR": return f"**{label}** — eşleşen tüm yıllarda toplam {count} ilan · en yeni yıl: {year or '—'}, {price or '—'}'dan başlayan"
            if lang=="RU": return f"**{label}** — всего {count} объявлений по подходящим годам · самый новый год: {year or '—'}, от {price or '—'}"
            return f"**{label}** — {count} matching listings across all eligible years · newest: {year or '—'}, from {price or '—'}"
        alt=evidence.get("alternative_models") or []
        if lang=="TR":
            intro="Bunlar mevcut kriterleriniz içinde daha aktif görünen seçeneklerden bazıları; piyasanın tamamı değil."
            tail=(" Daha az aktif model alternatifleri de var" + (", örneğin " + ", ".join(alt[:3]) if alt else "") + ".")
        elif lang=="RU":
            intro="Это несколько более активных вариантов в рамках ваших условий, а не весь рынок."
            tail=(" Есть и менее активные модели" + (", например " + ", ".join(alt[:3]) if alt else "") + ".")
        else:
            intro="These are some of the more active matches within your criteria, not the full market."
            tail=(" Other lower-activity model options also exist" + (", including " + ", ".join(alt[:3]) if alt else "") + ".")
        return intro+"\n"+"\n".join(line(m) for m in shown)+tail
    if kind=="listings":
        n=min(10,len(evidence.get("listings") or []))
        if not n:
            return {"EN":"No current listings match all your criteria. Which restriction would you like to change?",
                    "TR":"Tüm kriterlerinize uyan güncel ilan yok. Hangi kriteri değiştirmek istersiniz?",
                    "RU":"Нет актуальных объявлений по всем вашим условиям. Какое ограничение вы хотите изменить?"}[lang]
        total=int(evidence.get("count") or n)
        if lang=="TR": return f"Aşağıda {n} güncel ilan gösteriyorum" + (f" ({total} eşleşme içinden)." if total>n else ".") + " Aşağıdaki her satır tıklanabilir ve temel ilan bilgilerini içerir; sonuçları yıl, kilometre veya satıcı tipine göre daha da daraltabiliriz."
        if lang=="RU": return f"Ниже показаны {n} актуальных объявлений" + (f" из {total} совпадений." if total>n else ".") + " Каждая строка ниже кликабельна и содержит основные данные объявления; затем можно сузить выбор по году, пробегу или типу продавца."
        return f"I’ve put {n} current listings below" + (f" from {total} matches." if total>n else ".") + " Each row below is clickable and includes the key advert details; we can narrow the set further by year, mileage or seller type."
    if kind=="comparison":
        vs=evidence.get("vehicles") or []
        if len(vs)>=2:
            a,b=vs[0],vs[1]
            def comp_line(x):
                label=f"{x.get('brand')} {x.get('model')}"
                year=x.get('newest_year'); price=_money(x.get('newest_year_starting_price'))
                year_count=int(x.get('newest_year_count') or 0); total=int(x.get('count') or 0)
                year_med=_money(x.get('newest_year_median_price'))
                year_high=_money(x.get('newest_year_highest_price'))
                if year_count > 1 and year_med and year_high and year_high != price:
                    price_part = f"{price}–{year_high}"
                else:
                    price_part = price or '—'
                if lang=="TR": return f"**{label}** — {year or '—'} · {price_part} · {year_count} adet {year or ''} ilanı" + (f" · toplam {total} uygun ilan" if total!=year_count else "")
                if lang=="RU": return f"**{label}** — {year or '—'} · {price_part} · {year_count} объявл. за {year or '—'}" + (f" · {total} всего" if total!=year_count else "")
                return f"**{label}** — {year or '—'} · {price_part} · {year_count} option{'s' if year_count!=1 else ''} at {year or 'that year'}" + (f" · {total} total within your filters" if total!=year_count else "")
            return comp_line(a)+"\n"+comp_line(b)
    if kind in {"purchase_evaluation","sale_evaluation"}:
        verdict=decision.get("verdict")
        return {"EN":f"My current view is **{str(verdict).replace('_',' ').lower()}** based on comparable asking prices. Asking prices are not confirmed sale prices, so I'd use this as a negotiation/decision signal rather than an exact valuation.","TR":f"Benim mevcut görüşüm, karşılaştırılabilir ilan fiyatlarına göre **{str(verdict).replace('_',' ').lower()}**. İlan fiyatları doğrulanmış satış fiyatları değildir; bunu kesin değer yerine karar/pazarlık sinyali olarak kullanmak daha doğru olur.","RU":f"По текущим сопоставимым ценам объявлений мой вывод: **{str(verdict).replace('_',' ').lower()}**. Цены объявлений не являются подтверждёнными ценами сделок, поэтому это ориентир для решения/торга, а не точная оценка."}[lang]
    if kind=="clarification":
        awaiting=clarification(evidence.get("awaiting"))
        if isinstance(awaiting, Mapping):
            q=_text(awaiting.get("question"),500)
        elif isinstance(awaiting, str):
            q=_text(awaiting,500)
        else:
            q=None
        if q: return q
    return {"EN":"I don't have enough verified market evidence to give you a confident recommendation yet. Give me the missing vehicle or decision detail and I'll narrow it properly.","TR":"Henüz güvenli bir öneri vermek için yeterli doğrulanmış piyasa verim yok. Eksik araç veya karar detayını verirseniz doğru şekilde daraltırım.","RU":"Пока недостаточно проверенных рыночных данных для уверенной рекомендации. Дайте недостающую деталь по автомобилю или решению, и я уточню ответ."}[lang]


def _render_answer(message: str, language: str, state: Mapping[str, Any], action: str, evidence: Mapping[str, Any], decision: Mapping[str, Any], host: Mapping[str, Any]) -> str:
    post=_host(host,"_openai_post"); model=_host(host,"OPENAI_MODEL","gpt-5.6-luna")
    fallback=_fallback_answer(language,state,decision,evidence)
    # Guided UI prompts are deterministic questions; do not let the renderer turn
    # them into a search result or invent a bound.
    if evidence.get("kind") == "clarification":
        return fallback
    if not callable(post): return fallback
    instructions=f"""
You are OtoDeğer AI, a premium decision copilot for the North Cyprus vehicle market.
Respond entirely in {_lang(language)}. Do not switch languages because vehicle names are foreign.
The user is trying to accomplish a decision, not receive a market report.

V11 product contract:
- Utility over verbosity: provide exactly the evidence needed for the decision and a sensible next step.
- Hard constraints are budget, explicit year bounds, body type, mileage bounds, transmission and explicit brand constraints. Never silently relax them. Explicit regional exclusions must also be respected.
- If a request is too broad to be useful (for example only "find me a vehicle"), ask ONE high-value narrowing question such as body type before searching. Budget + body type is enough to begin discovery.
- If zero matches remain, state the active restrictions and ask which one the user wants to relax. Do not choose for them.
- If only one or two good matches remain, show them and also explain which single relaxation would most plausibly broaden choice.
- A small budget stretch may be mentioned only when verified evidence shows a materially better option just above budget; otherwise respect the budget without upselling.
- Purchase evaluation: judge price competitiveness against genuinely comparable vehicles, show useful comparables, and explain that trim, condition, history, equipment or damage can justify differences. Do not tell the user categorically to buy/not buy. A mechanical inspection matters before a purchase recommendation.
- Sale evaluation: compare an offer/asking price with relevant market evidence and support negotiation decisions. Never turn observed listing removals into confirmed sales.
- Business acquisition: prioritise strong observed demand/activity and faster observed exits, with extra value when current supply is low.
- Business pricing: when evidence permits, frame quick-sale / competitive-market / premium strategies and note vehicle-specific condition/trim adjustments.
- Aging stock: focus on actionable price-position changes that can improve competitiveness.
- ANALYZE_BUSINESS covers business health, stock in/out, asking-price changes, inventory count/value, stock age, mix and trends for arbitrary periods; month-to-date, YTD and trailing 12 months are especially useful. Surface material changes proactively.
- UNDERSTAND_MARKET should answer the decision context: personal users care about buying timing/negotiating room; businesses care whether their performance reflects the wider market.
- Outside proprietary market scope, only answer North-Cyprus vehicle-adjacent questions when reliable external evidence is actually available; otherwise say it cannot be reliably verified. Do not become a generic assistant.
- English, Turkish and Russian must follow the same decision policy.

Rules:
- Match the response depth to the stage of the decision. Do not force a recommendation before the buyer has supplied preferences that make one meaningful.
- DISCOVERY / vehicle_search: map the useful option set. For a broad budget + body-type request, show the useful active model families rather than naming a winner; usually 4-8, but do not force a fixed count. The deterministic model order already prioritises observed market activity (historical listing volume + observed market exits) before current supply. Respect that order. Explicitly make clear these are SOME of the more active matches, not the entire market. If VERIFIED_EVIDENCE contains alternative_models, briefly name those model-level lower-activity alternatives and offer an expanded list. Never call an observed exit a confirmed sale/transaction and never equate current listing count alone with popularity.
- When a budget exists, model discovery is about WHAT THAT BUDGET BUYS. Say: MODEL — N matching listings across all eligible years · newest: YEAR, from £PRICE. N is NOT the number of cars from the newest year. Do not lead with an old model's overall minimum price.
- When the user names brands (for example BMW or Mercedes), show the relevant models under those brands with newest affordable year + asking price at that year + option count. Do not introduce mileage yet unless the user asks for mileage or is filtering listings by mileage.
- Do not append routine caveats such as "asking prices are not confirmed transaction prices" to ordinary discovery/comparison replies. Preserve that distinction internally and mention it only when it materially affects the decision.
- When the filtered choice is thin (especially 1-2 cars at the newest viable year), do not pretend the market is broad. Say it is thin and guide the user toward choosing a model, relaxing the minimum year, widening brand scope, or another constraint that actually increases choice.
- COMPARISON: make it scan-friendly. Stay on the SAME newest affordable year when quoting price statistics. Give one compact line per model using newest affordable year, the price/range at that year, number of listings at that year, and optionally the total number matching the user's active filters. NEVER pair a newest-year headline with a median calculated across older years. Do not compare mileage unless the user explicitly asks about mileage. Only recommend a winner if the user's latest message asks which to choose/buy/prefer or their stated preferences clearly support one.
- SHOW_LISTINGS: the UI displays up to ten information-rich clickable listing rows below the prose. Each row contains the exact year/brand/model/variant plus available price, KM, transmission, location and seller/gallery. The prose must be ONE short introductory sentence only; NEVER repeat/list any vehicle, price, seller, mileage or model in the prose. The structured clickable rows are the single listing presentation.
- Ordinary response 35-130 words; simple answers may be shorter. Do not exceed 170 words unless essential.
- Use short paragraphs and compact model-per-line formatting. Avoid long prose comparisons.
- Mention only facts present in VERIFIED_EVIDENCE. Never invent prices, years, mileage, availability, counts, dealers or links. If a hard budget is active, do not mention above-budget alternatives unless the user explicitly asks what spending more would unlock.
- Only describe something as the user's requirement/criterion if it is present in the supplied constraints/preferences or explicitly stated in user_message. Evidence attributes (for example an automatic transmission on a listing) are facts about the vehicle, not automatically user requirements.
- Asking prices are not confirmed transaction prices, but do NOT add that disclaimer routinely in discovery/comparison/listing answers. Mention it only when the distinction is materially relevant to a valuation, negotiation or sale-price claim.
- Observed market exit is not a confirmed sale.
- Price reduction is asking-price pressure, not depreciation/value retention.
- Never promise profit, sale probability, exact sale time or guaranteed value.
- LOW/insufficient evidence must sound uncertain.
- Challenge the user's plan when the evidence clearly supports a better route.
- Do not dump metrics just because they exist. Keep evidence in reserve.
- If the action is SHOW_LISTINGS, do not print raw URLs in prose; the UI receives buttons separately.
- If there is an offered next action, you may end with ONE short contextual question. Never add multiple CTAs.
- If clarification is needed, ask exactly one concise question and nothing else.
- Do not mention internal job/action/state/tool names, internal scores or implementation details.
""".strip()
    packet={
        "user_message":message,"goal":state.get("goal"),"constraints":state.get("constraints") or {},
        "preferences":state.get("preferences") or {},"action":action,"decision":decision,
        "VERIFIED_EVIDENCE":_compact_evidence(evidence),
    }
    try:
        r=post(payload={"model":model,"reasoning":{"effort":"low"},"max_output_tokens":550,"instructions":instructions,"input":json.dumps(packet,ensure_ascii=False,default=str)},timeout=(2.0,12.0))
        r.raise_for_status(); answer=_response_text(r.json()).strip()
        return answer or fallback
    except Exception as exc:
        print(f"V10_RENDERER_FALLBACK: {exc}",flush=True)
        return fallback


def _evidence_validate(answer: str, evidence: Mapping[str, Any], state: Optional[Mapping[str, Any]]=None) -> Tuple[bool,str]:
    """Conservative post-render checks for high-risk hallucination classes."""
    low=answer.casefold()
    # No raw URL should be emitted; structured actions own listing links.
    if "http://" in low or "https://" in low:
        return False,"RAW_URL_IN_ANSWER"
    # If tool found rows, renderer cannot claim absolute absence.
    evidence_count=0
    for key in ("models","listings","vehicles","options","rows"):
        if isinstance(evidence.get(key),list): evidence_count=max(evidence_count,len(evidence.get(key)))
    if evidence_count>0 and re.search(r"\b(?:no|none|nothing)\b.{0,40}\b(?:listing|option|vehicle|car|match)|\b(?:ilan|seçenek|arac).{0,30}(?:yok|bulunm)|\bнет.{0,30}(?:объяв|вариант)",low,re.I):
        return False,"FALSE_ABSENCE_CLAIM"

    # A listing/model attribute must never be promoted into a user requirement.
    # This catches the high-value transmission case deterministically in EN/TR/RU.
    constraints=((state or {}).get("constraints") or {}) if isinstance(state, Mapping) else {}
    if not constraints.get("transmission"):
        transmission_terms=r"(?:automatic(?:-transmission)?|manual|otomatik|manuel|автомат(?:ическ)?|механическ)"
        requirement_terms=r"(?:requirement|requirements|criteria|criterion|kriter|gereksin|şart|требован|критер)"
        if (re.search(transmission_terms+r".{0,70}"+requirement_terms,low,re.I) or
                re.search(requirement_terms+r".{0,70}"+transmission_terms,low,re.I)):
            return False,"UNSUPPORTED_TRANSMISSION_REQUIREMENT"

    # With a hard buyer budget, do not let the renderer wander into unrelated
    # above-budget alternatives. Small formatting/rounding tolerance only.
    if evidence.get("kind") in {"vehicle_search", "comparison", "listings"}:
        budget = _finite(constraints.get("budget_max"))
        if budget is not None:
            for token in re.findall(r"£\s*([0-9][0-9,\.]*)", answer):
                try:
                    value = float(token.replace(",", ""))
                except Exception:
                    continue
                if value > budget * 1.02:
                    return False,"ABOVE_BUDGET_PRICE_MENTION"

    # Counts attached to 'options/listings' in discovery/comparison must come
    # from deterministic evidence, not model improvisation.
    if evidence.get("kind") in {"vehicle_search", "comparison"}:
        allowed_counts = {int(x.get(key) or 0) for x in ((evidence.get("models") or []) + (evidence.get("vehicles") or [])) for key in ('count', 'newest_year_count')}
        allowed_counts.add(int(evidence.get("count") or 0))
        for m in re.finditer(r"\b(\d{1,4})\s+(?:current\s+)?(?:options?|listings?|seçenek|ilan|вариант)", answer, re.I):
            if int(m.group(1)) not in allowed_counts:
                return False,"UNSUPPORTED_OPTION_COUNT"
    return True,"OK"


def _compat_mode(action: str, job: Optional[str]) -> str:
    if action==Action.SHOW_LISTINGS.value: return "SHOP"
    if action==Action.COMPARE_VEHICLES.value: return "COMPARE"
    if job==Job.EVALUATE_PURCHASE.value: return "EVALUATE_PURCHASE"
    if job==Job.EVALUATE_SALE.value: return "EVALUATE_SALE"
    if job in {Job.PRICE_STOCK.value,Job.MOVE_AGING_STOCK.value,Job.ACQUIRE_STOCK.value,Job.ANALYZE_BUSINESS.value}: return job
    if job==Job.UNDERSTAND_MARKET.value: return "MARKET"
    return "DISCOVER"


def _public_state_summary(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "job":state.get("job"),"goal":state.get("goal"),"constraints":state.get("constraints") or {},
        "preferences":state.get("preferences") or {},"focus_ids":((state.get("focus") or {}).get("object_ids") or []),
        "shortlist_ids":state.get("shortlist") or [],"offered_action":state.get("offered_action"),
    }


def handle_v10_request(data: Mapping[str, Any], host: Mapping[str, Any]) -> Tuple[Dict[str, Any], int]:
    started=time.perf_counter()
    if not isinstance(data, Mapping) or not isinstance(data.get('message'), str):
        return {'success':False, 'error':'INVALID_REQUEST'}, 400
    if len(data['message']) > int(_host(host,'ASSISTANT_MAX_MESSAGE_CHARS',1200)):
        return {'success':False, 'error':'MESSAGE_TOO_LONG'}, 413
    raw_id = data.get('conversation_id')
    if raw_id is not None and (not isinstance(raw_id, str) or not raw_id.strip() or len(raw_id) > 200):
        return {'success':False, 'error':'INVALID_CONVERSATION_ID'}, 400
    revision = data.get('state_revision')
    if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0):
        return {'success':False, 'error':'INVALID_STATE_REVISION'}, 400
    message=_text(data.get("message"), int(_host(host,"ASSISTANT_MAX_MESSAGE_CHARS",1200)))
    language=_lang(data.get("language"))
    if not message:
        return {"success":False,"error":"MESSAGE_REQUIRED"},400

    access_tier=str(data.get("access_tier") or data.get("tier") or "PERSONAL").upper()
    audience=_audience_from_access(access_tier)
    conversation_id=_text(data.get("conversation_id"),200) or None
    client_revision=_int(data.get("state_revision"))

    service=get_state_service()
    try:
        stored=service.get_or_create(conversation_id,audience=audience,language=language)
    except StateConflict:
        return {'success':False,'error':'STATE_CONFLICT','conversation_id':conversation_id},409
    state=copy.deepcopy(stored.state)
    # Existing conversation audience is authoritative. Client cannot switch it mid-thread.
    language=_lang(language)
    state["language"]=language
    state['awaiting'] = clarification(state.get('awaiting'))

    if stored.revision > 1 and client_revision != stored.revision:
        return {"success":False,"error":"STATE_STALE","conversation_id":state.get("conversation_id"),"state_revision":stored.revision},409

    # Business company comes from current integration plumbing for now; state owns it once set.
    # Production auth can replace this input without changing the V10 state/tool architecture.
    company=_text(data.get("business_company"),160)
    if state.get("audience")==Audience.BUSINESS.value and company and not (state.get("constraints") or {}).get("company"):
        state["constraints"]["company"]=company

    semantic_started=time.perf_counter()
    try:
        raw_plan=_semantic_plan(message,language,state,host)
        planned_state,resolved_plan=apply_turn_plan(state,raw_plan)
    except (InvalidTurnPlan, ContractError) as exc:
        # Invalid/contradictory interpretation is not permission to execute an old plan.
        print(f"V11_PLAN_REJECTED: {type(exc).__name__}",flush=True)
        return {'success':False,'error':'ASSISTANT_INTERPRETATION_FAILED','conversation_id':stored.state['conversation_id'],
                'state_revision':stored.revision, 'retryable':True},422
    except Exception as exc:
        # Preserve both the previous criteria and revision on upstream failures.
        if isinstance(exc, _host(host, 'AIUsageLimitExceeded', ())):
            raise
        print(f"V11_PLANNER_UNAVAILABLE: {type(exc).__name__}",flush=True)
        return {'success':False,'error':'AI_ASSISTANT_TEMPORARILY_UNAVAILABLE','conversation_id':stored.state['conversation_id'],
                'state_revision':stored.revision, 'retryable':True},503
    semantic_seconds=time.perf_counter()-semantic_started

    action=str(resolved_plan.get("action") or _default_action_for_job(planned_state.get("job"))).upper()
    try:
        evidence=_execute_tool(action,planned_state,resolved_plan,message,host)
    except Exception as exc:
        print(f"V11_TOOL_UNAVAILABLE: {type(exc).__name__}",flush=True)
        return {'success':False,'error':'ASSISTANT_DATA_TEMPORARILY_UNAVAILABLE',
                'conversation_id':stored.state['conversation_id'],'state_revision':stored.revision,'retryable':True},503
    if not isinstance(evidence, Mapping):
        return {'success':False,'error':'ASSISTANT_DATA_TEMPORARILY_UNAVAILABLE','retryable':True},503
    if evidence.get("kind") == "clarification":
        evidence = dict(evidence, awaiting=clarification(evidence.get('awaiting')))
        action = Action.ASK_CLARIFICATION.value
        resolved_plan = dict(resolved_plan, awaiting=evidence.get("awaiting"))

    # Evidence becomes durable objects before response generation so follow-ups bind
    # to data identities, never prose.
    updated_state=copy.deepcopy(planned_state)
    evidence_focus,evidence_shortlist=_register_evidence_objects(updated_state,evidence)
    if evidence_shortlist:
        updated_state["shortlist"]=evidence_shortlist[:30]
    if evidence_focus:
        updated_state["focus"]={"object_ids":evidence_focus[:20]}
    elif action == Action.SEARCH_VEHICLES.value and evidence.get("kind") == "vehicle_search":
        # Never let a failed/refined search leave old models as the active referent.
        # Otherwise a subsequent "compare them" can compare stale, unrelated cars.
        updated_state["focus"]={"object_ids":[]}
        updated_state["shortlist"]=[]

    decision=_decision_policy(updated_state,action,evidence,message)
    actions,offered=_actions_from_evidence(updated_state,action,evidence,language)
    if offered is not None:
        updated_state["offered_action"]=offered
    else:
        updated_state["offered_action"]=None

    result_targets=list(((updated_state.get("focus") or {}).get("object_ids") or []))
    updated_state["last_result"]={"type":str(evidence.get("kind") or action).upper(),"target_ids":result_targets[:20],"metadata":{"decision":decision.get("verdict")}}
    updated_state["awaiting"]=resolved_plan.get("awaiting") if action==Action.ASK_CLARIFICATION.value else None

    if evidence.get("kind") == "listings":
        # Listing rows are rendered through structured clickable actions. Keep the
        # prose deterministic so the LLM cannot duplicate those same vehicles.
        answer=_fallback_answer(language,updated_state,decision,evidence)
    else:
        answer=_render_answer(message,language,updated_state,action,evidence,decision,host)
    valid,reason=_evidence_validate(answer,evidence,updated_state)
    if not valid:
        print(f"V10_RENDER_VALIDATION_FALLBACK: {reason}",flush=True)
        answer=_fallback_answer(language,updated_state,decision,evidence)

    # Construct and serialize the complete response BEFORE committing the turn.
    # A presentation failure must never consume a revision the client did not receive.
    listings=evidence.get("listings") or []
    models=evidence.get("models") or []
    business_options=evidence.get("vehicles") or evidence.get("options") or evidence.get("rows") or []
    payload = {
        "success":True,"answer":answer,
        "conversation_id":updated_state["conversation_id"],"state_revision":stored.revision + 1,
        "v10_version":V10_VERSION,"assistant_build":ASSISTANT_BUILD,"job":updated_state.get("job"),"action":action,
        "decision":decision.get("verdict"),"decision_mode":_compat_mode(action,updated_state.get("job")),
        "stage":"v10_decision_agent","filters":_legacy_filters(updated_state),"preferences":_legacy_preferences(updated_state),
        "count":int(evidence.get("count") or len(listings) or len(models) or len(business_options) or 0),
        "returned":min(10,len(listings)) if evidence.get("kind")=="listings" else len(listings),
        "results":[] if evidence.get("kind")=="listings" else listings[:20],
        "model_options":models[:8],"business_options":business_options[:12],"actions":actions,
        "suggestions":_suggestions_from_context(updated_state,evidence,language),
        "assistant_state":_public_state_summary(updated_state),
    }
    payload = _json_safe(payload)
    updated_state = _json_safe(updated_state)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    try:
        saved=service.store.save(updated_state,expected_revision=stored.revision)
    except StateConflict:
        return {"success":False,"error":"STATE_CONFLICT","conversation_id":state.get("conversation_id")},409
    except StateNotFound:
        return {"success":False,"error":"STATE_EXPIRED","conversation_id":state.get("conversation_id")},409
    payload["state_revision"] = saved.revision
    total=time.perf_counter()-started
    print(f"V10_TIMING action={action} semantic={semantic_seconds:.2f}s total={total:.2f}s",flush=True)
    return payload,200


__all__=["V10_VERSION","handle_v10_request"]
