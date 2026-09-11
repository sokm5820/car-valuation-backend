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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from otodeger_v10_state import (
    Action, Audience, Job, ObjectType, Transition,
    InvalidTurnPlan, StateConflict, StorageUnavailable,
    apply_turn_plan, get_state_service, register_object,
)

V10_VERSION = "10.2-guided-market"
SUPPORTED_LANGUAGES = {"TR", "EN", "RU"}


# ---------------------------------------------------------------------------
# Small safe helpers
# ---------------------------------------------------------------------------

def _lang(value: Any) -> str:
    value = str(value or "EN").upper().strip()
    return value if value in SUPPORTED_LANGUAGES else "EN"


def _text(value: Any, max_len: int = 500) -> str:
    return str(value or "").strip()[:max_len]


def _finite(value: Any) -> Optional[float]:
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


def _deterministic_followup_plan(message: str, state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute explicit offered actions without making an LLM rediscover referents."""
    low = _text(message, 1200).casefold()
    offered = state.get("offered_action") or {}
    offered_type = str(offered.get("type") or "").upper()
    offered_targets = list(offered.get("target_ids") or [])
    focus_ids = list(((state.get("focus") or {}).get("object_ids") or []))

    acceptance = bool(re.fullmatch(r"\s*(?:yes|yeah|yep|sure|ok|okay|do it|go ahead|please|evet|tamam|olur|göster|goster|да|хорошо|давай)\s*[.!]?\s*", low, flags=re.I))
    show_links = bool(re.search(r"\b(?:show|see|send|give|open).{0,20}\b(?:links?|listings?|ads?)\b|\b(?:links?|listings?)\b|\b(?:ilanları|ilanlari|linkleri|göster|goster)\b|\b(?:ссылк|объявлен)", low, re.I))
    compare = bool(re.search(r"\bcompare\b|\bcomparison\b|\bcompare them\b|\bthose two\b|\bkarşılaştır|karsilastir|сравн", low, re.I))

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
ANALYZE_BUSINESS_PERIOD, EXPLAIN_RESULT, CHANGE_CONSTRAINTS, CHANGE_FOCUS,
START_NEW_GOAL, CONTINUE_CURRENT_GOAL.
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

Hard constraints keys allowed: budget_min,budget_max,vehicle_type,brands,models,min_year,max_year,max_km,min_km,
transmission,seller_type,location,fuel_type,category,company,period_start,period_end,asking_price,offer_price,
acquisition_price,desired_sale_price,currency.
Preference keys: economy,reliability,performance,luxury,comfort,practicality,family,commute,size,resale,low_mileage,newer.

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
""".strip()

    payload = {
        "latest_message": message,
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
    return parsed


def _authoritative_numeric_constraints(message: str, parsed: Mapping[str, Any], state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    """Protect directly typed numbers from LLM scaling/format errors.

    This is deliberately narrow: it only overrides a field when the surrounding
    wording makes the number's semantic role clear. Meaning remains the model's
    job; literal numeric values remain deterministic evidence.
    """
    out = copy.deepcopy(dict(parsed))
    delta = dict(out.get("constraints_delta") or {})
    low = str(message or "").casefold()
    parser = _host(host, "_parse_human_number")

    def parse_token(token: str) -> Optional[float]:
        if callable(parser):
            try:
                v = parser(token)
                return _finite(v)
            except Exception:
                pass
        t = str(token or "").strip().lower().replace(" ", "")
        mult = 1000 if t.endswith("k") else 1
        if mult == 1000: t = t[:-1]
        if "," in t and "." in t: t = t.replace(",", "")
        elif t.count(",") == 1:
            a,b=t.split(","); t=a+b if len(b)==3 else a+"."+b
        elif t.count(".") == 1:
            a,b=t.split("."); t=a+b if len(b)==3 else a+"."+b
        try: return float(t)*mult
        except Exception: return None

    money_tokens = re.findall(r"£\s*([\d.,]+\s*[kK]?)|\b([\d.,]+\s*[kK]?)\s*(?:GBP|pounds?|sterlin)\b", str(message), re.I)
    flat=[]
    for a,b in money_tokens:
        tok=(a or b).replace(" ","")
        v=parse_token(tok)
        if v is not None and 100 <= v <= 1_000_000: flat.append(v)
    # A bare 15k / 15.000 next to clear budget wording.
    if not flat:
        m=re.search(r"\b([\d.,]+\s*[kK]?)\b", str(message))
        if m and re.search(r"\b(?:budget|bütçe|butce|бюджет|under|below|up to|max(?:imum)?|ceiling|spend|afford)\b", low, re.I):
            v=parse_token(m.group(1).replace(" ",""))
            if v is not None and 100 <= v <= 1_000_000: flat.append(v)
    if flat:
        value=flat[0]
        if re.search(r"\b(?:offer(?:ed)?|teklif|предлож)\b", low, re.I):
            delta["offer_price"]=value
        elif state.get("audience")==Audience.BUSINESS.value and re.search(r"\b(?:buy in|bring in|acquir|stock|offered|tedarik|stok)\b", low, re.I):
            delta["acquisition_price"]=value
        elif re.search(r"\b(?:asking|advertised|listed at|for sale at|ilan fiyat|satış fiyat|satis fiyat|price is)\b", low, re.I):
            delta["asking_price"]=value
        elif re.search(r"\b(?:budget|bütçe|butce|бюджет|under|below|up to|max(?:imum)?|ceiling|spend|afford|can i buy|what can i buy)\b", low, re.I):
            delta["budget_max"]=value

    # Explicit year floors/ceilings.
    m=re.search(r"\b(?:newer than|from|minimum year|min year|en az|sonrası|sonrasi|не старше)\s*(20\d{2}|19\d{2})\b", low, re.I)
    if m: delta["min_year"]=int(m.group(1))
    m=re.search(r"\b(?:older than|max year|up to year|en fazla|до)\s*(20\d{2}|19\d{2})\b", low, re.I)
    if m: delta["max_year"]=int(m.group(1))
    # Explicit mileage cap.
    m=re.search(r"\b(?:under|below|max(?:imum)?|less than|altında|altinda|en fazla|до)\s*([\d.,]+\s*[kK]?)\s*(?:km|kilomet)", low, re.I)
    if m:
        v=parse_token(m.group(1).replace(" ",""))
        if v is not None and 0 <= v <= 2_000_000: delta["max_km"]=int(round(v))

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
        "transmission": "transmissions", "category": "categories", "company": "companies",
    }
    for src, dest in mapping.items():
        val = c.get(src)
        if val in (None, "", [], {}):
            continue
        if dest in {"locations", "transmissions", "categories", "companies"} and not isinstance(val, list):
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
        return {"success": False, "error": "MARKET_SEARCH_UNAVAILABLE", "count": 0, "results": []}
    f = _legacy_filters(state)
    return fn(
        budget=f.get("budget"), min_budget=f.get("min_budget"), brands=f.get("brands"),
        models=f.get("models"), categories=f.get("categories"), locations=f.get("locations"),
        companies=f.get("companies"), exclude_companies=f.get("exclude_companies"), transmissions=f.get("transmissions"),
        min_year=f.get("min_year"), max_year=f.get("max_year"), min_km=f.get("min_km"), max_km=f.get("max_km"),
        limit=5000, max_limit=5000, analysis_mode=True,
    )


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
        return {"kind":"vehicle_search", "count":0, "models":[], "listings":[]}

    shortlist = _host(host, "shortlist_models_for_preferences")
    selector = _host(host, "_select_model_options")
    options: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    reasons: List[Dict[str, Any]] = []
    qualified: Sequence[Mapping[str, Any]] = rows
    if callable(shortlist) and callable(selector):
        try:
            qualified, reasons, summaries = shortlist(
                message, state.get("language") or "EN",
                _legacy_filters(state), _legacy_preferences(state), rows
            )
            summaries = summaries or _model_summary(qualified or rows)
            options = selector(summaries, reasons or [], _legacy_filters(state), max_options=8)
        except Exception:
            options = []
            summaries = []
            reasons = []
    if not options:
        summaries = _model_summary(rows, max_groups=80)
        options = summaries[:8]

    constraints = state.get("constraints") or {}
    preferences = state.get("preferences") or {}
    has_named_vehicle = bool(constraints.get("brands") or constraints.get("models"))
    has_decision_priority = any(
        preferences.get(k) not in (None, False, 0, "", [], {})
        for k in ("economy","reliability","performance","luxury","comfort","practicality","family","commute","size","resale","low_mileage","newer")
    )

    # Broad discovery should favour places where the observed North Cyprus market
    # is genuinely active, not merely whichever niche model happens to have two
    # very new ads today. Rank the FULL matching model set by historical observed
    # listing activity + observed exits, then use current choice/newness as ties.
    if not has_named_vehicle and not has_decision_priority:
        broad = _model_summary(rows, max_groups=120)
        activity = _historical_model_activity(host)
        enriched = []
        for item in broad:
            x = dict(item)
            hist = activity.get((str(x.get("brand") or "").casefold(), str(x.get("model") or "").casefold()), {})
            x.update(hist)
            enriched.append(x)
        options = sorted(
            enriched,
            key=lambda x: (
                -float(x.get("activity_score") or 0.0),
                -int(x.get("historical_distinct_listings") or 0),
                -int(x.get("count") or 0),
                -int(x.get("newest_year") or 0),
                float(x.get("newest_year_starting_price") or 1e18),
            ),
        )

    return {
        "kind":"vehicle_search",
        "count":len(rows),
        "models":[dict(x) for x in options[:8]],
        "listings":[],
        "discovery_style":"OPTIONS" if not has_named_vehicle else "BRAND_OPTIONS",
        "ranking_basis":"observed_market_activity" if not has_named_vehicle and not has_decision_priority else "current_fit",
    }

def _target_listing_search(state: Mapping[str, Any], target_ids: Sequence[str], host: Mapping[str, Any]) -> Dict[str, Any]:
    targets = _model_targets_from_objects(state, target_ids)
    fn = _host(host, "_search_market_for_vehicle_targets")
    if targets and callable(fn):
        base = _legacy_filters(state)
        # target search applies brand/model target-by-target; remove global brand/model cross product.
        for k in ("brands", "models", "categories"):
            base.pop(k, None)
        try:
            result = fn(base, targets)
            rows = result.get("results") or []
        except Exception:
            rows = []
    else:
        rows = _search_all(state, host).get("results") or []

    # Ranking is deliberate: not just cheapest. Favor current focus, newer year,
    # sensible mileage, and price while preserving actual listing evidence.
    def rank(r):
        year = _int(r.get("year")) or 0
        km = _int(r.get("km"))
        price = _finite(r.get("price")) or 1e18
        return (-year, km if km is not None else 10**9, price)
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
            brands=[brand], models=[model], categories=f.get("categories"),
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
        if len(tight) >= 3:
            rows = tight
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
        kms=[_int(x.get("km")) for x in rows if _int(x.get("km")) is not None]
        best_newest = sorted(rows, key=lambda x:(-(_int(x.get("year")) or 0), _finite(x.get("price")) or 1e18))[0]
        comps.append({
            "object_id": oid, "brand":payload.get("brand"), "model":payload.get("model"),
            "count":len(rows), "starting_price":min(prices) if prices else None,
            "median_price":statistics.median(prices) if prices else None,
            "newest_year":max(years) if years else None,
            "newest_year_starting_price":_finite(best_newest.get("price")),
            "median_km":statistics.median(kms) if kms else None,
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
        for row in (evidence.get("listings") or [])[:5]:
            url=_text(row.get("link"),500)
            if not url: continue
            name=" ".join(str(x) for x in [row.get("year"),row.get("brand"),row.get("model")] if x not in (None,""))
            actions.append({"type":"LISTING","label":f"{labels['listing']}: {name}","url":url})
        return actions, None
    if evidence.get("kind") in {"vehicle_search","comparison"}:
        target_ids=list(((state.get("focus") or {}).get("object_ids") or []))
        if target_ids:
            return actions,{"type":Action.SHOW_LISTINGS.value,"target_ids":target_ids,"metadata":{}}
    if evidence.get("kind") in {"purchase_evaluation","sale_evaluation"}:
        shortlist=list(state.get("shortlist") or [])
        if shortlist:
            return actions,{"type":Action.SHOW_LISTINGS.value,"target_ids":shortlist[:5],"metadata":{}}
    return actions,None


def _suggestions_from_context(state: Mapping[str, Any], evidence: Mapping[str, Any], language: str) -> List[str]:
    """Deterministic, useful narrowing prompts. Keep them actionable and short."""
    lang=_lang(language)
    kind=evidence.get("kind")
    c=state.get("constraints") or {}
    models=evidence.get("models") or evidence.get("vehicles") or []
    brands=[]
    for item in models:
        b=_text(item.get("brand"),80)
        if b and b not in brands:
            brands.append(b)

    text={
        "EN": {"km":"Under 100,000 km","year":"2018 or newer","auto":"Automatic only","gallery":"Gallery sellers only","links":"Show listings","compare":"Compare these options"},
        "TR": {"km":"100.000 km altı","year":"2018 ve üzeri","auto":"Sadece otomatik","gallery":"Sadece galeriler","links":"İlanları göster","compare":"Bu seçenekleri karşılaştır"},
        "RU": {"km":"До 100 000 км","year":"2018 года и новее","auto":"Только автомат","gallery":"Только автосалоны","links":"Показать объявления","compare":"Сравнить эти варианты"},
    }[lang]
    out=[]
    if kind=="vehicle_search":
        if len(models)>=2:
            out.append(text["compare"])
        if not c.get("min_year"):
            out.append(text["year"])
        if not c.get("max_km"):
            out.append(text["km"])
        if not c.get("transmission") and not c.get("transmissions"):
            out.append(text["auto"])
    elif kind=="comparison":
        out.append(text["links"])
        if not c.get("max_km"):
            out.append(text["km"])
        if not c.get("min_year"):
            out.append(text["year"])
    elif kind=="listings":
        if not c.get("max_km"):
            out.append(text["km"])
        if not c.get("min_year"):
            out.append(text["year"])
        if not c.get("seller_type"):
            out.append(text["gallery"])
    return out[:3]


def _compact_evidence(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    e=copy.deepcopy(dict(evidence))
    # Keep model context rich but bounded.
    for key, limit in (("models",5),("listings",5),("vehicles",5),("options",5),("comparables",5),("rows",8),("daily",35)):
        if isinstance(e.get(key),list): e[key]=e[key][:limit]
    return e


def _fallback_answer(language: str, state: Mapping[str, Any], decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    lang=_lang(language); kind=evidence.get("kind")
    if kind=="vehicle_search":
        models=evidence.get("models") or []
        if not models:
            return {"EN":"I couldn't find a current match for those constraints. The most useful next step is to loosen one constraint.","TR":"Bu kriterlere uyan güncel bir seçenek bulamadım. En faydalı sonraki adım kriterlerden birini gevşetmek.","RU":"Я не нашёл актуальных вариантов по этим условиям. Лучше всего немного ослабить одно из ограничений."}[lang]
        shown=models[:5]
        def line(m):
            label=f"{m.get('brand')} {m.get('model')}"
            year=m.get('newest_year'); price=_money(m.get('newest_year_starting_price'))
            count=int(m.get('count') or 0)
            if lang=="TR": return f"**{label}** — {year or 'yıl bilgisi yok'} · {price or 'fiyat yok'}'dan · {count} seçenek"
            if lang=="RU": return f"**{label}** — до {year or '—'} · {price or '—'} · {count} вариантов"
            return f"**{label}** — up to {year or '—'} · {year or 'newest year'} from {price or '—'} · {count} options"
        intro={"EN":"Your budget reaches several current options. I’d narrow the field before choosing a favourite.","TR":"Bütçeniz birkaç güncel seçeneğe ulaşıyor. Bir favori seçmeden önce alanı daraltmak daha doğru.","RU":"Ваш бюджет охватывает несколько актуальных вариантов. Сначала лучше сузить выбор, а не выбирать фаворита."}[lang]
        return intro+"\n"+"\n".join(line(m) for m in shown)
    if kind=="listings":
        n=min(5,len(evidence.get("listings") or []))
        total=int(evidence.get("count") or n)
        if lang=="TR": return f"Aşağıda {n} güncel ilan gösteriyorum" + (f" ({total} eşleşme içinden)." if total>n else ".") + " Kartlardan ilanları açabilirsiniz; sonuçları yıl, kilometre veya satıcı tipine göre daha da daraltabiliriz."
        if lang=="RU": return f"Ниже показаны {n} актуальных объявлений" + (f" из {total} совпадений." if total>n else ".") + " Откройте их карточками ниже; затем можно сузить по году, пробегу или типу продавца."
        return f"I’ve put {n} current listings below" + (f" from {total} matches." if total>n else ".") + " Open them from the cards; we can narrow the set further by year, mileage or seller type."
    if kind=="comparison":
        vs=evidence.get("vehicles") or []
        if len(vs)>=2:
            a,b=vs[0],vs[1]
            def comp_line(x):
                label=f"{x.get('brand')} {x.get('model')}"
                year=x.get('newest_year'); price=_money(x.get('newest_year_starting_price'))
                med=_money(x.get('median_price')); count=int(x.get('count') or 0)
                if lang=="TR": return f"**{label}** — en yeni {year or '—'} · {year or 'bu yıl'} {price or '—'}'dan · medyan {med or '—'} · {count} seçenek"
                if lang=="RU": return f"**{label}** — новейший {year or '—'} · {price or '—'} · медиана {med or '—'} · {count} вариантов"
                return f"**{label}** — newest {year or '—'} · {year or 'newest year'} from {price or '—'} · median {med or '—'} · {count} options"
            return comp_line(a)+"\n"+comp_line(b)
    if kind in {"purchase_evaluation","sale_evaluation"}:
        verdict=decision.get("verdict")
        return {"EN":f"My current view is **{str(verdict).replace('_',' ').lower()}** based on comparable asking prices. Asking prices are not confirmed sale prices, so I'd use this as a negotiation/decision signal rather than an exact valuation.","TR":f"Benim mevcut görüşüm, karşılaştırılabilir ilan fiyatlarına göre **{str(verdict).replace('_',' ').lower()}**. İlan fiyatları doğrulanmış satış fiyatları değildir; bunu kesin değer yerine karar/pazarlık sinyali olarak kullanmak daha doğru olur.","RU":f"По текущим сопоставимым ценам объявлений мой вывод: **{str(verdict).replace('_',' ').lower()}**. Цены объявлений не являются подтверждёнными ценами сделок, поэтому это ориентир для решения/торга, а не точная оценка."}[lang]
    if kind=="clarification":
        awaiting=evidence.get("awaiting")
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
    if not callable(post): return fallback
    instructions=f"""
You are OtoDeğer AI, a premium decision copilot for the North Cyprus vehicle market.
Respond entirely in {_lang(language)}. Do not switch languages because vehicle names are foreign.
The user is trying to accomplish a decision, not receive a market report.

Rules:
- Match the response depth to the stage of the decision. Do not force a recommendation before the buyer has supplied preferences that make one meaningful.
- DISCOVERY / vehicle_search: map the useful option set. For a broad budget + body-type request, show 4-6 model families rather than naming a winner. The deterministic model order already prioritises observed market activity (historical listing volume + observed market exits) before current supply. Respect that order. You may describe these as more active parts of the observed market, but NEVER call an observed exit a confirmed sale/transaction and never equate current listing count alone with popularity.
- When a budget exists, model discovery is about WHAT THAT BUDGET BUYS. Prefer: MODEL — up to YEAR · YEAR from £PRICE · N options. Do not lead with an old model's overall minimum price.
- When the user names brands (for example BMW or Mercedes), show the relevant models under those brands with newest affordable year + asking price at that year + option count. Do not introduce mileage yet unless the user asks for mileage or is filtering listings by mileage.
- COMPARISON: make it scan-friendly. Give one compact line per model using newest affordable year, price at that year, median asking price and current option count. Do not compare median/representative mileage unless the user explicitly asks about mileage. Only recommend a winner if the user's latest message asks which to choose/buy/prefer or their stated preferences clearly support one.
- SHOW_LISTINGS: the UI displays at most five listing cards below the prose. Say how many are actually being shown (max 5), not merely the total number of matches. You may separately say 'from N matches'. DO NOT repeat/list the vehicles in the prose and do not create markdown links; the structured UI cards are the single listing presentation.
- Ordinary response 35-130 words; simple answers may be shorter. Do not exceed 170 words unless essential.
- Use short paragraphs and compact model-per-line formatting. Avoid long prose comparisons.
- Mention only facts present in VERIFIED_EVIDENCE. Never invent prices, years, mileage, availability, counts, dealers or links. If a hard budget is active, do not mention above-budget alternatives unless the user explicitly asks what spending more would unlock.
- Only describe something as the user's requirement/criterion if it is present in the supplied constraints/preferences or explicitly stated in user_message. Evidence attributes (for example an automatic transmission on a listing) are facts about the vehicle, not automatically user requirements.
- Current listing asking prices are not confirmed transaction prices.
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
        allowed_counts = {int(x.get("count") or 0) for x in ((evidence.get("models") or []) + (evidence.get("vehicles") or []))}
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
    message=_text(data.get("message"), int(_host(host,"ASSISTANT_MAX_MESSAGE_CHARS",1200)))
    language=_lang(data.get("language"))
    if not message:
        return {"success":False,"error":"MESSAGE_REQUIRED"},400

    access_tier=str(data.get("access_tier") or data.get("tier") or "PERSONAL").upper()
    audience=_audience_from_access(access_tier)
    conversation_id=_text(data.get("conversation_id"),200) or None
    client_revision=_int(data.get("state_revision"))

    service=get_state_service()
    stored=service.get_or_create(conversation_id,audience=audience,language=language)
    state=copy.deepcopy(stored.state)
    # Existing conversation audience is authoritative. Client cannot switch it mid-thread.
    language=_lang(language)
    state["language"]=language

    if client_revision is not None and stored.revision>1 and client_revision != stored.revision:
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
    except (InvalidTurnPlan, Exception) as exc:
        # One bounded second semantic attempt with a validation hint is preferable to
        # silently falling back into unrelated broad search.
        print(f"V10_PLAN_DEGRADED: {exc}",flush=True)
        fallback_action=_default_action_for_job(state.get("job"))
        if not state.get("job"):
            q={"EN":"What are you trying to decide — finding a car, checking a specific car, selling your car, or understanding the market?","TR":"Hangi kararı vermeye çalışıyorsunuz — araç bulmak, belirli bir aracı değerlendirmek, aracınızı satmak veya piyasayı anlamak mı?","RU":"Какое решение вы принимаете — подобрать машину, оценить конкретный автомобиль, продать свою машину или понять рынок?"}[language]
            raw_plan={"transition":"CONTINUE","action":"ASK_CLARIFICATION","job":None,"awaiting":{"question":q}}
        else:
            raw_plan={"transition":"CONTINUE","action":fallback_action,"job":state.get("job")}
        planned_state,resolved_plan=apply_turn_plan(state,raw_plan)
    semantic_seconds=time.perf_counter()-semantic_started

    action=str(resolved_plan.get("action") or _default_action_for_job(planned_state.get("job"))).upper()
    evidence=_execute_tool(action,planned_state,resolved_plan,message,host)

    # Evidence becomes durable objects before response generation so follow-ups bind
    # to data identities, never prose.
    updated_state=copy.deepcopy(planned_state)
    evidence_focus,evidence_shortlist=_register_evidence_objects(updated_state,evidence)
    if evidence_shortlist:
        updated_state["shortlist"]=evidence_shortlist[:30]
    if evidence_focus:
        updated_state["focus"]={"object_ids":evidence_focus[:20]}

    decision=_decision_policy(updated_state,action,evidence,message)
    actions,offered=_actions_from_evidence(updated_state,action,evidence,language)
    if offered is not None:
        updated_state["offered_action"]=offered
    elif action==Action.ASK_CLARIFICATION.value:
        updated_state["offered_action"]=None

    result_targets=list(((updated_state.get("focus") or {}).get("object_ids") or []))
    updated_state["last_result"]={"type":str(evidence.get("kind") or action).upper(),"target_ids":result_targets[:20],"metadata":{"decision":decision.get("verdict")}}
    updated_state["awaiting"]=resolved_plan.get("awaiting") if action==Action.ASK_CLARIFICATION.value else None

    answer=_render_answer(message,language,updated_state,action,evidence,decision,host)
    valid,reason=_evidence_validate(answer,evidence,updated_state)
    if not valid:
        print(f"V10_RENDER_VALIDATION_FALLBACK: {reason}",flush=True)
        answer=_fallback_answer(language,updated_state,decision,evidence)

    # Persist once, after semantic state + tool-derived focus/result/offered action are complete.
    # We intentionally do not call service.apply_plan twice; one CAS write per user turn.
    try:
        saved=service.store.save(updated_state,expected_revision=stored.revision)
    except StateConflict:
        return {"success":False,"error":"STATE_CONFLICT","conversation_id":state.get("conversation_id")},409

    listings=evidence.get("listings") or []
    models=evidence.get("models") or []
    business_options=evidence.get("vehicles") or evidence.get("options") or evidence.get("rows") or []
    compat_filters=_legacy_filters(saved.state)
    compat_preferences=_legacy_preferences(saved.state)
    total=time.perf_counter()-started
    print(f"V10_TIMING action={action} semantic={semantic_seconds:.2f}s total={total:.2f}s",flush=True)

    return {
        "success":True,"answer":answer,
        "conversation_id":saved.state.get("conversation_id"),"state_revision":saved.revision,
        "v10_version":V10_VERSION,"job":saved.state.get("job"),"action":action,
        "decision":decision.get("verdict"),"decision_mode":_compat_mode(action,saved.state.get("job")),
        "stage":"v10_decision_agent","filters":compat_filters,"preferences":compat_preferences,
        "count":int(evidence.get("count") or len(listings) or len(models) or len(business_options) or 0),
        "returned":len(listings),"results":listings[:20],"model_options":models[:8],
        "business_options":business_options[:12],"actions":actions,"suggestions":_suggestions_from_context(saved.state,evidence,language),
        "assistant_state":_public_state_summary(saved.state),
    },200


__all__=["V10_VERSION","handle_v10_request"]
