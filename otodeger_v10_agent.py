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
ASSISTANT_BUILD = "11.6-gold-release-candidate"
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
    allowed_words = {'make','the','my','budget','is','now','actually','set','to','add','increase','raise','reduce','decrease','subtract','by','from',
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
    # Contextual engine/variant edits such as "1.0" or "1.2L" belong to the
    # vehicle currently being evaluated; never reinterpret them as a budget/year.
    engine_only = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:l|litre|liter)?\s*", raw, flags=re.I)
    if engine_only and state.get("job") in {Job.EVALUATE_PURCHASE.value, Job.EVALUATE_SALE.value, Job.EVALUATE_TRADE_IN.value}:
        candidate_ids = focus_ids or list(state.get("shortlist") or [])
        if candidate_ids:
            current_id = candidate_ids[0]
            current = (state.get("objects") or {}).get(current_id) or {}
            if str(current.get("type") or "").upper() in {ObjectType.LISTING.value, ObjectType.OWNED_VEHICLE.value}:
                payload = {k:v for k,v in current.items() if k not in {"id","type"}}
                payload["category"] = engine_only.group(1).replace(",", ".")
                alias = "variant_vehicle"
                return {
                    "transition":Transition.REFINE.value,
                    "action":_default_action_for_job(state.get("job")),
                    "job":state.get("job"),
                    "objects":[{"type":str(current.get("type") or ObjectType.LISTING.value).upper(),"payload":payload,"alias":alias}],
                    "target_ids":[alias],
                    "clear_constraints":["category"],
                    "awaiting":None,
                }

    # Comparing several years of the currently selected model is a first-class
    # operation. Model-year objects are intentionally distinct in V11.3.
    years = []
    for token in re.findall(r"\b((?:19|20)\d{2})\b", raw):
        y = int(token)
        if y not in years:
            years.append(y)
    c = state.get("constraints") or {}
    selected_models = list(c.get("models") or [])
    selected_brands = list(c.get("brands") or [])
    year_compare_language = bool(re.search(r"\b(compare|versus|vs|between|amongst|among|decide|tell me about|what about)\b|karşılaştır|karsilastir|arasında|arasinda|сравн|между", low, re.I))
    if len(years) >= 2 and len(selected_models) == 1 and year_compare_language:
        brand = selected_brands[0] if len(selected_brands) == 1 else None
        model_name = selected_models[0]
        specs=[]; targets=[]
        for i, year in enumerate(years[:4]):
            alias=f"model_year_{i+1}"
            payload={"brand":brand,"model":model_name,"year":year}
            specs.append({"type":ObjectType.MODEL.value,"payload":payload,"alias":alias})
            targets.append(alias)
        return {
            "transition":Transition.SWITCH_SUBTASK.value,
            "action":Action.COMPARE_VEHICLES.value,
            "job":Job.COMPARE_CARS.value,
            "objects":specs,
            "target_ids":targets,
            "awaiting":None,
        }

    # Once a model has been explicitly selected, a lone model year is naturally a
    # year refinement ("what about 2024?"), not an ambiguous budget/year/km number.
    if len(years) == 1 and len(selected_models) == 1 and state.get("job") in {Job.FIND_A_CAR.value, Job.COMPARE_CARS.value}:
        year = years[0]
        if re.fullmatch(r"\s*(?:what about|how about|try|show me|maybe)?\s*(?:a\s+)?(?:19|20)\d{2}(?:\s+(?:one|model))?\s*[?.!]*\s*", low, re.I):
            return {
                "transition": Transition.REFINE.value,
                "action": Action.SEARCH_VEHICLES.value,
                "job": Job.FIND_A_CAR.value,
                "constraints_delta": {"min_year": year, "max_year": year},
                "awaiting": None,
            }

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
    compare_targets = offered_targets or focus_ids
    if compare and len(compare_targets) == 2:
        return {
            "transition": Transition.SWITCH_SUBTASK.value,
            "action": Action.COMPARE_VEHICLES.value,
            "job": Job.COMPARE_CARS.value if state.get("audience") == Audience.PERSONAL.value else state.get("job"),
            "target_ids": compare_targets,
        }
    if compare and len(compare_targets) > 2:
        objects = state.get("objects") or {}
        labels=[]
        for oid in compare_targets[:8]:
            obj=objects.get(oid) or {}
            label=" ".join(str(x) for x in [obj.get("year"),obj.get("brand"),obj.get("model")] if x not in (None,""))
            if label and label not in labels:
                labels.append(label)
        choices=", ".join(labels)
        q={
            "EN":f"Which 2-4 would you like to compare? Current options: {choices}",
            "TR":f"Hangi 2-4 seçeneği karşılaştırmak istersiniz? Mevcut seçenekler: {choices}",
            "RU":f"Какие 2-4 варианта сравнить? Текущие варианты: {choices}",
        }[lang]
        return {
            "transition":Transition.CONTINUE.value,
            "action":Action.ASK_CLARIFICATION.value,
            "job":state.get("job") or Job.FIND_A_CAR.value,
            "awaiting":{"field":"compare_targets","question":q},
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
    evidence["explicit_years"] = [int(x) for x in re.findall(r"\b((?:19|20)\d{2})\b", str(message or ""))][:8]
    engine = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:l|litre|liter|litres|liters)\b", str(message or ""), flags=re.I)
    if engine:
        evidence["engine_size"] = engine.group(1).replace(",", ".")
    return evidence


def _semantic_plan(message: str, language: str, state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    deterministic = _deterministic_followup_plan(message, state)
    if deterministic:
        return deterministic

    post = _host(host, "_openai_post")
    model = _host(host, "OPENAI_MODEL", "gpt-5.6-luna")

    authoritative = _authoritative_message_evidence(message, host)

    # Straightforward dealer trade-ins should not depend on an upstream planner
    # call merely to identify the task. If the vehicle family is authoritative,
    # build the trade-in subject deterministically and let Python own valuation.
    # This also provides graceful continuity during a transient AI planning outage.
    if state.get("audience") == Audience.BUSINESS.value and re.search(
        r"\btrade[ -]?in\b|\bpart exchange\b|\btakas\b|\bобмен\b",
        str(message or ""), re.I):
        targets = list(authoritative.get("vehicle_targets") or [])
        if targets:
            payload = dict(targets[0])
            years = list(authoritative.get("explicit_years") or [])
            if years and payload.get("year") in (None, ""):
                payload["year"] = years[0]
            if authoritative.get("engine_size") and not payload.get("category"):
                payload["category"] = authoritative["engine_size"]
            elif not payload.get("category"):
                # In normal dealer speech engine size is often written as
                # "Yaris 1.5 automatic" without an L suffix. A standalone
                # decimal in a resolved trade-in vehicle description is therefore
                # a safe category/engine hint; prices and mileage are handled by
                # their own units below.
                engine_decimal = re.search(r"(?<![\d,.])(\d{1,2}[.,]\d)(?![\d,.])", str(message or ""))
                if engine_decimal:
                    payload["category"] = engine_decimal.group(1).replace(",", ".")
            km_match = re.search(r"\b(\d[\d,.]*)\s*(?:km|kilomet(?:er|re)s?)\b", str(message or ""), re.I)
            if km_match:
                try:
                    payload["km"] = int(parse_number(km_match.group(1)))
                except Exception:
                    pass
            low_message = str(message or "").casefold()
            if re.search(r"\b(?:automatic|auto|otomatik|автомат)\b", low_message):
                payload["transmission"] = "Automatic"
            elif re.search(r"\b(?:manual|manuel|düz|duz|механик)\b", low_message):
                payload["transmission"] = "Manual"
            return {
                "transition": Transition.START_NEW_GOAL.value if not state.get("job") else Transition.SWITCH_SUBTASK.value,
                "action": Action.EVALUATE_TRADE_IN.value,
                "job": Job.EVALUATE_TRADE_IN.value,
                "goal_summary": "Evaluate an incoming trade-in",
                "constraints_delta": {},
                "clear_constraints": [],
                "preferences_delta": {},
                "clear_preferences": [],
                "objects": [{"type": ObjectType.OWNED_VEHICLE.value, "alias": "trade_in_vehicle", "payload": payload}],
                "target_ids": ["trade_in_vehicle"],
                "shortlist_ids": [],
                "awaiting": None,
            }

    if not callable(post):
        raise RuntimeError("V10_OPENAI_POST_UNAVAILABLE")

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
BUSINESS jobs: ACQUIRE_STOCK, PRICE_STOCK, MOVE_AGING_STOCK, ANALYZE_BUSINESS, EVALUATE_TRADE_IN, PROMOTE_STOCK, UNDERSTAND_MARKET.
Actions: ASK_CLARIFICATION, SEARCH_VEHICLES, SHOW_LISTINGS, COMPARE_VEHICLES,
EVALUATE_PURCHASE, EVALUATE_SALE, VALUE_VEHICLE, ANALYZE_MARKET,
ANALYZE_STOCK_PRICES, ANALYZE_AGING_STOCK, RECOMMEND_ACQUISITIONS,
ANALYZE_BUSINESS_PERIOD, EVALUATE_TRADE_IN, RECOMMEND_AD_CANDIDATE, EXPLAIN_RESULT.
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
- Personal owner asking what their vehicle is worth, what a good selling/listing price is, or how much they should sell/list it for: job EVALUATE_SALE + action VALUE_VEHICLE. This is a valuation request and does NOT require the user to supply an asking price or offer price. Never ask the user for the price they are asking OtoDeğer to determine.
- Personal owner asking whether a specific offer is fair, whether to accept/counter an offer, or whether an existing asking price should be reduced: job EVALUATE_SALE + action EVALUATE_SALE.
- Business user asking what to bring in / whether to acquire a specific car: ACQUIRE_STOCK + RECOMMEND_ACQUISITIONS.
- Business pricing question: PRICE_STOCK + ANALYZE_STOCK_PRICES.
- Business aging/sitting/slow-stock question: MOVE_AGING_STOCK + ANALYZE_AGING_STOCK.
- Business period/change/recap question: ANALYZE_BUSINESS + ANALYZE_BUSINESS_PERIOD.
- Business user asking how much to offer for an incoming trade-in: EVALUATE_TRADE_IN + EVALUATE_TRADE_IN.
- Business user asking which current stock vehicle to advertise/promote/boost: PROMOTE_STOCK + RECOMMEND_AD_CANDIDATE.
- If the user has explicitly selected/named a model and then changes only year, mileage, fuel, colour, transmission, seller type, location or budget, KEEP that model selected. Do not reopen unrelated models unless they ask for alternatives/other models or name a different model.
- If the user asks to compare multiple years of one model, treat each model-year as a distinct comparison target. Do not collapse them into one model family.
- AUTHORITATIVE_MESSAGE_EVIDENCE.engine_size is an explicit engine/variant token from forms such as 1.2L; use it as category when appropriate instead of asking the user to repeat it.
- In a focused PRICE_STOCK conversation, wording such as "sell it quickly", "get it gone", or "not maximum margin" is a pricing-strategy refinement: stay PRICE_STOCK + ANALYZE_STOCK_PRICES. Do not switch to MOVE_AGING_STOCK unless the user is asking which stock is old/slow/stuck.
- Historical questions about asking-price reductions/cuts (for example "which price cuts were biggest?") are ANALYZE_BUSINESS + ANALYZE_BUSINESS_PERIOD, not PRICE_STOCK. PRICE_STOCK is for what price to set now.
- A business question contrasting the company with the wider market (for example "was it us or the whole market?") is UNDERSTAND_MARKET + ANALYZE_MARKET. Do not ask for sales/revenue/margin metrics first.
- Month names without a year mean the most recent occurrence of that month relative to current_date supplied in the input. Do not ask which year when that interpretation is unambiguous.
- For North Cyprus vehicle discovery, location normally does not block a useful first answer. If a new broad vehicle goal has a body type but no numeric budget, ask only for maximum budget rather than also asking for a preferred location.

Hard constraints keys allowed: budget_min,budget_max,vehicle_type,brands,models,min_year,max_year,max_km,min_km,
transmission,seller_type,location,fuel_type,exclude_locations,exclude_fuels,colors,category,company,period_start,period_end,asking_price,offer_price,
acquisition_price,desired_sale_price,target_margin_pct,prep_allowance,currency.
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
    parsed = None
    last_planner_error = None
    for attempt in range(2):
        try:
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
            break
        except Exception as exc:
            last_planner_error = exc
            if attempt == 0:
                time.sleep(0.25)
                continue
            raise
    if not parsed:
        raise last_planner_error or RuntimeError("V10_SEMANTIC_PLAN_INVALID")

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

    if audience == Audience.PERSONAL.value:
        delta = dict(out.get("constraints_delta") or {})

        # Receiving an offer on the user's own vehicle is a seller-evaluation task,
        # even if the same sentence also asks what the car is worth. This is a
        # higher-utility conversational answer than redirecting to valuation.
        owner_offer = bool(re.search(
            r"\b(?:someone|somebody|dealer|buyer).{0,20}offered\s+me\b|"
            r"\bi(?:'ve| have)?\s+been\s+offered\b|\bi\s+(?:got|received|have)\s+(?:an?\s+)?offer\b|"
            r"\boffer\s+on\s+my\s+(?:car|vehicle)\b|"
            r"\bbana.{0,25}teklif\b|\bteklif\s+ald[ıi]m\b|\b(?:aracım|aracim|arabam).{0,25}teklif\b|"
            r"\bмне\s+предложил|\bполучил[аи]?\s+предложение",
            low, re.I))
        if owner_offer and (delta.get("offer_price") is not None or re.search(r"[£€$]|\b\d{4,6}\b", low)):
            out["job"] = Job.EVALUATE_SALE.value
            out["action"] = Action.EVALUATE_SALE.value
            out["transition"] = Transition.SWITCH_SUBTASK.value if current_job and current_job != Job.EVALUATE_SALE.value else Transition.CONTINUE.value
            out["awaiting"] = None

        # Explicit requests for ads/listings are a presentation intent, not merely
        # another discovery search. Keep the evidence retrieval in SHOW_LISTINGS so
        # users receive clickable rows rather than being bounced back to model cards.
        explicit_listing_request = bool(re.search(
            r"\b(?:show|see|send|give|find)\b.{0,100}\b(?:listings?|ads?)\b|"
            r"\b(?:listings?|ads?)\b.{0,100}\b(?:show|see|send|give|find)\b|"
            r"\b(?:ilanları|ilanlari|ilanlar|linkleri)\b|\b(?:объявления|ссылки)\b",
            low, re.I))
        if explicit_listing_request and str(out.get("action") or "").upper() in {Action.SEARCH_VEHICLES.value, Action.SHOW_LISTINGS.value, ""}:
            out["action"] = Action.SHOW_LISTINGS.value
            out["job"] = Job.FIND_A_CAR.value

        # A model explicitly chosen earlier is a durable subject. If the latest turn
        # only changes filters (year/km/fuel/colour/transmission/seller/location/
        # budget/variant), do not allow a semantic-plan wobble to reopen the market.
        current_constraints = state.get("constraints") or {}
        selected_models = list(current_constraints.get("models") or [])
        selected_brands = list(current_constraints.get("brands") or [])
        if len(selected_models) == 1 and current_job in {Job.FIND_A_CAR.value, Job.COMPARE_CARS.value}:
            new_model_names=[]
            for spec in list(out.get("objects") or []):
                if not isinstance(spec, Mapping) or str(spec.get("type") or "").upper() != ObjectType.MODEL.value:
                    continue
                payload = spec.get("payload") or {}
                name = _text(payload.get("model"), 120)
                if name:
                    new_model_names.append(name)
            if delta.get("models"):
                new_model_names.extend(str(x) for x in (delta.get("models") or []))
            explicitly_changed_model = any(str(x).casefold() != str(selected_models[0]).casefold() for x in new_model_names)
            asks_for_alternatives = bool(re.search(
                r"\b(?:other|another|different|alternatives?|what else|more models?|other brands?)\b|"
                r"\b(?:başka|baska|diğer|diger).{0,15}(?:model|marka|araç|arac)|"
                r"\b(?:друг(?:ой|ие)|альтернатив).{0,15}(?:модел|мар|автомоб)",
                low, re.I))
            refinement_keys={
                "budget_min","budget_max","min_year","max_year","min_km","max_km",
                "transmission","seller_type","location","exclude_locations","fuel_type",
                "exclude_fuels","colors","category","currency",
            }
            delta_keys={str(k) for k,v in delta.items() if v not in (None,"",[],{})}
            filter_only = bool(delta_keys) and delta_keys.issubset(refinement_keys)
            if not explicitly_changed_model and not asks_for_alternatives:
                clear=[str(x) for x in (out.get("clear_constraints") or [])]
                out["clear_constraints"]=[x for x in clear if x not in {"models","brands"}]
                if filter_only and str(out.get("action") or "").upper() in {Action.SEARCH_VEHICLES.value, Action.SHOW_LISTINGS.value, ""}:
                    if str(out.get("transition") or "").upper() == Transition.START_NEW_GOAL.value:
                        out["transition"] = Transition.REFINE.value
                    # Keep the durable selection explicit even if the planner omitted it.
                    if "models" not in delta:
                        delta["models"] = selected_models
                    if len(selected_brands) == 1 and "brands" not in delta:
                        delta["brands"] = selected_brands
                    out["constraints_delta"] = delta
                    out["job"] = Job.FIND_A_CAR.value
                    # Once the user is looking at actual listings, filter-only
                    # refinements should keep returning actual listings. Do not
                    # regress to model discovery just because the planner chose
                    # SEARCH_VEHICLES for the refinement turn.
                    last_type = str(((state.get("last_result") or {}).get("type") or "")).upper()
                    if last_type == "LISTINGS":
                        out["action"] = Action.SHOW_LISTINGS.value

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

_COLOR_FILTER_ALIASES = {
    # English -> canonical spellings used by the North Cyprus market dataset.
    "white": ["Beyaz"], "pearl white": ["İnci Beyaz"],
    "black": ["Siyah"], "silver": ["Gümüş"],
    "grey": ["Gri"], "gray": ["Gri"],
    "dark grey": ["Koyu Gri", "Füme"], "dark gray": ["Koyu Gri", "Füme"],
    "light grey": ["Açık Gri"], "light gray": ["Açık Gri"],
    "blue": ["Mavi"], "navy": ["Lacivert"], "navy blue": ["Lacivert"],
    "red": ["Kırmızı"], "burgundy": ["Bordo"],
    "green": ["Yeşil"], "yellow": ["Sarı"], "beige": ["Bej"],
    "brown": ["Kahverengi"], "orange": ["Turuncu"],
    # Russian equivalents.
    "белый": ["Beyaz"], "белая": ["Beyaz"], "чёрный": ["Siyah"], "черный": ["Siyah"],
    "серебристый": ["Gümüş"], "серый": ["Gri"], "синий": ["Mavi"],
    "красный": ["Kırmızı"], "зелёный": ["Yeşil"], "зеленый": ["Yeşil"],
    "жёлтый": ["Sarı"], "желтый": ["Sarı"], "бежевый": ["Bej"],
    "коричневый": ["Kahverengi"], "оранжевый": ["Turuncu"],
}

def _canonical_market_colors(values: Any) -> List[str]:
    """Translate user-facing colour names to the source-market vocabulary.

    The source listings are predominantly Turkish while the assistant can run in
    English/Russian. Colour is a hard filter, so localization must happen before
    search rather than merely when rendering a listing. Unknown values pass
    through unchanged so newly observed market colours still work.
    """
    if values in (None, "", [], {}):
        return []
    raw = values if isinstance(values, list) else [values]
    out: List[str] = []
    for value in raw:
        text = _text(value, 80)
        if not text:
            continue
        aliases = _COLOR_FILTER_ALIASES.get(text.casefold(), [text])
        for alias in aliases:
            if alias and alias.casefold() not in {x.casefold() for x in out}:
                out.append(alias)
    return out

def _legacy_filters(state: Mapping[str, Any]) -> Dict[str, Any]:
    c = dict(state.get("constraints") or {})
    out: Dict[str, Any] = {}
    mapping = {
        "budget_max": "budget", "budget_min": "min_budget", "brands": "brands",
        "models": "models", "min_year": "min_year", "max_year": "max_year",
        "min_km": "min_km", "max_km": "max_km", "location": "locations",
        "exclude_locations": "exclude_locations", "transmission": "transmissions", "fuel_type": "fuels", "colors": "colors", "category": "categories", "company": "companies",
    }
    for src, dest in mapping.items():
        val = c.get(src)
        if val in (None, "", [], {}):
            continue
        if dest in {"locations", "transmissions", "fuels", "colors", "categories", "companies"} and not isinstance(val, list):
            val = [val]
        if dest == "colors":
            val = _canonical_market_colors(val)
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
        "colors": f.get("colors"),
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
    """Exclude non-road / powersport inventory by default unless requested.

    The source market table does not carry a reliable vehicle-class column for
    every listing, so the stable model profile catalogue is the first line of
    defence. For an unprofiled model, a brand is also excluded when every known
    profile for that brand is powersport-only. Unknown models from mixed brands
    are retained (to avoid deleting legitimate cars) but profile-fit ranking
    keeps them below verified car families during buyer discovery.
    """
    vehicle_type = str((state.get("constraints") or {}).get("vehicle_type") or "").casefold()
    if vehicle_type in {"motorcycle", "scooter", "atv", "atv_utv", "quad", "utv"}:
        return [dict(x) for x in results]
    lookup = _host(host, "MODEL_PROFILE_LOOKUP", {}) or {}
    non_road={"motorcycle","scooter","atv","atv_utv","quad","utv"}
    road={"car","van","pickup","suv","commercial","minivan","mpv"}
    brand_types: Dict[str,set] = {}
    for k, prof in lookup.items():
        try:
            brand_key=str(k[0]).casefold()
        except Exception:
            continue
        vt=_text((prof or {}).get("VehicleType") or (prof or {}).get("vehicle_type"),40).casefold()
        if vt:
            brand_types.setdefault(brand_key,set()).add(vt)
    out = []
    for row in results:
        brand_key=_text(row.get("brand"),100).casefold()
        key = (brand_key, _text(row.get("model"),120).casefold())
        prof = lookup.get(key) or lookup.get((_text(row.get("brand"),100), _text(row.get("model"),120)))
        vt = _text((prof or {}).get("VehicleType") or (prof or {}).get("vehicle_type"), 40).casefold()
        if vt in non_road:
            continue
        known_types=brand_types.get(brand_key,set())
        if not vt and known_types and any(x in non_road for x in known_types) and not any(x in road for x in known_types):
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
        profile_lookup=_host(host,"MODEL_PROFILE_LOOKUP",{}) or {}
        level_rank={"HIGH":3,"MEDIUM":2,"MODERATE":2,"LOW":1,"UNKNOWN":0,"":0}
        size_pref=str(preferences.get("size") or "").strip().casefold()
        def profile_fit(item):
            prof=profile_lookup.get((str(item.get("brand") or "").casefold(),str(item.get("model") or "").casefold())) or {}
            item["profile_context"]={k:prof.get(k) for k in ("VehicleType","BodyStyle","SizeClass","Economy","Luxury","Comfort","Performance","Practicality","Family","Commute","Confidence") if prof.get(k) not in (None,"")}
            score=0
            size=str(prof.get("SizeClass") or "").upper()
            if size_pref in {"small","küçük","kucuk","маленький","small car","small_car"}:
                score += 5 if size in {"SMALL","MICRO"} else 2 if size=="COMPACT" else 0
            elif size_pref in {"compact","kompakt","компактный"}:
                score += 5 if size=="COMPACT" else 3 if size in {"SMALL","MICRO"} else 0
            elif size_pref in {"large","büyük","buyuk","большой"}:
                score += 5 if size=="LARGE" else 0
            pref_fields={"economy":"Economy","luxury":"Luxury","comfort":"Comfort","performance":"Performance","practicality":"Practicality","family":"Family","commute":"Commute"}
            for pref,field in pref_fields.items():
                if preferences.get(pref) not in (None,False,0,"",[],{}):
                    score += level_rank.get(str(prof.get(field) or "").upper(),0)
            item["profile_fit_score"]=score
            return score
        if not has_named_vehicle and has_decision_priority:
            for item in ranked: profile_fit(item)
            # Soft preferences decide relevance; current-year/budget depth break ties.
            # A model without a trustworthy profile is not silently deleted, but it
            # cannot outrank a verified family/economy/size match on newness alone.
            ranked.sort(key=lambda x:(
                -int(x.get("profile_fit_score") or 0),
                -int(x.get("count") or 0),
                -int(x.get("newest_year") or 0),
                float(x.get("newest_year_starting_price") or 1e18),
            ))
        else:
            # Named-brand refinement should primarily answer what the budget buys.
            for item in ranked: profile_fit(item)
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

    enrich=_host(host,"enrich_model_options_with_buyer_intelligence")
    if callable(enrich):
        try:
            shown=enrich(shown,_legacy_filters(state),_legacy_preferences(state),rows)
        except Exception:
            shown=[dict(x) for x in shown]

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


def _comparable_rows(state: Mapping[str, Any], target: Mapping[str, Any], host: Mapping[str, Any], *, year_window: int = 2, strict_year: bool = False, allow_category_fallback: bool = False) -> List[Dict[str, Any]]:
    brand, model = _text(target.get("brand"),100), _text(target.get("model"),120)
    if not brand or not model:
        return []
    fn = _host(host, "market_search")
    if not callable(fn):
        return []
    try:
        f = _legacy_filters(state)
        target_transmission = target.get("transmission") or (state.get("constraints") or {}).get("transmission")
        target_fuel = target.get("fuel_type") or target.get("fuel") or (state.get("constraints") or {}).get("fuel_type")
        kwargs={
            "budget":f.get("budget"), "min_budget":f.get("min_budget"),
            "brands":[brand], "models":[model],
            # Pull the model family first; variant fallback must be possible when
            # an exact trim/category has no current observations.
            "categories":None,
            "locations":f.get("locations"),
            "transmissions":[target_transmission] if target_transmission else f.get("transmissions"),
            "min_year":f.get("min_year"), "max_year":f.get("max_year"),
            "min_km":f.get("min_km"), "max_km":f.get("max_km"),
            "limit":5000, "max_limit":5000, "analysis_mode":True,
        }
        if target_fuel or f.get("fuels"):
            kwargs["fuels"]=[target_fuel] if target_fuel else f.get("fuels")
        if f.get("colors"):
            kwargs["colors"]=f.get("colors")
        result = fn(**kwargs)
        rows = [dict(x) for x in (result.get("results") or []) if _text(x.get("brand"),100).casefold()==brand.casefold() and _text(x.get("model"),120).casefold()==model.casefold()]
    except Exception:
        return []

    category = target.get("category") or (state.get("constraints") or {}).get("category")
    if category:
        exact_category=[x for x in rows if str(x.get("category") or "").strip().casefold()==str(category).strip().casefold()]
        if exact_category:
            rows=exact_category
        elif not allow_category_fallback:
            rows=[]

    year = _int(target.get("year"))
    if year is not None:
        exact = [x for x in rows if _int(x.get("year")) == year]
        if strict_year:
            rows = exact
        elif len(exact) >= 3:
            rows = exact
        else:
            nearby = [x for x in rows if _int(x.get("year")) is not None and abs((_int(x.get("year")) or 0)-year) <= year_window]
            rows = nearby or exact

    target_km = _int(target.get("km") or target.get("mileage"))
    km_rows = [x for x in rows if _int(x.get("km")) is not None]
    if target_km is not None and len(km_rows) >= 5:
        km_rows.sort(key=lambda x: abs((_int(x.get("km")) or target_km) - target_km))
        rows = km_rows[:min(20, len(km_rows))]

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
    if mode == "purchase_evaluation":
        price_to_evaluate = asking
    elif mode in {"vehicle_valuation", "trade_in_evaluation"}:
        price_to_evaluate = None
    else:
        price_to_evaluate = offer if offer is not None else asking
        # A seller asking us to determine the price has not omitted a required
        # input: that IS the output. Treat a sale evaluation with no supplied
        # price as a valuation rather than entering an asking-price loop.
        if price_to_evaluate is None:
            mode = "vehicle_valuation"
    requested_category = subject.get("category") or c.get("category")
    allow_category_fallback = mode in {"sale_evaluation", "vehicle_valuation", "trade_in_evaluation"}
    rows = _comparable_rows(state, subject, host, allow_category_fallback=allow_category_fallback)
    categories = sorted({str(x.get("category") or "").strip() for x in rows if str(x.get("category") or "").strip()})
    category_fallback_used = bool(requested_category and rows and not any(str(x.get("category") or "").strip().casefold()==str(requested_category).strip().casefold() for x in rows))
    if not requested_category and len(categories) > 1:
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
        "p75_asking_price":p75, "estimated_market_value":median if mode in {"vehicle_valuation", "trade_in_evaluation"} else None,
        "recommended_asking_price":median if mode == "vehicle_valuation" else None,
        "position":pos, "vs_median_pct":round(vs_med*100,1) if vs_med is not None else None,
        "comparables":closest,
        "category_fallback_used":category_fallback_used,
        "requested_category":requested_category,
        "basis":"current asking prices, not confirmed transaction prices",
    }


def _comparison_market_signal(host: Mapping[str, Any], brand: str, model: str, year: Optional[int]) -> Dict[str, Any]:
    df=_host(host,"buyer_model_df")
    if df is None or getattr(df,"empty",True):
        return {}
    try:
        work=df[(df["Brand"].fillna("").astype(str).str.casefold()==brand.casefold()) & (df["Model"].fillna("").astype(str).str.casefold()==model.casefold())].copy()
        if work.empty:
            return {}
        if year is not None and "Year" in work.columns:
            exact=work[work["Year"]==year]
            if not exact.empty:
                work=exact
        if "Year" in work.columns:
            work=work.sort_values("Year",ascending=False)
        row=work.iloc[0]
        eligible=_int(row.get("Buyer_Exit60EligibleListings")) or 0
        rate=_finite(row.get("Buyer_ObservedExitWithin60DaysRate"))
        if rate is not None and rate>1.5:
            rate=rate/100.0
        confidence=str(row.get("Buyer_LiquidityEvidenceConfidence") or "").upper()
        if rate is None or eligible < 10 or confidence in {"", "INSUFFICIENT"}:
            liquidity="LIMITED_EVIDENCE"
        elif rate >= 0.75:
            liquidity="STRONGER_OBSERVED_TURNOVER"
        elif rate >= 0.50:
            liquidity="MODERATE_OBSERVED_TURNOVER"
        else:
            liquidity="WEAKER_OBSERVED_TURNOVER"
        reduction=_finite(row.get("Buyer_PriceReductionRate"))
        if reduction is not None and reduction>1.5:
            reduction=reduction/100.0
        if reduction is None:
            pressure="LIMITED_EVIDENCE"
        elif reduction >= 0.35:
            pressure="HIGHER_PRICE_REDUCTION_PRESSURE"
        elif reduction >= 0.18:
            pressure="MODERATE_PRICE_REDUCTION_PRESSURE"
        else:
            pressure="LOWER_PRICE_REDUCTION_PRESSURE"
        return {
            "liquidity_signal":liquidity,
            "liquidity_confidence":confidence,
            "liquidity_eligible":eligible,
            "price_reduction_rate":round(reduction,4) if reduction is not None else None,
            "median_reduction_pct_when_reduced":_finite(row.get("Buyer_MedianReductionPctAmongReduced")),
            "price_pressure_signal":pressure,
            "price_pressure_confidence":str(row.get("Buyer_PricePressureEvidenceConfidence") or "").upper(),
        }
    except Exception:
        return {}


def _compare_models(state: Mapping[str, Any], target_ids: Sequence[str], host: Mapping[str, Any]) -> Dict[str, Any]:
    objects = state.get("objects") or {}
    ids = list(target_ids or ((state.get("focus") or {}).get("object_ids") or []))[:4]
    comps=[]
    for oid in ids:
        obj=objects.get(oid) or {}
        payload={k:v for k,v in obj.items() if k not in {"id","type"}}
        explicit_year=_int(payload.get("year"))
        rows=_comparable_rows(state,payload,host,strict_year=explicit_year is not None)
        if not rows:
            continue
        prices=[float(x["price"]) for x in rows if _finite(x.get("price")) is not None]
        years=[_int(x.get("year")) for x in rows if _int(x.get("year")) is not None]
        if not years or not prices:
            continue
        anchor_year=explicit_year if explicit_year is not None else max(years)
        anchor_rows=[x for x in rows if _int(x.get("year"))==anchor_year]
        anchor_prices=[float(x["price"]) for x in anchor_rows if _finite(x.get("price")) is not None]
        anchor_kms=[_int(x.get("km")) for x in anchor_rows if _int(x.get("km")) is not None]
        fuels=sorted({_text(x.get("fuel") or x.get("fuel_type"),80) for x in anchor_rows if _text(x.get("fuel") or x.get("fuel_type"),80)})
        colors=sorted({_text(x.get("color"),80) for x in anchor_rows if _text(x.get("color"),80)})
        transmissions=sorted({_text(x.get("transmission"),80) for x in anchor_rows if _text(x.get("transmission"),80)})
        best_anchor=sorted(anchor_rows,key=lambda x:_finite(x.get("price")) or 1e18)[0] if anchor_rows else None
        year_summaries=[]
        for y in sorted(set(years),reverse=True)[:6]:
            yr=[x for x in rows if _int(x.get("year"))==y]
            yp=[float(x["price"]) for x in yr if _finite(x.get("price")) is not None]
            yk=[_int(x.get("km")) for x in yr if _int(x.get("km")) is not None]
            if not yp:
                continue
            year_summaries.append({
                "year":y,"count":len(yr),"starting_price":min(yp),"median_price":statistics.median(yp),"highest_price":max(yp),
                "median_km":statistics.median(yk) if yk else None,
                "fuels":sorted({_text(x.get("fuel") or x.get("fuel_type"),80) for x in yr if _text(x.get("fuel") or x.get("fuel_type"),80)}),
                "colors":sorted({_text(x.get("color"),80) for x in yr if _text(x.get("color"),80)}),
            })
        signal=_comparison_market_signal(host,_text(payload.get("brand"),100),_text(payload.get("model"),120),anchor_year)
        profile_lookup=_host(host,"MODEL_PROFILE_LOOKUP",{}) or {}
        profile=profile_lookup.get((_text(payload.get("brand"),100).casefold(),_text(payload.get("model"),120).casefold())) or {}
        profile_context={k:profile.get(k) for k in ("VehicleType","BodyStyle","SizeClass","Economy","Luxury","Comfort","Performance","Practicality","Family","Commute","Confidence") if profile.get(k) not in (None,"")}
        comps.append({
            "object_id":oid,"brand":payload.get("brand"),"model":payload.get("model"),"year":explicit_year,
            "count":len(rows),"newest_year":anchor_year,"newest_year_count":len(anchor_rows),
            "newest_year_starting_price":min(anchor_prices) if anchor_prices else None,
            "newest_year_median_price":statistics.median(anchor_prices) if anchor_prices else None,
            "newest_year_highest_price":max(anchor_prices) if anchor_prices else None,
            "median_km":statistics.median(anchor_kms) if anchor_kms else None,
            "fuels":fuels,"colors":colors[:8],"transmissions":transmissions,
            "year_summaries":year_summaries,
            "profile_context":profile_context,
            "representative_listing":best_anchor,
            **signal,
        })

    # Comparing one selected model means the buyer is usually deciding between
    # years of that model. Expand the model's real year cohorts so 2023 vs 2024
    # vs 2025 is a genuine comparison instead of collapsing back to "newest".
    if len(comps)==1 and comps[0].get("year") is None and len(comps[0].get("year_summaries") or [])>=2:
        base=comps[0]; expanded=[]
        for summary in (base.get("year_summaries") or [])[:4]:
            y=_int(summary.get("year"))
            signal=_comparison_market_signal(host,_text(base.get("brand"),100),_text(base.get("model"),120),y)
            expanded.append({
                "object_id":base.get("object_id"),"brand":base.get("brand"),"model":base.get("model"),"year":y,
                "count":int(summary.get("count") or 0),"newest_year":y,"newest_year_count":int(summary.get("count") or 0),
                "newest_year_starting_price":_finite(summary.get("starting_price")),
                "newest_year_median_price":_finite(summary.get("median_price")),
                "newest_year_highest_price":_finite(summary.get("highest_price")),
                "median_km":_finite(summary.get("median_km")),
                "fuels":list(summary.get("fuels") or []),"colors":list(summary.get("colors") or [])[:8],
                "transmissions":base.get("transmissions") or [],"year_summaries":[summary],
                "profile_context":base.get("profile_context") or {},
                **signal,
            })
        comps=expanded
    return {"kind":"comparison","vehicles":comps}

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

    def anchor_quality(row):
        reasons=[]
        selection=str(row.get("ComparableSelection") or "").upper()
        category=_text(row.get("CategoryDetail"),160)
        comp_n=_int(row.get("ComparableListings")) or 0
        comp_conf=str(row.get("ComparableEvidenceConfidence") or "").upper()
        km=_finite(row.get("KM")); comp_km=_finite(row.get("ComparableMedianKM"))
        pct=abs(_finite(row.get("PriceVsMedianPct")) or 0.0)
        # A model-level fallback is useful for detecting that a price deserves
        # review, but not safe enough to set a concrete target when the listing
        # has a specific engine/trim category (e.g. E 500, 43 AMG, 530d).
        if category and selection and not selection.startswith("CATEGORY"):
            reasons.append("variant_not_matched")
        if km is not None and comp_km is not None and comp_km>0:
            ratio=km/comp_km
            if abs(km-comp_km)>=25000 and (ratio<0.5 or ratio>2.0):
                reasons.append("mileage_mismatch")
        if comp_n<3:
            reasons.append("too_few_comparables")
        hard={"variant_not_matched","mileage_mismatch","too_few_comparables"}
        if any(x in hard for x in reasons):
            return "WITHHELD",reasons
        caution=[]
        if comp_n<5: caution.append("thin_comparable_set")
        if comp_conf in {"LOW","INSUFFICIENT",""}: caution.append("lower_comparable_confidence")
        if selection and "PLUS_MINUS" in selection: caution.append("year_range_fallback")
        if pct>=0.5: caution.append("large_price_gap")
        return ("CAUTION" if caution else "READY"),caution

    def score(row):
        band=str(row.get("PricePositionBand") or "").upper()
        attention=str(row.get("AttentionLevel") or "").upper()
        pct=_finite(row.get("PriceVsMedianPct")) or 0
        return (2 if "HIGH" in band else 0)+(2 if "HIGH" in attention else 1 if "ATTENTION" in attention else 0)+max(0,pct)

    records=[]
    for raw in _df_records(rows,500):
        r=dict(raw)
        status,reasons=anchor_quality(r)
        r["reprice_anchor_status"]=status
        r["reprice_anchor_reasons"]=reasons
        p25=_finite(r.get("ComparableP25Price")); med=_finite(r.get("ComparableMedianPrice")); p75=_finite(r.get("ComparableP75Price"))
        if status=="WITHHELD":
            r["quick_sale_target"]=None; r["competitive_target"]=None; r["premium_target"]=None
        else:
            r["quick_sale_target"]=p25; r["competitive_target"]=med; r["premium_target"]=p75
        records.append(r)
    status_rank={"READY":2,"CAUTION":1,"WITHHELD":0}
    records.sort(key=lambda r:(status_rank.get(r.get("reprice_anchor_status"),0),score(r)),reverse=True)
    return {"kind":"business_pricing","company":company,"vehicles":records[:12],"status":"ok"}


def _business_aging_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    company, rows = _business_company_rows(state, host)
    if rows is None or getattr(rows,"empty",True):
        return {"kind":"business_aging", "company":company, "vehicles":[], "status":"company_or_stock_unavailable"}
    rows=_filter_business_rows_to_targets(state,rows,target_ids)
    records=_df_records(rows,500)
    age_rank={"VERY_AGED":5,"AGED":4,"ABOVE_TYPICAL":3,"NORMAL":2,"FRESH":1,"INSUFFICIENT_EVIDENCE":0,"":0}
    confidence_rank={"HIGH":3,"MEDIUM":2,"LOW":1,"INSUFFICIENT":0,"":0}
    # "Sitting too long" is a relative-market question. Rank a supported age
    # classification ahead of raw day count so an evidence-poor 260-day listing
    # does not masquerade as a proven slow seller for its segment.
    records.sort(key=lambda x:(
        age_rank.get(str(x.get("StockAgeBand") or "").upper(),0),
        confidence_rank.get(str(x.get("LiquidityEvidenceConfidence") or "").upper(),0),
        _finite(x.get("StockAgeDays")) or -1,
        _finite(x.get("PriceVsMedianPct")) or -999,
    ), reverse=True)
    return {"kind":"business_aging","company":company,"vehicles":records[:12],"status":"ok"}


def _business_acquire_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    df=_host(host,"business_market_df")
    if df is None or getattr(df,"empty",True):
        return {"kind":"business_acquisition","options":[],"status":"unavailable"}
    work=df.copy()
    c=state.get("constraints") or {}
    prefs=state.get("preferences") or {}
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

        # Business acquisition should default to normal road vehicles unless the
        # dealer explicitly asks for another class. The profile catalogue is the
        # canonical physical-class layer; the raw market VehicleType remains a
        # fallback for models without a profile.
        profile_lookup=_host(host,"MODEL_PROFILE_LOOKUP",{}) or {}
        requested_type=str(c.get("vehicle_type") or "").strip().casefold()
        if not requested_type and not work.empty:
            def road_vehicle(row):
                key=(str(row.get("Brand") or "").casefold(),str(row.get("Model") or "").casefold())
                prof=profile_lookup.get(key) or {}
                pvt=str(prof.get("VehicleType") or "").upper()
                if pvt in {"MOTORCYCLE","SCOOTER","ATV_UTV"}: return False
                rawvt=str(row.get("VehicleType") or "").casefold()
                if any(tok in rawvt for tok in ("motosiklet","motorcycle","scooter","atv","utv")): return False
                return True
            mask=work.apply(lambda r: road_vehicle(r),axis=1)
            if mask.any(): work=work[mask]

        # Transmission/fuel are not dimensions in the aggregate Business market
        # table. Verify them against live current listings so a hard request such
        # as "automatic only" cannot be silently ignored by the recommendation.
        live_search=_host(host,"market_search")
        hard_transmission=c.get("transmission")
        hard_fuel=c.get("fuel_type")
        verified_keys=None
        verified_counts={}
        if callable(live_search) and (hard_transmission or hard_fuel):
            kwargs={
                "budget":c.get("budget_max"),"min_budget":c.get("budget_min"),
                "transmissions":[hard_transmission] if hard_transmission else None,
                "fuels":[hard_fuel] if hard_fuel else None,
                "limit":5000,"max_limit":5000,"analysis_mode":True,
            }
            if c.get("brands"): kwargs["brands"]=c.get("brands")
            live=_checked_search(live_search(**kwargs)).get("results") or []
            for r in live:
                key=(str(r.get("brand") or "").casefold(),str(r.get("model") or "").casefold(),_int(r.get("year")))
                verified_counts[key]=verified_counts.get(key,0)+1
            verified_keys=set(verified_counts)
            if verified_keys:
                keys=work.apply(lambda r:(str(r.get("Brand") or "").casefold(),str(r.get("Model") or "").casefold(),_int(r.get("Year"))),axis=1)
                work=work[keys.isin(verified_keys)]
            else:
                work=work.iloc[0:0]

        # Prefer model/year granularity where available.
        if "BusinessGranularity" in work:
            preferred=work[work["BusinessGranularity"].fillna("").astype(str).str.upper().isin({"MODEL_YEAR","CATEGORY_YEAR","CATEGORY_NEAR_YEAR"})]
            if not preferred.empty: work=preferred

        records=_df_records(work,500)
        size_pref=str(prefs.get("size") or "").strip().casefold()
        def size_rank(rec):
            if not size_pref: return 1
            prof=profile_lookup.get((str(rec.get("Brand") or "").casefold(),str(rec.get("Model") or "").casefold())) or {}
            size=str(prof.get("SizeClass") or "").upper()
            if size_pref in {"small","küçük","kucuk","маленький","compact small"}:
                if size in {"SMALL","MICRO"}: return 3
                if size=="COMPACT": return 2
                return 0
            if size_pref in {"compact","kompakt","компактный"}:
                if size=="COMPACT": return 3
                if size in {"SMALL","MICRO"}: return 2
                return 0
            if size_pref in {"large","büyük","buyuk","большой"}:
                return 3 if size=="LARGE" else 0
            return 1
        evidence_rank={"HIGH":3,"MEDIUM":2,"LOW":1,"INSUFFICIENT":0,"":0}
        signal_rank={"VERY_STRONG":4,"STRONG":3,"MODERATE":2,"WEAK":1,"CAUTION":0,"INSUFFICIENT_EVIDENCE":0,"":0}
        benchmark_rank={"MODEL_YEAR":3,"CATEGORY_YEAR":3,"CATEGORY_NEAR_YEAR":2,"MODEL_ALL_YEARS":1,"CATEGORY_ALL_YEARS":1,"":0}
        def market_rank(rec):
            key=(str(rec.get("Brand") or "").casefold(),str(rec.get("Model") or "").casefold(),_int(rec.get("Year")))
            verified=verified_counts.get(key,0) if verified_counts else 0
            # A year-specific stocking recommendation should prefer evidence that
            # actually belongs to that year (or a near-year fallback). A HIGH
            # sample count inherited from MODEL_ALL_YEARS is useful context, but
            # it must not outrank true MODEL_YEAR evidence solely on volume.
            source_quality=min(
                benchmark_rank.get(str(rec.get("HistoricalBenchmarkSourceLiquidity") or "").upper(),0),
                benchmark_rank.get(str(rec.get("HistoricalBenchmarkSourcePricePressure") or "").upper(),0),
            )
            return (
                size_rank(rec),
                source_quality,
                evidence_rank.get(str(rec.get("EvidenceQuality") or "").upper(),0),
                signal_rank.get(str(rec.get("AcquisitionSignal") or "").upper(),0),
                _finite(rec.get("ConfidenceAdjustedOpportunityIndex")) or -1.0,
                min(verified,10),
                _finite(rec.get("OpportunityPercentile")) or -1.0,
            )
        records.sort(key=market_rank,reverse=True)
    except Exception:
        records=[]
    # Deduplicate model/year and avoid opaque scores in user-facing packet.
    out=[]; seen=set()
    for r in records:
        full_key=(str(r.get("Brand") or "").casefold(),str(r.get("Model") or "").casefold(),_int(r.get("Year")))
        # A broad "what should I stock?" answer should map the strongest model
        # families, not fill the list with six years of the same model. Year-level
        # evidence remains attached to the representative candidate and can be
        # drilled into on the next turn.
        dedupe_key=full_key if (specific_brand and specific_model) else full_key[:2]
        if dedupe_key in seen: continue
        seen.add(dedupe_key)
        prof=(_host(host,"MODEL_PROFILE_LOOKUP",{}) or {}).get((full_key[0],full_key[1])) or {}
        r=dict(r)
        if prof:
            r["profile_size_class"]=prof.get("SizeClass")
            r["profile_vehicle_type"]=prof.get("VehicleType")
        if verified_counts:
            r["verified_current_matches"]=verified_counts.get(full_key,0)
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

def _business_trade_in_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    base=_evaluate_price_subject(state,target_ids,host,"trade_in_evaluation")
    if base.get("kind")=="clarification":
        return base
    out=dict(base)
    out["kind"]="business_trade_in"
    c=state.get("constraints") or {}
    median=_finite(out.get("median_asking_price"))
    margin=_finite(c.get("target_margin_pct"))
    prep=_finite(c.get("prep_allowance")) or 0.0
    if margin is not None and median is not None:
        margin=max(0.0,min(80.0,margin))
        out["target_margin_pct"]=margin
        out["prep_allowance"]=prep
        out["suggested_offer_ceiling"]=max(0.0,median*(1.0-margin/100.0)-prep)
    else:
        out["target_margin_pct"]=margin
        out["prep_allowance"]=prep if prep else None
        out["suggested_offer_ceiling"]=None
    return out


def _business_ad_candidate_tool(state: Mapping[str, Any], host: Mapping[str, Any], target_ids: Sequence[str] = ()) -> Dict[str, Any]:
    company, rows=_business_company_rows(state,host)
    market=_host(host,"business_market_df")
    if rows is None or getattr(rows,"empty",True):
        return {"kind":"business_ad_candidate","company":company,"vehicles":[],"status":"company_or_stock_unavailable"}
    rows=_filter_business_rows_to_targets(state,rows,target_ids)
    if market is None or getattr(market,"empty",True):
        return {"kind":"business_ad_candidate","company":company,"vehicles":[],"status":"market_unavailable"}
    evidence_rank={"HIGH":3,"MEDIUM":2,"LOW":1,"INSUFFICIENT":0,"":0}
    out=[]
    for stock in _df_records(rows,500):
        brand=_text(stock.get("Brand"),100); model=_text(stock.get("Model"),120); year=_int(stock.get("Year")); category=_text(stock.get("CategoryDetail"),160)
        if not brand or not model:
            continue
        try:
            m=market[(market["Brand"].fillna("").astype(str).str.casefold()==brand.casefold()) & (market["Model"].fillna("").astype(str).str.casefold()==model.casefold())].copy()
            if year is not None and "Year" in m.columns:
                exact=m[m["Year"]==year]
                if not exact.empty: m=exact
            if category and "CategoryDetail" in m.columns:
                exact_cat=m[m["CategoryDetail"].fillna("").astype(str).str.casefold()==category.casefold()]
                if not exact_cat.empty: m=exact_cat
            if m.empty:
                continue
            if "OpportunityPercentile" in m.columns:
                m=m.sort_values("OpportunityPercentile",ascending=False,na_position="last")
            mr=m.iloc[0].to_dict()
        except Exception:
            continue
        quality=str(mr.get("EvidenceQuality") or "").upper()
        opportunity=_finite(mr.get("OpportunityPercentile"))
        price_vs=_finite(stock.get("PriceVsMedianPct"))
        age=_finite(stock.get("StockAgeDays")) or 0.0
        # Higher evidence/opportunity is better. A materially high asking price is
        # a reason to fix pricing before spending on promotion. Older suitable stock
        # gets a small tie-break benefit because ad spend can help unlock it.
        price_penalty=max(0.0,price_vs or 0.0)
        rank=(evidence_rank.get(quality,0), opportunity if opportunity is not None else -1.0, -price_penalty, age)
        rate=_finite(mr.get("ObservedExitWithin60DaysRate"))
        if rate is not None and rate>1.5: rate/=100.0
        eligible=_int(mr.get("Exit60EligibleListings")) or _int(mr.get("HistoricalDistinctListings")) or 0
        if rate is None or eligible<10 or quality in {"","INSUFFICIENT"}: activity="LIMITED_EVIDENCE"
        elif rate>=0.75: activity="STRONGER_OBSERVED_TURNOVER"
        elif rate>=0.50: activity="MODERATE_OBSERVED_TURNOVER"
        else: activity="WEAKER_OBSERVED_TURNOVER"
        out.append({
            "Link":stock.get("Link"),"Company":stock.get("Company"),"Brand":brand,"Model":model,"Year":year,"CategoryDetail":category,
            "CurrentAskingPrice":_finite(stock.get("CurrentAskingPrice")),"KM":_int(stock.get("KM")),"StockAgeDays":_int(stock.get("StockAgeDays")),
            "PricePositionBand":stock.get("PricePositionBand"),"PriceVsMedianPct":price_vs,
            "MarketActivitySignal":activity,"EvidenceQuality":quality,"CurrentListings":_int(mr.get("CurrentListings")),
            "PriceReductionRate":_finite(mr.get("PriceReductionRate")),"AcquisitionSignal":mr.get("AcquisitionSignal"),
            "_rank":rank,
        })
    out.sort(key=lambda x:x.get("_rank",(0,-1,0,0)),reverse=True)
    for x in out: x.pop("_rank",None)
    return {"kind":"business_ad_candidate","company":company,"vehicles":out[:8],"status":"ok" if out else "insufficient"}



def _fuel_from_listing_link(value: Any) -> str:
    text=_text(value,500).casefold()
    patterns=[
        (r"(?:^|-)mild-hibrit-benzin(?:-|$)", "Mild Hybrid Petrol"),
        (r"(?:^|-)mild-hibrit-dizel(?:-|$)", "Mild Hybrid Diesel"),
        (r"(?:^|-)plug-in-hibrit(?:-|$)", "Plug-in Hybrid"),
        (r"(?:^|-)elektrik(?:-|$)", "Electric"),
        (r"(?:^|-)hibrit(?:-|$)", "Hybrid"),
        (r"(?:^|-)dizel(?:-|$)", "Diesel"),
        (r"(?:^|-)benzin(?:-|$)", "Petrol"),
    ]
    for pattern,label in patterns:
        if re.search(pattern,text,re.IGNORECASE):
            return label
    return ""


def _brand_model_from_category_text(value: Any) -> Tuple[str,str]:
    parts=[x.strip() for x in _text(value,500).split('/') if x.strip()]
    if len(parts)>=3:
        return parts[1],parts[2]
    return "",""


def _price_band(value: Any) -> Optional[str]:
    price=_finite(value)
    if price is None:
        return None
    if price < 15000: return "UNDER_15K"
    if price < 25000: return "15K_25K"
    if price < 40000: return "25K_40K"
    return "40K_PLUS"


def _segment_dimensions(record: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str,str]:
    brand=_text(record.get("Brand") or record.get("brand"),100)
    model=_text(record.get("Model") or record.get("model"),120)
    if not brand or not model:
        b,m=_brand_model_from_category_text(record.get("Category") or record.get("category") or record.get("CategoryDetail"))
        brand=brand or b
        model=model or m
    profile=(_host(host,"MODEL_PROFILE_LOOKUP",{}) or {}).get((brand.casefold(),model.casefold()),{}) if brand and model else {}
    price=record.get("EventPrice")
    if price in (None,""): price=record.get("new_price")
    if price in (None,""): price=record.get("CurrentAskingPrice")
    if price in (None,""): price=record.get("CurrentStartingPrice")
    fuel=_text(record.get("Fuel") or record.get("fuel"),80)
    if not fuel:
        fuel=_fuel_from_listing_link(record.get("Link") or record.get("link") or record.get("listing_id"))
    dims={}
    pb=_price_band(price)
    if pb: dims["PRICE_BAND"]=pb
    body=_text(profile.get("BodyStyle"),80)
    size=_text(profile.get("SizeClass"),80)
    if body: dims["BODY_STYLE"]=body
    if size: dims["SIZE_CLASS"]=size
    if fuel: dims["FUEL"]=fuel
    if brand and model: dims["MODEL"]=f"{brand} {model}"
    combo=[x for x in [pb,size,fuel] if x]
    if len(combo)>=2: dims["COMBO"]=" | ".join(combo)
    return dims


def _business_segment_analysis(state: Mapping[str, Any], host: Mapping[str, Any], company_events: Mapping[str,List[Mapping[str,Any]]], period_start: str, period_end: str) -> List[Dict[str,Any]]:
    company=_text((state.get("constraints") or {}).get("company"),160)
    if not company:
        return []
    try:
        import pandas as pd
        counters={}
        def bucket(dim,value):
            key=(dim,str(value))
            if key not in counters:
                counters[key]={"dimension":dim,"segment":str(value),"company_new":0,"company_exits":0,"wider_new":0,"wider_exits":0,"company_stock":0,"wider_stock":0,"opportunities":[]}
            return counters[key]
        def add_records(records, field, prefix):
            for rec in records or []:
                for dim,value in _segment_dimensions(rec,host).items():
                    bucket(dim,value)[f"{prefix}_{field}"]+=1

        add_records(company_events.get("new"),"new","company")
        add_records(company_events.get("exits"),"exits","company")

        activity=_host(host,"business_activity_df")
        if activity is not None and not getattr(activity,"empty",True):
            work=activity.copy(); work["Date"]=pd.to_datetime(work["Date"],errors="coerce")
            a=pd.Timestamp(period_start); b=pd.Timestamp(period_end)
            wider=work[(work["Date"]>=a)&(work["Date"]<=b)&(work["Company"].fillna("").astype(str).str.casefold()!=company.casefold())]
            def parse(value):
                try:
                    x=json.loads(str(value or "[]")); return x if isinstance(x,list) else []
                except Exception: return []
            for row in wider.to_dict("records"):
                add_records(parse(row.get("NewListingsJson")),"new","wider")
                add_records(parse(row.get("ObservedExitListingsJson")),"exits","wider")

        stock=_host(host,"business_stock_df")
        if stock is not None and not getattr(stock,"empty",True):
            for rec in _df_records(stock,20000):
                prefix="company" if _text(rec.get("Company"),160).casefold()==company.casefold() else "wider"
                for dim,value in _segment_dimensions(rec,host).items():
                    bucket(dim,value)[f"{prefix}_stock"]+=1

        market=_host(host,"business_market_df")
        if market is not None and not getattr(market,"empty",True):
            for rec in _df_records(market,20000):
                opp=_finite(rec.get("OpportunityPercentile"))
                if opp is None: continue
                for dim,value in _segment_dimensions(rec,host).items():
                    bucket(dim,value)["opportunities"].append(opp)

        out=[]
        for row in counters.values():
            total_stock=row["company_stock"]+row["wider_stock"]
            total_exits=row["company_exits"]+row["wider_exits"]
            stock_share=(row["company_stock"]/total_stock) if total_stock>0 else None
            exit_share=(row["company_exits"]/total_exits) if total_exits>0 else None
            relative=(exit_share/stock_share) if stock_share and exit_share is not None else None
            evidence_events=total_exits+row["company_new"]+row["wider_new"]
            company_activity_events=row["company_exits"]+row["company_new"]
            # A dealership-specific "strongest segment" claim needs evidence from
            # the dealership itself, not merely a strong wider-market percentile.
            # One stock item + one exit is an early signal, never a proven strength.
            if row["company_stock"]>=3 and company_activity_events>=4:
                dealer_evidence="STRONG"
            elif row["company_stock"]>=2 and company_activity_events>=2:
                dealer_evidence="USABLE"
            else:
                dealer_evidence="EARLY"
            if relative is None or total_exits<3 or row["company_stock"]<1:
                performance="LIMITED_EVIDENCE"
            elif relative>=1.25 and row["company_exits"]>=2 and dealer_evidence in {"STRONG","USABLE"}:
                performance="STRONG_RELATIVE_ACTIVITY"
            elif relative<=0.75 and row["company_stock"]>=2 and company_activity_events>=2:
                performance="WEAK_RELATIVE_ACTIVITY"
            else:
                performance="BALANCED_RELATIVE_ACTIVITY"
            opp=(sum(row["opportunities"])/len(row["opportunities"])) if row["opportunities"] else None
            if performance=="STRONG_RELATIVE_ACTIVITY" and dealer_evidence in {"STRONG","USABLE"} and (opp is None or opp>=35):
                bet="FAVORABLE"
            elif opp is not None and opp>=70 and performance=="BALANCED_RELATIVE_ACTIVITY" and dealer_evidence in {"STRONG","USABLE"}:
                bet="FAVORABLE"
            elif (opp is not None and opp<35 and dealer_evidence!="EARLY") or performance=="WEAK_RELATIVE_ACTIVITY":
                bet="REVIEW"
            else:
                bet="NEUTRAL"
            out.append({
                "dimension":row["dimension"],"segment":row["segment"],
                "company_current_stock":row["company_stock"],"wider_current_stock":row["wider_stock"],
                "company_newly_observed":row["company_new"],"company_observed_exits":row["company_exits"],
                "wider_newly_observed":row["wider_new"],"wider_observed_exits":row["wider_exits"],
                "relative_activity_index":round(relative,2) if relative is not None else None,
                "performance_signal":performance,"market_opportunity_percentile":round(opp,1) if opp is not None else None,
                "bet_signal":bet,"evidence_events":evidence_events,
                "company_activity_events":company_activity_events,"dealer_evidence":dealer_evidence,
                "exit_definition":"Observed listing exits, not confirmed sales",
            })
        priority={"FAVORABLE":0,"NEUTRAL":1,"REVIEW":2}
        perf={"STRONG_RELATIVE_ACTIVITY":0,"BALANCED_RELATIVE_ACTIVITY":1,"LIMITED_EVIDENCE":2,"WEAK_RELATIVE_ACTIVITY":3}
        out.sort(key=lambda r:(priority.get(r["bet_signal"],9),perf.get(r["performance_signal"],9),-(r.get("evidence_events") or 0),-(r.get("company_current_stock") or 0)))
        return out[:15]
    except Exception:
        return []


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
            segment_analysis=_business_segment_analysis(
                state,host,events,
                summary.get("period_start") or str(daily_records[0].get("Date")),
                summary.get("period_end") or str(daily_records[-1].get("Date")),
            ) if daily_records else []
            return {"kind":"business_period","company":company,"status":"ok","summary":summary,"events":events,"daily":daily_records,"segments":segment_analysis}
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



def _business_relative_market_context(state: Mapping[str, Any], host: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Compare a dealership's observed listing activity with other galleries.

    This is intentionally about advert/listing activity, not confirmed sales.
    Rates are event counts per observed stock-day so a large dealership is not
    automatically treated as more active merely because it carries more cars.
    """
    if str(state.get("audience") or "").upper() != Audience.BUSINESS.value:
        return None
    company=_text((state.get("constraints") or {}).get("company"),160)
    activity=_host(host,"business_activity_df")
    if not company or activity is None or getattr(activity,"empty",True):
        return None
    try:
        import pandas as pd
        work=activity.copy()
        work["Date"]=pd.to_datetime(work["Date"],errors="coerce")
        work=work[work["Date"].notna()].copy()
        if work.empty:
            return None
        c=state.get("constraints") or {}
        end=pd.Timestamp(c.get("period_end")) if c.get("period_end") else work["Date"].max()
        start=pd.Timestamp(c.get("period_start")) if c.get("period_start") else end-pd.Timedelta(days=29)
        if start>end:
            start,end=end,start
        days=max(1,int((end-start).days)+1)
        prev_end=start-pd.Timedelta(days=1)
        prev_start=prev_end-pd.Timedelta(days=days-1)

        def select(frame,a,b):
            return frame[(frame["Date"]>=a)&(frame["Date"]<=b)].copy()

        def metrics(frame):
            if frame.empty:
                return {"days_with_data":0,"observed_stock_days":0.0,"new_listings":0,"observed_exits":0,"price_reductions":0,"exit_activity_rate":None,"new_listing_activity_rate":None,"price_reduction_activity_rate":None}
            stock=pd.to_numeric(frame.get("OpeningObservedStockCount"),errors="coerce").fillna(0).clip(lower=0)
            denom=float(stock.sum())
            def sm(col):
                return float(pd.to_numeric(frame.get(col),errors="coerce").fillna(0).clip(lower=0).sum())
            exits=sm("ObservedMarketExits"); new=sm("NewlyObservedListings"); reductions=sm("AskingPriceReductions")
            return {
                "days_with_data":int(frame["Date"].nunique()),"observed_stock_days":round(denom,2),
                "new_listings":int(round(new)),"observed_exits":int(round(exits)),"price_reductions":int(round(reductions)),
                "exit_activity_rate":round(exits/denom,6) if denom>0 else None,
                "new_listing_activity_rate":round(new/denom,6) if denom>0 else None,
                "price_reduction_activity_rate":round(reductions/denom,6) if denom>0 else None,
            }

        def delta(curr,prev,key):
            a=_finite(curr.get(key)); b=_finite(prev.get(key))
            if a is None or b is None or b<=0:
                return None
            return round((a/b)-1.0,4)

        own=work[work["Company"].fillna("").astype(str).str.casefold()==company.casefold()].copy()
        others=work[work["Company"].fillna("").astype(str).str.casefold()!=company.casefold()].copy()
        own_now,own_prev=metrics(select(own,start,end)),metrics(select(own,prev_start,prev_end))
        market_now,market_prev=metrics(select(others,start,end)),metrics(select(others,prev_start,prev_end))
        own_delta=delta(own_now,own_prev,"exit_activity_rate")
        market_delta=delta(market_now,market_prev,"exit_activity_rate")
        if own_now["days_with_data"]==0 or market_now["days_with_data"]==0:
            return None
        if own_delta is None or market_delta is None:
            diagnosis="INSUFFICIENT_TREND_BASELINE"
        elif own_delta<=-0.15 and market_delta<=-0.15:
            diagnosis="BROAD_ACTIVITY_SLOWDOWN"
        elif own_delta<=-0.15 and market_delta>-0.10:
            diagnosis="BUSINESS_SPECIFIC_WEAKNESS"
        elif own_delta>=0.10 and market_delta<=-0.15:
            diagnosis="OUTPERFORMING_WEAKER_MARKET"
        elif own_delta>=0.10 and market_delta>=0.10:
            diagnosis="BROAD_ACTIVITY_IMPROVEMENT"
        else:
            diagnosis="MIXED_OR_STABLE"

        # Corrective context should come from the dealer's actual stock, not
        # generic advice. These are suggestions to inspect, not automatic edits.
        pricing=_business_price_tool(state,host).get("vehicles") or []
        aging=_business_aging_tool(state,host).get("vehicles") or []
        return {
            "scope":"observed_gallery_listing_activity_not_confirmed_sales",
            "company":company,
            "current_period":{"start":start.date().isoformat(),"end":end.date().isoformat()},
            "comparison_period":{"start":prev_start.date().isoformat(),"end":prev_end.date().isoformat()},
            "company_activity":own_now,"company_previous":own_prev,
            "wider_gallery_activity":market_now,"wider_gallery_previous":market_prev,
            "company_exit_activity_change_pct":round(own_delta*100,1) if own_delta is not None else None,
            "wider_exit_activity_change_pct":round(market_delta*100,1) if market_delta is not None else None,
            "diagnosis":diagnosis,
            "repricing_candidates":pricing[:3],"aging_candidates":aging[:3],
        }
    except Exception:
        return None


def _market_understanding_tool(state: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    df=_host(host,"business_market_df")
    if df is None or getattr(df,"empty",True):
        search=_search_vehicle_models(state,"",host)
        return {"kind":"market_understanding","status":"current_market_only","models":search.get("models") or [],"supports_time_trend":False}
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
        sort_cols=[x for x in ["OpportunityPercentile","CurrentListings"] if x in work.columns]
        if sort_cols: work=work.sort_values(sort_cols,ascending=[False]*len(sort_cols))
        raw=_df_records(work,30)
    except Exception:
        raw=[]
    rows=[]
    for r in raw:
        rate=_finite(r.get("ObservedExitWithin60DaysRate"))
        if rate is not None and rate>1.5: rate/=100.0
        eligible=_int(r.get("Exit60EligibleListings")) or _int(r.get("HistoricalDistinctListings")) or 0
        quality=str(r.get("EvidenceQuality") or "").upper()
        if rate is None or eligible<10 or quality in {"","INSUFFICIENT"}: turnover="LIMITED_EVIDENCE"
        elif rate>=0.75: turnover="STRONGER_OBSERVED_TURNOVER"
        elif rate>=0.50: turnover="MODERATE_OBSERVED_TURNOVER"
        else: turnover="WEAKER_OBSERVED_TURNOVER"
        reduction=_finite(r.get("PriceReductionRate"))
        if reduction is not None and reduction>1.5: reduction/=100.0
        rows.append({
            "VehicleType":r.get("VehicleType"),"Brand":r.get("Brand"),"Model":r.get("Model"),"CategoryDetail":r.get("CategoryDetail"),"Year":_int(r.get("Year")),
            "CurrentListings":_int(r.get("CurrentListings")),"CurrentStartingPrice":_finite(r.get("CurrentStartingPrice")),"CurrentMedianPrice":_finite(r.get("CurrentMedianPrice")),
            "TurnoverSignal":turnover,"EvidenceQuality":quality,"EvidenceSample":eligible,
            "PriceReductionRate":round(reduction,4) if reduction is not None else None,
            "AcquisitionSignal":r.get("AcquisitionSignal"),
        })
    business_context=_business_relative_market_context(state,host)
    return {
        "kind":"market_understanding","status":"ok" if rows or business_context else "insufficient","rows":rows[:20],
        "supports_time_trend":bool(business_context),
        "time_trend_scope":"observed_gallery_listing_activity_not_confirmed_sales" if business_context else None,
        "business_context":business_context,
    }

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
    if action == Action.EVALUATE_SALE.value:
        return _evaluate_price_subject(state,targets,host,"sale_evaluation")
    if action == Action.VALUE_VEHICLE.value:
        return _evaluate_price_subject(state,targets,host,"vehicle_valuation")
    if action == Action.ANALYZE_STOCK_PRICES.value:
        return _business_price_tool(state,host,targets)
    if action == Action.ANALYZE_AGING_STOCK.value:
        return _business_aging_tool(state,host,targets)
    if action == Action.RECOMMEND_ACQUISITIONS.value:
        return _business_acquire_tool(state,host,targets)
    if action == Action.ANALYZE_BUSINESS_PERIOD.value:
        return _business_period_tool(state,host)
    if action == Action.EVALUATE_TRADE_IN.value:
        return _business_trade_in_tool(state,host,targets)
    if action == Action.RECOMMEND_AD_CANDIDATE.value:
        return _business_ad_candidate_tool(state,host,targets)
    if action == Action.ANALYZE_MARKET.value:
        return _market_understanding_tool(state,host)
    if action == Action.EXPLAIN_RESULT.value:
        return {"kind":"explain_result","last_result":state.get("last_result")}
    if action == Action.ASK_CLARIFICATION.value:
        return {"kind":"clarification","awaiting":resolved_plan.get("awaiting")}
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
        Job.EVALUATE_TRADE_IN.value: Action.EVALUATE_TRADE_IN.value,
        Job.PROMOTE_STOCK.value: Action.RECOMMEND_AD_CANDIDATE.value,
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
        # The models the user was just shown are all valid conversational targets.
        # Do not silently reduce a 5-8 option set to the first two; explicit compare
        # commands will ask the user which subset they want when needed.
        focus=shortlist[:8]
    elif kind == "listings":
        for item in (evidence.get("listings") or [])[:12]:
            oid=register_object(state,ObjectType.LISTING.value,dict(item))
            shortlist.append(oid)
        focus=shortlist[:5]
    elif kind == "comparison":
        focus=[str(x.get("object_id")) for x in (evidence.get("vehicles") or []) if x.get("object_id")]
    elif kind in {"purchase_evaluation","sale_evaluation","vehicle_valuation","business_trade_in"}:
        # Keep existing subject focus; register comparable listings for follow-up links.
        for item in (evidence.get("comparables") or [])[:8]:
            try:
                shortlist.append(register_object(state,ObjectType.LISTING.value,dict(item)))
            except Exception:
                pass
    elif kind in {"business_pricing","business_aging","business_ad_candidate"}:
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
    if kind == "vehicle_valuation":
        if evidence.get("status") != "ok" or _finite(evidence.get("median_asking_price")) is None:
            return {"verdict":"INSUFFICIENT_EVIDENCE","reason":"Need identifiable vehicle details and comparable market evidence."}
        median=_finite(evidence.get("median_asking_price"))
        p25=_finite(evidence.get("p25_asking_price"))
        p75=_finite(evidence.get("p75_asking_price"))
        return {
            "verdict":"MARKET_VALUE_ESTIMATE",
            "reason":f"Comparable asking-price median is {_money(median)}.",
            "estimated_value":median,
            "market_range_low":p25,
            "market_range_high":p75,
        }
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
        asks_for_choice=bool(re.search(r"\b(?:which|choose|pick|prefer|better|recommend|would you|best for me|decide)\b|hangisi|seç|sec|tercih|öner|oner|karar|какой|выбрать|лучше|предпоч", low, re.I))
        prefs=state.get("preferences") or {}
        active_prefs={k:v for k,v in prefs.items() if v not in (None,False,0,"",[],{})}
        if not asks_for_choice and not active_prefs:
            return {"verdict":"COMPARE_OPTIONS"}

        def level(value):
            t=str(value or "").strip().casefold()
            if t in {"high","strong","yüksek","yuksek","высокий","высокая"}: return 3
            if t in {"medium","moderate","orta","средний","средняя"}: return 2
            if t in {"low","weak","düşük","dusuk","низкий","низкая"}: return 1
            return 0
        profile_map={
            "economy":"Economy","luxury":"Luxury","comfort":"Comfort","performance":"Performance",
            "practicality":"Practicality","family":"Family","commute":"Commute",
        }
        scored=[]
        for v in vehicles:
            score=0.0; reasons=[]; prof=v.get("profile_context") or {}
            for pref,field in profile_map.items():
                if active_prefs.get(pref):
                    lv=level(prof.get(field)); score+=lv
                    if lv>=3: reasons.append(pref)
            if active_prefs.get("resale"):
                liq=str(v.get("liquidity_signal") or "")
                if liq=="STRONGER_OBSERVED_TURNOVER": score+=3; reasons.append("resale/liquidity")
                elif liq=="MODERATE_OBSERVED_TURNOVER": score+=2
                elif liq=="WEAKER_OBSERVED_TURNOVER": score+=1
            if active_prefs.get("newer"):
                score+=((_int(v.get("newest_year")) or 0)/1000.0)
            if active_prefs.get("low_mileage") and _finite(v.get("median_km")) is not None:
                score+=max(0.0,2.0-min(float(v.get("median_km"))/100000.0,2.0))
            scored.append((score,v,reasons))
        scored.sort(key=lambda x:(-x[0], -int(x[1].get("count") or 0), _finite(x[1].get("newest_year_starting_price")) or 1e18))
        if scored and active_prefs and scored[0][0] > 0:
            top=scored[0][1]
            return {
                "verdict":"PREFER",
                "preferred_object_id":top.get("object_id"),
                "preferred":{
                    "brand":top.get("brand"),
                    "model":top.get("model"),
                    "year":_int(top.get("year")) or _int(top.get("newest_year")),
                },
                "preference_reasons":scored[0][2],
            }
        if asks_for_choice and vehicles:
            # If the user explicitly asks us to choose, do not hide behind "it depends".
            # Use only supported market evidence: turnover signal, price pressure,
            # current choice depth, affordable year and then price as a tie-breaker.
            liq={"STRONGER_OBSERVED_TURNOVER":3.0,"MODERATE_OBSERVED_TURNOVER":2.0,"WEAKER_OBSERVED_TURNOVER":1.0,"LIMITED_EVIDENCE":0.0}
            pressure={"LOWER_PRICE_REDUCTION_PRESSURE":2.0,"MODERATE_PRICE_REDUCTION_PRESSURE":1.0,"HIGHER_PRICE_REDUCTION_PRESSURE":0.0,"LIMITED_EVIDENCE":0.0}
            market_scored=[]
            for v in vehicles:
                s0=liq.get(str(v.get("liquidity_signal") or "LIMITED_EVIDENCE"),0.0)*3.0
                s0+=pressure.get(str(v.get("price_pressure_signal") or "LIMITED_EVIDENCE"),0.0)
                s0+=min(int(v.get("count") or 0),50)/25.0
                s0+=((_int(v.get("newest_year")) or 0)-2000)/50.0
                market_scored.append((s0,v))
            market_scored.sort(key=lambda x:(-x[0], _finite(x[1].get("newest_year_starting_price")) or 1e18))
            top=market_scored[0][1]
            others=[v for v in vehicles if v is not top]
            reasons=[]
            if others:
                other=others[0]
                top_count=int(top.get("count") or 0); other_count=int(other.get("count") or 0)
                top_km=_finite(top.get("median_km")); other_km=_finite(other.get("median_km"))
                top_year=_int(top.get("newest_year")); other_year=_int(other.get("newest_year"))
                top_start=_finite(top.get("newest_year_starting_price")); other_start=_finite(other.get("newest_year_starting_price"))
                if top_count>=other_count*1.2 and top_count-other_count>=5:
                    reasons.append(f"more current choice ({top_count} vs {other_count} matching listings)")
                if top_km is not None and other_km is not None and top_km<=other_km*0.9:
                    reasons.append(f"lower median mileage ({top_km:,.0f} vs {other_km:,.0f} km)")
                if top_year is not None and other_year is not None and top_year>other_year:
                    reasons.append(f"your budget reaches a newer year ({top_year} vs {other_year})")
                if top_start is not None and other_start is not None and top_start<=other_start*0.95:
                    reasons.append(f"lower entry asking price ({_money(top_start)} vs {_money(other_start)})")
                top_liq=str(top.get("liquidity_signal") or ""); other_liq=str(other.get("liquidity_signal") or "")
                if liq.get(top_liq,0)>liq.get(other_liq,0):
                    reasons.append("stronger observed turnover evidence")
                top_pressure=str(top.get("price_pressure_signal") or ""); other_pressure=str(other.get("price_pressure_signal") or "")
                if pressure.get(top_pressure,0)>pressure.get(other_pressure,0):
                    reasons.append("less asking-price reduction pressure")
            if not reasons:
                reasons=["a small edge on the supported market signals"]
            return {
                "verdict":"PREFER",
                "preferred_object_id":top.get("object_id"),
                "preferred":{
                    "brand":top.get("brand"),
                    "model":top.get("model"),
                    "year":_int(top.get("year")) or _int(top.get("newest_year")),
                },
                "preference_reasons":reasons[:3],
            }
        return {"verdict":"COMPARE_OPTIONS","reason":"No supported preference clearly separates the options."}
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
    if kind == "business_trade_in":
        if evidence.get("status") != "ok" or _finite(evidence.get("median_asking_price")) is None:
            return {"verdict":"INSUFFICIENT_EVIDENCE"}
        if _finite(evidence.get("suggested_offer_ceiling")) is not None:
            return {"verdict":"TRADE_IN_OFFER_GUIDANCE","suggested_offer_ceiling":evidence.get("suggested_offer_ceiling")}
        return {"verdict":"RETAIL_BENCHMARK_READY_NEEDS_MARGIN"}
    if kind == "business_ad_candidate":
        return {"verdict":"PROMOTE_TOP_CANDIDATE" if evidence.get("vehicles") else "INSUFFICIENT_EVIDENCE"}
    if kind == "business_period":
        return {"verdict":"PERIOD_RECAP" if evidence.get("status")=="ok" else "HISTORICAL_ACTIVITY_NOT_READY"}
    return {"verdict":"INFORM"}


def _localized_listing_color(value: Any, language: str) -> str:
    text=_text(value,80)
    if not text:
        return ""
    lang=_lang(language)
    if lang == "TR":
        return text
    key=text.casefold().replace("ı","i").replace("ş","s").replace("ğ","g").replace("ü","u").replace("ö","o").replace("ç","c")
    en={
        "siyah":"Black","beyaz":"White","gumus":"Silver","gri":"Grey","fume":"Dark grey",
        "mavi":"Blue","kirmizi":"Red","yesil":"Green","sari":"Yellow","bej":"Beige",
        "kahverengi":"Brown","turuncu":"Orange","lacivert":"Navy","bordo":"Burgundy",
        "inci beyaz":"Pearl white","metalik gri":"Metallic grey","koyu gri":"Dark grey","acik gri":"Light grey",
    }
    ru={
        "siyah":"Чёрный","beyaz":"Белый","gumus":"Серебристый","gri":"Серый","fume":"Тёмно-серый",
        "mavi":"Синий","kirmizi":"Красный","yesil":"Зелёный","sari":"Жёлтый","bej":"Бежевый",
        "kahverengi":"Коричневый","turuncu":"Оранжевый","lacivert":"Тёмно-синий","bordo":"Бордовый",
        "inci beyaz":"Перламутровый белый","metalik gri":"Серый металлик","koyu gri":"Тёмно-серый","acik gri":"Светло-серый",
    }
    return (en if lang == "EN" else ru).get(key,text)

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
            if (state.get("constraints") or {}).get("colors"):
                color=_localized_listing_color(row.get("color"),language)
                if color:
                    details.append(color)
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
    if evidence.get("kind") == "vehicle_valuation":
        actions.append({"type":"VALUATION","label":labels["valuation"],"url":"https://otodeger.online"})
        return actions, None
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

    def vehicle_label(subject):
        return " ".join(str(x) for x in [subject.get("year"),subject.get("brand"),subject.get("model"),subject.get("category")] if x not in (None,""))

    def pct_text(value):
        x=_finite(value)
        return f"{abs(x):.1f}%" if x is not None else None

    if kind=="vehicle_search":
        models=evidence.get("models") or []
        if not models:
            return {"EN":"I couldn't find a current match for those constraints. Tell me which one you are most willing to relax and I'll keep the rest.","TR":"Bu kriterlere uyan güncel bir seçenek bulamadım. En kolay esnetebileceğiniz kriteri söyleyin; diğerlerini koruyayım.","RU":"Я не нашёл актуальных вариантов по всем условиям. Скажите, какое одно ограничение можно ослабить — остальные я сохраню."}[lang]
        shown=models[:8]
        def line(m):
            label=f"{m.get('brand')} {m.get('model')}".strip()
            year=m.get('newest_year'); price=_money(m.get('newest_year_starting_price')); count=int(m.get('count') or 0)
            if lang=="TR": return f"**{label}** — {count} uygun ilan · bütçedeki en yeni yıl {year or '—'}, {price or '—'}'dan başlayan"
            if lang=="RU": return f"**{label}** — {count} подходящих объявлений · самый новый доступный год {year or '—'}, от {price or '—'}"
            return f"**{label}** — {count} matching listings · newest affordable year {year or '—'}, from {price or '—'}"
        intro={"EN":"These are some of the stronger current matches for your criteria, not the full market.","TR":"Bunlar kriterlerinize uyan daha güçlü güncel seçeneklerden bazıları; piyasanın tamamı değil.","RU":"Это несколько наиболее подходящих актуальных вариантов по вашим критериям, а не весь рынок."}[lang]
        alt=evidence.get("alternative_models") or []
        if alt:
            tail={"EN":"Other lower-activity alternatives include ","TR":"Daha düşük aktiviteli diğer alternatifler: ","RU":"Другие варианты с меньшей активностью: "}[lang]+", ".join(str(x) for x in alt[:4])+"."
        else:
            tail=""
        return intro+"\n"+"\n".join(line(m) for m in shown)+("\n"+tail if tail else "")

    if kind=="listings":
        n=min(10,len(evidence.get("listings") or [])); total=int(evidence.get("count") or n)
        if not n:
            return {"EN":"No current listings match all of those filters. Tell me which one you would relax first and I'll keep the rest.","TR":"Bu filtrelerin tamamına uyan güncel ilan yok. Önce hangi kriteri esnetebileceğinizi söyleyin; diğerlerini koruyayım.","RU":"Нет актуальных объявлений, соответствующих всем фильтрам. Скажите, какое условие можно ослабить первым — остальные я сохраню."}[lang]
        if lang=="TR": return f"Aşağıda {n} güncel ilan gösteriyorum"+(f" ({total} eşleşme içinden)." if total>n else ".")
        if lang=="RU": return f"Ниже {n} актуальных объявлений"+(f" из {total} совпадений." if total>n else ".")
        return f"I’ve put {n} current listings below"+(f" from {total} matches." if total>n else ".")

    if kind=="comparison":
        vehicles=(evidence.get("vehicles") or [])[:4]
        if len(vehicles)<2:
            return {"EN":"I don't yet have two comparable options to put side by side.","TR":"Yan yana karşılaştırmak için henüz iki uygun seçenek yok.","RU":"Пока нет двух подходящих вариантов для сравнения."}[lang]
        headers={"EN":["Option","Year","Asking price","Choice","Median km","Fuel","Market signal"],"TR":["Seçenek","Yıl","İlan fiyatı","Seçenek sayısı","Medyan km","Yakıt","Piyasa sinyali"],"RU":["Вариант","Год","Цена","Выбор","Медиана км","Топливо","Рынок"]}[lang]
        rows=[]
        for v in vehicles:
            label=" ".join(x for x in [_text(v.get("brand"),80),_text(v.get("model"),100)] if x)
            yr=_int(v.get("year")) or _int(v.get("newest_year"))
            if _int(v.get("year")) is not None: label=f"{label} {yr}"
            lo=_money(v.get("newest_year_starting_price")); hi=_money(v.get("newest_year_highest_price"))
            price=(f"{lo}–{hi}" if lo and hi and lo!=hi else lo or "—")
            count=int(v.get("newest_year_count") or 0)
            km=_finite(v.get("median_km")); km_text=f"{km:,.0f}" if km is not None else "—"
            fuels=", ".join((v.get("fuels") or [])[:3]) or "—"
            signal=str(v.get("liquidity_signal") or "LIMITED_EVIDENCE")
            signal_labels={
                "EN":{"STRONGER_OBSERVED_TURNOVER":"Stronger turnover","MODERATE_OBSERVED_TURNOVER":"Moderate turnover","WEAKER_OBSERVED_TURNOVER":"Weaker turnover","LIMITED_EVIDENCE":"Limited evidence"},
                "TR":{"STRONGER_OBSERVED_TURNOVER":"Daha güçlü devir","MODERATE_OBSERVED_TURNOVER":"Orta devir","WEAKER_OBSERVED_TURNOVER":"Daha zayıf devir","LIMITED_EVIDENCE":"Sınırlı veri"},
                "RU":{"STRONGER_OBSERVED_TURNOVER":"Выше оборот","MODERATE_OBSERVED_TURNOVER":"Средний оборот","WEAKER_OBSERVED_TURNOVER":"Ниже оборот","LIMITED_EVIDENCE":"Мало данных"},
            }[lang]
            rows.append([label,str(yr or "—"),price,str(count),km_text,fuels,signal_labels.get(signal,"Limited evidence")])
        table="| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(r)+" |" for r in rows)
        recommendation=""
        if decision.get("verdict")=="PREFER":
            pref=decision.get("preferred") or {}
            pbrand=_text(pref.get("brand"),80); pmodel=_text(pref.get("model"),100); pyear=_int(pref.get("year"))
            preferred_label=" ".join(str(x) for x in [pbrand,pmodel,pyear] if x not in (None,""))
            reasons=[str(x) for x in (decision.get("preference_reasons") or []) if str(x).strip()]
            reason_text=", ".join(reasons[:2])
            if preferred_label:
                if lang=="TR":
                    recommendation=f"**Ben {preferred_label} seçerdim.**"+(f" Bunu destekleyen başlıca sinyaller: {reason_text}." if reason_text else " Mevcut piyasa kanıtı bu seçeneği öne çıkarıyor.")
                elif lang=="RU":
                    recommendation=f"**Я бы выбрал {preferred_label}.**"+(f" Основные подтверждающие сигналы: {reason_text}." if reason_text else " Текущие рыночные данные дают этому варианту преимущество.")
                else:
                    recommendation=f"**I would choose the {preferred_label}.**"+(f" The main supporting signals are {reason_text}." if reason_text else " The current market evidence gives it the edge.")
        tail={"EN":"If you want, I can now compare an older year against the newest one rather than assuming newer is automatically better.","TR":"İsterseniz şimdi daha eski bir yılı en yeni seçenekle karşılaştırabiliriz; en yeninin otomatik olarak en iyi olduğunu varsaymam.","RU":"Могу также сравнить более старый год с самым новым, не предполагая, что новее автоматически значит лучше."}[lang]
        return table+("\n\n"+recommendation if recommendation else "")+"\n\n"+tail

    if kind=="vehicle_valuation":
        return {"EN":"For your own car's value, use OtoDeğer’s dedicated valuation tool so the estimate comes from the full valuation flow.","TR":"Kendi aracınızın değeri için OtoDeğer'in özel değerleme aracını kullanın; böylece hesaplama tam değerleme akışından gelir.","RU":"Для оценки собственного автомобиля используйте специальный инструмент OtoDeğer — он применяет полный процесс оценки."}[lang]

    if kind in {"purchase_evaluation","sale_evaluation"}:
        if evidence.get("status")!="ok" or evidence.get("price") is None:
            return {"EN":"I understand the price decision, but I don't have enough comparable market evidence yet. Give me the missing vehicle detail and I'll broaden the comparison sensibly rather than guessing.","TR":"Fiyat kararını anlıyorum, ancak henüz yeterli karşılaştırılabilir piyasa verisi yok. Eksik araç detayını verin; tahmin yürütmek yerine karşılaştırmayı mantıklı şekilde genişleteyim.","RU":"Я понимаю ценовой вопрос, но пока недостаточно сопоставимых рыночных данных. Дайте недостающую характеристику автомобиля — я разумно расширю сравнение вместо догадки."}[lang]
        subject=evidence.get("subject") or {}; vehicle=vehicle_label(subject) or {"EN":"this vehicle","TR":"bu araç","RU":"этот автомобиль"}[lang]
        price=_money(evidence.get("price")); med=_money(evidence.get("median_asking_price")); p25=_money(evidence.get("p25_asking_price")); p75=_money(evidence.get("p75_asking_price")); n=int(evidence.get("comparables_count") or 0)
        vs=_finite(evidence.get("vs_median_pct")); fallback_note=""
        thin_note=""
        if n<=2:
            if lang=="EN": thin_note=f" Only {n} current comparable{' was' if n==1 else 's were'} available, so treat this as a directional benchmark rather than a precise valuation."
            elif lang=="TR": thin_note=f" Yalnızca {n} güncel karşılaştırılabilir ilan bulundu; bu nedenle bunu kesin bir değerlemeden çok yön gösteren bir referans olarak kullanın."
            else: thin_note=f" Доступно только {n} сопоставимых объявлени{'е' if n==1 else 'я'}, поэтому это ориентир, а не точная оценка."
        if evidence.get("category_fallback_used"):
            requested=_text(evidence.get("requested_category"),80)
            fallback_note={"EN":f" Exact {requested or 'trim'} evidence was limited, so I broadened the benchmark to similar {subject.get('brand') or ''} {subject.get('model') or ''} listings.","TR":f" Tam {requested or 'versiyon'} verisi sınırlı olduğu için kıyaslamayı benzer {subject.get('brand') or ''} {subject.get('model') or ''} ilanlarına genişlettim.","RU":f" Данных именно по версии {requested or ''} мало, поэтому ориентир расширен до похожих {subject.get('brand') or ''} {subject.get('model') or ''}."}[lang]
        if kind=="purchase_evaluation":
            if vs is not None and vs>=10:
                lead={"EN":f"**{price} looks expensive** for this {vehicle}.","TR":f"Bu {vehicle} için **{price} pahalı görünüyor**.","RU":f"**{price} выглядит дорого** для {vehicle}."}[lang]
                target=f"{med}–{p75}" if med and p75 and med!=p75 else med or p75
                action={"EN":f"I would negotiate toward roughly **{target}** before condition/history adjustments.","TR":f"Kondisyon/geçmiş ayarlamalarından önce yaklaşık **{target}** seviyesine pazarlık etmeye çalışırdım.","RU":f"До поправок на состояние/историю я бы торговался примерно к **{target}**."}[lang]
            elif vs is not None and vs<=-8:
                lead={"EN":f"**{price} looks attractive on asking price** for this {vehicle}.","TR":f"Bu {vehicle} için **{price} ilan fiyatı açısından cazip görünüyor**.","RU":f"**{price} выглядит привлекательно по цене объявления** для {vehicle}."}[lang]
                action={"EN":"The price is already below the comparable median, so condition, history and inspection matter more than forcing a further discount.","TR":"Fiyat zaten karşılaştırılabilir medyanın altında; bu nedenle daha fazla indirim zorlamaktan çok kondisyon, geçmiş ve ekspertiz önemli.","RU":"Цена уже ниже медианы сопоставимых объявлений, поэтому состояние, история и осмотр важнее дополнительной скидки."}[lang]
            else:
                lead={"EN":f"**{price} looks broadly fair** for this {vehicle}.","TR":f"Bu {vehicle} için **{price} genel olarak makul görünüyor**.","RU":f"**{price} выглядит в целом справедливо** для {vehicle}."}[lang]
                action={"EN":f"There may still be room to negotiate toward the comparable median of **{med}**.","TR":f"Yine de **{med}** olan karşılaştırılabilir medyana doğru pazarlık payı olabilir.","RU":f"При этом можно попробовать торговаться к медиане сопоставимых объявлений — **{med}**."}[lang]
        else:
            if vs is not None and vs<=-10:
                lead={"EN":f"**I would not accept {price} immediately.** It is {pct_text(vs)} below the comparable asking-price median for {vehicle}.","TR":f"**{price} teklifini hemen kabul etmezdim.** {vehicle} için karşılaştırılabilir ilan medyanının %{abs(vs):.1f} altında.","RU":f"**Я бы не принимал {price} сразу.** Это на {pct_text(vs)} ниже медианы сопоставимых объявлений для {vehicle}."}[lang]
                counter=f"{p25}–{med}" if p25 and med and p25!=med else med or p25
                action={"EN":f"A defensible counter zone is roughly **{counter}**, then adjust for condition/history and how urgently you want to sell.","TR":f"Savunulabilir karşı teklif aralığı yaklaşık **{counter}**; ardından kondisyon/geçmiş ve satış aciliyetinize göre ayarlayın.","RU":f"Обоснованный диапазон встречного предложения — примерно **{counter}**, затем корректируйте по состоянию, истории и срочности продажи."}[lang]
            elif vs is not None and vs>=5:
                lead={"EN":f"**{price} is a strong relative offer** for {vehicle}; it is above the comparable asking-price median.","TR":f"**{price}, {vehicle} için göreceli olarak güçlü bir teklif**; karşılaştırılabilir ilan medyanının üzerinde.","RU":f"**{price} — относительно сильное предложение** для {vehicle}; оно выше медианы сопоставимых объявлений."}[lang]
                action={"EN":"If you are happy with the vehicle-specific condition/history factors, there may be little market-based reason to push much higher.","TR":"Araca özel kondisyon/geçmiş faktörlerinden memnunsanız, piyasa açısından çok daha yukarısını zorlamak için sınırlı gerekçe var.","RU":"Если состояние и история автомобиля вас устраивают, рыночных оснований сильно повышать цену немного."}[lang]
            else:
                lead={"EN":f"**{price} is broadly reasonable**, but you still have some room to negotiate for {vehicle}.","TR":f"**{price} genel olarak makul**, ancak {vehicle} için hâlâ bir miktar pazarlık payınız var.","RU":f"**{price} в целом разумно**, но по {vehicle} ещё есть пространство для торга."}[lang]
                counter=f"{med}" if med else f"{p25}–{p75}"
                action={"EN":f"I would test a counter around **{counter}** if you are not in a rush.","TR":f"Aceleniz yoksa yaklaşık **{counter}** seviyesinde karşı teklif deneyebilirsiniz.","RU":f"Если не спешите, можно попробовать встречное предложение около **{counter}**."}[lang]
        if n<=2:
            if lang=="EN":
                direction="above" if (vs or 0)>0 else "below" if (vs or 0)<0 else "close to"
                lead=f"With only **{n} current comparable{'s' if n!=1 else ''}**, **{price} appears {direction} the limited observed benchmark** for this {vehicle}; confidence is lower than in a deeper market sample."
            elif lang=="TR":
                lead=f"Yalnızca **{n} güncel kıyas** olduğu için **{price}**, bu {vehicle} için sınırlı gözlemlenen referansa göre {'yüksek' if (vs or 0)>0 else 'düşük' if (vs or 0)<0 else 'yakın'} görünüyor; güven düzeyi daha düşük."
            else:
                lead=f"Доступно только **{n} сопоставимых объявлени{'е' if n==1 else 'я'}**; **{price}** выглядит {'выше' if (vs or 0)>0 else 'ниже' if (vs or 0)<0 else 'близко к'} ограниченному ориентиру для {vehicle}, поэтому уверенность ниже."
        benchmark={"EN":f"Comparable asking prices: median **{med}**, middle range **{p25}–{p75}** across {n} listing{'s' if n!=1 else ''}.","TR":f"Karşılaştırılabilir ilanlar: medyan **{med}**, orta aralık **{p25}–{p75}**, {n} ilan.","RU":f"Сопоставимые объявления: медиана **{med}**, средний диапазон **{p25}–{p75}**, {n} объявлений."}[lang]
        caveat={"EN":"These are asking-price benchmarks, not confirmed transaction prices.","TR":"Bunlar ilan fiyatı kıyaslarıdır; doğrulanmış satış fiyatları değildir.","RU":"Это ориентиры по ценам объявлений, а не подтверждённые цены сделок."}[lang]
        return lead+fallback_note+"\n\n"+benchmark+thin_note+" "+action+" "+caveat

    if kind=="business_trade_in":
        if evidence.get("status")!="ok":
            return {"EN":"I can price the trade-in once I have enough comparable market evidence for the vehicle.","TR":"Araç için yeterli karşılaştırılabilir piyasa verisi olduğunda takas teklifini hesaplayabilirim.","RU":"Я смогу рассчитать предложение по trade-in, когда будет достаточно сопоставимых рыночных данных."}[lang]
        subject=evidence.get("subject") or {}; vehicle=vehicle_label(subject); med=_money(evidence.get("median_asking_price")); lo=_money(evidence.get("p25_asking_price")); hi=_money(evidence.get("p75_asking_price")); ceiling=_money(evidence.get("suggested_offer_ceiling")); n=int(evidence.get("comparables_count") or 0)
        if ceiling:
            margin=_finite(evidence.get("target_margin_pct")); prep=_money(evidence.get("prep_allowance")) if _finite(evidence.get("prep_allowance")) else None
            return {"EN":f"For the {vehicle}, the retail asking benchmark is about **{med}** (middle range **{lo}–{hi}**, {n} comparables). With your {margin:.0f}% target margin"+(f" and {prep} prep allowance" if prep else "")+f", I would cap the trade-in offer around **{ceiling}**. That is an acquisition ceiling, not a guaranteed resale value.","TR":f"{vehicle} için perakende ilan kıyası yaklaşık **{med}** (orta aralık **{lo}–{hi}**, {n} kıyas). %{margin:.0f} hedef marjınızla"+(f" ve {prep} hazırlık payıyla" if prep else "")+f" takas teklifini yaklaşık **{ceiling}** seviyesinde sınırlandırırdım. Bu bir alım tavanıdır, garanti satış değeri değildir.","RU":f"Для {vehicle} ориентир розничной цены около **{med}** (средний диапазон **{lo}–{hi}**, {n} аналогов). При целевой марже {margin:.0f}%"+(f" и резерве на подготовку {prep}" if prep else "")+f" я бы ограничил trade-in примерно **{ceiling}**. Это потолок закупки, не гарантированная цена перепродажи."}[lang]
        return {"EN":f"For the {vehicle}, the retail asking benchmark is about **{med}**, with a middle comparable range of **{lo}–{hi}** across {n} listings. What gross margin do you normally target on trade-ins? Give me that percentage and I'll turn this into a maximum offer.","TR":f"{vehicle} için perakende ilan kıyası yaklaşık **{med}**; {n} ilanda orta aralık **{lo}–{hi}**. Takaslarda normalde hedeflediğiniz brüt marj yüzde kaç? Yüzdeyi söyleyin, bunu maksimum teklife çevireyim.","RU":f"Для {vehicle} ориентир розничной цены около **{med}**, средний диапазон **{lo}–{hi}** по {n} объявлениям. Какую валовую маржу вы обычно закладываете в trade-in? Назовите процент — я рассчитаю максимальное предложение."}[lang]

    if kind=="business_pricing":
        vehicles=evidence.get("vehicles") or []
        if not vehicles:
            if evidence.get("status") == "company_or_stock_unavailable":
                company=_text(evidence.get("company"),160)
                if company:
                    return {"EN":f"I couldn't match **{company}** to current Business stock data. Check the Business profile name against how the gallery appears on its listings, then try again.","TR":f"**{company}** adını güncel Business stok verisiyle eşleştiremedim. Business profilindeki galeri adını ilanlarda göründüğü şekliyle kontrol edip tekrar deneyin.","RU":f"Не удалось сопоставить **{company}** с текущими данными склада Business. Проверьте название автосалона в Business-профиле так, как оно указано в объявлениях, и повторите запрос."}[lang]
                return {"EN":"Link your gallery in the Business profile first, then I can analyse which stock needs repricing.","TR":"Önce Business profilinde galerinizi bağlayın; ardından hangi stokların yeniden fiyatlanması gerektiğini analiz edebilirim.","RU":"Сначала укажите автосалон в Business-профиле — после этого я смогу определить, какие позиции нужно переоценить."}[lang]
            return {"EN":"I couldn't identify any stock item with enough price evidence to recommend a change right now.","TR":"Şu anda fiyat değişikliği önermek için yeterli fiyat verisi olan bir stok aracı belirleyemedim.","RU":"Сейчас не удалось найти складской автомобиль с достаточными данными для рекомендации по цене."}[lang]
        lines=[]
        for r in vehicles[:6]:
            name=" ".join(str(x) for x in [r.get("Year"),r.get("Brand"),r.get("Model"),r.get("CategoryDetail")] if x not in (None,""))
            ask=_money(r.get("CurrentAskingPrice")); med=_money(r.get("competitive_target")); quick=_money(r.get("quick_sale_target")); premium=_money(r.get("premium_target")); band=_text(r.get("PricePositionBand"),60) or "—"
            status=str(r.get("reprice_anchor_status") or "READY").upper(); reasons=set(r.get("reprice_anchor_reasons") or [])
            if status=="WITHHELD":
                raw_med=_money(r.get("ComparableMedianPrice")); km=_int(r.get("KM")); comp_km=_int(r.get("ComparableMedianKM")); why=[]
                if "variant_not_matched" in reasons: why.append("the comparable set does not match this engine/trim")
                if "mileage_mismatch" in reasons and km is not None and comp_km is not None: why.append(f"mileage is {km:,} km vs {comp_km:,} km comparable median")
                if "too_few_comparables" in reasons: why.append("the comparable set is too small")
                why_text=" and ".join(why) or "the comparable set is not close enough"
                if lang=="EN": lines.append(f"**{name}** — now {ask}; flagged {band}. Raw comparable median {raw_med or '—'}, but **I would not set a target price from this set** because {why_text}.")
                elif lang=="TR": lines.append(f"**{name}** — şu an {ask}; {band} olarak işaretli. Ham kıyas medyanı {raw_med or '—'}, ancak **bu kıyas setinden hedef fiyat belirlemezdim**; kıyaslar araçla yeterince yakın değil.")
                else: lines.append(f"**{name}** — сейчас {ask}; позиция {band}. Сырая медиана аналогов {raw_med or '—'}, но **я бы не задавал целевую цену по этому набору**: аналоги недостаточно близки.")
            else:
                caution=(" Directional only: the comparable set is lower-confidence." if status=="CAUTION" and lang=="EN" else "")
                if lang=="EN": lines.append(f"**{name}** — now {ask}; market position {band}. Quick-sale anchor **{quick or '—'}**, competitive **{med or '—'}**, premium **{premium or '—'}**.{caution}")
                elif lang=="TR": lines.append(f"**{name}** — şu an {ask}; piyasa konumu {band}. Hızlı satış **{quick or '—'}**, rekabetçi **{med or '—'}**, premium **{premium or '—'}**."+(" Kıyas seti daha düşük güvenli; rakamları yön gösterici kullanın." if status=="CAUTION" else ""))
                else: lines.append(f"**{name}** — сейчас {ask}; позиция {band}. Быстрая продажа **{quick or '—'}**, конкурентная **{med or '—'}**, премиум **{premium or '—'}**."+(" Набор аналогов менее надёжен; используйте цифры как ориентир." if status=="CAUTION" else ""))
        intro={"EN":"I would start with the stock where the comparable set is close enough to support an actual price move. When trim or mileage is materially mismatched, I flag the vehicle for review but deliberately withhold a target rather than risk a destructive price cut.","TR":"Önce kıyas seti gerçek bir fiyat hamlesini destekleyecek kadar yakın olan stoktan başlardım. Versiyon veya kilometre ciddi şekilde uyuşmuyorsa aracı inceleme için işaretler, zararlı bir fiyat indirimi riskine karşı hedef fiyatı bilinçli olarak vermem.","RU":"Я бы начал с позиций, где аналоги достаточно близки для реального изменения цены. При существенном расхождении комплектации или пробега я отмечаю автомобиль для проверки, но намеренно не даю целевую цену, чтобы не спровоцировать ошибочное снижение."}[lang]
        return intro+"\n\n"+"\n".join(lines)

    if kind=="business_aging":
        vehicles=evidence.get("vehicles") or []
        if not vehicles:
            if evidence.get("status") == "company_or_stock_unavailable":
                company=_text(evidence.get("company"),160)
                if company:
                    return {"EN":f"I couldn't match **{company}** to current Business stock data. Check the gallery name in your Business profile before I judge stock age.","TR":f"**{company}** adını güncel Business stok verisiyle eşleştiremedim. Stok yaşını değerlendirmeden önce Business profilindeki galeri adını kontrol edin.","RU":f"Не удалось сопоставить **{company}** с текущими данными склада. Проверьте название автосалона в Business-профиле перед анализом возраста склада."}[lang]
                return {"EN":"Link your gallery in the Business profile first, then I can identify which stock has been sitting unusually long.","TR":"Önce Business profilinde galerinizi bağlayın; ardından hangi stokların alışılmadık derecede uzun süredir beklediğini belirleyebilirim.","RU":"Сначала укажите автосалон в Business-профиле — затем я смогу определить, какие машины стоят дольше обычного."}[lang]
            return {"EN":"I couldn't identify aging stock with enough evidence to call out right now.","TR":"Şu anda öne çıkaracak kadar veriye sahip yaşlanan stok belirleyemedim.","RU":"Сейчас не удалось выделить залежавшийся склад с достаточными данными."}[lang]
        lines=[]
        for r in vehicles[:4]:
            name=" ".join(str(x) for x in [r.get("Year"),r.get("Brand"),r.get("Model"),r.get("CategoryDetail")] if x not in (None,""))
            age=_int(r.get("StockAgeDays")); med=_finite(r.get("HistoricalMedianObservedDaysToExit")); p75=_finite(r.get("HistoricalP75ObservedDaysToExit")); band=_text(r.get("PricePositionBand"),60)
            age_band=str(r.get("StockAgeBand") or "").upper(); liq_conf=str(r.get("LiquidityEvidenceConfidence") or "").upper()
            if age_band=="VERY_AGED": concern="high"
            elif age_band in {"AGED","ABOVE_TYPICAL"}: concern="watch"
            elif age_band=="INSUFFICIENT_EVIDENCE" or liq_conf=="INSUFFICIENT": concern="evidence-limited"
            else: concern="normal"
            if lang=="EN": lines.append(f"**{name}** — {age if age is not None else '—'} days observed · price position {band or '—'} · age assessment **{concern}**"+(f" (historical benchmark evidence is insufficient)." if concern=="evidence-limited" else "."))
            elif lang=="TR": lines.append(f"**{name}** — {age if age is not None else '—'} gündür gözlemde · fiyat konumu {band or '—'} · stok yaşı değerlendirmesi **{concern}**"+(" (tarihsel kıyas verisi yetersiz)." if concern=="evidence-limited" else "."))
            else: lines.append(f"**{name}** — наблюдается {age if age is not None else '—'} дн. · ценовая позиция {band or '—'} · оценка возраста **{concern}**"+(" (исторических данных недостаточно)." if concern=="evidence-limited" else "."))
        tail={"EN":"I would reprice only where the vehicle is both aged and poorly positioned; if price is already competitive, improve presentation/promotion or review the stock choice before cutting again.","TR":"Fiyatı yalnızca araç hem yaşlanmış hem de kötü konumlanmışsa düşürürdüm; fiyat zaten rekabetçiyse tekrar indirimden önce sunum/reklam veya stok seçimini gözden geçirirdim.","RU":"Я бы снижал цену только если автомобиль и залежался, и плохо позиционирован. Если цена уже конкурентна, сначала стоит улучшить подачу/рекламу или пересмотреть сам выбор склада."}[lang]
        return "\n".join(lines)+"\n\n"+tail

    if kind in {"business_acquisition","business_acquisition_specific"}:
        options=(evidence.get("options") or []) if kind=="business_acquisition" else (evidence.get("market_evidence") or [])
        if not options:
            return {"EN":"I don't have enough market evidence to recommend an acquisition from those constraints yet.","TR":"Bu kriterlerle stok alımı önermek için henüz yeterli piyasa verisi yok.","RU":"Пока недостаточно рыночных данных, чтобы рекомендовать закупку по этим условиям."}[lang]
        lines=[]
        for r in options[:5]:
            name=" ".join(str(x) for x in [r.get("Year"),r.get("Brand"),r.get("Model"),r.get("CategoryDetail")] if x not in (None,"")); start=_money(r.get("CurrentStartingPrice")); supply=_int(r.get("verified_current_matches")); supply=supply if supply is not None else _int(r.get("CurrentListings")); signal=_text(r.get("AcquisitionSignal"),60) or "—"; quality=_text(r.get("EvidenceQuality"),40) or "—"
            if lang=="EN": lines.append(f"**{name}** — from {start or '—'} · {supply if supply is not None else '—'} current listings · market opportunity **{signal}** · evidence {quality}.")
            elif lang=="TR": lines.append(f"**{name}** — {start or '—'}'dan başlayan · {supply if supply is not None else '—'} güncel ilan · piyasa fırsatı **{signal}** · veri {quality}.")
            else: lines.append(f"**{name}** — от {start or '—'} · {supply if supply is not None else '—'} текущих объявлений · сигнал **{signal}** · данные {quality}.")
        return {"EN":"These are the strongest acquisition candidates from the supported market signals; I would verify your buy-in price before committing.\n\n","TR":"Desteklenen piyasa sinyallerine göre en güçlü alım adayları bunlar; karar vermeden önce alış fiyatınızı ayrıca kontrol ederdim.\n\n","RU":"Это самые сильные кандидаты по поддерживаемым рыночным сигналам; перед закупкой стоит отдельно проверить закупочную цену.\n\n"}[lang]+"\n".join(lines)

    if kind=="business_ad_candidate":
        vehicles=evidence.get("vehicles") or []
        if not vehicles:
            if evidence.get("status") == "company_or_stock_unavailable":
                company=_text(evidence.get("company"),160)
                if company:
                    return {"EN":f"I couldn't match **{company}** to current Business stock data. Check the Business profile name before I rank vehicles for advertising.","TR":f"**{company}** adını güncel Business stok verisiyle eşleştiremedim. Reklam için araçları sıralamadan önce Business profilindeki galeri adını kontrol edin.","RU":f"Не удалось сопоставить **{company}** с текущими данными склада. Проверьте название в Business-профиле перед выбором машины для рекламы."}[lang]
                return {"EN":"Link your gallery in the Business profile first, then I can rank your actual stock for ad spend.","TR":"Önce Business profilinde galerinizi bağlayın; ardından gerçek stokunuzu reklam bütçesi için sıralayabilirim.","RU":"Сначала укажите автосалон в Business-профиле — затем я смогу ранжировать ваш фактический склад для рекламы."}[lang]
            return {"EN":"I couldn't find a stock vehicle with enough market evidence to justify recommending ad spend right now.","TR":"Şu anda reklam bütçesi önermek için yeterli piyasa verisi olan bir stok aracı bulamadım.","RU":"Сейчас не нашлось складского автомобиля с достаточными данными, чтобы оправдать рекламный бюджет."}[lang]
        top=vehicles[0]; name=" ".join(str(x) for x in [top.get("Year"),top.get("Brand"),top.get("Model"),top.get("CategoryDetail")] if x not in (None,"")); ask=_money(top.get("CurrentAskingPrice")); age=_int(top.get("StockAgeDays")); activity=_text(top.get("MarketActivitySignal"),80); band=_text(top.get("PricePositionBand"),60)
        activity_label={"STRONGER_OBSERVED_TURNOVER":"stronger observed market turnover","MODERATE_OBSERVED_TURNOVER":"moderate observed market turnover","WEAKER_OBSERVED_TURNOVER":"weaker observed market turnover","LIMITED_EVIDENCE":"limited turnover evidence"}.get(activity,"limited turnover evidence")
        if lang=="EN": return f"My first ad candidate is **{name}** at **{ask or '—'}**. It has {activity_label}, is currently positioned **{band or '—'}** on price, and has been observed for {age if age is not None else '—'} days. That makes it the strongest supported use of ad spend among the stock I could evaluate—not a guarantee of a sale."
        if lang=="TR": return f"İlk reklam adayım **{name}**, fiyatı **{ask or '—'}**. Piyasa devir sinyali {activity_label}, fiyat konumu **{band or '—'}** ve {age if age is not None else '—'} gündür gözlemde. Değerlendirebildiğim stok içinde reklam bütçesinin en güçlü desteklenen kullanımı bu; satış garantisi değil."
        return f"Первый кандидат для рекламы — **{name}** по цене **{ask or '—'}**. Рыночный сигнал: {activity_label}, ценовая позиция **{band or '—'}**, наблюдается {age if age is not None else '—'} дней. Это наиболее обоснованное применение рекламного бюджета среди оценённого склада, но не гарантия продажи."

    if kind=="business_period":
        if evidence.get("status")=="ok":
            s=evidence.get("summary") or {}; company=evidence.get("company") or {"EN":"your business","TR":"işletmeniz","RU":"ваш бизнес"}[lang]
            if lang=="EN": base=f"For **{company}** in the selected period: observed advertised stock moved from **{s.get('opening_observed_stock','—')}** to **{s.get('closing_observed_stock','—')}**; **{s.get('newly_observed_listings','—')}** listings appeared, **{s.get('observed_market_exits','—')}** left the observed market, and there were **{s.get('asking_price_reductions','—')}** asking-price reductions. These exits are listing removals, not confirmed sales."
            elif lang=="TR": base=f"Seçili dönemde **{company}** için gözlemlenen ilan stoğu **{s.get('opening_observed_stock','—')}**'dan **{s.get('closing_observed_stock','—')}**'a geldi; **{s.get('newly_observed_listings','—')}** yeni ilan görüldü, **{s.get('observed_market_exits','—')}** ilan gözlemlenen piyasadan çıktı ve **{s.get('asking_price_reductions','—')}** fiyat indirimi oldu. Çıkışlar doğrulanmış satış değildir."
            else: base=f"За выбранный период у **{company}** наблюдаемый рекламный склад изменился с **{s.get('opening_observed_stock','—')}** до **{s.get('closing_observed_stock','—')}**; появилось **{s.get('newly_observed_listings','—')}** новых объявлений, **{s.get('observed_market_exits','—')}** исчезло из наблюдаемого рынка, было **{s.get('asking_price_reductions','—')}** снижений цены. Исчезновение объявления не означает подтверждённую продажу."
            segments=evidence.get("segments") or []
            strong=next((x for x in segments if x.get("bet_signal")=="FAVORABLE" and x.get("dealer_evidence") in {"STRONG","USABLE"}),None)
            weak=next((x for x in segments if x.get("performance_signal")=="WEAK_RELATIVE_ACTIVITY" and x.get("dealer_evidence") in {"STRONG","USABLE"}),None)
            extras=[]
            if strong:
                seg=str(strong.get("segment") or "").replace("_"," ")
                extras.append({"EN":f"A segment worth leaning into is **{seg}**: your observed listing activity is relatively healthy and the market opportunity signal is supportive.","TR":f"Ağırlık vermeye değer bir segment **{seg}**: gözlemlenen ilan aktiviteniz göreceli olarak sağlıklı ve piyasa fırsat sinyali destekleyici.","RU":f"Сегмент, в который можно усилить ставку: **{seg}** — относительная активность объявлений здорова, а рыночный сигнал поддерживает."}[lang])
            else:
                supported=[x for x in segments if x.get("dealer_evidence") in {"STRONG","USABLE"} and x.get("relative_activity_index") is not None and x.get("performance_signal")!="WEAK_RELATIVE_ACTIVITY"]
                if supported:
                    best=max(supported,key=lambda x:(float(x.get("relative_activity_index") or 0),int(x.get("company_activity_events") or 0),int(x.get("company_current_stock") or 0)))
                    seg=str(best.get("segment") or "").replace("_"," ")
                    rel=_finite(best.get("relative_activity_index")); stock_n=_int(best.get("company_current_stock")); ev_n=_int(best.get("company_activity_events"))
                    if lang=="EN": extras.append(f"I don't see enough dealer-specific evidence to call a clear winning segment yet. The best-supported positive read is **{seg}** (relative activity index {rel:.2f}, {stock_n} current stock, {ev_n} observed new/exit events), so I would treat it as directional rather than a proven strength.")
                    elif lang=="TR": extras.append(f"Henüz net bir kazanan segment demek için yeterli galeriye özel veri yok. En güçlü desteklenen olumlu sinyal **{seg}** (göreli aktivite endeksi {rel:.2f}, {stock_n} güncel stok, {ev_n} gözlemlenen yeni/çıkış olayı); bunu kanıtlanmış bir güçten çok yön gösteren bir sinyal olarak kullanırdım.")
                    else: extras.append(f"Пока недостаточно данных именно по автосалону, чтобы назвать явный лучший сегмент. Наиболее подтверждённый положительный сигнал — **{seg}** (индекс относительной активности {rel:.2f}, текущий склад {stock_n}, наблюдаемых новых/исчезнувших объявлений {ev_n}); это скорее направление, чем доказанное преимущество.")
            if weak and weak is not strong:
                seg=str(weak.get("segment") or "").replace("_"," ")
                extras.append({"EN":f"I would review **{seg}** exposure: observed activity is weak relative to your current stock share.","TR":f"**{seg}** maruziyetini gözden geçirirdim: gözlemlenen aktivite mevcut stok payınıza göre zayıf.","RU":f"Стоит пересмотреть экспозицию **{seg}**: наблюдаемая активность слабая относительно доли текущего склада."}[lang])
            return base+("\n\n"+" ".join(extras) if extras else "")
        if evidence.get("status")=="current_snapshot_only":
            return {"EN":"I can see the current business snapshot, but I don't have the historical activity series needed to truthfully recap what changed over that period.","TR":"Güncel işletme görünümünü görebiliyorum ancak o dönemde ne değiştiğini güvenilir şekilde özetlemek için gerekli geçmiş aktivite serisi yok.","RU":"Текущий снимок бизнеса доступен, но нет исторического ряда, чтобы достоверно описать изменения за период."}[lang]
        return {"EN":"I understand the business recap you want, but the historical activity data is unavailable right now.","TR":"İstediğiniz işletme özetini anlıyorum ancak geçmiş aktivite verisi şu anda kullanılamıyor.","RU":"Я понимаю, какой бизнес-отчёт нужен, но исторические данные сейчас недоступны."}[lang]

    if kind=="market_understanding":
        context=evidence.get("business_context") or {}
        if context:
            diagnosis=str(context.get("diagnosis") or "")
            own=context.get("company_exit_activity_change_pct"); wider=context.get("wider_exit_activity_change_pct")
            company=context.get("company") or {"EN":"your dealership","TR":"galeriniz","RU":"ваш автосалон"}[lang]
            labels={
                "EN":{
                    "BROAD_ACTIVITY_SLOWDOWN":"Both your dealership and the wider gallery market show weaker observed listing-exit activity than the previous comparable period.",
                    "BUSINESS_SPECIFIC_WEAKNESS":"Your dealership's observed listing-exit activity weakened while the wider gallery market did not show the same slowdown.",
                    "OUTPERFORMING_WEAKER_MARKET":"The wider gallery market weakened, while your dealership's observed listing-exit activity held up better.",
                    "BROAD_ACTIVITY_IMPROVEMENT":"Both your dealership and the wider gallery market show stronger observed listing-exit activity.",
                    "MIXED_OR_STABLE":"The comparison is mixed rather than showing a clear broad slowdown.",
                    "INSUFFICIENT_TREND_BASELINE":"I have current gallery activity but not a strong enough previous-period baseline to diagnose the trend confidently.",
                },
                "TR":{
                    "BROAD_ACTIVITY_SLOWDOWN":"Hem galerinizde hem de daha geniş galeri piyasasında önceki karşılaştırılabilir döneme göre gözlemlenen ilan çıkış aktivitesi zayıfladı.",
                    "BUSINESS_SPECIFIC_WEAKNESS":"Galerinizin gözlemlenen ilan çıkış aktivitesi zayıflarken daha geniş galeri piyasasında aynı yavaşlama görülmüyor.",
                    "OUTPERFORMING_WEAKER_MARKET":"Daha geniş galeri piyasası zayıflarken galerinizin gözlemlenen ilan çıkış aktivitesi daha iyi dayandı.",
                    "BROAD_ACTIVITY_IMPROVEMENT":"Hem galerinizde hem de daha geniş galeri piyasasında gözlemlenen ilan çıkış aktivitesi güçlendi.",
                    "MIXED_OR_STABLE":"Karşılaştırma net bir genel yavaşlama yerine karışık/dengeli bir tablo gösteriyor.",
                    "INSUFFICIENT_TREND_BASELINE":"Güncel galeri aktivitesi var ancak trendi güvenle teşhis etmek için önceki dönem tabanı yeterince güçlü değil.",
                },
                "RU":{
                    "BROAD_ACTIVITY_SLOWDOWN":"И у вашего автосалона, и у более широкого рынка дилеров наблюдаемая активность исчезновения объявлений снизилась относительно предыдущего сопоставимого периода.",
                    "BUSINESS_SPECIFIC_WEAKNESS":"У вашего автосалона активность исчезновения объявлений снизилась, тогда как широкий рынок дилеров не показывает такого же замедления.",
                    "OUTPERFORMING_WEAKER_MARKET":"Широкий рынок дилеров ослаб, а активность вашего автосалона держится лучше.",
                    "BROAD_ACTIVITY_IMPROVEMENT":"И у вашего автосалона, и на широком рынке дилеров наблюдаемая активность усилилась.",
                    "MIXED_OR_STABLE":"Картина смешанная и не указывает на явное общее замедление.",
                    "INSUFFICIENT_TREND_BASELINE":"Текущая активность есть, но предыдущего периода недостаточно для уверенного вывода о тренде.",
                },
            }
            lead=labels[lang].get(diagnosis,labels[lang]["MIXED_OR_STABLE"])
            changes=""
            if own is not None and wider is not None:
                if lang=="TR":
                    def _tr_signed_pct(value):
                        sign="-" if value < 0 else "+"
                        return f"{sign}%{abs(value):.1f}".replace(".", ",")
                    changes=f" Önceki döneme göre gözlemlenen ilan çıkış aktivitesi: {company} {_tr_signed_pct(own)}, daha geniş galeri piyasası {_tr_signed_pct(wider)}."
                else:
                    changes={"EN":f" Relative to the previous period: {company} {own:+.1f}%, wider gallery market {wider:+.1f}% on this observed listing-exit activity rate.","RU":f" Относительно предыдущего периода: {company} {own:+.1f}%, широкий дилерский рынок {wider:+.1f}% по этой метрике активности."}[lang]
            caveat={"EN":" This is advert/listing activity, not confirmed sales.","TR":" Bu, ilan aktivitesidir; doğrulanmış satış değildir.","RU":" Это активность объявлений, а не подтверждённые продажи."}[lang]
            pricing=context.get("repricing_candidates") or []; aging=context.get("aging_candidates") or []
            action=""
            if diagnosis=="BUSINESS_SPECIFIC_WEAKNESS" and (pricing or aging):
                candidate=(pricing or aging)[0]
                name=" ".join(str(x) for x in [candidate.get("Year"),candidate.get("Brand"),candidate.get("Model")] if x not in (None,""))
                if name:
                    action={"EN":f" The first corrective item I would inspect is **{name}**, then review its price position and stock age before changing your acquisition strategy.","TR":f" İlk düzeltici olarak **{name}** aracını incelerdim; stok alım stratejisini değiştirmeden önce fiyat konumu ve stok yaşına bakın.","RU":f" Первым делом я бы проверил **{name}** — его ценовую позицию и возраст склада, прежде чем менять стратегию закупок."}[lang]
            return lead+changes+caveat+action
        rows=evidence.get("rows") or []
        if not rows:
            return {"EN":"I can inspect current market options, but I don't have enough historical evidence here to make a reliable demand conclusion.","TR":"Güncel piyasa seçeneklerini inceleyebilirim ancak burada güvenilir bir talep sonucu çıkarmak için yeterli geçmiş veri yok.","RU":"Я могу оценить текущий рынок, но здесь недостаточно исторических данных для надёжного вывода о спросе."}[lang]
        lines=[]
        signal_labels={"STRONGER_OBSERVED_TURNOVER":"stronger observed turnover","MODERATE_OBSERVED_TURNOVER":"moderate observed turnover","WEAKER_OBSERVED_TURNOVER":"weaker observed turnover","LIMITED_EVIDENCE":"limited evidence"}
        for r in rows[:5]:
            name=" ".join(str(x) for x in [r.get("Year"),r.get("Brand"),r.get("Model")] if x not in (None,"")); supply=_int(r.get("CurrentListings")); sig=signal_labels.get(str(r.get("TurnoverSignal") or ""),"limited evidence"); quality=_text(r.get("EvidenceQuality"),40) or "—"
            lines.append(f"**{name}** — {sig} · {supply if supply is not None else '—'} current listings · evidence {quality}")
        intro={"EN":"I can rank current segments by relative market signals, but this evidence does **not** establish whether the overall market has sped up or slowed down over time.","TR":"Güncel segmentleri göreceli piyasa sinyallerine göre sıralayabilirim ancak bu veri genel piyasanın zaman içinde hızlanıp yavaşladığını **göstermez**.","RU":"Я могу сравнить текущие сегменты по относительным рыночным сигналам, но эти данные **не показывают**, ускорился или замедлился весь рынок со временем."}[lang]
        return intro+"\n\n"+"\n".join(lines)

    if kind=="clarification":
        awaiting=clarification(evidence.get("awaiting")); q=_text(awaiting.get("question"),500) if isinstance(awaiting,Mapping) else (_text(awaiting,500) if isinstance(awaiting,str) else None)
        if q: return q

    return {"EN":"I understand the decision you're trying to make, but I don't have enough verified data to answer it safely yet. Give me the missing vehicle or business detail and I'll keep the rest of the context.","TR":"Vermeye çalıştığınız kararı anlıyorum ancak henüz güvenli yanıt için yeterli doğrulanmış veri yok. Eksik araç veya işletme detayını verin; diğer bağlamı koruyayım.","RU":"Я понимаю, какое решение вы принимаете, но пока не хватает проверенных данных. Дайте недостающую деталь по автомобилю или бизнесу — остальной контекст я сохраню."}[lang]

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
- Vehicle valuation is handled by the dedicated OtoDeğer valuation flow before this renderer. Do not duplicate that journey in chat.
- Sale evaluation: when the user supplies an actual offer or existing asking price, compare that supplied price with relevant market evidence and support negotiation decisions. Never turn observed listing removals into confirmed sales.
- Business acquisition: prioritise strong observed market activity and attractive supply/price-pressure conditions; never call observed removals confirmed sales.
- Business pricing: when evidence permits, frame urgent-sale / competitive-market / premium strategies and note vehicle-specific condition/trim adjustments.
- Aging stock: say whether age is concerning relative to relevant market evidence and give a concrete next action.
- Trade-in: first establish a retail-market benchmark. If a user/account target margin and prep allowance are supplied, translate that benchmark into an acquisition ceiling; otherwise give the benchmark and ask only for the missing commercial assumption.
- Ad candidate: choose among the dealer's ACTUAL stock, favouring stronger market-activity evidence, sensible price position and sufficient evidence. Explain why the chosen vehicle is a better use of ad spend; never promise a sale.
- ANALYZE_BUSINESS covers business health, stock in/out, asking-price changes, inventory count/value, stock age, mix and trends for arbitrary periods; month-to-date, YTD and trailing 12 months are especially useful. Surface material changes proactively.
- When business_period VERIFIED_EVIDENCE contains segments, use them to explain where the dealer shows stronger/weaker OBSERVED LISTING ACTIVITY relative to current stock exposure and where market opportunity signals support leaning in or reviewing exposure. Never call segment exits sales. Prefer a few material segments over dumping every grouping.
- UNDERSTAND_MARKET should answer the decision context: personal users care about buying timing/negotiating room; businesses care whether their performance reflects the wider market.
- Outside proprietary market scope, only answer North-Cyprus vehicle-adjacent questions when reliable external evidence is actually available; otherwise say it cannot be reliably verified. Do not become a generic assistant.
- English, Turkish and Russian must follow the same decision policy.

Rules:
- Match the response depth to the stage of the decision. Do not force a recommendation before the buyer has supplied preferences that make one meaningful.
- DISCOVERY / vehicle_search: map the useful option set. For a broad budget + body-type request, show the useful active model families rather than naming a winner; usually 4-8, but do not force a fixed count. The deterministic model order already prioritises observed market activity (historical listing volume + observed market exits) before current supply. Respect that order. Explicitly make clear these are SOME of the more active matches, not the entire market. If VERIFIED_EVIDENCE contains alternative_models, briefly name those model-level lower-activity alternatives and offer an expanded list. Never call an observed exit a confirmed sale/transaction and never equate current listing count alone with popularity.
- When a budget exists, model discovery is about WHAT THAT BUDGET BUYS. Say: MODEL — N matching listings across all eligible years · newest: YEAR, from £PRICE. N is NOT the number of cars from the newest year. Do not lead with an old model's overall minimum price.
- When the user names brands (for example BMW or Mercedes), show the relevant models under those brands with newest affordable year + asking price at that year + option count. Do not introduce mileage yet unless the user asks for mileage or is filtering listings by mileage.
- Do not append routine caveats such as "asking prices are not confirmed transaction prices" to ordinary discovery/comparison replies. Preserve that distinction internally and mention it only when it materially affects the decision.
- When current choice is limited (especially 1-2 cars at the newest viable year), say choice is limited and show the most useful way to broaden it. Do not use internal phrases such as "thin choice".
- COMPARISON: make it decision-oriented and scan-friendly, preferably as a Markdown table when 2-4 targets are being compared. For each target use its explicit year when supplied; otherwise show what the budget reaches plus useful older-year alternatives from year_summaries. Useful columns can include year, starting/range price, current choice, median mileage, fuel availability, asking-price reduction pressure and a qualitative historical liquidity signal when VERIFIED_EVIDENCE supports them. Do not assume newest year is automatically best. Only recommend a winner when the user asks which to choose or their stated priorities clearly justify one.
- SHOW_LISTINGS: the UI displays up to ten information-rich clickable listing rows below the prose. Each row contains the exact year/brand/model/variant plus available price, KM, transmission, location and seller/gallery. The prose must be ONE short introductory sentence only; NEVER repeat/list any vehicle, price, seller, mileage or model in the prose. The structured clickable rows are the single listing presentation.
- Ordinary response 35-130 words; simple answers may be shorter. Do not exceed 170 words unless essential.
- Use short paragraphs and compact model-per-line formatting. Avoid long prose comparisons.
- Mention only facts present in VERIFIED_EVIDENCE. Never invent prices, years, mileage, availability, counts, dealers or links. If a hard budget is active, do not mention above-budget alternatives unless the user explicitly asks what spending more would unlock.
- Only describe something as the user's requirement/criterion if it is present in the supplied constraints/preferences or explicitly stated in user_message. Evidence attributes (for example an automatic transmission on a listing) are facts about the vehicle, not automatically user requirements.
- Asking prices are not confirmed transaction prices, but do NOT add that disclaimer routinely in discovery/comparison/listing answers. Mention it only when the distinction is materially relevant to a valuation, negotiation or sale-price claim.
- Observed market exit is not a confirmed sale. Never say "sold within X days" from listing-removal evidence.
- For general market/demand/liquidity answers, prefer qualitative relative signals (stronger/weaker observed turnover, supply, price pressure) over raw exit percentages. Do not surface suspicious 100% exit rates as if they were sales rates.
- If VERIFIED_EVIDENCE says supports_time_trend=false and the user asks whether the market has slowed, accelerated, increased/decreased recently, or changed over time, SAY that this evidence cannot establish the time trend. Do not answer yes/no. You may still describe current relative segment signals and explain what time-series evidence would be needed.
- If supports_time_trend=true only because time_trend_scope is observed_gallery_listing_activity_not_confirmed_sales, you MAY compare the dealer's observed listing-exit activity with the wider gallery market over the supplied periods, but NEVER call those exits sales. Use the business_context diagnosis and actual repricing/aging candidates for corrective action.
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
    if job in {Job.PRICE_STOCK.value,Job.MOVE_AGING_STOCK.value,Job.ACQUIRE_STOCK.value,Job.ANALYZE_BUSINESS.value,Job.EVALUATE_TRADE_IN.value,Job.PROMOTE_STOCK.value}: return job
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
