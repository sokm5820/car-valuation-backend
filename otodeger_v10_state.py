"""OtoDeğer V10 authoritative conversation-state engine.

Phase 1 of the V10 decision-agent architecture.

Design goals
------------
* Server-owned structured state; chat prose is not authoritative memory.
* Redis-first storage with optimistic concurrency and TTL.
* Thread-safe in-memory backend for local/private development only.
* Strict job/action/transition vocabularies.
* Stable object IDs for models, listings, inventory items, and companies.
* Deterministic state transitions: the LLM proposes a TurnPlan; this module
  validates and applies it.
* No dependency on valuation or market calculation code.

Production configuration
------------------------
REDIS_URL=<managed redis connection URL>
V10_STATE_BACKEND=redis
V10_REQUIRE_REDIS=true
V10_STATE_TTL_SECONDS=2592000       # 30 days
V10_STATE_NAMESPACE=otodeger:v10

For local/private development, omit REDIS_URL and use:
V10_STATE_BACKEND=memory
V10_REQUIRE_REDIS=false
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "10.0"
DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_OBJECTS = 250
MAX_SHORTLIST = 30
MAX_FOCUS_IDS = 20
MAX_STATE_BYTES = 256 * 1024


class V10StateError(RuntimeError):
    """Base exception for V10 state errors."""


class InvalidState(V10StateError):
    pass


class InvalidTurnPlan(V10StateError):
    pass


class StateConflict(V10StateError):
    """Raised when another request updated the same conversation first."""


class StateNotFound(V10StateError):
    pass


class StorageUnavailable(V10StateError):
    pass


class Audience(str, Enum):
    PERSONAL = "PERSONAL"
    BUSINESS = "BUSINESS"


class Job(str, Enum):
    FIND_A_CAR = "FIND_A_CAR"
    EVALUATE_PURCHASE = "EVALUATE_PURCHASE"
    EVALUATE_SALE = "EVALUATE_SALE"
    COMPARE_CARS = "COMPARE_CARS"
    UNDERSTAND_MARKET = "UNDERSTAND_MARKET"
    ACQUIRE_STOCK = "ACQUIRE_STOCK"
    PRICE_STOCK = "PRICE_STOCK"
    MOVE_AGING_STOCK = "MOVE_AGING_STOCK"
    ANALYZE_BUSINESS = "ANALYZE_BUSINESS"


PERSONAL_JOBS = {
    Job.FIND_A_CAR.value,
    Job.EVALUATE_PURCHASE.value,
    Job.EVALUATE_SALE.value,
    Job.COMPARE_CARS.value,
    Job.UNDERSTAND_MARKET.value,
}
BUSINESS_JOBS = {
    Job.ACQUIRE_STOCK.value,
    Job.PRICE_STOCK.value,
    Job.MOVE_AGING_STOCK.value,
    Job.ANALYZE_BUSINESS.value,
    Job.UNDERSTAND_MARKET.value,
}


class Transition(str, Enum):
    CONTINUE = "CONTINUE"
    REFINE = "REFINE"
    SWITCH_SUBTASK = "SWITCH_SUBTASK"
    START_NEW_GOAL = "START_NEW_GOAL"
    CORRECT = "CORRECT"


class Action(str, Enum):
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    SEARCH_VEHICLES = "SEARCH_VEHICLES"
    SHOW_LISTINGS = "SHOW_LISTINGS"
    COMPARE_VEHICLES = "COMPARE_VEHICLES"
    EVALUATE_PURCHASE = "EVALUATE_PURCHASE"
    EVALUATE_SALE = "EVALUATE_SALE"
    VALUE_VEHICLE = "VALUE_VEHICLE"
    ANALYZE_MARKET = "ANALYZE_MARKET"
    ANALYZE_STOCK_PRICES = "ANALYZE_STOCK_PRICES"
    ANALYZE_AGING_STOCK = "ANALYZE_AGING_STOCK"
    RECOMMEND_ACQUISITIONS = "RECOMMEND_ACQUISITIONS"
    ANALYZE_BUSINESS_PERIOD = "ANALYZE_BUSINESS_PERIOD"
    EXPLAIN_RESULT = "EXPLAIN_RESULT"


class ObjectType(str, Enum):
    MODEL = "MODEL"
    LISTING = "LISTING"
    OWNED_VEHICLE = "OWNED_VEHICLE"
    STOCK_ITEM = "STOCK_ITEM"
    COMPANY = "COMPANY"
    MARKET_SEGMENT = "MARKET_SEGMENT"


ALLOWED_CONSTRAINTS = {
    "budget_min",
    "budget_max",
    "vehicle_type",
    "brands",
    "models",
    "min_year",
    "max_year",
    "max_km",
    "min_km",
    "transmission",
    "seller_type",
    "location",
    "fuel_type",
    "exclude_locations",
    "exclude_fuels",
    "category",
    "company",
    "period_start",
    "period_end",
    "asking_price",
    "offer_price",
    "acquisition_price",
    "desired_sale_price",
    "currency",
}

ALLOWED_PREFERENCES = {
    "economy",
    "reliability",
    "performance",
    "luxury",
    "comfort",
    "practicality",
    "family",
    "commute",
    "size",
    "resale",
    "low_mileage",
    "newer",
    "avoid_fuel",
}


@dataclass(frozen=True)
class StoredState:
    state: Dict[str, Any]
    revision: int


def _utc_epoch() -> int:
    return int(time.time())


def _clean_string(value: Any, max_len: int = 500) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value[:max_len]


def _unique_strings(values: Any, max_items: int = 30) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        raise InvalidTurnPlan("Expected a list of strings")
    out: List[str] = []
    seen = set()
    for value in values:
        text = _clean_string(value, 120)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _safe_number(value: Any, *, integer: bool = False) -> Optional[float]:
    from otodeger_v11_contract import ContractError, parse_number
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise InvalidTurnPlan("Boolean is not a numeric constraint")
    try:
        number = parse_number(value)
    except (TypeError, ValueError, ContractError) as exc:
        raise InvalidTurnPlan(f"Invalid numeric value: {value!r}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise InvalidTurnPlan("Numeric constraint must be finite")
    if integer and not number.is_integer():
        raise InvalidTurnPlan("Year and mileage constraints must be whole numbers")
    return int(number) if integer else number


def _normalize_constraints(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise InvalidTurnPlan("constraints must be an object")
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_CONSTRAINTS:
            continue
        if key in {"brands", "models", "exclude_locations", "exclude_fuels"}:
            out[key] = _unique_strings(value)
        elif key in {"budget_min", "budget_max", "asking_price", "offer_price", "acquisition_price", "desired_sale_price"}:
            number = _safe_number(value)
            out[key] = None if number is None else round(number, 2)
        elif key in {"min_year", "max_year", "max_km", "min_km"}:
            out[key] = _safe_number(value, integer=True)
        else:
            out[key] = _clean_string(value, 160)
    return out


def _normalize_preferences(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise InvalidTurnPlan("preferences must be an object")
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_PREFERENCES:
            continue
        if value is None:
            out[key] = None
        elif isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = max(-1.0, min(1.0, float(value)))
        else:
            out[key] = _clean_string(value, 80)
    return out


def _object_fingerprint(object_type: str, payload: Mapping[str, Any]) -> str:
    """Stable identity based only on immutable/canonical identity fields."""
    fields_by_type = {
        ObjectType.MODEL.value: ("brand", "model"),
        ObjectType.LISTING.value: ("link", "listing_id", "brand", "model", "year", "price", "km"),
        ObjectType.OWNED_VEHICLE.value: ("vehicle_id", "brand", "model", "year"),
        ObjectType.STOCK_ITEM.value: ("link", "listing_id", "company", "brand", "model", "year"),
        ObjectType.COMPANY.value: ("company_id", "name"),
        ObjectType.MARKET_SEGMENT.value: ("name", "vehicle_type", "brand", "model", "category"),
    }
    identity_fields = fields_by_type.get(object_type)
    if not identity_fields:
        raise InvalidState(f"Unsupported object type {object_type!r}")
    identity = {k: payload.get(k) for k in identity_fields if payload.get(k) not in (None, "")}
    if not identity:
        raise InvalidState(f"Object of type {object_type} has no usable identity")
    material = json.dumps([object_type, identity], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def make_object_id(object_type: str, payload: Mapping[str, Any]) -> str:
    prefix = {
        ObjectType.MODEL.value: "model",
        ObjectType.LISTING.value: "listing",
        ObjectType.OWNED_VEHICLE.value: "owned",
        ObjectType.STOCK_ITEM.value: "stock",
        ObjectType.COMPANY.value: "company",
        ObjectType.MARKET_SEGMENT.value: "segment",
    }.get(object_type, "obj")
    return f"{prefix}_{_object_fingerprint(object_type, payload)}"


def new_state(
    conversation_id: Optional[str] = None,
    *,
    audience: str = Audience.PERSONAL.value,
    language: str = "EN",
) -> Dict[str, Any]:
    now = _utc_epoch()
    audience = str(audience or Audience.PERSONAL.value).upper()
    if audience not in {x.value for x in Audience}:
        audience = Audience.PERSONAL.value
    return {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": conversation_id or str(uuid.uuid4()),
        "revision": 0,
        "audience": audience,
        "language": str(language or "EN").upper()[:5],
        "job": None,
        "goal": {"summary": None, "status": "IDLE"},
        "constraints": {},
        "preferences": {},
        "objects": {},
        "shortlist": [],
        "focus": {"object_ids": []},
        "last_result": None,
        "offered_action": None,
        "awaiting": None,
        "created_at": now,
        "updated_at": now,
    }


def validate_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise InvalidState("State must be a JSON object")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise InvalidState("Unsupported state schema version")
    if not _clean_string(state.get("conversation_id"), 200):
        raise InvalidState("conversation_id is required")
    audience = state.get("audience")
    if audience not in {x.value for x in Audience}:
        raise InvalidState("Invalid audience")
    job = state.get("job")
    if job is not None:
        if job not in {x.value for x in Job}:
            raise InvalidState("Invalid job")
        if audience == Audience.PERSONAL.value and job not in PERSONAL_JOBS:
            raise InvalidState("Business-only job cannot run in Personal state")
        if audience == Audience.BUSINESS.value and job not in BUSINESS_JOBS:
            raise InvalidState("Personal-only job cannot run in Business state")
    constraints = state.get("constraints") or {}
    if not isinstance(constraints, Mapping):
        raise InvalidState("constraints must be an object")
    if set(constraints) - ALLOWED_CONSTRAINTS:
        raise InvalidState("State contains unsupported constraints")
    prefs = state.get("preferences") or {}
    if not isinstance(prefs, Mapping):
        raise InvalidState("preferences must be an object")
    if set(prefs) - ALLOWED_PREFERENCES:
        raise InvalidState("State contains unsupported preferences")
    objects = state.get("objects") or {}
    if not isinstance(objects, Mapping) or len(objects) > MAX_OBJECTS:
        raise InvalidState("Invalid object registry")
    for object_id, obj in objects.items():
        if not isinstance(obj, Mapping):
            raise InvalidState(f"Invalid object {object_id}")
        if obj.get("id") != object_id:
            raise InvalidState(f"Object id mismatch for {object_id}")
        if obj.get("type") not in {x.value for x in ObjectType}:
            raise InvalidState(f"Invalid object type for {object_id}")
    shortlist = state.get("shortlist") or []
    if not isinstance(shortlist, list) or len(shortlist) > MAX_SHORTLIST:
        raise InvalidState("Invalid shortlist")
    focus_ids = ((state.get("focus") or {}).get("object_ids") or [])
    if not isinstance(focus_ids, list) or len(focus_ids) > MAX_FOCUS_IDS:
        raise InvalidState("Invalid focus")
    all_refs = list(shortlist) + list(focus_ids)
    for object_id in all_refs:
        if object_id not in objects:
            raise InvalidState(f"Dangling object reference: {object_id}")
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_STATE_BYTES:
        raise InvalidState("Conversation state exceeds maximum size")


def register_object(state: Dict[str, Any], object_type: str, payload: Mapping[str, Any]) -> str:
    if object_type not in {x.value for x in ObjectType}:
        raise InvalidTurnPlan(f"Unsupported object type: {object_type}")
    if not isinstance(payload, Mapping):
        raise InvalidTurnPlan("Object payload must be an object")
    clean_payload = {str(k)[:80]: v for k, v in payload.items() if v is not None}
    object_id = make_object_id(object_type, clean_payload)
    obj = {"id": object_id, "type": object_type, **clean_payload}
    objects = state.setdefault("objects", {})
    objects[object_id] = obj
    # Bound registry growth while never deleting currently referenced objects.
    if len(objects) > MAX_OBJECTS:
        pinned = set(state.get("shortlist") or []) | set((state.get("focus") or {}).get("object_ids") or [])
        removable = [oid for oid in objects if oid not in pinned and oid != object_id]
        for oid in removable[: max(0, len(objects) - MAX_OBJECTS)]:
            objects.pop(oid, None)
    return object_id


def set_shortlist(state: Dict[str, Any], object_ids: Sequence[str]) -> None:
    unique = []
    seen = set()
    for oid in object_ids:
        if oid in seen:
            continue
        if oid not in (state.get("objects") or {}):
            raise InvalidTurnPlan(f"Unknown shortlist object: {oid}")
        seen.add(oid)
        unique.append(oid)
        if len(unique) >= MAX_SHORTLIST:
            break
    state["shortlist"] = unique


def set_focus(state: Dict[str, Any], object_ids: Sequence[str]) -> None:
    unique = []
    seen = set()
    for oid in object_ids:
        if oid in seen:
            continue
        if oid not in (state.get("objects") or {}):
            raise InvalidTurnPlan(f"Unknown focus object: {oid}")
        seen.add(oid)
        unique.append(oid)
        if len(unique) >= MAX_FOCUS_IDS:
            break
    state["focus"] = {"object_ids": unique}


def _reset_goal_context(state: Dict[str, Any]) -> None:
    """Clear task-specific state but retain session identity/audience/language."""
    state["job"] = None
    state["goal"] = {"summary": None, "status": "IDLE"}
    state["constraints"] = {}
    state["preferences"] = {}
    state["objects"] = {}
    state["shortlist"] = []
    state["focus"] = {"object_ids": []}
    state["last_result"] = None
    state["offered_action"] = None
    state["awaiting"] = None


def validate_turn_plan(plan: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    from otodeger_v11_contract import ContractError, clarification
    if not isinstance(plan, Mapping):
        raise InvalidTurnPlan("Turn plan must be a JSON object")
    transition = str(plan.get("transition") or Transition.CONTINUE.value).upper()
    action = str(plan.get("action") or "").upper()
    if transition not in {x.value for x in Transition}:
        raise InvalidTurnPlan(f"Unknown transition: {transition}")
    if action not in {x.value for x in Action}:
        raise InvalidTurnPlan(f"Unknown action: {action}")

    job = plan.get("job")
    if job is not None:
        job = str(job).upper()
        if job not in {x.value for x in Job}:
            raise InvalidTurnPlan(f"Unknown job: {job}")
        audience = state.get("audience", Audience.PERSONAL.value)
        allowed = PERSONAL_JOBS if audience == Audience.PERSONAL.value else BUSINESS_JOBS
        if job not in allowed:
            raise InvalidTurnPlan(f"Job {job} is not permitted for {audience}")

    target_ids = _unique_strings(plan.get("target_ids") or [], max_items=MAX_FOCUS_IDS)
    known_ids = set((state.get("objects") or {}).keys())
    # Plans may target objects introduced in the same turn; their temporary aliases
    # are resolved after object registration in apply_turn_plan.
    object_specs = plan.get("objects") or []
    if not isinstance(object_specs, list):
        raise InvalidTurnPlan("objects must be an array")
    if len(object_specs) > 30:
        raise InvalidTurnPlan("Too many objects in one turn")

    normalized_objects = []
    aliases = set()
    for spec in object_specs:
        if not isinstance(spec, Mapping):
            raise InvalidTurnPlan("Each object spec must be an object")
        typ = str(spec.get("type") or "").upper()
        payload = spec.get("payload") or {}
        alias = _clean_string(spec.get("alias"), 80)
        if typ not in {x.value for x in ObjectType}:
            raise InvalidTurnPlan(f"Unknown object type: {typ}")
        if not isinstance(payload, Mapping):
            raise InvalidTurnPlan("Object payload must be an object")
        if alias:
            if alias in aliases:
                raise InvalidTurnPlan(f"Duplicate object alias: {alias}")
            aliases.add(alias)
        normalized_objects.append({"type": typ, "payload": dict(payload), "alias": alias})

    for target in target_ids:
        if target not in known_ids and target not in aliases:
            raise InvalidTurnPlan(f"Unknown target object: {target}")

    clear_constraints = _unique_strings(plan.get("clear_constraints") or [], max_items=30)
    clear_preferences = _unique_strings(plan.get("clear_preferences") or [], max_items=30)
    if any(k not in ALLOWED_CONSTRAINTS for k in clear_constraints):
        raise InvalidTurnPlan("Unsupported clear_constraints key")
    if any(k not in ALLOWED_PREFERENCES for k in clear_preferences):
        raise InvalidTurnPlan("Unsupported clear_preferences key")

    try:
        awaiting = clarification(plan.get("awaiting"))
    except ContractError as exc:
        raise InvalidTurnPlan(str(exc)) from exc
    return {
        "transition": transition,
        "action": action,
        "job": job,
        "goal_summary": _clean_string(plan.get("goal_summary"), 500),
        "constraints_delta": _normalize_constraints(plan.get("constraints_delta")),
        "clear_constraints": clear_constraints,
        "preferences_delta": _normalize_preferences(plan.get("preferences_delta")),
        "clear_preferences": clear_preferences,
        "objects": normalized_objects,
        "target_ids": target_ids,
        "shortlist_ids": _unique_strings(plan.get("shortlist_ids") or [], max_items=MAX_SHORTLIST),
        "awaiting": awaiting,
        "offered_action": copy.deepcopy(plan.get("offered_action")),
        "result": copy.deepcopy(plan.get("result")),
    }


def _validate_numeric_consistency(constraints: Mapping[str, Any]) -> None:
    for key in ("budget_min", "budget_max", "asking_price", "offer_price", "acquisition_price", "desired_sale_price", "min_km", "max_km"):
        value = constraints.get(key)
        if value is not None and (not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise InvalidTurnPlan(f"{key} must be a finite nonnegative number")
    for key in ("min_year", "max_year"):
        value = constraints.get(key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 1900 <= value <= 2100):
            raise InvalidTurnPlan(f"{key} must be a model year between 1900 and 2100")
    bmin, bmax = constraints.get("budget_min"), constraints.get("budget_max")
    if bmin is not None and bmax is not None and float(bmin) > float(bmax):
        raise InvalidTurnPlan("budget_min cannot exceed budget_max")
    ymin, ymax = constraints.get("min_year"), constraints.get("max_year")
    if ymin is not None and ymax is not None and int(ymin) > int(ymax):
        raise InvalidTurnPlan("min_year cannot exceed max_year")
    kmin, kmax = constraints.get("min_km"), constraints.get("max_km")
    if kmin is not None and kmax is not None and int(kmin) > int(kmax):
        raise InvalidTurnPlan("min_km cannot exceed max_km")


def apply_turn_plan(state: Mapping[str, Any], plan: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate and deterministically apply a semantic TurnPlan.

    Returns (new_state, resolved_plan). The latter contains concrete object IDs
    after same-turn aliases have been resolved and is what the action planner
    should consume.
    """
    validate_state(state)
    normalized = validate_turn_plan(plan, state)
    next_state = copy.deepcopy(dict(state))

    if normalized["transition"] == Transition.START_NEW_GOAL.value:
        _reset_goal_context(next_state)

    if normalized["job"] is not None:
        next_state["job"] = normalized["job"]

    if normalized["goal_summary"]:
        next_state["goal"] = {
            "summary": normalized["goal_summary"],
            "status": "ACTIVE",
        }
    elif next_state.get("job") and (next_state.get("goal") or {}).get("status") == "IDLE":
        next_state["goal"] = {"summary": None, "status": "ACTIVE"}

    constraints = dict(next_state.get("constraints") or {})
    for key in normalized["clear_constraints"]:
        constraints.pop(key, None)
    for key, value in normalized["constraints_delta"].items():
        if value is None:
            constraints.pop(key, None)
        else:
            constraints[key] = value
    _validate_numeric_consistency(constraints)
    next_state["constraints"] = constraints

    preferences = dict(next_state.get("preferences") or {})
    for key in normalized["clear_preferences"]:
        preferences.pop(key, None)
    for key, value in normalized["preferences_delta"].items():
        if value is None:
            preferences.pop(key, None)
        else:
            preferences[key] = value
    next_state["preferences"] = preferences

    alias_map: Dict[str, str] = {}
    for spec in normalized["objects"]:
        oid = register_object(next_state, spec["type"], spec["payload"])
        if spec["alias"]:
            alias_map[spec["alias"]] = oid

    def resolve_ids(values: Sequence[str]) -> List[str]:
        out = []
        for value in values:
            oid = alias_map.get(value, value)
            if oid not in next_state.get("objects", {}):
                raise InvalidTurnPlan(f"Unknown resolved object: {value}")
            if oid not in out:
                out.append(oid)
        return out

    resolved_targets = resolve_ids(normalized["target_ids"])
    resolved_shortlist = resolve_ids(normalized["shortlist_ids"])
    if resolved_shortlist:
        set_shortlist(next_state, resolved_shortlist)
    if resolved_targets:
        set_focus(next_state, resolved_targets)

    next_state["awaiting"] = normalized["awaiting"]

    offered = normalized["offered_action"]
    if offered is not None:
        if not isinstance(offered, Mapping):
            raise InvalidTurnPlan("offered_action must be an object")
        offered_type = str(offered.get("type") or "").upper()
        if offered_type and offered_type not in {x.value for x in Action}:
            raise InvalidTurnPlan(f"Unknown offered action: {offered_type}")
        offered_targets = _unique_strings(offered.get("target_ids") or [], max_items=MAX_FOCUS_IDS)
        resolved_offered_targets = resolve_ids(offered_targets)
        next_state["offered_action"] = {
            "type": offered_type,
            "target_ids": resolved_offered_targets,
            "metadata": copy.deepcopy(offered.get("metadata") or {}),
        }
    elif normalized["transition"] == Transition.START_NEW_GOAL.value:
        next_state["offered_action"] = None

    if normalized["result"] is not None:
        result = normalized["result"]
        if not isinstance(result, Mapping):
            raise InvalidTurnPlan("result must be an object")
        result_targets = _unique_strings(result.get("target_ids") or [], max_items=MAX_FOCUS_IDS)
        next_state["last_result"] = {
            "type": _clean_string(result.get("type"), 100),
            "target_ids": resolve_ids(result_targets),
            "metadata": copy.deepcopy(result.get("metadata") or {}),
        }

    next_state["updated_at"] = _utc_epoch()
    # revision increments only when persisted; storage controls concurrency.
    validate_state(next_state)

    resolved_plan = copy.deepcopy(normalized)
    resolved_plan["target_ids"] = resolved_targets
    resolved_plan["shortlist_ids"] = resolved_shortlist
    resolved_plan["alias_map"] = alias_map
    return next_state, resolved_plan


class StateStore:
    """Storage contract. Implementations must provide optimistic concurrency."""

    def load(self, conversation_id: str) -> Optional[StoredState]:
        raise NotImplementedError

    def create(self, state: Mapping[str, Any]) -> StoredState:
        raise NotImplementedError

    def save(self, state: Mapping[str, Any], *, expected_revision: int) -> StoredState:
        raise NotImplementedError

    def delete(self, conversation_id: str) -> None:
        raise NotImplementedError


class MemoryStateStore(StateStore):
    """Thread-safe development store. Not suitable for paid production traffic."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = threading.RLock()
        self._items: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _purge_if_expired(self, conversation_id: str) -> None:
        item = self._items.get(conversation_id)
        if not item:
            return
        expires_at, _ = item
        if expires_at <= time.time():
            self._items.pop(conversation_id, None)

    def load(self, conversation_id: str) -> Optional[StoredState]:
        with self._lock:
            self._purge_if_expired(conversation_id)
            item = self._items.get(conversation_id)
            if not item:
                return None
            _, state = item
            return StoredState(copy.deepcopy(state), int(state.get("revision", 0)))

    def create(self, state: Mapping[str, Any]) -> StoredState:
        validate_state(state)
        conversation_id = str(state["conversation_id"])
        with self._lock:
            self._purge_if_expired(conversation_id)
            if conversation_id in self._items:
                raise StateConflict("Conversation already exists")
            stored = copy.deepcopy(dict(state))
            stored["revision"] = 1
            stored["updated_at"] = _utc_epoch()
            validate_state(stored)
            self._items[conversation_id] = (time.time() + self.ttl_seconds, stored)
            return StoredState(copy.deepcopy(stored), 1)

    def save(self, state: Mapping[str, Any], *, expected_revision: int) -> StoredState:
        validate_state(state)
        conversation_id = str(state["conversation_id"])
        with self._lock:
            self._purge_if_expired(conversation_id)
            item = self._items.get(conversation_id)
            if not item:
                raise StateNotFound(conversation_id)
            _, current = item
            current_revision = int(current.get("revision", 0))
            if current_revision != int(expected_revision):
                raise StateConflict(
                    f"Expected revision {expected_revision}, found {current_revision}"
                )
            stored = copy.deepcopy(dict(state))
            stored["revision"] = current_revision + 1
            stored["updated_at"] = _utc_epoch()
            validate_state(stored)
            self._items[conversation_id] = (time.time() + self.ttl_seconds, stored)
            return StoredState(copy.deepcopy(stored), stored["revision"])

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._items.pop(conversation_id, None)


class RedisStateStore(StateStore):
    """Redis-backed production store with atomic compare-and-set writes."""

    # Lua: compare revision and update atomically while refreshing TTL.
    _CAS_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then return -1 end
local current = cjson.decode(raw)
if tonumber(current['revision'] or 0) ~= tonumber(ARGV[1]) then return -2 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
return 1
"""

    def __init__(self, redis_url: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS, namespace: str = "otodeger:v10"):
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise StorageUnavailable(
                "Redis storage requested but the 'redis' package is not installed"
            ) from exc
        if not redis_url:
            raise StorageUnavailable("REDIS_URL is required")
        self.redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.namespace = (namespace or "otodeger:v10").strip(":")
        self._cas = self.redis.register_script(self._CAS_SCRIPT)

    def _key(self, conversation_id: str) -> str:
        safe = str(conversation_id).strip()
        if not safe or len(safe) > 200:
            raise InvalidState("Invalid conversation_id")
        return f"{self.namespace}:conversation:{safe}"

    @staticmethod
    def _decode(raw: str) -> Dict[str, Any]:
        try:
            state = json.loads(raw)
        except Exception as exc:
            raise StorageUnavailable("Stored conversation state is invalid JSON") from exc
        validate_state(state)
        return state

    def ping(self) -> bool:
        try:
            return bool(self.redis.ping())
        except Exception as exc:
            raise StorageUnavailable("Redis is unavailable") from exc

    def load(self, conversation_id: str) -> Optional[StoredState]:
        try:
            raw = self.redis.get(self._key(conversation_id))
        except Exception as exc:
            raise StorageUnavailable("Failed to load conversation state") from exc
        if raw is None:
            return None
        state = self._decode(raw)
        return StoredState(state, int(state.get("revision", 0)))

    def create(self, state: Mapping[str, Any]) -> StoredState:
        validate_state(state)
        stored = copy.deepcopy(dict(state))
        stored["revision"] = 1
        stored["updated_at"] = _utc_epoch()
        validate_state(stored)
        payload = json.dumps(stored, ensure_ascii=False, separators=(",", ":"))
        try:
            created = self.redis.set(
                self._key(str(stored["conversation_id"])),
                payload,
                ex=self.ttl_seconds,
                nx=True,
            )
        except Exception as exc:
            raise StorageUnavailable("Failed to create conversation state") from exc
        if not created:
            raise StateConflict("Conversation already exists")
        return StoredState(stored, 1)

    def save(self, state: Mapping[str, Any], *, expected_revision: int) -> StoredState:
        validate_state(state)
        stored = copy.deepcopy(dict(state))
        stored["revision"] = int(expected_revision) + 1
        stored["updated_at"] = _utc_epoch()
        validate_state(stored)
        payload = json.dumps(stored, ensure_ascii=False, separators=(",", ":"))
        try:
            result = int(
                self._cas(
                    keys=[self._key(str(stored["conversation_id"]))],
                    args=[int(expected_revision), payload, self.ttl_seconds],
                )
            )
        except Exception as exc:
            raise StorageUnavailable("Failed to save conversation state") from exc
        if result == -1:
            raise StateNotFound(str(stored["conversation_id"]))
        if result == -2:
            raise StateConflict("Conversation state changed concurrently")
        if result != 1:
            raise StorageUnavailable(f"Unexpected Redis CAS result: {result}")
        return StoredState(stored, stored["revision"])

    def delete(self, conversation_id: str) -> None:
        try:
            self.redis.delete(self._key(conversation_id))
        except Exception as exc:
            raise StorageUnavailable("Failed to delete conversation state") from exc


class ConversationStateService:
    """High-level state API used by the future V10 assistant endpoint."""

    def __init__(self, store: StateStore):
        self.store = store

    def get_or_create(
        self,
        conversation_id: Optional[str],
        *,
        audience: str = Audience.PERSONAL.value,
        language: str = "EN",
    ) -> StoredState:
        if conversation_id:
            loaded = self.store.load(conversation_id)
            if loaded is not None:
                # Client cannot silently switch the authoritative audience.
                return loaded
        state = new_state(conversation_id, audience=audience, language=language)
        return self.store.create(state)

    def apply_plan(
        self,
        stored: StoredState,
        plan: Mapping[str, Any],
    ) -> Tuple[StoredState, Dict[str, Any]]:
        next_state, resolved_plan = apply_turn_plan(stored.state, plan)
        saved = self.store.save(next_state, expected_revision=stored.revision)
        return saved, resolved_plan


def build_state_store_from_env() -> StateStore:
    backend = str(os.getenv("V10_STATE_BACKEND") or "auto").strip().lower()
    require_redis = str(os.getenv("V10_REQUIRE_REDIS") or "false").strip().lower() in {"1", "true", "yes", "on"}
    redis_url = str(os.getenv("REDIS_URL") or "").strip()
    ttl = int(os.getenv("V10_STATE_TTL_SECONDS") or DEFAULT_TTL_SECONDS)
    namespace = str(os.getenv("V10_STATE_NAMESPACE") or "otodeger:v10").strip()

    if backend not in {"auto", "redis", "memory"}:
        raise StorageUnavailable(f"Unsupported V10_STATE_BACKEND: {backend}")

    wants_redis = backend == "redis" or (backend == "auto" and bool(redis_url))
    if wants_redis:
        return RedisStateStore(redis_url, ttl_seconds=ttl, namespace=namespace)

    if require_redis:
        raise StorageUnavailable(
            "V10_REQUIRE_REDIS=true but no usable Redis configuration was supplied"
        )

    return MemoryStateStore(ttl_seconds=ttl)


# Singleton is deliberately lazy. Importing the module must not make network calls.
_state_service_lock = threading.Lock()
_state_service: Optional[ConversationStateService] = None


def get_state_service() -> ConversationStateService:
    global _state_service
    if _state_service is None:
        with _state_service_lock:
            if _state_service is None:
                _state_service = ConversationStateService(build_state_store_from_env())
    return _state_service


__all__ = [
    "SCHEMA_VERSION",
    "Audience",
    "Job",
    "Transition",
    "Action",
    "ObjectType",
    "StoredState",
    "V10StateError",
    "InvalidState",
    "InvalidTurnPlan",
    "StateConflict",
    "StateNotFound",
    "StorageUnavailable",
    "new_state",
    "validate_state",
    "validate_turn_plan",
    "register_object",
    "set_shortlist",
    "set_focus",
    "apply_turn_plan",
    "MemoryStateStore",
    "RedisStateStore",
    "ConversationStateService",
    "build_state_store_from_env",
    "get_state_service",
]
