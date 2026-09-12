import inspect
import pandas as pd
import otodeger_v10_agent as agent
from otodeger_v10_state import Action, Audience, Job


def _state():
    return {
        "audience": Audience.PERSONAL.value,
        "language": "EN",
        "job": None,
        "objects": {},
        "constraints": {},
        "preferences": {},
        "shortlist": [],
        "focus": {"object_ids": []},
    }


def _market():
    return pd.DataFrame([
        {"Brand": "Nissan", "Model": "March", "Category": "1.2"},
        {"Brand": "Nissan", "Model": "Note", "Category": "1.2"},
        {"Brand": "Toyota", "Model": "Yaris", "Category": "1.5"},
    ])


def _host(resolver):
    return {
        "resolve_market_vehicle_mentions": resolver,
        "_attach_explicit_years_to_vehicle_targets": lambda message, targets: targets,
        "_extract_explicit_market_brands": lambda message: ["Nissan"] if "Nissan" in message else [],
        "market_df": _market(),
    }


def test_purchase_route_survives_primary_resolver_returning_nothing():
    plan = agent._deterministic_purchase_price_plan(
        "Is £5,000 a good price for a 2007 Nissan March 1.2L?",
        _state(),
        _host(lambda message: []),
    )
    assert plan["action"] == Action.EVALUATE_PURCHASE.value
    assert plan["job"] == Job.EVALUATE_PURCHASE.value
    subject = plan["objects"][0]["payload"]
    assert subject["brand"] == "Nissan"
    assert subject["model"] == "March"
    assert subject["year"] == 2007
    assert subject["category"] == "1.2"
    assert subject["asking_price"] == 5000


def test_purchase_route_survives_ambiguous_primary_resolver_output():
    resolver_output = [
        {"brand": "Nissan", "model": "March", "category": None},
        {"brand": "Nissan", "model": "Note", "category": None},
    ]
    plan = agent._deterministic_purchase_price_plan(
        "Is £5,000 a good price for a 2007 Nissan March 1.2L?",
        _state(),
        _host(lambda message: resolver_output),
    )
    assert plan["action"] == Action.EVALUATE_PURCHASE.value
    assert plan["objects"][0]["payload"]["model"] == "March"


def test_purchase_signal_can_never_degrade_to_search_when_vehicle_unresolved():
    host = _host(lambda message: [])
    host["market_df"] = pd.DataFrame(columns=["Brand", "Model", "Category"])
    plan = agent._deterministic_purchase_price_plan(
        "Is £5,000 a good price for a 2007 Nissan March 1.2L?",
        _state(),
        host,
    )
    assert plan["job"] == Job.EVALUATE_PURCHASE.value
    assert plan["action"] == Action.ASK_CLARIFICATION.value
    assert plan["action"] != Action.SEARCH_VEHICLES.value


def test_seller_language_is_not_forced_into_purchase_mode():
    plan = agent._deterministic_purchase_price_plan(
        "Is £14,000 a good price to sell my 2017 BMW 118i for?",
        _state(),
        _host(lambda message: [{"brand": "BMW", "model": "1 Serisi", "category": "118i"}]),
    )
    assert plan is None


def test_request_boundary_precedes_semantic_planner():
    source = inspect.getsource(agent.handle_v10_request)
    forced = source.index("raw_plan=_deterministic_purchase_price_plan")
    semantic = source.index("raw_plan=_semantic_plan", forced)
    assert forced < semantic
