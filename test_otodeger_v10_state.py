import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from otodeger_v10_state import (
    Action,
    Audience,
    ConversationStateService,
    InvalidTurnPlan,
    Job,
    MemoryStateStore,
    ObjectType,
    StateConflict,
    Transition,
    apply_turn_plan,
    new_state,
    register_object,
    set_focus,
    set_shortlist,
    validate_state,
)


class V10StateTests(unittest.TestCase):
    def make_service(self):
        return ConversationStateService(MemoryStateStore(ttl_seconds=3600))

    def test_new_state_is_valid(self):
        state = new_state("c1")
        validate_state(state)
        self.assertEqual(state["audience"], "PERSONAL")
        self.assertIsNone(state["job"])

    def test_personal_refinement_preserves_prior_constraints(self):
        state = new_state("c2")
        state, _ = apply_turn_plan(state, {
            "transition": "START_NEW_GOAL",
            "job": "FIND_A_CAR",
            "action": "SEARCH_VEHICLES",
            "goal_summary": "Find a car within £15,000",
            "constraints_delta": {"budget_max": 15000},
        })
        state, _ = apply_turn_plan(state, {
            "transition": "REFINE",
            "job": "FIND_A_CAR",
            "action": "SEARCH_VEHICLES",
            "constraints_delta": {"vehicle_type": "SUV"},
        })
        state, _ = apply_turn_plan(state, {
            "transition": "REFINE",
            "job": "FIND_A_CAR",
            "action": "SEARCH_VEHICLES",
            "constraints_delta": {"brands": ["BMW", "Mercedes-Benz"]},
        })
        self.assertEqual(state["constraints"]["budget_max"], 15000)
        self.assertEqual(state["constraints"]["vehicle_type"], "SUV")
        self.assertEqual(state["constraints"]["brands"], ["BMW", "Mercedes-Benz"])

    def test_objects_are_stable_and_deduplicated(self):
        state = new_state("c3")
        a = register_object(state, "MODEL", {"brand": "BMW", "model": "X1"})
        b = register_object(state, "MODEL", {"brand": "BMW", "model": "X1", "note": "same model"})
        self.assertEqual(a, b)
        self.assertEqual(len(state["objects"]), 1)

    def test_same_turn_aliases_resolve_to_objects(self):
        state = new_state("c4")
        state, resolved = apply_turn_plan(state, {
            "transition": "START_NEW_GOAL",
            "job": "COMPARE_CARS",
            "action": "COMPARE_VEHICLES",
            "objects": [
                {"alias": "x1", "type": "MODEL", "payload": {"brand": "BMW", "model": "X1"}},
                {"alias": "gla", "type": "MODEL", "payload": {"brand": "Mercedes-Benz", "model": "GLA"}},
            ],
            "target_ids": ["x1", "gla"],
            "shortlist_ids": ["x1", "gla"],
            "offered_action": {"type": "SHOW_LISTINGS", "target_ids": ["x1", "gla"]},
            "result": {"type": "VEHICLE_COMPARISON", "target_ids": ["x1", "gla"]},
        })
        self.assertEqual(len(state["focus"]["object_ids"]), 2)
        self.assertEqual(state["focus"]["object_ids"], state["shortlist"])
        self.assertEqual(state["offered_action"]["target_ids"], state["focus"]["object_ids"])
        self.assertEqual(resolved["target_ids"], state["focus"]["object_ids"])

    def test_yes_can_execute_structured_offered_action_without_prose_recovery(self):
        state = new_state("c5")
        x1 = register_object(state, "MODEL", {"brand": "BMW", "model": "X1"})
        gla = register_object(state, "MODEL", {"brand": "Mercedes-Benz", "model": "GLA"})
        set_focus(state, [x1, gla])
        set_shortlist(state, [x1, gla])
        state["offered_action"] = {"type": "SHOW_LISTINGS", "target_ids": [x1, gla], "metadata": {}}
        state, resolved = apply_turn_plan(state, {
            "transition": "CONTINUE",
            "job": "FIND_A_CAR",
            "action": "SHOW_LISTINGS",
            "target_ids": [x1, gla],
        })
        self.assertEqual(resolved["action"], Action.SHOW_LISTINGS.value)
        self.assertEqual(resolved["target_ids"], [x1, gla])

    def test_new_goal_clears_stale_vehicle_context(self):
        state = new_state("c6")
        x1 = register_object(state, "MODEL", {"brand": "BMW", "model": "X1"})
        set_focus(state, [x1])
        set_shortlist(state, [x1])
        state["constraints"] = {"budget_max": 15000, "vehicle_type": "SUV"}
        state["job"] = "FIND_A_CAR"
        state, _ = apply_turn_plan(state, {
            "transition": "START_NEW_GOAL",
            "job": "EVALUATE_SALE",
            "action": "VALUE_VEHICLE",
            "goal_summary": "Decide what to do with my Aqua",
            "objects": [{"alias": "mine", "type": "OWNED_VEHICLE", "payload": {"brand": "Toyota", "model": "Aqua", "year": 2020}}],
            "target_ids": ["mine"],
        })
        self.assertEqual(state["job"], "EVALUATE_SALE")
        self.assertNotIn(x1, state["objects"])
        self.assertNotIn("budget_max", state["constraints"])
        self.assertEqual(len(state["focus"]["object_ids"]), 1)

    def test_correction_changes_only_requested_constraint(self):
        state = new_state("c7")
        state["job"] = "FIND_A_CAR"
        state["goal"] = {"summary": "Find an SUV", "status": "ACTIVE"}
        state["constraints"] = {"budget_max": 15000, "vehicle_type": "SUV", "brands": ["BMW"]}
        state, _ = apply_turn_plan(state, {
            "transition": "CORRECT",
            "job": "FIND_A_CAR",
            "action": "SEARCH_VEHICLES",
            "constraints_delta": {"budget_max": 18000},
        })
        self.assertEqual(state["constraints"], {"budget_max": 18000, "vehicle_type": "SUV", "brands": ["BMW"]})

    def test_invalid_cross_audience_job_is_rejected(self):
        state = new_state("c8", audience="PERSONAL")
        with self.assertRaises(InvalidTurnPlan):
            apply_turn_plan(state, {
                "transition": "START_NEW_GOAL",
                "job": "PRICE_STOCK",
                "action": "ANALYZE_STOCK_PRICES",
            })

    def test_contradictory_budget_is_rejected(self):
        state = new_state("c9")
        with self.assertRaises(InvalidTurnPlan):
            apply_turn_plan(state, {
                "transition": "START_NEW_GOAL",
                "job": "FIND_A_CAR",
                "action": "SEARCH_VEHICLES",
                "constraints_delta": {"budget_min": 20000, "budget_max": 15000},
            })

    def test_optimistic_concurrency_prevents_stale_tab_overwrite(self):
        service = self.make_service()
        original = service.get_or_create("c10")
        tab_a = original
        tab_b = original
        saved_a, _ = service.apply_plan(tab_a, {
            "transition": "START_NEW_GOAL",
            "job": "FIND_A_CAR",
            "action": "SEARCH_VEHICLES",
            "constraints_delta": {"budget_max": 15000},
        })
        self.assertEqual(saved_a.revision, original.revision + 1)
        with self.assertRaises(StateConflict):
            service.apply_plan(tab_b, {
                "transition": "START_NEW_GOAL",
                "job": "FIND_A_CAR",
                "action": "SEARCH_VEHICLES",
                "constraints_delta": {"budget_max": 20000},
            })

    def test_business_journey_state(self):
        state = new_state("b1", audience="BUSINESS")
        state, _ = apply_turn_plan(state, {
            "transition": "START_NEW_GOAL",
            "job": "ANALYZE_BUSINESS",
            "action": "ANALYZE_BUSINESS_PERIOD",
            "constraints_delta": {"period_start": "2026-08-01", "period_end": "2026-08-31", "company": "Example Motors"},
        })
        self.assertEqual(state["job"], Job.ANALYZE_BUSINESS.value)
        self.assertEqual(state["constraints"]["period_start"], "2026-08-01")

    def test_object_registry_supports_specific_listing_focus(self):
        state = new_state("c11")
        listing = register_object(state, "LISTING", {
            "link": "https://example.test/123",
            "brand": "Mercedes-Benz",
            "model": "GLA",
            "year": 2017,
            "price": 14000,
            "km": 85000,
        })
        set_focus(state, [listing])
        validate_state(state)
        obj = state["objects"][state["focus"]["object_ids"][0]]
        self.assertEqual(obj["model"], "GLA")
        self.assertEqual(obj["price"], 14000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
