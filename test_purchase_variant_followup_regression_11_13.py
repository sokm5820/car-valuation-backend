import unittest

import otodeger_v10_agent as agent
from otodeger_v10_state import (
    Action,
    Audience,
    Job,
    ObjectType,
    new_state,
    register_object,
    set_focus,
)


class PurchaseVariantFollowupRegression(unittest.TestCase):
    def _state(self):
        state = new_state(audience=Audience.PERSONAL.value, language="EN")
        state["job"] = Job.EVALUATE_PURCHASE.value
        state["goal"] = {"summary": "Evaluate 2007 Nissan March 1.2 at £5,000", "status": "ACTIVE"}
        state["constraints"] = {"asking_price": 5000.0}
        oid = register_object(
            state,
            ObjectType.LISTING.value,
            {
                "brand": "Nissan",
                "model": "March",
                "year": 2007,
                "category": "1.2",
                "asking_price": 5000.0,
            },
        )
        set_focus(state, [oid])
        return state

    def assert_variant(self, message, expected="1.0"):
        state = self._state()
        plan = agent._deterministic_followup_plan(message, state)
        self.assertIsNotNone(plan, message)
        self.assertEqual(plan.get("action"), Action.EVALUATE_PURCHASE.value, message)
        self.assertEqual(plan.get("job"), Job.EVALUATE_PURCHASE.value, message)
        self.assertEqual(plan.get("transition"), "REFINE", message)
        self.assertEqual(plan.get("target_ids"), ["variant_vehicle"], message)
        payload = (plan.get("objects") or [])[0]["payload"]
        self.assertEqual(payload.get("brand"), "Nissan", message)
        self.assertEqual(payload.get("model"), "March", message)
        self.assertEqual(payload.get("year"), 2007, message)
        self.assertEqual(payload.get("asking_price"), 5000.0, message)
        self.assertEqual(payload.get("category"), expected, message)

    def test_natural_engine_refinements_never_reopen_search(self):
        for message in (
            "1.0",
            "1.0L",
            "What about 1.0?",
            "How about the 1.0L?",
            "1.0 instead",
            "And what about 1.0?",
            "Try 1.0",
        ):
            with self.subTest(message=message):
                self.assert_variant(message)

    def test_semantic_planner_is_never_called_for_natural_variant_refinement(self):
        state = self._state()
        def explode(*args, **kwargs):
            raise AssertionError("semantic model must not be called")
        plan = agent._semantic_plan("What about 1.0?", "EN", state, {"_openai_post": explode})
        self.assertEqual(plan.get("action"), Action.EVALUATE_PURCHASE.value)
        self.assertEqual(plan.get("job"), Job.EVALUATE_PURCHASE.value)

    def test_missing_subject_clarifies_without_reopening_search(self):
        state = self._state()
        state["objects"] = {}
        state["focus"] = {"object_ids": []}
        state["shortlist"] = []
        plan = agent._deterministic_followup_plan("What about 1.0?", state)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.get("action"), Action.ASK_CLARIFICATION.value)
        self.assertEqual(plan.get("job"), Job.EVALUATE_PURCHASE.value)

    def test_engine_refinement_is_contextual_not_global(self):
        state = self._state()
        state["job"] = Job.FIND_A_CAR.value
        self.assertIsNone(agent._deterministic_followup_plan("What about 1.0?", state))

    def test_year_like_value_is_not_mistaken_for_engine(self):
        state = self._state()
        plan = agent._deterministic_followup_plan("What about 2010?", state)
        self.assertFalse(plan and plan.get("action") == Action.EVALUATE_PURCHASE.value and (plan.get("objects") or []))


if __name__ == "__main__":
    unittest.main()
