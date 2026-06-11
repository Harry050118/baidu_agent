import unittest

from src.agent.nodes import (
    planning_retrieval_node,
    review_node,
    review_retrieval_node,
    revise_node,
    screenplay_node,
    writing_retrieval_node,
)
from src.agent.router import HumanReviewAction, route_after_human_review, route_after_review
from src.evaluation.reviewer import build_review_report
from src.skills.screenplay.repair import JsonRepairExhaustedError
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay
from src.skills.screenplay.skill import ScreenplaySkillResult
from tests.screenplay_fixtures import valid_screenplay_dict
from tests.test_reviewer import score_payload


def base_state(**updates):
    state = {
        "story_plan": {"genre": "悬疑"},
        "requirement": {"genre": "悬疑"},
        "constraints": GenerationConstraints().model_dump(),
        "writing_guidelines": [],
        "user_preferences": None,
        "json_repair_count": 0,
        "content_revision_count": 0,
        "errors": [],
    }
    state.update(updates)
    return state


class SuccessfulSkill:
    def __init__(self, repairs=0):
        self.repairs = repairs

    def generate(self, **kwargs):
        return ScreenplaySkillResult(
            screenplay=Screenplay.model_validate(valid_screenplay_dict()),
            json_repair_attempts=self.repairs,
        )


class FailingSkill:
    def generate(self, **kwargs):
        raise JsonRepairExhaustedError("invalid")


class Reviewer:
    def __init__(self, report):
        self.report = report

    def review(self, *args):
        return self.report


class Retriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def __call__(self, query, purpose):
        self.calls.append((query, purpose))
        if self.error:
            raise self.error
        return self.results


class AgentRoutingTests(unittest.TestCase):
    def test_screenplay_node_accumulates_skill_json_repairs(self):
        update = screenplay_node(base_state(json_repair_count=2), skill=SuccessfulSkill(repairs=1))
        self.assertEqual(update["json_repair_count"], 3)
        self.assertIsInstance(update["screenplay"], dict)

    def test_screenplay_node_records_typed_skill_failure(self):
        update = screenplay_node(base_state(), skill=FailingSkill())
        self.assertEqual(update["errors"][0]["error_type"], "JsonRepairExhaustedError")

    def test_review_node_updates_current_and_best_versions(self):
        report = build_review_report(score_payload(9.0), [], pass_score=8.0)
        state = base_state(screenplay=valid_screenplay_dict(), review_guidelines=[])
        update = review_node(state, reviewer=Reviewer(report))
        self.assertEqual(update["best_screenplay"]["title"], "最后一课")
        self.assertEqual(update["best_review_report"]["total_score"], 9.0)

    def test_review_router_stops_at_revision_limit_and_exports_best(self):
        state = base_state(
            review_report={"passed": False},
            content_revision_count=2,
        )
        self.assertEqual(route_after_review(state), "export")

    def test_human_review_actions_have_three_explicit_values(self):
        self.assertEqual({item.value for item in HumanReviewAction}, {"approve", "revise", "pause"})
        self.assertEqual(route_after_human_review({"human_review_action": "approve"}), "screenplay")

    def test_revise_node_only_increments_content_revision_count(self):
        update = revise_node(base_state(content_revision_count=1))
        self.assertEqual(update, {"content_revision_count": 2})

    def test_retrieval_nodes_store_stage_guidelines(self):
        retriever = Retriever([{"chunk_id": "c1"}])
        planning = planning_retrieval_node(base_state(), retriever=retriever)
        writing = writing_retrieval_node(base_state(), retriever=retriever)
        review = review_retrieval_node(
            base_state(screenplay=valid_screenplay_dict()),
            retriever=retriever,
        )

        self.assertEqual(planning["planning_guidelines"][0]["chunk_id"], "c1")
        self.assertEqual(writing["writing_guidelines"][0]["chunk_id"], "c1")
        self.assertEqual(review["review_guidelines"][0]["chunk_id"], "c1")
        self.assertEqual([purpose for _, purpose in retriever.calls], ["planning", "writing", "review"])

    def test_retrieval_failure_records_recoverable_error_and_degrades(self):
        update = planning_retrieval_node(
            base_state(),
            retriever=Retriever(error=RuntimeError("offline")),
        )

        self.assertEqual(update["planning_guidelines"], [])
        self.assertEqual(update["errors"][0]["node"], "planning_retrieval")
        self.assertTrue(update["errors"][0]["recoverable"])

    def test_empty_retrieval_records_recoverable_error_and_degrades(self):
        update = planning_retrieval_node(base_state(), retriever=Retriever())

        self.assertEqual(update["planning_guidelines"], [])
        self.assertEqual(update["errors"][0]["error_type"], "EmptyKnowledgeBase")
        self.assertTrue(update["errors"][0]["recoverable"])


if __name__ == "__main__":
    unittest.main()
