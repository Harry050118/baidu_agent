import unittest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agent.graph import GraphDependencies, build_test_graph
from src.evaluation.reviewer import build_review_report
from src.skills.screenplay.schemas import Screenplay
from src.skills.screenplay.repair import JsonRepairExhaustedError
from src.skills.screenplay.skill import ScreenplaySkillResult
from tests.screenplay_fixtures import valid_screenplay_dict
from tests.test_reviewer import score_payload


def thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def initial_state():
    return {
        "thread_id": "thread",
        "user_id": "u1",
        "user_request": "校园悬疑",
        "requirement": {"genre": "悬疑"},
        "constraints": {
            "min_duration_seconds": 180,
            "max_duration_seconds": 300,
            "min_scene_count": 3,
            "max_scene_count": 6,
            "max_content_revisions": 2,
            "pass_score": 8.0,
        },
        "writing_guidelines": [],
        "review_guidelines": [],
        "json_repair_count": 0,
        "content_revision_count": 0,
        "errors": [],
    }


class Skill:
    def generate(self, **kwargs):
        return ScreenplaySkillResult(
            screenplay=Screenplay.model_validate(valid_screenplay_dict()),
            json_repair_attempts=0,
        )


class FailingSkill:
    def generate(self, **kwargs):
        raise JsonRepairExhaustedError("invalid")


class Reviewer:
    def review(self, *args):
        return build_review_report(score_payload(9.0), [], pass_score=8.0)


class SequenceReviewer:
    def __init__(self):
        self.calls = 0

    def review(self, *args):
        self.calls += 1
        score = 7.0 if self.calls == 1 else 9.0
        return build_review_report(score_payload(score), [], pass_score=8.0)


class RecordingRetriever:
    def __init__(self):
        self.purposes = []

    def __call__(self, query, purpose):
        self.purposes.append(purpose)
        return [{"chunk_id": purpose}]


class RecordingExporter:
    def __init__(self):
        self.state = None

    def __call__(self, state):
        self.state = state
        return exporter(state)


def story_planner(state):
    return {
        "story_plan": {
            "genre": "悬疑",
            "outline": "学生追查录音",
            "feedback": state.get("user_feedback"),
        }
    }


def exporter(state):
    return {
        "final_json_path": f"output/{state['thread_id']}/screenplay.json",
        "final_markdown_path": f"output/{state['thread_id']}/screenplay.md",
    }


def dependencies():
    return GraphDependencies(
        retriever=lambda query, purpose: [{"chunk_id": purpose}],
        story_planner=story_planner,
        screenplay_skill=Skill(),
        reviewer=Reviewer(),
        exporter=exporter,
    )


class GraphFlowTests(unittest.TestCase):
    def test_graph_interrupts_for_outline_review_and_resumes_on_approve(self):
        graph = build_test_graph(dependencies(), checkpointer=MemorySaver())

        first = graph.invoke(initial_state(), config=thread_config("t1"))
        self.assertTrue(first["__interrupt__"])
        resumed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t1"))

        self.assertIn("screenplay", resumed)
        self.assertTrue(resumed["final_json_path"])

    def test_pause_keeps_thread_resumable_without_advancing(self):
        graph = build_test_graph(dependencies(), checkpointer=MemorySaver())
        graph.invoke(initial_state(), config=thread_config("t2"))

        paused = graph.invoke(Command(resume={"action": "pause"}), thread_config("t2"))

        self.assertTrue(paused["__interrupt__"])
        self.assertIsNone(paused.get("screenplay"))
        resumed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t2"))
        self.assertIn("screenplay", resumed)

    def test_revise_returns_to_story_planning(self):
        graph = build_test_graph(dependencies(), checkpointer=MemorySaver())
        graph.invoke(initial_state(), config=thread_config("t3"))

        revised = graph.invoke(
            Command(resume={"action": "revise", "feedback": "加强结尾反转"}),
            thread_config("t3"),
        )

        self.assertEqual(revised["user_feedback"], "加强结尾反转")
        self.assertEqual(revised["story_plan"]["feedback"], "加强结尾反转")
        self.assertTrue(revised["__interrupt__"])

    def test_revise_updates_checkpoint_with_explicit_user_preferences(self):
        graph = build_test_graph(dependencies(), checkpointer=MemorySaver())
        graph.invoke(initial_state(), config=thread_config("t-preferences"))

        revised = graph.invoke(
            Command(
                resume={
                    "action": "revise",
                    "feedback": "加强结尾反转",
                    "user_preferences": {"preferred_tones": ["紧张"]},
                }
            ),
            thread_config("t-preferences"),
        )

        self.assertEqual(revised["user_preferences"]["preferred_tones"], ["紧张"])

    def test_generation_failure_stops_before_review_and_preserves_error(self):
        deps = dependencies()
        deps = GraphDependencies(
            retriever=deps.retriever,
            story_planner=deps.story_planner,
            screenplay_skill=FailingSkill(),
            reviewer=deps.reviewer,
            exporter=deps.exporter,
        )
        graph = build_test_graph(deps, checkpointer=MemorySaver())
        state = initial_state()
        state["screenplay"] = valid_screenplay_dict()
        graph.invoke(state, config=thread_config("t4"))

        failed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t4"))

        self.assertEqual(failed["errors"][0]["error_type"], "JsonRepairExhaustedError")
        self.assertIsNone(failed.get("review_report"))

    def test_graph_runs_three_retrieval_stages_and_exports_trace(self):
        retriever = RecordingRetriever()
        recording_exporter = RecordingExporter()
        deps = dependencies()
        deps = GraphDependencies(
            retriever=retriever,
            story_planner=deps.story_planner,
            screenplay_skill=deps.screenplay_skill,
            reviewer=deps.reviewer,
            exporter=recording_exporter,
        )
        graph = build_test_graph(deps, checkpointer=MemorySaver())

        graph.invoke(initial_state(), config=thread_config("t5"))
        graph.invoke(Command(resume={"action": "approve"}), thread_config("t5"))

        self.assertEqual(retriever.purposes, ["planning", "writing", "review"])
        self.assertEqual(recording_exporter.state["planning_guidelines"][0]["chunk_id"], "planning")
        self.assertEqual(recording_exporter.state["writing_guidelines"][0]["chunk_id"], "writing")
        self.assertEqual(recording_exporter.state["review_guidelines"][0]["chunk_id"], "review")

    def test_content_revision_repeats_writing_and_review_retrieval(self):
        retriever = RecordingRetriever()
        deps = dependencies()
        deps = GraphDependencies(
            retriever=retriever,
            story_planner=deps.story_planner,
            screenplay_skill=deps.screenplay_skill,
            reviewer=SequenceReviewer(),
            exporter=deps.exporter,
        )
        graph = build_test_graph(deps, checkpointer=MemorySaver())

        graph.invoke(initial_state(), config=thread_config("t6"))
        graph.invoke(Command(resume={"action": "approve"}), thread_config("t6"))

        self.assertEqual(
            retriever.purposes,
            ["planning", "writing", "review", "writing", "review"],
        )

    def test_story_planning_failure_is_recorded_and_stops(self):
        deps = dependencies()

        def failing_planner(state):
            raise RuntimeError("planner offline")

        graph = build_test_graph(
            GraphDependencies(
                retriever=deps.retriever,
                story_planner=failing_planner,
                screenplay_skill=deps.screenplay_skill,
                reviewer=deps.reviewer,
                exporter=deps.exporter,
            ),
            checkpointer=MemorySaver(),
        )

        state = initial_state()
        state["story_plan"] = {"genre": "旧大纲"}
        failed = graph.invoke(state, config=thread_config("t7"))

        self.assertEqual(failed["errors"][0]["node"], "story_planning")
        self.assertFalse(failed["errors"][0]["recoverable"])
        self.assertNotIn("__interrupt__", failed)

    def test_review_failure_is_recorded_and_stops_before_export(self):
        deps = dependencies()
        recording_exporter = RecordingExporter()

        class FailingReviewer:
            def review(self, *args):
                raise RuntimeError("review offline")

        graph = build_test_graph(
            GraphDependencies(
                retriever=deps.retriever,
                story_planner=deps.story_planner,
                screenplay_skill=deps.screenplay_skill,
                reviewer=FailingReviewer(),
                exporter=recording_exporter,
            ),
            checkpointer=MemorySaver(),
        )

        state = initial_state()
        state["review_report"] = {"passed": True}
        graph.invoke(state, config=thread_config("t8"))
        failed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t8"))

        self.assertEqual(failed["errors"][0]["node"], "review")
        self.assertIsNone(recording_exporter.state)

    def test_export_failure_is_recorded(self):
        deps = dependencies()

        def failing_exporter(state):
            raise OSError("disk full")

        graph = build_test_graph(
            GraphDependencies(
                retriever=deps.retriever,
                story_planner=deps.story_planner,
                screenplay_skill=deps.screenplay_skill,
                reviewer=deps.reviewer,
                exporter=failing_exporter,
            ),
            checkpointer=MemorySaver(),
        )

        graph.invoke(initial_state(), config=thread_config("t9"))
        failed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t9"))

        self.assertEqual(failed["errors"][0]["node"], "export")
        self.assertFalse(failed["errors"][0]["recoverable"])


if __name__ == "__main__":
    unittest.main()
