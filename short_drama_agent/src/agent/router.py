from enum import Enum

from .state import AgentState


class HumanReviewAction(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    PAUSE = "pause"


def route_after_human_review(state: AgentState) -> str:
    action = HumanReviewAction(state["human_review_action"])
    return {
        HumanReviewAction.APPROVE: "screenplay",
        HumanReviewAction.REVISE: "story_planning",
        HumanReviewAction.PAUSE: "pause",
    }[action]


def route_after_story_planning(state: AgentState) -> str:
    return "human_review" if state.get("story_plan") else "stop"


def route_after_review(state: AgentState) -> str:
    report = state.get("review_report") or {}
    if not report:
        return "stop"
    if report.get("passed"):
        return "export"
    max_revisions = state.get("constraints", {}).get("max_content_revisions", 2)
    if state.get("content_revision_count", 0) >= max_revisions:
        return "export"
    return "revise"
