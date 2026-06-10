from typing import Any

from src.evaluation.reviewer import choose_best
from src.evaluation.schemas import ReviewReport
from src.skills.screenplay.repair import ScreenplayGenerationError
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay

from .state import AgentState


def screenplay_node(state: AgentState, *, skill: Any) -> dict[str, Any]:
    try:
        result = skill.generate(
            story_plan=state["story_plan"],
            requirement=state["requirement"],
            constraints=GenerationConstraints.model_validate(state["constraints"]),
            writing_guidelines=state.get("writing_guidelines", []),
            user_preferences=state.get("user_preferences"),
        )
        return {
            "screenplay": result.screenplay.model_dump(),
            "json_repair_count": state.get("json_repair_count", 0)
            + result.json_repair_attempts,
        }
    except ScreenplayGenerationError as exc:
        errors = list(state.get("errors", []))
        errors.append(
            {
                "node": "screenplay",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "recoverable": True,
            }
        )
        return {"errors": errors}


def review_node(state: AgentState, *, reviewer: Any) -> dict[str, Any]:
    screenplay = Screenplay.model_validate(state["screenplay"])
    constraints = GenerationConstraints.model_validate(state["constraints"])
    report = reviewer.review(screenplay, constraints, state.get("review_guidelines", []))
    best_screenplay = (
        Screenplay.model_validate(state["best_screenplay"])
        if state.get("best_screenplay")
        else None
    )
    best_report = (
        ReviewReport.model_validate(state["best_review_report"])
        if state.get("best_review_report")
        else None
    )
    selected_screenplay, selected_report = choose_best(
        best_screenplay,
        best_report,
        screenplay,
        report,
    )
    return {
        "review_report": report.model_dump(),
        "best_screenplay": selected_screenplay.model_dump(),
        "best_review_report": selected_report.model_dump(),
    }


def revise_node(state: AgentState) -> dict[str, int]:
    return {"content_revision_count": state.get("content_revision_count", 0) + 1}
