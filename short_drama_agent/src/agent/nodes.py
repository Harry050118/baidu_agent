from typing import Any

from src.evaluation.reviewer import choose_best
from src.evaluation.schemas import ReviewReport
from src.rag.query_builder import build_planning_query, build_review_query, build_writing_query
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay

from .state import AgentState


def safe_state_node(
    state: AgentState,
    *,
    node: str,
    operation: Any,
    failure_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return operation(state)
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(
            {
                "node": node,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "recoverable": False,
            }
        )
        return {**(failure_update or {}), "errors": errors}


def planning_retrieval_node(state: AgentState, *, retriever: Any) -> dict[str, Any]:
    return _retrieve(
        state,
        retriever,
        build_planning_query(state["requirement"]),
        "planning",
        "planning_guidelines",
        "planning_retrieval",
    )


def writing_retrieval_node(state: AgentState, *, retriever: Any) -> dict[str, Any]:
    return _retrieve(
        state,
        retriever,
        build_writing_query(state["story_plan"]),
        "writing",
        "writing_guidelines",
        "writing_retrieval",
    )


def review_retrieval_node(state: AgentState, *, retriever: Any) -> dict[str, Any]:
    return _retrieve(
        state,
        retriever,
        build_review_query(state["screenplay"]),
        "review",
        "review_guidelines",
        "review_retrieval",
    )


def _retrieve(
    state: AgentState,
    retriever: Any,
    query: str,
    purpose: str,
    state_key: str,
    node: str,
) -> dict[str, Any]:
    try:
        results = retriever(query, purpose)
        if not results:
            errors = list(state.get("errors", []))
            errors.append(
                {
                    "node": node,
                    "error_type": "EmptyKnowledgeBase",
                    "message": f"No guidelines retrieved for {purpose}",
                    "recoverable": True,
                }
            )
            return {state_key: [], "errors": errors}
        return {
            state_key: [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in results
            ]
        }
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(
            {
                "node": node,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "recoverable": True,
            }
        )
        return {state_key: [], "errors": errors}


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
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(
            {
                "node": "screenplay",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "recoverable": True,
            }
        )
        return {"screenplay": None, "errors": errors}


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
