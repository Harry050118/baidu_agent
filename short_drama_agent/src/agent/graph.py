from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .nodes import (
    planning_retrieval_node,
    review_node,
    review_retrieval_node,
    revise_node,
    safe_state_node,
    screenplay_node,
    writing_retrieval_node,
)
from .router import (
    HumanReviewAction,
    route_after_human_review,
    route_after_review,
    route_after_story_planning,
)
from .state import AgentState


StateNode = Callable[[AgentState], dict[str, Any]]


@dataclass(frozen=True)
class GraphDependencies:
    retriever: Any
    story_planner: StateNode
    screenplay_skill: Any
    reviewer: Any
    exporter: StateNode


def human_review_node(state: AgentState) -> dict[str, Any]:
    response = interrupt({"story_plan": state.get("story_plan")})
    action = HumanReviewAction(response["action"])
    update: dict[str, Any] = {"human_review_action": action.value}
    if response.get("feedback") is not None:
        update["user_feedback"] = response["feedback"]
    if response.get("user_preferences") is not None:
        update["user_preferences"] = response["user_preferences"]
    if action is HumanReviewAction.APPROVE:
        update["outline_confirmed"] = True
    return update


def route_after_screenplay(state: AgentState) -> str:
    return "review" if state.get("screenplay") else "stop"


def build_graph(dependencies: GraphDependencies, *, checkpointer: Any):
    builder = StateGraph(AgentState)
    builder.add_node(
        "planning_retrieval",
        lambda state: planning_retrieval_node(state, retriever=dependencies.retriever),
    )
    builder.add_node(
        "story_planning",
        lambda state: safe_state_node(
            state,
            node="story_planning",
            operation=dependencies.story_planner,
            failure_update={"story_plan": None},
        ),
    )
    builder.add_node("human_review", human_review_node)
    builder.add_node(
        "writing_retrieval",
        lambda state: writing_retrieval_node(state, retriever=dependencies.retriever),
    )
    builder.add_node(
        "screenplay",
        lambda state: screenplay_node(state, skill=dependencies.screenplay_skill),
    )
    builder.add_node(
        "review_retrieval",
        lambda state: review_retrieval_node(state, retriever=dependencies.retriever),
    )
    builder.add_node(
        "review",
        lambda state: safe_state_node(
            state,
            node="review",
            operation=lambda current: review_node(current, reviewer=dependencies.reviewer),
            failure_update={"review_report": None},
        ),
    )
    builder.add_node("revise", revise_node)
    builder.add_node(
        "export",
        lambda state: safe_state_node(
            state,
            node="export",
            operation=dependencies.exporter,
        ),
    )

    builder.add_edge(START, "planning_retrieval")
    builder.add_edge("planning_retrieval", "story_planning")
    builder.add_conditional_edges(
        "story_planning",
        route_after_story_planning,
        {"human_review": "human_review", "stop": END},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "screenplay": "writing_retrieval",
            "story_planning": "story_planning",
            "pause": "human_review",
        },
    )
    builder.add_conditional_edges(
        "screenplay",
        route_after_screenplay,
        {"review": "review_retrieval", "stop": END},
    )
    builder.add_edge("writing_retrieval", "screenplay")
    builder.add_edge("review_retrieval", "review")
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"revise": "revise", "export": "export", "stop": END},
    )
    builder.add_edge("revise", "writing_retrieval")
    builder.add_edge("export", END)
    return builder.compile(checkpointer=checkpointer)


def build_test_graph(dependencies: GraphDependencies, *, checkpointer: Any):
    return build_graph(dependencies, checkpointer=checkpointer)


def sqlite_checkpointer(db_path: str | Path):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpoint support requires langgraph-checkpoint-sqlite"
        ) from exc
    import sqlite3

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)
