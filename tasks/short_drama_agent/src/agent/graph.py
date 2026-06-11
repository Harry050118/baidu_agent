from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .nodes import review_node, revise_node, screenplay_node
from .router import HumanReviewAction, route_after_human_review, route_after_review
from .state import AgentState


StateNode = Callable[[AgentState], dict[str, Any]]


@dataclass(frozen=True)
class GraphDependencies:
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
    if action is HumanReviewAction.APPROVE:
        update["outline_confirmed"] = True
    return update


def route_after_screenplay(state: AgentState) -> str:
    return "review" if state.get("screenplay") else "stop"


def build_graph(dependencies: GraphDependencies, *, checkpointer: Any):
    builder = StateGraph(AgentState)
    builder.add_node("story_planning", dependencies.story_planner)
    builder.add_node("human_review", human_review_node)
    builder.add_node(
        "screenplay",
        lambda state: screenplay_node(state, skill=dependencies.screenplay_skill),
    )
    builder.add_node(
        "review",
        lambda state: review_node(state, reviewer=dependencies.reviewer),
    )
    builder.add_node("revise", revise_node)
    builder.add_node("export", dependencies.exporter)

    builder.add_edge(START, "story_planning")
    builder.add_edge("story_planning", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "screenplay": "screenplay",
            "story_planning": "story_planning",
            "pause": "human_review",
        },
    )
    builder.add_conditional_edges(
        "screenplay",
        route_after_screenplay,
        {"review": "review", "stop": END},
    )
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"revise": "revise", "export": "export"},
    )
    builder.add_edge("revise", "screenplay")
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
