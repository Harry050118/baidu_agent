from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    project_id: str
    thread_id: str
    user_id: str
    user_request: str
    requirement: dict[str, Any] | None
    constraints: dict[str, Any]
    user_preferences: dict[str, Any] | None
    planning_guidelines: list[dict[str, Any]]
    story_plan: dict[str, Any] | None
    outline_confirmed: bool
    user_feedback: str | None
    human_review_action: str
    writing_guidelines: list[dict[str, Any]]
    screenplay: dict[str, Any] | None
    review_guidelines: list[dict[str, Any]]
    review_report: dict[str, Any] | None
    best_screenplay: dict[str, Any] | None
    best_review_report: dict[str, Any] | None
    json_repair_count: int
    content_revision_count: int
    final_json_path: str | None
    final_markdown_path: str | None
    errors: list[dict[str, Any]]
