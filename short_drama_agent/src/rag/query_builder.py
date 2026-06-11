def build_planning_query(requirement: dict) -> str:
    return f"{_genre(requirement)}短剧的开头钩子、人物目标、核心冲突与反转设计"


def build_writing_query(story_plan: dict) -> str:
    return f"{_genre(story_plan)}短剧的场景节奏、视觉动作、对白与低成本可拍性"


def build_review_query(screenplay: dict) -> str:
    return f"{_genre(screenplay)}短剧的结构、节奏、人物一致性与可拍性评价标准"


def _genre(payload: dict) -> str:
    conflict = payload.get("conflict")
    conflict_type = conflict.get("type") if isinstance(conflict, dict) else None
    return payload.get("genre") or conflict_type or "未指定"
