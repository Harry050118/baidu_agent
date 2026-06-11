def build_planning_query(requirement: dict) -> str:
    return f"{requirement['genre']}短剧的开头钩子、人物目标、核心冲突与反转设计"


def build_writing_query(story_plan: dict) -> str:
    return f"{story_plan['genre']}短剧的场景节奏、视觉动作、对白与低成本可拍性"


def build_review_query(screenplay: dict) -> str:
    return f"{screenplay['genre']}短剧的结构、节奏、人物一致性与可拍性评价标准"
