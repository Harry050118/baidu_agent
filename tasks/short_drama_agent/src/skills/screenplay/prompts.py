import json
from typing import Any


def build_generation_messages(
    story_plan: dict[str, Any],
    requirement: dict[str, Any],
    constraints: Any,
    writing_guidelines: list[Any],
    user_preferences: dict[str, Any] | None,
) -> list[dict[str, str]]:
    payload = {
        "story_plan": story_plan,
        "requirement": requirement,
        "constraints": constraints.model_dump(),
        "writing_guidelines": [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in writing_guidelines
        ],
        "user_preferences": user_preferences,
    }
    return [
        {
            "role": "system",
            "content": "你是短剧编剧。仅输出符合 Screenplay Schema 的 JSON 对象。",
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def build_repair_messages(raw_response: str, validation_error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "保留所有有效剧情内容，只修复 JSON 解析、字段类型和缺失的必填字段。"
                "不要进行内容改写或质量优化，仅输出修复后的 JSON。"
            ),
        },
        {
            "role": "user",
            "content": f"校验错误：{validation_error}\n\n待修复响应：\n{raw_response}",
        },
    ]
