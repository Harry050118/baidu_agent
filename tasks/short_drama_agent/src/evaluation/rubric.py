REVIEW_DIMENSIONS = (
    "structure",
    "character_consistency",
    "conflict",
    "dialogue",
    "pacing",
    "shootability",
    "rag_adherence",
    "schema_validity",
)


def build_review_messages(screenplay_json: str, review_guidelines: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是短剧质量审查员。按结构、人物一致性、冲突、对白、节奏、可拍性、"
                "RAG 方法遵循度和 Schema 有效性评分。仅输出 JSON 评分对象。"
            ),
        },
        {
            "role": "user",
            "content": f"审查指南：{review_guidelines}\n\n剧本：{screenplay_json}",
        },
    ]
