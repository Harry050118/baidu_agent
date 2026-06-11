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
                "RAG 方法遵循度和 Schema 有效性进行 0 到 10 分评分。"
                "仅输出 JSON 对象，必须包含 structure_score、character_consistency_score、"
                "conflict_score、dialogue_score、pacing_score、shootability_score、"
                "rag_adherence_score、schema_validity_score、total_score、issues、"
                "revision_instructions 和 summary。"
            ),
        },
        {
            "role": "user",
            "content": f"审查指南：{review_guidelines}\n\n剧本：{screenplay_json}",
        },
    ]
