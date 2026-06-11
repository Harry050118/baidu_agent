from src.rag.knowledge_base import WritingKnowledgeBase
from src.rag.schemas import RetrievedGuideline


def retrieve_guidelines(
    knowledge_base: WritingKnowledgeBase,
    query: str,
    purpose: str,
    candidate_k: int = 5,
    top_k: int = 3,
) -> list[RetrievedGuideline]:
    return knowledge_base.search(query, purpose, candidate_k=candidate_k, top_k=top_k)
