import hashlib
from pathlib import Path

from .chunker import Chunker
from .document_loader import DocumentLoader
from .retriever import RetrievalResult, Retriever
from .schemas import RetrievedGuideline


def guideline_from_result(
    result: RetrievalResult,
    query: str,
    purpose: str,
) -> RetrievedGuideline:
    chunk = result.chunk
    metadata = chunk.metadata
    document_title = metadata.get("document_title")
    parent_header = metadata.get("parent_header")
    current_header = metadata.get("current_header", metadata.get("header"))
    identity = "\n".join(
        [
            Path(chunk.source).as_posix().lower(),
            document_title or "",
            parent_header or "",
            current_header or "",
            chunk.text,
        ]
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return RetrievedGuideline(
        chunk_id=chunk_id,
        purpose=purpose,
        query=query,
        text=chunk.text,
        source=chunk.source,
        document_title=document_title,
        parent_header=parent_header,
        current_header=current_header,
        vector_score=result.vector_score,
        title_score=result.title_score,
        final_score=result.final_score,
    )


class WritingKnowledgeBase:
    def __init__(self, retriever: Retriever, chunker: Chunker | None = None):
        self.retriever = retriever
        self.chunker = chunker or Chunker()

    def build(self, document_dir: str) -> None:
        documents = DocumentLoader.load(document_dir)
        self.retriever.index(self.chunker.split(documents))

    def search(
        self,
        query: str,
        purpose: str,
        candidate_k: int = 5,
        top_k: int = 3,
    ) -> list[RetrievedGuideline]:
        if candidate_k < 1 or top_k < 1:
            raise ValueError("candidate_k and top_k must be positive")
        if top_k > candidate_k:
            raise ValueError("top_k cannot exceed candidate_k")
        results = self.retriever.query(query, candidate_k=candidate_k, top_k=top_k)
        return [guideline_from_result(result, query, purpose) for result in results]
