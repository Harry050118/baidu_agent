from pydantic import BaseModel


class RetrievedGuideline(BaseModel):
    chunk_id: str
    purpose: str
    query: str
    text: str
    source: str
    document_title: str | None = None
    parent_header: str | None = None
    current_header: str | None = None
    vector_score: float
    title_score: float
    final_score: float
