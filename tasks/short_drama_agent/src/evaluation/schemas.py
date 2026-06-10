from pydantic import BaseModel


class ReviewScorePayload(BaseModel):
    structure_score: float
    character_consistency_score: float
    conflict_score: float
    dialogue_score: float
    pacing_score: float
    shootability_score: float
    rag_adherence_score: float
    schema_validity_score: float
    total_score: float
    issues: list[str]
    revision_instructions: list[str]
    summary: str


class ReviewReport(ReviewScorePayload):
    deterministic_errors: list[str]
    passed: bool
