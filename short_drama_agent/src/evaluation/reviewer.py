import json
from typing import Any

from src.llm.base import LLM
from src.skills.screenplay.repair import extract_json_text
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay, validate_screenplay

from .rubric import REVIEW_DIMENSIONS, build_review_messages
from .schemas import ReviewReport, ReviewScorePayload


def build_review_report(
    score_payload: ReviewScorePayload | dict[str, Any],
    deterministic_errors: list[str],
    pass_score: float,
) -> ReviewReport:
    scores = ReviewScorePayload.model_validate(score_payload)
    return ReviewReport(
        **scores.model_dump(),
        deterministic_errors=deterministic_errors,
        passed=scores.total_score >= pass_score and not deterministic_errors,
    )


def normalize_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("review"), dict):
        payload = payload["review"]
    normalized = dict(payload)
    shorthand_scores = {
        dimension: payload.get(dimension)
        for dimension in REVIEW_DIMENSIONS
        if payload.get(dimension) is not None
    }
    use_five_point_scale = shorthand_scores and max(shorthand_scores.values()) <= 5
    for dimension in REVIEW_DIMENSIONS:
        score_key = f"{dimension}_score"
        if score_key not in normalized and dimension in shorthand_scores:
            score = float(shorthand_scores[dimension])
            normalized[score_key] = score * 2 if use_five_point_scale else score
    dimension_scores = [
        float(normalized[f"{dimension}_score"])
        for dimension in REVIEW_DIMENSIONS
        if f"{dimension}_score" in normalized
    ]
    normalized.setdefault(
        "total_score",
        sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0.0,
    )
    if float(normalized["total_score"]) > 10 and dimension_scores:
        normalized["total_score"] = sum(dimension_scores) / len(dimension_scores)
    normalized.setdefault("issues", [])
    normalized.setdefault("revision_instructions", [])
    if isinstance(normalized["revision_instructions"], str):
        normalized["revision_instructions"] = [normalized["revision_instructions"]]
    normalized.setdefault("summary", "自动归一化模型审查结果。")
    return normalized


def choose_best(
    best_screenplay: Screenplay | None,
    best_report: ReviewReport | None,
    candidate_screenplay: Screenplay,
    candidate_report: ReviewReport,
) -> tuple[Screenplay, ReviewReport]:
    if best_screenplay is None or best_report is None:
        return candidate_screenplay, candidate_report
    candidate_key = (
        candidate_report.total_score,
        -len(candidate_report.deterministic_errors),
    )
    best_key = (best_report.total_score, -len(best_report.deterministic_errors))
    if candidate_key > best_key:
        return candidate_screenplay, candidate_report
    return best_screenplay, best_report


class ScreenplayReviewer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def review(
        self,
        screenplay: Screenplay,
        constraints: GenerationConstraints,
        review_guidelines: list[Any],
    ) -> ReviewReport:
        guidelines = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in review_guidelines
        ]
        response = self.llm.generate(
            build_review_messages(screenplay.model_dump_json(), guidelines),
            temperature=0.0,
        )
        score_payload = normalize_score_payload(json.loads(extract_json_text(response)))
        return build_review_report(
            score_payload,
            validate_screenplay(screenplay, constraints),
            constraints.pass_score,
        )
