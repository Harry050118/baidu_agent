import json
from typing import Any

from src.llm.base import LLM
from src.skills.screenplay.repair import extract_json_text
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay, validate_screenplay

from .rubric import build_review_messages
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
        score_payload = json.loads(extract_json_text(response))
        return build_review_report(
            score_payload,
            validate_screenplay(screenplay, constraints),
            constraints.pass_score,
        )
