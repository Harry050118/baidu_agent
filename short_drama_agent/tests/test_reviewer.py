import json
import unittest

from src.evaluation.reviewer import ScreenplayReviewer, build_review_report, choose_best
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay
from tests.screenplay_fixtures import SequenceLLM, valid_screenplay_dict


def score_payload(total_score=9.0):
    return {
        "structure_score": total_score,
        "character_consistency_score": total_score,
        "conflict_score": total_score,
        "dialogue_score": total_score,
        "pacing_score": total_score,
        "shootability_score": total_score,
        "rag_adherence_score": total_score,
        "schema_validity_score": total_score,
        "total_score": total_score,
        "issues": [],
        "revision_instructions": [],
        "summary": "结构完整。",
    }


def screenplay(title):
    payload = valid_screenplay_dict()
    payload["title"] = title
    return Screenplay.model_validate(payload)


class ReviewerTests(unittest.TestCase):
    def test_passed_requires_score_and_no_deterministic_errors(self):
        report = build_review_report(score_payload(9.0), ["scene ids invalid"], pass_score=8.0)
        self.assertFalse(report.passed)

    def test_higher_score_replaces_best_version(self):
        old_screenplay = screenplay("旧版")
        new_screenplay = screenplay("新版")
        old_report = build_review_report(score_payload(7.0), [], pass_score=8.0)
        new_report = build_review_report(score_payload(9.0), [], pass_score=8.0)

        best, report = choose_best(old_screenplay, old_report, new_screenplay, new_report)

        self.assertIs(best, new_screenplay)
        self.assertIs(report, new_report)

    def test_equal_score_prefers_fewer_deterministic_errors(self):
        old_screenplay = screenplay("旧版")
        new_screenplay = screenplay("新版")
        old_report = build_review_report(score_payload(8.0), ["error"], pass_score=8.0)
        cleaner_report = build_review_report(score_payload(8.0), [], pass_score=8.0)

        best, report = choose_best(old_screenplay, old_report, new_screenplay, cleaner_report)

        self.assertIs(best, new_screenplay)
        self.assertIs(report, cleaner_report)

    def test_reviewer_combines_deterministic_errors_with_llm_scores(self):
        invalid = screenplay("无效版本")
        invalid.scenes[0].dialogues[0].character = "陌生人"
        llm = SequenceLLM([json.dumps(score_payload(9.5), ensure_ascii=False)])
        reviewer = ScreenplayReviewer(llm)

        report = reviewer.review(invalid, GenerationConstraints(), [])

        self.assertEqual(report.total_score, 9.5)
        self.assertTrue(report.deterministic_errors)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
