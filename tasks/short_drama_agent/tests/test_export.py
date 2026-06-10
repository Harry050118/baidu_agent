import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.reviewer import build_review_report
from src.memory.project_memory import ProjectSummary
from src.skills.screenplay.schemas import Screenplay
from src.tools.export import export_project
from tests.screenplay_fixtures import valid_screenplay_dict
from tests.test_reviewer import score_payload


class ExportTests(unittest.TestCase):
    def test_export_writes_best_screenplay_review_and_trace(self):
        best_screenplay = Screenplay.model_validate(valid_screenplay_dict())
        best_report = build_review_report(score_payload(9.0), [], pass_score=8.0)
        retrieval_trace = [{"chunk_id": "abc", "purpose": "writing"}]
        summary = ProjectSummary(
            project_id="p1",
            title=best_screenplay.title,
            genre=best_screenplay.genre,
            logline=best_screenplay.logline,
            user_feedback=None,
            final_score=best_report.total_score,
            output_paths=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_project(
                Path(temp_dir),
                best_screenplay,
                best_report,
                retrieval_trace,
                summary,
            )

            self.assertTrue(paths.screenplay_json.exists())
            self.assertIn(
                best_screenplay.title,
                paths.screenplay_markdown.read_text(encoding="utf-8"),
            )
            self.assertTrue(paths.retrieval_trace_json.exists())
            self.assertEqual(
                json.loads(paths.summary_json.read_text(encoding="utf-8"))["project_id"],
                "p1",
            )


if __name__ == "__main__":
    unittest.main()
