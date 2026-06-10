import tempfile
import unittest
from pathlib import Path

from src.memory.project_memory import ProjectMemoryRepository, ProjectSummary
from src.memory.user_memory import UserMemoryRepository
from src.tools.memory import MemoryTool


class MemoryRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "memory.sqlite"
        self.repo = ProjectMemoryRepository(db_path)
        self.user_repo = UserMemoryRepository(db_path)
        self.memory_tool = MemoryTool(self.user_repo, self.repo)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_project_writes_initial_record_without_inferred_preferences(self):
        self.repo.create_project("p1", "t1", "u1", "校园悬疑")

        self.assertEqual(self.repo.get_project("p1").status, "created")
        self.assertIsNone(self.user_repo.get("u1"))

    def test_explicit_outline_feedback_updates_preferences(self):
        self.memory_tool.record_outline_feedback("u1", {"preferred_genres": ["悬疑"]})

        self.assertEqual(self.user_repo.get("u1").preferred_genres, ["悬疑"])

    def test_successful_export_writes_project_summary(self):
        self.repo.create_project("p1", "t1", "u1", "校园悬疑")
        summary = ProjectSummary(
            project_id="p1",
            title="最后一课",
            genre="悬疑",
            logline="学生追查录音。",
            user_feedback=None,
            final_score=9.0,
            output_paths=["output/projects/p1/screenplay.json"],
        )

        self.memory_tool.record_export(summary)

        self.assertEqual(self.repo.history("u1")[0].title, summary.title)
        self.assertEqual(self.repo.get_project("p1").status, "exported")


if __name__ == "__main__":
    unittest.main()
