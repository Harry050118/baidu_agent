import tempfile
import unittest
from pathlib import Path

from src.app.runtime import ShortDramaApplication
from src.memory.project_memory import ProjectMemoryRepository
from src.memory.user_memory import UserMemoryRepository
from src.tools.memory import MemoryTool


class RecordingGraph:
    def __init__(self):
        self.invocations = []

    def invoke(self, payload, config):
        self.invocations.append((payload, config))
        return {"ok": True}


class RuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "memory.sqlite"
        self.projects = ProjectMemoryRepository(path)
        self.users = UserMemoryRepository(path)
        self.memory = MemoryTool(self.users, self.projects)
        self.graph = RecordingGraph()
        self.app = ShortDramaApplication(self.graph, self.projects, self.users, self.memory, {})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_loads_existing_user_preferences(self):
        self.users.update_explicit("u1", {"preferred_genres": ["悬疑"]})

        self.app.create(request="校园故事", user_id="u1", constraints={})

        state = self.graph.invocations[0][0]
        self.assertEqual(state["user_preferences"]["preferred_genres"], ["悬疑"])

    def test_resume_writes_only_explicit_preferences(self):
        self.projects.create_project("p1", "t1", "u1", "校园故事")

        self.app.resume(
            thread_id="t1",
            action="revise",
            feedback="加强反转",
            explicit_preferences={"preferred_tones": ["紧张"]},
        )

        self.assertEqual(self.users.get("u1").preferred_tones, ["紧张"])
        command = self.graph.invocations[0][0]
        self.assertEqual(command.resume["user_preferences"]["preferred_tones"], ["紧张"])

    def test_resume_does_not_write_preferences_without_revise_action(self):
        self.projects.create_project("p1", "t1", "u1", "校园故事")

        self.app.resume(
            thread_id="t1",
            action="approve",
            explicit_preferences={"preferred_tones": ["紧张"]},
        )

        self.assertIsNone(self.users.get("u1"))


if __name__ == "__main__":
    unittest.main()
