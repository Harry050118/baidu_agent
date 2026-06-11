import unittest

from src.app.cli import run_cli


class FakeApp:
    def __init__(self):
        self.created = None
        self.resumed_thread_id = None
        self.explicit_preferences = None

    def create(self, *, request, user_id, constraints):
        self.created = {
            "request": request,
            "user_id": user_id,
            **constraints,
        }
        return {"thread_id": "t-new", "status": "awaiting_outline_review"}

    def resume(self, *, thread_id, action=None, feedback=None, explicit_preferences=None):
        self.resumed_thread_id = thread_id
        self.explicit_preferences = explicit_preferences
        return {"thread_id": thread_id, "action": action, "feedback": feedback}

    def history(self, *, user_id):
        return [
            {
                "project_id": "p1",
                "title": "测试短剧",
                "genre": "悬疑",
                "final_score": 9.0,
            }
        ]


class CliFlowTests(unittest.TestCase):
    def setUp(self):
        self.app = FakeApp()

    def test_create_passes_request_and_constraints_to_agent(self):
        run_cli(["create", "--request", "校园悬疑", "--duration", "360"], self.app)

        self.assertEqual(self.app.created["target_duration_seconds"], 360)
        self.assertEqual(self.app.created["request"], "校园悬疑")

    def test_resume_uses_existing_thread_id(self):
        run_cli(["resume", "--thread-id", "t1", "--action", "approve"], self.app)

        self.assertEqual(self.app.resumed_thread_id, "t1")

    def test_history_lists_user_projects(self):
        output = run_cli(["history", "--user-id", "u1"], self.app)

        self.assertIn("测试短剧", output)

    def test_resume_passes_only_explicit_preference_arguments(self):
        run_cli(
            [
                "resume",
                "--thread-id",
                "t1",
                "--action",
                "revise",
                "--feedback",
                "加强反转",
                "--preferred-genre",
                "悬疑",
                "--production-constraint",
                "低成本",
            ],
            self.app,
        )

        self.assertEqual(
            self.app.explicit_preferences,
            {
                "preferred_genres": ["悬疑"],
                "production_constraints": ["低成本"],
            },
        )


if __name__ == "__main__":
    unittest.main()
