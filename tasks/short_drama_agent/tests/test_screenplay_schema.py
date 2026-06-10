import unittest

from src.skills.screenplay.schemas import (
    GenerationConstraints,
    Scene,
    Screenplay,
    validate_screenplay,
)


def valid_scene(scene_id=1, character="林夏", ending_hook="门外传来脚步声。"):
    return {
        "scene_id": scene_id,
        "location": "教室",
        "time": "夜",
        "purpose": "发现线索",
        "conflict": "有人试图阻止调查",
        "visual_action": "她推开门，看见桌上的录音笔。",
        "dialogues": [{"character": character, "line": "真相就在这里。"}],
        "estimated_seconds": 60,
        "ending_hook": ending_hook,
    }


def valid_screenplay():
    return Screenplay(
        title="最后一课",
        genre="悬疑",
        logline="学生追查消失的录音。",
        theme="信任",
        opening_hook="失踪者的声音突然响起。",
        main_conflict="林夏必须在老师阻挠下找到真相。",
        reversal="录音来自未来。",
        emotional_payoff="林夏选择公开真相。",
        characters=[{"name": "林夏", "role": "学生", "personality": "执着", "goal": "找到真相"}],
        scenes=[
            valid_scene(1),
            valid_scene(2, ending_hook=None),
            valid_scene(3, ending_hook=None),
        ],
        ending="真相被公开。",
    )


class ScreenplaySchemaTests(unittest.TestCase):
    def test_scene_supports_visual_action_and_optional_shooting_note(self):
        scene = Scene(**valid_scene())
        self.assertEqual(scene.visual_action, "她推开门，看见桌上的录音笔。")
        self.assertIsNone(scene.shooting_note)

    def test_constraints_are_defaults_not_schema_constants(self):
        constraints = GenerationConstraints()
        self.assertEqual(constraints.min_duration_seconds, 180)
        custom = constraints.model_copy(update={"max_scene_count": 8})
        self.assertEqual(custom.max_scene_count, 8)

    def test_valid_screenplay_has_no_deterministic_errors(self):
        self.assertEqual(validate_screenplay(valid_screenplay(), GenerationConstraints()), [])

    def test_deterministic_validation_rejects_unknown_dialogue_character(self):
        screenplay = valid_screenplay()
        screenplay.scenes[0].dialogues[0].character = "陌生人"

        errors = validate_screenplay(screenplay, GenerationConstraints())

        self.assertTrue(any("角色" in error for error in errors))

    def test_validation_checks_scene_ids_hooks_and_reversal(self):
        screenplay = valid_screenplay()
        screenplay.scenes[1].scene_id = 1
        for scene in screenplay.scenes:
            scene.ending_hook = None
        screenplay.reversal = ""

        errors = validate_screenplay(screenplay, GenerationConstraints())

        self.assertTrue(any("scene_id" in error for error in errors))
        self.assertTrue(any("钩子" in error for error in errors))
        self.assertTrue(any("反转" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
