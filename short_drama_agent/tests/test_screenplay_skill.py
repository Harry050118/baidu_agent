import unittest

from src.skills.screenplay.skill import ScreenplaySkill
from tests.screenplay_fixtures import SequenceLLM, skill_inputs, valid_screenplay_json


class ScreenplaySkillTests(unittest.TestCase):
    def test_generation_prompt_contains_required_screenplay_schema(self):
        llm = SequenceLLM([valid_screenplay_json()])

        ScreenplaySkill(llm).generate(**skill_inputs())

        prompt = "\n".join(message["content"] for message in llm.messages[0])
        self.assertIn("personality", prompt)
        self.assertIn("goal", prompt)
        self.assertIn("character", prompt)

    def test_valid_first_response_returns_screenplay_without_repair(self):
        llm = SequenceLLM([valid_screenplay_json()])

        result = ScreenplaySkill(llm).generate(**skill_inputs())

        self.assertEqual(result.screenplay.title, "最后一课")
        self.assertEqual(result.json_repair_attempts, 0)

    def test_repair_prompt_limits_changes_to_json_structure(self):
        llm = SequenceLLM(["not json", valid_screenplay_json()])

        ScreenplaySkill(llm, max_json_repair_attempts=1).generate(**skill_inputs())

        repair_prompt = "\n".join(message["content"] for message in llm.messages[1])
        self.assertIn("保留所有有效剧情内容", repair_prompt)
        self.assertIn("只修复 JSON", repair_prompt)
        self.assertIn("personality", repair_prompt)


if __name__ == "__main__":
    unittest.main()
