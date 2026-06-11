import unittest

from src.skills.screenplay.repair import (
    JsonRepairExhaustedError,
    extract_json_text,
)
from src.skills.screenplay.skill import ScreenplaySkill
from tests.screenplay_fixtures import SequenceLLM, skill_inputs, valid_screenplay_json


class JsonRepairTests(unittest.TestCase):
    def test_extract_json_text_supports_markdown_fence(self):
        fenced = f"```json\n{valid_screenplay_json()}\n```"
        self.assertEqual(extract_json_text(fenced), valid_screenplay_json())

    def test_skill_reports_repair_attempts_after_invalid_first_response(self):
        llm = SequenceLLM(["not json", valid_screenplay_json()])

        result = ScreenplaySkill(llm, max_json_repair_attempts=2).generate(**skill_inputs())

        self.assertEqual(result.json_repair_attempts, 1)

    def test_skill_raises_typed_error_after_repair_limit(self):
        llm = SequenceLLM(["bad", "still bad", "bad again"])

        with self.assertRaises(JsonRepairExhaustedError):
            ScreenplaySkill(llm, max_json_repair_attempts=2).generate(**skill_inputs())


if __name__ == "__main__":
    unittest.main()
