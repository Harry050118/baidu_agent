import json
import unittest

from src.skills.screenplay.repair import (
    JsonRepairExhaustedError,
    extract_json_text,
    parse_screenplay,
)
from src.skills.screenplay.skill import ScreenplaySkill
from tests.screenplay_fixtures import (
    SequenceLLM,
    skill_inputs,
    valid_screenplay_dict,
    valid_screenplay_json,
)


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

    def test_parse_screenplay_accepts_common_screenplay_wrapper(self):
        wrapped = json.dumps({"screenplay": valid_screenplay_dict()}, ensure_ascii=False)

        screenplay = parse_screenplay(wrapped)

        self.assertEqual(screenplay.title, "最后一课")


if __name__ == "__main__":
    unittest.main()
