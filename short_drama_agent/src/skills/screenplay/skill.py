from typing import Any

from pydantic import BaseModel

from src.llm.base import LLM

from .prompts import build_generation_messages, build_repair_messages
from .repair import JsonRepairExhaustedError, ScreenplayGenerationError, parse_screenplay
from .schemas import GenerationConstraints, Screenplay


class ScreenplaySkillResult(BaseModel):
    screenplay: Screenplay
    json_repair_attempts: int


class ScreenplaySkill:
    def __init__(self, llm: LLM, max_json_repair_attempts: int = 2):
        if max_json_repair_attempts < 0:
            raise ValueError("max_json_repair_attempts cannot be negative")
        self.llm = llm
        self.max_json_repair_attempts = max_json_repair_attempts

    def generate(
        self,
        story_plan: dict[str, Any],
        requirement: dict[str, Any],
        constraints: GenerationConstraints,
        writing_guidelines: list[Any],
        user_preferences: dict[str, Any] | None,
    ) -> ScreenplaySkillResult:
        messages = build_generation_messages(
            story_plan,
            requirement,
            constraints,
            writing_guidelines,
            user_preferences,
        )
        response = self.llm.generate(messages)
        for repair_attempts in range(self.max_json_repair_attempts + 1):
            try:
                return ScreenplaySkillResult(
                    screenplay=parse_screenplay(response),
                    json_repair_attempts=repair_attempts,
                )
            except ScreenplayGenerationError as exc:
                if repair_attempts == self.max_json_repair_attempts:
                    raise JsonRepairExhaustedError(str(exc)) from exc
                response = self.llm.generate(build_repair_messages(response, str(exc)))
        raise AssertionError("unreachable")
