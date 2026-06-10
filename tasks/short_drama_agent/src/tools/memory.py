from typing import Any

from src.memory.project_memory import ProjectMemoryRepository, ProjectSummary
from src.memory.user_memory import UserMemoryRepository, UserPreferences


class MemoryTool:
    def __init__(
        self,
        user_repository: UserMemoryRepository,
        project_repository: ProjectMemoryRepository,
    ):
        self.user_repository = user_repository
        self.project_repository = project_repository

    def record_outline_feedback(
        self,
        user_id: str,
        explicit_preferences: dict[str, Any],
    ) -> UserPreferences:
        return self.user_repository.update_explicit(user_id, explicit_preferences)

    def record_export(self, summary: ProjectSummary) -> None:
        self.project_repository.record_export(summary)
