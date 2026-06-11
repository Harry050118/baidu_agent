import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    preferred_genres: list[str] = Field(default_factory=list)
    preferred_tones: list[str] = Field(default_factory=list)
    preferred_endings: list[str] = Field(default_factory=list)
    dialogue_style: str | None = None
    production_constraints: list[str] = Field(default_factory=list)


class UserMemoryRepository:
    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, user_id: str) -> UserPreferences | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT preferences_json FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return UserPreferences.model_validate_json(row[0]) if row else None

    def update_explicit(self, user_id: str, updates: dict[str, Any]) -> UserPreferences:
        current = self.get(user_id) or UserPreferences()
        unknown = set(updates) - set(UserPreferences.model_fields)
        if unknown:
            raise ValueError(f"Unknown preference fields: {', '.join(sorted(unknown))}")
        updated = current.model_copy(update=updates)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO user_preferences (user_id, preferences_json)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences_json = excluded.preferences_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, json.dumps(updated.model_dump(), ensure_ascii=False)),
            )
        return updated
