import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    genre: str
    logline: str
    user_feedback: str | None
    final_score: float
    output_paths: list[str]


class ProjectRecord(BaseModel):
    project_id: str
    thread_id: str
    user_id: str
    user_request: str
    status: str
    summary: ProjectSummary | None = None


class ProjectMemoryRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

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
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_project(
        self,
        project_id: str,
        thread_id: str,
        user_id: str,
        user_request: str,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO projects
                    (project_id, thread_id, user_id, user_request, status)
                VALUES (?, ?, ?, ?, 'created')
                """,
                (project_id, thread_id, user_id, user_request),
            )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def record_export(self, summary: ProjectSummary) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE projects SET
                    status = 'exported',
                    summary_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                """,
                (json.dumps(summary.model_dump(), ensure_ascii=False), summary.project_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown project: {summary.project_id}")

    def history(self, user_id: str) -> list[ProjectSummary]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT summary_json FROM projects
                WHERE user_id = ? AND summary_json IS NOT NULL
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [ProjectSummary.model_validate_json(row["summary_json"]) for row in rows]

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ProjectRecord:
        summary = (
            ProjectSummary.model_validate_json(row["summary_json"])
            if row["summary_json"]
            else None
        )
        return ProjectRecord(
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            user_request=row["user_request"],
            status=row["status"],
            summary=summary,
        )
