import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.evaluation.schemas import ReviewReport
from src.memory.project_memory import ProjectSummary
from src.skills.screenplay.schemas import Screenplay


class ExportPaths(BaseModel):
    screenplay_json: Path
    screenplay_markdown: Path
    review_json: Path
    retrieval_trace_json: Path
    summary_json: Path


def export_project(
    output_root: str | Path,
    screenplay: Screenplay,
    review_report: ReviewReport,
    retrieval_trace: list[Any],
    summary: ProjectSummary,
) -> ExportPaths:
    project_dir = Path(output_root) / summary.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    paths = ExportPaths(
        screenplay_json=project_dir / "screenplay.json",
        screenplay_markdown=project_dir / "screenplay.md",
        review_json=project_dir / "review.json",
        retrieval_trace_json=project_dir / "retrieval_trace.json",
        summary_json=project_dir / "summary.json",
    )
    _write_json(paths.screenplay_json, screenplay.model_dump())
    paths.screenplay_markdown.write_text(_screenplay_markdown(screenplay), encoding="utf-8")
    _write_json(paths.review_json, review_report.model_dump())
    _write_json(
        paths.retrieval_trace_json,
        [item.model_dump() if hasattr(item, "model_dump") else item for item in retrieval_trace],
    )
    _write_json(paths.summary_json, summary.model_dump())
    return paths


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _screenplay_markdown(screenplay: Screenplay) -> str:
    lines = [
        f"# {screenplay.title}",
        "",
        f"- 类型：{screenplay.genre}",
        f"- 一句话梗概：{screenplay.logline}",
        f"- 主题：{screenplay.theme}",
        f"- 开头钩子：{screenplay.opening_hook}",
        f"- 核心冲突：{screenplay.main_conflict}",
        f"- 反转：{screenplay.reversal}",
        "",
    ]
    for scene in screenplay.scenes:
        lines.extend(
            [
                f"## 场景 {scene.scene_id}：{scene.location} / {scene.time}",
                "",
                scene.visual_action,
                "",
            ]
        )
        for dialogue in scene.dialogues:
            lines.append(f"**{dialogue.character}**：{dialogue.line}")
        if scene.ending_hook:
            lines.extend(["", f"结尾钩子：{scene.ending_hook}"])
        lines.append("")
    lines.extend(["## 结局", "", screenplay.ending, ""])
    return "\n".join(lines)
