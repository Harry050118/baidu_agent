import argparse
import json
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="短剧生成 Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="创建短剧项目")
    create.add_argument("--request", required=True, help="创作需求")
    create.add_argument("--user-id", default="default")
    create.add_argument("--duration", type=int)
    create.add_argument("--min-scenes", type=int)
    create.add_argument("--max-scenes", type=int)

    resume = subparsers.add_parser("resume", help="恢复已有工作流")
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--action", choices=["approve", "revise", "pause"])
    resume.add_argument("--feedback")
    resume.add_argument("--preferred-genre", action="append")
    resume.add_argument("--preferred-tone", action="append")
    resume.add_argument("--preferred-ending", action="append")
    resume.add_argument("--dialogue-style")
    resume.add_argument("--production-constraint", action="append")

    history = subparsers.add_parser("history", help="查看用户项目历史")
    history.add_argument("--user-id", default="default")
    return parser


def run_cli(argv: Sequence[str], app: Any) -> str:
    args = build_parser().parse_args(argv)
    result = _dispatch(args, app)
    return _format_output(result)


def _dispatch(args: argparse.Namespace, app: Any) -> Any:
    if args.command == "create":
        constraints = {
            key: value
            for key, value in {
                "target_duration_seconds": args.duration,
                "min_scene_count": args.min_scenes,
                "max_scene_count": args.max_scenes,
            }.items()
            if value is not None
        }
        return app.create(
            request=args.request,
            user_id=args.user_id,
            constraints=constraints,
        )
    if args.command == "resume":
        explicit_preferences = {
            key: value
            for key, value in {
                "preferred_genres": args.preferred_genre,
                "preferred_tones": args.preferred_tone,
                "preferred_endings": args.preferred_ending,
                "dialogue_style": args.dialogue_style,
                "production_constraints": args.production_constraint,
            }.items()
            if value is not None
        }
        return app.resume(
            thread_id=args.thread_id,
            action=args.action,
            feedback=args.feedback,
            explicit_preferences=explicit_preferences or None,
        )
    return app.history(user_id=args.user_id)


def _format_output(result: Any) -> str:
    if isinstance(result, list):
        return "\n".join(
            f"{item.get('title', item.get('project_id', ''))} | "
            f"{item.get('genre', '')} | {item.get('final_score', '')}"
            for item in result
        )
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from src.app.runtime import build_production_app
    from src.config import load_config

    app = build_production_app(load_config("config/default.yaml"))
    print(_format_output(_dispatch(args, app)))
    return 0
