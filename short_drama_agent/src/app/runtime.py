import json
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from src.agent.graph import GraphDependencies, build_graph, sqlite_checkpointer
from src.evaluation.reviewer import ScreenplayReviewer
from src.evaluation.schemas import ReviewReport
from src.llm import DeepSeekLLM, LLMClient, RetryingLLM
from src.memory.project_memory import ProjectMemoryRepository, ProjectSummary
from src.memory.user_memory import UserMemoryRepository
from src.rag import Chunker, EmbeddingModel, Retriever, WritingKnowledgeBase
from src.skills.screenplay.repair import extract_json_text
from src.skills.screenplay.schemas import GenerationConstraints, Screenplay
from src.skills.screenplay.skill import ScreenplaySkill
from src.tools.export import export_project
from src.tools.memory import MemoryTool
from src.tools.retrieve import retrieve_guidelines


class ShortDramaApplication:
    def __init__(
        self,
        graph: Any,
        projects: ProjectMemoryRepository,
        users: UserMemoryRepository,
        memory: MemoryTool,
        default_constraints: dict,
    ):
        self.graph = graph
        self.projects = projects
        self.users = users
        self.memory = memory
        self.default_constraints = default_constraints

    def create(self, *, request: str, user_id: str, constraints: dict) -> dict:
        project_id = uuid.uuid4().hex
        thread_id = uuid.uuid4().hex
        merged_constraints = {**self.default_constraints, **constraints}
        self.projects.create_project(project_id, thread_id, user_id, request)
        state = {
            "project_id": project_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "user_request": request,
            "requirement": {"genre": "未指定", "request": request},
            "constraints": merged_constraints,
            "user_preferences": _model_dump_or_none(self.users.get(user_id)),
            "planning_guidelines": [],
            "writing_guidelines": [],
            "review_guidelines": [],
            "json_repair_count": 0,
            "content_revision_count": 0,
            "errors": [],
        }
        return self.graph.invoke(state, config=_thread_config(thread_id))

    def resume(
        self,
        *,
        thread_id: str,
        action: str | None = None,
        feedback: str | None = None,
        explicit_preferences: dict | None = None,
    ) -> dict:
        updated_preferences = None
        if action == "revise" and explicit_preferences:
            project = self.projects.get_project_by_thread(thread_id)
            if project is None:
                raise KeyError(f"Unknown thread: {thread_id}")
            updated_preferences = self.memory.record_outline_feedback(
                project.user_id,
                explicit_preferences,
            )
        resume = {"action": action or "pause"}
        if feedback is not None:
            resume["feedback"] = feedback
        if updated_preferences is not None:
            resume["user_preferences"] = updated_preferences.model_dump()
        return self.graph.invoke(Command(resume=resume), config=_thread_config(thread_id))

    def history(self, *, user_id: str) -> list[dict]:
        return [summary.model_dump() for summary in self.projects.history(user_id)]


def build_production_app(config: dict[str, Any]) -> ShortDramaApplication:
    llm_config = config["llm"]
    base_llm = DeepSeekLLM(
        LLMClient(
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config["base_url"],
            model=llm_config["model"],
        )
    )
    retry_config = config.get("retry", {})
    llm = RetryingLLM(
        base_llm,
        max_attempts=retry_config.get("max_attempts", 3),
        initial_delay_seconds=retry_config.get("initial_delay_seconds", 1.0),
        backoff_multiplier=retry_config.get("backoff_multiplier", 2.0),
    )
    constraints = {
        **config["generation"],
        **config["review"],
    }
    projects = ProjectMemoryRepository(config["storage"]["memory_db"])
    users = UserMemoryRepository(config["storage"]["memory_db"])
    memory = MemoryTool(users, projects)
    retriever = _build_guideline_retriever(config)
    dependencies = GraphDependencies(
        retriever=retriever,
        story_planner=_StoryPlanner(llm),
        screenplay_skill=ScreenplaySkill(
            llm,
            max_json_repair_attempts=constraints["max_json_repair_attempts"],
        ),
        reviewer=ScreenplayReviewer(llm),
        exporter=_Exporter(config["storage"]["output_dir"], memory),
    )
    graph = build_graph(
        dependencies,
        checkpointer=sqlite_checkpointer(config["storage"]["checkpoint_db"]),
    )
    return ShortDramaApplication(graph, projects, users, memory, constraints)


class _StoryPlanner:
    def __init__(self, llm: Any):
        self.llm = llm

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        response = self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "根据需求生成短剧故事大纲，仅输出 JSON 对象。",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": state["user_request"],
                            "feedback": state.get("user_feedback"),
                            "planning_guidelines": state.get("planning_guidelines", []),
                            "user_preferences": state.get("user_preferences"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        return {"story_plan": json.loads(extract_json_text(response))}


class _Exporter:
    def __init__(self, output_dir: str, memory: MemoryTool):
        self.output_dir = Path(output_dir)
        self.memory = memory

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        screenplay = Screenplay.model_validate(state["best_screenplay"])
        report = ReviewReport.model_validate(state["best_review_report"])
        summary = ProjectSummary(
            project_id=state["project_id"],
            title=screenplay.title,
            genre=screenplay.genre,
            logline=screenplay.logline,
            user_feedback=state.get("user_feedback"),
            final_score=report.total_score,
            output_paths=[],
        )
        paths = export_project(
            self.output_dir,
            screenplay,
            report,
            [
                *state.get("planning_guidelines", []),
                *state.get("writing_guidelines", []),
                *state.get("review_guidelines", []),
            ],
            summary,
        )
        summary.output_paths = [str(path) for path in paths.model_dump().values()]
        export_project(
            self.output_dir,
            screenplay,
            report,
            [
                *state.get("planning_guidelines", []),
                *state.get("writing_guidelines", []),
                *state.get("review_guidelines", []),
            ],
            summary,
        )
        self.memory.record_export(summary)
        return {
            "final_json_path": str(paths.screenplay_json),
            "final_markdown_path": str(paths.screenplay_markdown),
        }


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _model_dump_or_none(value: Any) -> dict[str, Any] | None:
    return value.model_dump() if value is not None else None


def _build_guideline_retriever(config: dict[str, Any]):
    rag_config = config["rag"]
    try:
        knowledge_base = WritingKnowledgeBase(
            Retriever(EmbeddingModel(rag_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))),
            Chunker(
                chunk_size=rag_config.get("chunk_size", 500),
                overlap=rag_config.get("chunk_overlap", 50),
            ),
        )
        knowledge_base.build(rag_config["documents_path"])
    except Exception as exc:
        def failed_retriever(query: str, purpose: str, error: Exception = exc):
            raise RuntimeError(f"RAG initialization failed: {error}") from error

        return failed_retriever

    def search(query: str, purpose: str):
        return retrieve_guidelines(
            knowledge_base,
            query,
            purpose,
            candidate_k=rag_config["candidate_k"],
            top_k=rag_config["top_k"],
        )

    return search
