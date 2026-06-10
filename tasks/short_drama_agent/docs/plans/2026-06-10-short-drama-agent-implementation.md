# Short Drama Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI short-drama Agent that extends its own independent RAG baseline, pauses for outline review, generates validated structured screenplays, reviews and revises content, persists memory, and exports the best result.

**Architecture:** Keep the task-local `src/rag` public API stable and add a `WritingKnowledgeBase` adapter. Use focused Pydantic models for contracts, a standalone `ScreenplaySkill` for structured generation and JSON repair, and a LangGraph workflow for orchestration, interrupt/resume, review routing, and export. Dependencies are injected so tests use fake LLM and embedding implementations without network access.

**Tech Stack:** Python 3.10+, Pydantic 2, LangGraph, PyYAML, OpenAI-compatible DeepSeek client, SQLite, unittest.

---

## File Map

**Entry and configuration**

- Create: `task2_main.py` - thin executable entry point.
- Create: `.env.example` - documented environment variables without secrets.
- Create: `config/default.yaml` - Agent, RAG, generation, review, and storage defaults.
- Create: `src/app/cli.py` - `create`, `resume`, and `history` commands.
- Create: `src/config.py` - YAML and environment configuration loader.
- Modify: `requirements.txt` - add Pydantic, LangGraph, SQLite checkpointer, and PyYAML.

**LLM and shared models**

- Create: `src/llm/base.py` - protocol for replaceable LLM clients.
- Create: `src/llm/deepseek.py` - adapter around the existing OpenAI-compatible client.
- Modify: `src/llm/__init__.py` - export new interfaces while retaining `LLMClient`.

**RAG adapter**

- Create: `src/rag/schemas.py` - `RetrievedGuideline`.
- Create: `src/rag/knowledge_base.py` - task-local RAG adapter, stable chunk IDs, query parameter validation.
- Create: `src/rag/query_builder.py` - planning, writing, and review queries.
- Modify: `src/rag/__init__.py` - export adapter types without breaking existing exports.

**Screenplay Skill**

- Create: `src/skills/screenplay/schemas.py` - screenplay and generation constraint models.
- Create: `src/skills/screenplay/repair.py` - JSON extraction and repair exceptions.
- Create: `src/skills/screenplay/skill.py` - structured screenplay generation.
- Create: `src/skills/screenplay/prompts.py` - generation and repair prompts.
- Create: package `__init__.py` files under `src/skills/` and `src/skills/screenplay/`.

**Review, memory, tools, workflow**

- Create: `src/evaluation/schemas.py`, `rubric.py`, `reviewer.py`.
- Create: `src/memory/user_memory.py`, `project_memory.py`.
- Create: `src/tools/export.py`, `retrieve.py`, `memory.py`.
- Create: `src/agent/state.py`, `nodes.py`, `router.py`, `graph.py`.
- Create: package `__init__.py` files for each package.

**Knowledge and tests**

- Create: `rag_docs_short_drama/*.md` - four focused writing-method documents.
- Create: `tests/` files listed in the approved design.

---

### Task 1: Configuration and Replaceable LLM Contract

**Files:**
- Create: `.env.example`
- Create: `config/default.yaml`
- Create: `src/config.py`
- Create: `src/llm/base.py`
- Create: `src/llm/deepseek.py`
- Modify: `src/llm/__init__.py`
- Modify: `requirements.txt`
- Test: `tests/test_config_and_llm.py`

- [ ] **Step 1: Write the failing configuration and LLM contract tests**

```python
import unittest

from src.config import load_config
from src.llm.base import LLM


class EchoLLM:
    def generate(self, messages, *, temperature=0.7):
        return messages[-1]["content"]


class ConfigAndLLMTests(unittest.TestCase):
    def test_default_config_contains_generation_and_storage_defaults(self):
        config = load_config("config/default.yaml")
        self.assertEqual(config["generation"]["target_duration_seconds"], 240)
        self.assertEqual(config["review"]["max_content_revisions"], 2)
        self.assertIn("checkpoint_db", config["storage"])

    def test_fake_implementation_satisfies_llm_protocol(self):
        llm: LLM = EchoLLM()
        self.assertEqual(llm.generate([{"role": "user", "content": "hello"}]), "hello")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_config_and_llm -v`

Expected: FAIL because `src.config` and `src.llm.base` do not exist.

- [ ] **Step 3: Add dependencies and minimal configuration files**

Add to `requirements.txt`:

```text
pydantic>=2.7.0
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=2.0.0
PyYAML>=6.0.0
```

Create `.env.example` with `DEEPSEEK_API_KEY=`.

Create `config/default.yaml` with defaults from the approved design, including RAG, generation, review, and SQLite paths.

- [ ] **Step 4: Implement minimal config loader and LLM protocol**

```python
class LLM(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
    ) -> str:
        raise NotImplementedError
```

`DeepSeekLLM` must adapt this method to the existing `LLMClient.chat()` method.

- [ ] **Step 5: Run tests and task-local RAG regression suite**

Run: `python -m unittest tests.test_config_and_llm tests.test_rag_baseline -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add .env.example config/default.yaml requirements.txt src/config.py src/llm tests/test_config_and_llm.py
git commit -m "Add agent configuration and LLM contract"
```

### Task 2: Existing-RAG Writing Knowledge Adapter

**Files:**
- Create: `src/rag/schemas.py`
- Create: `src/rag/knowledge_base.py`
- Create: `src/rag/query_builder.py`
- Modify: `src/rag/__init__.py`
- Test: `tests/test_existing_rag_compatibility.py`
- Test: `tests/test_writing_knowledge_base.py`

- [ ] **Step 1: Write failing adapter tests**

```python
class WritingKnowledgeBaseTests(unittest.TestCase):
    def test_search_forwards_candidate_and_top_k_and_generates_chunk_id(self):
        retriever = RecordingRetriever()
        kb = WritingKnowledgeBase(retriever=retriever)
        result = kb.search("冲突设计", "planning", candidate_k=7, top_k=2)
        self.assertEqual(retriever.last_query, ("冲突设计", 7, 2))
        self.assertTrue(result[0].chunk_id)

    def test_missing_titles_become_none(self):
        guideline = guideline_from_result(result_without_titles, "q", "writing")
        self.assertIsNone(guideline.document_title)

    def test_top_k_cannot_exceed_candidate_k(self):
        with self.assertRaises(ValueError):
            kb.search("q", "review", candidate_k=2, top_k=3)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_writing_knowledge_base -v`

Expected: FAIL because adapter modules do not exist.

- [ ] **Step 3: Implement `RetrievedGuideline` and adapter**

Use SHA-256 over normalized `source`, optional title hierarchy, and chunk text. `WritingKnowledgeBase` must generate the ID when the task-local RAG result has none and call:

```python
self.retriever.query(query, candidate_k=candidate_k, top_k=top_k)
```

- [ ] **Step 4: Add query builders**

Create deterministic functions:

```python
def build_planning_query(requirement: dict) -> str:
    return f"{requirement['genre']}短剧的开头钩子、人物目标、核心冲突与反转设计"


def build_writing_query(story_plan: dict) -> str:
    return f"{story_plan['genre']}短剧的场景节奏、视觉动作、对白与低成本可拍性"


def build_review_query(screenplay: dict) -> str:
    return f"{screenplay['genre']}短剧的结构、节奏、人物一致性与可拍性评价标准"
```

- [ ] **Step 5: Run adapter and task-local RAG tests**

Run: `python -m unittest tests.test_writing_knowledge_base tests.test_existing_rag_compatibility tests.test_rag_baseline -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/rag tests/test_existing_rag_compatibility.py tests/test_writing_knowledge_base.py
git commit -m "Add short drama writing knowledge adapter"
```

### Task 3: Screenplay Schemas and Deterministic Validation

**Files:**
- Create: `src/skills/__init__.py`
- Create: `src/skills/screenplay/__init__.py`
- Create: `src/skills/screenplay/schemas.py`
- Test: `tests/test_screenplay_schema.py`

- [ ] **Step 1: Write failing schema tests**

```python
class ScreenplaySchemaTests(unittest.TestCase):
    def test_scene_supports_visual_action_and_optional_shooting_note(self):
        scene = Scene(**valid_scene())
        self.assertEqual(scene.visual_action, "她推开门，看见桌上的录音笔。")
        self.assertIsNone(scene.shooting_note)

    def test_constraints_are_defaults_not_schema_constants(self):
        constraints = GenerationConstraints()
        self.assertEqual(constraints.min_duration_seconds, 180)
        custom = constraints.model_copy(update={"max_scene_count": 8})
        self.assertEqual(custom.max_scene_count, 8)

    def test_deterministic_validation_rejects_unknown_dialogue_character(self):
        errors = validate_screenplay(valid_screenplay_with_unknown_speaker(), GenerationConstraints())
        self.assertTrue(any("角色" in error for error in errors))
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_screenplay_schema -v`

Expected: FAIL because screenplay schemas do not exist.

- [ ] **Step 3: Implement Pydantic schemas and validation**

Implement `Character`, `Dialogue`, `Scene`, `Screenplay`, `GenerationConstraints`, and:

```python
def validate_screenplay(
    screenplay: Screenplay,
    constraints: GenerationConstraints,
) -> list[str]:
    errors: list[str] = []
    duration = sum(scene.estimated_seconds for scene in screenplay.scenes)
    if not constraints.min_duration_seconds <= duration <= constraints.max_duration_seconds:
        errors.append("总时长不符合约束")
    return errors
```

Validate duration, scene count, speaker references, continuous unique IDs, hook presence, and reversal.

- [ ] **Step 4: Run schema tests**

Run: `python -m unittest tests.test_screenplay_schema -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/skills tests/test_screenplay_schema.py
git commit -m "Add structured screenplay schemas"
```

### Task 4: JSON Repair and Screenplay Skill

**Files:**
- Create: `src/skills/screenplay/prompts.py`
- Create: `src/skills/screenplay/repair.py`
- Create: `src/skills/screenplay/skill.py`
- Test: `tests/test_json_repair.py`
- Test: `tests/test_screenplay_skill.py`

- [ ] **Step 1: Write failing JSON repair tests**

```python
class JsonRepairTests(unittest.TestCase):
    def test_skill_reports_repair_attempts_after_invalid_first_response(self):
        llm = SequenceLLM(["not json", valid_screenplay_json()])
        result = ScreenplaySkill(llm, max_json_repair_attempts=2).generate(**inputs())
        self.assertEqual(result.json_repair_attempts, 1)

    def test_skill_raises_typed_error_after_repair_limit(self):
        llm = SequenceLLM(["bad", "still bad", "bad again"])
        with self.assertRaises(JsonRepairExhaustedError):
            ScreenplaySkill(llm, max_json_repair_attempts=2).generate(**inputs())
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_json_repair tests.test_screenplay_skill -v`

Expected: FAIL because Skill modules do not exist.

- [ ] **Step 3: Implement typed errors and minimal JSON extraction**

Implement:

```python
class ScreenplayGenerationError(RuntimeError):
    pass


class JsonRepairExhaustedError(ScreenplayGenerationError):
    pass

class ScreenplaySkillResult(BaseModel):
    screenplay: Screenplay
    json_repair_attempts: int
```

The Skill returns only success results. Exhausted repair attempts must raise `JsonRepairExhaustedError`.

- [ ] **Step 4: Implement generation and repair prompts**

The repair prompt must instruct the model to preserve valid content and only fix JSON parsing, types, and missing required fields.

- [ ] **Step 5: Run Skill tests**

Run: `python -m unittest tests.test_json_repair tests.test_screenplay_skill tests.test_screenplay_schema -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/skills tests/test_json_repair.py tests/test_screenplay_skill.py
git commit -m "Add structured screenplay generation skill"
```

### Task 5: Reviewer and Best-Version Selection

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/schemas.py`
- Create: `src/evaluation/rubric.py`
- Create: `src/evaluation/reviewer.py`
- Test: `tests/test_reviewer.py`

- [ ] **Step 1: Write failing reviewer tests**

```python
class ReviewerTests(unittest.TestCase):
    def test_passed_requires_score_and_no_deterministic_errors(self):
        report = build_review_report(score_payload(9.0), ["scene ids invalid"], pass_score=8.0)
        self.assertFalse(report.passed)

    def test_higher_score_replaces_best_version(self):
        best, report = choose_best(old_screenplay, old_report, new_screenplay, new_report)
        self.assertIs(best, new_screenplay)

    def test_equal_score_prefers_fewer_deterministic_errors(self):
        best, report = choose_best(old_screenplay, old_report, new_screenplay, cleaner_report)
        self.assertIs(best, new_screenplay)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_reviewer -v`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement `ReviewReport`, pass calculation, and best selection**

Ensure `passed` is computed by code:

```python
passed = total_score >= pass_score and not deterministic_errors
```

- [ ] **Step 4: Implement LLM reviewer adapter**

`ScreenplayReviewer.review()` must combine deterministic errors with parsed LLM score output and return `ReviewReport`.

- [ ] **Step 5: Run reviewer tests**

Run: `python -m unittest tests.test_reviewer tests.test_screenplay_schema -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/evaluation tests/test_reviewer.py
git commit -m "Add screenplay quality reviewer"
```

### Task 6: SQLite Memory Repositories and Write Timing

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/user_memory.py`
- Create: `src/memory/project_memory.py`
- Create: `src/tools/memory.py`
- Test: `tests/test_memory_repository.py`

- [ ] **Step 1: Write failing memory timing tests**

```python
class MemoryRepositoryTests(unittest.TestCase):
    def test_create_project_writes_initial_record_without_inferred_preferences(self):
        repo.create_project("p1", "t1", "u1", "校园悬疑")
        self.assertEqual(repo.get_project("p1").status, "created")
        self.assertIsNone(user_repo.get("u1"))

    def test_explicit_outline_feedback_updates_preferences(self):
        memory_tool.record_outline_feedback("u1", {"preferred_genres": ["悬疑"]})
        self.assertEqual(user_repo.get("u1").preferred_genres, ["悬疑"])

    def test_successful_export_writes_project_summary(self):
        memory_tool.record_export(summary)
        self.assertEqual(repo.history("u1")[0].title, summary.title)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_memory_repository -v`

Expected: FAIL because memory modules do not exist.

- [ ] **Step 3: Implement minimal SQLite repositories**

Use `sqlite3` from the standard library. Provide explicit methods for project creation, explicit preference updates, successful-export summaries, and history reads. Do not infer preferences from model output.

- [ ] **Step 4: Run memory tests**

Run: `python -m unittest tests.test_memory_repository -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/memory src/tools/memory.py tests/test_memory_repository.py
git commit -m "Add project and user memory repositories"
```

### Task 7: Export Tools and Short-Drama Knowledge Documents

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/export.py`
- Create: `src/tools/retrieve.py`
- Create: `rag_docs_short_drama/01_短剧结构与节奏.md`
- Create: `rag_docs_short_drama/02_人物与冲突设计.md`
- Create: `rag_docs_short_drama/03_对白与场景写作.md`
- Create: `rag_docs_short_drama/04_短剧质量审查标准.md`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write failing export test**

```python
class ExportTests(unittest.TestCase):
    def test_export_writes_best_screenplay_review_and_trace(self):
        paths = export_project(tmp_path, best_screenplay, best_report, retrieval_trace, summary)
        self.assertTrue(paths.screenplay_json.exists())
        self.assertIn(best_screenplay.title, paths.screenplay_markdown.read_text(encoding="utf-8"))
        self.assertTrue(paths.retrieval_trace_json.exists())
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_export -v`

Expected: FAIL because export module does not exist.

- [ ] **Step 3: Implement deterministic JSON and Markdown export**

Write files under `output/projects/<project_id>/` and return typed paths. Export must receive the selected best version, not select it internally.

- [ ] **Step 4: Add focused RAG documents**

Each Markdown file must use clear `#`, `##`, and `###` headings so the existing hierarchy-aware chunker preserves retrieval metadata.

- [ ] **Step 5: Run export and RAG tests**

Run: `python -m unittest tests.test_export tests.test_writing_knowledge_base tests.test_rag_baseline -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/tools rag_docs_short_drama tests/test_export.py
git commit -m "Add screenplay exports and writing knowledge"
```

### Task 8: Agent State, Nodes, and Routing

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/state.py`
- Create: `src/agent/router.py`
- Create: `src/agent/nodes.py`
- Test: `tests/test_graph_routing.py`

- [ ] **Step 1: Write failing routing and node tests**

```python
class AgentRoutingTests(unittest.TestCase):
    def test_screenplay_node_accumulates_skill_json_repairs(self):
        state = base_state(json_repair_count=2)
        update = screenplay_node(state, skill=SuccessfulSkill(repairs=1))
        self.assertEqual(update["json_repair_count"], 3)

    def test_screenplay_node_records_typed_skill_failure(self):
        update = screenplay_node(base_state(), skill=FailingSkill())
        self.assertEqual(update["errors"][0]["error_type"], "JsonRepairExhaustedError")

    def test_review_router_stops_at_revision_limit_and_exports_best(self):
        state = reviewed_state(passed=False, content_revision_count=2)
        self.assertEqual(route_after_review(state), "export")

    def test_human_review_actions_have_three_explicit_values(self):
        self.assertEqual({item.value for item in HumanReviewAction}, {"approve", "revise", "pause"})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_graph_routing -v`

Expected: FAIL because Agent modules do not exist.

- [ ] **Step 3: Implement state contracts and routers**

Use serializable dictionaries in `AgentState`. Implement `HumanReviewAction`, `route_after_human_review()`, and `route_after_review()`.

- [ ] **Step 4: Implement dependency-injected nodes**

Nodes must return partial state updates. `ScreenplaySkillNode` catches typed Skill errors and accumulates `json_repair_count`; `ReviewNode` updates current and best versions; `ReviseNode` increments only `content_revision_count`.

- [ ] **Step 5: Run routing and component tests**

Run: `python -m unittest tests.test_graph_routing tests.test_screenplay_skill tests.test_reviewer -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/agent tests/test_graph_routing.py
git commit -m "Add short drama agent nodes and routing"
```

### Task 9: LangGraph Interrupt, Resume, and Workflow Assembly

**Files:**
- Create: `src/agent/graph.py`
- Test: `tests/test_graph_flow.py`

- [ ] **Step 1: Write failing graph-flow tests**

```python
class GraphFlowTests(unittest.TestCase):
    def test_graph_interrupts_for_outline_review_and_resumes_on_approve(self):
        graph = build_test_graph(dependencies, checkpointer=MemorySaver())
        first = graph.invoke(initial_state(), config={"configurable": {"thread_id": "t1"}})
        self.assertTrue(first["__interrupt__"])
        resumed = graph.invoke(Command(resume={"action": "approve"}), thread_config("t1"))
        self.assertIn("screenplay", resumed)

    def test_pause_keeps_thread_resumable_without_advancing(self):
        graph = build_test_graph(dependencies, checkpointer=MemorySaver())
        graph.invoke(initial_state(), config=thread_config("t2"))
        paused = graph.invoke(Command(resume={"action": "pause"}), thread_config("t2"))
        self.assertIsNone(paused.get("screenplay"))

    def test_revise_returns_to_story_planning(self):
        graph = build_test_graph(dependencies, checkpointer=MemorySaver())
        graph.invoke(initial_state(), config=thread_config("t3"))
        revised = graph.invoke(
            Command(resume={"action": "revise", "feedback": "加强结尾反转"}),
            thread_config("t3"),
        )
        self.assertEqual(revised["user_feedback"], "加强结尾反转")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_graph_flow -v`

Expected: FAIL because graph assembly does not exist.

- [ ] **Step 3: Assemble LangGraph**

Add nodes and conditional edges matching the approved workflow. Use `interrupt()` for outline review and accept `Command(resume={"action": action, "feedback": feedback})`. Start with an in-memory checkpointer in tests and expose SQLite checkpointer construction for CLI use.

- [ ] **Step 4: Run graph tests**

Run: `python -m unittest tests.test_graph_flow tests.test_graph_routing -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/agent/graph.py tests/test_graph_flow.py
git commit -m "Assemble interruptible short drama workflow"
```

### Task 10: CLI Create, Resume, and History

**Files:**
- Create: `src/app/__init__.py`
- Create: `src/app/cli.py`
- Create: `task2_main.py`
- Test: `tests/test_cli_flow.py`

- [ ] **Step 1: Write failing CLI tests**

```python
class CliFlowTests(unittest.TestCase):
    def test_create_passes_request_and_constraints_to_agent(self):
        result = run_cli(["create", "--request", "校园悬疑", "--duration", "360"], fake_app)
        self.assertEqual(fake_app.created["target_duration_seconds"], 360)

    def test_resume_uses_existing_thread_id(self):
        run_cli(["resume", "--thread-id", "t1"], fake_app)
        self.assertEqual(fake_app.resumed_thread_id, "t1")

    def test_history_lists_user_projects(self):
        output = run_cli(["history", "--user-id", "u1"], fake_app)
        self.assertIn("测试短剧", output)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_cli_flow -v`

Expected: FAIL because CLI modules do not exist.

- [ ] **Step 3: Implement CLI parser and thin entry point**

`task2_main.py` must only call `src.app.cli.main()`. CLI must load `config/default.yaml`, instantiate production dependencies, and support `create`, `resume`, and `history`.

- [ ] **Step 4: Run CLI tests and help smoke test**

Run: `python -m unittest tests.test_cli_flow -v`

Run: `python task2_main.py --help`

Expected: tests PASS and help lists all three commands.

- [ ] **Step 5: Commit**

```powershell
git add src/app task2_main.py tests/test_cli_flow.py
git commit -m "Add short drama agent CLI"
```

### Task 11: Full Verification, Documentation, and Push

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-06-10-short-drama-agent-design.md` only if implementation discovered a necessary contract correction

- [ ] **Step 1: Add usage documentation**

Document setup, `.env` configuration, the three CLI commands, output artifacts, RAG reuse, JSON Repair versus Content Revision, and offline test commands.

- [ ] **Step 2: Run complete offline test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with no network calls.

- [ ] **Step 3: Run existing task-one regression tests**

Run: `python -m unittest tests/test_rag_baseline.py -v`

Expected: 3 tests PASS.

- [ ] **Step 4: Run syntax and CLI smoke checks**

Run: `python -m compileall src task2_main.py`

Run: `python task2_main.py --help`

Expected: both commands exit 0.

- [ ] **Step 5: Inspect repository state**

Run: `git status --short`

Run: `git diff --check`

Expected: only intended documentation changes remain and no whitespace errors.

- [ ] **Step 6: Commit final documentation**

```powershell
git add README.md docs
git commit -m "Document short drama agent workflow"
```

- [ ] **Step 7: Push verified implementation**

Run: `git push origin main`

Expected: local `main` and `origin/main` point to the same verified commit.
