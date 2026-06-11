# 短剧生成 Agent 实施计划

> **供 Agent 执行者使用：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。步骤使用复选框（`- [x]`）跟踪。

**目标：** 构建一个命令行短剧 Agent：扩展自身独立的 RAG 基线，在大纲审阅处暂停，生成经过校验的结构化剧本，审查并修订内容，持久化记忆，最后导出最佳结果。

**架构：** 保持任务目录内 `src/rag` 的公共 API 稳定，并增加 `WritingKnowledgeBase` 适配器。使用职责明确的 Pydantic 模型定义契约，使用独立 `ScreenplaySkill` 完成结构化生成与 JSON 修复，并使用 LangGraph 工作流负责编排、中断/恢复、审查路由和导出。依赖通过注入提供，使测试可使用虚假 LLM 与 Embedding 实现，无需访问网络。

**技术栈：** Python 3.10+、Pydantic 2、LangGraph、PyYAML、兼容 OpenAI 接口的 DeepSeek 客户端、SQLite、unittest。

---

## 文件规划

**入口与配置**

- 新建： `task2_main.py` - 轻量可执行入口。
- 新建： `.env.example` - 记录环境变量，不包含密钥。
- 新建： `config/default.yaml` - Agent、RAG、生成、审查和存储默认配置。
- 新建： `src/app/cli.py` - `create`、`resume` 和 `history` 命令。
- 新建： `src/config.py` - YAML 与环境变量配置加载器。
- 修改： `requirements.txt` - 添加 Pydantic、LangGraph、SQLite checkpointer 和 PyYAML。

**LLM 与共享模型**

- 新建： `src/llm/base.py` - 可替换 LLM 客户端协议。
- 新建： `src/llm/deepseek.py` - 对现有 OpenAI 兼容客户端的适配器。
- 修改： `src/llm/__init__.py` - 在保留 `LLMClient` 的同时导出新接口。

**RAG 适配器**

- 新建： `src/rag/schemas.py` - `RetrievedGuideline`.
- 新建： `src/rag/knowledge_base.py` - 任务内 RAG 适配器、稳定 chunk ID 与查询参数校验。
- 新建： `src/rag/query_builder.py` - 策划、写作和审查查询。
- 修改： `src/rag/__init__.py` - 导出适配器类型且不破坏现有导出。

**剧本 Skill**

- 新建： `src/skills/screenplay/schemas.py` - 剧本与生成约束模型。
- 新建： `src/skills/screenplay/repair.py` - JSON 提取与修复异常。
- 新建： `src/skills/screenplay/skill.py` - 结构化剧本生成。
- 新建： `src/skills/screenplay/prompts.py` - 生成与修复 Prompt。
- 新建：在 `src/skills/` 和 `src/skills/screenplay/` 下创建包级 `__init__.py` 文件。

**审查、记忆、工具与工作流**

- 新建： `src/evaluation/schemas.py`, `rubric.py`, `reviewer.py`.
- 新建： `src/memory/user_memory.py`, `project_memory.py`.
- 新建： `src/tools/export.py`, `retrieve.py`, `memory.py`.
- 新建： `src/agent/state.py`, `nodes.py`, `router.py`, `graph.py`.
- 新建：为每个包创建 `__init__.py` 文件。

**知识库与测试**

- 新建： `rag_docs_short_drama/*.md` - 四份聚焦的写作方法文档。
- 新建：已批准设计中列出的 `tests/` 文件。

---

### 任务 1：配置与可替换 LLM 契约

**文件：**
- 新建： `.env.example`
- 新建： `config/default.yaml`
- 新建： `src/config.py`
- 新建： `src/llm/base.py`
- 新建： `src/llm/deepseek.py`
- 修改： `src/llm/__init__.py`
- 修改： `requirements.txt`
- 测试： `tests/test_config_and_llm.py`

- [x] **步骤 1：编写失败的配置与 LLM 契约测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_config_and_llm -v`

预期：失败，因为 `src.config` 和 `src.llm.base` 尚不存在。

- [x] **步骤 3：添加依赖与最小配置文件**

向 `requirements.txt` 添加：

```text
pydantic>=2.7.0
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=2.0.0
PyYAML>=6.0.0
```

创建 `.env.example`，内容包含 `DEEPSEEK_API_KEY=`。

根据已批准设计创建 `config/default.yaml`，其中包含 RAG、生成、审查和 SQLite 路径的默认值。

- [x] **步骤 4：实现最小配置加载器与 LLM 协议**

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

`DeepSeekLLM` 必须将此方法适配到现有的 `LLMClient.chat()` 方法。

- [x] **步骤 5：运行测试与任务内 RAG 回归套件**

运行：`python -m unittest tests.test_config_and_llm tests.test_rag_baseline -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add .env.example config/default.yaml requirements.txt src/config.py src/llm tests/test_config_and_llm.py
git commit -m "Add agent configuration and LLM contract"
```

### 任务 2：基础 RAG 写作知识适配器

**文件：**
- 新建： `src/rag/schemas.py`
- 新建： `src/rag/knowledge_base.py`
- 新建： `src/rag/query_builder.py`
- 修改： `src/rag/__init__.py`
- 测试： `tests/test_existing_rag_compatibility.py`
- 测试： `tests/test_writing_knowledge_base.py`

- [x] **步骤 1：编写失败的适配器测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_writing_knowledge_base -v`

预期：失败，因为适配器模块尚不存在。

- [x] **步骤 3：实现 `RetrievedGuideline` 与适配器**

使用规范化后的 `source`、可选标题层级和 chunk 文本计算 SHA-256。当任务内 RAG 结果没有 ID 时，`WritingKnowledgeBase` 必须生成该 ID，并调用：

```python
self.retriever.query(query, candidate_k=candidate_k, top_k=top_k)
```

- [x] **步骤 4：添加查询构建器**

创建确定性函数：

```python
def build_planning_query(requirement: dict) -> str:
    return f"{requirement['genre']}短剧的开头钩子、人物目标、核心冲突与反转设计"


def build_writing_query(story_plan: dict) -> str:
    return f"{story_plan['genre']}短剧的场景节奏、视觉动作、对白与低成本可拍性"


def build_review_query(screenplay: dict) -> str:
    return f"{screenplay['genre']}短剧的结构、节奏、人物一致性与可拍性评价标准"
```

- [x] **步骤 5：运行适配器与任务内 RAG 测试**

运行：`python -m unittest tests.test_writing_knowledge_base tests.test_existing_rag_compatibility tests.test_rag_baseline -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add src/rag tests/test_existing_rag_compatibility.py tests/test_writing_knowledge_base.py
git commit -m "Add short drama writing knowledge adapter"
```

### 任务 3：剧本 Schema 与确定性校验

**文件：**
- 新建： `src/skills/__init__.py`
- 新建： `src/skills/screenplay/__init__.py`
- 新建： `src/skills/screenplay/schemas.py`
- 测试： `tests/test_screenplay_schema.py`

- [x] **步骤 1：编写失败的 Schema 测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_screenplay_schema -v`

预期：失败，因为剧本 Schema 尚不存在。

- [x] **步骤 3：实现 Pydantic Schema 与校验**

实现 `Character`、`Dialogue`、`Scene`、`Screenplay`、`GenerationConstraints`，以及：

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

校验时长、场景数量、说话角色引用、连续且唯一的 ID、开头钩子和反转。

- [x] **步骤 4：运行 Schema 测试**

运行：`python -m unittest tests.test_screenplay_schema -v`

预期：所有测试均通过。

- [x] **步骤 5：提交**

```powershell
git add src/skills tests/test_screenplay_schema.py
git commit -m "Add structured screenplay schemas"
```

### 任务 4：JSON 修复与剧本 Skill

**文件：**
- 新建： `src/skills/screenplay/prompts.py`
- 新建： `src/skills/screenplay/repair.py`
- 新建： `src/skills/screenplay/skill.py`
- 测试： `tests/test_json_repair.py`
- 测试： `tests/test_screenplay_skill.py`

- [x] **步骤 1：编写失败的 JSON 修复测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_json_repair tests.test_screenplay_skill -v`

预期：失败，因为 Skill 模块尚不存在。

- [x] **步骤 3：实现类型化错误与最小 JSON 提取**

实现：

```python
class ScreenplayGenerationError(RuntimeError):
    pass


class JsonRepairExhaustedError(ScreenplayGenerationError):
    pass

class ScreenplaySkillResult(BaseModel):
    screenplay: Screenplay
    json_repair_attempts: int
```

Skill 只返回成功结果。修复次数耗尽时必须抛出 `JsonRepairExhaustedError`。

- [x] **步骤 4：实现生成与修复 Prompt**

修复 Prompt 必须要求模型保留有效内容，只修复 JSON 解析、类型和缺失的必填字段。

- [x] **步骤 5：运行 Skill 测试**

运行：`python -m unittest tests.test_json_repair tests.test_screenplay_skill tests.test_screenplay_schema -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add src/skills tests/test_json_repair.py tests/test_screenplay_skill.py
git commit -m "Add structured screenplay generation skill"
```

### 任务 5：审查器与最佳版本选择

**文件：**
- 新建： `src/evaluation/__init__.py`
- 新建： `src/evaluation/schemas.py`
- 新建： `src/evaluation/rubric.py`
- 新建： `src/evaluation/reviewer.py`
- 测试： `tests/test_reviewer.py`

- [x] **步骤 1：编写失败的审查器测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_reviewer -v`

预期：失败，因为 evaluation 模块尚不存在。

- [x] **步骤 3：实现 `ReviewReport`、通过判定与最佳版本选择**

确保 `passed` 由代码计算：

```python
passed = total_score >= pass_score and not deterministic_errors
```

- [x] **步骤 4：实现 LLM 审查器适配器**

`ScreenplayReviewer.review()` 必须合并确定性错误和解析后的 LLM 评分输出，并返回 `ReviewReport`。

- [x] **步骤 5：运行审查器测试**

运行：`python -m unittest tests.test_reviewer tests.test_screenplay_schema -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add src/evaluation tests/test_reviewer.py
git commit -m "Add screenplay quality reviewer"
```

### 任务 6：SQLite Memory 仓库与写入时机

**文件：**
- 新建： `src/memory/__init__.py`
- 新建： `src/memory/user_memory.py`
- 新建： `src/memory/project_memory.py`
- 新建： `src/tools/memory.py`
- 测试： `tests/test_memory_repository.py`

- [x] **步骤 1：编写失败的 Memory 写入时机测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_memory_repository -v`

预期：失败，因为 Memory 模块尚不存在。

- [x] **步骤 3：实现最小 SQLite 仓库**

使用标准库中的 `sqlite3`。为项目创建、显式偏好更新、成功导出摘要和历史记录读取提供明确方法。不得从模型输出推断用户偏好。

- [x] **步骤 4：运行 Memory 测试**

运行：`python -m unittest tests.test_memory_repository -v`

预期：所有测试均通过。

- [x] **步骤 5：提交**

```powershell
git add src/memory src/tools/memory.py tests/test_memory_repository.py
git commit -m "Add project and user memory repositories"
```

### 任务 7：导出工具与短剧知识文档

**文件：**
- 新建： `src/tools/__init__.py`
- 新建： `src/tools/export.py`
- 新建： `src/tools/retrieve.py`
- 新建： `rag_docs_short_drama/01_短剧结构与节奏.md`
- 新建： `rag_docs_short_drama/02_人物与冲突设计.md`
- 新建： `rag_docs_short_drama/03_对白与场景写作.md`
- 新建： `rag_docs_short_drama/04_短剧质量审查标准.md`
- 测试： `tests/test_export.py`

- [x] **步骤 1：编写失败的导出测试**

```python
class ExportTests(unittest.TestCase):
    def test_export_writes_best_screenplay_review_and_trace(self):
        paths = export_project(tmp_path, best_screenplay, best_report, retrieval_trace, summary)
        self.assertTrue(paths.screenplay_json.exists())
        self.assertIn(best_screenplay.title, paths.screenplay_markdown.read_text(encoding="utf-8"))
        self.assertTrue(paths.retrieval_trace_json.exists())
```

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_export -v`

预期：失败，因为导出模块尚不存在。

- [x] **步骤 3：实现确定性的 JSON 与 Markdown 导出**

将文件写入 `output/projects/<project_id>/` 并返回类型化路径。导出函数必须接收已选定的最佳版本，不得在内部选择版本。

- [x] **步骤 4：添加聚焦的 RAG 文档**

每份 Markdown 文件必须使用清晰的 `#`、`##` 和 `###` 标题，使现有的层级感知分块器能够保留检索元数据。

- [x] **步骤 5：运行导出与 RAG 测试**

运行：`python -m unittest tests.test_export tests.test_writing_knowledge_base tests.test_rag_baseline -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add src/tools rag_docs_short_drama tests/test_export.py
git commit -m "Add screenplay exports and writing knowledge"
```

### 任务 8：Agent 状态、节点与路由

**文件：**
- 新建： `src/agent/__init__.py`
- 新建： `src/agent/state.py`
- 新建： `src/agent/router.py`
- 新建： `src/agent/nodes.py`
- 测试： `tests/test_graph_routing.py`

- [x] **步骤 1：编写失败的路由与节点测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_graph_routing -v`

预期：失败，因为 Agent 模块尚不存在。

- [x] **步骤 3：实现状态契约与路由器**

在 `AgentState` 中使用可序列化字典。实现 `HumanReviewAction`、`route_after_human_review()` 和 `route_after_review()`。

- [x] **步骤 4：实现依赖注入节点**

节点必须返回部分状态更新。`ScreenplaySkillNode` 捕获类型化 Skill 错误并累加 `json_repair_count`；`ReviewNode` 更新当前版本和最佳版本；`ReviseNode` 只递增 `content_revision_count`。

- [x] **步骤 5：运行路由与组件测试**

运行：`python -m unittest tests.test_graph_routing tests.test_screenplay_skill tests.test_reviewer -v`

预期：所有测试均通过。

- [x] **步骤 6：提交**

```powershell
git add src/agent tests/test_graph_routing.py
git commit -m "Add short drama agent nodes and routing"
```

### 任务 9：LangGraph 中断、恢复与工作流组装

**文件：**
- 新建： `src/agent/graph.py`
- 测试： `tests/test_graph_flow.py`

- [x] **步骤 1：编写失败的图流程测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_graph_flow -v`

预期：失败，因为图组装尚不存在。

- [x] **步骤 3：组装 LangGraph**

添加符合已批准工作流的节点与条件边。使用 `interrupt()` 进行大纲审阅，并接受 `Command(resume={"action": action, "feedback": feedback})`。测试中先使用内存 checkpointer，同时暴露 SQLite checkpointer 构造方式供 CLI 使用。

- [x] **步骤 4：运行图测试**

运行：`python -m unittest tests.test_graph_flow tests.test_graph_routing -v`

预期：所有测试均通过。

- [x] **步骤 5：提交**

```powershell
git add src/agent/graph.py tests/test_graph_flow.py
git commit -m "Assemble interruptible short drama workflow"
```

### 任务 10：CLI 创建、恢复与历史记录

**文件：**
- 新建： `src/app/__init__.py`
- 新建： `src/app/cli.py`
- 新建： `task2_main.py`
- 测试： `tests/test_cli_flow.py`

- [x] **步骤 1：编写失败的 CLI 测试**

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

- [x] **步骤 2：运行测试并确认 RED**

运行：`python -m unittest tests.test_cli_flow -v`

预期：失败，因为 CLI 模块尚不存在。

- [x] **步骤 3：实现 CLI 解析器与轻量入口**

`task2_main.py` 只能调用 `src.app.cli.main()`。CLI 必须加载 `config/default.yaml`、实例化生产依赖，并支持 `create`、`resume` 和 `history`。

- [x] **步骤 4：运行 CLI 测试与帮助信息冒烟测试**

运行：`python -m unittest tests.test_cli_flow -v`

运行：`python task2_main.py --help`

预期：测试通过，帮助信息列出全部三个命令。

- [x] **步骤 5：提交**

```powershell
git add src/app task2_main.py tests/test_cli_flow.py
git commit -m "Add short drama agent CLI"
```

### 任务 11：完整验证、文档与推送

**文件：**
- 新建： `README.md`
- 修改：仅当实现过程中发现必须修正的契约时，修改 `docs/specs/2026-06-10-short-drama-agent-design.md`

- [x] **步骤 1：添加使用文档**

记录安装方式、`.env` 配置、三个 CLI 命令、输出产物、RAG 使用方式、JSON Repair 与 Content Revision 的区别，以及离线测试命令。

- [x] **步骤 2：运行完整离线测试套件**

运行：`python -m unittest discover -s tests -v`

预期：所有测试均通过，且不发生网络调用。

- [x] **步骤 3：运行基础 RAG 回归测试**

运行：`python -m unittest tests/test_rag_baseline.py -v`

预期：3 个测试通过。

- [x] **步骤 4：运行语法与 CLI 冒烟检查**

运行：`python -m compileall src task2_main.py`

运行：`python task2_main.py --help`

预期：两个命令均以退出码 0 结束。

- [x] **步骤 5：检查仓库状态**

运行：`git status --short`

运行：`git diff --check`

预期：仅剩预期的文档变更，且没有空白字符错误。

- [x] **步骤 6：提交最终文档**

```powershell
git add README.md docs
git commit -m "Document short drama agent workflow"
```

- [x] **步骤 7：推送已验证的实现**

运行：`git push origin main`

预期：本地 `main` 与 `origin/main` 指向同一个已验证提交。
