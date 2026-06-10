# 短剧生成 Agent 设计规格

日期：2026-06-10

## 1. 目标与范围

本项目实现一个命令行短剧生成 Agent。用户输入创作需求后，系统使用短剧创作方法知识库辅助策划与写作，在故事大纲阶段等待人工确认，随后生成经过 Pydantic 校验的结构化单集短剧，并通过质量审查与有限次数的内容修订输出最佳版本。

第一版面向单集、约 3 至 5 分钟的短剧。时长和场景数量是可覆盖的默认约束，不固化在剧本 Schema 中。

系统需要明确体现以下能力：

- Agent：规划、检索、生成、人工确认、审查、修订和导出的完整工作流。
- RAG：在策划、写作和审查阶段提供可追踪的短剧创作知识。
- Skill：稳定生成符合 Pydantic Schema 的结构化剧本。
- Memory：保存用户创作偏好与项目历史。
- Tool Use：将检索、Memory 访问和导出封装为明确能力。
- Persistence：支持人工中断、失败恢复和按 `thread_id` 继续执行。

第一版不包含 Web UI、多 Agent、多集连续剧、外部向量数据库、自动联网素材搜索，以及视频、音频或分镜图生成。

## 2. 技术方案

采用“确定性 LangGraph 工作流 + 节点内 Agent 能力”的混合架构。

LangGraph 管理共享状态、条件路由、人工中断、检查点和有限修订循环。LLM 不负责自由决定整个流程，只在需求解析、故事策划、结构化生成、内容审查和内容修订等受控节点中工作。

DeepSeek 是第一版默认模型。模型调用通过统一接口封装，后续允许替换为其他兼容模型。

## 3. 总体工作流

```text
CLI
 └─ ShortDramaAgent
     └─ LangGraph Orchestrator
         ├─ ParseRequirementNode
         ├─ LoadUserMemoryNode
         ├─ RetrievePlanningKnowledgeNode
         ├─ StoryPlanningNode
         ├─ HumanReviewNode (interrupt)
         ├─ RetrieveWritingKnowledgeNode
         ├─ ScreenplaySkillNode
         ├─ RetrieveReviewCriteriaNode
         ├─ ReviewNode
         ├─ ReviseNode (有限循环)
         └─ ExportNode
```

`HumanReviewNode` 使用 LangGraph `interrupt()` 暂停工作流并保存 checkpoint。用户可确认大纲、提交修改意见，或退出后使用相同 `thread_id` 恢复。

## 4. 目录规划

```text
src/
├── app/
│   └── cli.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   └── router.py
├── llm/
│   ├── base.py
│   └── deepseek.py
├── rag/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── retriever.py
│   ├── knowledge_base.py
│   ├── query_builder.py
│   ├── schemas.py
│   └── __init__.py
├── memory/
│   ├── project_memory.py
│   └── user_memory.py
├── skills/
│   └── screenplay/
│       ├── skill.py
│       ├── schemas.py
│       ├── prompts.py
│       └── repair.py
├── evaluation/
│   ├── reviewer.py
│   ├── rubric.py
│   └── schemas.py
└── tools/
    ├── retrieve.py
    ├── memory.py
    └── export.py

rag_docs_short_drama/
├── 01_短剧结构与节奏.md
├── 02_人物与冲突设计.md
├── 03_对白与场景写作.md
└── 04_短剧质量审查标准.md

tests/
├── test_existing_rag_compatibility.py
├── test_writing_knowledge_base.py
├── test_screenplay_schema.py
├── test_json_repair.py
├── test_screenplay_skill.py
├── test_reviewer.py
├── test_memory_repository.py
├── test_graph_routing.py
└── test_cli_flow.py

config/
└── default.yaml

.env.example
task2_main.py
```

## 5. 现有 RAG 兼容设计

现有 `src/rag` 是任务一的稳定基础，必须保留其文件、接口和测试：

```python
documents = DocumentLoader.load(path)
chunks = Chunker(...).split(documents)

retriever = Retriever(EmbeddingModel(model_name))
retriever.index(chunks)
results = retriever.query(query, candidate_k=5, top_k=3)
```

第一版不重命名 `embedding.py`，不重写标题层级切分与混合评分逻辑，也不引入独立向量数据库。

新增 `WritingKnowledgeBase` 作为 Agent 适配层。它负责初始化并复用现有 Retriever，避免每个节点重复加载 Embedding 模型和建立索引：

```python
class WritingKnowledgeBase:
    def build(self, document_dir: str) -> None: ...
    def search(
        self,
        query: str,
        purpose: str,
        candidate_k: int = 5,
        top_k: int = 3,
    ) -> list[RetrievedGuideline]: ...
```

`candidate_k` 和 `top_k` 显式透传给现有 `Retriever.query()`。节点可以按阶段覆盖参数，但默认值与当前 RAG 配置保持一致。`top_k` 不得大于 `candidate_k`。

RAG 在三个阶段分别使用：

| 阶段 | 检索目标 |
|---|---|
| 故事策划前 | 开头钩子、核心冲突、人物目标和反转方法 |
| 剧本生成前 | 场景节奏、视觉动作、对白技巧和低成本可拍性 |
| 质量审查前 | 短剧评价标准、结构完整性和节奏标准 |

检索结果保留来源层级及现有 Retriever 的三类分数：

```python
class RetrievedGuideline(BaseModel):
    chunk_id: str
    purpose: str
    query: str
    text: str
    source: str
    document_title: str | None = None
    parent_header: str | None = None
    current_header: str | None = None
    vector_score: float
    title_score: float
    final_score: float
```

标题字段可空，以兼容无 Markdown 标题或缺少层级元数据的文档。`chunk_id` 是检索块的稳定标识，默认由规范化后的 `source`、标题层级和块文本内容计算哈希生成，用于追踪、去重和测试；不得使用查询结果列表位置作为 ID。若现有 RAG 返回结果中没有 `chunk_id`，由 `WritingKnowledgeBase` 负责兜底生成。

## 6. 状态、Checkpoint 与 Memory

三类状态严格分离：

| 类型 | 内容 | 生命周期 | 存储 |
|---|---|---|---|
| AgentState | 当前运行中的创作数据 | 单次工作流 | LangGraph State |
| Checkpoint | 节点级状态快照 | 项目运行期间 | LangGraph SQLite Checkpointer |
| Long-term Memory | 用户偏好与项目历史摘要 | 跨项目 | 独立 SQLite Repository |

核心状态：

```python
class AgentState(TypedDict):
    thread_id: str
    user_id: str
    user_request: str

    requirement: dict | None
    constraints: dict
    user_preferences: dict | None

    planning_guidelines: list[dict]
    story_plan: dict | None
    outline_confirmed: bool
    user_feedback: str | None

    writing_guidelines: list[dict]
    screenplay: dict | None

    review_guidelines: list[dict]
    review_report: dict | None
    best_screenplay: dict | None
    best_review_report: dict | None

    json_repair_count: int
    content_revision_count: int

    final_json_path: str | None
    final_markdown_path: str | None
    errors: list[dict]
```

Pydantic 对象在写入 State 和 checkpoint 前转换为可序列化字典。

长期 Memory 只保存高价值信息：

```python
class UserPreferences(BaseModel):
    preferred_genres: list[str]
    preferred_tones: list[str]
    preferred_endings: list[str]
    dialogue_style: str | None
    production_constraints: list[str]


class ProjectSummary(BaseModel):
    project_id: str
    title: str
    genre: str
    logline: str
    user_feedback: str | None
    final_score: float
    output_paths: list[str]
```

长期 Memory 不承担工作流恢复，Checkpoint 也不自动转化为用户偏好。

### Memory 写入时机

Memory 写入必须发生在明确事件之后，不允许每个节点都隐式修改长期记忆：

| 时机 | 写入内容 | 说明 |
|---|---|---|
| 创建新项目后 | 初始项目记录、`thread_id`、原始需求 | 用于历史列表和运行关联，不写入推断偏好 |
| 用户在大纲确认阶段提交明确反馈后 | 用户明确表达的题材、风格、结局或制作约束偏好 | 仅保存可从用户反馈直接确认的偏好 |
| 最终导出成功后 | `ProjectSummary`、最终分数和输出路径 | 只保存最佳版本摘要，不保存完整 checkpoint |
| 用户显式要求更新偏好时 | 指定偏好字段 | 覆盖或合并时保留更新时间 |

失败运行、自动审查意见和模型自行推断出的偏好不得直接写入长期 Memory。写入失败不应破坏已生成剧本；系统记录可恢复错误，并允许稍后重试。

## 7. 生成约束

默认约束可通过 CLI 或用户需求覆盖：

```python
class GenerationConstraints(BaseModel):
    target_duration_seconds: int = 240
    min_duration_seconds: int = 180
    max_duration_seconds: int = 300

    target_scene_count: int = 4
    min_scene_count: int = 3
    max_scene_count: int = 6

    max_json_repair_attempts: int = 2
    max_content_revisions: int = 2
    pass_score: float = 8.0
```

时长和场景数量不固化在 `Screenplay` Schema 中，由本次运行的约束执行确定性校验。

## 8. Screenplay Skill

`ScreenplaySkill` 是独立于 LangGraph 的可复用、可测试能力：

```python
ScreenplaySkill.generate(
    story_plan,
    requirement,
    constraints,
    writing_guidelines,
    user_preferences,
) -> ScreenplaySkillResult


class ScreenplaySkillResult(BaseModel):
    screenplay: Screenplay
    json_repair_attempts: int
```

核心 Schema：

```python
class Character(BaseModel):
    name: str
    role: str
    personality: str
    goal: str


class Dialogue(BaseModel):
    character: str
    line: str
    emotion: str | None = None


class Scene(BaseModel):
    scene_id: int
    location: str
    time: str
    purpose: str
    conflict: str
    visual_action: str
    dialogues: list[Dialogue]
    estimated_seconds: int
    ending_hook: str | None = None
    shooting_note: str | None = None


class Screenplay(BaseModel):
    title: str
    genre: str
    logline: str
    theme: str
    opening_hook: str
    main_conflict: str
    reversal: str
    emotional_payoff: str
    characters: list[Character]
    scenes: list[Scene]
    ending: str
```

`visual_action` 必须描述可被镜头表现的动作。`shooting_note` 用于可选的运镜、声音、转场或表演建议。

确定性校验包含：

- 总预计时长和场景数量满足本次运行约束。
- 对白角色存在于角色列表。
- `scene_id` 连续且唯一。
- 至少一个场景包含结尾钩子。
- 剧本存在明确反转。

## 9. JSON Repair 与 Content Revision

两类修订严格分离：

| 流程 | 触发条件 | 修改范围 | 是否进入 Reviewer |
|---|---|---|---|
| JSON Repair | JSON 无法解析或不符合 Pydantic Schema | 仅修复格式、类型和缺失字段 | 否 |
| Content Revision | 合法剧本未通过内容审查 | 按审查意见改进剧情内容 | 是 |

执行流程：

```text
LLM 生成原始 JSON
→ JSON 解析与 Pydantic 校验
→ 失败：JSON Repair，最多 max_json_repair_attempts 次
→ 成功：得到合法 Screenplay
→ Reviewer 内容审查
→ 未通过：Content Revision，最多 max_content_revisions 次
→ 修订结果先重新通过 JSON 解析和 Pydantic 校验
→ 再次进入 Reviewer
```

JSON Repair 不消耗内容修订次数，且不得主动改变合法剧情内容。

`ScreenplaySkill` 不直接访问或修改 `AgentState`。`ScreenplaySkillNode` 调用 Skill 后，将结果显式写回状态：

```python
result = screenplay_skill.generate(...)

return {
    "screenplay": result.screenplay.model_dump(),
    "json_repair_count": (
        state["json_repair_count"] + result.json_repair_attempts
    ),
}
```

`json_repair_count` 表示当前工作流累计发生的 JSON Repair 次数，包括初次生成和每次 Content Revision 后的结构修复。达到单次 Skill 调用的修复上限时，Skill 抛出类型化异常，例如 `ScreenplayGenerationError` 或 `JsonRepairExhaustedError`。`ScreenplaySkillNode` 捕获异常，将结构化错误写入 `AgentState.errors`，并依赖 checkpoint 保留当前状态。Skill 不返回 failure result，也不得把未通过 Schema 校验的对象写入 `screenplay`。

## 10. Reviewer 与最佳版本

Reviewer 包含确定性检查和 LLM 内容审查。

确定性检查负责 Schema、约束、角色引用、场景编号和必填内容。LLM 审查负责结构、人物一致性、冲突、对白、节奏、可拍性和 RAG 方法遵循度。

```python
class ReviewReport(BaseModel):
    structure_score: float
    character_consistency_score: float
    conflict_score: float
    dialogue_score: float
    pacing_score: float
    shootability_score: float
    rag_adherence_score: float
    schema_validity_score: float
    total_score: float

    deterministic_errors: list[str]
    issues: list[str]
    revision_instructions: list[str]

    passed: bool
    summary: str
```

`passed` 由程序计算，而不是直接信任 LLM：

```python
passed = (
    total_score >= constraints.pass_score
    and not deterministic_errors
)
```

每次审查后按以下顺序选择最佳版本：

1. 首个拥有有效审查报告的版本成为当前最佳版本。
2. 总分更高的版本替换当前最佳版本。
3. 总分相同时，确定性错误更少的版本替换当前最佳版本。

达到内容修订上限后，系统导出 `best_screenplay` 和 `best_review_report`，而不是盲目导出最后一次版本。

## 11. 人工确认

`HumanReviewNode` 通过 LangGraph `interrupt()` 暂停。恢复输入使用明确的动作枚举：

```python
class HumanReviewAction(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    PAUSE = "pause"
```

| 动作 | 行为 |
|---|---|
| `approve` | 确认当前大纲并继续生成剧本 |
| `revise` | 携带用户修改意见，返回 `StoryPlanningNode` |
| `pause` | 不推进工作流，保留 checkpoint，等待稍后通过相同 `thread_id` 恢复 |

流程如下：

```text
生成 StoryPlan
→ 保存 checkpoint
→ interrupt 返回故事大纲
→ CLI 展示大纲
→ 用户确认、提交修改意见或暂停
→ Command(resume=反馈)
→ 确认后继续，修改后返回 StoryPlanningNode
```

CLI 中可以使用输入交互收集用户选择，但工作流语义必须建立在 interrupt 和 checkpoint 上。

## 12. 错误处理

```python
class AgentError(BaseModel):
    node: str
    error_type: str
    message: str
    recoverable: bool
```

| 错误 | 处理 |
|---|---|
| RAG 知识库为空 | 记录警告，允许无 RAG 降级执行 |
| Embedding 或 LLM 调用失败 | 指数退避重试，仍失败则保存 checkpoint 并退出 |
| JSON Repair 达到上限 | 保存原始响应、验证错误和 checkpoint |
| Content Revision 达到上限 | 导出最佳剧本与最佳审查报告 |
| 人工确认期间退出 | 保留 checkpoint，通过 `thread_id` 恢复 |
| 导出失败 | 保留状态并报告明确路径错误 |

## 13. CLI 与输出

核心命令：

```powershell
python task2_main.py create --request "创作一部校园悬疑短剧"
python task2_main.py resume --thread-id <id>
python task2_main.py history --user-id default
```

可选约束覆盖示例：

```powershell
python task2_main.py create `
  --request "创作一部校园悬疑短剧" `
  --duration 360 `
  --min-scenes 5 `
  --max-scenes 8
```

最终产物：

```text
output/projects/<project_id>/
├── screenplay.json
├── screenplay.md
├── review_report.json
├── retrieval_trace.json
└── run_summary.json
```

`retrieval_trace.json` 保存三个创作阶段使用的 RAG 来源和评分。

## 14. 测试策略

自动化测试默认使用 Fake LLM 和 Fake Embedding，不依赖真实 DeepSeek API。

必须覆盖：

- 现有 `src/rag` 接口与任务一测试不被破坏。
- `WritingKnowledgeBase` 复用索引，并保留检索来源和评分。
- 三阶段检索结果可追踪。
- JSON Repair 只修复结构问题，不执行内容修订。
- Content Revision 不消耗 JSON Repair 次数。
- 时长和场景约束可配置。
- `ReviewReport.passed` 由程序正确计算。
- 修订后分数下降时仍保留最佳版本。
- 达到修订上限后工作流必然结束。
- interrupt 后可通过相同 `thread_id` 恢复。
- CLI 能创建、恢复和查看历史项目。

## 15. 验收标准

第一版完成时，应满足：

1. 用户可通过 CLI 输入单集短剧需求。
2. 系统在策划、写作和审查三个阶段使用现有 RAG 能力。
3. 大纲生成后工作流通过 interrupt 等待人工确认或修改。
4. Screenplay Skill 返回通过 Pydantic 校验的结构化剧本。
5. JSON Repair 与 Content Revision 拥有独立计数与职责。
6. Reviewer 输出结构化报告，并由程序计算 `passed`。
7. 有限内容修订结束后导出最佳版本。
8. JSON、Markdown、审查报告和检索追踪均成功导出。
9. 用户偏好和项目历史可跨运行读取。
10. 现有任务一 RAG 测试和新 Agent 测试全部通过。

## 16. 实现优先级与 MVP 范围

实现按以下顺序推进。每一级必须保持现有 RAG 测试通过，并形成可运行增量。

### P0：可运行 MVP

目标是完整演示“RAG 辅助 + 结构化 Skill + Agent 工作流”的核心闭环：

1. 扩展统一 LLM 接口，支持 DeepSeek 和可测试的 Fake LLM。
2. 实现 `WritingKnowledgeBase`、`RetrievedGuideline`、稳定 `chunk_id` 和三阶段查询参数透传。
3. 建立短剧创作方法知识库。
4. 实现 Screenplay Pydantic Schema、JSON Repair 和 `ScreenplaySkillResult`。
5. 实现基础 AgentState、需求解析、故事策划、剧本生成、审查、有限内容修订和导出节点。
6. 实现 CLI `create`，在大纲阶段完成一次人工确认。
7. 导出最佳剧本 JSON、Markdown、审查报告和检索追踪。
8. 为 RAG 兼容、Schema、JSON Repair、Skill 和核心路由编写测试。

P0 可以使用内存 checkpointer，长期 Memory 可使用最小 SQLite Repository；但接口必须与后续持久化设计一致。

### P1：完整作业版本

目标是补齐完整 Agent 框架能力：

1. 使用 SQLite Checkpointer 支持 interrupt 后按 `thread_id` 恢复。
2. 实现 CLI `resume` 和 `history`。
3. 完成长短期 Memory 分离，以及规定事件上的 Memory 写入。
4. 完善指数退避、结构化错误、失败恢复和降级执行。
5. 补齐 Reviewer 各维度评分、最佳版本选择和修订上限行为。
6. 完成图路由、Memory、interrupt 恢复和 CLI 端到端测试。

### P2：增强项

以下能力仅在 P0 和 P1 稳定后实现：

- 查询改写、去重和更复杂的 RAG 重排。
- 更丰富的项目历史检索和偏好合并策略。
- 运行 tracing、成本统计和质量对比报告。
- Web UI、多 Agent、多集连续剧和多媒体生成。
