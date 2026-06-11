# 短剧生成 Agent

任务二是一个独立的命令行短剧 Agent。它复用本目录的层级感知 RAG 基线，使用
LangGraph 管理大纲确认、中断恢复、剧本生成、质量审查、内容修订和最终导出。

## 安装与配置

在本目录中执行：

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=your-api-key
```

默认模型、RAG 参数、生成约束、审查阈值和 SQLite 路径位于
`config/default.yaml`。

## CLI

创建项目并在故事大纲确认阶段暂停：

```powershell
python task2_main.py create --request "创作一部校园悬疑短剧"
```

可覆盖时长和场景数量约束：

```powershell
python task2_main.py create `
  --request "创作一部校园悬疑短剧" `
  --duration 360 `
  --min-scenes 5 `
  --max-scenes 8
```

恢复已有线程：

```powershell
python task2_main.py resume --thread-id <thread-id> --action approve
python task2_main.py resume --thread-id <thread-id> --action revise --feedback "加强结尾反转"
python task2_main.py resume --thread-id <thread-id> --action pause
```

修订大纲时可以显式记录长期偏好；普通 `--feedback` 不会自动推断或写入偏好：

```powershell
python task2_main.py resume --thread-id <thread-id> --action revise `
  --feedback "加强结尾反转" `
  --preferred-genre "悬疑" `
  --preferred-tone "紧张" `
  --preferred-ending "反转" `
  --dialogue-style "短句" `
  --production-constraint "低成本"
```

查看用户项目历史：

```powershell
python task2_main.py history --user-id default
```

## 工作流与持久化

- `data/checkpoints.sqlite` 保存 LangGraph 节点级 checkpoint，用于中断恢复。
- `data/memory.sqlite` 保存显式用户偏好和成功导出的项目摘要。
- 创建项目不会推断用户偏好；只有明确的大纲反馈才会更新偏好。
- 失败运行和模型自行推断的信息不会自动写入长期 Memory。
- LLM 调用使用配置化指数退避；尝试次数耗尽后记录结构化错误并保留 checkpoint。

## RAG

`src/rag/` 保留独立 RAG 基线的公共接口。`WritingKnowledgeBase` 在其上提供 Agent
适配层，显式透传 `candidate_k` 和 `top_k`，并为检索块生成稳定 ID。

LangGraph 在故事策划、剧本写作和质量审查前分别执行检索。检索失败或知识库为空时，
系统记录可恢复错误并使用空指南降级继续；非 RAG 节点失败时记录不可恢复错误并停止。

短剧写作知识位于 `rag_docs_short_drama/`，分别覆盖结构与节奏、人物与冲突、
对白与场景写作、质量审查标准。

## JSON Repair 与 Content Revision

- JSON Repair 只修复 JSON 解析、字段类型和缺失字段，不主动改写合法剧情内容。
- Content Revision 根据 Reviewer 意见修改剧本内容，并单独计算修订次数。
- 达到修订上限时导出评分最高的最佳版本，而不是直接导出最后一次版本。

## 输出

成功导出后，项目产物位于 `output/projects/<project_id>/`：

```text
screenplay.json
screenplay.md
review.json
retrieval_trace.json
summary.json
```

## 离线验证

测试使用假 LLM 和假 Embedding，不访问网络：

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_rag_baseline -v
python -m compileall src task2_main.py
python task2_main.py --help
```

设计和实施计划位于：

- `docs/specs/2026-06-10-short-drama-agent-design.md`
- `docs/plans/2026-06-10-short-drama-agent-implementation.md`
