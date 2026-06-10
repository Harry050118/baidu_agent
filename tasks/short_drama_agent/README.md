# 短剧生成 Agent

任务二目录与高速追尾事故 RAG 完全隔离。这里包含一份独立的基础 RAG/LLM 源码，
后续 Agent 功能只在本目录内扩展。

当前状态：

- 已有独立的 `src/rag/` 和 `src/llm/` 基线。
- 已有基础 RAG 回归测试 `tests/test_rag_baseline.py`。
- Agent 设计与实施计划位于 `docs/`。
- Agent CLI、工作流、短剧知识库和持久化功能尚未实现。

## 当前验证

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_rag_baseline -v
```

## 后续实现

实施 Agent 前先阅读：

- `docs/specs/2026-06-10-short-drama-agent-design.md`
- `docs/plans/2026-06-10-short-drama-agent-implementation.md`

