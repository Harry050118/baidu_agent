# 独立任务目录拆分设计

日期：2026-06-10

## 1. 目标

将当前仓库拆分为两个可以独立理解、安装、运行和测试的任务：

- `tasks/rag_accident/`：任务一，高速追尾事故 RAG。
- `tasks/short_drama_agent/`：任务二，短剧生成 Agent。

两个任务不得依赖根目录下的共享 `src/`、`config.py` 或
`requirements.txt`。每个任务拥有自己的源码、配置、依赖、知识库、测试、
文档和输出目录。

## 2. 目标目录结构

```text
.
├── .gitignore
├── README.md
├── homework_2.docx
└── tasks/
    ├── rag_accident/
    │   ├── README.md
    │   ├── config.py
    │   ├── mission1_main.py
    │   ├── rag_pipeline_demo.ipynb
    │   ├── requirements.txt
    │   ├── test_rag_enhancements.py
    │   ├── 任务一_RAG学习与完成指南.md
    │   ├── rag_docs_accident/
    │   ├── output/
    │   └── src/
    │       ├── llm/
    │       └── rag/
    └── short_drama_agent/
        ├── README.md
        ├── requirements.txt
        ├── docs/
        │   ├── plans/
        │   └── specs/
        └── src/
            ├── llm/
            └── rag/
```

任务二后续实现时，其 `src/`、配置、测试、知识库和输出均继续放在
`tasks/short_drama_agent/` 内，不得重新引用任务一目录。

## 3. 文件归属

### 3.1 高速追尾事故 RAG

以下现有内容归入 `tasks/rag_accident/`：

- `mission1_main.py`
- `rag_pipeline_demo.ipynb`
- `config.py`
- `requirements.txt`
- `test_rag_enhancements.py`
- `任务一_RAG学习与完成指南.md`
- `rag_docs_accident/`
- `mission1_output/`，迁移后统一命名为 `output/`
- 当前 `src/rag/` 和 `src/llm/`

迁移后修正入口、Notebook、配置和文档中的路径，使其从
`tasks/rag_accident/` 目录执行时可用。

当前工作区中旧的 `task1_main.py` 和旧的 `output/` 已处于删除状态。拆分过程
不恢复这些旧文件，使用现有 `mission1_main.py` 和 `mission1_output/` 作为任务一
的当前版本。

### 3.2 短剧生成 Agent

以下现有内容归入 `tasks/short_drama_agent/`：

- `docs/superpowers/specs/2026-06-10-short-drama-agent-design.md`
- `docs/superpowers/plans/2026-06-10-short-drama-agent-implementation.md`

任务二当前尚未实现。为保证目录边界从一开始就是独立的，任务二获得一份当前
基础 `src/rag/` 和 `src/llm/` 的复制，后续 Agent 可在自身目录中扩展或修改，
不会影响任务一。

短剧 Agent 的设计和实施计划必须更新所有目标路径，使计划创建的文件均位于
`tasks/short_drama_agent/` 内。文档中不再将根目录 `src/`、`tests/`、
`rag_docs_short_drama/` 或 `task2_main.py` 作为 Agent 文件位置。

### 3.3 仓库根目录

根目录只保留：

- `.gitignore`
- `README.md`：说明两个任务的用途、入口和独立运行命令。
- `homework_2.docx`：原始作业说明。
- `tasks/`
- Git 和工具自身所需的隐藏目录。

根目录不保留共享 Python 源码、共享依赖文件、任务入口、任务输出或任务专属文档。

## 4. 独立运行约束

任务一的运行与测试命令以其目录为工作目录：

```powershell
Set-Location tasks\rag_accident
python mission1_main.py
python -m unittest test_rag_enhancements.py -v
```

任务二后续的运行与测试命令同样以其目录为工作目录：

```powershell
Set-Location tasks\short_drama_agent
python task2_main.py --help
python -m unittest discover -s tests -v
```

两个任务可以使用相同的 Python 包名，例如各自拥有 `src.rag`，因为运行时工作
目录和源码树互相隔离。任何测试不得通过修改 `sys.path` 引用另一个任务或仓库
根目录的源码。

## 5. 配置与输出

- 任务一的知识库路径和输出路径相对于 `tasks/rag_accident/`。
- 任务二的配置、知识库、SQLite、检查点和生成结果相对于
  `tasks/short_drama_agent/`。
- 根目录 `.env` 不纳入版本控制。每个任务需要的环境变量应由各自 README 和
  `.env.example` 说明；本次拆分不复制真实密钥文件。
- 根目录 `.gitignore` 继续提供仓库级通用忽略规则，并增加两个任务的运行时输出
  规则时不得忽略需要保留的演示产物。

## 6. 迁移安全规则

- 禁止批量删除文件或目录。
- 不使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse` 或 `rm -rf`。
- 文件迁移只使用明确源路径和明确目标路径。
- 目录通过创建目标目录、逐个移动文件完成；不通过先删除旧目录完成迁移。
- 不恢复或覆盖用户当前工作区中的删除和未跟踪状态。
- 如果迁移后留下空目录，不为清理空目录而执行批量删除。

## 7. 验证

拆分完成后必须验证：

1. 根目录不存在任务共享的 `src/`、`config.py` 或 `requirements.txt`。
2. `tasks/rag_accident/` 包含完整任务一源码、知识库、文档和当前输出。
3. `tasks/short_drama_agent/` 包含独立基础源码和更新后的 Agent 设计、实施计划。
4. 在 `tasks/rag_accident/` 内运行现有 RAG 单元测试通过。
5. 从仓库根目录检索 Agent 文档，不存在要求在根目录创建 Agent 专属源码或数据的
   有效路径说明。
6. `git status` 中的变更与本设计一致，并保留拆分前已有的用户工作区状态。

任务一主程序需要外部模型和 API 密钥，因此本次拆分以导入检查、单元测试和路径
检查作为离线验收；不要求实际调用远程 LLM。

## 8. 非目标

- 本次不实现短剧 Agent。
- 本次不重构 RAG 算法、LLM 客户端或测试逻辑。
- 本次不合并两个任务的依赖或配置。
- 本次不清理缓存目录或其他生成文件。
