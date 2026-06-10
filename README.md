# 作业任务目录

本仓库包含两个彼此独立的任务。每个任务拥有自己的源码、依赖、配置、测试和文档，
运行时不依赖仓库根目录中的共享 Python 模块。

## 任务一：高速追尾事故 RAG

目录：`tasks/rag_accident/`

```powershell
Set-Location tasks\rag_accident
python -m pip install -r requirements.txt
python -m unittest test_rag_enhancements.py -v
python mission1_main.py
```

## 任务二：短剧生成 Agent

目录：`tasks/short_drama_agent/`

当前目录包含独立的基础 RAG/LLM 源码、回归测试以及 Agent 的设计和实施计划。
Agent 主流程尚未实现。

```powershell
Set-Location tasks\short_drama_agent
python -m pip install -r requirements.txt
python -m unittest tests.test_rag_baseline -v
```

