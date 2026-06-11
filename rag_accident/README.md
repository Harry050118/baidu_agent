# 高速追尾事故 RAG

任务一实现最小 RAG 流程：加载高速追尾事故知识文档、分块、生成向量、检索相关片段，
并调用 LLM 回答“什么是高速追尾事故？”。

## 运行

在本目录中执行：

```powershell
python -m pip install -r requirements.txt
python mission1_main.py
```

远程回答生成需要在环境变量 `DEEPSEEK_API_KEY` 中提供 API 密钥。

## 测试

```powershell
python -m unittest test_rag_enhancements.py -v
```

Notebook `rag_pipeline_demo.ipynb` 与 `mission1_main.py` 使用相同的任务内源码。
知识库位于 `rag_docs_accident/`，演示输出位于 `output/`。

