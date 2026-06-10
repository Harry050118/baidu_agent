"""
任务1：最简单的RAG
回答"什么是高速追尾事故？"
"""

from src.rag import DocumentLoader, Chunker, EmbeddingModel, Retriever
from src.llm import LLMClient
from config import CONFIG


def build_rag_prompt(question: str, chunks: list) -> str:
    """构建RAG prompt"""
    context = "\n\n".join([
        f"【文档片段{i+1}】\n{result.chunk.text}"
        for i, result in enumerate(chunks)
    ])

    prompt = f"""请根据以下文档内容回答问题。

    文档内容：
    {context}

    问题：{question}

    请仅根据文档内容回答，如果文档中没有相关信息，请说明。回答要准确、完整。"""

    return prompt


def main():
    print("=" * 50)
    print("任务1：最简单的RAG系统")
    print("=" * 50)

    # 1. 加载文档
    print("\n[1] 加载文档...")
    docs = DocumentLoader.load(CONFIG["knowledge_base"]["accident"])
    print(f"    加载了 {len(docs)} 篇文档")

    # 2. 切分文档
    print("\n[2] 切分文档...")
    chunker = Chunker(
        chunk_size=CONFIG["rag"]["chunk_size"],
        overlap=CONFIG["rag"]["chunk_overlap"],
    )
    chunks = chunker.split(docs)
    print(f"    切分为 {len(chunks)} 个文档块")

    # 3. 初始化Embedding模型
    print("\n[3] 初始化Embedding模型...")
    embedding_model = EmbeddingModel(CONFIG["embedding"]["model_name"])
    print("    模型加载完成")

    # 4. 构建检索器并索引
    print("\n[4] 构建检索索引...")
    retriever = Retriever(embedding_model)
    retriever.index(chunks)
    print("    索引构建完成")

    # 5. 初始化LLM客户端
    print("\n[5] 初始化LLM客户端...")
    llm_client = LLMClient(
        api_key=CONFIG["llm"]["api_key"],
        base_url=CONFIG["llm"]["base_url"],
        model=CONFIG["llm"]["model"],
    )
    print("    LLM客户端就绪")

    # 6. 用户提问
    question = "什么是高速追尾事故？"
    print(f"\n[6] 用户问题：{question}")

    # 7. 检索相关文档
    print("\n[7] 检索相关文档...")
    top_k = CONFIG["rag"]["top_k"]
    results = retriever.query(
        question,
        candidate_k=CONFIG["rag"]["candidate_k"],
        top_k=top_k,
    )
    print(f"    检索到 {len(results)} 个相关文档块")
    for i, result in enumerate(results):
        print(
            f"    - [{i+1}] 综合: {result.final_score:.4f}, "
            f"向量: {result.vector_score:.4f}, 标题: {result.title_score:.4f}, "
            f"来源: {result.chunk.metadata.get('filename', 'unknown')}"
        )

    # 8. 构建prompt并调用LLM
    print("\n[8] 调用LLM生成回答...")
    prompt = build_rag_prompt(question, results)
    messages = [{"role": "user", "content": prompt}]
    answer = llm_client.chat(messages)

    # 9. 输出结果
    print("\n" + "=" * 50)
    print("回答：")
    print("=" * 50)
    print(answer)
    print("=" * 50)


if __name__ == "__main__":
    main()
