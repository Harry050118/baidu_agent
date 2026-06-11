import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # LLM配置
    "llm": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
    },

    # Embedding配置
    "embedding": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    },

    # RAG配置
    "rag": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "candidate_k": 5,
        "top_k": 3,
    },

    # 知识库路径
    "knowledge_base": {
        "accident": "rag_docs_accident/",
    },
}
