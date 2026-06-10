import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """文本向量化模型"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化Embedding模型

        Args:
            model_name: sentence-transformers模型名称
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组，shape=(len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        将单个文本转换为向量

        Args:
            text: 文本字符串

        Returns:
            向量数组，shape=(embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
