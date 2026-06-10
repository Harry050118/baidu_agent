import numpy as np
import re
from dataclasses import dataclass
from typing import List, Tuple
from .embedding import EmbeddingModel
from .chunker import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    vector_score: float
    title_score: float
    final_score: float


class Retriever:
    """向量检索器"""

    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化检索器

        Args:
            embedding_model: Embedding模型实例
        """
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: np.ndarray = None

    def index(self, chunks: List[Chunk]) -> None:
        """
        索引文档块

        Args:
            chunks: Chunk列表
        """
        self.chunks = chunks
        texts = [self._embedding_text(chunk) for chunk in chunks]
        self.chunk_embeddings = self.embedding_model.encode(texts)

    def query(self, question: str, candidate_k: int = 5, top_k: int = 3) -> List[RetrievalResult]:
        """
        检索相关文档块

        Args:
            question: 用户问题
            top_k: 返回的文档块数量

        Returns:
            (Chunk, 相似度分数) 列表
        """
        if not self.chunks:
            return []

        # 将问题向量化
        question_embedding = self.embedding_model.encode_single(question)

        # 计算余弦相似度
        similarities = self._cosine_similarity(question_embedding, self.chunk_embeddings)

        candidate_indices = np.argsort(similarities)[::-1][:candidate_k]
        results = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            vector_score = float(similarities[idx])
            title_score = self._title_bigram_score(question, chunk)
            results.append(RetrievalResult(
                chunk=chunk,
                vector_score=vector_score,
                title_score=title_score,
                final_score=0.85 * vector_score + 0.15 * title_score,
            ))

        return sorted(results, key=lambda result: result.final_score, reverse=True)[:top_k]

    def get_all_similarities(self, question: str) -> np.ndarray:
        """获取问题与所有 chunk 的向量相似度。"""
        question_embedding = self.embedding_model.encode_single(question)
        return self._cosine_similarity(question_embedding, self.chunk_embeddings)

    def _embedding_text(self, chunk: Chunk) -> str:
        metadata = chunk.metadata
        hierarchy = [
            metadata.get("document_title", ""),
            metadata.get("parent_header", ""),
            metadata.get("current_header", metadata.get("header", "")),
        ]
        return "\n".join([text for text in hierarchy if text] + [chunk.text])

    def _title_bigram_score(self, question: str, chunk: Chunk) -> float:
        metadata = chunk.metadata
        title = " ".join([
            metadata.get("document_title", ""),
            metadata.get("parent_header", ""),
            metadata.get("current_header", metadata.get("header", "")),
        ])
        question_bigrams = self._bigrams(question)
        title_bigrams = self._bigrams(title)
        if not question_bigrams or not title_bigrams:
            return 0.0
        return len(question_bigrams & title_bigrams) / len(question_bigrams)

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())
        return {normalized[i:i + 2] for i in range(len(normalized) - 1)}

    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # 归一化
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        docs_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

        # 计算相似度
        similarities = np.dot(docs_norm, query_norm)
        return similarities
