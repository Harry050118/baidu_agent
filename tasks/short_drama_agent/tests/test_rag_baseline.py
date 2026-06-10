import unittest

import numpy as np

from src.rag.chunker import Chunk, Chunker
from src.rag.document_loader import Document
from src.rag.retriever import Retriever


class RecordingEmbeddingModel:
    def __init__(self):
        self.indexed_texts = []

    def encode(self, texts):
        self.indexed_texts = list(texts)
        return np.array([
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.10, 0.90],
        ])

    def encode_single(self, text):
        return np.array([1.0, 0.0])


class RagEnhancementTests(unittest.TestCase):
    def test_chunker_preserves_document_parent_and_current_headers(self):
        document = Document(
            content=(
                "# 高速公路事故\n\n"
                "## 三、赔偿项目与标准\n\n"
                "### 医疗费\n\n医疗费用说明。" + "详细内容。" * 30
            ),
            metadata={"source": "law.md", "filename": "law.md"},
        )

        chunks = Chunker(chunk_size=120, overlap=20, min_chunk_size=0).split([document])

        medical_chunk = next(chunk for chunk in chunks if "医疗费" in chunk.text)
        self.assertEqual(medical_chunk.metadata["document_title"], "# 高速公路事故")
        self.assertEqual(medical_chunk.metadata["parent_header"], "## 三、赔偿项目与标准")
        self.assertEqual(medical_chunk.metadata["current_header"], "### 医疗费")

    def test_small_section_does_not_merge_into_a_different_parent_section(self):
        document = Document(
            content=(
                "# 事故知识\n\n"
                "## 二、责任划分说明\n\n" + "责任说明。" * 30 + "\n\n"
                "## 三、赔偿项目与标准\n\n"
                "### 1. 财产损失赔偿\n\n- 车辆维修费用\n- 车载物品损失\n\n"
                "### 2. 人身损害赔偿项目\n\n" + "医疗费用。" * 80
            ),
            metadata={"source": "law.md", "filename": "law.md"},
        )

        chunks = Chunker(chunk_size=300, overlap=30, min_chunk_size=120).split([document])

        property_chunk = next(chunk for chunk in chunks if "车辆维修费用" in chunk.text)
        self.assertEqual(property_chunk.metadata["parent_header"], "## 三、赔偿项目与标准")
        self.assertEqual(property_chunk.metadata["current_header"], "### 1. 财产损失赔偿")
        human_chunk = next(chunk for chunk in chunks if "医疗费用" in chunk.text)
        self.assertEqual(human_chunk.metadata["current_header"], "### 2. 人身损害赔偿项目")

    def test_retriever_indexes_hierarchy_and_returns_three_scores(self):
        chunks = [
            Chunk(
                text=f"正文 {i}",
                source=f"{i}.md",
                metadata={
                    "document_title": "# 事故知识",
                    "parent_header": "## 其他主题",
                    "current_header": f"### 条目{i}",
                },
            )
            for i in range(6)
        ]
        chunks[4].metadata["parent_header"] = "## 赔偿项目与标准"
        model = RecordingEmbeddingModel()
        retriever = Retriever(model)

        retriever.index(chunks)
        results = retriever.query("赔偿项目有哪些", candidate_k=5, top_k=3)

        self.assertIn("赔偿项目与标准", model.indexed_texts[4])
        self.assertEqual(len(results), 3)
        self.assertIs(results[0].chunk, chunks[4])
        self.assertGreater(results[0].title_score, 0)
        self.assertAlmostEqual(
            results[0].final_score,
            0.85 * results[0].vector_score + 0.15 * results[0].title_score,
        )


if __name__ == "__main__":
    unittest.main()
