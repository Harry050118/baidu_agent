import unittest

from src.rag.chunker import Chunk
from src.rag.knowledge_base import WritingKnowledgeBase, guideline_from_result
from src.rag.query_builder import (
    build_planning_query,
    build_review_query,
    build_writing_query,
)
from src.rag.retriever import RetrievalResult


class RecordingRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None

    def query(self, query, candidate_k=5, top_k=3):
        self.last_query = (query, candidate_k, top_k)
        return self.results


def retrieval_result(metadata=None):
    return RetrievalResult(
        chunk=Chunk(text="冲突应逐步升级。", source="./guide.md", metadata=metadata or {}),
        vector_score=0.8,
        title_score=0.4,
        final_score=0.74,
    )


class WritingKnowledgeBaseTests(unittest.TestCase):
    def test_search_forwards_candidate_and_top_k_and_generates_chunk_id(self):
        retriever = RecordingRetriever([retrieval_result()])
        kb = WritingKnowledgeBase(retriever=retriever)

        result = kb.search("冲突设计", "planning", candidate_k=7, top_k=2)

        self.assertEqual(retriever.last_query, ("冲突设计", 7, 2))
        self.assertTrue(result[0].chunk_id)

    def test_missing_titles_become_none(self):
        guideline = guideline_from_result(retrieval_result(), "q", "writing")

        self.assertIsNone(guideline.document_title)
        self.assertIsNone(guideline.parent_header)
        self.assertIsNone(guideline.current_header)

    def test_stable_chunk_id_does_not_depend_on_query(self):
        result = retrieval_result({"document_title": "# 短剧"})

        first = guideline_from_result(result, "first", "planning")
        second = guideline_from_result(result, "second", "review")

        self.assertEqual(first.chunk_id, second.chunk_id)

    def test_top_k_cannot_exceed_candidate_k(self):
        kb = WritingKnowledgeBase(retriever=RecordingRetriever())

        with self.assertRaises(ValueError):
            kb.search("q", "review", candidate_k=2, top_k=3)

    def test_query_builders_include_genre_and_stage_goal(self):
        requirement = {"genre": "悬疑"}
        story_plan = {"genre": "喜剧"}
        screenplay = {"genre": "都市"}

        self.assertIn("开头钩子", build_planning_query(requirement))
        self.assertIn("场景节奏", build_writing_query(story_plan))
        self.assertIn("评价标准", build_review_query(screenplay))

    def test_writing_query_supports_real_story_plan_without_top_level_genre(self):
        story_plan = {
            "title": "消失的课表",
            "conflict": {
                "type": "校园悬疑",
                "description": "学生调查异常课程表。",
            },
        }

        query = build_writing_query(story_plan)

        self.assertIn("校园悬疑", query)
        self.assertIn("场景节奏", query)


if __name__ == "__main__":
    unittest.main()
