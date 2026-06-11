import unittest

from src.rag import Chunk, Retriever


class ExistingRagCompatibilityTests(unittest.TestCase):
    def test_existing_public_exports_remain_available(self):
        self.assertIsNotNone(Chunk)
        self.assertIsNotNone(Retriever)


if __name__ == "__main__":
    unittest.main()
