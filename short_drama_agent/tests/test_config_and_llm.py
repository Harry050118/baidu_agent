import unittest
from unittest.mock import patch

from src.config import load_config
from src.llm.base import LLM
from src.llm.deepseek import DeepSeekLLM


class EchoLLM:
    def generate(self, messages, *, temperature=0.7):
        return messages[-1]["content"]


class RecordingClient:
    def __init__(self):
        self.call = None

    def chat(self, messages, temperature=0.7):
        self.call = (messages, temperature)
        return "generated"


class ConfigAndLLMTests(unittest.TestCase):
    def test_default_config_contains_generation_and_storage_defaults(self):
        config = load_config("config/default.yaml")
        self.assertEqual(config["generation"]["target_duration_seconds"], 240)
        self.assertEqual(config["review"]["max_content_revisions"], 2)
        self.assertIn("checkpoint_db", config["storage"])

    def test_fake_implementation_satisfies_llm_protocol(self):
        llm: LLM = EchoLLM()
        self.assertEqual(llm.generate([{"role": "user", "content": "hello"}]), "hello")

    def test_environment_api_key_overrides_llm_config(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            config = load_config("config/default.yaml")

        self.assertEqual(config["llm"]["api_key"], "test-key")

    def test_deepseek_adapter_forwards_to_existing_client(self):
        client = RecordingClient()
        llm = DeepSeekLLM(client)
        messages = [{"role": "user", "content": "hello"}]

        result = llm.generate(messages, temperature=0.2)

        self.assertEqual(result, "generated")
        self.assertEqual(client.call, (messages, 0.2))


if __name__ == "__main__":
    unittest.main()
