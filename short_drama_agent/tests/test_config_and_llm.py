import unittest
from unittest.mock import patch

from src.config import load_config
from src.llm.base import LLM
from src.llm.deepseek import DeepSeekLLM
from src.llm.retrying import LLMRetryExhaustedError, RetryingLLM


class EchoLLM:
    def generate(self, messages, *, temperature=0.7):
        return messages[-1]["content"]


class RecordingClient:
    def __init__(self):
        self.call = None

    def chat(self, messages, temperature=0.7):
        self.call = (messages, temperature)
        return "generated"


class FlakyLLM:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def generate(self, messages, *, temperature=0.7):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary")
        return "generated"


class ConfigAndLLMTests(unittest.TestCase):
    def test_default_config_contains_generation_and_storage_defaults(self):
        config = load_config("config/default.yaml")
        self.assertEqual(config["generation"]["target_duration_seconds"], 240)
        self.assertEqual(config["review"]["max_content_revisions"], 2)
        self.assertEqual(config["retry"]["max_attempts"], 3)
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

    def test_retrying_llm_retries_with_exponential_delays(self):
        llm = FlakyLLM(failures=2)
        delays = []
        retrying = RetryingLLM(
            llm,
            max_attempts=3,
            initial_delay_seconds=0.5,
            backoff_multiplier=2.0,
            sleep=delays.append,
        )

        result = retrying.generate([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "generated")
        self.assertEqual(delays, [0.5, 1.0])

    def test_retrying_llm_raises_typed_error_after_attempt_limit(self):
        retrying = RetryingLLM(
            FlakyLLM(failures=3),
            max_attempts=2,
            initial_delay_seconds=0,
            backoff_multiplier=2,
            sleep=lambda _: None,
        )

        with self.assertRaises(LLMRetryExhaustedError):
            retrying.generate([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
