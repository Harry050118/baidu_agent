import time
from collections.abc import Callable

from .base import LLM


class LLMRetryExhaustedError(RuntimeError):
    pass


class RetryingLLM:
    def __init__(
        self,
        llm: LLM,
        *,
        max_attempts: int = 3,
        initial_delay_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        self.llm = llm
        self.max_attempts = max_attempts
        self.initial_delay_seconds = initial_delay_seconds
        self.backoff_multiplier = backoff_multiplier
        self.sleep = sleep

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
    ) -> str:
        delay = self.initial_delay_seconds
        for attempt in range(self.max_attempts):
            try:
                return self.llm.generate(messages, temperature=temperature)
            except Exception as exc:
                if attempt == self.max_attempts - 1:
                    raise LLMRetryExhaustedError(str(exc)) from exc
                self.sleep(delay)
                delay *= self.backoff_multiplier
        raise AssertionError("unreachable")
