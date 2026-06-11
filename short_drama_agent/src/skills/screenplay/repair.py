import json
import re
from typing import Any

from pydantic import ValidationError

from .schemas import Screenplay


class ScreenplayGenerationError(RuntimeError):
    pass


class JsonRepairExhaustedError(ScreenplayGenerationError):
    pass


def extract_json_text(response: str) -> str:
    stripped = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def parse_screenplay(response: str) -> Screenplay:
    try:
        payload: Any = json.loads(extract_json_text(response))
        if isinstance(payload, dict) and isinstance(payload.get("screenplay"), dict):
            payload = payload["screenplay"]
        return Screenplay.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ScreenplayGenerationError(str(exc)) from exc
