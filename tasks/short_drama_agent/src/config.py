import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        config.setdefault("llm", {})["api_key"] = api_key
    return config
