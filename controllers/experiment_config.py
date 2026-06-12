"""Load experiment settings for training and inference."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "default_experiment.json")


def load_experiment_config(path: str | None = None) -> dict[str, Any]:
    """Load the default experiment config and overlay an optional JSON file."""
    config = _read_json(DEFAULT_CONFIG_PATH)
    if path is not None:
        config = _deep_update(config, _read_json(path))
    return config


def env_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = deepcopy(config.get("env", {}))
    reward_config = deepcopy(config.get("reward", {}))
    if reward_config:
        kwargs["reward_weights"] = reward_config
    return kwargs


def model_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = deepcopy(config.get("model", {}))
    kwargs.pop("algorithm", None)
    kwargs.pop("policy", None)
    if "policy_kwargs" in kwargs:
        kwargs["policy_kwargs"] = deepcopy(kwargs["policy_kwargs"])
    return kwargs


def model_algorithm_from_config(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("algorithm", "ppo"))


def model_policy_from_config(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("policy", "MultiInputPolicy"))


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
