import os
import yaml
from types import SimpleNamespace


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            return _to_namespace(config)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {config_path}: {exc}") from exc
