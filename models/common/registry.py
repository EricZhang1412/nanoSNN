from __future__ import annotations

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_model_cls(name: str):
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key]
