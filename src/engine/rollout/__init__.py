# Lazy imports only — avoid loading unused engines at module import time

__all__ = [
    "ModelOutput",
    "RolloutEngine",
]


def __getattr__(name):
    if name in ("ModelOutput", "RolloutEngine"):
        from .rollout_engine import ModelOutput, RolloutEngine
        return ModelOutput if name == "ModelOutput" else RolloutEngine
    raise AttributeError(name)
