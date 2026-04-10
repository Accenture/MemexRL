"""Engine module for Memex.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

__all__ = [
    "TokenStatsManager",
]


def __getattr__(name):
    if name == "AgentExecutionEngine" or name == "AsyncAgentExecutionEngine":
        from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine
        if name == "AgentExecutionEngine":
            return AgentExecutionEngine
        return AsyncAgentExecutionEngine
    if name in ("ModelOutput", "RolloutEngine"):
        from .rollout.rollout_engine import ModelOutput, RolloutEngine
        return ModelOutput if name == "ModelOutput" else RolloutEngine
    raise AttributeError(name)
