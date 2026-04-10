"""
Async-compatible parallel ALFWorld environment.

This provides a drop-in replacement for ALFWorldEnv that runs each environment
instance in a separate process, avoiding the TextWorld global lock bottleneck.

Usage in train_alfworld_rl.py:
    from src.environments.alfworld.async_parallel_env import AsyncParallelALFWorldEnv

    # Replace ALFWorldEnv with AsyncParallelALFWorldEnv
    trainer = AgentTrainer(
        agent_class=ALFWorldAgentWithMemory,
        env_class=AsyncParallelALFWorldEnv,  # Changed!
        ...
    )
"""

import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue

from src.environments.base.base_env import BaseEnv
from typing import Any, Optional, Callable
import traceback
import os
import threading


def _env_worker_main(
    worker_id: int,
    cmd_queue: Queue,
    result_queue: Queue,
    max_steps: int,
    ready_event: mp.Event
):
    """Worker process main function."""
    import random
    import numpy as np

    # Set unique random seed
    seed = os.getpid() + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))

    env = None
    ready_event.set()  # Signal that we're ready

    while True:
        try:
            msg = cmd_queue.get()
            if msg is None:  # Shutdown
                break

            cmd, args = msg

            if cmd == "init":
                from src.environments.alfworld.env import ALFWorldEnv
                env = ALFWorldEnv(
                    task=args["task"],
                    max_steps=max_steps,
                    use_process_lock=False  # No lock - isolated process
                )
                result_queue.put(("ok", True))

            elif cmd == "reset":
                if env is None:
                    result_queue.put(("error", "Not initialized"))
                else:
                    result = env.reset()
                    result_queue.put(("ok", result))

            elif cmd == "step":
                if env is None:
                    result_queue.put(("error", "Not initialized"))
                else:
                    result = env.step(args["action"])
                    result_queue.put(("ok", result))

            elif cmd == "reward":
                if env is None:
                    result_queue.put(("ok", 0.0))
                else:
                    result = env.compute_final_reward()
                    result_queue.put(("ok", result))

            elif cmd == "close":
                if env is not None:
                    try:
                        env.close()
                    except:
                        pass
                    env = None
                result_queue.put(("ok", True))

        except Exception as e:
            traceback.print_exc()
            result_queue.put(("error", str(e)))


class AsyncParallelALFWorldEnv(BaseEnv):
    """Parallel ALFWorld environment with async-compatible interface.

    Each instance runs in its own subprocess, enabling true parallelism
    without the TextWorld lock bottleneck.

    This class is a drop-in replacement for ALFWorldEnv in the Memex framework.
    """

    # Class-level process management
    _ctx = None
    _initialized = False

    @classmethod
    def _get_mp_context(cls):
        if cls._ctx is None:
            cls._ctx = mp.get_context("spawn")
        return cls._ctx

    def __init__(
        self,
        task: Optional[dict] = None,
        reward_fn: Optional[Callable] = None,
        max_steps: int = 50,
        **kwargs
    ):
        """Initialize parallel environment.

        Args:
            task: Task configuration dict
            reward_fn: Optional reward function (ignored, uses env's internal)
            max_steps: Maximum steps per episode
        """
        self.task = task or {}
        self.reward_fn = reward_fn
        self.max_steps = max_steps

        # Create dedicated process for this env instance
        ctx = self._get_mp_context()
        self._cmd_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._ready_event = ctx.Event()

        self._process = ctx.Process(
            target=_env_worker_main,
            args=(
                id(self) % 10000,
                self._cmd_queue,
                self._result_queue,
                max_steps,
                self._ready_event
            ),
            daemon=True
        )
        self._process.start()

        # Wait for worker to be ready
        self._ready_event.wait(timeout=30)

        # State tracking (for compatibility)
        self.current_step = 0
        self.done = False
        self.won = False
        self._closed = False  # Must be set before _send_cmd

        # Initialize env with task
        if task:
            self._send_cmd("init", {"task": task})

    def _send_cmd(self, cmd: str, args: dict = None) -> Any:
        """Send command to worker and get result."""
        if self._closed:
            raise RuntimeError("Environment is closed")

        # Check if worker process is still alive
        if not self._process.is_alive():
            raise RuntimeError(f"Env worker process died unexpectedly (cmd={cmd})")

        self._cmd_queue.put((cmd, args or {}))

        # Wait with periodic alive checks (10 second intervals, max 60 seconds total)
        import queue
        max_wait = 60  # seconds
        check_interval = 10  # seconds
        waited = 0

        while waited < max_wait:
            try:
                status, result = self._result_queue.get(timeout=check_interval)
                if status == "error":
                    raise RuntimeError(f"Env worker error: {result}")
                return result
            except queue.Empty:
                waited += check_interval
                # Check if process died while waiting
                if not self._process.is_alive():
                    raise RuntimeError(f"Env worker process died while executing {cmd}")
                # Still waiting, continue

        # Timeout reached
        raise RuntimeError(f"Env worker timeout after {max_wait}s waiting for {cmd}")

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment."""
        self.current_step = 0
        self.done = False
        self.won = False
        return self._send_cmd("reset")

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute an action."""
        self.current_step += 1
        obs, reward, done, info = self._send_cmd("step", {"action": action})
        self.done = done
        if info.get("won"):
            self.won = True
        return obs, reward, done, info

    def compute_final_reward(self) -> float:
        """Compute final reward."""
        return self._send_cmd("reward")

    def get_tools(self) -> list:
        """Get environment tools."""
        from src.environments.alfworld.env import get_alfworld_tools
        return get_alfworld_tools()

    @staticmethod
    def is_multithread_safe() -> bool:
        """This env IS safe - each instance has its own process."""
        return True

    @staticmethod
    def from_dict(info: dict) -> "AsyncParallelALFWorldEnv":
        """Create environment from dictionary.

        Args:
            info: Dictionary containing task configuration.

        Returns:
            AsyncParallelALFWorldEnv instance.
        """
        task = {
            "task_id": info.get("task_id", ""),
            "task_type": info.get("task_type", ""),
            "game_file": info.get("game_file", ""),
            "task_description": info.get("task_description", ""),
            "data_source": info.get("data_source", "alfworld"),
        }

        return AsyncParallelALFWorldEnv(
            task=task,
            reward_fn=info.get("reward_fn"),
            max_steps=info.get("max_steps", 50),
        )

    def close(self):
        """Close the environment and worker process."""
        if self._closed:
            return

        self._closed = True
        try:
            self._cmd_queue.put(None)  # Shutdown signal
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
        except:
            pass

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


# For compatibility - export same interface as original
def get_alfworld_tools():
    from src.environments.alfworld.env import get_alfworld_tools as _get_tools
    return _get_tools()
