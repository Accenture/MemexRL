"""
Multiprocessing-based parallel environment worker for ALFWorld.

This module provides a simple way to run ALFWorld environments in parallel
using Python's multiprocessing module. Each worker process has its own
TextWorld instance, avoiding the global lock bottleneck.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from typing import Any, Optional
import traceback
import os


class EnvWorkerProcess:
    """A worker process that owns and operates an ALFWorld environment.

    Communication happens via multiprocessing queues.
    """

    def __init__(self, worker_id: int, task_queue: Queue, result_queue: Queue, max_steps: int = 50):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.max_steps = max_steps
        self.env = None

    def run(self):
        """Main worker loop."""
        # Set unique random seed per worker
        import random
        import numpy as np
        random.seed(os.getpid())
        np.random.seed(os.getpid() % (2**32))

        while True:
            try:
                task = self.task_queue.get()

                if task is None:  # Shutdown signal
                    break

                cmd, args = task
                result = self._handle_command(cmd, args)
                self.result_queue.put((self.worker_id, result))

            except Exception as e:
                traceback.print_exc()
                self.result_queue.put((self.worker_id, ("error", str(e))))

    def _handle_command(self, cmd: str, args: dict) -> tuple:
        """Handle a command from the main process."""
        if cmd == "set_task":
            return self._set_task(args["task"])
        elif cmd == "reset":
            return self._reset()
        elif cmd == "step":
            return self._step(args["action"])
        elif cmd == "compute_final_reward":
            return self._compute_final_reward()
        elif cmd == "close":
            return self._close()
        else:
            return ("error", f"Unknown command: {cmd}")

    def _set_task(self, task: dict) -> tuple:
        """Create environment with task."""
        try:
            from src.environments.alfworld.env import ALFWorldEnv
            self.env = ALFWorldEnv(
                task=task,
                max_steps=self.max_steps,
                use_process_lock=False  # No lock needed - isolated process
            )
            return ("ok", True)
        except Exception as e:
            return ("error", str(e))

    def _reset(self) -> tuple:
        """Reset the environment."""
        if self.env is None:
            return ("error", "Environment not initialized")
        try:
            result = self.env.reset()
            return ("ok", result)
        except Exception as e:
            return ("error", str(e))

    def _step(self, action: Any) -> tuple:
        """Execute action."""
        if self.env is None:
            return ("error", "Environment not initialized")
        try:
            result = self.env.step(action)
            return ("ok", result)
        except Exception as e:
            return ("error", str(e))

    def _compute_final_reward(self) -> tuple:
        """Compute final reward."""
        if self.env is None:
            return ("ok", 0.0)
        try:
            result = self.env.compute_final_reward()
            return ("ok", result)
        except Exception as e:
            return ("error", str(e))

    def _close(self) -> tuple:
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
            self.env = None
        return ("ok", True)


def _worker_main(worker_id: int, task_queue: Queue, result_queue: Queue, max_steps: int):
    """Entry point for worker process."""
    worker = EnvWorkerProcess(worker_id, task_queue, result_queue, max_steps)
    worker.run()


class ParallelEnvPool:
    """Pool of parallel environment workers using multiprocessing.

    This provides a simpler alternative to Ray actors that can work
    without Ray cluster overhead.

    Usage:
        pool = ParallelEnvPool(num_workers=16)

        # Set tasks
        pool.set_task(0, task_dict)

        # Reset and step (async-friendly)
        obs, info = pool.reset(0)
        obs, reward, done, info = pool.step(0, action)

        # Cleanup
        pool.shutdown()
    """

    def __init__(self, num_workers: int = 16, max_steps: int = 50):
        """Initialize the parallel environment pool.

        Args:
            num_workers: Number of parallel worker processes
            max_steps: Max steps per environment episode
        """
        self.num_workers = num_workers
        self.max_steps = max_steps

        # Use spawn to ensure clean process state
        ctx = mp.get_context("spawn")

        # Create queues for each worker
        self.task_queues = [ctx.Queue() for _ in range(num_workers)]
        self.result_queue = ctx.Queue()

        # Start worker processes
        self.workers = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_worker_main,
                args=(i, self.task_queues[i], self.result_queue, max_steps),
                daemon=True
            )
            p.start()
            self.workers.append(p)

        print(f"[ParallelEnvPool] Started {num_workers} worker processes")

    def set_task(self, worker_idx: int, task: dict) -> bool:
        """Set task for a worker."""
        self.task_queues[worker_idx].put(("set_task", {"task": task}))
        worker_id, result = self.result_queue.get()
        status, value = result
        if status == "error":
            raise RuntimeError(f"Worker {worker_idx} set_task failed: {value}")
        return value

    def reset(self, worker_idx: int) -> tuple[dict, dict]:
        """Reset environment for a worker."""
        self.task_queues[worker_idx].put(("reset", {}))
        worker_id, result = self.result_queue.get()
        status, value = result
        if status == "error":
            raise RuntimeError(f"Worker {worker_idx} reset failed: {value}")
        return value

    def step(self, worker_idx: int, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute step for a worker."""
        self.task_queues[worker_idx].put(("step", {"action": action}))
        worker_id, result = self.result_queue.get()
        status, value = result
        if status == "error":
            raise RuntimeError(f"Worker {worker_idx} step failed: {value}")
        return value

    def compute_final_reward(self, worker_idx: int) -> float:
        """Compute final reward for a worker."""
        self.task_queues[worker_idx].put(("compute_final_reward", {}))
        worker_id, result = self.result_queue.get()
        status, value = result
        if status == "error":
            raise RuntimeError(f"Worker {worker_idx} compute_final_reward failed: {value}")
        return value

    def close(self, worker_idx: int):
        """Close environment for a worker."""
        self.task_queues[worker_idx].put(("close", {}))
        worker_id, result = self.result_queue.get()

    def shutdown(self):
        """Shutdown all workers."""
        for q in self.task_queues:
            q.put(None)  # Shutdown signal

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        self.workers = []
        print("[ParallelEnvPool] All workers shut down")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass
