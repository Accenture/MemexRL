"""
Data preparation for ALFWorld benchmark.

This module loads ALFWorld game files and registers them with the DatasetRegistry.
"""

import os
from pathlib import Path
from typing import Optional

from src.data.dataset import DatasetRegistry


# ALFWorld task types (6 types)
ALFWORLD_TASK_TYPES = [
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]


def get_alfworld_data_path() -> str:
    """Get ALFWorld data path from environment variable.

    Returns:
        Path to ALFWorld data directory.

    Raises:
        EnvironmentError: If ALFWORLD_DATA is not set.
    """
    alfworld_data = os.environ.get("ALFWORLD_DATA")
    if not alfworld_data:
        raise EnvironmentError(
            "ALFWORLD_DATA environment variable not set. "
            "Please run: alfworld-download -f"
        )
    if not os.path.exists(alfworld_data):
        raise FileNotFoundError(
            f"ALFWorld data directory not found: {alfworld_data}"
        )
    return alfworld_data


def prepare_alfworld_data(
    alfworld_data_path: Optional[str] = None,
    max_train_size: Optional[int] = None,
    max_test_size: Optional[int] = None,
    seed: int = 42,
) -> tuple:
    """Load ALFWorld game files and register with DatasetRegistry.

    Args:
        alfworld_data_path: Path to ALFWorld data. If None, uses ALFWORLD_DATA env var.
        max_train_size: Maximum number of training examples.
        max_test_size: Maximum number of test examples.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    import random

    if alfworld_data_path is None:
        alfworld_data_path = get_alfworld_data_path()

    alfworld_data_path = Path(alfworld_data_path)
    json_dir = alfworld_data_path / "json_2.1.1"

    if not json_dir.exists():
        raise FileNotFoundError(
            f"ALFWorld json_2.1.1 directory not found: {json_dir}"
        )

    train_tasks = []
    test_tasks = []

    # Scan splits
    splits = {
        "train": train_tasks,
        "valid_seen": test_tasks,
        "valid_unseen": test_tasks,
    }

    for split_name, task_list in splits.items():
        split_dir = json_dir / split_name

        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            continue

        # Find all game files (sorted for deterministic ordering)
        for game_file in sorted(split_dir.rglob("*.tw-pddl")):
            # Extract task type from path
            # Path format: json_2.1.1/train/pick_and_place_simple/trial_T20190906_123456/game.tw-pddl
            relative_path = game_file.relative_to(split_dir)
            parts = relative_path.parts

            if len(parts) >= 2:
                task_type = parts[0]
                trial_name = parts[1]
            else:
                task_type = "unknown"
                trial_name = game_file.stem

            task_id = f"alfworld_{split_name}_{task_type}_{trial_name}"

            task = {
                "task_id": task_id,
                "task_type": task_type,
                "game_file": str(game_file.absolute()),
                "split": split_name,
                "max_steps": 50,
                "data_source": "alfworld",
            }

            task_list.append(task)

    print(f"Found {len(train_tasks)} training games")
    print(f"Found {len(test_tasks)} test games")

    # Shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(train_tasks)
    rng.shuffle(test_tasks)

    # Apply size limits
    if max_train_size is not None:
        train_tasks = train_tasks[:max_train_size]
    if max_test_size is not None:
        test_tasks = test_tasks[:max_test_size]

    print(f"Train: {len(train_tasks)} tasks, Test: {len(test_tasks)} tasks")

    # Register with DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("alfworld", train_tasks, "train")
    test_dataset = DatasetRegistry.register_dataset("alfworld", test_tasks, "test")

    return train_dataset, test_dataset


def load_alfworld_data(
    alfworld_data_path: Optional[str] = None,
    split: str = "test",
) -> list[dict]:
    """Load ALFWorld data without registering to DatasetRegistry.

    Args:
        alfworld_data_path: Path to ALFWorld data.
        split: Which split to return ('train' or 'test').

    Returns:
        List of task dictionaries.
    """
    train_dataset, test_dataset = prepare_alfworld_data(
        alfworld_data_path=alfworld_data_path
    )
    return train_dataset.get_data() if split == "train" else test_dataset.get_data()


def get_task_type_distribution(tasks: list[dict]) -> dict[str, int]:
    """Get distribution of task types.

    Args:
        tasks: List of task dictionaries.

    Returns:
        Dictionary mapping task type to count.
    """
    from collections import Counter
    return dict(Counter(t["task_type"] for t in tasks))


if __name__ == "__main__":
    print("Testing ALFWorld data preparation...")
    print("=" * 80)

    try:
        train_dataset, test_dataset = prepare_alfworld_data(
            max_train_size=10,
            max_test_size=5,
        )

        train_data = train_dataset.get_data()
        test_data = test_dataset.get_data()

        print("\n" + "=" * 80)
        print("Sample training task:")
        print("=" * 80)
        if train_data:
            sample = train_data[0]
            print(f"Task ID: {sample['task_id']}")
            print(f"Task Type: {sample['task_type']}")
            print(f"Game File: {sample['game_file']}")
            print(f"Split: {sample['split']}")

        print("\n" + "=" * 80)
        print("Task type distribution (train):")
        print("=" * 80)
        dist = get_task_type_distribution(train_data)
        for task_type, count in sorted(dist.items()):
            print(f"  {task_type}: {count}")

    except (EnvironmentError, FileNotFoundError) as e:
        print(f"Error: {e}")
        print("\nPlease ensure ALFWorld is installed and data is downloaded:")
        print("  pip install alfworld")
        print("  alfworld-download -f")
