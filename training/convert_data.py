"""Convert Memex ALFWorld datasets to Slime-compatible JSONL format.

Usage:
    python convert_data.py --output-dir /path/to/output [--max-train N] [--max-test N]

Each output line: {"prompt": "<task_dict_json>"}
The generate function parses sample.prompt via json.loads() to get the task dict.
"""

import argparse
import json
import os
import sys


def convert_alfworld(output_dir: str, max_train: int | None = None, max_test: int | None = None):
    from src.data.alfworld import prepare_alfworld_data

    train_dataset, val_dataset = prepare_alfworld_data(
        max_train_size=max_train,
        max_test_size=max_test,
    )

    os.makedirs(output_dir, exist_ok=True)

    for split_name, dataset in [("train", train_dataset), ("test", val_dataset)]:
        data = dataset.get_data()
        output_path = os.path.join(output_dir, f"alfworld_{split_name}.jsonl")

        with open(output_path, "w") as f:
            for task in data:
                line = {"prompt": json.dumps(task, ensure_ascii=False)}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"Wrote {len(data)} tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Memex data to Slime JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--max-train", type=int, default=None, help="Max training examples")
    parser.add_argument("--max-test", type=int, default=None, help="Max test examples")
    args = parser.parse_args()

    convert_alfworld(args.output_dir, args.max_train, args.max_test)


if __name__ == "__main__":
    main()
