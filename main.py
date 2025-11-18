"""Simple CLI to run different parts of the ML project."""

from __future__ import annotations

import argparse

from eda import run_eda
from export_notebook import export_notebook
from problem_description import describe_problem
from train_models import train_and_select_best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud detection project helper CLI.")
    parser.add_argument(
        "--command",
        choices=["describe", "eda", "train", "export"],
        default="describe",
        help="Which stage to run. Defaults to 'describe'.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used during training (train command only). Defaults to 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "describe":
        print(describe_problem())
    elif args.command == "eda":
        run_eda()
    elif args.command == "train":
        train_and_select_best(random_state=args.random_state)
    elif args.command == "export":
        export_notebook()
    else:
        raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
