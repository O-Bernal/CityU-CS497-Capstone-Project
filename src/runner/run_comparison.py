"""Render the configured task/library/condition matrix for comparison runs."""

import argparse

from src.core.config import load_config


def main() -> None:
    """Read comparison config and print all planned experiment combinations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tasks = cfg.get("comparison", {}).get("tasks", [])
    libraries = cfg.get("comparison", {}).get("libraries", [])
    conditions = cfg.get("comparison", {}).get("conditions", [])

    print("Planned comparison matrix:")
    for task in tasks:
        for lib in libraries:
            for cond in conditions:
                print(f"- task={task}, library={lib}, condition={cond}")


if __name__ == "__main__":
    main()
