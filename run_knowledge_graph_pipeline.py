from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the post-relation-extraction knowledge graph pipeline.")
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root path.")
    return parser.parse_args()


def run_step(script_path: Path) -> None:
    env = dict(os.environ)
    env.setdefault("PANDAS_NO_USE_NUMEXPR", "1")
    env.setdefault("PANDAS_NO_USE_BOTTLENECK", "1")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent.parent if script_path.parent.parent.exists() else script_path.parent,
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    steps = [
        args.project_root / "关系消歧" / "main.py",
        args.project_root / "属性抽取" / "main.py",
        args.project_root / "属性消歧" / "main.py",
        args.project_root / "知识存储" / "main.py",
    ]
    for step in steps:
        print(f"[run] {step}")
        run_step(step)


if __name__ == "__main__":
    main()
