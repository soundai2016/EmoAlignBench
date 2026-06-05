from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command reproduction for the EmoAlignBench paper artifact.")
    parser.add_argument("--config", default="configs/emobench_config.json")
    parser.add_argument("--recompute-stats", action="store_true", help="Recompute statistical tests instead of using cached statistics_tables.json.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    fig_cmd = [py, "03_generate_paper_figures_tables.py", "--config", args.config]
    if args.recompute_stats:
        fig_cmd.append("--recompute-stats")
    run(fig_cmd, root)
    run([py, "04_validate_release.py", "--config", args.config], root)


if __name__ == "__main__":
    main()
