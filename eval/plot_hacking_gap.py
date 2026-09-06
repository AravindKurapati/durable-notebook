"""Plot the naive-vs-hardened hacking-gap curves from regrade_traces.py's
summary.json. See docs/ANALYSIS_hacking_gap.md for the full writeup.

Usage: python eval/plot_hacking_gap.py eval/regrade_results/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_stage_step(file_path: str) -> tuple[int, int]:
    name = Path(file_path).stem  # e.g. "stage1_step50"
    m = re.match(r"stage(\d)_step(\d+)", name)
    if not m:
        raise ValueError(f"unrecognized trace filename: {name}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/hacking_gap_curves.png"))
    args = parser.parse_args()

    data = json.loads(args.summary.read_text())
    by_stage: dict[int, list[dict]] = {1: [], 2: []}
    for d in data:
        stage, step = parse_stage_step(d["file"])
        by_stage[stage].append({**d, "step": step})
    for stage in by_stage:
        by_stage[stage].sort(key=lambda d: d["step"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = {1: "#d62728", 2: "#1f77b4"}
    labels = {1: "Stage 1 (trained vs naive)", 2: "Stage 2 (trained vs hardened)"}

    for stage, rows in by_stage.items():
        steps = [r["step"] for r in rows]
        naive = [r["mean_naive_overall"] for r in rows]
        hardened = [r["mean_hardened_overall"] for r in rows]
        c = colors[stage]
        ax1.plot(steps, naive, "-o", color=c, label=f"{labels[stage]} -- naive score")
        ax1.plot(steps, hardened, "--o", color=c, alpha=0.55, label=f"{labels[stage]} -- hardened score")

    ax1.set_xlabel("training step")
    ax1.set_ylabel("mean grader score (re-graded offline)")
    ax1.set_title("Naive vs. hardened score, same rollouts")
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.25)

    for stage, rows in by_stage.items():
        steps = [r["step"] for r in rows]
        cheat = [r["cheating_rate_naive_high_hardened_low"] * 100 for r in rows]
        ax2.plot(steps, cheat, "-o", color=colors[stage], label=labels[stage])

    ax2.set_xlabel("training step")
    ax2.set_ylabel("% rollouts: naive >= 0.75 and hardened <= 0.25")
    ax2.set_title("Cheating rate (strict: looks done, isn't)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 70)
    ax2.grid(alpha=0.25)

    fig.suptitle("durable-notebook: the naive/hardened hacking gap", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
