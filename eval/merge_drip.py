"""Merge the overnight drip's per-seed rollout dumps into one dataset and
score it as a single Gate 1 run.

The drip harvests one episode at a time because Groq's TPD is a slowly
refilling token bucket (see eval/overnight_gate1_drip.sh). Each cycle
therefore writes its own dump and scores it alone, which is meaningless --
one episode is 3 questions and Gate 1's MIN_PER_BUCKET is 15. The numbers
only mean anything once the cycles are pooled, which is what this does.

Rollouts that errored (usually a 429 that aborted the episode mid-way) are
dropped rather than scored as zeros. Scoring an aborted rollout as a wrong
answer would silently turn a quota failure into an apparent task failure --
exactly the confound that voided the earlier Gate 1 numbers.

    python eval/merge_drip.py [--dir eval/results/drip] [--out merged.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from gate1_persistence_ablation import score_outputs  # noqa: E402


def merge(drip_dir: Path | None, out_path: Path | None,
          files: list[Path] | None = None) -> int:
    # Explicit files, not just a directory glob: the useful 120b rollouts
    # are scattered across dumps named for the run that produced them
    # (gate1_120b_batch200.json, gate1_postfix_120b_n30.json), and
    # renaming them to seed*.json purely to satisfy a glob would destroy
    # the record of which run each came from.
    if files:
        dumps = sorted(files)
    elif drip_dir is not None:
        dumps = sorted(drip_dir.glob("seed*.json"))
    else:
        dumps = []
    if not dumps:
        print(f"no dumps found in {drip_dir}")
        return 1
    n_questions: set = set()

    outputs: list[dict] = []
    window: int | None = None
    model: str | None = None
    n_err = 0

    for d in dumps:
        payload = json.loads(d.read_text(encoding="utf-8"))
        m = payload.get("model", "unknown")
        if model is None:
            model = m
        elif m != model:
            # Pooling models would report one model's name over a mixture of
            # two, and the models differ enormously here (120b answers the
            # visible bucket, 20b and qwen3.6-27b floor at 0.000), so the
            # mixture would be meaningless as well as mislabelled.
            print(f"refusing to merge: {d.name} is {m}, expected {model}")
            return 1
        w = payload.get("compaction_window")
        if window is None:
            window = w
        elif w != window:
            # Pooling across different compaction windows would compare
            # rollouts whose visible/compacted split was drawn differently.
            print(f"refusing to merge: {d.name} has window {w}, expected {window}")
            return 1
        n_questions.add(payload.get("n_questions"))
        for o in payload.get("outputs", []):
            if o.get("error"):
                n_err += 1
                continue
            outputs.append(o)

    print(f"merged {len(dumps)} dumps: {len(outputs)} usable rollouts, "
          f"{n_err} dropped as errored")
    if len(n_questions) > 1:
        # Allowed, unlike a model or window mismatch: the scored unit is
        # the question, and task-averaged accuracy averages within type,
        # so episodes contributing 3 vs 6 questions pool correctly. Said
        # out loud rather than silently, since it does mean episodes are
        # weighted unequally.
        print(f"note: pooling across n_questions={sorted(x for x in n_questions if x)} "
              f"-- legitimate (the question is the scored unit), but episode "
              f"weights differ")
    if not outputs:
        print("nothing usable to score")
        return 1

    if out_path:
        out_path.write_text(
            json.dumps(
                {"model": model, "compaction_window": window, "outputs": outputs},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"merged dump written to {out_path}")

    score_outputs(outputs, window, verbose=False, label=f"{model} (merged drip)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="eval/results/drip")
    p.add_argument("--files", nargs="*", default=None,
                   help="explicit dump paths; overrides --dir")
    p.add_argument("--out", default="eval/results/drip/merged.json")
    a = p.parse_args()
    return merge(
        Path(a.dir) if not a.files else None,
        Path(a.out) if a.out else None,
        [Path(f) for f in a.files] if a.files else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
