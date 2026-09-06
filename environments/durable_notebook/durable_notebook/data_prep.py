"""Generate the fixed durable-notebook dataset once and write it to disk.

Run as a script: `python -m durable_notebook.data_prep`

The split is a pure function of seed (see generator._split_for_seed), so
re-running this with the same arguments always reproduces the same files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from durable_notebook.generator import Episode, generate_dataset

DATA_DIR = Path(__file__).parent / "data"


def _write_jsonl(path: Path, episodes: list[Episode]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode.to_dict()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-turns", type=int, default=15)
    parser.add_argument("--n-questions", type=int, default=6)
    args = parser.parse_args()

    train, held_out = generate_dataset(
        n_episodes=args.n_episodes,
        start_seed=args.start_seed,
        n_turns=args.n_turns,
        n_questions=args.n_questions,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATA_DIR / "train.jsonl", train)
    _write_jsonl(DATA_DIR / "held_out.jsonl", held_out)

    print(f"wrote {len(train)} train episodes -> {DATA_DIR / 'train.jsonl'}")
    print(f"wrote {len(held_out)} held_out episodes -> {DATA_DIR / 'held_out.jsonl'}")


if __name__ == "__main__":
    main()
