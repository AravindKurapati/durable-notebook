"""Gate 1 -- persistence-necessity ablation.

Runs durable-notebook with the filesystem tools disabled
(disable_memory_tools=True): the policy can only answer from whatever is
still inside the compaction window by the time the QA turn arrives. If
accuracy on facts that were compacted away is no better than guessing --
and no worse than accuracy on facts that stayed visible -- the task
doesn't actually force persistence, and nothing downstream (the
naive/hardened grader contrast, GRPO training) means anything. This must
pass before any GPU spend is authorized.

Eval-only, no Modal, no GPU -- a handful of cheap Groq API calls. Run:

    python eval/gate1_persistence_ablation.py --n-episodes 20 --model llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ENV_PKG_DIR = EVAL_DIR.parents[0] / "environments" / "durable_notebook"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ENV_PKG_DIR))

# Model output routinely contains characters (narrow no-break space U+202F,
# smart quotes, em dashes) that Windows' default cp1252 stdout cannot encode.
# Without this, --verbose crashes mid-run with UnicodeEncodeError *after* the
# API calls have already been paid for, losing the whole diagnostic. Replace
# rather than raise: a mangled character in a debug dump is harmless, a lost
# run is not.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import Dataset  # noqa: E402
from groq_client import get_groq_vf_client  # noqa: E402

from durable_notebook.env import DurableNotebookEnv  # noqa: E402
from durable_notebook.generator import Episode, episode_from_dict, generate_dataset  # noqa: E402


def _visible_fact_ids_at_qa(episode: Episode, compaction_window: int) -> set[str]:
    """Fact ids whose turn was still inside the compaction window by the
    time the QA turn was shown. Mirrors DurableNotebookEnv.get_prompt_messages's
    own window arithmetic: len(episode.turns) scenario turns plus 1 QA turn
    are "delivered" in total; only the last `compaction_window` of those
    stay visible."""
    total_turns = len(episode.turns) + 1
    first_visible_turn_index = max(0, total_turns - compaction_window)
    return {f.id for f in episode.facts if f.turn_index >= first_visible_turn_index}


def _get(msg, key, default=None):
    """completion messages may come back as typed vf.Message objects or
    plain dicts depending on the pipeline path -- handle both rather than
    assuming, since this is purely a debugging aid and shouldn't itself
    become a source of unverified assumptions."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _render_completion_transcript(completion: list) -> str:
    """Render the model's full turn-by-turn transcript for a rollout --
    what it actually said/reasoned before submitting, not just the final
    manifest. Exists specifically to investigate cases like a tool-less
    model correctly answering a question about a fact it was never shown
    in its visible context: is it guessing, leaking from elsewhere in its
    own reasoning, or something else? The final manifest alone can't
    answer that."""
    lines = []
    for msg in completion or []:
        role = _get(msg, "role", "?")
        content = _get(msg, "content")
        tool_calls = _get(msg, "tool_calls")
        reasoning = _get(msg, "reasoning_content")
        parts = [f"[{role}]"]
        if reasoning:
            parts.append(f"(reasoning: {reasoning[:300]})")
        if content:
            parts.append(str(content)[:500])
        if tool_calls:
            for tc in tool_calls:
                name = _get(tc, "name", "?")
                args = _get(tc, "arguments", "")
                parts.append(f"<call {name}({str(args)[:300]})>")
        lines.append(" ".join(parts))
    return "\n    ".join(lines)


def _value_present(text: str, value: str) -> bool:
    """Whether `value` appears in `text` as a standalone value.

    Deliberately not equality: a model that answers "The badge color is
    yellow" knows the fact just as well as one that answers "yellow", and
    Gate 1 is measuring knowledge, not phrasing. Deliberately not plain
    substring either: bare `"5" in "1500"` would be true, which would hand
    out credit for values the model never said. The lookarounds require the
    match not to be glued to an adjacent alphanumeric, so "$1200" matches
    inside "is $1200 and" but "5" does not match inside "1500"."""
    if not text or not value:
        return False
    pattern = r"(?<![0-9a-z])" + re.escape(value.strip().lower()) + r"(?![0-9a-z])"
    return re.search(pattern, text.strip().lower()) is not None


def _score_question_from_manifest(episode: Episode, question, manifest: dict) -> float:
    """Direct answer-vs-ground-truth scoring. There is no workspace in this
    ablation (no filesystem tools were offered), so naive/hardened grading
    doesn't apply -- score the manifest's claimed answer directly against
    the true value for each supporting slot.

    Two lookups per slot, in order:

    1. An entry whose `slot` matches exactly. Normal, well-behaved case.
    2. Failing that, an entry whose slot NAME CONTAINS the target slot.
       Real models merge: asked a multi-hop question naming two keys,
       gpt-oss-120b submitted a single entry
       `slot="sponsor_budget_and_badge_color"` answering both correctly in
       prose. That is a manifest-protocol violation, but scoring it 0 would
       be recording "did not know the fact", which is false, and it is what
       made the first 20-episode Gate 1 run read FAIL (visible accuracy
       0.381) when the model's actual recall was fine.

    The containment requirement in (2) is what keeps this from becoming a
    free pass: one dump-everything entry under an unrelated slot name is
    never consulted, so it cannot score every question at once. A model that
    renames a slot beyond recognition still gets no credit -- accepted
    strictness, since the alternative is matching on values alone."""
    fact_by_id = {f.id: f for f in episode.facts}
    entries = [e for e in manifest.get("entries", []) if isinstance(e, dict)]
    entries_by_slot = {e.get("slot"): e for e in entries}

    slots: list[str] = []
    for fid in question.supporting_fact_ids:
        slot = fact_by_id[fid].slot
        if slot not in slots:
            slots.append(slot)

    hits = []
    for slot in slots:
        true_fact = next(
            f for f in episode.facts if f.slot == slot and f.id in question.supporting_fact_ids
        )
        candidates = []
        exact = entries_by_slot.get(slot)
        if exact is not None:
            candidates.append(exact)
        else:
            candidates.extend(
                e for e in entries if slot in str(e.get("slot", ""))
            )
        hits.append(any(
            _value_present(str(c.get("answer", "")), true_fact.value) for c in candidates
        ))
    return sum(hits) / len(hits) if hits else 0.0


def _question_visibility(episode: Episode, question, visible_ids: set[str]) -> str:
    """"visible" | "compacted" | "mixed" for one question.

    A question is only "visible" if EVERY slot it depends on was still in
    the window, and only "compacted" if none were. Anything in between is
    "mixed" and is excluded from the headline comparison.

    This replaces an `any()` rule that labelled a two-slot question
    "visible" when a single slot survived -- while still scoring it on
    getting BOTH right. That made "visible multi_hop" strictly harder than
    "compacted multi_hop" and produced a NEGATIVE gap (0.000 vs 0.333) in
    the 2026-08-17 gpt-oss-20b run: an artifact of the label, not a
    property of the task."""
    flags = [fid in visible_ids for fid in question.supporting_fact_ids]
    if all(flags):
        return "visible"
    if not any(flags):
        return "compacted"
    return "mixed"


def _task_averaged(scores_by_kind: dict[str, list[float]]) -> float:
    """Mean of per-question-type accuracies, NOT pooled accuracy.

    LongMemEval (ICLR 2025) reports exactly this alongside overall
    accuracy, with the guidance to "prefer task_averaged_accuracy to
    overall_accuracy -- it corrects for type imbalance". Type imbalance is
    the confound here: the visible bucket was 16/20 multi_hop +
    temporal_update while the compacted bucket was 16/34 single_hop, so a
    pooled comparison measured question difficulty rather than visibility.

    NaN, not 0.0, when there is nothing to average -- "no data" and "scored
    zero" are different claims and must not collapse into the same FAIL."""
    means = [
        statistics.mean(scores) for scores in scores_by_kind.values() if scores
    ]
    return statistics.mean(means) if means else float("nan")


def _permutation_test(a: list[float], b: list[float], n_iter: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation test on the difference of means.

    MemDelta (arXiv 2606.29914) faults this whole field for not reporting
    "whether observed differences survive a significance test". A
    permutation test is the right primitive here: it assumes nothing about
    the score distribution (these are bounded partial-credit scores, not
    normal), and it correctly refuses to find significance at the tiny
    per-type sample sizes these runs produce."""
    if not a or not b:
        return float("nan")
    rng = random.Random(seed)
    pool = list(a) + list(b)
    n_a = len(a)
    observed = abs(statistics.mean(a) - statistics.mean(b))
    hits = 0
    for _ in range(n_iter):
        rng.shuffle(pool)
        diff = abs(statistics.mean(pool[:n_a]) - statistics.mean(pool[n_a:]))
        if diff >= observed - 1e-12:
            hits += 1
    return hits / n_iter


async def run(n_episodes: int, n_turns: int, n_questions: int, compaction_window: int, model: str, max_concurrent: int, verbose: bool = False, dump_path: Path | None = None, start_seed: int = 0):
    client = get_groq_vf_client()
    train, held_out = generate_dataset(
        n_episodes=n_episodes, start_seed=start_seed, n_turns=n_turns, n_questions=n_questions
    )
    # Gate 1 is an eval-only sanity check, not a train/held-out-sensitive
    # training run -- use every generated episode. (A run with n_episodes=3
    # silently used only 2 of them here before this fix, since seed 0 lands
    # in held_out by generate_dataset's own seed%5==0 rule and was being
    # discarded.)
    episodes = list(train) + list(held_out)
    rows = [
        {"question": e.turns[0].text, "answer": "", "info": e.to_dict()}
        for e in episodes
    ]
    dataset = Dataset.from_list(rows)

    env = DurableNotebookEnv(
        dataset=dataset,
        compaction_window=compaction_window,
        disable_memory_tools=True,
        max_turns=n_turns + 10,
    )

    results = await env.generate(
        env.get_dataset(),  # NOT the raw `dataset` -- this is missing
        # example_id/prompt columns that only get added by
        # Environment._format_dataset() inside build_dataset(), which
        # get_dataset() triggers and the raw dataset never goes through.
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        state_columns=["dn_manifest"],
    )

    # Persist the raw rollouts BEFORE scoring anything.
    #
    # This exists because of a concrete, expensive mistake: a 20-episode run
    # surfaced a scorer bug, the scorer was fixed, and re-scoring the same
    # rollouts should have been free -- except only the summary had been
    # kept, so the run had to be repeated, which exhausted the model's
    # 200,000 token/day cap and blocked the re-check entirely.
    #
    # Rollouts are the expensive artifact and the scorer is the cheap,
    # frequently-wrong part. Never couple them: dump first, score after, and
    # re-score offline with `--rescore <path>` for free.
    if dump_path:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "n_episodes": n_episodes,
            "n_turns": n_turns,
            "n_questions": n_questions,
            "compaction_window": compaction_window,
            "outputs": [
                {
                    "info": o["info"],
                    "dn_manifest": o.get("dn_manifest"),
                    "error": o.get("error"),
                    # kept so --verbose transcripts work offline too; that
                    # diagnostic is what found the merged-slot scorer bug.
                    "completion": o.get("completion"),
                }
                for o in results["outputs"]
            ],
        }
        dump_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"raw rollouts written to {dump_path}")

    score_outputs(results["outputs"], compaction_window, verbose, label=model)


# PASS requires gap > 0.3 AND ta_comp < 0.35 (see _verdict). Since
# gap = ta_vis - ta_comp and ta_comp >= 0, PASS is arithmetically
# unreachable for any ta_vis <= 0.30. That number is derived from the PASS
# rule, not picked.
FLOOR_VISIBLE = 0.30


def _verdict(ta_vis: float, ta_comp: float, n_vis: int, n_comp: int,
             min_per_bucket: int) -> tuple[str, str]:
    """The gate's decision, as a pure function so it can be tested.

    Three ways to be INCONCLUSIVE, and they are different claims:

    * too few questions to say anything at all;
    * NaN, i.e. a bucket that produced no scores;
    * the policy is at floor on the VISIBLE bucket, so the run has no
      power to detect a compaction effect even if one exists.

    The third case is why this function exists. It used to fall through to
    an `else: FAIL`, which asserted "the task may not actually force
    persistence" about a policy that could not recall facts still on
    screen. Live example, 2026-08-18: re-scoring the 35-episode
    gpt-oss-20b run gave ta_vis=0.139, ta_comp=0.264 -- printed as FAIL,
    with all three per-type permutation p-values non-significant (0.193,
    0.726, 1.000). A model answering 14% of questions whose answers are
    still visible is not evidence about the task design; it is evidence
    about the model. FAIL is only a meaningful verdict when the policy has
    demonstrated it can do the task under the easy condition.
    """
    if n_vis < min_per_bucket or n_comp < min_per_bucket:
        return ("INCONCLUSIVE", (
            f"need >= {min_per_bucket} questions per bucket to say anything "
            f"(have visible={n_vis}, compacted={n_comp}). Run more episodes "
            f"or widen --compaction-window."))
    if ta_vis != ta_vis or ta_comp != ta_comp:  # NaN
        return ("INCONCLUSIVE", (
            "a bucket produced no scored questions, so there is no "
            "like-for-like comparison to make."))
    if ta_vis <= FLOOR_VISIBLE:
        return ("INCONCLUSIVE", (
            f"the policy is at floor on the VISIBLE bucket "
            f"(task-averaged {ta_vis:.3f} <= {FLOOR_VISIBLE:.2f}), i.e. it "
            f"cannot answer even when the facts are still on screen. This "
            f"run has no power to detect a compaction effect -- PASS is "
            f"arithmetically unreachable below {FLOOR_VISIBLE:.2f} -- so it "
            f"says nothing about whether the task forces persistence. Re-run "
            f"with a policy that clears the visible bucket first."))
    if ta_vis - ta_comp > 0.3 and ta_comp < 0.35:
        return ("PASS-looking", (
            "on a like-for-like, type-matched comparison the policy does "
            "markedly worse on compacted facts, so the task does appear to "
            "require persistence. Confirm the per-type p values above are "
            "small before trusting this."))
    return ("FAIL-looking", (
        "the policy clears the visible bucket but compacted-away accuracy is "
        "not much worse on a type-matched comparison -- the task may not "
        "actually force persistence. Investigate before spending GPU."))


def score_outputs(outputs: list, compaction_window: int, verbose: bool = False, label: str = "") -> None:
    """Score a batch and report it the way the memory-eval literature does.

    Reporting follows LongMemEval (ICLR 2025): per-question-type accuracy
    plus TASK-AVERAGED accuracy (mean over types), because pooled accuracy
    is dominated by whichever type happens to be most numerous in a bucket.
    Significance via a permutation test, per MemDelta (arXiv 2606.29914),
    which faults this field for reporting differences without one.

    `MIN_PER_BUCKET` exists so an underpowered run reports INCONCLUSIVE
    rather than PASS or FAIL. A gate that silently returns a verdict from
    n=4 is worse than one that refuses to answer.
    """
    MIN_PER_BUCKET = 15

    by_bucket: dict[str, dict[str, list[float]]] = {
        "visible": {}, "compacted": {}, "mixed": {},
    }
    errors = 0

    for output in outputs:
        episode = episode_from_dict(output["info"])
        manifest = output.get("dn_manifest") or {"entries": []}
        if output.get("error"):
            errors += 1
            if verbose:
                print(f"[{episode.episode_id}] ERROR: {output['error']}")
            continue
        visible_ids = _visible_fact_ids_at_qa(episode, compaction_window)
        if verbose:
            print(f"\n=== {episode.episode_id} ===")
            print("submitted manifest:", manifest)
            print("transcript:")
            print("    " + _render_completion_transcript(output.get("completion")))
            fact_by_id = {f.id: f for f in episode.facts}
        for q in episode.questions:
            score = _score_question_from_manifest(episode, q, manifest)
            bucket = _question_visibility(episode, q, visible_ids)
            by_bucket[bucket].setdefault(q.kind, []).append(score)
            if verbose:
                true_values = {
                    fact_by_id[fid].slot: fact_by_id[fid].value for fid in q.supporting_fact_ids
                }
                print(f"  [{bucket:<9}] {q.kind:<16} score={score:.2f}  "
                      f"truth={true_values}  q={q.text!r}")

    print(f"model={label}  episodes={len(outputs)}  "
          f"compaction_window={compaction_window}  errors={errors}")
    print()

    # -- per-type breakdown (the diagnostic that exposed the confound) --
    print(f"{'bucket':<11}{'question type':<18}{'n':>4}{'mean':>9}")
    for bucket in ("visible", "compacted", "mixed"):
        for kind, scores in sorted(by_bucket[bucket].items()):
            print(f"{bucket:<11}{kind:<18}{len(scores):>4}{statistics.mean(scores):>9.3f}")
    print()

    vis, comp = by_bucket["visible"], by_bucket["compacted"]
    n_vis = sum(len(v) for v in vis.values())
    n_comp = sum(len(v) for v in comp.values())
    ta_vis, ta_comp = _task_averaged(vis), _task_averaged(comp)

    print(f"task-averaged accuracy, VISIBLE   : {ta_vis:.3f}  (n={n_vis})")
    print(f"task-averaged accuracy, COMPACTED : {ta_comp:.3f}  (n={n_comp})")
    if by_bucket["mixed"]:
        n_mixed = sum(len(v) for v in by_bucket["mixed"].values())
        print(f"(excluded: {n_mixed} 'mixed' questions -- some slots visible, "
              f"some compacted; not a clean contrast either way)")
    print()

    # -- verdict, compared WITHIN question type ------------------------
    shared_kinds = sorted(set(vis) & set(comp))
    if not shared_kinds:
        print("INCONCLUSIVE: no question type appears in both buckets, so "
              "there is nothing to compare like-for-like.")
        return

    print(f"{'question type':<18}{'visible':>10}{'compacted':>11}{'gap':>8}{'p':>8}")
    per_kind_gaps = []
    for kind in shared_kinds:
        v, c = vis[kind], comp[kind]
        gap = statistics.mean(v) - statistics.mean(c)
        p = _permutation_test(v, c)
        per_kind_gaps.append(gap)
        print(f"{kind:<18}{statistics.mean(v):>10.3f}{statistics.mean(c):>11.3f}"
              f"{gap:>8.3f}{p:>8.3f}")
    print()

    gap = ta_vis - ta_comp
    print(f"task-averaged visible-minus-compacted gap: {gap:.3f}")

    verdict, message = _verdict(ta_vis, ta_comp, n_vis, n_comp, MIN_PER_BUCKET)
    print(f"{verdict}: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=20)
    # Groq's TPD is a continuously-refilling token bucket (200k/day =
    # ~139 tok/min), so a capped day can only be harvested in small
    # chunks spread over hours. Without a seed offset every chunk would
    # regenerate the SAME episodes and the chunks would not accumulate.
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-turns", type=int, default=10)
    parser.add_argument("--n-questions", type=int, default=3)
    parser.add_argument("--compaction-window", type=int, default=3)
    # Groq's catalog is not stable: both Llama models (the previous default)
    # were removed and now 404. Re-check `client.models.list()` before a run
    # rather than trusting this default.
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    # Sequential by default. All Groq models here share an 8000 TPM cap, and
    # concurrency is what turns a clean batch into a half-errored one.
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dump", type=str, default=None,
        help="Write raw rollouts here before scoring, so a later scorer change "
             "can be re-checked offline for free via --rescore.",
    )
    parser.add_argument(
        "--rescore", type=str, default=None,
        help="Score a saved rollout dump instead of calling the API. Free, "
             "offline, and the correct way to evaluate any scorer change.",
    )
    args = parser.parse_args()

    if args.rescore:
        payload = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        window = payload.get("compaction_window", args.compaction_window)
        print(f"re-scoring {args.rescore} (no API calls)")
        score_outputs(
            payload["outputs"], window, args.verbose,
            label=payload.get("model", "unknown"),
        )
        return

    dump_path = Path(args.dump) if args.dump else None
    asyncio.run(run(
        args.n_episodes, args.n_turns, args.n_questions,
        args.compaction_window, args.model, args.max_concurrent, args.verbose,
        dump_path, args.start_seed,
    ))


if __name__ == "__main__":
    main()
