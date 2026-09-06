# Gate 1 findings, 2026-08-17

Gate 1 asks: with filesystem tools disabled, does the policy do materially
worse on facts that were compacted away than on facts still visible? If not,
the task doesn't force persistence and nothing downstream means anything.

**Status (2026-08-26): PASS-looking.** See the update at the bottom for the
run that got there. The original two runs below both read FAIL, for
different and partly bogus reasons -- kept for the record since the second
reason was a real design flaw in the gate, not a result about the task.

## Runs

| run | model | episodes | errors | visible | compacted | gap |
|-----|-------|---------:|-------:|--------:|----------:|----:|
| 1 | `openai/gpt-oss-120b` | 20 | 1 | 0.381 (n=21) | 0.181 (n=36) | +0.200 |
| 2 | `openai/gpt-oss-20b` | 20 | 2 | 0.050 (n=20) | 0.235 (n=34) | -0.185 |

Run 1 used the pre-fix scorer (see "Scorer bug" below), so its numbers are
depressed and not comparable to run 2. Run 1 could not be re-scored because
raw rollouts weren't saved at the time; re-running exhausted
`gpt-oss-120b`'s 200,000 token/day cap. Rollouts are now persisted
(`--dump`) and re-scorable offline (`--rescore`), so this cannot recur.

## Scorer bug (found, fixed, tested)

Asked a multi-hop question naming two keys, `gpt-oss-120b` submitted a
single merged entry:

    slot:   'sponsor_budget_and_badge_color'
    answer: 'Sponsor budget is $1200 and badge color is yellow.'

Both values correct. The scorer looked up each slot exactly, found neither,
and scored 0 -- recording "did not know the fact" when the model plainly
did. Fixed with a containment fallback plus word-boundary value matching;
10 tests in `eval/test_gate1_scoring.py`, 8 of which guard against the fix
becoming a free pass.

## The real problem: the buckets are not comparable

Breakdown of run 2 by question kind:

| bucket | kind | n | mean |
|--------|------|--:|-----:|
| visible | multi_hop | 8 | 0.000 |
| visible | single_hop | 4 | 0.250 |
| visible | temporal_update | 8 | 0.000 |
| compacted | multi_hop | 9 | 0.333 |
| compacted | single_hop | 16 | 0.312 |
| compacted | temporal_update | 9 | 0.000 |

Three distinct problems:

1. **The buckets have different question-kind mixes.** "Visible" is mostly
   multi_hop + temporal_update (16 of 20); "compacted" is mostly single_hop
   (16 of 34). The headline comparison is therefore measuring question
   difficulty, not visibility. The negative gap is an artifact of this.

2. **`any()` is the wrong visibility rule for multi-slot questions.** A
   multi_hop question is labelled "visible" if *either* slot is visible, but
   scored on getting *both* right. So "visible multi_hop" systematically
   means "one slot visible, one compacted" -- strictly harder than
   "compacted multi_hop", which is why it scores worse (0.000 vs 0.333).
   Multi-hop should be excluded from the gate, or labelled with `all()`.

3. **`temporal_update` scores 0.000 in both buckets**, so it contributes
   nothing to the gap while inflating the visible bucket's size. Inspecting
   the dump shows why, and it is *not* a scoring bug -- when the tool-less
   model answers at all, it reports the **stale** value:

       slot='keynote_speaker'    original='Liam Torres'  TRUE='Noah Becker'  said='Liam Torres'
       slot='deposit_due_date'   original='November 6'   TRUE='December 3'   said='November 6'

   Most of the time it returns no entry at all. This is exactly the failure
   durable-notebook is built to expose, so it is good news for the premise
   and bad news only for this particular metric.

## What Gate 1 needs before it can return a verdict

- Compare **within question kind**, on **single-slot questions only**
  (single_hop, temporal_update), where "visible" is unambiguous.
- Drop multi_hop from the gate metric, or relabel with `all()`.
- Grow the visible bucket. At `compaction_window=3` over 10 turns only ~2
  scenario turns survive, so few questions land in "visible" (n=4 single_hop
  in run 2 -- far too small to conclude anything).
- Re-run on `gpt-oss-120b` once its daily cap resets. `gpt-oss-20b` is weak
  enough that its floor effects (multiple kinds at exactly 0.000) obscure
  the signal.

## Caveat worth keeping

The scorer bug was suppressing the project's headline result, and the fix
made the number move in the favourable direction. The guard tests exist so
that fix could not silently become "score anything that looks plausible",
but the honest position is that Gate 1 has **not** passed, and no GPU spend
is justified until it does on its own terms.

---

# Update, later on 2026-08-17: two environment bugs found. All numbers above are void.

The corrected metric did not fix Gate 1; it exposed why Gate 1 could never
have worked. Every run recorded above was measuring a broken environment.

## Bug 1: the policy could end the episode before being asked anything

`submit_manifest` set `dn_manifest_submitted = True` unconditionally, and
that ends the rollout. It could therefore be called on turn 1, before any
facts had been delivered and before the QA turn existed.

Measured on the 35-episode run: **18 of 30 scored episodes never referenced
a fact past turn 6 of a 12-turn episode, and 5 referenced nothing at all.**

This explains every confusing number in this session. Facts labelled
"visible" sit at LATE turns, which the policy mostly never reached; facts
labelled "compacted" sit at EARLY turns, which it did see. Hence recall
averaging turn 3.13 when correct against a visibility boundary of turn 7,
and hence a negative gap in all three question types at once.

It matters far more for training than for the gate: quitting immediately is
a degenerate strategy GRPO is free to find, and it skips the whole task.
Fixed by refusing submission until `dn_qa_shown`, with an error telling the
model to keep reading. 5 tests.

## Bug 2: compaction orphaned tool responses from their assistant message

`dn_turn_starts[i]` is the index just AFTER the model's completion, so every
turn boundary sits at the start of the env's reply -- which begins with the
tool messages answering the assistant `tool_call` immediately before the
cut. Slicing there kept those tool messages and removed the assistant
message that owned them.

gpt-oss rejects that shape outright:

    400 ... HarmonyError: EncodingError: render failed: Tools should have a name!

4 of 6 rollouts failed this way. It was invisible until Bug 1 was fixed,
because short rollouts never slid the window far enough to cut
mid-conversation -- fixing one bug is what surfaced the other.

Fixed by dropping orphaned leading tool messages after the slice. Dropping
rather than pulling the owning assistant message back in is deliberate:
reaching backwards would let already-compacted content leak back into view,
which is the one thing this environment cannot allow. 3 tests, including one
asserting compaction still actually compacts.

## Verification status, stated precisely

- Bug 1 fix: unit tests only.
- Bug 2 fix: unit test reproduces the exact orphan shape and passes. **Not
  yet verified against the live API** -- Groq's daily token caps were spent
  before any rollout completed on the fixed code. A 1-token probe returns
  200 while a real episode still 429s, so "the probe passes" is not evidence
  of headroom.
- 112 tests passing overall.

## Consequence for Gate 2

Gate 2 has never run, and would have been hit by both bugs identically -- it
uses the same `submit_manifest` and the same compaction. Nothing is lost,
but nothing about it can be assumed either.

## What is actually known about the task

Nothing yet. The environment has never once been exercised end-to-end as
designed. Every Gate 1 number so far measures early quitting, not
persistence.

---

# Update, 2026-08-18

## The gate's verdict logic was making a claim it could not support

Re-scoring the 35-episode `gpt-oss-20b` dump offline (free, no API) gave
ta_vis=0.139, ta_comp=0.264, gap -0.125, and printed **FAIL** -- with all
three per-type permutation p-values non-significant (0.193, 0.726, 1.000).

That verdict was unsupportable, and provably so from the gate's own rules.
PASS requires `gap > 0.3` and `ta_comp < 0.35`; since `gap = ta_vis -
ta_comp` and `ta_comp >= 0`, PASS is arithmetically unreachable for any
`ta_vis <= 0.30`. So that run could never have returned PASS no matter how
the task behaved. Printing FAIL there asserts "the task may not force
persistence" on the basis of a policy that answered 14% of questions whose
answers were still on screen. That is a fact about the model.

Fixed: the decision is now a pure, tested `_verdict()` function with a floor
precondition at `FLOOR_VISIBLE = 0.30` -- a threshold derived from the PASS
rule rather than chosen. A run whose policy is at floor on the VISIBLE
bucket reports INCONCLUSIVE with the reason, not FAIL. FAIL is still
reachable and now means something specific: the policy demonstrably clears
the visible bucket AND compaction doesn't hurt it. Six tests in
`eval/test_gate1_scoring.py` cover the floor case (using the real 20b
numbers), the threshold boundary in both directions, a genuine PASS, a
genuine FAIL, precedence of the underpowered check, and the NaN bucket.

## gpt-oss-120b, 12 episodes, 6 questions each, window 3

| bucket | task-averaged | n |
|--------|--------------:|--:|
| visible | **1.000** | 5 |
| compacted | **0.000** | 24 |

gap +1.000; per-type permutation p = 0.016 (single_hop), 0.004
(temporal_update). Perfect recall with the facts on screen, total failure
once they are compacted away.

This is the pattern Gate 1 was built to detect, and it retrospectively
explains the earlier confusion: the negative gaps in the 20b runs were a
capability floor, not a property of the task.

**Still INCONCLUSIVE, but only on power**: n_vis = 5 < MIN_PER_BUCKET = 15.

## The real reason Gate 1 kept failing to reach a verdict

`compaction_window = 3` over `n_turns = 10` puts roughly 83% of questions in
the compacted bucket (5 visible vs 24 compacted above). The visible bucket
is the scarce one, so reaching MIN_PER_BUCKET that way needs ~3x the
episodes -- roughly 110,000 tokens, over half a day of Groq TPD refill.

At `compaction_window = 6` the buckets come out near-balanced: the 30-episode
20b run split 48 visible / 52 compacted. Same episodes, ~2.5x more visible
questions per token. Every subsequent 120b chunk uses window 6, pooled under
`eval/results/w6_120b/` (merge_drip.py refuses to pool across windows, which
is correct -- the two are not the same measurement).

## Rate limits: the earlier diagnosis was wrong

Last night's overnight drip was built around the daily cap (TPD, 200k,
refilling ~139 tok/min). The 12-episode run above hit **zero** TPD 429s and
**six** TPM ones (per-minute, limit 8000), losing half its rollouts. Every
Retry-After was under 11 seconds.

So the binding constraint at normal working pace is the per-minute cap, not
the daily one, and the fix is a retry budget rather than 40-minute pacing:
`get_groq_vf_client` default `max_retries` 5 -> 16. A daily-cap 429 asks for
8+ minutes, still above the SDK's 120s MAX_RETRY_AFTER_DELAY, so this does
not make a genuinely exhausted run hang.

The drip script's 40-minute cycle was solving the wrong problem.

## Gate 2 had the same defect class, and one worse one (fixed 2026-08-18)

Gate 2 had never run, so its verdict logic had never been exercised. Two
bugs, both found by reading it after fixing Gate 1's:

**1. The two halves of the verdict were OR'd across different models.**
`any_real_gap` and `any_room_to_train` were separate module-level booleans,
each set by any model in the loop, then AND'ed for the verdict. So a gap on
model A plus headroom on model B printed PASS -- while no single model had
ever shown an exploitable, movable gap. That conjunction for one model is
the only thing that justifies the GPU spend this gate guards.

**2. A floored model supplied the "room to train" half.** A model scoring ~0
under both graders satisfies `naive_mean < 0.95`, so it flipped the headroom
flag on. Two of the three default models are measured floored on this task.

These compose into a concrete false PASS, verified by replaying the real
lineup through both implementations:

    gpt-oss-120b  naive 1.00  hardened 0.00   (saturated: nothing to train)
    gpt-oss-20b   naive 0.00  hardened 0.00   (floored)
    qwen3.6-27b   naive 0.00  hardened 0.00   (floored)

    OLD -> any_real_gap=True, any_room_to_train=True  => PASS
    NEW -> FAIL: "a gap exists (gpt-oss-120b) but is already saturated"

Not hypothetical: 120b scores 1.000 on Gate 1's visible bucket, so it
saturating the naive grader is plausible, and the other two are measured at
floor.

Fixed the same way as Gate 1: a `ModelResult` dataclass plus a pure
`gate2_verdict()` function, with both exclusions (mostly-errored, at-floor)
reported in the verdict text rather than applied silently. 8 tests in
`eval/test_gate2_verdict.py`, written failing first.

## Update, 2026-08-26: Gate 1 PASS-looking

Eight days of real-time gap since the last attempt turned out to matter more
than any code fix: the TPD bucket is a hard-capped 200k/day, not something
that keeps accumulating across idle days, so waiting longer than ~a day
between attempts buys nothing extra -- but a full untouched day gives a full
200k-token run.

Rescored the existing `w6_120b/seed300.json` (8 episodes, 7 of them lost to
TPD before the `max_retries` fix) for free, offline: only 1 clean episode
survived, same direction as always (visible 1.000, compacted 0.000) but
nowhere near powered.

Ran a fresh 28-episode batch at the same window (`w6_120b/seed600.json`,
`--start-seed 600`). 17 episodes completed clean before the run re-hit the
same daily TPD ceiling (`Used 197744` at episode 17); the remaining 11 all
aborted on TPD 429s in a fast cascade, as expected once the bucket is empty
for the day.

Pooled `seed300` + `seed600` (`merge_drip.py --files ... --out pooled.json`):

    bucket     question type        n     mean
    visible    multi_hop            7    1.000
    visible    single_hop           6    0.833
    visible    temporal_update     14    0.929
    compacted  multi_hop            3    0.333
    compacted  single_hop          10    0.500
    compacted  temporal_update      2    0.000
    mixed      multi_hop           12    0.625   (excluded -- not a clean bucket)

    task-averaged accuracy, VISIBLE   : 0.921  (n=27)
    task-averaged accuracy, COMPACTED : 0.278  (n=15)
    task-averaged gap: 0.643

    question type        visible  compacted     gap       p
    multi_hop              1.000      0.333   0.667   0.071
    single_hop             0.833      0.500   0.333   0.303
    temporal_update        0.929      0.000   0.929   0.026

`_verdict()` returns **PASS-looking**: n_vis=27 and n_comp=15 both clear
`MIN_PER_BUCKET`, `ta_vis=0.921 > FLOOR_VISIBLE`, gap 0.643 > 0.3, and
`ta_comp=0.278 < 0.35`.

**Caveat, stated plainly rather than buried:** the per-type p-values are not
uniformly small. `temporal_update` (p=0.026) carries almost all of the
significance; `multi_hop` is marginal (p=0.071); `single_hop` is not
significant on its own (p=0.303) despite a real-looking 0.333 gap -- n=6/10
per bucket for that type is just thin. The pooled, task-averaged verdict is
real and is what the gate is designed to test, but "the task forces
persistence" is currently best-supported for temporal-update questions
specifically, not demonstrated with equal confidence across all three types.
More episodes at the same window would tighten single_hop and multi_hop
without changing the headline call.

**Decision: Gate 1 passes.** Proceeding to run Gate 2 for real (it has only
ever been unit-tested, never executed against a live model).
