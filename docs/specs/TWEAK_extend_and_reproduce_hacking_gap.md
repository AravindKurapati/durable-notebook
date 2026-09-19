# TWEAK: strengthen the hacking-gap result (denser sampling, more steps, second seed)

Date: 2026-09-06
Status: **A done. B dropped. C deferred (blocked on credit). Real
follow-up is #1 below, not any of these.**

## Outcome (2026-09-06)

- **A (dense re-grade) — done, free.** All 60 steps of both runs re-graded
  under both graders. Result got *stronger*: rollout-weighted cheat rate
  29.0% (naive-trained) vs 10.3% (hardened-trained), a 64.5% relative
  reduction (was 45.8% on the 7-step sample). The "step-50 spike" was a
  sparse-sampling artifact and is gone from the writeup. README +
  `ANALYSIS_hacking_gap.md` + `docs/hacking_gap_curves.png` +
  `eval/regrade_results/` all updated to the 60-step data. Found and fixed
  3 real crashes in `eval/regrade_traces.py` on malformed tool calls
  (`eval/test_regrade_traces.py`, 6 tests).
- **B (extend to 120 steps) — dropped.** The dense data made it
  non-load-bearing: there is no spike to chase past step 60, and the
  step-60 dip is just noise. Config edits reverted to `max_steps = 60`.
- **C (second seed, `start_seed=1000`) — deferred.** Still the single most
  valuable check (one seed, one run per grader is the main open weakness),
  ≈$13 / ≈3h. Blocked: the primary Modal account (`akasharshininyu`) is at
  its monthly credit cap ($30.04 spent, resets ~Oct 1). The secondary
  account (`arvind-kurapati`, full ~$30) has the app deployed and ready
  (`modal deploy` done 2026-09-06). To run it: set `start_seed = 1000` in
  both configs' `[[orchestrator.train.env]]` `args`, then
  `MODAL_PROFILE=arvind-kurapati python modal/trigger.py
  configs/stage1_naive.toml --run-tag seed1k` (and stage2). Regrade with
  `--start-seed 1000`. Decided not worth doing now: it hardens a
  confirmatory result rather than making it interesting.

## The follow-up that actually matters (#1): attack the hardened grader

Not in the original scope of this spec. The current result is guaranteed
by construction (gameable grader gets gamed). The version with tension:
apply optimization pressure directly against the **hardened** grader and
see whether the policy finds its presence-not-slotting gap
(`hardened.py` accepts the true value appearing anywhere in the workspace,
not under the right question). If it does — harden v2 (require the value
under the correct slot; penalise workspace bloat) and show v2 holds where
v1 broke. That is a finding. Everything in this spec is polish by
comparison.

## Original plan (for reference)

## Why

`docs/ANALYSIS_hacking_gap.md` already lists its own three biggest
weaknesses:

1. **Sparse sampling.** The cheat-rate curve is 7 points out of 60 training
   steps per stage. The step-50 spike (Stage 1 cheat rate 64.5%, n=31) then
   the step-60 drop (16.7%, n=30) is the whole shape the analysis hangs on,
   and it is drawn through 2 points.
2. **One run per stage.** The spike-then-recovery shape has never been
   checked against a second data seed. Could be a seed artifact.
3. **Short horizon.** Training stopped at step 60. Whether the naive-grader
   cheat rate re-diverges, plateaus, or keeps falling after step 60 is
   unknown, and "it dropped at step 60" is currently load-bearing.

All three are addressable now. The `stage1_naive-deployed1` /
`stage2_hardened-deployed1` runs saved a full per-step `traces.jsonl`
(`run_default/rollouts/step_<n>/train/effective/traces.jsonl`) for every
one of the 60 steps, not just the 7 that were regraded. The resume path is
now fixed (`modal/trigger.py:_patch_ckpt_output_dir`, verified this session
on the Stage 0 test: picked up `step_5`, continued from step 6).

## The three sub-tasks

### A. Denser regrade (free, no GPU, no API)

Both graders are pure deterministic functions of
`(episode, workspace, manifest)` (`eval/regrade_traces.py`). Pull all 60
steps' `traces.jsonl` for both stages off the `durable-notebook-rl-outputs`
Volume and regrade every step (or every 2nd step if runtime is
unreasonable). Regenerate `docs/hacking_gap_curves.png` at up to 60-point
resolution and rewrite the ANALYSIS "honest limitations" bullet #1.

Also regrade `train/all/traces.jsonl` (the full pre-filter rollout batch,
larger n per step) alongside `train/effective` and report both — the
`effective` set is filtered by GRPO's zero-advantage drop and its n swings
15–64, which is most of weakness #1.

Cost: zero. Runtime: a few minutes of local CPU.

### B. Extend both runs to step 120 (primary Modal account)

Bump `max_steps` 60 -> 120 in `stage1_naive.toml` and `stage2_hardened.toml`
(identical change, preserves the Stage 1 vs Stage 2 comparison — grader is
still the only difference). Relaunch both with the **same run_tag**
(`deployed1`) so `resume_step = -1` + the trigger's ckpt-path patch resume
from `step_60` rather than restarting. `[ckpt] interval = 10` already set,
so the new range is sampled at the same cadence.

Runs on the primary account (`akasharshininyu`) — the `step_60` checkpoints
live on that account's Volume, so resume has to happen there.

Cost: ~60 more steps/stage. Stage 1 did 60 steps in 1h22m on A100-40GB:2
(~$4.27/hr, both GPUs), so ~$6/stage, ~$12 total. Primary account had
~$14 remaining this cycle (resets ~Oct 1) — fits, tight.

### C. Second data seed, fresh 60-step run (secondary Modal account)

New run_tag `seed1k`. Set `start_seed = 1000` in the `[[orchestrator.train.env]]`
`args` table of both configs (plumbs through `load_environment(start_seed=...)`
-> `generate_dataset`). Everything else identical to `deployed1`. Deploy the
app once to the secondary account (`arvind-kurapati`, already authenticated,
~$30 credit) and launch both stages there.

Regrade with `--start-seed 1000` so the offline dataset regeneration
matches.

Cost: 2 full 60-step runs, ~$12, on the secondary account's ~$30.

## Execution order

1. A first (free, immediate signal on whether the spike shape even holds up
   under dense sampling — if step 50 was a 2-point mirage, that reframes B
   and C).
2. B and C launched in parallel (different accounts, no contention).
3. Regrade B's and C's new steps as they land; update ANALYSIS + curves +
   README headline table once.

## Config changes (exact)

| file | change | applies to |
|---|---|---|
| `configs/stage1_naive.toml` | `max_steps = 60` -> `120` | B |
| `configs/stage2_hardened.toml` | `max_steps = 60` -> `120` | B |
| both | `args = { ..., start_seed = 1000 }` | C only — a **separate** edit on a branch/stash, NOT combined with the max_steps bump; C is a fresh run, not a resume |

B and C must not share a config edit: B resumes `deployed1` (needs
`max_steps=120`, no seed change), C starts `seed1k` fresh (needs
`start_seed=1000`, `max_steps` either value is fine since it runs from 0).
Simplest: do B's launch first with just the `max_steps` bump committed,
then add `start_seed` and launch C.

## Revert / cleanup

`max_steps = 120` stays (it is the real training horizon now). `start_seed`
for C is passed via a throwaway edit and reverted after launch — the
canonical configs stay at `start_seed = 0`. New run_tags never overwrite
`deployed1`.

## Database impact

None. This project has no database; state is per-rollout in-memory
workspace plus JSON/bin rollout dumps on the Modal Volume. No `SCHEMA.md`.
