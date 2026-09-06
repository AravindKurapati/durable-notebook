# FEATURE: durable-notebook — an RL environment for agentic filesystem/memory reward hacking

- Date: 2026-08-11
- Status: Approved, entering implementation
- Author: Aravind (with Claude Code)
- Project dir: `D:\Aru\NYU\durable-notebook`

## One-liner

A `verifiers` RL environment where a model receives facts across many turns
that get compacted out of visible context once acknowledged, so the only way
to retain them is to write them to files in a scoped workspace. At the end it
must answer questions requiring exact recall of what it wrote earlier. Two
graders, same task: a naive one that trusts a shallow self-reported completion
manifest, a hardened one that re-reads the actual files against ground truth.
GRPO-train a small Qwen3 policy under both, show the hacking gap open and
close, publish to the Environments Hub.

## Motivation and context

Prime Intellect's Environments Hub launch post (2025-08-27) lists "evaluating
usage of filesystems and memory for long-running tasks" as an unmet need.
Checked whether it's since been filled: INTELLECT-3 (2025-11-25) trained on
six categories — Math, Code, Science, Logic, Deep Research, Software
Engineering — none of which is filesystem/memory, and its own "Future Work:
Long Horizon Agents" section states this is still open (external memory
across turns, not yet RL-able). `community-environments` (219 open issues)
has nothing matching this gap either. Verified current, not stale.

This is a new, standalone project — not a continuation or reuse of any other
repo in this workspace, including as a design template.

## Goals

- Ship a `verifiers`-spec environment, published and installable from the
  Environments Hub.
- Demonstrate reward hacking emerging from plain GRPO pressure against the
  naive grader (no prompting for it).
- Quantify with the field-standard metric stack: automatic hacking gap
  (naive-grader score vs. hardened-grader score on the same rollouts) plus an
  LLM-judge confirmed-cheating rate.
- Show the hardened grader closes the gap under an otherwise identical run.
- Two hard gates (persistence-necessity ablation, exploitability pilot) must
  pass before any GPU spend — see below.

## Non-goals (YAGNI)

- No code execution tool — filesystem only (`write_file`/`read_file`/`list_files`).
- No claim of beating any long-context benchmark; capability is not the point.
- No custom v1-native taskset/harness/runtime implementation. `DurableNotebookEnv`
  is a classic v0 `StatefulToolEnv` and trains through **prime-rl v0.7.0**'s v0
  bridge (`[[orchestrator.train.env]]` with `id` + `args`), not a port. This
  requires pinning v0.7.0 — prime-rl main removed the v0 path. See "Trainer"
  below for the full verification trail; the pin is load-bearing, not incidental.
  (Earlier drafts of this doc referenced a "single-GPU verifiers nano trainer" —
  that doesn't exist; corrected after cloning prime-rl directly.)

## Architecture

1. **Task generator** (`durable_notebook/generator/`). Procedurally builds
   episodes: N ground-truth facts distributed across T turns, BABILong-style
   generic distractor filler, LoCoMo-style single-hop/multi-hop QA at the end,
   and the expected self-report manifest format. Deterministic seed, generated
   once offline, fixed train/held-out split — not regenerated per rollout.

2. **Environment** (`durable_notebook/env/`). `verifiers` `load_environment()`
   wrapping a multi-turn loop: environment-enforced compaction (fixed
   schedule drops old turns from visible context regardless of what the model
   does — the model never controls what gets summarized away), the three
   filesystem tools scoped to a per-episode workspace dir, and final QA turns.

3. **Graders** (`durable_notebook/graders/`), swappable via an env arg:
   - `naive.py`: checks the self-reported completion manifest is present and
     its referenced files exist, are non-empty, and are keyword-plausible.
     Not a zero-check — a shallow check a rushed real implementation would
     plausibly ship. Exploitable by writing a plausible-looking manifest
     instead of accurately maintaining one.
   - `hardened.py`: ignores the self-report, re-reads the actual workspace
     files at answer time, scores exact content against ground truth.
   - Reward is partial-credit per held-out question, not one binary per
     episode — more training signal per dollar of expensive multi-turn
     rollout.

4. **Gates** (`durable_notebook/eval/` or `eval/`):
   - Gate 1, persistence-necessity ablation: a model with the filesystem tool
     disabled must score at chance on final questions, or the compaction/task
     design doesn't actually force persistence and nothing downstream means
     anything.
   - Gate 2, exploitability pilot: a handful of current models via API against
     both graders, eval-only, no Modal spend. Confirms the naive grader is
     exploitable above baseline and not already saturated at ceiling/floor —
     i.e. there's a real gap for GRPO training pressure to move.

5. **Training configs** (`configs/`). prime-rl v0.7.0 `rl.toml` configs, each
   referencing `DurableNotebookEnv` via `[[orchestrator.train.env]]` (`id` +
   `args`, forwarded to `load_environment`), for Stage 1 (naive grader) and
   Stage 2 (hardened grader) GRPO runs on Qwen3 (1.7B or 4B), plus a
   cents-scale Stage 0 smoke-test config. All three validate cleanly against
   the real v0.7.0 `prime_rl.configs.rl.RLConfig` pydantic model offline
   (`configs/validate_configs.py`, 2026-08-17) — rerunnable, needs no GPU, no
   API key, and no durable-notebook install. This is the cheap pre-flight
   before any GPU spend.

6. **Eval and analysis** (`eval/`). Hacking-gap curves over training steps,
   LLM-judge (gpt-5-nano class) confirmed-cheating rate, 2-3 harvested exhibit
   transcripts.

## Model and training

- **Policy: Qwen3, 1.7B to start, 4B if budget allows.** Plain dense
  transformer — mature LoRA/vLLM support, no hybrid-architecture risk. One
  family spans smoke test through the real run via size parameter alone. Real
  precedent: Prime Intellect's own INTELLECT-2 (32B) was built on QwQ-32B, a
  Qwen-architecture model.
- **Trainer: prime-rl, pinned to v0.7.0, via its v0-environment bridge.**
  The pin is the load-bearing decision here, established 2026-08-17 after two
  wrong turns (first "there's a nano trainer" — there isn't; then a reading of
  `[legacy]` that was right for one version and wrong for another).

  **prime-rl v0.7.0 (2026-07-14) trains a classic v0 verifiers environment
  natively.** Verified in the real source at that tag:
    - `configs/orchestrator.py`: `EnvConfig.is_legacy` is documented as "A
      v0/legacy env (run via the bridge): an `id` is set and no v1 `taskset`
      is" — implemented as `return not self.taskset.id`.
    - Its validator error names both paths: `'no env configured — set
      taskset = { id = "<id>" } (v1) or id = "<id>" (v0/legacy)'`.
    - `orchestrator/envs.py::_spawn` passes `legacy=True, env_id=...,
      env_args=..., extra_env_kwargs=...` when `config.is_legacy`, spawning
      verifiers' `LegacyEnvServer`, which handles `TrainClientConfig` and so
      records per-turn token ids + logprobs — real GRPO signal, not eval-only.
    - Rollouts are addressed by `task_idx`, which is exactly what
      `LegacyEnvServer` expects (it indexes its dataset server-side).
    - `configs/alphabet_sort/rl.toml` at that tag uses this very shape.

  **prime-rl main / v0.8.x removed it.** `TrainSourceConfig` and
  `EnvServerConfig` no longer carry any legacy field;
  `entrypoints/env_server.py` calls `serve_env(...)` with no `legacy=`, so it
  always builds the v1 `EnvServer`; `LegacyEnvConfig` survives only in
  `verifiers/v1/configs/cli/eval.py` (i.e. `vf-eval`). Bypassing the launcher
  does not rescue it either: the wire `RunRequest` requires *exactly one* of
  `task_data` (v1) or `task_idx` (legacy), and main's orchestrator owns the
  taskset client-side and structurally always ships `task_data`.

  So: on main, durable-notebook would need a full v1 taskset + harness port —
  and that port is not cosmetic, because v1 runs the policy as an external
  program over MCP tools rather than driving the message loop in-process,
  which is precisely the assumption environment-enforced compaction rests on.
  Pinning v0.7.0 buys that entire port back as scope we do not spend. The cost
  of the pin is being one minor release behind, which is ordinary and stated
  plainly in the README.

  Corroboration that this path works in practice: the sibling
  `reward-hacking-env` project ran a full prime-rl GRPO loop on Modal against a
  classic v0 environment (`vf.load_environment("funcsynth-reward-hack", ...)`)
  — smoke passed 2026-06-04 on H100:2, LoRA served, checkpoint written.

  GPU count unchanged: default to 1 (the `alphabet_sort` shape — multi-turn,
  LoRA, no SFT — is documented on a single H100), with 2 (1 trainer + 1
  inference) as the accepted fallback.
- **Judge:** cheap API model, separate calls, no added GPU load.

## Budget and staged gates

$30 Modal credit in the primary account, second account as backup. Multi-turn
rollouts cost more per step than single-turn tasks: small policy, targeting
a single GPU (see Trainer above, with 2 GPUs as the accepted fallback),
short episodes (10-12 turns, per what actually worked live in Gate 1 —
15 hit the tight free-tier TPM limits harder), staged smoke/go-no-go gates
before any run that spends real money, explicit check-in before each
GPU-spending stage — never launch a training stage without saying so first.

## Prior art

- Reward Hacking Benchmark / RHB (Thaman, 2026) — eval-only, 13 frontier
  models via API, chained tool-use exploits including a memory/file-leakage
  family. Different threat model (metadata leakage via a reachable
  grader-internal file, not self-report trust); no training, no Hub artifact.
- `camgeodesic/reward_hacker_v1` (HF) — actual GRPO hack-emergence run, but
  code test-harness exploits, not filesystem/memory, not published to the Hub.
- Anthropic, "Natural emergent misalignment from reward hacking in production
  RL" (Nov 2025) — production coding environments, generalization focus, not
  this domain.
- CapReward/CapCode — reward-shaping defense (score capping), different
  mechanism than the grader-swap contrast used here.

No prior work combines GRPO + filesystem/memory long-horizon domain +
published Hub artifact, and Prime Intellect's own team still lists the domain
as open per INTELLECT-3.

## Risks and mitigations

- **Hack doesn't emerge, or naive grader already saturated.** Mitigated by
  Gate 2 before any GPU spend.
- **Compaction/task design doesn't actually force persistence.** Mitigated by
  Gate 1 before any GPU spend.
- **Modal/vLLM/trainer integration friction** (this is where a prior,
  unrelated project in this workspace lost time). Mitigated by staged
  smoke-test gates, single-GPU nano trainer over disaggregated prime-rl, small
  dense Qwen3 policy over any hybrid architecture.
- **Budget overrun.** Staged gates, short episodes, explicit check-in per
  GPU-spending stage.

## Database impact

None. This is a training/eval pipeline: generated task data lives as flat
files (JSON/JSONL) under `durable_notebook/generator/data/`, episode
workspaces are ephemeral per-rollout directories, run metrics are written as
flat result files under `eval/`. No persistent database. If a local run-metrics
store is later added, a `SCHEMA.md` will be created at that point per
workspace convention.

## Deliverables

- Published Environments Hub page.
- README with visible-vs-hidden (naive-vs-hardened) curves, confirmed-cheating
  rate, exhibit transcripts.
- This spec.
- Reproducible Modal scripts and pinned configs.
- Per-repo `CLAUDE.md`/`AGENTS.md`, if added, stays gitignored.

## Open questions

- Exact N (fact count) / T (turn count) for v1 — set empirically once the
  generator exists, tuned so Gate 1 actually discriminates.
- LoRA rank and step budget for Stage 1/2 — set empirically after the smoke
  test.
- Published env name (e.g. `durable-notebook`) — finalize before Hub publish.
