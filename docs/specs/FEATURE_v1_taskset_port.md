# FEATURE: port durable-notebook to a verifiers v1 taskset

Status: planning. No code yet. Awaiting approval of approach.

## Why

Two outcomes, both optional and both after the v0 project is committed and re-pushed:

1. A second Hub listing that runs on the current verifiers architecture (v1), so the
   env stays usable as `verifiers` drops the `verifiers.legacy` tree (already gone on
   `main`, still shipped in the 0.3.1 release).
2. The real prize: a `docs/v1/migration.md` PR to `PrimeIntellect-ai/verifiers`. That
   file does not exist. Issue #1982 ("Hub quality scan tests the v0 API") is open since
   2026-07-13 with an unfulfilled maintainer promise from July. A migration guide built
   from a concrete port is a documented gap we can fill.

## Database impact

None. No database in this project. The env generates its dataset in-process.

## Current v0 shape (what we are porting from)

`environments/durable_notebook/durable_notebook/`

- `env.py` — `DurableNotebookEnv(vf.StatefulToolEnv)` plus `load_environment()`. Heavy
  subclass: overrides `setup_state`, `env_response`, `get_prompt_messages`,
  `update_tool_args`, `no_tools_called`, and a `_score_episode` rubric func.
- `tools.py` — `write_file`, `read_file`, `list_files`, `submit_manifest`. Each takes a
  hidden `state` arg carrying the per-rollout `Workspace`; `state` is stripped from the
  model-facing schema via `add_tool(..., args_to_skip=["state"])`.
- `workspace.py` — `Workspace`: a path-scoped filesystem sandbox, no code execution.
- `generator.py` — `Episode` / `episode_from_dict` / `generate_dataset`. Pure logic.
- `naive.py` / `hardened.py` — `grade(episode, workspace, manifest) -> dict`. Pure logic.
- `manifest.py`, `compaction.py` — pure logic.

The mechanism the whole environment exists to demonstrate: `get_prompt_messages` rebuilds
the visible prompt every step from only the last `compaction_window` scripted turns plus
the system message. Older turns are absent, not summarized. Persist to a file or lose it.

## Target v1 shape

v1 has no `MultiTurnEnv`, no `StatefulToolEnv`, no `env_response`, no `get_prompt_messages`.
The pieces are:

- **Taskset** (`taskset.py`): `DurableNotebookData(vf.TaskData)` holds the serialized
  `Episode`. `DurableNotebookTask(vf.Task[...])` carries `@vf.stop` and `@vf.reward`.
  `DurableNotebookConfig(vf.TasksetConfig)` exposes `grader`, `compaction_window`,
  `n_episodes`, `start_seed`, `n_turns`, `n_questions`. `load()` builds/generates the
  tasks (can be a generator).
- **Toolset** (`servers/tool.py`): `write_file` / `read_file` / `list_files` /
  `submit_manifest` as a `vf.Toolset`, exposed to the harness as an MCP server.
  Per-rollout isolation via `self.state` (the pattern the built-in `scratchpad` env
  demonstrates), not a hidden `state` arg.
- **Harness**: either a custom harness program (the built-in `compact` env is a working
  reference for exactly this) or a custom `vf.Env`. This is where the scripted-turn
  delivery and the compaction window move.
- Grader / generator / manifest / compaction modules port with near-zero change. They
  are already pure functions.

## The hard parts (these become the migration guide)

1. **No message-shaping hooks.** In v1 the harness owns the model loop; the environment
   cannot rewrite what the model sees per turn. Enforced compaction, the entire point of
   this env, has to be reimplemented inside a custom harness program or a custom Env.
   Undocumented. This is the whole reason a migration guide is worth writing.

2. **`StatefulToolEnv` + `update_tool_args(state)` is gone.** Per-rollout tool state is
   now `self.state` on the `Toolset`. The `args_to_skip` schema-filtering dance
   disappears (good), but you can no longer hand a live Python object to a tool.

3. **Tools are MCP servers in a separate process (and, on prime/modal runtimes, a
   separate machine).** The `Workspace` that was a live object in `state` now lives
   inside the tool server. The grader runs client-side and needs the workspace contents
   after the rollout. Options:
   - (a) shared filesystem path — only works on `subprocess` / local `docker`.
   - (b) at `submit_manifest` time, the tool server serializes every workspace file into
     the tool result text, so it lands in the trace; the reward function parses it back
     and builds an in-memory `Workspace`. Runtime-agnostic. Workspace is small (a few
     note files). **Recommended.**

4. **Reward functions take `trace: vf.Trace`, not `state`.** The v0 grader read
   `state["dn_workspace"]` and `state["dn_manifest"]`. In v1:
   - manifest: recovered from the `submit_manifest` tool call in the trace's message graph.
   - workspace: rebuilt from the serialized dump per point 3(b).

5. **Scripted multi-turn user delivery.** v0 did this in `env_response`. v1 options:
   the custom harness program drives it (compact pattern), or `UserSimEnv` with a
   scripted user agent. The guide should show both; the port uses the first.

6. **Dataset row shape.** v0: `{"question", "answer", "info"}` with a JSON `info` blob and
   a hand-rolled `episode_from_dict`. v1: a typed `vf.TaskData` subclass (Pydantic). The
   v0 workaround comment about the reserved `task` field becomes moot.

7. **Stop conditions.** v0 overrode a `@vf.stop` named `no_tools_called` to force "a turn
   may be a plain acknowledgement." v1: `@vf.stop` methods on the Task read the trace.
   The "acknowledgement is allowed" logic moves into the harness program's turn protocol.

## Approaches

### Approach A — custom harness program (recommended)

Port the scripted-turn loop + compaction into a `program.py` uv script plus a thin
`harness.py`, mirroring `environments/compact/`. The program:

- reads the serialized episode from the task prompt / an env var
- delivers episode turns one at a time as fresh `[system] + last N turns` prompts
- lets the model call the filesystem MCP tools or reply in plain text
- forces the QA turn once scripted turns are exhausted
- captures `submit_manifest` and ends the run

Filesystem tools as an MCP `Toolset` with per-rollout workspace under `self.state`.
Workspace contents serialized into the final tool result (point 3b) for the grader.

- Pro: `compact` is a maintained, working reference for this exact "rebuild context each
  turn" pattern. Tightest 1:1 with the v0 mechanism. Full control of the compaction rule.
- Con: you maintain a harness program. Must solve the workspace-across-process boundary
  (3b handles it). More surface than a pure taskset.

### Approach B — custom `vf.Env` subclass

Write `DurableNotebookEnv(vf.Env)` whose `run(task, agents)` invokes a minimal agent once
per scripted turn, rebuilding context between invocations from the Env side.

- Pro: matches v1's "the Env owns control flow" framing more than a harness hack does.
- Con: no close working example in-repo; the built-in Envs (`AgenticJudgeEnv`,
  `UserSimEnv`, `BestOfNEnv`) do not do per-turn context rewriting. Heavier, more
  invention, more risk. Multi-agent `Env` is really aimed at solver/judge/user setups,
  not single-agent context control.

### Recommendation

Approach A. It has a working reference, it is the smaller lift, and the friction it
exposes (harness program vs message hooks) is precisely what the migration guide needs
to document. Revisit B only if A hits a wall in the harness/runtime contract.

## File plan (Approach A)

```
environments/durable_notebook_v1/
  pyproject.toml                       # verifiers >=0.3.1,<0.4 ; name durable-notebook-v1
  durable_notebook_v1/
    __init__.py                        # exports DurableNotebookTaskset
    taskset.py                         # Data / Task / Config / Taskset
    harness.py                         # thin Harness wrapper
    program.py                         # the compaction + scripted-turn loop (uv script)
    servers/tool.py                    # filesystem Toolset (write/read/list/submit)
    generator.py  compaction.py  manifest.py  naive.py  hardened.py  workspace.py
                                       # copied from v0, near-unchanged
    tests/                             # port test_env / test_full_episode intent
```

## Validation (no GPU, no Modal)

- `uv run init` sanity, then `uv run validate durable-notebook-v1`.
- `uv run eval durable-notebook-v1 -n 5 -r 1` against a hosted model API
  (`gpt-4.1-mini` or `claude-haiku`), token cost only. Target: non-zero reward, naive
  grader loose, hardened grader stricter on the same rollouts.
- Port the v0 `test_full_episode` intent as a subprocess-runtime test.
- `prime env push` to `aravind-k` namespace only after eval looks right, and only with
  explicit go-ahead.

## Prerequisites and sequencing

1. Commit today's v0 changes in this repo (dense regrade, doc reframe, bug fixes).
2. `prime env push` the final v0 (needs the local `prime` CLI dependency conflict fixed).
3. Fresh branch for the v1 port. Build Approach A. Validate with `uv run eval`.
4. While mid-port, with the friction points above turned into concrete before/after
   snippets, open the `docs/v1/migration.md` PR against `PrimeIntellect-ai/verifiers`.
   File it during the port, not after; maintainer response time is the long pole.
5. If the v1 `prime env push` fails the Hub quality scan the way strickvl's
   `isaf-extraction` did, add that as a second data point on issue #1982.

Nothing in steps 2 through 5 happens without the user's explicit approval to act under
their Prime / GitHub identity.

## Migration guide outline (the PR deliverable)

`docs/v1/migration.md`:

1. When you need this (you have a v0 `load_environment` returning a `vf.*Env` subclass).
2. The mental model shift: env-owns-messages to harness-owns-messages.
3. Mechanical mappings
   - dataset rows to `vf.TaskData`
   - `Rubric` funcs to `@vf.reward` on the Task
   - `@vf.stop` overrides to Task stop methods
   - in-process tool callables to a `vf.Toolset` MCP server
   - per-rollout `state` object to `Toolset.self.state`
4. The hard case: environments that shaped the model's context (`get_prompt_messages`,
   custom `env_response`). Two patterns: custom harness program (with `compact` as the
   worked example) and custom `Env`.
5. Recovering post-rollout state in a reward function from the `Trace` when tools ran
   out of process.
6. Dual-exposing v0 `load_environment` and the v1 taskset from one package during the
   transition, and the Hub-scan implications (link issue #1982).
