# TWEAK: sequence budget for Stage 1 / Stage 2

Date: 2026-08-18
Status: Option A applied 2026-09-01, approved by user via AskUserQuestion.
Applied identically to `stage1_naive.toml`, `stage2_hardened.toml`, and
also `stage0_smoke.toml` (not originally in scope here, but re-running
Stage 0 to confirm the fix only means something if it uses the same
max_turns/max_completion_tokens/seq_len shape stage1/2 will use).

**Option A alone did not fix it.** Re-running Stage 0 with max_turns=22
still showed Truncation 100.0% on all 5 steps, with mean turns 17-20 --
well under the new cap, so the turn count was never the actual bottleneck.
Root cause (source-verified against the real v0.7.0 RLConfig schema, not
guessed): `Qwen3RendererConfig.enable_thinking` defaults `true`, and none
of these configs ever set `[orchestrator.renderer]`. At
max_completion_tokens=320, Qwen3's own thinking trace burns most or all of
that budget on nearly every turn, independent of max_turns. Second fix
(also approved via AskUserQuestion, after weighing the research-validity
trade-off explicitly): `[orchestrator.renderer] name="qwen3"
enable_thinking=false`, applied identically to all three configs.

**Confirmed fixed 2026-09-01.** Stage 0 re-run with both fixes: Truncation
0.0% on all 5 steps (down from 100.0%). Reward 0.15-0.25, Trainable rate
noisy (17-100%) but nonzero every step, Turns 14.5-19.2. Run tagged
`v3nothink` under the `durable-notebook-rl-outputs` volume.

Deliberate, not permanent: re-enabling thinking with a properly-sized
budget is a reasonable follow-up ablation once past smoke tests, if it
turns out to change the hacking-gap result. Judged low risk for the core
result either way -- gaming the naive grader's shallow self-report check
is a policy shortcut, not a deduction, so it plausibly needs less
reasoning than doing the task honestly, not more. Gate 1/2 are unaffected
regardless, since those ran against real deployed models over the Groq
API, independent of this training-time renderer setting.

## The finding

Stage 0 (smoke, Modal A10G:2, prime-rl v0.7.0, Qwen3-0.6B) passed end to end
but reported `Truncation 100.0%` on all 5 steps. Every reward in that run was
therefore measured on a truncated episode.

Measured cause, not guessed. The task's own prompt is tiny:

| config | all turn text | question block | system msg | uncompacted floor |
|--------|--------------:|---------------:|-----------:|------------------:|
| stage0 (`n_turns=10, nq=3`) | 177 | 50 | 138 | ~365 |
| stage1/2 (`n_turns=12, nq=4`) | 199 | 63 | 138 | ~400 |

(mean over 20 seeds, cl100k_base, measured offline with no API calls.)

So ~400 tokens of the 4096 was the task. The other ~3700 was the policy's own
output: `DurableNotebookEnv.max_turns` defaults to **40** (`env.py:77`), and
Stage 0 rollouts really did run 18-29 turns. At `max_completion_tokens = 256`
the worst case is 40 x 256 = 10,240 tokens of completion alone, 2.5x the
`seq_len` it was given.

`max_turns` is not currently exposed by `load_environment()`, but it reaches
`DurableNotebookEnv` through `**kwargs`, so it IS settable from the toml
`args` table with no code change.

## Why the current Stage 1/2 configs do not fix it

Both already have `seq_len = 8192` and `group_size = 8`, which is why this
looked handled. It is not. They also raise `max_completion_tokens` to 512 and
leave `max_turns` at the default 40:

    40 turns x 512 tokens = 20,480  +  ~400 prompt  =  ~21k  >>  8192

Stage 1 and Stage 2 as written would truncate at least as badly as Stage 0
did, on a longer and much more expensive run.

## Two options

**A. Bound the turn budget to what the task needs (recommended).**

    seq_len = 8192          (unchanged)
    max_turns = 22          (new, via env args)
    max_completion_tokens = 320

    22 x 320 = 7,040  +  ~400 prompt  +  ~700 tool results  =  ~8,150

Rationale: a well-behaved rollout is one write_file per fact turn (12) plus a
manifest, so ~15 assistant turns is the honest ceiling and 22 leaves real
slack. The Gate 1 rollouts (memory tools disabled) used 4-8 messages. 40 is
not a safety margin, it is room for a small model to loop, and a looping
rollout is not training signal we want to pay for. Costs nothing extra in
GPU memory over what is already budgeted.

Risk: a policy that would have solved the task in 25 turns gets truncated
instead. Mitigated by the fact that truncation is already reported per step,
so a persistent nonzero truncation rate at 22 is visible immediately.

**B. Raise the sequence budget to fit 40 turns.**

    seq_len = 16384
    max_turns = 40          (unchanged default)
    max_completion_tokens = 384

    40 x 384 = 15,360  +  ~1,100  =  ~16,460, so really seq_len = 20480

Rationale: never truncates, imposes no prior on how many turns the task
"should" take, and lets the hacking-gap result speak without an artificial
turn cap that a reviewer could call a confound.

Cost: 2-2.5x the activation memory of option A on a task whose actual content
is 400 tokens, on an A10G:2 with ~$27 of credit remaining and two full runs
(Stage 1 naive + Stage 2 hardened) still to pay for.

## Recommendation

Option A. The turn cap is the variable that actually blew the budget, and
bounding it is both cheaper and more honest than paying to store 20k tokens
of a 0.6B model repeating itself. Option B's "no artificial cap" argument is
real but is better answered by reporting the truncation rate at 22 turns than
by buying headroom we have evidence is only consumed by degenerate rollouts.

Whichever is chosen, apply it identically to `stage1_naive.toml` and
`stage2_hardened.toml`. Stage 2 differs from Stage 1 in `grader` only; any
other difference destroys the comparison.

## Database impact

None. This project has no database; state is per-rollout in-memory workspace
plus JSON rollout dumps under `eval/results/`. No `SCHEMA.md` change.
