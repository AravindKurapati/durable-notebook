# TWEAK: disable optimizer CPU offload for LoRA training

Date: 2026-09-05
Status: **optim_cpu_offload=false alone did NOT fix it.** Applied, smoke-
tested clean on Stage 0, relaunched Stage 1 for real -- Step 1 still took
7m58s at 2338 tok/s, Peak Mem 8.9 GiB, essentially unchanged from the
9m19s/2555 tok/s/8.9 GiB baseline with offload on. Confirmed live via
`modal container exec` reading trainer.log directly out of the running
container (mid-run Volume reads are unreliable -- `outputs.commit()` in
`modal/app.py` only fires once, at the end). New leading suspect below:
`ac_offloading`, not `optim_cpu_offload`. `optim_cpu_offload=false` is left
in place (harmless, matches the reasoning even if it wasn't the dominant
cost) while this second candidate is tested.

## The finding

Stage 1 (naive grader, Qwen3-1.7B, A10G:2) was relaunched with `--detach`
after the earlier session's sequence-budget fixes. The orchestrator log
showed what looked like a periodic ~10-12min stall ("0 inflight rollouts")
recurring roughly every other step starting at step 4. Initial hypothesis
(dispatcher/rollout-buffer bug, or `torch.compile` recompiling on new
sequence shapes) turned out to be wrong in emphasis -- the orchestrator log
only shows the *orchestrator* process; `uv run rl` writes the *trainer*
process's own log to a separate file
(`{output_dir}/logs/trainer.log`, confirmed in
`prime_rl/entrypoints/rl.py`) that was never inspected until now.

Pulled `stage1_naive-retry1/logs/trainer.log` from the outputs Volume after
stopping the run (`modal volume get`, survives an interrupted run -- no
Volume commit issue here). It shows the real picture:

    Step 1 |  9m 19s | Throughput 2555 tokens/s | MFU 10.4% | Peak Mem. 8.9 GiB
    Step 2 |  6m 42s | Throughput 2566 tokens/s | MFU 10.5% | Peak Mem. 8.9 GiB
    Step 3 |  6m 42s | Throughput 2556 tokens/s | MFU 10.4% | Peak Mem. 8.9 GiB
    Step 4 |  4m 31s | Throughput 2569 tokens/s | MFU 10.5% | Peak Mem. 8.9 GiB
    Step 5 |  5m 44s | Throughput 2514 tokens/s | MFU 10.2% | Peak Mem. 8.9 GiB
    Step 6 |  4m 13s | Throughput 2564 tokens/s | MFU 10.4% | Peak Mem. 8.9 GiB

**Every trainer step took 4-9 minutes.** This is not a periodic anomaly --
it is the trainer's real steady-state speed. The orchestrator's own
"Step 1/2/3" timings (40s-100s, looked healthy) were measuring how fast it
could drain a backlog of rollouts it had already generated while racing
ahead of the much slower trainer, not real training throughput. Once that
backlog drained and the orchestrator hit `TARGET_LAG=1`
(`orchestrator.py`: "Maximum batches the orchestrator may run ahead of the
trainer"), it paused and waited for the trainer, which is what showed up as
"0 inflight rollouts" for ~10 minutes at a time. The dispatcher was working
correctly; the trainer was just this slow the whole time.

(`MFU 10.4%` is also computed against the wrong reference: line in the log
--"Peak FLOPS undefined for `NVIDIA A10`. Falling back to A100 (312
TFLOPS)" -- A10G's real peak bf16 FLOPS is roughly 40% of A100's, so true
MFU relative to A10G's own ceiling is higher than the raw number, maybe
~25%. Still low, and the throughput number (2500-2600 tok/s) is the real,
reference-independent evidence.)

## Root cause (source-verified against the real v0.7.0 pydantic schema)

`prime_rl.configs.trainer.ModelConfig.optim_cpu_offload: bool = True` --
default on, never overridden in any of our three configs. Per its own
docstring: "Offload only optimizer states (momentum, variance) to CPU,
keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU
offload while still saving GPU memory." It round-trips optimizer state to
CPU on every step to save GPU memory.

But `Peak Mem. 8.9 GiB` was reported on every one of the 6 steps -- well
under A10G's 24 GiB, 15 GiB of headroom to spare. With LoRA, the trainer
log itself confirms the optimizer only has to track a small state:

    LoRA enabled: 34,865,152 adapter params adapting 1,409,286,144 base params

~35M trainable params means AdamW's optimizer state (momentum + variance,
fp32) is roughly 3 x 35M x 4 bytes =~ 420 MB. There is no memory pressure
here that offloading is solving. It is pure overhead: every optimizer step
pays a CPU<->GPU transfer round trip for a few hundred MB of state that
would easily fit on-GPU alongside the rest of the 8.9 GiB already in use.

This is a plausible, well-reasoned candidate for most of the observed
per-step slowness, though not yet proven by an A/B measurement -- that is
what the smoke test below is for.

## Proposed fix

Add to `[trainer.model]` in all three configs:

    optim_cpu_offload = false

Same discipline as `TWEAK_stage12_sequence_budget.md`: applied identically
to `stage0_smoke.toml`, `stage1_naive.toml`, `stage2_hardened.toml`, so
Stage 1 and Stage 2 stay comparable and Stage 0 stays a real pre-check of
whatever shape Stage 1/2 will actually run with.

## Verification plan

1. Apply the change to all three configs.
2. Re-run `configs/validate_configs.py` offline (free) to confirm the
   field name and type are accepted by the real v0.7.0 schema before any
   GPU spend.
3. Smoke-test on `stage0_smoke.toml` (Qwen3-0.6B, 5 steps, ~$0.05-0.10) --
   cheap enough to iterate on if this alone doesn't fix it. Compare
   per-step trainer time and `Peak Mem.` against a mental baseline (a 0.6B
   model has even more headroom, so this mainly confirms no crash / no
   regression, not the full magnitude of the fix -- the real test is
   Stage 1 itself).
4. If Stage 0 looks healthy (no OOM, step time not obviously worse),
   relaunch Stage 1 for real and compare trainer.log step times against
   this run's 4m13s-9m19s baseline.

## Risk

Low. If 8.9 GiB was already safely under the 24 GiB ceiling with offload
*on*, keeping the optimizer state on-GPU too can only add a few hundred MB
-- nowhere near the ceiling. The `cpu_offload_mutual_exclusion` validator
confirms `fsdp_cpu_offload` (a different, heavier offload path) is already
off (`False` by default, unchanged), so there's no interaction to worry
about there.

If this alone doesn't explain the full slowdown, activation offloading
(`ac_offloading`, also on by default, also CPU-round-trip-based) is the
next candidate to look at -- deliberately not touched in this pass, to
change one variable at a time and keep the A/B clean.

## Update 2026-09-05: fix #1 failed, real culprit is likely `ac_offloading`

`optim_cpu_offload=false` only touches the optimizer's momentum/variance
state, which is round-tripped to CPU **once per step**. It genuinely wasn't
the bottleneck -- the live relaunch's Step 1 was statistically the same as
the baseline (7m58s vs 9m19s, 2338 vs 2555 tok/s, both 8.9 GiB peak).

The next candidate, source-verified: `ModelConfig.ac_offloading:
ActivationOffloadingConfig | None = ActivationOffloadingConfig()` -- also
on by default, also never overridden in any of our configs. Unlike the
optimizer offload, this wraps **every forward pass**
(`trainer/rl/train.py`: `with maybe_record_function("forward"),
maybe_activation_offloading(config.model.ac_offloading):`), moving
checkpointed activations to pinned CPU memory and back. Combined with
`ac.freq=1` (full activation checkpointing -- every layer's activations are
discarded and recomputed during backward, itself already a large compute
multiplier by design), this means CPU<->GPU round trips happening at
checkpoint-segment granularity, many times per step, not once per step like
the optimizer state. Much higher frequency, much better fit for a
systematic 4-10x slowdown.

Same justification as before: `Peak Mem. 8.9 GiB` out of A10G's 24 GiB
leaves 15 GiB of headroom. There is no memory pressure this offload is
solving for our LoRA setup.

### Proposed fix #2

Add to `[trainer.model]` in all three configs (alongside the existing
`optim_cpu_offload = false`):

    ac_offloading = "None"

(String `"None"` is this codebase's documented convention for explicitly
nulling an `Optional[Config]` field from TOML -- same pattern already used
for `compile = "None"` in the repo's own `configs/debug/rl/train.toml`.)

Same verification plan as before: validate offline, smoke-test on Stage 0,
then relaunch Stage 1 and compare Step 1's trainer.log numbers against both
the original baseline (9m19s/2555 tok/s) and the failed-fix run (7m58s/2338
tok/s).

If disabling `ac_offloading` alone doesn't close the gap either,
`ac.freq=1` itself (full activation checkpointing) becomes the next and
much bigger lever -- raising it (e.g. to 4 or 8, checkpointing less
aggressively) trades some of that same 15 GiB headroom for a potentially
large compute-time win, since full recompute at freq=1 is the single most
compute-expensive AC setting available.

## Update 2026-09-05 (later): fix #2 also failed

`ac_offloading = "None"` was applied identically, smoke-tested clean on
Stage 0 (steps 17.8-23.6s, Peak Mem 3.9 GiB, up slightly from 3.5 as
expected), then tested live on Stage 1. Step 1: **9m6s, 2436 tok/s, Peak
Mem 9.8 GiB** -- statistically indistinguishable from both the original
baseline (9m19s/2555 tok/s) and fix #1 (7m58s/2338 tok/s). Neither
CPU-offload flag is the dominant cost. Both were reasonable, cheap,
well-justified guesses given the memory headroom -- both wrong. Stopped the
run (`ap-UuNEuo9ntPT0rra11M6CEP`).

Cost note: at ~9min/step and ~$2.47/hr for A10G:2, a 60-step run costs
roughly $22, and Stage 2 needs the identical config again -- ~$44
combined, more than this month's remaining Modal budget at the time this
was measured. Flagged to the user; a second Modal account exists as a
backup if needed, which reduces the urgency but not the value of finding
the real cause.

Next candidate, still untested: `ac.freq=1` (full activation checkpointing
-- every layer's activations discarded and recomputed in backward, roughly
doubling forward-pass compute by design). Unlike the two offload flags,
this directly cuts FLOPs rather than just moving data around, and we've
now confirmed real headroom to spend (Peak Mem never exceeded 9.8 GiB of
24 GiB across any variant tested). This is the next thing to try.

## Update 2026-09-05 (later still): fix #3 (`ac.freq=4`) OOM'd on Stage 1

Smoke-tested on Stage 0 first: clean run, no crash, throughput up ~25-30%
(steps hit 5300-5700 tok/s vs ~4100-4400 with the offload-only fix) but
Peak Mem roughly tripled at this tiny 0.6B scale (3.9 -> 11.7 GiB). That
memory-scaling signal was real: tested directly on Stage 1
(`ap-9ONaYDmsrDwtJr1mdpUaem`, chosen over a more conservative first step
since a CUDA OOM fails fast and is cheap to detect either way) and it
crashed ~2m20s into training with:

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 594.00
    MiB. GPU 0 has a total capacity of 22.06 GiB of which 563.38 MiB is
    free. Process 1 has 21.50 GiB memory in use.

(Note: A10G's usable capacity is 22.06 GiB, not the nominal 24 -- some is
reserved by the driver/system.) Cheap failure as predicted -- cost was
minutes, not a full step. `freq=4` is ruled out for this model/GPU
combination. Next: `freq=2`, a smaller step down from full checkpointing,
same verification order (Stage 0 smoke, then a direct Stage 1 test given
OOM fails fast and cheap).

## Update 2026-09-05 (final): `freq=2` also failed -- all three candidates ruled out

Smoke-tested clean on Stage 0 (Peak Mem 9.0 GiB, between freq=1's 3.9 and
freq=4's 11.7, as expected; throughput 4633-5110 tok/s, comparable to
freq=4's gain). Tested live on Stage 1
(`ap-ObKhPo7dulbu69s8Ty1mf5`): Step 1 completed at **8m23s, 2628 tok/s,
Peak Mem 18.1 GiB**.

That is statistically the same speed as the untouched `freq=1` baseline
(9m19s, 2555 tok/s) -- well within normal step-to-step noise, not a real
improvement -- while Peak Mem roughly doubled (8.9 -> 18.1 GiB), leaving
only ~4 GiB of headroom for the rest of a 60-step run. Worse risk profile,
no speed benefit. Stopped the run and reverted `ac.freq` to `1` in all
three configs (equally fast, safest memory footprint).

**Conclusion: all three trainer-level candidates in this document
(`optim_cpu_offload`, `ac_offloading`, `ac.freq`) are ruled out.** Across
five live Stage 1 measurements at Qwen3-1.7B scale, Step 1 landed in a
tight band regardless of setting:

| Variant | Step 1 time | Throughput | Peak Mem |
|---|---|---|---|
| Baseline (both offloads on, freq=1) | 9m19s | 2555 tok/s | 8.9 GiB |
| optim_cpu_offload=false | 7m58s | 2338 tok/s | 8.9 GiB |
| + ac_offloading=None | 9m6s | 2436 tok/s | 9.8 GiB |
| + ac.freq=4 | OOM'd (~2m20s in) | -- | crashed at 21.5 GiB |
| ac.freq=2 (offloads still off) | 8m23s | 2628 tok/s | 18.1 GiB |

None of these moved throughput meaningfully (2338-2628 tok/s across every
variant that didn't crash -- a ~12% spread, consistent with step-to-step
noise, not a causal effect). This is no longer "a config bug we haven't
found yet" -- it looks like the genuine compute-bound cost of training a
1.7B-parameter model with this batch/sequence shape (batch_size=64,
group_size=8, seq_len up to 8192, LoRA rank=32) on a single A10G GPU. Real
next levers, not yet tried: a faster GPU (A10G is an inference-oriented
card, not a training-optimized one; Modal's A100/H100 options have
meaningfully higher real bf16 FLOPS and might be net cheaper in total
dollars despite a higher $/hr rate), or accepting the current pace and
reducing `max_steps` to fit budget as-is.

Final state: `optim_cpu_offload = false` and `ac_offloading = "None"`
remain applied (harmless, no downside observed) in all three configs;
`ac.freq` is back to `1` (the default, and empirically no slower than the
alternatives tested).

## Update 2026-09-05 (GPU switch): A100-40GB:2 confirmed real, worthwhile speedup

Given the trainer-level dead end above, tested `A100-40GB:2` in place of
`A10G:2` (real Modal cost, measured empirically via `modal billing
report` rather than looked up, since Exa wasn't connected this session).

Smoke scale (Qwen3-0.6B) looked ambiguous: ~$4.27/hr vs A10G's ~$2.6-2.7/hr
(~1.6x), but only ~45-50% higher throughput (6200-6500 vs 4100-4400
tok/s) -- roughly a wash, understating the real effect because a tiny
model's step time is dominated by fixed per-step overhead, not raw
compute.

**Stage 1 scale (Qwen3-1.7B) told a different story.** Step 2 (steady
state, past warmup): **3m26s at 4937 tok/s**, vs A10G's steady-state
6m42s at similar batch composition -- essentially half the time, ~1.9x
throughput. That ratio holds: a 60-step run now projects to roughly
3-3.5hrs / ~$13-14 total, actually *cheaper* than A10G's own ~6hr/~$15.6
projection despite A100's higher hourly rate, because the throughput gain
outpaces the price increase at this model scale. `SMOKE_GPU` in
`modal/app.py` switched to `"A100-40GB:2"` for Stage 1/2.

**A genuine live run hit an unexplained interruption.** Running the real
`stage1_naive.toml` on A100, 11 steps completed cleanly (reward climbing
0.60 -> 0.83, a real and healthy signal) before the app terminated at
17:14:21 with no application-level error anywhere in trainer.log or
orchestrator.log -- the raw log instead shows:

    [modal-client] Received a cancellation signal while processing input...
    Runner terminated.

This was not issued by us (no `modal app stop` call on that app id in this
session) and shows no OOM/exception/traceback -- it looks like a
Modal-platform-level event (possible A100 preemption), cause unconfirmed.
A checkpoint had been saved at step 10 just before this happened.

**Resume attempt (`resume_step = -1`) revealed a second, unrelated
issue.** Relaunching with the same run_tag to resume from that checkpoint:
the orchestrator's own resume-path resolution correctly found
`checkpoints: 10` under `{output_dir}/run_default/checkpoints`, but the
trainer's own `CheckpointManager` (in `prime_rl/trainer/ckpt.py`) resolves
its resume path from the plain `{output_dir}/checkpoints` -- a real path
mismatch inside prime-rl v0.7.0 between the orchestrator's and trainer's
own checkpoint-path resolution when LoRA + its multi-run manager
(`run_default` subdir) are both in play. The trainer logged "No
checkpoints found... Starting from scratch" and silently retrained steps
1+ from zero instead of resuming -- not a crash, just a silent no-op on
the resume feature. Confirmed via source (`entrypoints/rl.py`'s
`ckpt_base = config.output_dir` vs `trainer/ckpt.py`'s
`CheckpointManager.ckpt_dir = get_ckpt_dir(output_dir)`, neither going
through `MultiRunManager.get_run_dir()`, while the orchestrator's own
internal resume check apparently does).

Not worth patching -- prime-rl is a pinned external dependency and this is
deep internal plumbing outside this project's scope. Practical takeaway:
`resume_step = -1` is safe to leave in (degrades gracefully when no
checkpoint exists) but should not be relied on to actually save GPU time
after an interruption in this exact config shape (LoRA + filesystem
broadcast); a future interruption means accepting a from-scratch restart,
same as if the setting weren't there at all.

## Update 2026-09-05 (root cause of the interruptions, finally confirmed)

Checked the Modal web dashboard (not just the CLI) for the second failed
run: **Status: Cancelled, Execution time: 31m 27s** -- an exact, precise
record, not a guess. The full log timeline showed why: at exactly 31m27s
into the run, `[modal-client]` logged "Received a cancellation signal
while processing input", followed ~53s later by "Received SIGTERM,
terminating all processes." Both A100 attempts died at almost identical
elapsed wall-clock time (31m46s and 31m27s) despite reaching very
different step counts (11 vs 25) -- a fixed-duration signature, not a
step-count or GPU-availability one.

**Not an A100-specific limit.** Checked the account's billing/plan page
directly (Starter plan, no A100-specific restriction visible) -- ruled
out. The real mechanism: `modal run --detach` still creates an *ephemeral*
app whose lifetime is owned by the local CLI process that launched it.
`--detach` protects against a *clean* disconnect (Ctrl+C, closed
terminal), but this machine's own low-memory killer was SIGKILLing that
local process outright (confirmed via task notifications: "stopped
because the system is running low on memory", observed for both A100
attempts). A hard kill never lets the client send a clean detach/keepalive
signal, and Modal's backend apparently cancels the invocation after some
delay once the owning client goes fully silent -- regardless of
`--detach` having been requested at launch.

**Fix: switch from `modal run --detach` to `modal deploy` +
`Function.spawn()`.** A deployed app is a standing entity on Modal's own
infrastructure with no owning CLI session at all -- there is no local
liveness to lose. Added `modal/trigger.py`: deploys once
(`modal deploy modal/app.py`), then looks up the deployed `run_training`
function and calls `.spawn()` (fire-and-forget, returns a FunctionCall id
immediately, does not block). The local script exits the instant the job
is submitted; the remote run has zero dependency on it afterward. All
existing monitoring (`modal app logs`, `modal volume get`,
`modal container exec`) works unchanged against the resulting app/container
-- only the launch mechanism changed.

## Database impact

None. Same as `TWEAK_stage12_sequence_budget.md` -- this project has no
database.
