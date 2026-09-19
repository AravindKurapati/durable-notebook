"""Fire-and-forget trigger for a DEPLOYED durable-notebook-rl app.

Why this exists (docs/specs/TWEAK_trainer_optim_offload.md): `modal run
--detach` still ties an ephemeral app's lifetime to the local CLI process
that launched it -- `--detach` protects against a clean disconnect, but a
hard OS-level kill of that local process (observed twice: this machine's
own low-memory killer SIGKILLing the local `modal run`) eventually gets
treated by Modal's backend as an abandoned invocation and cancels it. Both
real Stage 1 attempts on A100 died this way at ~31 minutes, confirmed via
the Modal dashboard (status: Cancelled, exact execution time logged).

A DEPLOYED app has no such dependency: the app is a standing entity on
Modal's infrastructure, not owned by any one CLI invocation. This script
looks up the deployed function and calls `.spawn()` -- fire-and-forget,
returns a FunctionCall id immediately without blocking -- so this local
script can exit (or crash) the instant the job is submitted with zero
effect on the remote run.

Usage:
    modal deploy modal/app.py                      # once, or after any code change
    python modal/trigger.py configs/stage1_naive.toml --run-tag a100run1

Check on it later with the returned FunctionCall id:
    python -c "import modal; print(modal.FunctionCall.from_id('<id>').get(timeout=1))"
or just use the same modal app logs / volume get / container exec workflow
already established for monitoring -- spawn doesn't change any of that,
only how the job is launched.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import modal
import tomli_w

APP_NAME = "durable-notebook-rl"
_HERE = Path(__file__).parent


def _patch_ckpt_output_dir(config_contents: str, out_dir: str) -> str:
    """Fix a real prime-rl v0.7.0 quirk (docs/ANALYSIS_hacking_gap.md /
    TWEAK_trainer_optim_offload.md): OrchestratorConfig.output_dir defaults
    to "outputs/run_default" but TrainerConfig.output_dir defaults to plain
    "outputs" -- two different baked-in defaults for what our TOML treats
    as one shared output_dir. The orchestrator's own resume check (using
    its own output_dir) finds a checkpoint under `run_default/checkpoints`;
    the trainer's own separate resume check (using its own output_dir)
    looks in the plain `checkpoints` dir and finds nothing, silently
    retraining from scratch instead of resuming.

    Fix: explicitly set the shared [ckpt] table's `output_dir` to the same
    `run_default` path the orchestrator already resolves to, so both
    processes' resume-checkpoint lookups agree. Done here (locally, in
    Python, before the config ever reaches Modal) rather than hardcoded
    into the static .toml files, since the correct absolute path depends
    on --output-dir/run_tag and isn't known until launch time.
    """
    config = tomllib.loads(config_contents)
    ckpt = config.setdefault("ckpt", {})
    ckpt["output_dir"] = f"{out_dir}/run_default"
    return tomli_w.dumps(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a config .toml, relative to repo root")
    parser.add_argument("--run-tag", default="", help="Namespaces this run's output dir")
    args = parser.parse_args()

    config_path = Path(args.config)
    abs_path = config_path if config_path.is_absolute() else _HERE.parent / config_path
    config_contents = abs_path.read_text()

    out_name = config_path.stem + (f"-{args.run_tag}" if args.run_tag else "")
    out_dir = f"/outputs/{out_name}"
    config_contents = _patch_ckpt_output_dir(config_contents, out_dir)

    run_training = modal.Function.from_name(APP_NAME, "run_training")
    call = run_training.spawn(
        config_path=args.config,
        config_contents=config_contents,
        run_tag=args.run_tag,
    )
    print(f"Spawned. FunctionCall id: {call.object_id}")
    print("This is now fully decoupled from this local process -- safe to close this terminal.")
    print(f"Monitor via: modal app logs, or modal container exec against run_tag={args.run_tag!r}")


if __name__ == "__main__":
    main()
