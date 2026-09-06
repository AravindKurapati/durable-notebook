"""Modal app for durable-notebook's prime-rl GRPO training stages.

Run order, cheapest first. Do not skip ahead: each step exists because the
previous one cannot catch its class of failure.

    modal run modal/app.py::build_check          # CPU only, ~free
    modal run modal/app.py::env_load_check       # CPU only, ~free
    modal run modal/app.py --config configs/stage0_smoke.toml   # GPU, costs money

Per this project's budget rules ($27 Modal credit), the GPU entrypoint needs
an explicit go-ahead for that specific stage. The two CPU checks do not.

Recipe notes, mostly inherited from the sibling reward-hacking-env project
which paid for these lessons already:

- CUDA devel base, NOT debian_slim. prime-rl's flash-attn extra is a
  prebuilt cu128/torch2.11 wheel and the stack expects a real CUDA install.
- `uv sync --extra flash-attn`, NOT `--all-extras`. The other extras
  (flash-attn-3, flash-attn-cute) build from git and are slow and fragile.
  flash-attn itself is not optional in practice: ring-flash-attn is a core
  dependency that imports it.
- `uv run --no-sync` at runtime, ALWAYS. Without it uv re-syncs on every
  invocation and uninstalls the flash-attn extra that was just installed.
- Submodules use git@github.com: URLs and the builder has no SSH key, so
  rewrite to https. Init only the four core submodules, never --recursive
  (configs/private is SSH-only and fails).
- prime-rl is 2 GPUs minimum: SingleNodeDeploymentConfig defaults to
  num_train_gpus=1 + num_infer_gpus=1 (verified against the real v0.7.0
  schema). A single-GPU config does not exist. An earlier draft of this
  file said H100:1 and would simply have failed.

prime-rl is PINNED to v0.7.0 deliberately: it trains a classic v0 verifiers
environment through its v0 bridge, which main/v0.8.x removed. The full
verification trail is in configs/stage0_smoke.toml's header.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "durable-notebook-rl"
PRIME_RL_TAG = "v0.7.0"

# A10G, not H100, for the smoke: Qwen3-0.6B + LoRA fits in 24GB with room to
# spare, and A10G is roughly a fifth the price. Revisit only for Stage 1/2.
#
# A100-40GB:2 measured (docs/specs/TWEAK_trainer_optim_offload.md): real
# Modal cost ~$4.27/hr (vs A10G:2's ~$2.6-2.7/hr) and only ~45-50% higher
# throughput at Qwen3-0.6B smoke scale (6200-6500 vs 4100-4400 tok/s) --
# roughly a wash at this scale, though a small model may understate A100's
# real advantage since A10G's ~9min/step at 1.7B scale looks compute-bound
# in a way this tiny smoke test isn't.
#
# TEMP: testing A100-40GB:2 directly on Stage 1 (1.7B) to get the real
# answer at the scale that matters. Revert to "A10G:2" after unless this
# clearly justifies switching.
SMOKE_GPU = "A100-40GB:2"

_HERE = Path(__file__).parent
_ENV_PKG = _HERE.parent / "environments" / "durable_notebook"

_GIT_HTTPS = 'git -c url."https://github.com/".insteadOf="git@github.com:"'

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "curl")
    .pip_install("uv")
    .run_commands(
        f"{_GIT_HTTPS} clone --depth 1 -b {PRIME_RL_TAG} "
        "https://github.com/PrimeIntellect-ai/prime-rl.git /prime-rl",
        # NOT --depth 1 on submodules: a shallow fetch does not reliably land
        # the exact pinned commit, which leaves the workspace out of step with
        # the committed uv.lock and forces uv to re-resolve from scratch.
        f"cd /prime-rl && {_GIT_HTTPS} submodule update --init "
        "deps/verifiers deps/renderers deps/research-environments deps/pydantic-config",
        # --frozen: use the committed uv.lock, do not re-resolve.
        #
        # prime-rl auto-discovers every bundled environment under
        # deps/*/environments/* as a workspace member, and a workspace shares
        # ONE lock. Those members are not mutually resolvable from scratch:
        #   wordle-v1 depends on verifiers[ta]==0.0.1.dev1
        #   aime24-v1 depends on verifiers>=0.2.0
        # so a fresh resolve dies with "No solution found ... your workspace's
        # requirements are unsatisfiable" (seen on the first Modal build,
        # 2026-08-17). The maintainers' lock already encodes a working
        # resolution; --frozen makes uv use it instead of rediscovering the
        # conflict. It also makes the build reproducible, which matters more
        # here than picking up newer patch releases.
        "cd /prime-rl && uv sync --frozen --extra flash-attn",
    )
    .add_local_dir(str(_ENV_PKG), remote_path="/durable-notebook-env", copy=True)
    .run_commands(
        # Installed into prime-rl's own uv environment so the config's
        # `id = "durable-notebook"` resolves by plain import rather than
        # being fetched from the Hub (verifiers' LegacyEnvServer: "a local
        # id is already importable"). --no-deps because everything it needs
        # (verifiers, datasets) is already present, and letting pip resolve
        # them risks pulling a second, conflicting verifiers.
        "cd /prime-rl && uv pip install --no-deps -e /durable-notebook-env",
    )
    .env({"WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)

outputs = modal.Volume.from_name(f"{APP_NAME}-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)


def _sh(cmd: str, cwd: str = "/prime-rl") -> int:
    import subprocess

    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd).returncode


@app.function(timeout=60 * 20)
def build_check() -> None:
    """CPU only, no GPU billed. Confirms the image actually built and the
    two packages that matter are importable. Catches submodule, uv-sync and
    install failures for free, which is most of what goes wrong."""
    assert _sh("uv run --no-sync python -c \"import prime_rl; print('prime_rl OK')\"") == 0
    assert _sh("uv run --no-sync python -c \"import verifiers; print('verifiers', verifiers.__version__)\"") == 0
    assert _sh("uv run --no-sync python -c \"import durable_notebook; print('durable_notebook OK')\"") == 0
    print("build_check PASSED", flush=True)


@app.function(timeout=60 * 20)
def env_load_check() -> None:
    """CPU only, no GPU billed. The check that actually matters before
    paying for a GPU: can prime-rl's v0 bridge load OUR environment by the
    id the rl.toml uses, and does it expose a dataset to train on?

    A failure here is a packaging or entrypoint problem and would otherwise
    only surface after the GPU container has spun up and started billing.
    """
    script = (
        "import verifiers as vf; "
        "env = vf.load_environment('durable-notebook', grader='naive', "
        "compaction_window=3, n_episodes=8, n_turns=10, n_questions=3); "
        "ds = env.get_dataset(); "
        "print('env loaded:', type(env).__name__); "
        "print('dataset rows:', len(ds)); "
        "print('columns:', ds.column_names); "
        "print('tools:', sorted(env.tool_map))"
    )
    assert _sh(f'uv run --no-sync python -c "{script}"') == 0
    print("env_load_check PASSED", flush=True)


@app.function(
    gpu=SMOKE_GPU,
    timeout=60 * 60 * 2,
    volumes={"/outputs": outputs, "/hf-cache": hf_cache},
)
def run_training(config_path: str, config_contents: str, run_tag: str = "") -> None:
    """Spends money. The config's TEXT is passed in, not just its path, so
    the exact TOML reviewed locally is the one that runs -- no chance of a
    stale copy baked into the image executing instead.

    `run_tag` namespaces this run's checkpoints under a subdirectory of the
    shared `outputs` Volume (`/outputs/<config-stem>` or
    `/outputs/<config-stem>-<run_tag>`), instead of writing straight to
    `/outputs`. prime-rl refuses to silently overwrite a prior run's
    checkpoints there (FileExistsError) -- discovered live re-running Stage 0
    after the truncation-budget fix, since the first (pre-fix) smoke attempt
    had already written to that path. A unique subdir per run avoids the
    collision without ever deleting a previous run's checkpoints, which
    matters more once Stage 1/2 crash-recovery is real money on the line."""
    import os

    os.environ["HF_HOME"] = "/hf-cache"

    dest = Path("/prime-rl") / config_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_contents)
    print(f"--- running {config_path} ---\n{config_contents}\n---", flush=True)

    out_name = Path(config_path).stem + (f"-{run_tag}" if run_tag else "")
    out_dir = f"/outputs/{out_name}"
    code = _sh(f"uv run --no-sync rl @ {dest} --output-dir {out_dir}")
    outputs.commit()
    if code != 0:
        raise RuntimeError(f"prime-rl exited with code {code}")
    print("training finished", flush=True)


@app.local_entrypoint()
def main(config: str = "configs/stage0_smoke.toml", run_tag: str = "") -> None:
    path = Path(config)
    if not path.is_absolute():
        path = _HERE.parent / config
    run_training.remote(config_path=config, config_contents=path.read_text(), run_tag=run_tag)
