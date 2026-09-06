"""Validate durable-notebook's rl.toml configs against the REAL prime-rl
pydantic schema. Offline, no GPU, no network, no API key.

This is the cheap pre-flight before any Modal spend: it catches config-shape
bugs for free that would otherwise surface partway into a paid GPU run. It
already earned its keep: it's what caught that the drafted
`[orchestrator.train.source.env.legacy]` block was not a real prime-rl block
at all (extra_forbidden), which in turn forced the version investigation that
established the v0.7.0 pin.

Setup (one-off), in a throwaway venv so it can't disturb the project env.
Pin v0.7.0 -- that is the version these configs target, and its schema differs
from main's (see stage0_smoke.toml's header for why we pin):

    python -m venv /tmp/cfg070_venv
    git -c url."https://github.com/".insteadOf="git@github.com:" \\
        clone --depth 1 -b v0.7.0 \\
        https://github.com/PrimeIntellect-ai/prime-rl.git /tmp/prime-rl-070
    /tmp/cfg070_venv/bin/pip install /tmp/prime-rl-070/packages/prime-rl-configs
    /tmp/cfg070_venv/bin/pip install "verifiers==0.2.1"

Two gotchas, both already hit:
  - Installing prime-rl-configs straight from the git URL fails: v0.7.0's
    submodules use SSH URLs and the build hook runs `git submodule update
    --init --recursive`. Clone with the insteadOf rewrite and install the
    local subdirectory instead.
  - `verifiers` must be 0.2.1, not 0.3.0. v0.7.0's config declares only
    `verifiers>=0.2.0` but imports `vf.EnvServerConfig`, which 0.3.0 removed.

No durable-notebook install is needed: a v0/legacy env (`id` set, no
`taskset`) is not resolved at config-validation time.

Usage:  python configs/validate_configs.py configs/

Exits non-zero if any config fails, so it can gate a run.
"""
import sys
import types
import tomllib
from pathlib import Path

# fcntl is POSIX-only; verifiers.v1.runtimes.modal imports it unconditionally.
# Only flock/LOCK_EX/LOCK_UN are used, none of which matter for config validation.
if "fcntl" not in sys.modules:
    stub = types.ModuleType("fcntl")
    stub.LOCK_EX, stub.LOCK_UN, stub.LOCK_SH, stub.LOCK_NB = 2, 8, 1, 4
    stub.flock = stub.lockf = lambda *a, **k: None
    stub.fcntl = stub.ioctl = lambda *a, **k: 0
    sys.modules["fcntl"] = stub

from pydantic import ValidationError  # noqa: E402
from prime_rl.configs.rl import RLConfig  # noqa: E402

configs = sorted(Path(sys.argv[1]).glob("*.toml"))
if not configs:
    raise SystemExit(f"no .toml found under {sys.argv[1]}")

failures = 0
for path in configs:
    data = tomllib.loads(path.read_text())
    try:
        RLConfig(**data)
        print(f"PASS  {path.name}")
    except ValidationError as e:
        failures += 1
        print(f"FAIL  {path.name}")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            print(f"        {loc}: {err['msg']} [{err['type']}]")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"ERROR {path.name}: {type(e).__name__}: {e}")

print(f"\n{len(configs) - failures}/{len(configs)} configs valid")
sys.exit(1 if failures else 0)
