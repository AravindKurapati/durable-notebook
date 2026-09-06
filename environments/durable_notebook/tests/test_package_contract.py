"""The packaging contract verifiers' loader actually depends on.

`verifiers.load_environment(env_id, **args)` imports the module named by the
id and calls `module.load_environment(**args)` on it. If the package does not
expose that name at TOP level, the load fails with

    AttributeError: Module 'durable_notebook' does not expose load_environment.

which is exactly what happened on Modal (2026-08-17): `__init__.py` was empty
and `load_environment` sat unexported in `env.py`. Every local test imported
`durable_notebook.env` directly, so nothing caught it. It would have failed on
a billing GPU container had a free CPU check not run first.
"""

import importlib
import inspect


def test_package_exposes_load_environment_at_top_level():
    module = importlib.import_module("durable_notebook")
    assert hasattr(module, "load_environment"), (
        "verifiers' loader does getattr(module, 'load_environment') on the "
        "package itself; exporting it only from durable_notebook.env is not enough"
    )
    assert callable(module.load_environment)


def test_load_environment_accepts_every_arg_the_training_configs_pass():
    """configs/*.toml forward their `args` table straight to this function.
    A name mismatch there is a paid-GPU failure, so pin the signature."""
    module = importlib.import_module("durable_notebook")
    params = inspect.signature(module.load_environment).parameters
    for name in ("grader", "compaction_window", "n_episodes", "n_turns", "n_questions"):
        assert name in params, f"config passes {name!r} but load_environment has no such arg"


def test_load_environment_returns_a_usable_env():
    module = importlib.import_module("durable_notebook")
    env = module.load_environment(
        grader="naive", compaction_window=3, n_episodes=4, n_turns=10, n_questions=3
    )
    assert len(env.get_dataset()) > 0
    assert "submit_manifest" in env.tool_map


def test_the_toml_configs_only_pass_args_load_environment_accepts():
    """Read the real config files rather than a copy of them, so drift in
    either direction is caught."""
    import tomllib
    from pathlib import Path

    module = importlib.import_module("durable_notebook")
    params = inspect.signature(module.load_environment).parameters
    configs = sorted((Path(__file__).parents[3] / "configs").glob("*.toml"))
    assert configs, "no configs found; path assumption is wrong"
    for path in configs:
        data = tomllib.loads(path.read_text())
        for source in data.get("orchestrator", {}).get("train", {}).get("env", []):
            for key in (source.get("args") or {}):
                assert key in params, f"{path.name} passes {key!r}, unknown to load_environment"
