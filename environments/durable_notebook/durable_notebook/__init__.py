"""durable-notebook: an RL environment for reward hacking in agentic
filesystem/memory persistence.

`load_environment` MUST be exported here, not just from `.env`. verifiers'
loader resolves an environment id by importing the package and then doing
`getattr(module, "load_environment")` on the package itself, so an empty
`__init__.py` fails with "Module 'durable_notebook' does not expose
load_environment" no matter what `env.py` defines. This is the entrypoint
prime-rl's v0 bridge uses to load the environment for training.
"""

from durable_notebook.env import DurableNotebookEnv, load_environment

__all__ = ["DurableNotebookEnv", "load_environment"]
