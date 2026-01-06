"""
Legacy execution environments (upstream mirror).
"""

from __future__ import annotations

from typing import Any, Literal

from rlm._legacy.environments.base_env import BaseEnv
from rlm._legacy.environments.local_repl import LocalREPL

__all__ = ["BaseEnv", "LocalREPL", "get_environment"]


def get_environment(
    environment: Literal["local", "modal", "docker", "prime"],
    environment_kwargs: dict[str, Any],
) -> BaseEnv:
    """
    Routes an environment name + kwargs dict to the appropriate environment implementation.

    Supported in the Phase 1 legacy port: local + docker.
    """
    if environment == "local":
        return LocalREPL(**environment_kwargs)
    if environment == "docker":
        from rlm._legacy.environments.docker_repl import DockerREPL

        return DockerREPL(**environment_kwargs)
    if environment in {"modal", "prime"}:
        raise NotImplementedError(
            f"Legacy environment '{environment}' is not ported/enabled in Phase 1."
        )
    raise ValueError(
        f"Unknown environment: {environment}. Supported: ['local', 'docker'] (Phase 1)"
    )
