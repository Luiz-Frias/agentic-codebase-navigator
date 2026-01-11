"""
rlm

This repository is migrating an upstream snapshot in `references/rlm/**` into a
src-layout Python package (`src/rlm/**`) and refactoring toward a hexagonal
modular monolith.
"""

from __future__ import annotations

from rlm._meta import __version__
from rlm.api import create_rlm, create_rlm_from_config
from rlm.api.rlm import RLM
from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig
from rlm.domain.models import ChatCompletion

__all__ = [
    "ChatCompletion",
    "EnvironmentConfig",
    "LLMConfig",
    "LoggerConfig",
    "RLM",
    "RLMConfig",
    "__version__",
    "create_rlm",
    "create_rlm_from_config",
]
