"""
Public API layer (hexagonal entrypoints).

This will expose the stable user-facing surface (Python API, optional CLI).
"""

from __future__ import annotations

from rlm.api.factory import create_rlm, create_rlm_from_config
from rlm.api.rlm import RLM
from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig
from rlm.domain.models import ChatCompletion

__all__ = [
    "RLM",
    "ChatCompletion",
    "EnvironmentConfig",
    "LLMConfig",
    "LoggerConfig",
    "RLMConfig",
    "create_rlm",
    "create_rlm_from_config",
]
