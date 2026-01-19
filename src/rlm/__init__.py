"""
rlm

This repository is migrating an upstream snapshot in `references/rlm/**` into a
src-layout Python package (`src/rlm/**`) and refactoring toward a hexagonal
modular monolith.
"""

from __future__ import annotations

from rlm._meta import __version__

# Default policy implementations (Phase 2.7) - ready-to-use defaults
from rlm.adapters.policies import (
    DefaultStoppingPolicy,
    NoOpContextCompressor,
    SimpleNestedCallPolicy,
)
from rlm.api import create_rlm, create_rlm_from_config
from rlm.api.rlm import RLM
from rlm.application.config import (
    EnvironmentConfig,
    LLMConfig,
    LoggerConfig,
    RLMConfig,
)

# Extension protocols (Phase 2.7) - for external apps to implement
from rlm.domain.agent_ports import (
    AgentModeName,
    ContextCompressor,
    NestedCallPolicy,
    NestedConfig,
    StoppingPolicy,
)
from rlm.domain.models import ChatCompletion

__all__ = [
    "RLM",
    "AgentModeName",
    "ChatCompletion",
    "ContextCompressor",
    "DefaultStoppingPolicy",
    "EnvironmentConfig",
    "LLMConfig",
    "LoggerConfig",
    "NestedCallPolicy",
    "NestedConfig",
    "NoOpContextCompressor",
    "RLMConfig",
    "SimpleNestedCallPolicy",
    "StoppingPolicy",
    "__version__",
    "create_rlm",
    "create_rlm_from_config",
]
