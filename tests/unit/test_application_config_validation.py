from __future__ import annotations

import pytest

from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig


@pytest.mark.unit
def test_llm_config_validates_backend_kwargs_is_dict() -> None:
    with pytest.raises(ValueError, match="backend_kwargs must be a dict"):
        LLMConfig(backend="openai", backend_kwargs="nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_environment_config_validates_environment_kwargs_is_dict() -> None:
    with pytest.raises(ValueError, match="environment_kwargs must be a dict"):
        EnvironmentConfig(environment="local", environment_kwargs="nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_logger_config_validates_logger_kwargs_is_dict() -> None:
    with pytest.raises(ValueError, match="logger_kwargs must be a dict"):
        LoggerConfig(logger="none", logger_kwargs="nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_rlm_config_validates_other_llms_shape_and_item_types() -> None:
    with pytest.raises(ValueError, match="other_llms must be a list"):
        RLMConfig(llm=LLMConfig(backend="openai"), other_llms="nope")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must contain only LLMConfig"):
        RLMConfig(llm=LLMConfig(backend="openai"), other_llms=[1])  # type: ignore[list-item]
