from __future__ import annotations

import pytest

from rlm.adapters.logger.console import ConsoleLoggerAdapter
from rlm.adapters.logger.noop import NoopLoggerAdapter
from rlm.domain.models import Iteration, RunMetadata


@pytest.mark.unit
def test_console_logger_disabled_is_noop(capsys: pytest.CaptureFixture[str]) -> None:
    logger = ConsoleLoggerAdapter(enabled=False)
    logger.log_metadata(
        RunMetadata(
            root_model="m",
            max_depth=1,
            max_iterations=1,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=None,
            correlation_id="cid-1",
        )
    )
    logger.log_iteration(
        Iteration(prompt="p", response="r", iteration_time=0.0, correlation_id="cid-1")
    )
    out = capsys.readouterr().out
    assert out == ""


@pytest.mark.unit
def test_console_logger_enabled_prints_metadata_and_iteration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = ConsoleLoggerAdapter(enabled=True)
    logger.log_metadata(
        RunMetadata(
            root_model="m",
            max_depth=1,
            max_iterations=2,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=None,
            correlation_id="cid-1",
        )
    )
    logger.log_iteration(
        Iteration(prompt="p", response="r", iteration_time=0.1, correlation_id="cid-1")
    )
    out = capsys.readouterr().out
    assert "cid=cid-1" in out
    assert "max_iterations=2" in out


@pytest.mark.unit
def test_noop_logger_accepts_calls() -> None:
    logger = NoopLoggerAdapter()
    logger.log_metadata(
        RunMetadata(
            root_model="m",
            max_depth=1,
            max_iterations=1,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=None,
            correlation_id=None,
        )
    )
    logger.log_iteration(Iteration(prompt="p", response="r", iteration_time=0.0))


@pytest.mark.unit
def test_console_logger_validates_enabled_flag_type() -> None:
    with pytest.raises(ValueError, match="enabled"):
        ConsoleLoggerAdapter(enabled="yes")  # type: ignore[arg-type]
