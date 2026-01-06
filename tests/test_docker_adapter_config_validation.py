from __future__ import annotations

import pytest


@pytest.mark.unit
def test_legacy_docker_environment_adapter_defaults_image(monkeypatch: pytest.MonkeyPatch) -> None:
    import rlm.adapters.environments.docker as docker_mod

    class _FakeDockerREPL:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    monkeypatch.setattr(docker_mod, "DockerREPL", _FakeDockerREPL)

    adapter = docker_mod.LegacyDockerEnvironmentAdapter()
    assert adapter._env.kwargs["image"] == "python:3.12-slim"


@pytest.mark.unit
def test_legacy_docker_environment_adapter_passes_through_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.adapters.environments.docker as docker_mod

    class _FakeDockerREPL:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    monkeypatch.setattr(docker_mod, "DockerREPL", _FakeDockerREPL)

    adapter = docker_mod.LegacyDockerEnvironmentAdapter(
        image="python:3.12-alpine",
        subprocess_timeout_s=12.34,
        lm_handler_address=("127.0.0.1", 12345),
    )
    assert adapter._env.kwargs["image"] == "python:3.12-alpine"
    assert adapter._env.kwargs["subprocess_timeout_s"] == 12.34
    assert adapter._env.kwargs["lm_handler_address"] == ("127.0.0.1", 12345)
