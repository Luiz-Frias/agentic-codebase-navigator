from __future__ import annotations

from abc import ABC, abstractmethod

from rlm._legacy.core.types import REPLResult


class BaseEnv(ABC):
    """
    Base REPL-like environment that the RLM uses to interact with. The primary types are isolated and non-isolated,
    where isolated environments are on a separate machine from the LM.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class IsolatedEnv(BaseEnv, ABC):
    """
    These environments (e.g. Prime Envs, Modal Envs) sit on a completely separate machine from the LM,
    guaranteeing complete isolation from the LM process.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class NonIsolatedEnv(BaseEnv, ABC):
    """
    These environments run on the same machine as the LM, and provide different levels of isolation
    depending on the choice of environment. The simplest, default is a local Python REPL that runs
    as a subprocess.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError
