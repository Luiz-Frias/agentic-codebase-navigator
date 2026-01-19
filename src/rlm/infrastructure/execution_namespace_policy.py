from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_allowed_import_roots() -> frozenset[str]:
    """
    Default allowlist for `import ...` inside execution environments.

    Notes:
    - This is intentionally conservative (safe-ish, not a security boundary).
    - Environments may override/extend this allowlist via configuration.

    """
    # Keep sorted for deterministic repr/debugging.
    return frozenset(
        [
            "collections",
            "dataclasses",
            "datetime",
            "decimal",
            "functools",
            "itertools",
            "json",
            "math",
            "pathlib",
            "random",
            "re",
            "statistics",
            "string",
            "textwrap",
            "typing",
            "uuid",
        ],
    )


@dataclass(frozen=True, slots=True)
class ExecutionNamespacePolicy:
    """
    Infrastructure policy for building a restricted-ish Python execution namespace.

    Responsibilities:
    - Provide "safe-ish" builtins (block eval/exec/input/etc).
    - Provide a controlled `__import__` allowing only whitelisted modules.
    - Provide a restricted `open()` that only permits file I/O under a session directory.

    Non-goals:
    - This is NOT a hardened sandbox. It is a best-effort safety policy for the
      Local environment adapter.
    """

    allowed_import_roots: frozenset[str] = field(default_factory=_default_allowed_import_roots)

    def build_builtins(self, *, session_dir: Path) -> dict[str, Any]:
        """Build a builtins dict suitable for passing as `globals()['__builtins__']` to `exec`."""
        session_dir = Path(session_dir).resolve()

        orig_import = builtins.__import__

        def _controlled_import(  # type: ignore[no-untyped-def]
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            # Very small validation surface: allow only absolute imports.
            if level != 0:
                raise ImportError("Relative imports are not allowed in this environment")
            if not isinstance(name, str) or not name.strip():
                raise ImportError("Invalid import name")
            root = name.split(".", 1)[0]
            if root not in self.allowed_import_roots:
                raise ImportError(f"Import of {root!r} is not allowed in this environment")
            return orig_import(name, globals, locals, fromlist, level)

        orig_open = builtins.open

        def _restricted_open(  # type: ignore[no-untyped-def]
            file,
            mode="r",
            buffering=-1,
            encoding=None,
            errors=None,
            newline=None,
            closefd=True,
            opener=None,
        ):
            # Disallow opening by fd (harder to reason about path constraints).
            if isinstance(file, int):
                raise PermissionError("open() with file descriptor is not allowed")

            # Path-like support.
            try:
                p = Path(file)
            except TypeError as exc:
                raise PermissionError("open() requires a path-like argument") from exc

            # Resolve relative paths under the current working directory.
            # The Local environment adapter changes cwd to `session_dir` before execution.
            p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()

            try:
                p.relative_to(session_dir)
            except ValueError as exc:
                raise PermissionError(
                    f"open() is restricted to the session directory: {session_dir}",
                ) from exc

            return orig_open(
                p,
                mode,
                buffering,
                encoding,
                errors,
                newline,
                closefd,
                opener,
            )

        # Safe-ish builtins: blocks dangerous operations like eval/exec/input.
        safe: dict[str, Any] = {
            # Core types and functions
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "__build_class__": builtins.__build_class__,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "any": any,
            "all": all,
            "pow": pow,
            "divmod": divmod,
            "chr": chr,
            "ord": ord,
            "hex": hex,
            "bin": bin,
            "oct": oct,
            "repr": repr,
            "ascii": ascii,
            "format": format,
            "hash": hash,
            "id": id,
            "iter": iter,
            "next": next,
            "slice": slice,
            "callable": callable,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "delattr": delattr,
            "dir": dir,
            "vars": vars,
            "bytes": bytes,
            "bytearray": bytearray,
            "memoryview": memoryview,
            "complex": complex,
            "object": object,
            "super": super,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            # Controlled IO + imports
            "__import__": _controlled_import,
            "open": _restricted_open,
            # Exceptions
            "Exception": Exception,
            "BaseException": BaseException,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "FileNotFoundError": FileNotFoundError,
            "OSError": OSError,
            "IOError": IOError,
            "RuntimeError": RuntimeError,
            "NameError": NameError,
            "ImportError": ImportError,
            "StopIteration": StopIteration,
            "AssertionError": AssertionError,
            "NotImplementedError": NotImplementedError,
            "ArithmeticError": ArithmeticError,
            "LookupError": LookupError,
            "Warning": Warning,
            # Blocked / disabled
            "input": None,
            "eval": None,
            "exec": None,
            "compile": None,
            "globals": None,
            "locals": None,
        }
        return safe
