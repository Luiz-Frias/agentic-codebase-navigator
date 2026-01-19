from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal, cast

from rlm.api import (
    EnvironmentConfig,
    LLMConfig,
    LoggerConfig,
    RLMConfig,
    create_rlm_from_config,
)


@dataclass(frozen=True, slots=True)
class CompletionArgs:
    """Typed arguments for the completion subcommand."""

    prompt: str
    backend: str
    model_name: str
    final: str
    environment: Literal["local", "docker", "modal", "prime"]
    max_iterations: int
    max_depth: int
    jsonl_log_dir: str | None
    json_output: bool


EnvironmentType = Literal["local", "docker", "modal", "prime"]


def _extract_completion_args(ns: argparse.Namespace) -> CompletionArgs:
    """
    Extract and validate completion args from Namespace with explicit types.

    Uses cast() for type-safe extraction from argparse.Namespace which
    returns dict[str, Any] via vars(). The casts are safe because argparse
    guarantees the types based on add_argument() configuration.
    """
    raw: dict[str, object] = vars(ns)

    # Extract with defaults, then cast to target types
    prompt = cast("str", raw.get("prompt") or "")
    backend = cast("str", raw.get("backend") or "openai")
    model_name = cast("str", raw.get("model_name") or "gpt-5-nano")
    final = cast("str", raw.get("final") or "ok")
    environment = cast("EnvironmentType", raw.get("environment") or "docker")
    max_iterations = cast("int", raw.get("max_iterations") or 30)
    max_depth = cast("int", raw.get("max_depth") or 1)
    jsonl_log_dir = cast("str | None", raw.get("jsonl_log_dir"))
    json_output = cast("bool", raw.get("json") or False)

    return CompletionArgs(
        prompt=prompt,
        backend=backend,
        model_name=model_name,
        final=final,
        environment=environment,
        max_iterations=max_iterations,
        max_depth=max_depth,
        jsonl_log_dir=jsonl_log_dir,
        json_output=json_output,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlm", description="RLM CLI (Phase 06)")
    p.add_argument("--version", action="store_true", help="Print package version and exit")

    sub = p.add_subparsers(dest="command", required=False)

    completion = sub.add_parser("completion", help="Run a single completion")
    completion.add_argument("prompt", help="Prompt (string)")
    completion.add_argument(
        "--backend",
        default="openai",
        help="LLM backend (default: openai). Other providers remain optional extras.",
    )
    completion.add_argument(
        "--model-name",
        default="gpt-5-nano",
        help="Model name (used for routing + usage; default: gpt-5-nano)",
    )
    completion.add_argument(
        "--final",
        default="ok",
        help="For backend=mock, the FINAL(...) answer to return (default: ok)",
    )
    completion.add_argument(
        "--environment",
        default="docker",
        choices=("local", "docker", "modal", "prime"),
        help="Execution environment (default: docker)",
    )
    completion.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max orchestrator iterations (default: 30)",
    )
    completion.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Max recursion depth (default: 1)",
    )
    completion.add_argument(
        "--jsonl-log-dir",
        default=None,
        help="Enable JSONL logging by writing logs under this directory",
    )
    completion.add_argument(
        "--json",
        action="store_true",
        help="Print full ChatCompletion as JSON (otherwise prints only the response text)",
    )
    return p


def _completion(ns: argparse.Namespace) -> int:
    args = _extract_completion_args(ns)

    backend_kwargs: dict[str, str | list[str]] = {}
    if args.backend == "mock":
        backend_kwargs = {"script": [f"FINAL({args.final})"]}

    logger_cfg = LoggerConfig(logger="none")
    if args.jsonl_log_dir is not None:
        logger_cfg = LoggerConfig(
            logger="jsonl",
            logger_kwargs={"log_dir": args.jsonl_log_dir, "file_name": "rlm"},
        )

    cfg = RLMConfig(
        llm=LLMConfig(
            backend=args.backend,
            model_name=args.model_name,
            backend_kwargs=backend_kwargs,
        ),
        env=EnvironmentConfig(environment=args.environment),
        logger=logger_cfg,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        verbose=False,
    )
    rlm = create_rlm_from_config(cfg)
    cc = rlm.completion(args.prompt)

    if args.json_output:
        print(json.dumps(cc.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(cc.response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    raw: dict[str, object] = vars(ns)

    if cast("bool", raw.get("version") or False):
        # Lazy import to avoid circular dependency at module load time
        import rlm as rlm_pkg

        print(rlm_pkg.__version__)
        return 0

    command = cast("str | None", raw.get("command"))
    if command == "completion":
        return _completion(ns)

    parser.print_help()
    return 0
