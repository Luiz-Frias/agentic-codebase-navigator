from __future__ import annotations

import argparse
import json
from typing import Any

from rlm.api import (
    EnvironmentConfig,
    LLMConfig,
    LoggerConfig,
    RLMConfig,
    create_rlm_from_config,
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


def _completion(args: argparse.Namespace) -> int:
    backend_kwargs: dict[str, Any] = {}
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
    if args.json:
        print(json.dumps(cc.to_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(cc.response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        import rlm

        print(rlm.__version__)
        return 0

    if args.command == "completion":
        return _completion(args)

    parser.print_help()
    return 0
