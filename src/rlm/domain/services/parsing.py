from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.domain.models.iteration import Iteration
    from rlm.domain.models.repl import ReplResult
    from rlm.domain.ports import EnvironmentPort


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks.

    We only execute blocks explicitly tagged as `repl`:

    ```repl
    print("hi")
    ```
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]


# =============================================================================
# Balanced Parenthesis Parser (Shared by sync/async final answer extraction)
# =============================================================================
#
# This module-level function handles the core parsing complexity:
# - Track parenthesis depth
# - Handle single/double quote strings
# - Handle escape sequences within strings
#
# The complexity is inherent to correct lexer behavior and cannot be reduced
# without using a parsing library.


def _find_marker_start(text: str, call_name: str) -> int | None:
    """
    Find the start position for parsing after a marker like `FINAL(` at line start.

    Returns the index after the opening parenthesis, or None if marker not found
    at a line start.
    """
    marker = f"{call_name}("
    marker_pos = text.rfind(marker)
    if marker_pos == -1:
        return None

    # Find the start of the line containing this marker
    line_start = text.rfind("\n", 0, marker_pos) + 1

    # Verify marker is at start of line (with optional whitespace)
    line_prefix = text[line_start:marker_pos]
    if line_prefix and not line_prefix.isspace():
        # Marker not at line start; fall back to forward regex search
        m = re.search(rf"^\s*{re.escape(call_name)}\(", text, re.MULTILINE)
        if not m:
            return None
        return m.end()

    return marker_pos + len(marker)


def _parse_balanced_parens(text: str, start: int) -> int | None:
    r"""
    Parse balanced parentheses starting from `start`, returning end index.

    Handles:
    - Nested parentheses: `FINAL(f(x))`
    - Single-quoted strings: `FINAL('text (with parens)')`
    - Double-quoted strings: `FINAL("text (with parens)")`
    - Escape sequences: `FINAL("escaped \"quote\"")`

    Returns index of closing paren, or None if not found.
    """
    depth = 1
    in_single = False
    in_double = False
    i = start
    text_len = len(text)

    while i < text_len:
        ch = text[i]

        # Handle escape sequences in strings (skip next char)
        if ch == "\\" and (in_single or in_double):
            i += 2
            continue

        # Quote state transitions
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        # Parenthesis handling (only outside quotes)
        elif not (in_single or in_double):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1

    return None


def _extract_call_arg(text: str, call_name: str) -> str | None:
    """
    Extract the argument string from `CALL_NAME(<arg...>)` at line start.

    This is a two-phase parser:
    1. Find marker position (handles line-start requirement)
    2. Parse balanced parentheses (handles nested parens + quotes)
    """
    start = _find_marker_start(text, call_name)
    if start is None:
        return None

    end = _parse_balanced_parens(text, start)
    if end is None:
        return None

    return text[start:end]


def find_final_answer(text: str, *, environment: EnvironmentPort | None = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) at the start of a line.

    - FINAL(answer) returns the answer substring (stripped).
    - FINAL_VAR(name) optionally queries the environment to resolve a variable.

    Performance: Since FINAL() typically appears at the end of LLM responses,
    we search backwards to avoid scanning through potentially large prefixes.
    """
    final_var_arg = _extract_call_arg(text, "FINAL_VAR")
    if final_var_arg is not None:
        variable_name = final_var_arg.strip().strip('"').strip("'")
        if environment is None:
            return None
        result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
        final_answer = result.stdout.strip()
        return final_answer if final_answer else (result.stderr.strip() or "")

    final_arg = _extract_call_arg(text, "FINAL")
    return final_arg.strip() if final_arg is not None else None


async def afind_final_answer(
    text: str,
    *,
    environment: EnvironmentPort | None = None,
) -> str | None:
    """
    Async variant of `find_final_answer`.

    This avoids blocking the event loop when FINAL_VAR needs environment execution.

    Performance: Since FINAL() typically appears at the end of LLM responses,
    we search backwards to avoid scanning through potentially large prefixes.
    """
    final_var_arg = _extract_call_arg(text, "FINAL_VAR")
    if final_var_arg is not None:
        variable_name = final_var_arg.strip().strip('"').strip("'")
        if environment is None:
            return None
        result = await asyncio.to_thread(
            environment.execute_code,
            f"print(FINAL_VAR({variable_name!r}))",
        )
        final_answer = result.stdout.strip()
        return final_answer if final_answer else (result.stderr.strip() or "")

    final_arg = _extract_call_arg(text, "FINAL")
    return final_arg.strip() if final_arg is not None else None


def format_execution_result(result: ReplResult) -> str:
    """Format a `ReplResult` for inclusion in the next prompt."""
    parts: list[str] = []

    if result.stdout:
        parts.append(f"\n{result.stdout}")
    if result.stderr:
        parts.append(f"\n{result.stderr}")

    # Show variable names (not values) for non-internal locals.
    important_vars: dict[str, str] = {}
    for key, value in result.locals.items():
        if key.startswith("_") or key in {"__builtins__", "__name__", "__doc__"}:
            continue
        if isinstance(value, (str, int, float, bool, list, dict, tuple)):
            important_vars[key] = ""

    if important_vars:
        parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(parts) if parts else "No output"


def format_iteration(
    iteration: Iteration,
    *,
    max_character_length: int = 20000,
) -> list[dict[str, str]]:
    """
    Format an iteration to append to the next prompt message history.

    Mirrors the legacy prompt-shaping behavior and truncates long REPL outputs.
    """
    messages: list[dict[str, str]] = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = format_execution_result(code_block.result)

        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        messages.append(
            {
                "role": "user",
                "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
            },
        )

    return messages
