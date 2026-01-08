from __future__ import annotations

import asyncio
import re

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


def find_final_answer(text: str, *, environment: EnvironmentPort | None = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) at the start of a line.

    - FINAL(answer) returns the answer substring (stripped).
    - FINAL_VAR(name) optionally queries the environment to resolve a variable.
    """
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is None:
            return None
        result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
        final_answer = result.stdout.strip()
        if final_answer == "":
            final_answer = result.stderr.strip() or ""
        return final_answer

    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def afind_final_answer(
    text: str, *, environment: EnvironmentPort | None = None
) -> str | None:
    """
    Async variant of `find_final_answer`.

    This avoids blocking the event loop when FINAL_VAR needs environment execution.
    """
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is None:
            return None
        result = await asyncio.to_thread(
            environment.execute_code, f"print(FINAL_VAR({variable_name!r}))"
        )
        final_answer = result.stdout.strip()
        if final_answer == "":
            final_answer = result.stderr.strip() or ""
        return final_answer

    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


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
    iteration: Iteration, *, max_character_length: int = 20000
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
            }
        )

    return messages
