from __future__ import annotations

import asyncio
import re

from rlm.domain.models.iteration import Iteration
from rlm.domain.models.repl import ReplResult
from rlm.domain.ports import EnvironmentPort


def find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks in text wrapped in triple backticks.

    We only execute blocks explicitly tagged as `repl`:

    ```repl
    print("hi")
    ```
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]


def find_final_answer(text: str, *, environment: EnvironmentPort | None = None) -> str | None:
    """Find FINAL(...) or FINAL_VAR(...) at the start of a line.

    - FINAL(answer) returns the answer substring (stripped).
    - FINAL_VAR(name) optionally queries the environment to resolve a variable.

    Performance: Since FINAL() typically appears at the end of LLM responses,
    we search backwards to avoid scanning through potentially large prefixes.
    """

    def _extract_call_arg(call_name: str) -> str | None:
        """Extract the argument string from a call like `CALL_NAME(<arg...>)` that
        appears at the start of a line.

        We can't use a naive regex like `.*?` because answers may contain
        parentheses (e.g. `FINAL(f(x) == 5)`).

        Optimization: Search backwards from end of text since FINAL markers
        typically appear at the end of responses.
        """
        # Fast path: check if marker exists at all
        marker = f"{call_name}("
        marker_pos = text.rfind(marker)
        if marker_pos == -1:
            return None

        # Find the start of the line containing this marker
        line_start = text.rfind("\n", 0, marker_pos) + 1  # +1 to skip the newline itself

        # Verify marker is at start of line (with optional whitespace)
        line_prefix = text[line_start:marker_pos]
        if line_prefix and not line_prefix.isspace():
            # Marker not at line start; search for earlier occurrences
            # Fall back to forward regex search for correctness
            m = re.search(rf"^\s*{re.escape(call_name)}\(", text, re.MULTILINE)
            if not m:
                return None
            start = m.end()
        else:
            # Marker is at line start (with optional whitespace)
            start = marker_pos + len(marker)

        # Parse balanced parentheses
        depth = 1
        in_single = False
        in_double = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue
            if ch == "\\" and (in_single or in_double):
                escape = True
                continue

            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue

            if in_single or in_double:
                continue

            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start:i]
                continue

        return None

    final_var_arg = _extract_call_arg("FINAL_VAR")
    if final_var_arg is not None:
        variable_name = final_var_arg.strip().strip('"').strip("'")
        if environment is None:
            return None
        result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
        final_answer = result.stdout.strip()
        if final_answer == "":
            final_answer = result.stderr.strip() or ""
        return final_answer

    final_arg = _extract_call_arg("FINAL")
    if final_arg is not None:
        return final_arg.strip()
    return None


async def afind_final_answer(
    text: str,
    *,
    environment: EnvironmentPort | None = None,
) -> str | None:
    """Async variant of `find_final_answer`.

    This avoids blocking the event loop when FINAL_VAR needs environment execution.

    Performance: Since FINAL() typically appears at the end of LLM responses,
    we search backwards to avoid scanning through potentially large prefixes.
    """

    def _extract_call_arg(call_name: str) -> str | None:
        # Fast path: check if marker exists at all
        marker = f"{call_name}("
        marker_pos = text.rfind(marker)
        if marker_pos == -1:
            return None

        # Find the start of the line containing this marker
        line_start = text.rfind("\n", 0, marker_pos) + 1  # +1 to skip the newline itself

        # Verify marker is at start of line (with optional whitespace)
        line_prefix = text[line_start:marker_pos]
        if line_prefix and not line_prefix.isspace():
            # Marker not at line start; search for earlier occurrences
            # Fall back to forward regex search for correctness
            m = re.search(rf"^\s*{re.escape(call_name)}\(", text, re.MULTILINE)
            if not m:
                return None
            start = m.end()
        else:
            # Marker is at line start (with optional whitespace)
            start = marker_pos + len(marker)

        # Parse balanced parentheses
        depth = 1
        in_single = False
        in_double = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue
            if ch == "\\" and (in_single or in_double):
                escape = True
                continue

            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue

            if in_single or in_double:
                continue

            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start:i]
                continue

        return None

    final_var_arg = _extract_call_arg("FINAL_VAR")
    if final_var_arg is not None:
        variable_name = final_var_arg.strip().strip('"').strip("'")
        if environment is None:
            return None
        result = await asyncio.to_thread(
            environment.execute_code,
            f"print(FINAL_VAR({variable_name!r}))",
        )
        final_answer = result.stdout.strip()
        if final_answer == "":
            final_answer = result.stderr.strip() or ""
        return final_answer

    final_arg = _extract_call_arg("FINAL")
    if final_arg is not None:
        return final_arg.strip()
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
    iteration: Iteration,
    *,
    max_character_length: int = 20000,
) -> list[dict[str, str]]:
    """Format an iteration to append to the next prompt message history.

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
