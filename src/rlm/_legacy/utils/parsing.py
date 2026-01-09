"""
Parsing utilities for RLM trjaectories.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rlm._legacy.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm._legacy.environments.base_env import BaseEnv


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def find_final_answer(text: str, environment: BaseEnv | None = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """

    def _extract_call_arg(call_name: str) -> str | None:
        """
        Extract the argument string from a call like `CALL_NAME(<arg...>)` that
        appears at the start of a line.

        We can't use a naive regex like `.*?` because answers may contain
        parentheses (e.g. `FINAL(f(x) == 5)`).
        """
        # Find the opening `CALL_NAME(` anchored to line-start.
        m = re.search(rf"^\s*{re.escape(call_name)}\(", text, re.MULTILINE)
        if not m:
            return None

        start = m.end()
        depth = 1
        in_single = False
        in_double = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            # Ignore escaped characters inside quoted strings.
            if ch == "\\" and (in_single or in_double):
                escape = True
                continue

            # Track basic quote state so parentheses inside quotes don't affect depth.
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

    # Check for FINAL_VAR pattern first - must be at start of line
    final_var_arg = _extract_call_arg("FINAL_VAR")
    if final_var_arg is not None:
        variable_name = final_var_arg.strip().strip('"').strip("'")
        if environment is not None:
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                final_answer = result.stderr.strip() or ""
            return final_answer
        return None

    # Check for FINAL pattern - must be at start of line
    final_arg = _extract_call_arg("FINAL")
    if final_arg is not None:
        return final_arg.strip()

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> str | None:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
