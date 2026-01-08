from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.llm_request import LLMRequest
from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.ports import EnvironmentPort, LLMPort, LoggerPort
from rlm.domain.services.parsing import (
    afind_final_answer,
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.domain.services.prompts import (
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.domain.types import Prompt


@dataclass(slots=True, frozen=True)
class RLMOrchestrator:
    """
    Pure domain orchestrator (Phase 2).

    This implements the legacy iteration loop semantics using only domain ports.
    Environment/broker lifecycle is handled outside (composition root).
    """

    llm: LLMPort
    environment: EnvironmentPort
    logger: LoggerPort | None = None
    system_prompt: str = RLM_SYSTEM_PROMPT

    def completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_depth: int = 1,
        depth: int = 0,
        max_iterations: int = 30,
    ) -> ChatCompletion:
        time_start = time.perf_counter()

        # Fallback: if we're at max depth, treat as a plain LM call.
        if depth >= max_depth:
            cc = self.llm.complete(LLMRequest(prompt=prompt))
            final_answer = find_final_answer(cc.response)
            response = final_answer if final_answer is not None else cc.response
            time_end = time.perf_counter()
            return ChatCompletion(
                root_model=cc.root_model,
                prompt=prompt,
                response=response,
                usage_summary=self.llm.get_usage_summary(),
                execution_time=time_end - time_start,
            )

        # Load the prompt as "context" into the environment (legacy-compatible semantics).
        self.environment.load_context(prompt)  # type: ignore[arg-type]

        # Build initial message history (system + metadata hint).
        query_metadata = QueryMetadata.from_context(prompt)
        message_history: list[dict[str, str]] = build_rlm_system_prompt(
            self.system_prompt, query_metadata
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()

            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i)
            ]

            llm_cc = self.llm.complete(LLMRequest(prompt=current_prompt))
            response = llm_cc.response

            code_block_strs = find_code_blocks(response)
            code_blocks: list[CodeBlock] = []
            for code in code_block_strs:
                repl_result = self.environment.execute_code(code)
                code_blocks.append(CodeBlock(code=code, result=repl_result))

            final_answer = find_final_answer(response, environment=self.environment)
            iteration_time = time.perf_counter() - iter_start

            iteration = Iteration(
                prompt=current_prompt,
                response=response,
                code_blocks=code_blocks,
                final_answer=final_answer,
                iteration_time=iteration_time,
            )

            if self.logger is not None:
                self.logger.log_iteration(iteration)

            if final_answer is not None:
                time_end = time.perf_counter()
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response=final_answer,
                    usage_summary=self.llm.get_usage_summary(),
                    execution_time=time_end - time_start,
                )

            # Carry state into the next iteration.
            message_history.extend(format_iteration(iteration))

        # Out of iterations: ask one final time for an answer.
        final_prompt: Prompt = message_history + [
            {
                "role": "user",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        last_cc = self.llm.complete(LLMRequest(prompt=final_prompt))
        extracted = find_final_answer(last_cc.response)
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=extracted if extracted is not None else last_cc.response,
            usage_summary=self.llm.get_usage_summary(),
            execution_time=time_end - time_start,
        )

    async def acompletion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_depth: int = 1,
        depth: int = 0,
        max_iterations: int = 30,
    ) -> ChatCompletion:
        """
        Async variant of `completion()`.

        Notes:
        - We still execute code blocks sequentially to preserve environment semantics.
        - We use `asyncio.TaskGroup` + `asyncio.to_thread` to avoid blocking the event loop
          while loading context / executing code.
        """
        time_start = time.perf_counter()

        if depth >= max_depth:
            cc = await self.llm.acomplete(LLMRequest(prompt=prompt))
            final_answer = await afind_final_answer(cc.response)
            response = final_answer if final_answer is not None else cc.response
            time_end = time.perf_counter()
            return ChatCompletion(
                root_model=cc.root_model,
                prompt=prompt,
                response=response,
                usage_summary=self.llm.get_usage_summary(),
                execution_time=time_end - time_start,
            )

        query_metadata = QueryMetadata.from_context(prompt)
        message_history: list[dict[str, str]] = build_rlm_system_prompt(
            self.system_prompt, query_metadata
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()
            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i)
            ]

            # On the first iteration, load context and run the LLM call concurrently.
            if i == 0:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        asyncio.to_thread(self.environment.load_context, prompt)  # type: ignore[arg-type]
                    )
                    llm_task = tg.create_task(self.llm.acomplete(LLMRequest(prompt=current_prompt)))
                llm_cc = llm_task.result()
            else:
                llm_cc = await self.llm.acomplete(LLMRequest(prompt=current_prompt))

            response = llm_cc.response
            code_block_strs = find_code_blocks(response)
            code_blocks: list[CodeBlock] = []

            for code in code_block_strs:
                repl_result = await asyncio.to_thread(self.environment.execute_code, code)
                code_blocks.append(CodeBlock(code=code, result=repl_result))

            final_answer = await afind_final_answer(response, environment=self.environment)
            iteration_time = time.perf_counter() - iter_start

            iteration = Iteration(
                prompt=current_prompt,
                response=response,
                code_blocks=code_blocks,
                final_answer=final_answer,
                iteration_time=iteration_time,
            )
            if self.logger is not None:
                self.logger.log_iteration(iteration)

            if final_answer is not None:
                time_end = time.perf_counter()
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response=final_answer,
                    usage_summary=self.llm.get_usage_summary(),
                    execution_time=time_end - time_start,
                )

            message_history.extend(format_iteration(iteration))

        final_prompt: Prompt = message_history + [
            {
                "role": "user",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        last_cc = await self.llm.acomplete(LLMRequest(prompt=final_prompt))
        extracted = await afind_final_answer(last_cc.response)
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=extracted if extracted is not None else last_cc.response,
            usage_summary=self.llm.get_usage_summary(),
            execution_time=time_end - time_start,
        )
