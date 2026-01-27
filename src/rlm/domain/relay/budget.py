from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TokenBudget:
    max_tokens: int
    per_state_estimates: dict[str, int] = field(default_factory=dict)
    consumed: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("TokenBudget.max_tokens must be >= 0")
        if self.consumed < 0:
            raise ValueError("TokenBudget.consumed must be >= 0")

    @property
    def remaining(self) -> int:
        return max(self.max_tokens - self.consumed, 0)

    def estimate_for(self, state_name: str) -> int:
        return self.per_state_estimates.get(state_name, 0)

    def can_consume(self, amount: int) -> bool:
        return amount <= self.remaining

    def with_consumed(self, amount: int) -> TokenBudget:
        if amount < 0:
            raise ValueError("TokenBudget.with_consumed requires amount >= 0")
        return TokenBudget(
            max_tokens=self.max_tokens,
            per_state_estimates=self.per_state_estimates,
            consumed=self.consumed + amount,
        )
