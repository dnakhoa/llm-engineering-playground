"""
Budget-aware execution loop.

Injects remaining budget into each agent call, enabling the model
to self-regulate — spending more tokens on hard items, fewer on easy ones.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BudgetTracker:
    """Tracks token / cost budget across an agent session."""
    total_tokens: int
    warn_threshold: float = 0.8
    stop_threshold: float = 0.97   # keep 3% as safety buffer
    spent_tokens: int = 0
    spent_cost_usd: float = 0.0

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        self.spent_tokens += input_tokens + output_tokens
        self.spent_cost_usd += cost_usd

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - self.spent_tokens)

    @property
    def pct_used(self) -> float:
        return self.spent_tokens / self.total_tokens

    @property
    def is_warned(self) -> bool:
        return self.pct_used >= self.warn_threshold

    @property
    def is_exhausted(self) -> bool:
        return self.pct_used >= self.stop_threshold

    def context_hint(self) -> str:
        """
        Inject this into the system prompt each turn.
        Lets the model self-regulate verbosity as budget shrinks.
        """
        pct_remaining = 1.0 - self.pct_used
        if pct_remaining > 0.5:
            guidance = "Budget ample — be thorough."
        elif pct_remaining > 0.2:
            guidance = "Budget moderate — prioritize key findings."
        else:
            guidance = "Budget nearly exhausted — wrap up concisely."

        return (
            f"[Budget: {self.remaining:,} tokens remaining "
            f"({pct_remaining:.0%}). {guidance}]"
        )

    def report(self) -> str:
        bar_len = 30
        filled = int(bar_len * self.pct_used)
        bar = "█" * filled + "░" * (bar_len - filled)
        return (
            f"Budget [{bar}] "
            f"{self.spent_tokens:,}/{self.total_tokens:,} tokens "
            f"({self.pct_used:.1%} used)"
        )


class BudgetLoop:
    """
    Runs a callable in a loop, stopping when the token budget is exhausted.

    On each iteration:
    1. Injects remaining budget hint into kwargs
    2. Calls the worker function
    3. Records token usage from the result
    4. Checks stopping conditions

    The worker function must return an object with:
    - .done: bool — signals task completion
    - .input_tokens: int — tokens consumed in this call
    - .output_tokens: int — tokens generated in this call
    - .result: Any — work product of this iteration
    """

    def __init__(
        self,
        total_tokens: int,
        max_iterations: int = 100,
        warn_threshold: float = 0.8,
    ):
        self.budget = BudgetTracker(
            total_tokens=total_tokens,
            warn_threshold=warn_threshold,
        )
        self.max_iterations = max_iterations
        self.results: list[Any] = []

    def run(self, worker: Callable, *args, **kwargs) -> dict:
        """
        Execute worker in a loop until done, budget exhausted, or max_iter reached.

        Each call to worker receives:
          budget_hint=<str>  — budget context to include in system prompt
          iteration=<int>    — current iteration number
        """
        stop_reason = "max_iter"

        for i in range(self.max_iterations):
            if self.budget.is_exhausted:
                stop_reason = "budget_exhausted"
                break

            if self.budget.is_warned and i > 0:
                logger.warning(
                    f"⚠️  {self.budget.pct_used:.0%} budget consumed "
                    f"({self.budget.remaining:,} tokens remaining)"
                )

            # Inject budget context into every call
            step_kwargs = {
                **kwargs,
                "budget_hint": self.budget.context_hint(),
                "iteration": i,
            }

            try:
                outcome = worker(*args, **step_kwargs)
            except Exception as e:
                logger.error(f"Worker error on iteration {i}: {e}")
                stop_reason = "error"
                break

            # Record usage
            self.budget.record(
                input_tokens=getattr(outcome, "input_tokens", 0),
                output_tokens=getattr(outcome, "output_tokens", 0),
                cost_usd=getattr(outcome, "cost_usd", 0.0),
            )

            if hasattr(outcome, "result"):
                self.results.append(outcome.result)

            logger.info(f"  Iter {i+1}: {self.budget.report()}")

            if getattr(outcome, "done", False):
                stop_reason = "done"
                break

        return {
            "stop_reason": stop_reason,
            "iterations": i + 1,
            "results": self.results,
            "budget": {
                "total": self.budget.total_tokens,
                "spent": self.budget.spent_tokens,
                "remaining": self.budget.remaining,
                "pct_used": round(self.budget.pct_used, 3),
            },
        }
