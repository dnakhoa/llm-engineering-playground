"""
Autonomous research loop with novelty gate, budget tracking,
durable journaling, and self-repair.
"""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..harness.journal import Journal
from ..harness.novelty_gate import NoveltyGate

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    findings: list[Any]
    total_rounds: int
    stop_reason: str   # "dry", "budget", "max_iter", "goal_reached", "error"
    stats: dict = field(default_factory=dict)


class ResearchLoop:
    """
    Exhaustive research loop with four interlocking stopping conditions:

    1. Loop-until-dry:   stops after `dry_threshold` rounds with nothing new
    2. Budget-aware:     stops when token budget is exhausted
    3. Max iterations:   hard cap to prevent runaway loops
    4. Goal detection:   optional callable that signals task completion

    On each round, the loop:
      a. Calls the finder to get candidate items
      b. Passes candidates through the novelty gate (dedup)
      c. Optionally verifies new items
      d. Journals all results for crash-proof resume
    """

    def __init__(
        self,
        run_id: str,
        finder: Callable[..., list],
        verifier: Optional[Callable[[Any], bool]] = None,
        goal_check: Optional[Callable[[list], bool]] = None,
        dry_threshold: int = 3,
        max_iterations: int = 50,
        token_budget: int = 200_000,
    ):
        self.finder = finder
        self.verifier = verifier
        self.goal_check = goal_check
        self.max_iterations = max_iterations
        self.token_budget = token_budget

        self.gate = NoveltyGate(dry_threshold=dry_threshold)
        self.journal = Journal(run_id)
        self.confirmed: list[Any] = []
        self._tokens_spent = 0

    def record_usage(self, input_tokens: int = 0, output_tokens: int = 0):
        self._tokens_spent += input_tokens + output_tokens

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self._tokens_spent)

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_remaining < 2_000  # 2k safety buffer

    def run(self, *finder_args, **finder_kwargs) -> LoopResult:
        """
        Execute the research loop.

        Args:
            *finder_args: Passed to finder on each round
            **finder_kwargs: Passed to finder on each round

        Returns:
            LoopResult with confirmed findings and stop reason
        """
        stop_reason = "max_iter"
        round_num = 0

        for round_num in range(self.max_iterations):
            logger.info(f"Round {round_num + 1} | "
                       f"seen={len(self.gate.seen)} | "
                       f"budget={self.budget_remaining:,} remaining")

            # ── Budget check ──────────────────────────────────────────────
            if self.budget_exhausted:
                stop_reason = "budget"
                logger.warning("Budget exhausted — stopping loop")
                break

            # ── Find candidates ───────────────────────────────────────────
            step_name = f"find_round_{round_num}"
            try:
                candidates = self.journal.execute(
                    step_name,
                    self.finder,
                    *finder_args,
                    round=round_num,
                    **finder_kwargs,
                )
            except Exception as e:
                logger.error(f"Finder error on round {round_num}: {e}")
                stop_reason = "error"
                break

            # ── Novelty gate ──────────────────────────────────────────────
            fresh = self.gate.filter_new(candidates or [])
            logger.info(f"  {len(candidates)} candidates → {len(fresh)} new")

            # ── Verify (optional) ─────────────────────────────────────────
            if fresh and self.verifier:
                for item in fresh:
                    verify_step = f"verify_{self.gate.fingerprint(item)}"
                    is_real = self.journal.execute(verify_step, self.verifier, item)
                    if is_real:
                        self.confirmed.append(item)
            elif fresh:
                self.confirmed.extend(fresh)

            # ── Goal check ────────────────────────────────────────────────
            if self.goal_check and self.goal_check(self.confirmed):
                stop_reason = "goal_reached"
                break

            # ── Dry check ────────────────────────────────────────────────
            if self.gate.is_dry:
                stop_reason = "dry"
                break

        return LoopResult(
            findings=self.confirmed,
            total_rounds=round_num + 1,
            stop_reason=stop_reason,
            stats={
                "gate": self.gate.stats(),
                "tokens_spent": self._tokens_spent,
                "budget_remaining": self.budget_remaining,
                "journal": self.journal.stats(),
            },
        )
