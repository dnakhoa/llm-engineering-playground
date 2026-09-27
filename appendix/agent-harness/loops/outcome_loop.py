"""
Outcome-driven loop: iterate → grade → revise, until a rubric is satisfied.

The distinguishing feature is the *grader*: a separate LLM call with a fresh
context that never sees how the artifact was built. It can only judge the
artifact against explicit, independently checkable criteria — so it cannot be
talked into approving the worker's own reasoning.

    outcome + rubric ──▶ worker ──▶ artifact ──▶ grader ──┬─ satisfied      → done
                            ▲                             ├─ needs_revision → loop
                            └────── gaps only ────────────┘  (gaps fed back)

Run:
    python loops/outcome_loop.py            # uses your configured provider
    python loops/outcome_loop.py --mock     # no API key needed (scripted grader)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

MAX_ITERATIONS_CAP = 20  # hard ceiling regardless of what the caller asks for


# ─────────────────────────────────────────────────────────────────────────────
# Verdict model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gap:
    criterion: str
    why: str


@dataclass
class Verdict:
    satisfied: bool
    gaps: list[Gap] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: str) -> "Verdict":
        """
        Parse a grader response. Graders are LLMs: assume the JSON may be
        wrapped in prose or a code fence, and treat an unparseable verdict as
        'not satisfied' rather than crashing the loop.
        """
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return cls(satisfied=False, gaps=[Gap("grader", "unparseable verdict")])
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return cls(satisfied=False, gaps=[Gap("grader", "invalid JSON verdict")])
        return cls(
            satisfied=bool(data.get("satisfied")),
            gaps=[
                Gap(str(g.get("criterion", "?")), str(g.get("why", "")))
                for g in data.get("gaps", [])
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# The loop
# ─────────────────────────────────────────────────────────────────────────────

WORKER_SYSTEM = (
    "You produce work products. Deliver the artifact only — no preamble, no "
    "explanation of your process. If revision feedback is provided, address "
    "every listed gap."
)

GRADER_SYSTEM = (
    "You are a grader. Score the artifact against each rubric criterion "
    "independently. You did not write the artifact and must not infer intent "
    "that is not visible in it.\n"
    'Reply with JSON only: {"satisfied": bool, '
    '"gaps": [{"criterion": str, "why": str}]}\n'
    "satisfied is true only if EVERY criterion is met."
)


def outcome_loop(
    task: str,
    rubric: str,
    *,
    worker,
    grader,
    max_iterations: int = 5,
    verbose: bool = True,
) -> tuple[str, str, int]:
    """
    Run the iterate → grade → revise loop.

    Args:
        task:     what to produce.
        rubric:   the stop condition, as independently checkable criteria.
        worker:   fn(task, rubric, prior, feedback) -> artifact
        grader:   fn(rubric, artifact) -> Verdict
        max_iterations: iteration cap (mechanical stop condition).

    Returns:
        (artifact, result, iterations_used) where result is one of
        "satisfied" | "max_iterations_reached".
    """
    max_iterations = max(1, min(max_iterations, MAX_ITERATIONS_CAP))
    artifact, feedback = "", ""

    for i in range(1, max_iterations + 1):
        artifact = worker(task, rubric, artifact, feedback)
        verdict = grader(rubric, artifact)

        if verbose:
            status = "satisfied" if verdict.satisfied else f"{len(verdict.gaps)} gap(s)"
            print(f"  [iteration {i}] {status}")
            for gap in verdict.gaps:
                print(f"      ✗ {gap.criterion}: {gap.why}")

        if verdict.satisfied:
            return artifact, "satisfied", i

        # Feed back ONLY the unmet criteria. Replaying the whole rubric teaches
        # the model nothing about what it missed.
        feedback = "\n".join(f"- {g.criterion}: {g.why}" for g in verdict.gaps)

    return artifact, "max_iterations_reached", max_iterations


# ─────────────────────────────────────────────────────────────────────────────
# Live worker / grader (any provider via shared/provider.py)
# ─────────────────────────────────────────────────────────────────────────────

def live_worker(task: str, rubric: str, prior: str, feedback: str) -> str:
    from provider import chat

    if prior and feedback:
        prompt = (
            f"TASK:\n{task}\n\nRUBRIC:\n{rubric}\n\n"
            f"YOUR PREVIOUS ATTEMPT:\n{prior}\n\n"
            f"UNMET CRITERIA — fix each one:\n{feedback}\n\n"
            "Return the corrected artifact in full."
        )
    else:
        prompt = f"TASK:\n{task}\n\nRUBRIC (your output is graded on this):\n{rubric}"

    return chat(prompt, system=WORKER_SYSTEM, max_tokens=1500)


def live_grader(rubric: str, artifact: str) -> Verdict:
    from provider import chat

    raw = chat(
        f"RUBRIC:\n{rubric}\n\nARTIFACT:\n{artifact}",
        system=GRADER_SYSTEM,
        temperature=0.0,   # graders should be as deterministic as possible
        max_tokens=800,
    )
    return Verdict.from_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Mock worker / grader — runs with no API key, for testing the harness itself
# ─────────────────────────────────────────────────────────────────────────────

def _mock_pair():
    """A worker that improves by one criterion per round, and a strict grader."""
    criteria = ["has a title", "has exactly 3 bullets", "ends with a one-line summary"]
    state = {"round": 0}

    def worker(task, rubric, prior, feedback):
        state["round"] += 1
        met = criteria[: state["round"]]
        parts = []
        if "has a title" in met:
            parts.append("# Quarterly Update")
        if "has exactly 3 bullets" in met:
            parts += ["- Revenue up 12%", "- Churn flat", "- Two new regions"]
        if "ends with a one-line summary" in met:
            parts.append("Summary: growth held, retention stable.")
        return "\n".join(parts)

    def grader(rubric, artifact):
        gaps = []
        if not artifact.startswith("#"):
            gaps.append(Gap("has a title", "no markdown heading found"))
        bullets = [ln for ln in artifact.splitlines() if ln.startswith("- ")]
        if len(bullets) != 3:
            gaps.append(Gap("has exactly 3 bullets",
                            f"found {len(bullets)} bullet(s), expected 3"))
        if "Summary:" not in artifact:
            gaps.append(Gap("ends with a one-line summary", "no Summary line"))
        return Verdict(satisfied=not gaps, gaps=gaps)

    return worker, grader


# ─────────────────────────────────────────────────────────────────────────────

DEMO_TASK = "Write a short internal update on Q4 performance for the exec team."

DEMO_RUBRIC = """- Starts with a markdown H1 title
- Contains exactly 3 bullet points
- Ends with a single line beginning "Summary:"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="run with a scripted worker/grader (no API key)")
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()

    if args.mock:
        worker, grader = _mock_pair()
    else:
        worker, grader = live_worker, live_grader

    print("=== Outcome loop ===")
    print(f"Task:   {DEMO_TASK}")
    print(f"Rubric:\n{DEMO_RUBRIC}")

    artifact, result, iterations = outcome_loop(
        DEMO_TASK, DEMO_RUBRIC,
        worker=worker, grader=grader, max_iterations=args.max_iterations,
    )

    print(f"\nResult: {result} after {iterations} iteration(s)")
    print("─" * 50)
    print(artifact)
    print("─" * 50)
    if result != "satisfied":
        print("Loop hit its mechanical stop. Either the rubric is unachievable "
              "or the cap is too low — check the gap list above to tell which.")


if __name__ == "__main__":
    main()
