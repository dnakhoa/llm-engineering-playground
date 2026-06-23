"""
Module 13: Agent Harness & Loop Engineering
Complete demonstration of harness patterns:
  1. Novelty gate / loop-until-dry
  2. Budget-aware loop
  3. Durable journal with crash-proof resume
  4. Self-repair loop
  5. Human approval checkpoint
  6. Multi-phase pipeline
"""
import os
import sys
import time
import random
import logging
from dataclasses import dataclass
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from provider import get_llm_client

client, model = get_llm_client()

# ──────────────────────────────────────────────────────────────────────────────
# 1. NOVELTY GATE — LOOP UNTIL DRY
# ──────────────────────────────────────────────────────────────────────────────

from harness.novelty_gate import NoveltyGate


def demo_novelty_gate():
    """
    Simulates exhaustive bug-finding with loop-until-dry.
    The loop stops only when 3 consecutive rounds find nothing new.
    """
    print("\n=== Demo 1: Loop-Until-Dry with Novelty Gate ===")

    # Simulated knowledge base of bugs (in reality, LLM finds these)
    all_bugs = [
        {"id": "B001", "file": "auth.py",   "description": "SQL injection in login"},
        {"id": "B002", "file": "api.py",    "description": "Missing rate limiting"},
        {"id": "B003", "file": "cache.py",  "description": "Cache key collision"},
        {"id": "B004", "file": "auth.py",   "description": "Weak session token"},
        {"id": "B005", "file": "upload.py", "description": "No file type validation"},
    ]

    def simulated_finder(round: int = 0) -> list[dict]:
        """Simulates an LLM finding bugs — fewer new bugs each round."""
        if round == 0: return all_bugs[:3]
        if round == 1: return all_bugs[2:]  # B003+B004+B005 (B003 is duplicate)
        if round == 2: return all_bugs[4:]  # B005 (duplicate)
        return []  # rounds 3+ find nothing

    gate = NoveltyGate(dry_threshold=3)
    confirmed = []
    round_num = 0

    while not gate.is_dry:
        candidates = simulated_finder(round=round_num)
        fresh = gate.filter_new(candidates)

        print(f"  Round {round_num+1}: {len(candidates)} candidates → {len(fresh)} new | "
              f"dry_rounds={gate.dry_rounds}")

        confirmed.extend(fresh)
        round_num += 1

    print(f"\n  Final: {len(confirmed)} confirmed bugs after {round_num} rounds")
    print(f"  Stop reason: {gate.dry_rounds} consecutive dry rounds")
    for bug in confirmed:
        print(f"    [{bug['id']}] {bug['file']}: {bug['description']}")
    return gate.stats()


# ──────────────────────────────────────────────────────────────────────────────
# 2. DURABLE JOURNAL — CRASH-PROOF RESUME
# ──────────────────────────────────────────────────────────────────────────────

from harness.journal import Journal


def demo_journal():
    """
    5-step pipeline where each step is journaled.
    On second run, already-completed steps are skipped.
    """
    print("\n=== Demo 2: Durable Journal (run twice to see resume) ===")

    run_id = "demo_pipeline_001"
    journal = Journal(run_id, journal_dir=".demo_journals")

    step_calls = {"fetch": 0, "clean": 0, "analyze": 0, "rank": 0, "report": 0}

    def fetch(url: str) -> dict:
        step_calls["fetch"] += 1
        time.sleep(0.1)  # simulate API call
        return {"url": url, "data": [1, 2, 3, 4, 5]}

    def clean(data: dict) -> list:
        step_calls["clean"] += 1
        return [x * 2 for x in data["data"]]

    def analyze(cleaned: list) -> dict:
        step_calls["analyze"] += 1
        return {"mean": sum(cleaned) / len(cleaned), "max": max(cleaned)}

    def rank(analysis: dict) -> list:
        step_calls["rank"] += 1
        return sorted(analysis.items(), key=lambda x: x[1])

    def report(ranked: list) -> str:
        step_calls["report"] += 1
        return f"Report: {', '.join(f'{k}={v}' for k,v in ranked)}"

    # Execute pipeline — journaled
    raw      = journal.execute("fetch",   fetch,   "https://api.example.com/data")
    cleaned  = journal.execute("clean",   clean,   raw)
    analysis = journal.execute("analyze", analyze, cleaned)
    ranked   = journal.execute("rank",    rank,    analysis)
    result   = journal.execute("report",  report,  ranked)

    print(f"\n  Result: {result}")
    print(f"  Steps actually executed: {dict((k,v) for k,v in step_calls.items() if v > 0)}")
    print(f"  (Run again to see resume — only new/changed steps will execute)")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. BUDGET-AWARE LOOP
# ──────────────────────────────────────────────────────────────────────────────

from loops.budget_loop import BudgetLoop, BudgetTracker


def demo_budget_loop():
    """
    Demonstrates a loop that stops when token budget is exhausted.
    Budget hint is injected into each call so the model can self-regulate.
    """
    print("\n=== Demo 3: Budget-Aware Loop ===")

    budget = BudgetTracker(total_tokens=50_000, warn_threshold=0.7)

    items = [
        {"id": i, "complexity": random.choice(["low", "medium", "high"])}
        for i in range(20)
    ]

    results = []
    for item in items:
        if budget.is_exhausted:
            print(f"  Budget exhausted — stopping at item {item['id']}")
            break

        # Inject budget context into LLM call
        system_prompt = (
            f"You are an analyst. {budget.context_hint()}\n"
            f"Analyze item complexity '{item['complexity']}' briefly."
        )

        # Simulate LLM call with variable token usage by complexity
        token_cost = {"low": 800, "medium": 2000, "high": 4000}[item["complexity"]]
        budget.record(input_tokens=token_cost // 2, output_tokens=token_cost // 2)

        results.append({"item": item["id"], "tokens": token_cost})
        print(f"  Item {item['id']:2d} ({item['complexity']:6s}) "
              f"— {token_cost:,} tokens | {budget.report()}")

        if budget.is_warned:
            print(f"  ⚠️  80% budget consumed — wrapping up soon")

    print(f"\n  Processed {len(results)}/{len(items)} items")
    print(f"  Final: {budget.report()}")
    return budget


# ──────────────────────────────────────────────────────────────────────────────
# 4. SELF-REPAIR LOOP
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    content: str
    error: str = ""


def demo_self_repair():
    """
    Demonstrates a tool that fails 50% of the time.
    The self-repair loop retries with error context fed back to agent.
    """
    print("\n=== Demo 4: Self-Repair Loop ===")

    MAX_RETRIES = 3
    success_count = 0
    failure_count = 0

    def flaky_tool(query: str, attempt: int = 0) -> ToolResult:
        """Fails 50% of the time on first attempt, 25% on retry."""
        fail_prob = 0.5 / (attempt + 1)
        if random.random() < fail_prob:
            return ToolResult(success=False, content="", error="Rate limit exceeded")
        return ToolResult(success=True, content=f"Result for: {query}")

    def agent_with_repair(task: str) -> str:
        """Agent that repairs by retrying with error context."""
        context = [{"role": "user", "content": task}]
        last_error = None

        for attempt in range(MAX_RETRIES):
            result = flaky_tool(task, attempt=attempt)

            if result.success:
                return result.content

            # Inject error back — let agent see what went wrong
            last_error = result.error
            context.append({
                "role": "system",
                "content": f"Tool failed (attempt {attempt+1}/{MAX_RETRIES}): "
                           f"{result.error}. Trying again..."
            })
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            time.sleep(0.05 * wait)  # scaled down for demo

        return f"FAILED after {MAX_RETRIES} attempts. Last error: {last_error}"

    tasks = [f"task_{i}" for i in range(10)]
    for task in tasks:
        result = agent_with_repair(task)
        if result.startswith("FAILED"):
            failure_count += 1
            print(f"  ✗ {task}: {result}")
        else:
            success_count += 1
            print(f"  ✓ {task}: {result}")

    print(f"\n  Success rate: {success_count}/{len(tasks)} ({success_count/len(tasks):.0%})")


# ──────────────────────────────────────────────────────────────────────────────
# 5. ADVERSARIAL VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def demo_adversarial_verify():
    """
    Demonstrates 3-voter adversarial verification of findings.
    A claim survives only if the majority cannot refute it.
    """
    print("\n=== Demo 5: Adversarial Verification (3-voter) ===")

    # Simulate findings with known truth values
    claims = [
        {"claim": "Python is the most popular language for ML", "truth": True},
        {"claim": "RAG always outperforms fine-tuning",           "truth": False},
        {"claim": "Context window has U-shaped attention",        "truth": True},
        {"claim": "Prompt caching reduces costs by 90%",          "truth": True},
        {"claim": "Agents should always use maximum context",     "truth": False},
    ]

    def simulated_voter(claim: str, voter_id: int) -> bool:
        """
        Simulates an LLM voter. True = claim is valid.
        In production: call LLM with "try to refute this claim".
        """
        # Use client to call LLM
        response = client.chat.completions.create(
            model=model,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Is this claim accurate? Answer only YES or NO.\n"
                    f"Claim: {claim}"
                )
            }]
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    def adversarial_verify(claim: str, n_voters: int = 3) -> bool:
        """Majority vote across N independent LLM voters."""
        votes = [simulated_voter(claim, i) for i in range(n_voters)]
        affirm_count = sum(votes)
        return affirm_count >= (n_voters // 2 + 1)  # majority must affirm

    confirmed = []
    for item in claims:
        is_real = adversarial_verify(item["claim"])
        correct = is_real == item["truth"]
        status = "✓" if correct else "✗"
        verdict = "CONFIRMED" if is_real else "REJECTED"
        print(f"  {status} [{verdict}] {item['claim'][:55]:<55} "
              f"(truth={item['truth']})")
        if is_real:
            confirmed.append(item["claim"])

    accuracy = sum(
        1 for item in claims if adversarial_verify(item["claim"]) == item["truth"]
    ) / len(claims)
    print(f"\n  Confirmed: {len(confirmed)}/{len(claims)} claims")
    return confirmed


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 13: Agent Harness & Loop Engineering")
    print("=" * 60)

    # Demo 1: Novelty gate
    stats = demo_novelty_gate()
    print(f"\n  Gate stats: {stats}")

    # Demo 2: Journal (run twice to see resume behavior)
    demo_journal()

    # Demo 3: Budget-aware loop
    demo_budget_loop()

    # Demo 4: Self-repair
    demo_self_repair()

    # Demo 5: Adversarial verification
    demo_adversarial_verify()

    print("\n✅ All harness demos complete.")
    print("\nKey patterns:")
    print("  1. Novelty gate  — exhaustive without over-running")
    print("  2. Journal       — idempotent steps, crash-safe")
    print("  3. Budget loop   — model self-regulates verbosity")
    print("  4. Self-repair   — retry with error context fed back")
    print("  5. Adversarial   — majority-vote verification")
