# Module 13: Agent Harness & Loop Engineering


> **Why this matters:** Autonomous agents crash, loop infinitely, and waste budgets. The harness is what makes agents reliable enough for production — stopping conditions, crash recovery, and cost control.


## 🎯 Learning Objectives
- Understand what an agent harness is and why it's necessary
- Implement loop-until-dry research loops with novelty gates
- Build budget-aware loops that respect token and cost limits
- Create durable journals for crash-proof agent runs with resume capability
- Design self-repair loops with retry logic and rollback
- Add human approval checkpoints for irreversible actions
- Choose between deterministic orchestration and model-driven autonomy

## 📚 What is an Agent Harness?

An **agent harness** is the scaffolding that wraps an LLM agent in a controlled execution loop. It provides:

- **Stopping conditions**: When does the loop end? (goal reached, budget exhausted, N dry rounds)
- **State management**: What persists between iterations?
- **Safety rails**: Max iterations, rollback, human approval
- **Observability**: Logs, traces, cost tracking

Without a harness, an autonomous agent is just a while-loop with no brakes. The harness is what makes agents reliable enough for production.

```
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT HARNESS                             │
│                                                                 │
│  ┌──────────┐   plan    ┌──────────┐  tool call ┌──────────┐   │
│  │ Journal  │◄─────────│   LLM    │────────────▶│  Tools   │   │
│  │ (durable)│          │  Agent   │◄────────────│  (env)   │   │
│  └──────────┘          └──────────┘  observation└──────────┘   │
│       │                     │                                   │
│  ┌────▼────────┐    ┌───────▼────────┐                          │
│  │  Novelty   │    │ Budget Tracker  │                          │
│  │   Gate     │    │ (tokens / cost) │                          │
│  └────────────┘    └────────────────┘                          │
│                                                                 │
│  Stopping conditions: dry_rounds ≥ N  |  budget exhausted      │
│                       max_iterations  |  goal_reached           │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Loop Patterns

### 1. Loop-Until-Dry (Novelty Gate)

Used when you don't know in advance how many items exist. Keep running until K consecutive rounds produce nothing new.

```python
# harness/novelty_gate.py
import hashlib

class NoveltyGate:
    """Tracks seen items and detects when a loop has gone dry."""
    
    def __init__(self, dry_threshold: int = 3):
        self.seen: set[str] = set()
        self.dry_rounds = 0
        self.dry_threshold = dry_threshold
    
    def fingerprint(self, item: any) -> str:
        return hashlib.md5(str(item).encode()).hexdigest()
    
    def filter_new(self, items: list) -> list:
        """Return only items not seen before. Update dry counter."""
        new_items = [
            item for item in items
            if self.fingerprint(item) not in self.seen
        ]
        if new_items:
            for item in new_items:
                self.seen.add(self.fingerprint(item))
            self.dry_rounds = 0
        else:
            self.dry_rounds += 1
        return new_items
    
    @property
    def is_dry(self) -> bool:
        return self.dry_rounds >= self.dry_threshold


# Usage pattern
gate = NoveltyGate(dry_threshold=3)
confirmed = []

while not gate.is_dry:
    findings = agent_search_round()       # returns a list
    fresh = gate.filter_new(findings)     # only new ones
    if fresh:
        verified = verify_findings(fresh) # expensive — only on new items
        confirmed.extend(verified)
    log(f"Round complete: {len(fresh)} new, {gate.dry_rounds} dry rounds")
```

**When to use**: Bug finding, research sweeps, vulnerability audits — any task where you need to be exhaustive but don't know the search space size.

### 2. Budget-Aware Loop

Stop the loop when a token or cost budget is consumed. Inject remaining budget into the agent's context so it self-regulates.

```python
# harness/budget_tracker.py
from dataclasses import dataclass

@dataclass
class BudgetTracker:
    total_tokens: int
    spent_tokens: int = 0
    
    def record(self, input_tokens: int, output_tokens: int):
        self.spent_tokens += input_tokens + output_tokens
    
    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - self.spent_tokens)
    
    @property
    def is_exhausted(self) -> bool:
        return self.remaining < 1000  # keep 1k tokens as safety buffer
    
    @property
    def pct_used(self) -> float:
        return self.spent_tokens / self.total_tokens * 100


# Usage in a loop
budget = BudgetTracker(total_tokens=100_000)

while not budget.is_exhausted:
    # Tell the agent how much budget remains (enables self-regulation)
    system_prompt = (
        f"You have {budget.remaining:,} tokens remaining. "
        f"Be proportionally thorough — deeper analysis for complex items, "
        f"brief notes for simple ones. Stop gracefully if nearly exhausted."
    )
    response = llm_call(system_prompt=system_prompt, ...)
    budget.record(response.usage.input_tokens, response.usage.output_tokens)
    
    if budget.pct_used > 80:
        log("⚠️  80% budget consumed — wrapping up")
```

**When to use**: Research runs, autonomous coding sessions, any long-running task where you must control cost.

### 3. Durable Journal (Crash-Proof Resume)

Write each completed step to an append-only log before proceeding. On restart, replay the journal to skip already-completed work.

```python
# harness/journal.py
import json
import hashlib
from pathlib import Path
from datetime import datetime

class Journal:
    """
    Append-only task journal for durable, resumable agent runs.
    
    Each completed step is written to JSONL before the next step starts.
    On resume, completed steps are replayed from disk — no re-execution.
    """
    
    def __init__(self, run_id: str, journal_dir: str = ".journals"):
        self.run_id = run_id
        self.path = Path(journal_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(exist_ok=True)
        self._completed: dict[str, any] = {}
        self._load()
    
    def _load(self):
        """Replay existing journal on startup."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                entry = json.loads(line)
                self._completed[entry["step_id"]] = entry["result"]
        if self._completed:
            print(f"  [journal] Resumed: {len(self._completed)} steps already done")
    
    def step_id(self, step_name: str, *args) -> str:
        """Deterministic ID for a step + its inputs."""
        payload = f"{step_name}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]
    
    def is_done(self, sid: str) -> bool:
        return sid in self._completed
    
    def get_result(self, sid: str) -> any:
        return self._completed[sid]
    
    def record(self, sid: str, step_name: str, result: any):
        """Write a completed step to the journal."""
        entry = {
            "step_id": sid,
            "step_name": step_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._completed[sid] = result
    
    def execute(self, step_name: str, fn, *args, **kwargs) -> any:
        """
        Run a step only if not already journaled.
        On resume, returns the cached result without calling fn.
        """
        sid = self.step_id(step_name, *args)
        if self.is_done(sid):
            print(f"  [journal] ↩ Skipping '{step_name}' (already done)")
            return self.get_result(sid)
        
        result = fn(*args, **kwargs)
        self.record(sid, step_name, result)
        print(f"  [journal] ✓ Completed '{step_name}'")
        return result


# Usage — crash-safe research workflow
journal = Journal(run_id="research_2026_06_23")

# Each call to journal.execute() is idempotent across crashes
topics = ["context engineering", "agent harness", "MCP"]
all_summaries = []

for topic in topics:
    summary = journal.execute(
        "summarize_topic",       # step name
        agent_summarize,         # function to call (skipped on resume)
        topic                    # args (used for dedup ID)
    )
    all_summaries.append(summary)

final_report = journal.execute("synthesize", agent_synthesize, all_summaries)
```

**When to use**: Any agent run that might be interrupted — long research tasks, overnight batch jobs, expensive multi-step pipelines.

### 4. Self-Repair Loop

When a tool call fails, the agent reformulates and retries. The harness enforces max retries and prevents infinite loops.

```python
# Pattern: self-repair with exponential backoff
import time

MAX_RETRIES = 3

def execute_with_repair(agent, task: str, tools: list) -> str:
    """
    Run the agent. On tool error, let the agent see the error
    and reformulate. Cap retries to avoid infinite loops.
    """
    messages = [{"role": "user", "content": task}]
    
    for attempt in range(MAX_RETRIES):
        response = agent.run(messages=messages, tools=tools)
        
        if response.stop_reason == "end_turn":
            return response.content  # success
        
        if response.stop_reason == "tool_use":
            tool_result = execute_tool(response.tool_calls)
            
            if tool_result.is_error:
                # Inject error back — let agent reformulate
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool '{tool_result.tool}' failed: {tool_result.error}\n"
                        "Please try a different approach or tool."
                    )
                })
                wait = 2 ** attempt  # exponential backoff
                time.sleep(wait)
                continue
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_result.content})
    
    return "MAX_RETRIES_EXCEEDED"
```

### 5. Human Approval Checkpoint

Pause the loop before irreversible actions and wait for explicit human confirmation.

```python
# Pattern: blocking checkpoint for high-stakes actions
def human_checkpoint(action_description: str, preview: str = "") -> bool:
    """
    Block execution until human approves or rejects.
    Returns True for proceed, False for abort.
    """
    print("\n" + "═" * 60)
    print("⚠️  HUMAN APPROVAL REQUIRED")
    print("═" * 60)
    print(f"Proposed action: {action_description}")
    if preview:
        print(f"Preview:\n{preview}")
    print("─" * 60)
    
    while True:
        choice = input("Proceed? [y/n/modify]: ").strip().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            print("Action rejected — agent will try an alternative approach.")
            return False
        elif choice == 'modify':
            modification = input("Enter modification: ")
            # Return modification signal for agent to incorporate
            return modification


# In the agent loop
action = agent.plan_next_action(state)
if action.is_irreversible:
    approved = human_checkpoint(action.description, action.preview)
    if not approved:
        agent.reject_action(action, reason="human rejected")
        continue  # agent re-plans
execute_action(action)
```

---

## 🔀 Orchestration Spectrum

Choosing between deterministic and autonomous control depends on task structure:

```
Deterministic ←──────────────────────────────────→ Autonomous
     │               │               │                    │
  Scripted        LangGraph      Supervisor           Full Loop
  Pipeline        / Workflow      + Workers            Agent
     │               │               │                    │
 Predictable    Branching       Complex tasks        Open-ended
 subtasks        control         undefined            problems
                flow             subtasks
```

| Factor | Go Deterministic | Go Autonomous |
|--------|-----------------|---------------|
| **Subtask structure** | Known upfront | Emerges from results |
| **Latency requirements** | Strict | Flexible |
| **Auditability** | Required | Optional |
| **Error surface** | Minimize | Acceptable |
| **Task complexity** | Bounded | Unbounded |

### Phase Parameter (OpenAI GPT-5.5+)

For long-running or tool-heavy flows, use the `phase` field to distinguish intermediate commentary from final answers:

```python
# GPT-5.5+ Responses API
response = client.responses.create(
    model="gpt-5.6",
    input=[
        {"role": "assistant", "phase": "commentary",
         "content": "I'll inspect the logs and then summarize root cause."},
        {"role": "assistant", "phase": "final_answer",
         "content": "Root cause: cache invalidation race."},
        {"role": "user", "content": "Great — now give me a rollout-safe fix plan."}
    ]
)
```

- `phase: "commentary"` — intermediate updates (preambles before tool calls)
- `phase: "final_answer"` — the completed answer
- Missing or dropped `phase` can cause preambles to be treated as final answers

### Background Mode (OpenAI)

For tasks that take minutes to hours, use background mode — the API returns immediately and you poll for completion:

```python
# Start a long-running task
response = client.responses.create(
    model="gpt-5.6",
    background=True,  # returns immediately
    input="Analyze this 500-page codebase for security vulnerabilities."
)

# Poll for completion
import time
while response.status == "in_progress":
    time.sleep(5)
    response = client.responses.retrieve(response.id)

print(response.output_text)
```

### Multi-Phase Pipelines

Chain phases sequentially with programmatic gates between them:

```python
# Phase 1: Discover
phase("Discover")
items = await agent("Find all relevant files", schema=ITEMS_SCHEMA)

if not items:
    log("Nothing found — stopping early")
    return {}

# Phase 2: Analyze (parallel over discovered items)
phase("Analyze")
analyses = await parallel([
    lambda item=item: agent(f"Analyze: {item}", schema=ANALYSIS_SCHEMA)
    for item in items.files
])

# Phase 3: Synthesize
phase("Synthesize")
report = await agent(
    f"Synthesize {len(analyses)} analyses into a report",
    schema=REPORT_SCHEMA
)
```

### Fan-Out / Fan-In

```
          ┌─── Worker A ──┐
Input ────┼─── Worker B ──┼──── Synthesizer ──→ Output
          └─── Worker C ──┘

Pipeline (no barrier):
  Item 1: A1 → B1 → C1              ← fastest, starts C1 as soon as B1 done
  Item 2: A2 → B2 → C2

Barrier (wait for all):
  A1, A2, A3 all complete → merge → B on merged result
```

Use **pipeline** when each item is independent (most tasks).  
Use **barrier** only when synthesis genuinely needs all prior results (dedup, comparison, voting).

---

## 🛡️ Adversarial Verification

For high-stakes outputs, spawn independent verifiers to challenge each finding:

```python
# Pattern: 3-voter adversarial check
async def adversarial_verify(claim: str, n_voters: int = 3) -> bool:
    """
    Spawn N independent agents to REFUTE the claim.
    Claim survives only if majority cannot refute it.
    """
    votes = await parallel([
        lambda: agent(
            f"Try hard to refute this claim. Default to refuted=True if uncertain. "
            f"Claim: {claim}",
            schema=VERDICT_SCHEMA
        )
        for _ in range(n_voters)
    ])
    
    refuted_count = sum(1 for v in votes if v and v.refuted)
    return refuted_count < n_voters // 2  # majority must FAIL to refute


# Usage: verify each finding before including in report
confirmed = []
for finding in raw_findings:
    if await adversarial_verify(finding.description):
        confirmed.append(finding)
    else:
        log(f"Rejected (refuted): {finding.description}")
```

---

## 🏗️ Project Structure

```
13-agent-harness/
├── README.md
├── requirements.txt
├── harness_example.py           ★ Full research loop demo
├── harness/
│   ├── __init__.py
│   ├── journal.py               Durable task journal (JSONL)
│   └── novelty_gate.py          Loop-until-dry gate
└── loops/
    ├── __init__.py
    ├── research_loop.py         Full autonomous research loop
    └── budget_loop.py           Budget-aware execution loop
```



## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Agent crashes and loses progress | Implement durable journal with JSONL checkpointing |
| Budget exhausted too quickly | Inject remaining budget into agent prompt for self-regulation |
| Novelty gate stops too early | Increase dry_threshold; check fingerprint function |
| Journal replay is slow | Use deterministic step IDs; skip completed steps efficiently |

## 📚 Resources

- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/) — durable agent state
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — agent patterns
- [CrewAI](https://docs.crewai.com/) — multi-agent orchestration

## 🧪 Hands-On Exercises

1. **Loop-Until-Dry**: Build a `NoveltyGate` with threshold=3. Simulate an agent that finds 5 new items in round 1, 3 in round 2, 0 in round 3, 0 in round 4, 0 in round 5 — verify the loop stops after round 5, not round 3 or 4.

2. **Journal Resume Test**: Write a 5-step pipeline using `Journal.execute()`. After step 3, simulate a crash (raise Exception). On restart, verify steps 1-3 are skipped and only steps 4-5 re-run. Measure the token savings.

3. **Budget Governor**: Build a `BudgetTracker` with 50,000 tokens. Run a loop that uses ~5,000 tokens/iteration. Verify it stops before exceeding the budget, AND that the system prompt correctly reflects remaining budget each turn.

4. **Self-Repair**: Design a tool that fails 50% of the time with a retriable error. Wrap it in the self-repair loop (max 3 retries). Run 20 tasks and measure: (a) success rate with vs. without repair, (b) average retries per success.

5. **Adversarial Verify Calibration**: Generate 20 claims — 10 true, 10 false. Run adversarial verification with N=3 voters on each. Measure precision and recall. What's the accuracy at different majority thresholds (1/3 must refute, 2/3, all 3)?

6. **Pipeline vs Barrier**: Process 10 items through a 3-stage pipeline. Measure wall-clock time for: (a) pure sequential, (b) parallel with barrier after each stage, (c) pipeline (no barriers). Plot the speedup.

---

## 📚 References

- Anthropic: "Building effective agents" (2024) — ACI design principles
- OpenAI: "Reasoning models" (2026) — effort levels, pro mode, persisted reasoning
- OpenAI: "Responses API" (2026) — phase parameter, background mode
- Google DeepMind: "Mastering Complex Multi-Step Tasks with LLM Agents" (2025)
- LangGraph: Adaptive RAG and agentic RAG patterns
- Anthropic Claude Code harness engineering (this tool's own architecture)
- "Proof-or-Stop: Loop Engineering for Verifiable Evidence-Gated Lifecycle Control" (2026)

## 🔗 Integration with Other Modules

- **Module 07 (Agents)**: Harness patterns extend basic LangGraph flows
- **Module 12 (Context Engineering)**: Harnesses must budget context per iteration
- **Module 08 (LLM Ops)**: Add tracing to each harness iteration
- **Module 09 (EvalOps)**: Evaluate harness quality end-to-end

---

**The harness is what separates a demo from a production agent. Build it first.**
