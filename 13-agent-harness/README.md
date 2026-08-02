# Module 13: Agent Harness & Loop Engineering
> **Why this matters:** Autonomous agents crash, loop infinitely, and waste budgets. The harness is what makes agents reliable enough for production — stopping conditions, crash recovery, and cost control. **Loop engineering** is the discipline of designing that iteration: what advances it, what ends it, what survives a crash, and what it's allowed to spend.


## Learning Objectives
- Understand what an agent harness is and why it's necessary
- Answer the five questions every production loop must answer
- Implement loop-until-dry research loops with novelty gates
- Build budget-aware loops — both hand-rolled and API-native (`task_budget`)
- Create durable journals for crash-proof agent runs with resume capability
- Design self-repair loops with retry logic and rollback
- Run outcome-driven loops that iterate against a graded rubric until "done"
- Manage the context lifecycle *inside* a loop (compaction, context editing)
- Decide who owns the loop: your code, the SDK, or a hosted agent runtime
- Add human approval checkpoints for irreversible actions
- Recognize and fix the standard loop failure modes
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

## 🧭 Loop Engineering: The Five Questions

A "loop" in production is not `while True`. Before you write one, answer these five questions explicitly. If you can't answer one of them, that's your next bug.

| # | Question | Mechanism | Failure if unanswered |
|---|----------|-----------|----------------------|
| 1 | **What advances it?** | New information per iteration — a tool result, a fresh finding, a grader verdict | Iterations repeat identical work (thrash) |
| 2 | **What ends it?** | Goal predicate, novelty gate, budget, max iterations, grader says *satisfied* | Infinite loop, or premature exit on the first plausible answer |
| 3 | **What survives a crash?** | Durable journal, checkpointer, or hosted session state | 40 minutes of work lost to one timeout |
| 4 | **What may it spend?** | Token/cost budget, `task_budget`, `effort`, iteration cap | A $200 overnight run that produced nothing |
| 5 | **When does it escalate?** | Human approval checkpoint, permission policy, deny-with-reason | Agent does something irreversible at 3am |

**Design rule:** every loop needs *at least two* independent stop conditions — one semantic (goal reached / gone dry) and one mechanical (budget or iteration cap). The semantic one is what you want; the mechanical one is what saves you when the semantic one is wrong.

### Loop taxonomy

Most agent loops are one of six shapes. Naming the shape tells you which controls you need.

| Shape | Advances on | Stops on | Use for |
|-------|-------------|----------|---------|
| **Tool loop** (ReAct) | Tool results | `stop_reason == "end_turn"` | Anything with tools — the base case |
| **Loop-until-dry** | New unseen items | K consecutive dry rounds | Exhaustive discovery (bugs, sources, entities) |
| **Budget loop** | Any work | Budget exhausted | Cost-capped research and coding runs |
| **Self-repair loop** | Error fed back to the model | Success or max retries | Flaky tools, compiling/testing, schema fixes |
| **Outcome loop** | Grader feedback | Rubric satisfied / max iterations | Deliverables with checkable "done" criteria |
| **Scheduled loop** | Wall-clock / event | Never (each firing is bounded) | Monitors, nightly triage, recurring reports |

They compose: a nightly (scheduled) outcome loop whose worker is a budget-capped tool loop with self-repair on test failures is an ordinary production design.

---

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

## 💰 API-Native Loop Controls

Patterns 1–5 are all *client-side*: your code counts, gates, and stops. Newer APIs push two of those controls into the model itself, which changes how you write the loop.

### `effort` — the per-call depth dial

Instead of a thinking-token budget, current models take an **effort** level that controls how much they think *and* act:

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=64000,
    thinking={"type": "adaptive"},              # model decides when to think
    output_config={"effort": "high"},           # low | medium | high | xhigh | max
    tools=tools,
    messages=messages,
)
```

Loop-relevant behaviour:

| Effort | Loop effect | Use in a harness |
|--------|-------------|------------------|
| `low` | Fewer, more consolidated tool calls; scopes to exactly what was asked | Subagents, classification steps, cheap verification passes |
| `medium` | Balanced | High-volume worker nodes |
| `high` | Default; more tool use, more thorough | Main agent loop |
| `xhigh` | Best for coding and agentic work | Long-horizon coding runs |
| `max` | Deepest reasoning, can overthink simple tasks | Rare, correctness-critical steps |

> ⚠️ **`effort` is not a verbosity dial.** Lowering it to shorten user-facing output is unreliable — instruct conciseness in the prompt instead. And at `xhigh`/`max`, raise `max_tokens` (start at 64K): thinking counts against the same ceiling as the answer, so a tight `max_tokens` truncates mid-thought.

### `task_budget` — a budget the *model* can see

Pattern 2 injected the remaining budget into the system prompt. That works, but it invalidates your prompt cache every turn and the number is only as accurate as your own accounting. The API-native version hands the model a server-tracked countdown for a whole agentic loop:

```python
with client.beta.messages.stream(
    model="claude-opus-5",
    max_tokens=128000,
    betas=["task-budgets-2026-03-13"],
    output_config={
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 64000},   # minimum 20,000
    },
    tools=tools,
    messages=messages,
) as stream:
    response = stream.get_final_message()
```

- **`task_budget` ≠ `max_tokens`.** `max_tokens` is an enforced per-response ceiling the model is *unaware* of — hitting it truncates. `task_budget` is a target the model *sees* and paces itself against, so it wraps up gracefully instead of being cut off mid-task.
- The countdown covers what the model generates plus the tool results it reads **this turn** — not the full history you resend each request. Leave `remaining` unset in a normal loop; the server tracks it.
- Stream it. A 128K `max_tokens` non-streaming request will hit HTTP timeouts.
- Keep your own `BudgetTracker` too, accumulating `usage.output_tokens` — you still need a hard mechanical stop and a number to show the user.

### Choosing between them

```
Need the model to pace itself over a long loop?    → task_budget
Need a hard ceiling you enforce?                   → max_tokens + BudgetTracker
Need to trade quality for cost per call?           → effort
Need cumulative spend across many runs?            → your own accounting (Module 08)
```

---

## 🎯 Outcome-Driven Loops

The patterns above stop on *mechanical* conditions. An **outcome loop** stops on a *quality* condition: you declare what "done" looks like as a gradeable rubric, and a separate grader — with its own context window — scores each iteration and feeds per-criterion gaps back to the agent.

```
                ┌──────────────────────────────────────────┐
                │                                          │
  outcome ──▶ agent works ──▶ artifact ──▶ grader ──┬── needs_revision (loop, with gaps)
  + rubric                                          ├── satisfied           → done
                                                    ├── max_iterations_reached
                                                    └── failed (rubric ≠ task)
```

Why it's different from self-critique: the grader has **no memory of how the artifact was built**, so it can't be talked into approving its own reasoning. Fresh-context verification consistently beats self-review.

```python
# Anthropic Managed Agents — the harness runs the iterate → grade → revise loop
session = client.beta.sessions.create(
    agent=AGENT_ID,                     # created once, versioned
    environment_id=ENVIRONMENT_ID,
    title="Q4 revenue model",
)

client.beta.sessions.events.send(
    session_id=session.id,
    events=[{
        "type": "user.define_outcome",
        "description": "Build a DCF model for Costco as an .xlsx file",
        "rubric": {"type": "text", "content": RUBRIC_MD},   # required
        "max_iterations": 5,                                # default 3, max 20
    }],
)
```

Watch `span.outcome_evaluation_end` on the event stream. Its `result` drives the loop:

| `result` | Meaning | Next |
|----------|---------|------|
| `satisfied` | Every criterion met | Terminal — collect the artifact |
| `needs_revision` | Gaps reported back to the agent | Another iteration runs |
| `max_iterations_reached` | Cap hit | One final revision may run, then idle |
| `failed` | Rubric doesn't match the task at all | Terminal — fix your rubric |
| `interrupted` | You sent `user.interrupt` | Terminal |

### Writing a rubric the grader can actually use

The rubric *is* the loop's stop condition. Vague rubrics produce noisy loops that burn iterations.

```markdown
❌ "The report should look good and be accurate."

✅ - Output is a single .xlsx file with sheets: Assumptions, Model, Summary
   - Revenue projection covers exactly 5 forward years
   - Every projected figure traces to a cell in Assumptions (no hardcoded numbers)
   - WACC is stated as a number between 0 and 1 in Assumptions!B4
   - Summary contains an enterprise value and an implied share price
```

Each line is independently checkable, so a failure names a specific gap the next iteration can close. Rule of thumb: if two competent reviewers could disagree on whether a criterion is met, the grader will too.

### Rolling your own outcome loop

You don't need a hosted runtime — the pattern is portable:

```python
def outcome_loop(task: str, rubric: str, max_iterations: int = 5):
    """Iterate → grade with a FRESH context → revise. Stop when satisfied."""
    artifact, feedback = None, ""

    for i in range(max_iterations):
        artifact = worker_agent(task=task, prior=artifact, feedback=feedback)

        verdict = grader_agent(          # separate call, no worker history
            rubric=rubric,
            artifact=artifact,
            schema=VERDICT_SCHEMA,       # {satisfied: bool, gaps: [{criterion, why}]}
        )
        log(f"iteration {i}: satisfied={verdict.satisfied} gaps={len(verdict.gaps)}")

        if verdict.satisfied:
            return artifact, "satisfied"

        # Feed back ONLY the unmet criteria — not the whole rubric again
        feedback = "\n".join(f"- {g.criterion}: {g.why}" for g in verdict.gaps)

    return artifact, "max_iterations_reached"
```

Two things make or break it: the grader must not see the worker's reasoning, and the feedback must be the *gaps only*. Replaying the full rubric each round teaches the model nothing about what it missed.

---

## 🧹 Context Lifecycle Inside a Loop

A loop that runs for 60 iterations accumulates 60 iterations of tool output. Left alone, the context window fills with stale observations, the model's attention degrades (Module 12), and eventually the request fails outright. A harness needs an explicit policy for what leaves the context.

| Mechanism | What it does | Loop-side cost |
|-----------|--------------|----------------|
| **Observation masking** (Module 12) | Compresses tool output *before* it enters context | Cheapest — do this first |
| **Context editing** | Server-side **clears** old tool results / thinking blocks | Cleared content is gone; keep it in your journal if you need it |
| **Compaction** | Server-side **summarizes** earlier history into a compaction block | Lossy; summary quality varies |
| **Journal + fresh context** | Restart the loop with a curated state summary you control | Most control, most code |

```python
# Context editing — prune stale tool results as the loop runs
response = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    betas=["context-management-2025-06-27"],
    context_management={"edits": [
        {"type": "clear_tool_uses_20250919"},   # drop old tool results
        {"type": "clear_thinking_20251015"},    # drop old thinking blocks
    ]},
    tools=tools,
    messages=messages,
)

# Compaction — summarize earlier turns instead of dropping them
response = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112"}]},
    messages=messages,
)
messages.append({"role": "assistant", "content": response.content})  # ← full content!
```

> ⚠️ **The one-line bug that breaks compaction:** append `response.content`, not just the text. The compaction block lives in `content`, and the API uses it to replace the compacted history on the next request. Extract only `.text` and you silently lose the compaction state — the conversation regrows and the next call is billed at full price.

**Pairing rule:** context editing and compaction manage the *model's* view; the journal manages *your* view. Facts the loop must not forget (confirmed findings, decisions, file paths) belong in the journal or a memory file — never only in the conversation history, because the history is designed to be pruned.

---

## 🧑‍✈️ Who Owns the Loop?

Four options, and the choice is mostly about how much of the harness you want to maintain.

| Approach | You write | Runs where | Reach for it when |
|----------|-----------|-----------|-------------------|
| **Manual loop** | The whole `while stop_reason == "tool_use"` cycle | Your process | You need control the helpers don't expose, or no beta dependency |
| **SDK tool runner** (`client.beta.messages.tool_runner`) | Just the tool functions | Your process | Most custom-tool agents — approval gates and retries are per-turn hooks |
| **Agent SDK / framework** | A prompt + options | Your process | You want a batteries-included coding/filesystem agent |
| **Hosted agent runtime** (Managed Agents) | Agent config + tool results | Provider-hosted session + sandbox | Long-running sessions, persisted versioned configs, scheduled runs |

Two rules that catch most people out:

1. **"I need fine-grained control" is rarely a reason to hand-write the loop.** The tool runner yields the assistant message *before* tools execute, so approval gating, error interception, result rewriting (e.g. adding `cache_control`), and per-turn retries all work without owning the loop.
2. **A harness-only helper is not a deployment.** The tool runner and agent frameworks still run on your infrastructure; only a hosted runtime takes over process lifetime, sandboxing, and scheduling.

### Durable execution: journal vs. engine

Pattern 3's journal is the do-it-yourself version of **durable execution**. The same guarantee — completed steps are never re-run — is offered by dedicated engines:

| Option | Persistence unit | Good fit |
|--------|-----------------|----------|
| Hand-rolled JSONL journal (pattern 3) | Your step IDs | Scripts, batch jobs, learning the mechanics |
| Graph checkpointer (Module 16) | Every node transition + state snapshot | Branching workflows, time-travel debugging, human interrupts |
| Workflow engine (Temporal-style) | Every activity, with retries and timers | Multi-day processes, strict SLAs, existing workflow infra |
| Hosted session state | The session itself | You don't want to run the process at all |

Pick the cheapest one that survives your worst realistic failure. A 20-minute research script needs a journal, not a workflow cluster.

### The idle-break gate (hosted loops)

When the provider runs the loop, your code becomes a stream consumer — and the single most common bug is breaking on the first idle event:

```python
for event in stream:
    handle(event)
    if event.type == "session.status_terminated":
        break
    if event.type == "session.status_idle":
        if event.stop_reason.type == "requires_action":
            continue          # waiting on YOU — send a tool result or approval
        break                 # end_turn / retries_exhausted → genuinely done
```

Sessions go idle transiently — between parallel tool calls, or while waiting for your approval. Breaking on bare `status_idle` truncates the run and looks like the agent gave up.

---

## ⏰ Loops That Start Themselves

Not every loop is triggered by a user. Two production shapes:

**Scheduled loops** — a cron-style deployment fires a fresh session per firing. Each run is bounded; the *schedule* is the outer loop.

```python
deployment = client.beta.deployments.create(
    name="Weekly compliance scan",
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    initial_events=[{
        "type": "user.message",
        "content": [{"type": "text", "text": "Run the weekly compliance scan."}],
    }],
    schedule={"type": "cron", "expression": "0 20 * * 5", "timezone": "America/New_York"},
)
```

Operational details that bite:
- **Firings are jittered** to spread load — don't build a downstream deadline that assumes the exact scheduled minute.
- **Missed firings are not backfilled** after a pause. If gap-free coverage matters, make each run compute its own window from persisted state rather than assuming "since last run".
- **DST is literal wall-clock**: a 2AM schedule can be skipped on spring-forward and fire twice on fall-back. Schedule outside 1–3AM local, or use UTC.
- Each firing writes a **run record** — audit failures there, not in session logs, because a failed firing may never create a session.

**Event-driven loops** — a webhook (or queue message) wakes the harness, it does one bounded pass, and it exits. Cheaper than a poller and the right default for "react to X". Verify the signature, dedupe on event ID, and treat delivery as at-least-once and unordered: drive state from the resource you fetch, not from arrival order.

> **Anti-pattern: polling as a loop.** A `while True: sleep(5); check()` loop is a scheduled loop with worse cost and no audit trail. If something can notify you, be notified.

---

## ☠️ Loop Failure Modes

The catalogue below is what actually goes wrong. Each has a specific fix; none are fixed by "add more instructions".

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Thrash** | Same tool, same args, every iteration | Novelty gate on the *action*, not just results; feed the failure back explicitly |
| **Ping-pong** | Agent alternates between two fixes forever | Record attempted approaches in state; forbid repeats; cap iterations |
| **Premature exit** | Loop ends on the first plausible answer | Add a verification gate; make the stop predicate a checked criterion, not the model's opinion |
| **Never-dry loop** | Novelty gate never trips | Fingerprint is too sensitive (timestamps, IDs in the hash) — normalize before hashing |
| **Silent truncation** | Output stops mid-sentence, no error | `max_tokens` too low for thinking + answer; raise it and check `stop_reason` |
| **Budget starvation** | Loop stops with 80% of work undone | Budget spent on exploration; lower `effort` on workers, mask observations, cap fan-out |
| **Context rot** | Quality degrades after ~20 iterations | Context editing / compaction + journal for must-keep facts (Module 12) |
| **Verification loop** | Loop spends more on checking than doing | On models that self-verify, *delete* "double-check your work" instructions and separate verification steps |
| **Lost work on crash** | Restart re-runs everything | Durable journal or checkpointer keyed on deterministic step IDs |
| **Runaway fan-out** | 40 subagents for a 3-file change | Explicit delegation policy + hard spawn cap in the harness, not just the prompt |
| **Approval deadlock** | Run hangs forever, no error | Stream dropped while a tool awaited approval — reconnect with history consolidation and re-resolve pending calls |

**Instrument before you tune.** Log per iteration: iteration number, tokens in/out, tools called, new items found, stop-condition state. Nearly every failure above is obvious in that table and invisible without it.

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
    ├── budget_loop.py           Budget-aware execution loop
    └── outcome_loop.py          ★ Rubric-graded iterate → grade → revise loop
```

Run the outcome loop with no API key to inspect the harness mechanics:

```bash
cd 13-agent-harness
python loops/outcome_loop.py --mock     # scripted worker + grader
python loops/outcome_loop.py            # live, uses your configured provider
```



## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Agent crashes and loses progress | Durable journal with JSONL checkpointing, or a graph checkpointer (Module 16) |
| Budget exhausted too quickly | Lower `effort` on worker steps, mask observations, cap fan-out — then re-measure |
| Model ignores the budget you injected | Use API-native `task_budget` so the server tracks the countdown; prompt-injected numbers also break prompt caching |
| Novelty gate stops too early | Increase `dry_threshold`; check the fingerprint function |
| Novelty gate never trips | Fingerprint includes volatile data (timestamps, UUIDs) — normalize before hashing |
| Journal replay is slow | Use deterministic step IDs; skip completed steps efficiently |
| Compaction seems to do nothing | You appended only `.text` — append the full `response.content` so the compaction block survives |
| Output truncates mid-thought | `max_tokens` too low for thinking + answer at high `effort`; raise it and check `stop_reason` |
| Outcome loop never reaches `satisfied` | Rubric criteria aren't independently checkable — rewrite them so a failure names a specific gap |
| Hosted run appears to give up early | You broke on bare `session.status_idle`; check `stop_reason.type != "requires_action"` |
| Scheduled run didn't fire on time | Firings are jittered and DST is literal wall-clock — check the run records, not the session log |

## 📚 Resources

- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/) — durable agent state
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — agent patterns
- [Anthropic: Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — hosted agent loops, outcomes, scheduled deployments
- [Anthropic: Adaptive thinking & effort](https://platform.claude.com/docs/en/build-with-claude/effort) — the per-call depth dial
- [Anthropic: Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) and [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Temporal: Durable execution](https://docs.temporal.io/evaluate/understanding-temporal) — the industrial version of the journal
- [CrewAI](https://docs.crewai.com/) — multi-agent orchestration

## 🧪 Hands-On Exercises

1. **Loop-Until-Dry**: Build a `NoveltyGate` with threshold=3. Simulate an agent that finds 5 new items in round 1, 3 in round 2, 0 in round 3, 0 in round 4, 0 in round 5 — verify the loop stops after round 5, not round 3 or 4.

2. **Journal Resume Test**: Write a 5-step pipeline using `Journal.execute()`. After step 3, simulate a crash (raise Exception). On restart, verify steps 1-3 are skipped and only steps 4-5 re-run. Measure the token savings.

3. **Budget Governor**: Build a `BudgetTracker` with 50,000 tokens. Run a loop that uses ~5,000 tokens/iteration. Verify it stops before exceeding the budget, AND that the system prompt correctly reflects remaining budget each turn.

4. **Self-Repair**: Design a tool that fails 50% of the time with a retriable error. Wrap it in the self-repair loop (max 3 retries). Run 20 tasks and measure: (a) success rate with vs. without repair, (b) average retries per success.

5. **Adversarial Verify Calibration**: Generate 20 claims — 10 true, 10 false. Run adversarial verification with N=3 voters on each. Measure precision and recall. What's the accuracy at different majority thresholds (1/3 must refute, 2/3, all 3)?

6. **Pipeline vs Barrier**: Process 10 items through a 3-stage pipeline. Measure wall-clock time for: (a) pure sequential, (b) parallel with barrier after each stage, (c) pipeline (no barriers). Plot the speedup.

7. **Outcome Loop**: Run `python loops/outcome_loop.py --mock` and confirm it stops at `satisfied` on iteration 3. Then write a rubric for a real task of your own with 5 criteria, run the live loop, and record how many iterations each criterion took to satisfy. Rewrite the slowest criterion to be more concrete and re-measure.

8. **Rubric Ambiguity**: Take one vague criterion ("the summary should be clear") and grade the *same* artifact 5 times with `temperature=0`. Count how often the verdict flips. Now rewrite it as a checkable criterion and repeat. Ambiguous criteria are why outcome loops burn iterations.

9. **Budget Comparison**: Run the same research task three ways — (a) no budget control, (b) prompt-injected remaining budget, (c) API-native `task_budget`. Compare total tokens spent, tokens *wasted after* the task was effectively complete, and cache-read hit rate. Explain why (b) hurts the cache.

10. **Context Lifecycle**: Run a 30-iteration tool loop with no pruning, then with observation masking, then with masking + context editing. Plot context size per iteration and note where answer quality starts to degrade in the unpruned run.

11. **Failure-Mode Bingo**: Deliberately induce four failures from the failure-mode table (thrash, premature exit, silent truncation, never-dry loop). For each, write down the *single log line* that would have identified it. That set of log lines is your harness's minimum instrumentation.

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

- **Module 07 (Agents)**: Harness patterns extend basic agent and multi-agent flows
- **Module 12 (Context Engineering)**: Harnesses must budget context per iteration; compaction and context editing are loop-level policies
- **Module 16 (Graph Engineering)**: A graph is the *other* way to express control flow — explicit topology instead of an open loop. Graph checkpointers are the productized version of pattern 3
- **Module 08 (LLM Ops)**: Add tracing to each harness iteration — the per-iteration log table is what makes failure modes visible
- **Module 09 (EvalOps)**: Evaluate harness quality end-to-end; the outcome-loop grader is an LLM-as-judge with a rubric (Module 04)
- **Module 14 (MCP & Tool Design)**: Tool descriptions decide how often your loop calls the right tool

### Loop or graph?

Both control iteration; they answer different questions.

| Use a **loop** (this module) when… | Use a **graph** (Module 16) when… |
|-----------------------------------|-----------------------------------|
| The number of steps is unknown up front | The steps and their order are known |
| The next action emerges from the last result | Branching is conditional but enumerable |
| You want exhaustiveness (until dry / until satisfied) | You want auditability and replay |
| State is a running transcript | State is a typed object with reducers |

In practice production systems nest them: a graph whose nodes contain bounded loops, or a loop whose each iteration executes a graph.

---

**The harness is what separates a demo from a production agent. Build it first.**

## Resources
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [CrewAI](https://docs.crewai.com/)
