"""
A state graph in ~150 lines — no dependencies, no API key.

The point is not to replace LangGraph. It is to make the four mechanics that
matter concrete enough to reason about:

    1. supersteps      — nodes scheduled together run together, then merge
    2. reducers        — how parallel updates combine instead of clobbering
    3. conditional edges — branch and stop conditions as testable Python
    4. checkpointing   — state after every superstep, so you can resume

Run:
    python examples/state_graph.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"
START = "__start__"


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

def last_write_wins(old: Any, new: Any) -> Any:
    return new


def append(old: list, new: list) -> list:
    return list(old) + list(new)


def add_int(old: int, new: int) -> int:
    return old + new


@dataclass
class Checkpoint:
    step: int
    frontier: list[str]
    state: dict


@dataclass
class StateGraph:
    """A minimal superstep executor with reducers and conditional edges."""

    reducers: dict[str, Callable[[Any, Any], Any]] = field(default_factory=dict)
    nodes: dict[str, Callable[[dict], dict]] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    branches: dict[str, Callable[[dict], list[str] | str]] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    max_steps: int = 50

    # ── build ────────────────────────────────────────────────────────────
    def add_node(self, name: str, fn: Callable[[dict], dict]) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges.setdefault(src, []).append(dst)
        return self

    def add_conditional_edges(self, src: str, router: Callable[[dict], list[str] | str]):
        """`router` returns the next node name(s) — this is where stop conditions live."""
        self.branches[src] = router
        return self

    # ── run ──────────────────────────────────────────────────────────────
    def _merge(self, state: dict, update: dict) -> dict:
        """Apply one node's partial update through the per-field reducers."""
        merged = dict(state)
        for key, value in (update or {}).items():
            reducer = self.reducers.get(key, last_write_wins)
            merged[key] = reducer(merged[key], value) if key in merged else value
        return merged

    def _next(self, node: str, state: dict) -> list[str]:
        if node in self.branches:
            result = self.branches[node](state)
            return [result] if isinstance(result, str) else list(result)
        return list(self.edges.get(node, []))

    def invoke(self, initial: dict, *, verbose: bool = True) -> dict:
        state = dict(initial)
        frontier = self._next(START, state)
        step = 0

        while frontier and END not in frontier:
            if step >= self.max_steps:
                raise RuntimeError(
                    f"graph exceeded {self.max_steps} supersteps — "
                    "a cycle is missing its mechanical stop condition"
                )
            step += 1

            # One superstep: every node on the frontier sees the SAME input state.
            # Their updates are collected, then merged — that is why two writers
            # to the same field need a reducer.
            updates = [(name, self.nodes[name](state)) for name in frontier]
            if verbose:
                print(f"  superstep {step}: {', '.join(frontier)}")

            for name, update in updates:
                state = self._merge(state, update)

            self.checkpoints.append(Checkpoint(step, list(frontier), dict(state)))

            nxt: list[str] = []
            for name, _ in updates:
                for target in self._next(name, state):
                    if target not in nxt:
                        nxt.append(target)
            frontier = [n for n in nxt if n != END]
            if not nxt:
                break

        return state

    # ── time travel ──────────────────────────────────────────────────────
    def resume_from(self, step: int, *, patch: dict | None = None,
                    verbose: bool = True) -> dict:
        """Re-run from a stored checkpoint, optionally editing state first."""
        cp = next(c for c in self.checkpoints if c.step == step)
        state = dict(cp.state)
        state.update(patch or {})
        frontier: list[str] = []
        for name in cp.frontier:
            for target in self._next(name, state):
                if target not in frontier and target != END:
                    frontier.append(target)

        if verbose:
            print(f"  ↩ resuming after step {step} with frontier {frontier}")
        sub = StateGraph(self.reducers, self.nodes, self.edges, self.branches,
                         max_steps=self.max_steps)
        sub.edges = self.edges
        sub.branches = dict(self.branches)
        sub.branches[START] = lambda _s, f=tuple(frontier): list(f)
        return sub.invoke(state, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 — corrective retrieval with a bounded cycle
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = {
    "churn": "Q4 churn was 3.1%, down from 4.0% in Q3.",
    "revenue": "Q4 revenue was $210M, up 12% YoY.",
    "headcount": "Headcount grew to 1,240 at end of Q4.",
}


def build_corrective_graph() -> StateGraph:
    g = StateGraph(reducers={"docs": append, "attempts": add_int})

    def retrieve(state: dict) -> dict:
        hits = [text for key, text in CORPUS.items() if key in state["question"].lower()]
        return {"docs": hits, "attempts": 1}

    def grade(state: dict) -> dict:
        # A real grader is an LLM call; the shape is identical.
        return {"relevant": bool(state["docs"])}

    def rewrite(state: dict) -> dict:
        # Naive query expansion — enough to show the cycle closing.
        return {"question": state["question"] + " churn"}

    def generate(state: dict) -> dict:
        if not state["docs"]:
            return {"answer": "I don't have evidence for that."}
        return {"answer": " ".join(state["docs"])}

    def route(state: dict) -> str:
        """Both stop conditions live here, in plain testable code."""
        if state["relevant"]:
            return "generate"
        if state["attempts"] >= 3:      # mechanical stop — never rely on the semantic one alone
            return "generate"
        return "rewrite"

    (g.add_node("retrieve", retrieve)
      .add_node("grade", grade)
      .add_node("rewrite", rewrite)
      .add_node("generate", generate))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route)
    g.add_edge("rewrite", "retrieve")     # the cycle
    g.add_edge("generate", END)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 — the lost-update bug, with and without a reducer
# ─────────────────────────────────────────────────────────────────────────────

def build_parallel_graph(*, with_reducer: bool) -> StateGraph:
    g = StateGraph(reducers={"docs": append} if with_reducer else {})

    g.add_node("search_web", lambda s: {"docs": ["web:a", "web:b"]})
    g.add_node("search_kb", lambda s: {"docs": ["kb:x"]})
    g.add_node("synthesize", lambda s: {"answer": f"{len(s['docs'])} docs used"})

    g.add_conditional_edges(START, lambda s: ["search_web", "search_kb"])  # fan out
    g.add_edge("search_web", "synthesize")                                 # fan in
    g.add_edge("search_kb", "synthesize")
    g.add_edge("synthesize", END)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 — dynamic fan-out with an explicit, logged cap
# ─────────────────────────────────────────────────────────────────────────────

def build_fanout_graph(max_width: int = 3) -> StateGraph:
    g = StateGraph(reducers={"findings": append})
    files = ["auth.py", "db.py", "api.py", "ui.py", "jobs.py"]

    g.add_node("discover", lambda s: {"files": files})

    def make_worker(name: str):
        return lambda s: {"findings": [f"{name}: ok"]}

    for f in files[:max_width]:
        g.add_node(f"analyze::{f}", make_worker(f))
        g.add_edge(f"analyze::{f}", "report")

    def report(state: dict) -> dict:
        dropped = len(state["files"]) - max_width
        note = f" ({dropped} file(s) NOT analyzed — fan-out cap {max_width})" if dropped > 0 else ""
        return {"answer": f"{len(state['findings'])} findings{note}"}

    g.add_node("report", report)
    g.add_edge(START, "discover")
    # Fan out at runtime over whatever `discover` found, bounded by the cap.
    g.add_conditional_edges(
        "discover",
        lambda s: [f"analyze::{f}" for f in s["files"][:max_width]],
    )
    g.add_edge("report", END)
    return g


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== 1. Corrective retrieval (bounded cycle) ===")
    g = build_corrective_graph()
    out = g.invoke({"question": "what happened to revenue?", "docs": [], "attempts": 0,
                    "relevant": False, "answer": ""})
    print(f"  answer: {out['answer']}")
    print(f"  attempts: {out['attempts']}, checkpoints: {len(g.checkpoints)}")

    print("\n  same graph, a question the corpus can't answer:")
    g2 = build_corrective_graph()
    out2 = g2.invoke({"question": "what is our NPS?", "docs": [], "attempts": 0,
                      "relevant": False, "answer": ""})
    print(f"  answer: {out2['answer']}  (attempts={out2['attempts']})")
    print("  Note the real bug the trace exposes: the naive rewrite appended a")
    print("  keyword, retrieved an UNRELATED doc, and the grader passed it. The")
    print("  cycle terminated correctly and still answered the wrong question —")
    print("  which is why `grade` belongs in your eval suite, not just `route`.")

    print("\n=== 2. Lost update: reducers are not optional ===")
    for flag in (False, True):
        graph = build_parallel_graph(with_reducer=flag)
        result = graph.invoke({"docs": [], "answer": ""}, verbose=False)
        label = "with reducer   " if flag else "without reducer"
        print(f"  {label}: docs={result['docs']} → {result['answer']}")
    print("  Without a reducer the second branch overwrites the first — silently.")

    print("\n=== 3. Dynamic fan-out with a logged cap ===")
    fan = build_fanout_graph(max_width=3)
    res = fan.invoke({"files": [], "findings": [], "answer": ""})
    print(f"  {res['answer']}")
    print("  A cap you don't report reads as 'we covered everything'.")

    print("\n=== 4. Time travel: resume from a checkpoint with edited state ===")
    g3 = build_corrective_graph()
    g3.invoke({"question": "what is our NPS?", "docs": [], "attempts": 0,
               "relevant": False, "answer": ""}, verbose=False)
    replayed = g3.resume_from(1, patch={"docs": ["NPS was 46 in Q4."], "relevant": True})
    print(f"  replayed answer: {replayed['answer']}")
    print("  Same prefix, edited state, different outcome — this is the agent debugger.")

    print("\n=== Checkpoint log (demo 1) ===")
    for cp in g.checkpoints:
        print(f"  step {cp.step:>2} after {','.join(cp.frontier):<12} "
              f"{json.dumps({k: v for k, v in cp.state.items() if k != 'docs'})}")


if __name__ == "__main__":
    main()
