"""
Knowledge graph + Graph RAG, dependency-free.

Covers the whole pipeline with a scripted extractor so it runs with no API key
and no graph database:

    text ──▶ extract ──▶ resolve ──▶ store ──▶ retrieve (local / global / path)
                                        │
                                   temporal edges (valid_from / valid_to)

Swap `scripted_extract` for an LLM call with a strict schema and the rest of the
file is unchanged — that is the point of keeping extraction behind one function.

Run:
    python examples/knowledge_graph.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# A closed relation vocabulary. Without this, every document invents new edge
# types and the graph becomes unqueryable.
RELATION_TYPES = {
    "WORKS_AT", "ACQUIRED", "SUPPLIES", "COMPETES_WITH", "PARTNERED_WITH", "LOCATED_IN",
}
ENTITY_TYPES = {"PERSON", "COMPANY", "PRODUCT", "LOCATION"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Edge:
    source: str
    type: str
    target: str
    evidence: str                 # the sentence it came from — never optional
    valid_from: str | None = None
    valid_to: str | None = None   # None → still believed true
    invalidated_by: str | None = None

    def active_at(self, as_of: str | None) -> bool:
        if as_of is None:
            return self.valid_to is None
        if self.valid_from and as_of < self.valid_from:
            return False
        if self.valid_to and as_of >= self.valid_to:
            return False
        return True

    def render(self) -> str:
        window = ""
        if self.valid_from or self.valid_to:
            window = f"  [{self.valid_from or '?'} → {self.valid_to or 'now'}]"
        return f"{self.source} —[{self.type}]→ {self.target}{window}"


@dataclass
class KnowledgeGraph:
    entities: dict[str, str] = field(default_factory=dict)        # canonical name -> type
    aliases: dict[str, str] = field(default_factory=dict)         # variant -> canonical
    edges: list[Edge] = field(default_factory=list)
    merge_log: list[str] = field(default_factory=list)

    # ── resolution ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize(name: str) -> str:
        n = name.lower().strip()
        n = re.sub(r"\b(inc|corp|corporation|co|ltd|llc|plc)\b\.?", "", n)
        return re.sub(r"[^a-z0-9 ]", "", n).strip()

    def resolve(self, name: str) -> str | None:
        """Cascade cheapest-first: exact → alias → normalized. Log every merge."""
        if name in self.entities:
            return name
        if name in self.aliases:
            return self.aliases[name]
        target = self._normalize(name)
        for canonical in self.entities:
            if self._normalize(canonical) == target:
                self.aliases[name] = canonical
                self.merge_log.append(f"{name!r} → {canonical!r} (normalized match)")
                return canonical
        return None

    def add_entity(self, name: str, etype: str) -> str:
        if etype not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type: {etype}")
        existing = self.resolve(name)
        if existing:
            return existing
        self.entities[name] = etype
        return name

    def add_edge(self, source: str, rtype: str, target: str, evidence: str,
                 *, valid_from: str | None = None) -> None:
        if rtype not in RELATION_TYPES:
            raise ValueError(f"relation {rtype!r} not in the closed vocabulary")
        if not evidence:
            raise ValueError("refusing an edge with no evidence — it would be unauditable")
        src = self.resolve(source) or source
        tgt = self.resolve(target) or target

        # A new fact INVALIDATES the old one rather than overwriting it.
        if valid_from:
            for i, e in enumerate(self.edges):
                if (e.source, e.type) == (src, rtype) and e.target != tgt and e.valid_to is None:
                    self.edges[i] = Edge(e.source, e.type, e.target, e.evidence,
                                         e.valid_from, valid_from, invalidated_by=evidence[:40])
        self.edges.append(Edge(src, rtype, tgt, evidence, valid_from))

    # ── retrieval ────────────────────────────────────────────────────────
    def neighbors(self, name: str, as_of: str | None = None) -> list[Edge]:
        canon = self.resolve(name)
        return [e for e in self.edges
                if canon in (e.source, e.target) and e.active_at(as_of)]

    def expand(self, anchors: list[str], hops: int = 2,
               as_of: str | None = None) -> list[Edge]:
        """LOCAL search: the subgraph around the entities named in the question."""
        seen_nodes = {self.resolve(a) or a for a in anchors}
        collected: list[Edge] = []
        for _ in range(hops):
            frontier: set[str] = set()
            for node in seen_nodes:
                for e in self.neighbors(node, as_of):
                    if e not in collected:
                        collected.append(e)
                    frontier |= {e.source, e.target}
            if frontier <= seen_nodes:
                break
            seen_nodes |= frontier
        return collected

    def paths(self, start: str, end: str, max_hops: int = 4,
              as_of: str | None = None) -> list[list[Edge]]:
        """PATH search: 'how is X connected to Y?' — BFS over active edges."""
        s = self.resolve(start) or start
        t = self.resolve(end) or end
        queue: list[tuple[str, list[Edge]]] = [(s, [])]
        found: list[list[Edge]] = []
        while queue:
            node, trail = queue.pop(0)
            if len(trail) >= max_hops:
                continue
            for e in self.neighbors(node, as_of):
                if e in trail:
                    continue
                nxt = e.target if e.source == node else e.source
                if nxt == t:
                    found.append(trail + [e])
                else:
                    queue.append((nxt, trail + [e]))
        return found

    def communities(self) -> dict[int, set[str]]:
        """
        Connected components stand in for community detection here. Real global
        search uses Leiden clustering plus an LLM summary per community, computed
        once at index time.
        """
        parent: dict[str, str] = {n: n for n in self.entities}

        def find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                x = parent[x]
            return x

        for e in self.edges:
            a, b = find(e.source), find(e.target)
            if a != b:
                parent[a] = b

        groups: dict[str, set[str]] = defaultdict(set)
        for node in list(parent):
            groups[find(node)].add(node)
        return {i: members for i, members in enumerate(groups.values())}

    def community_summaries(self) -> dict[int, str]:
        """GLOBAL search input: one summary per community (index-time cost)."""
        summaries = {}
        for cid, members in self.communities().items():
            inner = [e for e in self.edges if e.source in members and e.target in members]
            kinds = sorted({e.type for e in inner})
            summaries[cid] = (
                f"{len(members)} entities ({', '.join(sorted(members)[:4])}"
                f"{'...' if len(members) > 4 else ''}) linked by {', '.join(kinds) or 'no'} relations"
            )
        return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Extraction — replace this one function with an LLM + strict schema
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    ("doc_1", "Alice Chen joined Acme Corp as VP Engineering in February 2023.",
     [("Alice Chen", "PERSON"), ("Acme Corp", "COMPANY")],
     [("Alice Chen", "WORKS_AT", "Acme Corp", "2023-02-01")]),
    ("doc_2", "ACME Corporation acquired Bolt Systems in 2024 to expand in Berlin.",
     [("ACME Corporation", "COMPANY"), ("Bolt Systems", "COMPANY"), ("Berlin", "LOCATION")],
     [("ACME Corporation", "ACQUIRED", "Bolt Systems", "2024-05-01"),
      ("Bolt Systems", "LOCATED_IN", "Berlin", None)]),
    ("doc_3", "Bolt Systems supplies sensor modules to Cirrus Robotics.",
     [("Cirrus Robotics", "COMPANY")],
     [("Bolt Systems", "SUPPLIES", "Cirrus Robotics", None)]),
    ("doc_4", "Cirrus Robotics competes with Northwind Automation in warehouse robotics.",
     [("Northwind Automation", "COMPANY")],
     [("Cirrus Robotics", "COMPETES_WITH", "Northwind Automation", None)]),
    ("doc_5", "Alice Chen left Acme and started at Northwind Automation in July 2025.",
     [], [("Alice Chen", "WORKS_AT", "Northwind Automation", "2025-07-01")]),
    # A second, disconnected cluster — so community detection has something to find.
    ("doc_6", "Vela Pharma partnered with Kestrel Labs on trial logistics.",
     [("Vela Pharma", "COMPANY"), ("Kestrel Labs", "COMPANY")],
     [("Vela Pharma", "PARTNERED_WITH", "Kestrel Labs", None)]),
]


def scripted_extract(doc):
    """
    Stands in for: client.messages.create(..., output_config={"format": {...}})
    with the strict EXTRACTION_SCHEMA from the module README.
    """
    doc_id, text, entities, relations = doc
    return {
        "doc_id": doc_id,
        "entities": [{"name": n, "type": t} for n, t in entities],
        "relations": [
            {"source": s, "type": r, "target": o, "evidence": text, "valid_from": vf}
            for s, r, o, vf in relations
        ],
    }


def build_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for doc in DOCUMENTS:
        extracted = scripted_extract(doc)
        for ent in extracted["entities"]:
            kg.add_entity(ent["name"], ent["type"])
        for rel in extracted["relations"]:
            kg.add_edge(rel["source"], rel["type"], rel["target"],
                        rel["evidence"], valid_from=rel["valid_from"])
    return kg


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    kg = build_graph()

    print("=== Entity resolution ===")
    print(f"  canonical entities: {len(kg.entities)}")
    for line in kg.merge_log:
        print(f"  merged {line}")
    print("  'Acme Corp' and 'ACME Corporation' collapsed — without this the")
    print("  acquisition edge would dangle off a duplicate node.")

    print("\n=== LOCAL search: 'tell me about Bolt Systems' (2 hops) ===")
    for e in kg.expand(["Bolt Systems"], hops=2):
        print(f"  {e.render()}")

    print("\n=== PATH search: 'how is Alice Chen connected to Berlin?' ===")
    for as_of in ("2024-06-01", "2024-01-01"):
        found = kg.paths("Alice Chen", "Berlin", max_hops=4, as_of=as_of)
        print(f"  as of {as_of}:")
        if not found:
            print("    no active path (the acquisition hadn't happened yet)")
        for trail in found:
            print("    " + "  ⇒  ".join(e.render() for e in trail))
    print("  This is the multi-hop join vector search cannot do: no single chunk")
    print("  contains both 'Alice Chen' and 'Berlin'. Note the path also depends")
    print("  on WHEN you ask — temporal edges make that explicit instead of wrong.")

    print("\n=== GLOBAL search: community summaries (corpus-level questions) ===")
    for cid, summary in kg.community_summaries().items():
        print(f"  community {cid}: {summary}")

    print("\n=== TEMPORAL: 'where does Alice Chen work?' as-of three dates ===")
    for as_of in ("2024-01-01", "2025-01-01", "2026-01-01"):
        active = [e for e in kg.neighbors("Alice Chen", as_of) if e.type == "WORKS_AT"]
        answer = active[0].target if active else "unknown"
        print(f"  as of {as_of}: {answer}")
    print("  The 2023 edge was invalidated, not overwritten — so history survives")
    print("  and 'what did we believe last year?' is still answerable.")

    print("\n=== Guardrails ===")
    for bad in [
        ("Alice Chen", "MENTORS", "Bob", "some sentence"),   # relation not in vocabulary
        ("Alice Chen", "WORKS_AT", "Acme Corp", ""),         # no evidence
    ]:
        try:
            kg.add_edge(*bad)
        except ValueError as exc:
            print(f"  rejected: {exc}")


if __name__ == "__main__":
    main()
