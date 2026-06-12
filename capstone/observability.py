"""Cost + latency tracking — Module 08 applied."""
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_FILE = Path("./request_log.jsonl")

PRICING = {
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4o":      (2.50e-6, 10.0e-6),
}


@dataclass
class RequestMetrics:
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    question: str = ""
    model: str = "gpt-4o-mini"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    cost_usd: float = 0.0
    quality_score: Optional[float] = None
    error: Optional[str] = None

    def calculate_cost(self):
        ip, op = PRICING.get(self.model, PRICING["gpt-4o-mini"])
        self.cost_usd = self.input_tokens * ip + self.output_tokens * op

    def log(self):
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")


class ObservabilityTracker:
    def __init__(self):
        self._metrics: list[RequestMetrics] = []

    def record(self, m: RequestMetrics):
        m.calculate_cost()
        m.log()
        self._metrics.append(m)

    def summary(self) -> dict:
        if not self._metrics:
            return {}
        costs = [m.cost_usd for m in self._metrics]
        latencies = [m.latency_ms for m in self._metrics if not m.cache_hit]
        hits = sum(1 for m in self._metrics if m.cache_hit)
        return {
            "total_requests": len(self._metrics),
            "cache_hit_rate": hits / len(self._metrics),
            "total_cost_usd": sum(costs),
            "avg_cost_usd": sum(costs) / len(costs),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0,
        }
