"""Compliance logging — audit trails for LLM interactions."""

import hashlib
import json
from datetime import datetime
from typing import Optional


class ComplianceLogger:
    """Log LLM interactions for compliance and auditing."""

    def __init__(self, retention_days: int = 365):
        self.retention_days = retention_days
        self._logs: list[dict] = []

    def log_interaction(
        self,
        user_id: str,
        request: dict,
        response: dict,
        guardrail_results: dict,
        metadata: dict = None,
    ) -> str:
        interaction_id = hashlib.sha256(
            f"{user_id}:{json.dumps(request, default=str)}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        record = {
            "interaction_id": interaction_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "request_hash": hashlib.sha256(json.dumps(request, default=str).encode()).hexdigest(),
            "response_hash": hashlib.sha256(json.dumps(response, default=str).encode()).hexdigest(),
            "guardrail_passed": guardrail_results.get("passed", False),
            "guardrail_issues": guardrail_results.get("issues", []),
            "metadata": metadata or {},
        }
        self._logs.append(record)
        return interaction_id

    def get_audit_trail(
        self,
        user_id: str = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        results = self._logs
        if user_id:
            results = [r for r in results if r["user_id"] == user_id]
        if start_date:
            results = [r for r in results if r["timestamp"] >= start_date.isoformat()]
        if end_date:
            results = [r for r in results if r["timestamp"] <= end_date.isoformat()]
        return results

    def summary(self) -> dict:
        return {
            "total_interactions": len(self._logs),
            "passed": sum(1 for r in self._logs if r["guardrail_passed"]),
            "blocked": sum(1 for r in self._logs if not r["guardrail_passed"]),
        }
