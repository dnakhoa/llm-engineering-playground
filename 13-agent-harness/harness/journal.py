"""
Durable append-only journal for crash-proof agent runs.

Each completed step is persisted to JSONL before the next begins.
On restart, completed steps are replayed from disk — the function
is NOT re-called for journaled steps.
"""
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Journal:
    """
    Provides idempotent step execution with durable state.

    Usage:
        journal = Journal("run_2026_06_23_001")
        result = journal.execute("fetch_data", fetch_fn, url)
        # If run again after a crash, 'fetch_data' is skipped — returns cached result
    """

    def __init__(self, run_id: str, journal_dir: str = ".journals"):
        self.run_id = run_id
        self.path = Path(journal_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: dict[str, Any] = {}
        self._step_count = 0
        self._load()

    def _load(self) -> None:
        """Replay existing journal entries on startup."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    self._completed[entry["step_id"]] = entry["result"]
        if self._completed:
            logger.info(
                f"[journal] Resumed run '{self.run_id}': "
                f"{len(self._completed)} steps already completed"
            )
            print(
                f"  [journal] ↩  Resumed: {len(self._completed)} "
                f"step(s) already done for run '{self.run_id}'"
            )

    def step_id(self, step_name: str, *args: Any) -> str:
        """Deterministic step ID based on step name + arguments."""
        payload = json.dumps({"step": step_name, "args": [str(a) for a in args]},
                             sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:16]

    def is_done(self, step_name: str, *args: Any) -> bool:
        return self.step_id(step_name, *args) in self._completed

    def get_result(self, step_name: str, *args: Any) -> Any:
        sid = self.step_id(step_name, *args)
        if sid not in self._completed:
            raise KeyError(f"Step '{step_name}' not in journal")
        return self._completed[sid]

    def record(self, sid: str, step_name: str, result: Any) -> None:
        """Append a completed step to the JSONL journal."""
        entry = {
            "step_id": sid,
            "step_name": step_name,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._completed[sid] = result
        self._step_count += 1

    def execute(self, step_name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a step only if not already journaled.
        On resume, returns the cached result without calling fn.

        Args:
            step_name: Human-readable name (also used for dedup key)
            fn: The function to execute
            *args: Arguments passed to fn (also hashed for dedup)

        Returns:
            fn(*args, **kwargs) result (or cached result on resume)
        """
        sid = self.step_id(step_name, *args)

        if sid in self._completed:
            print(f"  [journal] ↩  Skip '{step_name}' (cached)")
            return self._completed[sid]

        print(f"  [journal] ▶  Executing '{step_name}'...")
        result = fn(*args, **kwargs)
        self.record(sid, step_name, result)
        print(f"  [journal] ✓  Completed '{step_name}'")
        return result

    def delete(self) -> None:
        """Remove the journal file (start fresh on next run)."""
        if self.path.exists():
            self.path.unlink()
            self._completed.clear()

    def stats(self) -> dict:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "completed_steps": len(self._completed),
        }
