"""
Novelty gate for loop-until-dry agent patterns.

Tracks which items have already been seen and detects when a loop
has gone 'dry' (N consecutive rounds with nothing new).
"""
import hashlib
import json
from typing import Any


class NoveltyGate:
    """
    Filters items to only those not previously seen.
    Counts consecutive 'dry' rounds to determine loop termination.

    Pattern: keep running until `dry_threshold` consecutive rounds
    return nothing new. This ensures exhaustive coverage without
    a fixed iteration count.
    """

    def __init__(self, dry_threshold: int = 3):
        self.dry_threshold = dry_threshold
        self.seen: set[str] = set()
        self.dry_rounds: int = 0
        self.total_rounds: int = 0
        self.total_new: int = 0

    def fingerprint(self, item: Any) -> str:
        """Stable hash for any item — used for deduplication."""
        if isinstance(item, (dict, list)):
            canonical = json.dumps(item, sort_keys=True)
        else:
            canonical = str(item)
        return hashlib.md5(canonical.encode()).hexdigest()

    def filter_new(self, items: list[Any]) -> list[Any]:
        """
        Return only items not seen before.
        Updates dry_rounds counter accordingly.
        """
        self.total_rounds += 1
        new_items = []
        for item in items:
            fp = self.fingerprint(item)
            if fp not in self.seen:
                self.seen.add(fp)
                new_items.append(item)

        if new_items:
            self.dry_rounds = 0  # reset on any new find
            self.total_new += len(new_items)
        else:
            self.dry_rounds += 1

        return new_items

    @property
    def is_dry(self) -> bool:
        """True when we've had `dry_threshold` consecutive dry rounds."""
        return self.dry_rounds >= self.dry_threshold

    def stats(self) -> dict:
        return {
            "total_rounds": self.total_rounds,
            "dry_rounds": self.dry_rounds,
            "dry_threshold": self.dry_threshold,
            "total_seen": len(self.seen),
            "total_new": self.total_new,
            "is_dry": self.is_dry,
        }
