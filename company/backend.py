"""The Acme Notes Backend: deterministic, in-memory, reset for every Case.

The Backend holds accounts and their plans. The Flagship Agent changes it only
through Actions, and every Action checks who is asking *inside the Action*. The
prompt can say whatever it likes; a request for someone else's account is
refused here, because here is the only place a refusal cannot be talked out of.

    backend = Backend.seeded()
    actions = backend.actions_for("acct_1001")      # the Case's own customer
    actions.run(ToolCall(id="c1", name="change_plan",
                         arguments={"account_id": "acct_1001", "plan": "pro"}))
    backend.export_state()["accounts"]["acct_1001"]["plan"]   # "pro"
"""
from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from llm.types import ToolCall, ToolResult, ToolSpec

SEED_PATH = Path(__file__).resolve().parent / "fixtures" / "accounts.json"

PLANS = ("free", "pro", "team")


class ActionRefused(Exception):
    """An Action declined to run. The message goes back to the agent verbatim."""


def _load_seed(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "accounts": {
            entry["account_id"]: {k: v for k, v in entry.items() if k != "account_id"}
            for entry in data["accounts"]
        }
    }


class Backend:
    """Acme Notes' accounts and plans. Build a fresh one per Case."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state: Dict[str, Any] = copy.deepcopy(dict(state))

    @classmethod
    def seeded(cls, path: Optional[Path] = None) -> "Backend":
        """A Backend in the seed state. Same seed, same state, every time."""
        return cls(_load_seed(path or SEED_PATH))

    def export_state(self) -> Dict[str, Any]:
        """A deep copy of the whole state, safe for Checks to assert on."""
        return copy.deepcopy(self._state)

    def actions_for(self, customer_account_id: str) -> "Actions":
        """The Actions, bound to one Case's customer."""
        return Actions(self, customer_account_id)

    # Operations. They assume authorization already happened; Actions do that.

    def _account(self, account_id: str) -> Dict[str, Any]:
        account = self._state["accounts"].get(account_id)
        if account is None:
            raise ActionRefused("There is no account {}.".format(account_id))
        return account

    def _set_plan(self, account_id: str, plan: str) -> None:
        self._account(account_id)["plan"] = plan


@dataclass(frozen=True)
class ActionRecord:
    """One Action the agent asked for, and whether it ran."""

    name: str
    arguments: Mapping[str, Any]
    executed: bool
    result: str


# ── The Actions, as the agent sees them ───────────────────────────────────────

_ACCOUNT_ID = {
    "type": "string",
    "description": "The Acme Notes account ID, for example acct_1001.",
}

TOOL_SPECS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="look_up_account",
        description=(
            "Look up an Acme Notes account: its name, current plan and seats. "
            "Only the customer's own account can be looked up."
        ),
        input_schema={
            "type": "object",
            "properties": {"account_id": _ACCOUNT_ID},
            "required": ["account_id"],
        },
    ),
    ToolSpec(
        name="change_plan",
        description=(
            "Move an Acme Notes account to another plan: free, pro or team. "
            "Only the customer's own account can be changed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "account_id": _ACCOUNT_ID,
                "plan": {"type": "string", "enum": list(PLANS)},
            },
            "required": ["account_id", "plan"],
        },
    ),
)


@dataclass
class Actions:
    """The Backend's Actions for one customer, with a log of every attempt."""

    backend: Backend
    customer_account_id: str
    log: List[ActionRecord] = field(default_factory=list)

    @property
    def tools(self) -> Tuple[ToolSpec, ...]:
        return TOOL_SPECS

    def run(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Refusals come back as an error result, never raised."""
        handlers: Dict[str, Callable[..., str]] = {
            "look_up_account": self._look_up_account,
            "change_plan": self._change_plan,
        }
        handler = handlers.get(call.name)
        try:
            if handler is None:
                raise ActionRefused("There is no Action called {}.".format(call.name))
            try:
                inspect.signature(handler).bind(**dict(call.arguments))
            except TypeError:
                raise ActionRefused(
                    "{} was called with the wrong arguments: {}.".format(
                        call.name, json.dumps(dict(call.arguments), sort_keys=True)
                    )
                )
            content = handler(**dict(call.arguments))
            executed = True
        except ActionRefused as refusal:
            content, executed = str(refusal), False
        self.log.append(
            ActionRecord(
                name=call.name,
                arguments=dict(call.arguments),
                executed=executed,
                result=content,
            )
        )
        return ToolResult(
            call_id=call.id, name=call.name, content=content, is_error=not executed
        )

    def _authorize(self, account_id: str) -> None:
        # The whole point of ticket 03's fourth criterion: this check lives in
        # the Action, so no prompt can get around it.
        if account_id != self.customer_account_id:
            raise ActionRefused(
                "Refused: {} is not the account of the customer on this Case.".format(
                    account_id
                )
            )

    def _look_up_account(self, account_id: str) -> str:
        self._authorize(account_id)
        account = self.backend._account(account_id)
        return json.dumps(dict(account, account_id=account_id), sort_keys=True)

    def _change_plan(self, account_id: str, plan: str) -> str:
        self._authorize(account_id)
        if plan not in PLANS:
            raise ActionRefused(
                "{} is not a plan. Acme Notes has: {}.".format(plan, ", ".join(PLANS))
            )
        current = self.backend._account(account_id)["plan"]
        if current == plan:
            raise ActionRefused("{} is already on {}.".format(account_id, plan))
        self.backend._set_plan(account_id, plan)
        return "{} moved from {} to {}.".format(account_id, current, plan)
