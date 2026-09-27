"""The Acme Notes Backend: deterministic, in-memory, reset for every Case.

The Backend holds accounts, their plans, their invoices and the refunds issued
against them. The Flagship Agent changes it only through Actions, and every
Action checks who is asking *inside the Action*. The prompt can say whatever it
likes; a request for someone else's account is refused here, because here is
the only place a refusal cannot be talked out of. Policy limits live here for
the same reason: ``issue_refund`` enforces the refund window, proration and the
maximum refund itself, whatever the agent was told.

    backend = Backend.seeded()
    actions = backend.actions_for("acct_1001")      # the Case's own customer
    actions.run(ToolCall(id="c1", name="change_plan",
                         arguments={"account_id": "acct_1001", "plan": "pro"}))
    backend.export_state()["accounts"]["acct_1001"]["plan"]   # "pro"
"""
from __future__ import annotations

import copy
import datetime
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from llm.types import ToolCall, ToolResult, ToolSpec

SEED_PATH = Path(__file__).resolve().parent / "fixtures" / "accounts.json"

PLANS = ("free", "pro", "team")

# The refund policy, as ``company/knowledge_base/refund-policy.md`` states it to
# customers. The article is what the agent reads; these are what the Action
# enforces. tests/test_company_knowledge.py keeps the two in step.

#: A refund is possible up to this many days after the invoice date.
REFUND_WINDOW_DAYS = 30

#: No single refund may be larger than this; bigger ones go to the billing team.
MAX_REFUND_CENTS = 20000


class ActionRefused(Exception):
    """An Action declined to run. The message goes back to the agent verbatim."""


def _load_seed(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "today": data["today"],
        "accounts": {
            entry["account_id"]: {k: v for k, v in entry.items() if k != "account_id"}
            for entry in data["accounts"]
        },
        "refunds": {},
    }


def _days_between(earlier: str, later: str) -> int:
    return (datetime.date.fromisoformat(later) - datetime.date.fromisoformat(earlier)).days


def refundable_cents(invoice: Mapping[str, Any], today: str) -> int:
    """The most the refund policy allows back on ``invoice`` today, in cents.

    Outside the refund window that is nothing. Inside it, it is the unused part
    of the billing period, prorated by day and rounded down to the cent, less
    what was already refunded, and never more than the maximum refund.
    """
    used_days = _days_between(invoice["date"], today)
    if used_days > REFUND_WINDOW_DAYS:
        return 0
    unused_days = max(invoice["period_days"] - used_days, 0)
    prorated = invoice["amount_cents"] * unused_days // invoice["period_days"]
    return max(min(prorated - invoice["refunded_cents"], MAX_REFUND_CENTS), 0)


class Backend:
    """Acme Notes' accounts, plans, invoices and refunds. Build a fresh one per Case."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state: Dict[str, Any] = copy.deepcopy(dict(state))

    @classmethod
    def seeded(cls, path: Optional[Path] = None) -> "Backend":
        """A Backend in the seed state. Same seed, same state, every time."""
        return cls(_load_seed(path or SEED_PATH))

    def export_state(self) -> Dict[str, Any]:
        """A deep copy of the whole state, safe for Checks to assert on."""
        return copy.deepcopy(self._state)

    def actions_for(
        self, customer_account_id: str, allowed: Optional[Sequence[str]] = None
    ) -> "Actions":
        """The Actions, bound to one Case's customer.

        ``allowed`` is the Actions the Case declares, in the order the agent is
        offered them; any other Action is refused. ``None`` allows them all.
        """
        return Actions(
            self,
            customer_account_id,
            allowed=tuple(ACTION_NAMES if allowed is None else allowed),
        )

    # Operations. They assume authorization already happened; Actions do that.

    def _account(self, account_id: str) -> Dict[str, Any]:
        account = self._state["accounts"].get(account_id)
        if account is None:
            raise ActionRefused("There is no account {}.".format(account_id))
        return account

    def _set_plan(self, account_id: str, plan: str) -> None:
        self._account(account_id)["plan"] = plan

    @property
    def _today(self) -> str:
        return self._state["today"]

    def _invoice(self, account_id: str, invoice_id: str) -> Dict[str, Any]:
        invoice = self._account(account_id).get("invoices", {}).get(invoice_id)
        if invoice is None:
            raise ActionRefused(
                "{} has no invoice {}.".format(account_id, invoice_id)
            )
        return invoice

    def _refund(self, account_id: str, invoice_id: str, amount_cents: int) -> str:
        refunds = self._state["refunds"]
        refund_id = "rf_{:04d}".format(len(refunds) + 1)
        refunds[refund_id] = {
            "account_id": account_id,
            "invoice_id": invoice_id,
            "amount_cents": amount_cents,
        }
        self._invoice(account_id, invoice_id)["refunded_cents"] += amount_cents
        return refund_id


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
    ToolSpec(
        name="issue_refund",
        description=(
            "Refund part or all of one invoice on an Acme Notes account, in US "
            "dollars. Refunds follow Acme Notes' refund policy, so read it in the "
            "Knowledge Base first; a refund the policy does not allow is refused. "
            "Only the customer's own account can be refunded."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "account_id": _ACCOUNT_ID,
                "invoice_id": {
                    "type": "string",
                    "description": "The invoice to refund, for example inv_2002.",
                },
                "amount_usd": {
                    "type": "number",
                    "description": "The amount to refund, in dollars and cents, for example 4.50.",
                },
            },
            "required": ["account_id", "invoice_id", "amount_usd"],
        },
    ),
)

#: Every Action the Backend has. A Case offers the agent only the ones it declares.
ACTION_NAMES: Tuple[str, ...] = tuple(spec.name for spec in TOOL_SPECS)
_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


@dataclass
class Actions:
    """The Backend's Actions for one customer, with a log of every attempt."""

    backend: Backend
    customer_account_id: str
    allowed: Tuple[str, ...] = ACTION_NAMES
    log: List[ActionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = [name for name in self.allowed if name not in _SPECS_BY_NAME]
        if unknown:
            raise ValueError(
                "There is no Action called {}. The Backend has: {}.".format(
                    ", ".join(unknown), ", ".join(ACTION_NAMES)
                )
            )

    @property
    def tools(self) -> Tuple[ToolSpec, ...]:
        """The Actions this Case allows, as tools, in the order it declares them."""
        return tuple(_SPECS_BY_NAME[name] for name in self.allowed)

    def run(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Refusals come back as an error result, never raised."""
        handlers: Dict[str, Callable[..., str]] = {
            "look_up_account": self._look_up_account,
            "change_plan": self._change_plan,
            "issue_refund": self._issue_refund,
        }
        handler = handlers.get(call.name)
        try:
            if handler is None:
                raise ActionRefused("There is no Action called {}.".format(call.name))
            if call.name not in self.allowed:
                # ADR 0005: the Case did not offer it, so it does not run, even
                # though the Backend has it. The attempt is still logged.
                raise ActionRefused(
                    "Refused: {} is not available on this Case.".format(call.name)
                )
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
        account = copy.deepcopy(self.backend._account(account_id))
        for invoice in account.get("invoices", {}).values():
            invoice["days_since_invoice"] = _days_between(invoice["date"], self.backend._today)
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

    def _issue_refund(self, account_id: str, invoice_id: str, amount_usd: Any) -> str:
        self._authorize(account_id)
        invoice = self.backend._invoice(account_id, invoice_id)
        try:
            cents = float(amount_usd) * 100
        except (TypeError, ValueError):
            cents = math.nan
        if isinstance(amount_usd, bool) or not math.isfinite(cents):
            raise ActionRefused("{!r} is not an amount in dollars.".format(amount_usd))
        amount_cents = int(round(cents))
        if abs(cents - amount_cents) > 1e-6:
            raise ActionRefused(
                "{!r} is not an amount in dollars and cents.".format(amount_usd)
            )
        if amount_cents <= 0:
            raise ActionRefused("A refund must be more than $0.00.")
        if _days_between(invoice["date"], self.backend._today) > REFUND_WINDOW_DAYS:
            raise ActionRefused(
                "Refused: {} is from {}, more than {} days ago, so the refund policy "
                "allows no refund on it.".format(invoice_id, invoice["date"], REFUND_WINDOW_DAYS)
            )
        if amount_cents > refundable_cents(invoice, self.backend._today):
            # The limit itself is not in the message: the agent is meant to get
            # the amount right from the policy, not by bargaining with the error.
            raise ActionRefused(
                "Refused: ${:.2f} on {} is more than the refund policy allows.".format(
                    amount_cents / 100, invoice_id
                )
            )
        refund_id = self.backend._refund(account_id, invoice_id, amount_cents)
        return "Refund {}: ${:.2f} on {} for {}.".format(
            refund_id, amount_cents / 100, invoice_id, account_id
        )
