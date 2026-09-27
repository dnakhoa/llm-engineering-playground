"""The Acme Notes Backend, tested directly on Action policy.

The spec keeps direct Backend tests to Action policy, because every Check relies
on the Backend being deterministic and on its Actions enforcing authorization
themselves. Everything else about the Backend is exercised through the Case
runner (tests/test_case_runner.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from company.backend import Backend  # noqa: E402
from llm.types import ToolCall  # noqa: E402

CUSTOMER = "acct_1001"  # on Free in the seed
SOMEONE_ELSE = "acct_1002"  # on Pro in the seed


def _call(name, **arguments):
    return ToolCall(id="call_1", name=name, arguments=arguments)


def test_a_customer_can_change_their_own_plan():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    result = actions.run(_call("change_plan", account_id=CUSTOMER, plan="pro"))

    assert not result.is_error
    assert backend.export_state()["accounts"][CUSTOMER]["plan"] == "pro"


def test_a_cross_account_plan_change_is_refused_by_the_action():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    result = actions.run(_call("change_plan", account_id=SOMEONE_ELSE, plan="free"))

    assert result.is_error
    assert "not the account" in result.content
    assert backend.export_state()["accounts"][SOMEONE_ELSE]["plan"] == "pro"
    assert actions.log[-1].executed is False


def test_a_cross_account_lookup_is_refused_too():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    result = actions.run(_call("look_up_account", account_id=SOMEONE_ELSE))

    assert result.is_error
    assert "Northwind Studio" not in result.content  # nothing about the other account leaks


def test_an_unknown_plan_is_refused():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    result = actions.run(_call("change_plan", account_id=CUSTOMER, plan="platinum"))

    assert result.is_error
    assert backend.export_state()["accounts"][CUSTOMER]["plan"] == "free"


def test_an_action_called_with_the_wrong_arguments_is_refused_not_raised():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    result = actions.run(_call("change_plan", account=CUSTOMER))

    assert result.is_error
    assert "wrong arguments" in result.content
    assert actions.log[-1].executed is False


def test_every_seeded_backend_starts_from_the_same_state():
    changed = Backend.seeded()
    changed.actions_for(CUSTOMER).run(_call("change_plan", account_id=CUSTOMER, plan="team"))

    assert Backend.seeded().export_state() == Backend.seeded().export_state()
    assert Backend.seeded().export_state() != changed.export_state()


# ── issue_refund: the refund window, proration and the maximum refund ─────────
#
# The seed's "today" is 2026-09-20. Northwind Studio (acct_1002) paid $12.00 for
# a 30-day Pro month on 2026-09-13, so 7 days are used and 23 unused: the policy
# allows 12.00 x 23 / 30 = $9.20. Harbor Robotics Club (acct_1003) paid $960.00
# for a year of Team on 2026-08-06, 45 days ago: outside the 30-day window.

PRO_MONTHLY = "acct_1002"
TEAM_ANNUAL = "acct_1003"


def _refund(actions, account_id, invoice_id, amount_usd):
    return actions.run(
        _call("issue_refund", account_id=account_id, invoice_id=invoice_id,
              amount_usd=amount_usd)
    )


def test_a_prorated_refund_within_the_window_is_issued_once():
    backend = Backend.seeded()
    actions = backend.actions_for(PRO_MONTHLY)

    result = _refund(actions, PRO_MONTHLY, "inv_2002", 9.20)

    assert not result.is_error, result.content
    state = backend.export_state()
    assert state["refunds"] == {
        "rf_0001": {"account_id": PRO_MONTHLY, "invoice_id": "inv_2002", "amount_cents": 920}
    }
    assert state["accounts"][PRO_MONTHLY]["invoices"]["inv_2002"]["refunded_cents"] == 920


def test_a_refund_above_the_prorated_amount_is_refused():
    backend = Backend.seeded()
    actions = backend.actions_for(PRO_MONTHLY)

    result = _refund(actions, PRO_MONTHLY, "inv_2002", 12.00)  # the full month

    assert result.is_error
    assert "refund policy" in result.content
    assert "9.20" not in result.content  # the agent has to read the policy, not the error
    assert backend.export_state()["refunds"] == {}
    assert actions.log[-1].executed is False


def test_a_second_refund_cannot_take_the_invoice_past_the_policy_amount():
    backend = Backend.seeded()
    actions = backend.actions_for(PRO_MONTHLY)

    assert not _refund(actions, PRO_MONTHLY, "inv_2002", 5.00).is_error
    assert _refund(actions, PRO_MONTHLY, "inv_2002", 5.00).is_error
    assert not _refund(actions, PRO_MONTHLY, "inv_2002", 4.20).is_error

    assert backend.export_state()["accounts"][PRO_MONTHLY]["invoices"]["inv_2002"][
        "refunded_cents"] == 920


def test_a_refund_outside_the_window_is_refused_by_the_action():
    backend = Backend.seeded()
    actions = backend.actions_for(TEAM_ANNUAL)

    result = _refund(actions, TEAM_ANNUAL, "inv_3001", 10.00)

    assert result.is_error
    assert "30 days" in result.content
    assert backend.export_state()["refunds"] == {}


def test_a_refund_above_the_maximum_is_refused_even_inside_the_window():
    # A Team year bought yesterday: $960.00, 364 of 365 days unused. Proration
    # alone would allow $957.36, but no single refund may exceed $200.00.
    state = Backend.seeded().export_state()
    state["accounts"][TEAM_ANNUAL]["invoices"]["inv_3001"]["date"] = "2026-09-19"
    backend = Backend(state)
    actions = backend.actions_for(TEAM_ANNUAL)

    assert _refund(actions, TEAM_ANNUAL, "inv_3001", 200.01).is_error
    assert not _refund(actions, TEAM_ANNUAL, "inv_3001", 200.00).is_error


def test_a_refund_on_someone_elses_invoice_is_refused():
    backend = Backend.seeded()
    actions = backend.actions_for(CUSTOMER)

    for account_id in (PRO_MONTHLY, CUSTOMER):  # their account, or theirs named as mine
        assert _refund(actions, account_id, "inv_2002", 1.00).is_error
    assert backend.export_state()["refunds"] == {}


def test_a_refund_of_nothing_or_of_fractions_of_a_cent_is_refused():
    backend = Backend.seeded()
    actions = backend.actions_for(PRO_MONTHLY)

    for amount in (0, -1.00, 1.005, "a lot", "nan", "inf", True):
        assert _refund(actions, PRO_MONTHLY, "inv_2002", amount).is_error, amount
    assert backend.export_state()["refunds"] == {}


# ── A Case offers only the Actions it declares (ADR 0005) ─────────────────────


def test_an_undeclared_action_is_refused_and_logged_as_attempted():
    backend = Backend.seeded()
    actions = backend.actions_for(PRO_MONTHLY, allowed=("look_up_account", "change_plan"))

    result = _refund(actions, PRO_MONTHLY, "inv_2002", 9.20)

    assert [spec.name for spec in actions.tools] == ["look_up_account", "change_plan"]
    assert result.is_error
    assert "not available on this Case" in result.content
    assert actions.log[-1].name == "issue_refund" and actions.log[-1].executed is False
    assert backend.export_state()["refunds"] == {}
