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
