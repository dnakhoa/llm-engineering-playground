# Spine 3 · Graded

_Last verified: 2026-09-27_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnakhoa/llm-engineering-playground/blob/main/spine/03-graded/graded.ipynb)

**What your agent can do at the end:** nothing new, on purpose. What changes
is that you can prove what it does. Its behaviour is pinned by a suite of Cases
that covers every Action it has, graded on every PR without a key, and a
regression that leaves the Backend looking perfect still fails the build.

**What you build:** the Graded suite, a judge rubric for what state cannot
show, a Case of your own written from an incident, and a CI step. The agent
from [Spine 2](../02-knowledge/README.md) does not change.

## Two ways to grade, and when to use each

**Backend-state assertions** read what the Backend ended up as and what the
agent tried to do to it. "Refunded exactly $9.20 on inv_2002, once, and
nothing else" is true or false, the same on every run, for free. Use them for
every outcome that lives in the Backend: plans, refunds, anything an Action
changes. They cannot be talked into a pass, because they never read the reply.

**LLM-as-judge rubrics** read the reply. A second model gets the conversation,
the Actions the agent attempted and a rubric ("declines the refund, gives the
30-day window as the reason, never implies money was refunded") and says pass
or fail. Use them for what state cannot show: whether the agent told the
customer the truth, in words they can act on. They cost a model call, they
need a live model, and they can be wrong, so they never replace a state Check.
A judge is the second opinion, not the verdict.

A rule of thumb: if the Backend can tell, assert on the Backend. Reach for a
judge only for the part of "did it help" that lives in the words.

## One verdict

Before this module, "did the agent do the right thing" had three definitions
that disagreed. `Outcome.resolved` counted a run as resolved when every
expected change had happened, and a Case that expects no change has no
expected changes, so an agent that downgraded the account unasked was
resolved. The Knowledge Checks failed that same run. The Budgeted module
(cost per resolved Case) and the Scoreboard (resolution rate) read
`resolved`, so they would have counted it as a success.

Now there is one: `outcome.verdict`, computed once by the Case runner
(`company/runner.py`). `outcome.resolved` is `verdict.resolved`, and each
Graded Check reports one part of it. Resolved means every part is empty:

| Part | What it holds | The Check that reports it |
|---|---|---|
| `unfinished` | the step limit, the Spend Cap, or a customer turn answered with an empty reply | the agent finishes the Case |
| `missing` | an expected change that did not happen, or happened with another value | Backend reaches the expected state |
| `unexpected` | a change the Case did not ask for | nothing changes beyond the expected state |
| `forbidden` | an attempt at an Action the Case forbids, even one the Backend refused | no forbidden Action is attempted |
| `repeated` | a change sent twice with the same arguments, even if the repeat was refused | each change is attempted once |

A test runs a panel of good and bad agents over every Graded Case and asserts
that `resolved` is true exactly when every Offline Graded Check passes.

## Grade the attempt, not only the result

The Backend refuses a refund outside the 30-day window whatever the agent
asks. So on `annual-refund-outside-window`, an agent that tries a $100 refund,
is refused, and tells the customer "Refunded $100" leaves the Backend exactly
as it found it. A Check that reads only what ran passes it. The Outcome keeps
every attempt (`actions_attempted`), and a forbidden Action counts when it was
attempted. The agent that lied to the customer fails.

The same goes for a change sent twice. The notebook seeds a regression into
the reference agent, the kind a retry wrapper introduces: every refund goes to
the Backend twice. On the prorated-refund Case the first refund uses up what
the invoice owes, so the Backend refuses the second, and the final state is
perfect. The Knowledge suite passes it, and the run prints "Passed through
module 2 of 3". The Graded suite fails it on the attempts: the Backend
refusing the repeat this time was luck. A $5.00 refund sent twice would have
run twice.

## An agent that never finishes fails

An agent that hits the step limit, or answers a customer turn with nothing,
fails every Case, including the Case where nothing was meant to change. Without
that rule, an agent that gives up at once passes `annual-refund-outside-window`
by doing nothing, and a customer who got silence counts as helped. The Check
is "the agent finishes the Case", and it runs in the Knowledge suite as well as
this one, because the multi-turn Case is where an empty second reply hides.

## The judge rubric is live only

`reply meets the judge rubric` (`checks/spine_3_graded.py`) is marked
`@live_only`. Offline, the Checks CLI reports it as SKIP, because a recording
holds the agent's model calls, not a judge's. Live, the CLI hands it a judge
(`checks/judge.py`) on the same model and key as the agent, and every judge
call counts against the run's Spend Cap. A course about budgets does not get
to hide the cost of its own grading. The judge answers with a JSON verdict and
a one-sentence reason, and an answer that is not a verdict is a fail.

```bash
python -m checks --modules 3 --mode live --spend-cap 0.50    # judge included, capped
```

## Offline costs nothing, and says so

An Offline run prices the recorded usage at registry prices and stops where
the same run would stop live, so you can rehearse a Spend Cap for free. It
reports that number as what the run would have cost, never as spend:

```
Would have cost $0.05522 live. Offline, nothing was spent.
Passed through module 3 of 3.
```

A live run reports `Spent $… of the $… Spend Cap.`

## The Case format

A Case is one JSON file in `company/cases/`. The runner loads it with
`load_case("its-id")`, or `load_case("path/to/case.json")` for a file anywhere
else. These are its fields:

| Field | Required | What it says |
|---|---|---|
| `id` | yes | The Case's name, in kebab-case. It is how suites and reports refer to it. |
| `customer` | yes | Who is asking. `customer.account_id` is the one account every Action on this Case may touch; the Actions refuse any other. |
| `opening_message` | yes | The customer's first turn, word for word. |
| `follow_ups` | no | Later customer turns, in order. The agent is called once per turn with the same `env`. |
| `actions` | yes | The Actions the Case offers the agent, in the order it is offered them ([ADR 0005](../../docs/adr/0005-cases-declare-their-actions.md)). Anything else is refused and logged as attempted. |
| `knowledge_base` | no | `true` offers the `search_knowledge_base` tool as well. Default `false`. |
| `expected_state_change` | yes | The Backend change a right answer makes, as dotted paths into the exported state and their values after the Case, for example `"accounts.acct_1002.invoices.inv_2002.refunded_cents": 920`. Nothing else may change. `{}` means nothing may change at all. |
| `forbidden_actions` | no | Actions the agent must not even attempt on this Case. |
| `judge_rubric` | no | What a good reply does, in one or two sentences, for the live-only judge. |
| `recording` | no | The file in `company/recordings/` its Offline run replays. Without one, the Case runs live only, and Offline reports its Checks as skipped. |
| `tags` | no | Free-form labels: `module`, `category`, and later `attack`. |
| `$comment` | no | Notes for the next person, such as the arithmetic behind the expected change. The runner ignores it. |

Two rules keep a Case honest. **Write the expected change from the policy, not
from what the agent did**: the arithmetic in `$comment`, then the number. A
Case written by copying a run's diff grades the agent against itself. And
**keep the expected change exact**: every path the right answer changes, and
no path it does not, because "nothing changes beyond the expected state" holds
the agent to exactly that list.

## From an incident to a Case

A production incident is a Case you did not have yet. The notebook walks
through one: a customer asked for a refund they could not have, and an agent
downgraded them to Free instead.

1. **Keep the customer's words.** Copy the message into `opening_message`, and
   any later messages into `follow_ups`. Note the account.
2. **Write the right Backend change from the policy.** Here: none, so
   `"expected_state_change": {}`.
3. **Forbid what went wrong.** `"forbidden_actions": ["issue_refund",
   "change_plan"]`, so even a refused attempt fails.
4. **Say what a good reply does** in `judge_rubric`, for the live judge.
5. **Run it.** Against the agent that caused the incident, it must fail; against
   the fixed agent, it must pass. A Case that passes both catches nothing.
6. **Record it** once, live, with `RecordingTransport` (see the
   [provider layer](../../llm/README.md)), and set `recording`. From then on
   it runs Offline, in CI, for free. The notebook's Case skips this step by
   keeping another Case's customer message and tools, so the reference agent
   sends exactly the requests that Case's recording already holds.

## The Graded suite

`checks/spine_3_graded.py` runs six Checks on four Cases that between them
use every Action built so far:

| Case | Actions the reference agent runs |
|---|---|
| `upgrade-to-pro` | `look_up_account`, `change_plan` |
| `downgrade-with-prorated-refund` | `look_up_account`, `change_plan`, `issue_refund` |
| `annual-refund-outside-window` | `look_up_account` (and no refund) |
| `upgrade-after-a-question` | `look_up_account`, `change_plan`, over two turns |

A module's run includes every earlier module's suite, so `--modules 3` runs
the Loop and Knowledge Checks too. Each Case runs once however many suites use
it.

## CI

`.github/workflows/ci.yml` runs the Offline Checks on every PR:

```yaml
- name: Offline Checks (every Spine module, no API key)
  run: python -m checks --mode offline
```

It names no module, so each later module's suite joins CI by existing. It
needs no secrets: the recordings stand in for the model, and a PR from a fork
gets the same result as one from the maintainer. The step fails the build on
any failing Check, which is what makes the suite a regression test and not a
report.

## Run the Checks on your agent

```bash
python -m checks --modules 3 --agent path/to/my_agent.py:run      # modules 1 to 3, Offline
python -m checks --modules 3 --agent path/to/my_agent.py:run \
    --mode live --spend-cap 0.50                                   # your key, judge included
```

Offline replays only the requests the reference agent sends, so your own
agent is graded live, or recorded once and replayed from then on.

## Try it

- Run the notebook, Offline, top to bottom.
- Write the Case for the refund-twice regression with a $5.00 refund, where
  the Backend lets both refunds run. Which Checks fail now?
- Turn one of your own production incidents into a Case, following the six
  steps above.
- Run the judge rubric live, then make your agent's replies worse on purpose.
  Does the judge notice? Does any state Check?

## Going further

The Appendix goes deeper on evaluation beyond one agent's Cases:
[evaluation](../../appendix/evaluation/) for metrics and judge design, and
[EvalOps](../../appendix/evalops/) for running evals as part of shipping.
