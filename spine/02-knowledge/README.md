# Spine 2 · Knowledge

_Last verified: 2026-09-27_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnakhoa/llm-engineering-playground/blob/main/spine/02-knowledge/knowledge.ipynb)

**What your agent can do at the end:** resolve Cases whose right answer is in
Acme Notes' policies. It refunds a downgrade by exactly the prorated amount,
declines a refund the policy does not allow, and remembers what a customer said
two turns ago.

**What you build:** retrieval over the Knowledge Base, without an embeddings
API, and memory across a Case's turns. The loop from
[Spine 1](../01-loop/README.md) changes by two lines.

## What is new in the world

- **The Knowledge Base** (`company/knowledge_base/`) is Acme Notes' help-centre
  and policy articles, one markdown file each: the refund policy, plans and
  pricing, changing plans, billing, exporting notes.
- **A new Action, `issue_refund`** (`company/backend.py`). Invoices now sit on
  paid accounts, and the seed fixes "today" at 2026-09-20, so every refund
  calculation gives the same answer on every run.
- **Three new Cases** (`company/cases/`), each with a right answer that only
  the policy gives:

| Case | The customer asks | The right Backend change |
|---|---|---|
| `downgrade-with-prorated-refund` | "downgrade us, and refund whatever we're owed" | Free, and one refund of $9.20 |
| `annual-refund-outside-window` | a refund on an annual plan bought 45 days ago | nothing |
| `upgrade-after-a-question` | what Pro costs; then, next turn, "go ahead" | Pro |

## Retrieval, without embeddings

`company/knowledge.py` has one interface, `search(query, k)`, returning `Hit`s
(an article and a score), and two retrievers behind it.

```python
from company.knowledge import LexicalRetriever, load_articles

retriever = LexicalRetriever(load_articles())
retriever.search("Can I get a refund for this month?", k=2)
# [Hit(article=Article(id='refund-policy', ...), score=1.749), ...]
```

**`LexicalRetriever` is the default.** It is BM25: an article scores for each
query word it contains, more for a word that is rare across the Knowledge Base,
more for a word it repeats, a little less for being long. It is pure Python,
needs no key, and returns the same ranking every time, which is what an Offline
Check needs. Its weakness is the one you would guess: it only matches words.
"I want my money back" finds the billing article, only because it says refunds
go "back to the card", and never finds the refund policy.

**`EmbeddingsRetriever` is the upgrade**, behind the same interface. It ranks by
cosine similarity between embedding vectors, so a question can find an article
that shares its meaning but not its words. You pass the `embed` function:

```python
from company.knowledge import EmbeddingsRetriever

retriever = EmbeddingsRetriever(load_articles(), embed=my_embed)  # texts -> vectors
outcome = run_case(case, "flagship.knowledge:run", mode="live", retriever=retriever)
```

`my_embed` is your provider's embeddings endpoint, or a local embedding model.
The articles are embedded once, when the retriever is built; each search embeds
its query. That is also why it is not the Offline default: a live embeddings
call costs money, and a different embedding model can rank differently, which
changes the articles the agent reads, which changes the request, which is a
replay mismatch. The notebook runs it with a toy `embed` so you can compare the
two for free.

## How the agent reaches the Knowledge Base

A Case says whether the agent may use the Knowledge Base:

```json
"actions": ["look_up_account", "change_plan", "issue_refund"],
"knowledge_base": true,
```

When it may, `env.tools` holds a `search_knowledge_base` tool after the Case's
Actions, and `env.act` answers it from the retriever with the best two articles
in full. The model decides when to look a policy up, exactly as it decides when
to look an account up. A search reads and changes nothing, so it is not an
Action: it is in the transcript but not in `actions_attempted`. `env.knowledge`
is the retriever itself, for an agent that would rather retrieve up front.

## Each Case declares its Actions

Adding `issue_refund` to the Backend could have broken every Spine 1 Check.
A recording holds the whole request, tool list included, so one global tool
list would have put `issue_refund` into the upgrade-to-Pro request and turned
its replay into a mismatch. Fixing that would mean re-recording on a live key,
for every Case, every time a module adds an Action.

So each Case lists the Actions it allows, and the runner offers the agent only
those ([ADR 0005](../../docs/adr/0005-cases-declare-their-actions.md)).
`upgrade-to-pro` declares `look_up_account` and `change_plan`, so its request
and its recording are byte for byte what they were in Spine 1. A test proves it:
it checks `issue_refund` exists, then replays the old Case.

An agent that calls an Action its Case did not declare gets a refusal, and the
Outcome lists the call under `actions_attempted`, not `actions_executed`. That
is least privilege, and the Attacked module comes back to it as a defence.

## Policy belongs in the Action

The refund policy article is what the agent reads. `issue_refund` is what
enforces it, the same way Spine 1's Actions enforce whose account it is:

- **The refund window.** Up to 30 days after the invoice date, and never after.
- **Proration.** At most the unused days of the billing period, rounded down to
  the cent, less anything already refunded on that invoice.
- **The maximum refund.** $200.00 per refund, whatever proration gives.

A refund the policy does not allow is refused, and the refusal does not say
what the right amount would have been. The agent is meant to get the amount
from the policy, not by bargaining with the error message.

The Action only bounds the refund from above, though. A refund of $5.00 on an
invoice that owes $9.20 is inside every limit, so it runs, and it is still
wrong. That is what the Checks are for.

## Memory across a Case's turns

A Case can have follow-up turns. The runner sends them one at a time, calling
your agent once per turn with the same `env`, and records each reply on
`outcome.replies`. The Spine 1 loop starts every turn from an empty transcript,
so when the customer says "please go ahead and switch us over", it has no idea
which account they mean.

The Spine 2 agent, `flagship/knowledge.py`, is the Spine 1 loop with its
transcript moved into `env.memory`:

```python
def run(customer_turn, env):
    messages = env.memory.setdefault("transcript", [])  # memory: kept across turns
    messages.append(Message.user(customer_turn))

    for _ in range(env.step_limit):
        response = env.complete(messages, tools=env.tools, system=SYSTEM)
        ...  # the Spine 1 loop, unchanged
```

`env.memory` is a dictionary the runner creates empty for each Case and keeps
for all of its turns, so nothing leaks from one Case into the next. Keeping the
whole transcript is the simplest memory there is. It grows with every turn, and
every turn resends it; the Budgeted module comes back to what that costs.

## The system prompt stays in the recording

`flagship/knowledge.py` uses Spine 1's system prompt word for word. That is not
laziness. The system prompt is part of every recorded request, so a replay of a
changed system prompt is a mismatch that names `body.system`, and a new system
prompt means you re-record every Case it touches. That is the replay doing its
job: a recording that still passed after the prompt changed would be grading
behaviour nobody ran. Guidance that only one Case needs, such as "read the
refund policy before refunding", lives in the tool descriptions, which only the
Cases that offer that tool carry.

## The Checks

`checks/spine_2_knowledge.py` runs both of its Checks on each of the three
Cases:

- **Backend reaches the expected state.** Every change the Case expects
  happened, with exactly the expected value: the refund is 920 cents, not 900.
- **Nothing changes beyond the expected state.** No second refund, no stray
  plan change. Two refunds of $9.00 and $0.20 add up to the right total and
  still fail.

Both read Backend state. Neither reads the reply, so "I've refunded you $9.20"
without a refund that ran fails.

## Run the Checks on your agent

```bash
python -m checks --modules 2 --agent path/to/my_agent.py:run      # modules 1 and 2, Offline
python -m checks --modules 2 --agent path/to/my_agent.py:run \
    --mode live --spend-cap 0.50                                   # your key, capped
```

`--modules 2` runs module 1's Check too. Without `--agent`, the CLI grades the
reference agent at the latest module, `flagship.knowledge:run`. Offline replays
only the requests the reference agent sends, so your own agent is graded live,
or recorded once with `RecordingTransport` and replayed from then on.

## Try it

- Run the notebook, Offline, top to bottom.
- Give your Spine 1 agent memory and run it live on `upgrade-after-a-question`.
- Change your system prompt and replay any Case Offline. Read the mismatch.
- Write a Case where proration gives more than $200.00. What should the
  expected Backend change be, and what should the agent tell the customer?
- Swap in `EmbeddingsRetriever` with a real embedding model and run the three
  Cases live. Does the agent read the same articles?

## Going further

The Appendix goes deeper on both halves of this module:
[RAG](../../appendix/rag/) for chunking, vector stores and hybrid search, and
[memory](../../appendix/memory/) for memory that outlives one conversation. The
Flagship Agent needs none of that yet: five short articles fit in a prompt, and
a lexical retriever finds the right one.
