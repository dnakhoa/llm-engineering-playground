# LLM Engineering Playground

An open-source course that teaches engineers to take an agent to production by building one agent, step by step, and hardening it for evals, security and cost.

Thesis: _Your agent isn't done until it's graded, attacked and budgeted._

## Language

### Audience

**Reader**:
A software engineer who already calls LLM APIs and must now ship an agent to production.
_Avoid_: student, beginner, learner (as the primary audience)

### Structure

**Flagship Agent**:
The single support/operations agent the Reader builds from scratch and carries through the whole course. It answers from a knowledge base and takes actions through tools. Every Spine module changes it.
_Avoid_: capstone, demo, sample app

**Spine**:
The ordered sequence of modules that each add one production capability to the Flagship Agent: loop, knowledge, graded, observed, attacked, budgeted, shipped.
_Avoid_: track, path, curriculum

**Appendix**:
Modules kept as reference material because they do not change the Flagship Agent.
_Avoid_: bonus, extra, legacy modules

**Drop**:
A dated, monthly public update to the course with a fixed, small shape.
_Avoid_: release, version, changelog entry

### The agent's world

**Company**:
The fictional business the Flagship Agent works for: Acme Notes, a note-taking SaaS with Free, Pro and Team plans. It owns the Knowledge Base and the Backend.
_Avoid_: client, customer org, tenant

**Case**:
One customer request, from arrival to resolution. A Case declares which Actions it lets the agent use; the agent is offered only those.
_Avoid_: ticket, conversation, session, request

**Knowledge Base**:
The Company's documents the Flagship Agent retrieves from. The Attacked Checks may plant hostile articles in it.
_Avoid_: corpus, docs, vector store

**Backend**:
The Company's deterministic mock systems (accounts, plans, billing) that the Flagship Agent changes through Actions. Checks assert on its state.
_Avoid_: API, sandbox, database

**Action**:
A tool call that changes Backend state, such as issuing a refund or changing a plan.
_Avoid_: side effect, write tool, operation

### Hardening

**Check**:
A runnable pass/fail test the Reader runs against their own Flagship Agent at the end of a Spine module.
_Avoid_: exercise, quiz, assignment

**Graded**:
The Flagship Agent has passed an eval suite that catches behavioural regressions.
_Avoid_: tested, evaluated (as a milestone name)

**Attacked**:
The Flagship Agent has withstood a fixed set of adversarial inputs aimed at its tools and knowledge.
_Avoid_: secured, hardened, red-teamed (as a milestone name)

**Budgeted**:
The Flagship Agent keeps its cost per resolved Case within a stated limit.
_Avoid_: optimized, cheap

**Spend Cap**:
The hard limit on model spend for one run of Checks on the Reader's own key. A run stops when it reaches the limit.
_Avoid_: budget (reserved for Budgeted), quota, limit

**Offline Checks**:
The subset of Checks that replay recorded model responses, so they need no API key.
_Avoid_: mock mode, dry run

**Scoreboard**:
The dated table, published with each Drop, of quality and cost per resolved Case for each provider and model, measured on this course's own Cases and Checks. It is not a general benchmark.
_Avoid_: leaderboard, benchmark, ranking

**Grader**:
The Claude Code skill that runs a Reader's Checks and explains each failure.
_Avoid_: tutor, judge (reserved for LLM-as-judge inside Checks)
