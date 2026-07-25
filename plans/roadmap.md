# Roadmap

Work that has not started yet. For what past releases did, see
[`docs/changelog.md`](../docs/changelog.md).

---

## Phase 10 — Production hardening

Every release so far added a capability and then measured what it broke. This one does the same
thing to the parts of the system that were left alone on purpose while the AI layer was being built.

The gaps below came out of a security review of the running code. None of them change answer
quality, which is why they were safe to defer. All of them matter before this runs anywhere real.

The rule for this phase is the same as every other phase: **measure the failure first, then fix it,
then publish the difference.** A fix with no before-number is not evidence of anything.

### 10a — Concurrency

Two people using the bot at once is not currently a tested state.

`PostgresCustomerStore` and `PostgresConversationStore` each hold one connection for the life of the
process, with no pool, no health check, and no reconnect. Meanwhile `_repo_for_mode()` in `agent.py`
opens a **new** repository, and a new connection, on every single request for phases 2 through 4. So
the system manages to be both contended and leaky at the same time.

- [ ] Write a load script: N chat sessions at once, mixed question types
- [ ] Record the failure first — p95 latency, error rate, and open connection count at 1, 2, 10 and
      50 sessions at once. Commit the table before changing any code
- [ ] Replace the single connections with a real pool
- [ ] Cache the per-mode repository instead of building one per request
- [ ] Re-run the load script and publish the difference
- [ ] Add a p95 latency floor to the CI gate

### 10b — Cap the tool loop

The agent loop is a `while True` with no iteration limit. A prompt that talks the model into calling
tools over and over has nothing stopping it, and the bill is the only signal.

- [ ] Cap iterations, and return a clear hand-off message when the cap is hit
- [ ] Add a token and cost budget per conversation
- [ ] Add an adversarial fixture that tries to cause a loop, and gate on it

### 10c — Bound what is held in memory

`SESSION_MEMORY` and the in-memory stores grow forever. Nothing is ever evicted. Given enough time,
the process runs out of memory.

- [ ] Add a time limit and a maximum size to session storage
- [ ] Add a test that fills past the limit and checks that old sessions are dropped

### 10d — Stop leaking internals to the client

The streaming error handler sends the raw exception text to the browser. That can include database
connection details and file paths.

- [ ] Send a generic message to the client, log the real error on the server with an id
- [ ] Add a test asserting no exception text reaches the client

### 10e — Validate what comes in

`session_id`, `mode` and `message` are accepted as-is. `session_id` can be any length and is used as
a dictionary key. `mode` is not checked against the list of real modes. `message` has no size limit,
so a very large body goes straight to the model.

- [ ] Length and format limits on every field, enforced by the request model
- [ ] `mode` restricted to the modes that exist
- [ ] Server-generated session ids, so one person cannot read another's conversation by guessing

### 10f — Clean memory facts before they reach the prompt

This is the most interesting one, because the repo already has the tool to prove it.

Customer facts are placed above the system prompt with no checks. If a hostile instruction gets
saved as a fact in one session, it is silently re-injected into every later session for that
customer. The fact extractor does not look for this.

- [ ] Write an adversarial fixture: session 1 saves a hostile instruction as a fact, session 2 checks
      the agent still behaves. **Watch it fail first**
- [ ] Filter and wrap facts so stored text cannot read as instructions
- [ ] Gate the fixture in CI

### 10g — Small deferred items

- [ ] Cache settings instead of reading `.env` on every request
- [ ] Escape every value written into the eval dashboard HTML

### Acceptance criteria

- [ ] Load numbers before and after, committed
- [ ] Tool loop cannot run past its cap
- [ ] Session storage has a ceiling
- [ ] No internal error text reaches a client
- [ ] Poisoned-fact fixture fails on the old code and passes on the new
- [ ] A decision doc in `plans/decisions/` covering what was measured

### Still out of scope, on purpose

Authentication, rate limiting, and the wide-open CORS setting stay out. They belong to whatever sits
in front of this service — a gateway, a token check, an auth service. Building them here would bury
what each phase is meant to show. Verifying that someone owns the email they typed belongs to that
same layer.

This is a deliberate boundary, not an oversight. It stays written down.

---

## Later, not scheduled

Rough order of value, not yet planned in detail.

**Make it runnable in one command.** Right now the full version needs a Supabase project, three API
keys, migrations applied by hand, and a CSV import. Most people will not do that. A compose file
with Postgres and pgvector, migrations applied on start, and a small seeded dataset would fix it.
Committing the 15 pre-computed document embeddings would let hybrid retrieval work with no Voyage
key, so the no-key path stops being the weakest one.

**Policy as code.** Refund rules currently live in an `if` statement, and the model effectively
decides who gets money back. A separate rules layer should decide, and the agent should only explain
the decision. That also opens up a new thing to measure: how often the reply matches what the rules
actually said, including when the customer pushes back.

**More of the real job.** Four tools is a thin slice of support work. Tracking a parcel, changing an
address inside a cutoff window, cancelling, exchanging, pausing a subscription, generating a return
label. With enough tools that have real preconditions, tool choice becomes a genuine test.

**Numbers a business would ask for.** How many conversations end without a human. How much each one
costs. How long they take. These sit next to the retrieval metrics, not instead of them.

**A real hand-off.** `create_ticket` returns a made-up id and nothing else happens. It should write a
ticket, show up in a queue, and carry a summary of the conversation for whoever picks it up.
