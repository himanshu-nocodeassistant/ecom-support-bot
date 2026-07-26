# Decision: When the bot opens a ticket

**Status:** Agreed, not yet implemented
**Owner:** Himanshu

---

## Why this exists

The bot has four tools. Plenty of things customers ask for are not among them. Until now the rule
was "create a ticket for each unsupported part", written into the system prompt in Phase 9.2.

Two problems with that rule.

First, the bot does not follow it. Measured across two eval suites: it declines correctly and then
never files the ticket. It happens in the multi-intent set (5 of 10 scenarios) and in agent fixture
af08. The customer is told no, and nobody at support ever hears about it.

Second, and more important: **the rule is wrong.** "Ticket for everything unsupported" would file a
ticket every time somebody asks for a discount. No human can act on that. A support queue full of
requests nobody can action is worse than no queue.

So the question is not "how do we make the bot file more tickets". It is "which requests deserve a
human, and how do we make that decision reliably".

## The rule

A ticket is for work a human can actually do. Nothing else.

| Customer asks for | Ticket? | Reason |
|---|---|---|
| A new order placed for them | Yes | An agent can place it |
| A receipt emailed | Yes | An agent can send it |
| An order cancelled | Only if it has not shipped | See below |
| A delivery address changed | Only if it has not shipped | See below |
| A discount | No | "We do not offer discounts" is a complete answer |

### The condition on cancel and address change

Both depend on one thing: **has the order left the warehouse?**

- **Not yet shipped** — an agent can still catch it. Open a ticket, include the order ID, mark it
  time-sensitive.
- **Already shipped or delivered** — nobody can change it. The parcel is with the carrier. Do not
  open a ticket, because there is nothing for a human to do. Explain that plainly and point to the
  process that does apply: a return once it arrives, or the carrier's own redirect service.

This keeps the rule honest in both directions. A customer who could still be helped is not dropped,
and a ticket is not filed for something nobody can act on.

## How the decision gets made

**Not by the bot.** The bot's judgment is what failed the first time, and a second attempt at
wording it more firmly would be the same bet at longer odds.

Instead `lookup_order` gains one computed field, decided in code:

```
status:        "processing"
delivered:     false
can_intercept: true          <- new
```

`can_intercept` is true only for `created`, `invoiced` and `processing`. It is false for `shipped`,
`delivered`, `canceled` and `unavailable`.

This matters because `delivered: false` covers two opposite situations — an order still sitting in
the warehouse and an order already on a truck. Asking the bot to tell those apart by reading the
status word is handing the policy decision straight back to the thing that got it wrong.

With the field in place the bot's job collapses to something it can do reliably: *if `can_intercept`
is true, open a ticket; if not, explain why not.* One rule, in one function, changed in one place.

### Consequence: the bot must look before it escalates

Today the bot decides whether to hand off from the wording of the question alone. Under this rule it
has to look the order up first, then decide. That is a real behaviour change, not a rewording.

It also means a cancel or address-change request with no order ID has to ask for the ID first. There
is no way to apply the rule without one.

## What the live data looks like

Out of the 10,000 imported Olist orders (the six `ORD-*` demo rows below are additional):

| status | count | can_intercept |
|---|---|---|
| delivered | 9,719 | no |
| shipped | 106 | no |
| canceled | 58 | no |
| unavailable | 55 | no |
| processing | 32 | **yes** |
| invoiced | 28 | **yes** |
| created | 2 | **yes** |

**62 orders out of 10,000 — 0.6% — can be intercepted.** The branch this whole decision turns on is
rare in the data. That is realistic, and it is also why it was never noticed: almost every order
anyone tests with is already delivered.

Demo orders now cover every branch, so both paths can be demonstrated and tested:

| order | status | can_intercept |
|---|---|---|
| ORD-1001 | shipped | no |
| ORD-1002 | delivered | no |
| ORD-1003 | processing | **yes** |
| ORD-1004 | invoiced | **yes** |
| ORD-1005 | created | **yes** |
| ORD-1006 | canceled | no |

## What this makes wrong in the current tests

Several existing tests encode "ticket for everything", which this decision retires. They have to be
corrected **before** the bot changes, not after. Changing both at once means a moving score with no
way to tell whether the bot improved or the target moved.

- **af08** — *"Can I change the delivery address after my order was shipped?"* currently expects a
  ticket. Under this rule it is a no-ticket case, because it has already shipped. Split it into two
  fixtures: one before shipping that expects a ticket, one after that expects a clear decline.
- **adv-mi-06** — the 20% discount scenario currently requires a ticket. It should not.
- **adv-mi-09** — the emailed-receipt part does deserve a ticket, so this one stands.
- **af09** — unrelated to this policy, but stale for the same underlying reason: it expects a refund
  to be processed without a stated reason, which Phase 9.2 deliberately blocked. Update it while
  the fixtures are open.

## Order of work

1. This document.
2. `can_intercept` in the repository layer and on `lookup_order`, with unit tests. No API calls.
3. Fixtures and contracts corrected to match the rule above. No API calls.
4. Prompt changed to key off `can_intercept`, then measured live.

Steps 1 to 3 are deterministic and verifiable offline. Only step 4 needs a live run, and by then
the target is fixed, so the number means something.

## What this is the beginning of

"Look up the state, apply a rule in code, let the bot explain the result" is the shape of the policy
engine in [`plans/roadmap.md`](../roadmap.md). Refund eligibility currently sits in an `if` statement
inside the tool, and the model effectively decides who gets money back. This is the same fix applied
to escalation first, because escalation is where it is measurably broken today. Worth building it in
the right shape now rather than as prompt wording to be redone later.
