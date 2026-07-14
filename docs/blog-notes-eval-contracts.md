# Blog Notes: When an Eval Rewards the Wrong Behaviour

## The incident

The support bot had a multi-intent adversarial metric called `multi_tool_rate`. It counted a case as
successful whenever the model used at least two distinct tools. The local result was 0.60 against a
0.80 CI floor.

At first glance that looked like a straightforward model-routing problem. Reading the failed cases
showed a more important problem: several requests bundled a valid support question with an action
the bot cannot perform, such as placing a new order, changing an address, or emailing a receipt.
One request also asked for a refund without supplying the reason required by the tool.

## The learning

More tool calls are not inherently better. For these cases, forcing a second tool call could reward
an unsafe action: issuing an under-specified refund or claiming an unsupported change succeeded.
The metric was measuring activity, not a safe customer outcome.

## The correction

Each multi-intent case now carries a behavioural contract:

- tools the agent must call for supported work;
- tools it must not call;
- whether it must ask for missing information; and
- whether it must escalate an unsupported part to a human.

The new metric is `multi_intent_contract_pass_rate`. It does not make the model deterministic; it
makes the definition of success explicit and reviewable in version control.

## A useful client conversation

This is the distinction worth showing in a client demo: an eval suite is not proof merely because
it has a score. First ask whether the score rewards the customer-safe behaviour you want. Then keep
the scenario, expectation, raw tool trace, model, and dataset version together so a later number can
be interpreted honestly.
