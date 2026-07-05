# Eval System

Custom evaluation pipeline for retrieval quality, agent behaviour, memory recall, and adversarial robustness. No external framework (e.g. RAGAS) is used — all metrics are hand-rolled.

---

## Directory layout

```
backend/eval/
  run.py                  # Main eval runner (all modes, LLM-judge, agent eval, benchmarks)
  check_regression.py     # CI regression gate
  memory_eval.py          # Memory recall evaluator
  generate_queries.py     # Synthetic query generator
  queries.json            # Gold query set (62 queries, hand-labelled with expected_document_id)
  adversarial_queries.json
  agent_fixtures.json
  memory_fixtures.json
  thresholds.json         # CI gate thresholds and best_mode pointer
  results/
    baseline.json         # Committed CI anchor — do not delete
    hybrid.json           # Latest run results per mode (gitignored except baseline)
    hybrid_rerank.json
    ...
```

---

## Running evals

```bash
# Single mode
python -m backend.eval.run --mode hybrid

# All modes side-by-side
python -m backend.eval.run --all-modes

# With LLM-as-judge (context relevance, scored on the top retrieved chunk)
python -m backend.eval.run --all-modes --llm-judge

# Agent fixture eval
python -m backend.eval.run --agent-eval

# Commit benchmark.md
python -m backend.eval.run --all-modes --benchmark

# Regression gate (used by CI)
python -m backend.eval.check_regression

# Save current results as new baseline
python -m backend.eval.check_regression --save-baseline
```

---

## Metrics

### Retrieval

| Metric | Description |
|---|---|
| `P@3 (doc)` | Precision@3 by document ID — primary signal |
| `R@3 (doc)` | Recall@3 by document ID |
| `NDCG@5` | Normalised discounted cumulative gain at rank 5 |
| `Hit@1` | Whether the top result is the correct document |
| `MRR` | Mean reciprocal rank |

> **Note:** title-based P@3/R@3 metrics exist but are deprecated (they penalise chunked retrieval unfairly — see `plans/decisions/retrieval-finding.md`).

### KwCorr (keyword-overlap answer correctness)

`avg_answer_correctness_kw` — fraction of expected keywords found in the concatenated top-3 retrieved content. **Not valid for cross-mode comparison.** Modes that return whole documents (e.g. `keyword`) search 5–10x more text per query than modes returning ~220-char chunks, so a higher score reflects text volume, not retrieval quality. `benchmark.md` publishes `avg_kw_context_chars` alongside it so a KwCorr gap is legible as a volume artifact. Only compare KwCorr across runs of the *same* mode over time (e.g. regression tracking), never across modes.

### Context relevance

| Metric | Description |
|---|---|
| `avg_context_relevance` | Cosine similarity between query and retrieved chunk embeddings |
| `avg_context_relevance_llm` | Claude-as-judge relevance score (0–1) for the top retrieved chunk vs. the query, via `--llm-judge` |

Both metrics score retrieval quality, not a generated answer — the retrieval eval loop never calls the agent, so there is no answer to judge. An earlier version of `avg_context_relevance_llm` was named `avg_answer_correctness_llm` and implied it scored a generated answer; it was renamed after an audit found the label overclaimed what was measured (`plans/decisions/eval-audit.md`). A prior `avg_faithfulness` metric was removed for the same reason — it was fed the top retrieved chunk as both the "context" and the "answer" being checked, so it could only ever score its own consistency with itself. Faithfulness (is the *agent's actual reply* grounded in what it retrieved) is a property of generation, not retrieval, and belongs in the agent-fixture eval path if reintroduced.

### Agent

| Metric | Description |
|---|---|
| `avg_tool_accuracy` | Correct tool selected for fixture scenarios |
| `avg_refusal_accuracy` | Correct refusal on out-of-scope queries |

### Memory

| Metric | Description |
|---|---|
| `memory_recall_rate` | Fraction of fixtures where all expected context fragments appear in the system prompt |

### Adversarial (absolute floor, not baseline-relative)

| Metric | Minimum |
|---|---|
| `injection_refusal_rate` | 0.80 |
| `clarification_rate` | 0.80 |
| `multi_tool_rate` | 0.80 |
| `oos_refusal_rate` | 0.80 |

---

## CI regression gate

`check_regression.py` runs after every eval. It compares current results against `results/baseline.json`.

- **Retrieval & agent:** fails if any gated metric drops more than `regression_max_drop` (10%) below baseline.
- **Memory:** fails if `memory_recall_rate` drops below `memory_recall_rate_min` (0.75) — absolute floor, not relative.
- **Adversarial:** absolute floor per metric, not baseline-relative.

To update the baseline after a deliberate improvement:

```bash
python -m backend.eval.check_regression --save-baseline
```

---

## Retrieval modes compared

| Mode | NDCG@5 | Hit@1 | P@3 (doc) |
|---|---|---|---|
| hybrid | 0.253 | 0.231 | 0.109 |
| hybrid+rerank | 0.288 | 0.288 | 0.141 |

`best_mode` in `thresholds.json` is currently `hybrid` (reranking disabled by default due to ~50 ms latency and 15× Voyage cost). See `plans/decisions/reranking.md`.

Hybrid search uses a **weighted linear combination** (not RRF):
```
score = 0.3 × ts_rank + 0.7 × (1 − cosine_distance)
```

---

## Query sets

- **Gold** (`queries.json`) — 62 hand-labelled queries with `expected_document_id`. Primary eval signal.
- **Synthetic** (`queries_synthetic.json`, gitignored) — LLM-generated queries for scale testing.
- **Adversarial** (`adversarial_queries.json`) — prompt injection, out-of-scope, multi-tool, clarification scenarios.
- **Agent fixtures** (`agent_fixtures.json`) — tool-use and refusal scenarios.
- **Memory fixtures** (`memory_fixtures.json`) — CustomerStore pre-population + system prompt fragment checks.
