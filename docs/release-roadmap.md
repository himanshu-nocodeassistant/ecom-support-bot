# Release Roadmap

This project should be built as a sequence of small GitHub-visible releases. Each release should:

1. Add one meaningful capability
2. Keep the system runnable
3. Include a short note about what improved and what remains weak

## `v0.1.0-core`

Ship a working baseline:

- Minimal support API
- Keyword retrieval over a small knowledge base
- Order lookup tool
- Refund and ticket stubs
- Session memory
- Basic tests

Why this release matters:

- It proves the product loop end-to-end
- It gives you something real to demo and push immediately
- It creates stable seams for future upgrades

Known weaknesses to note in the release:

- Retrieval quality is shallow
- Tool routing is rule-based, not model-driven
- No streaming
- No observability
- Sample data only

## `v0.2.0-retrieval-upgrade`

Improve answer quality:

- Document chunking
- Metadata-aware ingestion
- Embeddings
- Hybrid retrieval
- Confidence scoring

What to note:

- Which queries improved
- Which failure modes still exist
- Whether fallback behavior got safer

## `v0.3.0-agent-tool-loop`

Make the assistant feel more agentic:

- Model-driven tool selection
- Multi-step tool chains
- Better refund handling
- Structured tool logs

What to note:

- Which complex flows now work in one turn
- Where tool misuse still happens
- What guardrails were added

## `v0.4.0-streaming-ui`

Improve the product feel:

- Frontend chat interface
- Streaming output
- Tool activity timeline
- Better UX states

What to note:

- First-token latency
- UX trust improvements
- Remaining rough edges in mobile/error states
