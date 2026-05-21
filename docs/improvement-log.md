# Improvement Log

Use this file after each release. Keep it short and honest.

## Current Snapshot

### Release

- Version: working tree after `v0.1.0-core`
- Date: 2026-05-21

### What changed

- Added: Supabase schema, repository layer, Olist loader CLI, Postgres import path
- Improved: order lookup now works with real Olist IDs and database-backed records
- Improved: knowledge retrieval can now read persisted documents and chunks from Supabase
- Removed: hard dependency on in-memory orders for order status checks

### Why it matters

- User impact: order questions can now resolve against real imported data instead of only demo IDs
- Engineering impact: storage and agent logic are now decoupled, which makes retrieval and tool upgrades much easier

### What is still weak

- Knowledge retrieval is now persisted, but still keyword/full-text based rather than embedding-based
- Refund/ticket flows are still deterministic and not yet model-driven

## Template

### Release

- Version:
- Date:

### What changed

- Added:
- Improved:
- Removed:

### Why it matters

- User impact:
- Engineering impact:

### What is still weak

- Weakness 1:
- Weakness 2:

### Next release focus

- Next target:
- Success signal:
