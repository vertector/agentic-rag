---
name: audit-and-recovery
description: >
  Handles Merkle integrity auditing, Redis state recovery, version history
  retrieval, and permanent document purge for the Chanoch Clerk ingestion
  pipeline. Use when verifying ingested data integrity, recovering from Redis
  data loss, inspecting document version audit trails, performing point-in-time
  searches, or permanently deleting all document data. Activate on phrases like
  "verify integrity", "check merkle", "audit", "history", "version trail",
  "point-in-time", "redis recovery", "reconcile", "sync redis", "purge",
  "delete document", or "remove all versions".
compatibility: Requires active ingestion_pipeline_mcp server.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Audit and Recovery Skill

## When to activate
- Orchestrator requests Merkle integrity verification for a page.
- `audit` returned FAILED and recovery is needed.
- Redis data was lost (eviction, flush, restart without persistence).
- Orchestrator needs a list of version_root values for point-in-time search.
- Orchestrator requests permanent deletion of all data for a document.

## Steps — Integrity Audit

1. Call `audit` with `filename` (basename only) and `page_index` (1-indexed).
2. **PASSED**: return result and note that data is verified.
3. **FAILED**: proceed to Redis recovery sequence below.

## Steps — Redis Recovery

1. Call `sync` with the `filename`.
2. Call `audit` again for the same page.
3. **Now PASSED**: recovery successful. Persist updated root to
   `session.state["ingestor:version_roots"]`.
4. **Still FAILED**: data integrity compromised. Emit escalation immediately:
   `{"escalate": true, "agent": "ingestion_agent", "reason": "Integrity check FAILED after sync", ...}`

## Steps — Version History

1. Call `history` with `filename`.
2. On success: persist `versions[]` array to
   `session.state["ingestor:version_roots"][filename]` (cap 20 filenames).
3. Return history. If orchestrator needs point-in-time search: extract
   `version_root` from the desired version and persist to
   `session.state["ingestor:version_root"]` for the session.

## Steps — Point-in-Time Search

1. Confirm `ingestor:version_root` is set in state OR extract from `history`.
2. Call `search` with `version_root` forwarded.
3. Return results. Do not modify or summarise chunk content.

## Steps — Purge

1. **Before calling**: confirm `session.state["ingestor:purge_confirmed"] == True`.
   If not set: return the confirmation gate message and stop.
2. Call `purge` with `filename` and `confirm=True`.
3. On success: clear `ingestor:last_ingested_file` from state if it matches.
   Clear `ingestor:purge_confirmed` from state.
4. On error: surface message verbatim. Do NOT retry — partial purges require
   manual inspection of Qdrant and Redis.

## Gotchas

- `page_index` in `audit` is **1-indexed**.
- `sync` is non-destructive: it only re-seeds Redis keys from Qdrant.
  It will not recover data that was deleted from Qdrant itself.
- `history` returns versions newest-first. `history[-1]` is the oldest
  (v1) root. `history[0]` is the current active root.
- `purge` deletes BOTH leaf chunk points AND root anchor points from
  Qdrant (OR filter covers `metadata.filename` and `filename` fields).
  It also deletes ALL Redis keys matching `state:{model_id}:doc:{encoded_fn}:page:*`.
- After `configure` with a new `model_name`, the collection name changes.
  History and integrity operations on the old collection require reverting
  `model_name` to the original value.
- Read `references/search-schema.md` for the `search` response schema
  if the orchestrator needs to route results to reranker_agent.

## Constraints

- Never call `purge` without `session.state["ingestor:purge_confirmed"] == True`.
- Never retry a failed purge — escalate instead.
- Never fabricate version_root values — only use values from `history` results.
- Out of scope: ingesting new documents, parsing, reranking.