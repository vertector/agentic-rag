---
name: rerank-execution
description: >
  Executes parameterised hybrid rerank searches against the Chanoch Clerk Qdrant
  collection using rerank_search. Use when constructing any retrieval call that
  requires non-default parameters: point-in-time Merkle snapshot pinning,
  cross-collection targeting, high-recall top-k settings, or category filtering.
  Activate on phrases like "point-in-time query", "pin to version", "search
  collection", "high recall rerank", or any request supplying a version_root.
compatibility: Requires active reranker_mcp FastMCP server on stdio transport.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Rerank Execution Skill

## When to activate
- Request includes a `version_root` hash (point-in-time query).
- Request specifies a non-default `collection_name`.
- Request requires `retrieval_top_k` > 100 (high-recall mode).
- Request combines category filter + version pin + custom top-n (complex parameterisation).
- Default `rerank_search` call returned 0 results and retry with adjusted params is warranted.
- Request specifies a `corpus_id` to limit search to a specific knowledge base.

## Steps

1. **Parse request fields**: extract `query`, `retrieval_top_k`, `rerank_top_n`,
   `category`, `version_root`, `collection_name`, `include_citations_text`, `corpus_id`.

2. **Apply state defaults**: read `reranker:active_category` and
   `reranker:version_root` from `session.state`; use as defaults for any
   fields not provided in the request.

3. **Validate cross-field constraint**: confirm `rerank_top_n <= retrieval_top_k`.
   If violated, auto-correct by setting `retrieval_top_k = max(retrieval_top_k, rerank_top_n * 3)`.

4. **For point-in-time queries**: pass `version_root` explicitly to `rerank_search`.
   After a successful call, persist the `version_root` to `session.state["reranker:version_root"]`
   so subsequent calls in the session stay pinned.

5. **For high-recall mode** (`retrieval_top_k` > 100): set `include_citations_text=false`
   to keep the response payload within MCP message size limits (~1 MB).

6. **Call `rerank_search`** with the fully assembled parameter dict.

7. **On success**:
   - Persist `query` → `session.state["reranker:last_query"]`.
   - Persist serialised `results` → `session.state["reranker:last_results"]`.
   - Append `{query, top_score, result_count}` to `session.state["reranker:session_scores"]`
     (cap array at 50 entries).
   - Save artifact `rerank_results_{session_id}.json` via `tool_context.save_artifact`.

8. **On empty results (`result_count == 0`)**:
   - Do NOT retry automatically with looser params.
   - Call `rerank_status` to verify connectivity.
   - Return warning + status to orchestrator.

9. **On error response** (JSON contains `"error"` key):
   - Extract `message` and `suggestion` fields.
   - Return structured error — do not retry.
   - Read `references/error-handling.md` for error-specific recovery guidance.

## Gotchas

- `rerank_top_n` must be ≤ `retrieval_top_k`. The MCP server returns a
  `ValidationError` if violated, not a Python exception.
- `version_root` values are full 64-char SHA-256 hex strings. Truncated roots
  (e.g. from log output) will cause silent misses, not errors.
- `collection_name` is case-sensitive and must match exactly the Qdrant
  collection created by the ingestor (format: `{COLLECTION_BASE}_{embed_model_slug}`).
- BM25 sparse leg is silently disabled if `rank_bm25` is not installed —
  check `pipeline.bm25_active` in the response to confirm dual-leg fusion ran.
- Cross-encoder batch size is fixed at 64 pairs. With `retrieval_top_k=200`,
  expect 3–4 CE forward passes. First call after cold start downloads weights (~90 MB).
- The CE score cache is in-process and per-server instance. Cache hits are
  only effective within the same running MCP server process.
- Read `references/error-handling.md` if the tool returns `"error": "IngestorError"`.
- Read `references/api-reference.md` for the full `rerank_search` response schema.
- Citations now include `blob_cid` and `corpus_id` for enhanced provenance.

## Constraints

- Never call `rerank_search` with a fabricated or inferred `version_root` —
  only use values retrieved from actual tool results or provided by the orchestrator.
- Never set `retrieval_top_k` above 200 (server-enforced cap). If the
  orchestrator requests more, inform them of the limit.
- Out of scope: ingestion, document parsing, answer synthesis.