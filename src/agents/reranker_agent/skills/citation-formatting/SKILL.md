---
name: citation-formatting
description: >
  Formats CitationEnvelope objects from rerank_search results into human-readable
  inline citation strings suitable for direct insertion into LLM context windows,
  audit logs, or PDF deep-link overlays. Use when the orchestrator requests
  formatted citations, citation text rendering, human-readable result summaries,
  or structured citation lists from reranker output. Activate on phrases like
  "format citations", "render results", "citation strings", "human-readable",
  "audit trail", or "PDF coordinates".
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Citation Formatting Skill

## When to activate
- Orchestrator requests `include_citations_text=true` equivalent output from a prior
  `reranker:last_results` state value (without re-running the search).
- Orchestrator needs results rendered as an ordered citation list.
- Audit log generation over session score history (`reranker:session_scores`).
- PDF annotation prep — orchestrator needs bbox coordinates extracted.

## Steps

1. **Read results source**: use `reranker:last_results` from `session.state` if
   available. If absent, call `rerank_search` first (activate `rerank-execution` skill).

2. **For each result**, render the citation string using this template:
   ```
   [{index}] {citation.filename} · p.{citation.page_index} · chunk {citation.chunk_index} · score {final_score:.3f}
         CE={ce_score:.3f}  RRF={rrf_score:.4f}  sources={retrieval_sources}
         Merkle: {citation.version_root[:12]}...  hash: {citation.chunk_hash[:12]}...
         Audit: blob_cid={citation.blob_cid[:12]}...  corpus_id={citation.corpus_id}
         Time: {citation.timestamp}
   ```
   If `citation.bbox` is non-empty, append:
   ```
         bbox: [{x0}, {y0}, {x1}, {y1}]
   ```

3. **For PDF deep-link output**: extract `{filename}`, `{page_index}`, and `{bbox}`
   per result. Format as `{filename}#page={page_index}&rect={x0},{y0},{x1},{y1}`.
   Note: `page_index` is 1-indexed.

4. **For audit log output**: flatten each result to:
   ```json
   {
     "rank": N,
     "filename": "...",
     "page": N,
     "chunk": N,
     "final_score": 0.000000,
     "version_root": "...",
     "chunk_hash": "...",
     "blob_cid": "...",
     "corpus_id": "...",
     "timestamp": "ISO-8601"
   }
   ```

5. Return the formatted output to the orchestrator as a string (inline citation
   list) or JSON array (audit log).

## Gotchas

- `version_root` and `chunk_hash` in results are full 64-char SHA-256 hex.
  Display only the first 12 chars in human-readable output; always preserve
  the full value in audit JSON.
- `bbox` may be an empty list `[]` — check before formatting; omit the bbox
  line if empty.
- `timestamp` is ISO-8601 UTC (e.g. `2025-03-01T12:00:00+00:00`).
  Display as-is; do not reformat.
- `retrieval_sources` is a list: `["vector"]`, `["sparse"]`, or `["vector","sparse"]`.
  Render as a comma-joined string.
- Score precision: `final_score` and `ce_score` → 3 decimal places.
  `rrf_score` → 4 decimal places (smaller values, needs more precision).

## Constraints

- Never modify or re-rank results when formatting — preserve original ordering.
- Never truncate `chunk_hash` or `version_root` in audit JSON output.
- Out of scope: re-executing the search or modifying session state.