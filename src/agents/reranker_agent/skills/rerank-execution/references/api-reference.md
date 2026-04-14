# API Reference — rerank_search

## Tool: rerank_search

### Input: RerankSearchInput

| Field | Type | Default | Constraints | Notes |
|-------|------|---------|-------------|-------|
| query | str | required | 1–2000 chars | Natural-language query |
| retrieval_top_k | int | 50 | 1–200 | Candidates per leg before CE |
| rerank_top_n | int | 5 | 1–50, ≤ retrieval_top_k | Final returned results |
| category | str | null | Must match ingestion category | e.g. "research", "medical" |
| corpus_id | str | null | Knowledge Base container ID | Scopes search to this corpus |
| version_root | str | null | 64-char SHA-256 hex | null = active version |
| collection_name | str | null | Exact Qdrant collection name | null = default |
| include_citations_text | bool | false | — | Adds pre-formatted citation strings |

### Output Schema

```json
{
  "query": "string",
  "result_count": 0,
  "results": [
    {
      "content": "string",
      "final_score": 0.000000,
      "ce_score": 0.000000,
      "rrf_score": 0.000000,
      "retrieval_sources": ["vector", "sparse"],
      "citation": {
        "filename": "string",
        "page_index": 1,
        "page_count": 0,
        "chunk_index": 0,
        "chunk_hash": "sha256hex",
        "version_root": "sha256hex",
        "category": "string",
        "bbox": [0, 0, 0, 0],
        "timestamp": "ISO-8601",
        "blob_cid": "sha256hex",
        "corpus_id": "string"
      },
      "citation_text": "string (only if include_citations_text=true)"
    }
  ],
  "pipeline": {
    "retrieval_top_k": 50,
    "rerank_top_n": 5,
    "alpha": 0.7,
    "ce_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bm25_active": true,
    "corpus_id": null,
    "version_root": null,
    "category": null
  },
  "warning": "string (only present when result_count == 0)"
}
```

### Error Envelope (any error condition)

```json
{
  "error": "ErrorClassName",
  "message": "Human-readable description",
  "suggestion": "Actionable recovery step"
}
```

## Tool: rerank_configure

### Input: RerankConfigureInput

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| alpha | float | null (no change) | 0.0–1.0; 0.7 = CE-dominant |
| cross_encoder_model_name | str | null | Triggers model reload |
| cache_size | int | null | 64–65536 |
| qdrant_url | str | null | Triggers ingestor rebuild |
| redis_host | str | null | Triggers ingestor rebuild |
| redis_port | int | null | 1–65535 |
| embed_model_name | str | null | Changes target collection |

### Output: JSON with `applied_updates` and `active_settings`.

## Tool: rerank_status

No input. Returns connectivity + config snapshot (see system_prompt.xml §4).

## Tool: rerank_cache_clear

No input. Returns `{"cleared": N, "message": "..."}`.