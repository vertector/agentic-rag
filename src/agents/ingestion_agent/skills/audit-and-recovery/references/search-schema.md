# Search Schema Reference — search

## Input: SearchInput

| Field | Type | Default | Constraints |
|---|---|---|---|
| query | str | required | 1–1000 chars |
| category | str | null | Must match category set at ingest time |
| version_root | str | null | 64-char SHA-256 hex; null = active version |
| limit | int | 5 | 1–50 |

## Output Schema

```json
[
  {
    "score": 0.000000,
    "content": "chunk markdown text",
    "metadata": {
      "filename": "string",
      "page_index": 1,
      "page_count": 1,
      "category": "string"
    },
    "chunk_hash": "sha256hex",
    "version_root": "sha256hex",
    "timestamp": "ISO-8601"
  }
]
```

## history Output Schema

```json
{
  "filename": "string",
  "version_count": 2,
  "versions": [
    {
      "version_root": "sha256hex",
      "timestamp": "ISO-8601",
      "page_index": 1,
      "chunk_count": 5
    }
  ]
}
```
`versions[0]` = newest. `versions[-1]` = oldest (v1).

## audit Output Schema

```json
{
  "filename": "string",
  "page_index": 1,
  "integrity": "PASSED",
  "detail": "human-readable explanation"
}
```

## status Output Schema

```json
{
  "collection": "secure_rag_baai-bge-small-en-v1.5",
  "model_name": "BAAI/bge-small-en-v1.5",
  "embedding_engine": "fastembed",
  "qdrant_reachable": true,
  "redis_reachable": true,
  "qdrant_url": "http://localhost:6333",
  "redis_host": "localhost",
  "redis_port": 6379
}
```

## ingest Output Schema

```json
{
  "ingested": 3,
  "skipped": 1,
  "total_pages": 4,
  "errors": [],
  "collection": "secure_rag_baai-bge-small-en-v1.5",
  "warning": "optional, only present when errors non-empty"
}
```