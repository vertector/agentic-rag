# Error Handling Reference — ingestion_pipeline_mcp

## ingest_data Errors

### FileNotFoundError
**Cause**: `file_path` does not exist.  
**Recovery**: Confirm document_parser_agent ran successfully and the path matches
the parser's output folder (e.g. `{stem}/documents.json`). Do not retry until fixed.

### ValueError (schema)
**Cause**: Inline Document data failed Pydantic validation, or JSON in file_path is malformed.  
**Recovery**: Surface the full error message. If inline: orchestrator must fix the Document
structure. If file: parser output may be corrupt — re-run parse.

### IngestorError (Qdrant)
**Cause**: Qdrant unreachable, collection missing, or upsert failed.  
**Recovery**:
1. Call `ingest_status` to check `qdrant_reachable`.
2. If false: Qdrant is down. Escalate — do not retry.
3. If true but still failing: collection may be in a bad state. Escalate.

### IngestorError (Redis GET/SET)
**Cause**: Redis unreachable or key operation failed.  
**Recovery**:
1. Call `ingest_status` to check `redis_reachable`.
2. If false: Redis is down. Without Redis, idempotency checking and Merkle integrity
   are disabled. Escalate — do not ingest blind.
3. If true but still failing: Redis command error. Escalate.

### IngestorError (Embedding)
**Cause**: fastembed or sentence-transformers failed (model load, OOM, CUDA error).  
**Recovery**: Escalate immediately. Embedding engine failure requires server restart.

### Partial errors (errors[] non-empty)
**Cause**: Some pages failed within a multi-page ingest; others succeeded.  
**Recovery**: Surface the per-page error list. Do NOT re-run the entire batch.
Identify failed page_indexes and re-run `ingest_data` with only those pages
(inline delivery) after fixing the underlying cause.

## ingest_audit Errors

### FAILED integrity
**Cause**: Merkle root reconstructed from Qdrant does not match Redis-stored root.  
**Recovery sequence**:
1. Call `ingest_sync` for the filename.
2. Call `ingest_audit` again.
3. If still FAILED: data tamper suspected. Escalate immediately with both audit results.

### IngestorError during audit
**Cause**: Qdrant scroll failed or Redis GET failed during root reconstruction.  
**Recovery**: Call `ingest_status`. If connectivity confirmed, escalate.

## ingest_configure Errors

### Re-initialisation failure
**Cause**: New Qdrant URL/Redis host unreachable, or model name invalid.  
**Recovery**: The old ingestor has already been torn down. The server is in a degraded state.
Orchestrator must call `ingest_configure` again with valid (original) settings.

## Redis Durability Warning
If `ingest_status` logs show "neither AOF nor RDB persistence", integrity checks are
at risk after Redis restart. Recommend enabling `appendonly yes` in Redis config.