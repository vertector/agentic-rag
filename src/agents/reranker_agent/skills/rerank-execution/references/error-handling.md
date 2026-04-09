# Error Handling Reference — rerank-execution

## Error Classes from rerank_search / rerank_configure

### ValidationError
**Cause**: `rerank_top_n > retrieval_top_k` or field value out of bounds.  
**Recovery**: Auto-correct `retrieval_top_k = rerank_top_n * 3` and retry once.

### IngestorError
**Cause**: Qdrant or Redis unreachable; collection not yet created.  
**Recovery**:
1. Call `rerank_status` to isolate which service is down.
2. If `qdrant_reachable: false` — Qdrant is down or URL is wrong. Do not retry rerank_search.
3. If `redis_reachable: false` — Redis is down. Ingestor deduplication will fail; reranker may still work. Attempt rerank_search once.
4. If collection missing: escalate to orchestrator to run ingestion_agent first.

### RuntimeError (HybridReranker not initialised)
**Cause**: MCP server lifespan failed — sentence-transformers not installed or
CE model download failed.  
**Recovery**: Escalate immediately. The server must be restarted.

### Unexpected exception (any other error key)
**Recovery**: Surface `message` + `suggestion` verbatim. Escalate after one failure.

## Empty Results (not an error — result_count == 0)

Possible causes (check in order):
1. Collection is empty → run ingestion_agent.
2. `category` filter has no matching documents → retry without `category`.
3. `version_root` references a pruned snapshot → confirm root with ingestor history.
4. Query is too specific → suggest broader phrasing to orchestrator.

## Cache Issues

If CE scores appear stale after a CE model change:
- Call `rerank_cache_clear` (the skill should have done this automatically).
- Confirm `cleared > 0` in the response.

## Connectivity Degradation

`rerank_status` response fields to check:
- `qdrant_reachable` — false means no results are possible.
- `redis_reachable` — false means Merkle deduplication is disabled but reranking may still work.
- `bm25_active` — false means only single-leg (vector) retrieval; recall will be lower.