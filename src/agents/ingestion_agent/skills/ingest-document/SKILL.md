---
name: ingest-document
description: >
  Executes document ingestion into the Qdrant Merkle vector store via ingest.
  Use when ingesting parsed documents from a filename (discovery via find_manifest), 
  a manifest.json file path, or an inline Document array. Activate on phrases 
  like "ingest", "store", "index", "load into qdrant", "upload to vector store", 
  "process documents", or any request supplying a filename, manifest.json path 
  or inline Document objects.
compatibility: Requires active ingestion_pipeline_mcp server. Qdrant + Redis must be reachable.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Ingest Document Skill

## When to activate
- Request supplies a `filename` (e.g. "bbs.pdf") but no manifest path.
- Request supplies a `file_path` to a manifest.json file.
- Request supplies an inline `documents` array of Document objects.
- Request asks to re-ingest (idempotency check needed).
- Large batch ingest (many pages) where progress logging matters.
- Category tagging required alongside ingestion.
- Knowledge Base (Corpus) selection required.

## Steps

1. **Resolve delivery mode** (mutually exclusive):
   - `filename` only → call `find_manifest` to resolve to a `file_path`.
   - `file_path` → confirm file exists and has `.json` extension. Never pass a raw PDF.
   - `documents` inline → confirm it is a list, not a single dict. Max 500 documents.
   - Both `file_path` and `documents` provided → return ValidationError.

2. **Check connectivity**: if `ingestor:connected` is absent in `session.state`,
   call `status` first to confirm Qdrant + Redis are reachable. Set
   `ingestor:connected = True` on success.

3. **Build params dict**:
   - `file_path` (resolved to absolute path) OR `documents`.
   - `corpus_id` and `category` if provided/pinned in state.
   - Do not add extra fields — `IngestInput` has `extra="forbid"`.

4. **Call `ingest`**.

5. **On success** (`ingested` or `skipped` > 0, no top-level `error` key):
   - Extract `filename` from the manifest.json metadata or from the inline first doc.
   - Persist to `session.state["ingestor:last_ingested_file"]`.
   - Persist full result to `session.state["ingestor:last_ingest_summary"]`.
   - Append `{file, ingested, skipped, errors}` to `ingestor:session_ingest_log`
     (cap at 100 entries).
   - Save artifact `ingest_{stem}_{invocation_id}.json`.
   - Set `session.state["ingestor:connected"] = True`.
   - Return the full result JSON.

6. **On partial errors** (`errors[]` non-empty but `ingested > 0`):
   - Surface the error list. Do NOT treat as total failure.
   - Recommend re-running `ingest` for the failed pages after root-cause fix.

7. **On total failure** (top-level `error` key):
   - Surface `message` + `suggestion`.
   - Read `references/error-handling.md` for error-class-specific recovery.

## Gotchas

- `process_document` is idempotent: if the Merkle root of the incoming page matches
  the Redis-stored root, the page is skipped (`skipped++`). Re-ingesting is always safe.
- Merkle root is deterministic from chunk content order — inserting/reordering chunks
  produces a new root and triggers a new version (soft-delete of old).
- `file_path` is resolved to absolute by the server's `_map_legacy_fields` validator.
  Pass the path as-is; do not pre-resolve.
- Inline `documents` max is 500 per call (server-enforced). Split larger sets.
- Empty-chunk pages are auto-skipped server-side (`skipped++`), not errored.
- The ingestor embeds chunks in batches of 100 using fastembed or sentence-transformers.
  First call after cold server start incurs model download if not cached.
- Collection name = `{COLLECTION_BASE}_{model_id}` — changing `model_name` via
  `configure` targets a completely different collection.
- Read `references/error-handling.md` if `ingest` returns any `"error"` key.

## Constraints

- Never pass a raw PDF, image, or non-JSON file as `file_path`.
- Never call with both `file_path` and `documents` simultaneously.
- Never infer or fabricate Document content — only pass data produced by document_parser_agent.
- Out of scope: parsing, reranking, purge, integrity audit.