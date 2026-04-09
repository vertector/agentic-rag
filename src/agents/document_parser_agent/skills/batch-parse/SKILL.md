---
name: batch-parse
description: >
  Coordinates multi-document parallel parsing via parse_batch. Use when the
  orchestrator needs to parse 2 or more documents in a single operation, or
  when managing worker process count for memory-constrained environments. Activate
  on phrases like "parse multiple", "batch parse", "process these files", "parse
  in parallel", "parse all PDFs", or any request with a list of file paths.
  Handles pre-flight path validation, worker sizing warnings, result alignment,
  and per-file log persistence.
compatibility: Requires active document_parser_mcp server. Max 16 files per batch.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Batch Parse Skill

## When to activate
- Request includes 2 or more file paths to parse.
- Orchestrator asks to process a directory of documents.
- Orchestrator needs results aligned to input order for downstream ingestion.
- Worker count needs to be controlled (memory-constrained environment).

## Steps

1. **Validate paths pre-flight**: for each path in the list, confirm:
   - Extension is in `{.pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp}`.
   - Surface any unsupported extensions as a list before calling the server.
   - Do NOT call `parse_batch` until all extensions are valid.
   - Missing paths will be caught server-side, but reject upfront to save latency.

2. **Cap batch size**: maximum 16 files per call. If > 16 files, split into
   sub-batches and call `parse_batch` sequentially per sub-batch.

3. **Size worker count**:
   - ≤ 4 files or orchestrator is silent on workers → use `max_workers=null` (auto).
   - > 4 files AND `max_workers` not explicitly set → warn: "Each worker loads a
     PaddleOCRVL instance (~1–2 GB RAM/VRAM). Auto-sizing to {min(n, cpu_count)}.
     Set `max_workers` explicitly to limit memory usage."
   - `max_workers > 8` → require explicit orchestrator confirmation.

4. **Check cold-start**: if `parser:warm` is absent, warn about ~30s initialisation
   on the first worker.

5. **Call `parse_batch`** with validated `file_paths`, `max_workers`, `include_page_images`.

6. **On success**:
   - Results are aligned to `file_paths` input order — outer index = file index.
   - For each file result: append `{file, page_count, merkle_roots[]}` to
     `parser:session_parse_log` (cap at 100 entries).
   - Set `parser:warm = True`.
   - Save artifact `batch_{invocation_id}.json` via `tool_context.save_artifact`.
   - Return the aligned result array to the orchestrator.

7. **On partial failure** (some files returned error dicts within the result array):
   - Surface the per-file errors. Do not re-run the entire batch.
   - Suggest re-running `parse_document` on the failed files individually.

8. **On total failure** (top-level error JSON):
   - If OOM: escalate with recommendation to reduce `max_workers`.
   - Otherwise: surface message + suggestion, escalate after one failure.

## Gotchas

- `parse_batch` uses `ProcessPoolExecutor` internally. Each worker process calls
  `_init_worker()` at startup — this loads the full PaddleOCRVL model.
  With 8 workers and a 1.5 GB model: ~12 GB RAM required. Plan accordingly.
- Results array index alignment is guaranteed by `pool.map()` — rely on it,
  but verify `len(results) == len(file_paths)` before persisting.
- `configure_parser` called AFTER `parse_batch` workers are spawned will not
  affect the in-flight batch. Workers snapshot settings at spawn time.
- `include_page_images=True` in batch mode multiplies response size by N pages ×
  each page image. For a 10-page × 8-doc batch: expect potentially hundreds of MB.
  Avoid unless explicitly required.
- On macOS, `spawn` start method is default for ProcessPoolExecutor — model
  re-download is NOT triggered (weights are cached by HuggingFace).

## Constraints

- Maximum 16 files per `parse_batch` call (server enforced).
- Never call with `max_workers > 16` (server enforced at max 16).
- Require explicit orchestrator confirmation for `max_workers > 8`.
- Out of scope: ingestion, reranking, content Q&A.