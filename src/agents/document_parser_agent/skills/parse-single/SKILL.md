---
name: parse-single
description: >
  Executes a single-document parse via parse_document with full parameter
  resolution. Use when parsing requires non-default settings: base64 file
  delivery, per-call VLM backend override, layout threshold tuning, pixel
  budget adjustment, or any combination of PipelineSettings overrides.
  Activate on phrases like "parse with base64", "override settings for this
  parse", "tune layout threshold", "change VLM model for this file", "use
  chart recognition", or any single-file parse request that specifies non-
  default PipelineSettings fields.
compatibility: Requires active document_parser_mcp FastMCP server on stdio transport.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Parse Single Skill

## When to activate
- File is provided as base64 bytes + filename (not a local path).
- Request specifies any non-default PipelineSettings field for this call only.
- Request uses an external VLM backend (`vl_rec_backend` not `local`/`native`).
- First parse call after server startup (cold-start latency management).
- Parse returned unexpected results and retry with adjusted layout or VLM params.

## Steps

1. **Resolve delivery mode** (mutually exclusive):
   - `file_path` provided → use path directly. Verify extension is in
     `{.pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp}` before calling.
   - `file_content_base64` + `filename` → base64 mode. Size cap: 50 MB decoded.
   - Both provided → error. Ask orchestrator to pick one.

2. **Check cold-start state**: if `parser:warm` is absent in `session.state`,
   prepend a cold-start warning to the response:
   "PaddleOCRVL initialises on first call — expect ~30s latency."

3. **Assemble per-call overrides** (only include fields the caller explicitly set):
   Candidate override fields — see `references/pipeline-settings.md` for defaults.
   Do NOT include fields that were not in the request; they will be inherited
   from the server's current settings.

4. **Call `parse_document`** with the resolved params dict.

5. **On success**:
   - Set `session.state["parser:warm"] = True`.
   - Set `session.state["parser:last_parsed_file"]` to `file_path` or `filename`.
   - Set `session.state["parser:last_results"]` to the JSON result string.
   - Append `{file, page_count, merkle_roots}` to `parser:session_parse_log`.
   - Save artifact `parsed_{stem}_{invocation_id}.json` via `tool_context.save_artifact`.
   - Return the full result JSON to the orchestrator.

6. **On empty result (0 pages)**:
   - Do NOT retry automatically.
   - Check if file format is supported. Return warning with suggestion.

7. **On error JSON** (contains `"error"` key):
   - Extract `message` and `suggestion`. Surface verbatim.
   - Read `references/error-handling.md` for error-class-specific recovery.

## Gotchas

- `file_path` is resolved to an absolute path by the server's `_map_legacy_fields`
  validator. You don't need to resolve it — pass the path as-is.
- `prompt_label` override is dangerous: it forces ALL layout blocks through a
  single prompt and breaks Markdown table extraction. Warn before setting.
- `include_page_images=True` returns base64 PNGs per page and can make the
  response several MB. Only request it when the orchestrator explicitly needs
  page thumbnails.
- PaddleOCRVL v1.5 uses `pipeline_version="v1.5"` — if `parse_batch` is
  called concurrently with `configure_parser`, the batch workers may use
  stale settings (process isolation). Serialise configure → parse.
- `vl_rec_backend="local"` uses the bundled PaddleOCR weights. External
  backends (`vllm-server`, `mlx-vlm-server`) require a running server at
  `vl_rec_server_url`. Check connectivity before calling.
- If `layout_threshold` is lowered below 0.2, expect many false-positive
  layout boxes (noise regions detected as content). Start at 0.3 default.
- Read `references/pipeline-settings.md` for the full default values table.
- Read `references/error-handling.md` if the tool returns any `"error"` key.

## Constraints

- Never pass `file_path` and `file_content_base64` simultaneously.
- Never fabricate chunk content, bbox values, or Merkle roots.
- Never retry a failed parse more than once without orchestrator instruction.
- Out of scope: ingestion, reranking, content Q&A.