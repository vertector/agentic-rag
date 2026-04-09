# Error Handling Reference — document_parser_mcp

## Error Classes from parse_document / parse_batch

### FileNotFoundError
**Cause**: `file_path` does not exist on the server filesystem.  
**Recovery**: Surface `message` + `suggestion`. Ask orchestrator to correct path. Do not retry.

### PermissionError
**Cause**: Server process lacks read access to the file.  
**Recovery**: Surface both fields. Suggest: "Verify server process UID has read permission on the file." Do not retry.

### Base64DecodeError
**Cause**: Invalid base64 string, or decoded payload exceeds 50 MB.  
**Recovery**:
- If size > 50 MB: "File exceeds 50 MB limit. Save to disk and use `file_path` instead."
- Otherwise: "Base64 content appears malformed. Re-encode the file and retry."

### InvalidInput
**Cause** (parse_batch): `file_paths` is empty, or > 16 files.  
**Recovery**: Surface message. Adjust batch size.

### ValidationError (Pydantic)
**Cause**: Both `file_path` and `file_content_base64` provided, or unsupported extension.  
**Recovery**: Fix the param conflict per the suggestion field.

### Generic exception (VLM unreachable, corrupted file, OOM)
**Recovery**:
1. Read the `suggestion` field.
2. If suggestion mentions "VLM server": call `get_parser_settings` to check `vl_rec_backend`
   and `vl_rec_server_url`. Verify the server is running before retrying.
3. If suggestion mentions "corrupted file": escalate — parser cannot recover from this.
4. If OOM (out of memory in parse_batch): escalate with reduced `max_workers` recommendation.

## Cold-Start Latency (~30 seconds)
Not an error. Expected on first parse_document call after server start.  
If orchestrator reports timeout: instruct them to retry once with a longer timeout.  
After first success, set `parser:warm = True` in state.

## Empty Results (0 pages)
Not an error from the MCP server. Possible causes:
1. File is empty or has no extractable content.
2. Wrong `pipeline_version` for the file format.
3. `use_layout_detection=False` with a complex multi-column layout (no blocks detected).

Recovery: retry with `use_layout_detection=True` + default `pipeline_version="v1.5"`.

## Settings Drift (configure_parser + parse_batch race)
parse_batch workers use the PipelineSettings snapshot from when they were spawned.  
If configure_parser was called after batch workers started, the new settings will
NOT apply to the in-flight batch. Wait for batch completion before reconfiguring.