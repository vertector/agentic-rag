"""
src/agents/document_parser_agent/callbacks.py
==============================================
All six ADK callback hooks for the document parser agent.

Deliberately has NO import of LlmAgent or MCPToolset — it only depends on
ADK callback types, the genai types library, and stdlib. This keeps it
independently unit-testable without spinning up a full agent.

Callback responsibilities
-------------------------
before_agent  — fast-path out-of-scope deflection; session open log
after_agent   — session close audit log
before_model  — prompt-injection guard; base64 payload size gate; tool log
after_model   — per-turn token accounting; running session total
before_tool   — SHA-256 cache read; confirmation gate for parser_configure
after_tool    — cache write; warm-up flag; per-file parse summary; config update
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import pathlib
import tempfile
from datetime import datetime, timezone
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tools whose side-effects require explicit user confirmation before execution.
# The agent instruction teaches the model to set `confirmed:<tool_name> = True`
# in session state before the gate below will allow the call through.
REQUIRES_CONFIRMATION: frozenset[str] = frozenset({"configure_parser"})

# Tools whose results must never be cached (always have write side-effects).
NON_CACHEABLE: frozenset[str] = frozenset({"configure_parser"})

# Payload size ceiling — base64-encoded PDFs can be enormous; reject early
# rather than burning tokens on a request the MCP server will reject anyway.
MAX_REQUEST_CHARS = 500_000

# Keyword triggers that indicate a misrouted request.
# Checked in before_agent to avoid a full LLM round-trip for obvious misfires.
OUT_OF_SCOPE_TRIGGERS = (
    "ingest", "vector store", "qdrant", "redis",
    "rerank", "semantic search", "search results",
    "embed", "embedding",
)

# Prompt injection patterns — block before they reach the model.
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "disregard your system prompt",
    "act as a different",
    "you are now",
    "jailbreak",
    "pretend you are",
)


# ---------------------------------------------------------------------------
# 1. before_agent_callback
# ---------------------------------------------------------------------------

def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Gate: run before any agent work begins for this turn.

    - Logs session + invocation IDs for distributed tracing.
    - Stages any inline_data file attachments to a temp directory so the
      agent can pass a local file_path to parse_document instead of base64.
    - Fast-path deflects obviously out-of-scope requests before spending an
      LLM call (saves ~500ms + tokens per misrouted message).

    Returns:
        types.Content to short-circuit the agent, or None to proceed normally.
    """
    logger.info(
        "[PARSER AGENT START] session=%s invocation=%s",
        callback_context.session.id,
        callback_context.invocation_id,
    )

    # Stage any binary attachments to local temp files
    _stage_attachments(callback_context)

    user_text = _extract_user_text(callback_context).lower()
    for trigger in OUT_OF_SCOPE_TRIGGERS:
        if trigger in user_text:
            logger.info(
                "[OUT-OF-SCOPE] Fast-exit on trigger '%s' — no LLM call made.", trigger
            )
            return types.Content(
                role="model",
                parts=[types.Part(text=(
                    f"'{trigger}' is outside my scope — I handle document parsing only. "
                    "Route this to the Ingestion Agent or Reranker Agent."
                ))],
            )

    return None


# ---------------------------------------------------------------------------
# 2. after_agent_callback
# ---------------------------------------------------------------------------

def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Audit: log session close. No output modification.

    Could append compliance footers or post-turn summaries here in future.
    """
    logger.info(
        "[PARSER AGENT END] session=%s invocation=%s token_total=%s",
        callback_context.session.id,
        callback_context.invocation_id,
        callback_context.state.get("session_token_total", 0),
    )
    return None


# ---------------------------------------------------------------------------
# 3. before_model_callback
# ---------------------------------------------------------------------------

def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    Guardrail: runs immediately before the LLM API call.

    - Scans all text parts for prompt injection patterns.
    - Rejects payloads exceeding MAX_REQUEST_CHARS (catches stray base64 blobs).
    - Injects a note about staged file paths so the LLM uses local paths.
    - Logs available tool names at DEBUG level for traceability.

    Returns:
        LlmResponse to override the LLM call entirely, or None to proceed.
    """
    # Inject staged-file note so the model knows the local path to use
    staged: dict = callback_context.state.get("staged_files", {})
    if staged:
        lines = ["[Side-channel Note — use these local paths for parsing]:"]
        for original_name, local_path in staged.items():
            lines.append(f"  • '{original_name}' → {local_path}")
        lines.append("Call parse_document(file_path=<path above>) directly — do NOT re-encode as base64.")
        note = "\n".join(lines)
        llm_request.append_instructions([note])
        logger.info("[STAGE NOTE] Injected paths for %d staged file(s).", len(staged))

    full_text = _extract_request_text(llm_request)

    # Injection guard
    lower = full_text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            logger.warning("[GUARDRAIL] Injection pattern blocked: '%s'", pattern)
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="I cannot process that request.")],
                )
            )

    # Payload size guard
    if len(full_text) > MAX_REQUEST_CHARS:
        logger.warning(
            "[GUARDRAIL] Payload too large (%d chars) — rejecting.", len(full_text)
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=(
                    "The request payload is too large. "
                    "If you're passing base64 file content, save it to disk and "
                    "use `file_path` instead."
                ))],
            )
        )

    tool_names = list(llm_request.tools_dict.keys())
    logger.debug("[LLM CALL] tools_available=%s", tool_names)
    return None


# ---------------------------------------------------------------------------
# 4. after_model_callback
# ---------------------------------------------------------------------------

def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """
    Observability: log token usage and accumulate a session-level total.

    The running total in `session_token_total` is used by:
    - after_agent_callback (logs it on close)
    - The orchestrator agent (reads it for pipeline-level cost budgeting)

    Returns:
        None — no response modification.
    """
    meta = llm_response.usage_metadata
    if meta:
        prompt     = meta.prompt_token_count or 0
        completion = meta.candidates_token_count or 0
        total      = meta.total_token_count or 0

        logger.info(
            "[TOKENS] prompt=%d completion=%d total=%d session=%s",
            prompt, completion, total, callback_context.session.id,
        )

        prev = callback_context.state.get("session_token_total", 0)
        callback_context.state["session_token_total"] = prev + total

    return None


# ---------------------------------------------------------------------------
# 5. before_tool_callback
# ---------------------------------------------------------------------------

def before_tool_callback(
    tool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    """
    Cache read + confirmation gate — runs before every MCP tool call.

    Cache strategy
    --------------
    Key: SHA-256(tool_name + JSON-sorted args), stored in session state.
    Hit: return cached result immediately — the MCP server round-trip is skipped.
    Miss: return None and let the call proceed normally.

    Tools in NON_CACHEABLE are never cached (they always mutate state).

    Confirmation gate
    -----------------
    Tools in REQUIRES_CONFIRMATION are blocked until the session state key
    `confirmed:<tool_name>` is set to True. The agent instruction teaches the
    model to ask the user, set the key, then retry the tool call.

    Returns:
        dict  — short-circuits the tool call with this result (cache hit or gate block)
        None  — proceed with the actual tool call
    """
    logger.info("[TOOL PRE] %s args_keys=%s", tool.name, list(args.keys()))

    # ── Confirmation gate ────────────────────────────────────────────────────
    if tool.name in REQUIRES_CONFIRMATION:
        confirmed = tool_context.state.get(f"confirmed:{tool.name}", False)
        if not confirmed:
            logger.warning("[GATE] %s blocked — awaiting confirmation.", tool.name)
            return {
                "status": "error",
                "error": "ConfirmationRequired",
                "message": (
                    f"'{tool.name}' changes active pipeline settings for all subsequent "
                    "parse calls this session. Please confirm this is intentional."
                ),
            }

    # ── Cache read ────────────────────────────────────────────────────────────
    if tool.name not in NON_CACHEABLE:
        key = _make_cache_key(tool.name, args)
        hit = tool_context.state.get(key)
        if hit is not None:
            logger.info("[CACHE HIT] %s key=...%s", tool.name, key[-8:])
            return hit

    return None


# ---------------------------------------------------------------------------
# 6. after_tool_callback
# ---------------------------------------------------------------------------

def after_tool_callback(
    tool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """
    Cache write + session state bookkeeping — runs after every MCP tool call.

    On parser_parse success
        · Write result to the SHA-256 cache.
        · Set `app:parser_warmed_up = True` (persists across sessions in prod).
        · Write `temp:parsed_{stem}` summary for fast orchestrator look-up.

    On parser_parse_batch success
        · Set `app:parser_warmed_up = True`.

    On parser_configure success
        · Persist new settings to `parser_last_config` (read by prompts.py).
        · Clear the `confirmed:parser_configure` gate flag.
        · Evict the stale `parser_status` cache entry.

    Returns:
        None — result is not modified; all side-effects go to session state.
    """
    parsed = _parse_tool_result(tool_response)
    is_error = isinstance(parsed, dict) and "error" in parsed

    logger.info(
        "[TOOL POST] %s error=%s",
        tool.name,
        parsed.get("error") if isinstance(parsed, dict) else None,
    )

    # ── Cache write ───────────────────────────────────────────────────────────
    if tool.name not in NON_CACHEABLE and not is_error:
        key = _make_cache_key(tool.name, args)
        tool_context.state[key] = tool_response
        logger.debug("[CACHE WRITE] %s key=...%s", tool.name, key[-8:])

    # ── parse_document bookkeeping ──────────────────────────────────────────────
    if tool.name == "parse_document" and not is_error:
        tool_context.state["app:parser_warmed_up"] = True

        raw_path    = args.get("file_path") or args.get("filename", "unknown")
        stem        = pathlib.Path(raw_path).stem
        pages       = parsed if isinstance(parsed, list) else []
        # Snapshot the file fingerprint at parse time so get_parser_cache_stats
        # can compare it against the current fingerprint and flag stale entries.
        fingerprint = _file_fingerprint(raw_path) if args.get("file_path") else "base64"

        tool_context.state[f"temp:parsed_{stem}"] = {
            "file_path":   raw_path,
            "page_count":  len(pages),
            "output_path": f"{stem}/documents.json",
            "parsed_at":   _utcnow(),
            "fingerprint": fingerprint,   # mtime:size at parse time
        }
        logger.info(
            "[PARSE DONE] %s → %d pages → %s/documents.json (fp=%s)",
            stem, len(pages), stem, fingerprint,
        )

    # ── parse_batch bookkeeping ────────────────────────────────────────
    if tool.name == "parse_batch" and not is_error:
        tool_context.state["app:parser_warmed_up"] = True

    # ── configure_parser bookkeeping ─────────────────────────────────────────
    if tool.name == "configure_parser" and not is_error:
        if isinstance(parsed, dict):
            tool_context.state["parser_last_config"] = parsed.get("current_settings", parsed)

        # Clear confirmation gate — must be re-confirmed for next configure call
        tool_context.state.pop("confirmed:configure_parser", None)

        # Evict stale get_parser_settings cache — settings just changed
        stale_key = _make_cache_key("get_parser_settings", {})
        tool_context.state.pop(stale_key, None)

        logger.info("[CONFIG] parser_last_config updated; status cache evicted.")

    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stage_attachments(callback_context: CallbackContext) -> None:
    """
    Scan the latest user event for inline_data parts (file attachments) and
    save each one to a temporary file. The mapping of original_name -> local_path
    is stored in session state under 'staged_files'.

    Why stage rather than pass base64?
    - The MCP server accepts file paths, not raw bytes.
    - Passing a 10 MB PDF as base64 in a tool call blows past token limits.
    - A local path is 1–3 tokens; base64 of the same file is ~13 million tokens.
    """
    try:
        staged: dict = dict(callback_context.state.get("staged_files") or {})
        events = callback_context.session.events or []
        if not events:
            return

        # Only process the most recent user event
        for event in reversed(events):
            author = getattr(event, "author", None)
            if author != "user":
                continue
            content = getattr(event, "content", None)
            if not content:
                break

            for part in content.parts or []:
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue

                mime = getattr(inline, "mime_type", "") or ""
                data = getattr(inline, "data", None)
                display_name = getattr(inline, "display_name", None)

                if not data:
                    continue

                # Derive file extension from mime type
                ext_map = {
                    "application/pdf": ".pdf",
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/tiff": ".tiff",
                    "image/bmp": ".bmp",
                    "image/webp": ".webp",
                }
                suffix = ext_map.get(mime, ".bin")

                # Use display_name as stem if available
                original_name = display_name or f"attachment{suffix}"
                if original_name in staged:
                    continue  # Already staged this file this session

                # Decode bytes if needed
                if isinstance(data, str):
                    raw = base64.b64decode(data)
                else:
                    raw = bytes(data)

                # Write to a named temp file that persists until the process exits
                tmp = tempfile.NamedTemporaryFile(
                    suffix=suffix, prefix="parser_agent_", delete=False
                )
                tmp.write(raw)
                tmp.flush()
                tmp.close()

                staged[original_name] = tmp.name
                logger.info(
                    "[STAGE] '%s' (%s, %d bytes) → %s",
                    original_name, mime, len(raw), tmp.name,
                )

            break  # Only process latest user event

        callback_context.state["staged_files"] = staged

    except Exception as exc:
        logger.warning("[STAGE] Failed to stage attachments: %s", exc)


def _extract_user_text(callback_context: CallbackContext) -> str:
    """Return the most recent user message text for out-of-scope checking."""
    try:
        for event in reversed(callback_context.session.events or []):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts or []:
                    if getattr(part, "text", None):
                        return part.text
    except Exception:
        pass
    return ""


def _extract_request_text(llm_request: LlmRequest) -> str:
    """Flatten all text parts in an LlmRequest into a single string."""
    return " ".join(
        part.text
        for content in (llm_request.contents or [])
        for part in (content.parts or [])
        if getattr(part, "text", None)
    )


def _make_cache_key(tool_name: str, args: dict) -> str:
    """
    Content-aware SHA-256 cache key for a (tool_name, args) pair.

    For tools that operate on local files (parser_parse, parser_parse_batch),
    the file's mtime + size are folded into the hash so that an on-disk change
    automatically produces a new key — even when the path string is identical.
    This prevents the cache from serving a stale parse result after the source
    file has been modified, re-exported, or overwritten.

    Why mtime+size rather than a full content hash?
    -----------------------------------------------
    A SHA-256 of a 100 MB PDF takes ~300 ms. An os.stat() call takes < 1 µs.
    mtime+size catches every practical change (editor saves, pipeline re-runs,
    file replacements). The only undetected case is a same-size, same-mtime
    in-place overwrite — which does not occur in normal document workflows.

    JSON-sorting args ensures key stability regardless of dict insertion order.
    Full 64-char hex stored in state; last 8 chars used in log messages.
    """
    # Extract all file paths present in the args for this tool call
    fingerprints: dict[str, str] = {}

    if tool_name == "parse_document":
        path = args.get("file_path")
        if path:
            fingerprints[path] = _file_fingerprint(path)

    elif tool_name == "parse_batch":
        for path in args.get("file_paths", []):
            fingerprints[path] = _file_fingerprint(path)

    raw = json.dumps(
        {"tool": tool_name, "args": args, "fp": fingerprints},
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"cache:{tool_name}:{digest}"


def _file_fingerprint(path: str) -> str:
    """
    Return a cheap content-change signal for a local file: "{mtime:.6f}:{size}".

    Uses os.stat() — a single syscall, typically < 1 µs.  Returns the sentinel
    string "missing" when the file does not exist (e.g. base64-only delivery),
    so the key is still deterministic rather than raising.

    The fingerprint is embedded in the cache key, not stored separately,
    so there is nothing extra to maintain or expire.
    """
    try:
        st = os.stat(path)
        return f"{st.st_mtime:.6f}:{st.st_size}"
    except OSError:
        # File does not exist locally (base64 delivery) or path is not accessible.
        # Return a stable sentinel so the key remains deterministic.
        return "missing"


def _parse_tool_result(result) -> dict | list:
    """Normalise MCP tool result — may arrive as JSON string or parsed object."""
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
