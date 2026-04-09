"""
callbacks.py — Ingestion Agent Callbacks
All 6 ADK hooks with correct signatures (callback_context, not ctx).

Guards:
  before_tool_callback:
    - ingest_purge: requires ingestor:purge_confirmed=true in state AND confirm=true in params
    - ingest_configure: requires ingestor:configure_acknowledged=true for any field change
  after_tool_callback:
    - ingest_data: persist summary, log entry, artifact
    - ingest_status: set ingestor:connected
    - ingest_configure: clear ingestor:connected (re-init needed)
    - ingest_purge: clear last_ingested_file if matching
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger("ingestion_agent.callbacks")

_MAX_SESSION_LOG = 100
_MAX_VERSION_ROOTS_FILES = 20
_PURGE_CONFIRMED_KEY = "ingestor:purge_confirmed"
_CONFIGURE_ACK_KEY = "ingestor:configure_acknowledged"


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    if state.get("pipeline:halt"):
        reason = state.get("pipeline:halt_reason", "Pipeline halted by orchestrator.")
        logger.warning("[INGESTOR] Blocked by pipeline:halt — %s", reason)
        return types.Content(
            role="model",
            parts=[types.Part(text=json.dumps({"error": "PipelineHalt", "message": reason}))],
        )
    if "ingestor:session_ingest_log" not in state:
        state["ingestor:session_ingest_log"] = []
    logger.info(
        "[INGESTOR START] session=%s connected=%s category=%s",
        callback_context.session.id,
        state.get("ingestor:connected", False),
        state.get("ingestor:active_category", "<none>"),
    )
    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    log: list = state.get("ingestor:session_ingest_log", [])
    if len(log) > _MAX_SESSION_LOG:
        state["ingestor:session_ingest_log"] = log[-_MAX_SESSION_LOG:]
    logger.info(
        "[INGESTOR END] session=%s total_ingests=%d",
        callback_context.session.id,
        len(state.get("ingestor:session_ingest_log", [])),
    )
    return None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    state = callback_context.state
    ctx_parts = [f"connected={state.get('ingestor:connected', False)}"]
    if state.get("ingestor:active_category"):
        ctx_parts.append(f"category={state['ingestor:active_category']}")
    if state.get("ingestor:version_root"):
        ctx_parts.append(f"version_root={state['ingestor:version_root'][:12]}...")
    if state.get("ingestor:last_ingested_file"):
        ctx_parts.append(f"last_file={Path(state['ingestor:last_ingested_file']).name}")
    ctx_parts.append(f"prior_ingests={len(state.get('ingestor:session_ingest_log', []))}")

    if llm_request.contents:
        last_msg = llm_request.contents[-1]
        if last_msg.role == "user" and last_msg.parts:
            last_msg.parts.append(types.Part(text=f"[ingestor_state: {', '.join(ctx_parts)}]"))
    return None


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not llm_response.content or not llm_response.content.parts:
        return None
    text = " ".join(p.text for p in llm_response.content.parts if p.text)
    if '"escalate": true' in text or '"escalate":true' in text:
        logger.warning("[INGESTOR ESCALATION] session=%s", callback_context.session.id)
        callback_context.state["ingestor:escalation_pending"] = True
    return None


def before_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    # ── ingest_purge: dual gate (state key + params.confirm)
    if tool_name == "ingest_purge":
        params = args.get("params", {})
        filename = params.get("filename", "<unknown>") if isinstance(params, dict) else "<unknown>"
        confirm_in_params = params.get("confirm", False) if isinstance(params, dict) else False

        if not state.get(_PURGE_CONFIRMED_KEY):
            logger.info("[TOOL BLOCKED] ingest_purge — awaiting purge_confirmed for %s", filename)
            return {
                "confirmation_required": True,
                "message": (
                    f"ingest_purge for '{filename}' is IRREVERSIBLE — deletes ALL versions, "
                    "all pages, and all Qdrant vectors + Redis keys for this document. "
                    f"Set `ingestor:purge_confirmed = True` in session state to proceed."
                ),
                "filename": filename,
            }

        if not confirm_in_params:
            # State key set but params.confirm not forwarded — fix params
            if isinstance(params, dict):
                params["confirm"] = True
                logger.info("[TOOL AUTO-FIX] ingest_purge confirm=True injected")

        # Consume the ack
        del state[_PURGE_CONFIRMED_KEY]

    # ── ingest_configure: connection change gate
    if tool_name == "ingest_configure":
        params = args.get("params", {})
        if isinstance(params, dict):
            updates = {k: v for k, v in params.items() if v is not None}
            if updates and not state.get(_CONFIGURE_ACK_KEY):
                model_change = "model_name" in updates
                extra_warning = (
                    " Changing `model_name` targets a DIFFERENT Qdrant collection — "
                    "existing ingested data will not be visible until model_name is reverted."
                    if model_change else ""
                )
                logger.info("[TOOL BLOCKED] ingest_configure — awaiting configure_acknowledged")
                return {
                    "confirmation_required": True,
                    "message": (
                        f"ingest_configure will tear down and re-initialise the ingestor. "
                        f"Fields to change: {list(updates.keys())}.{extra_warning} "
                        "Set `ingestor:configure_acknowledged = True` in session state to proceed."
                    ),
                    "fields_to_change": list(updates.keys()),
                }
            if _CONFIGURE_ACK_KEY in state:
                del state[_CONFIGURE_ACK_KEY]

    logger.debug("[TOOL PRE] %s", tool_name)
    return None


def after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    # MCP tools return a wrapper: {"content": [{"type": "text", "text": "<json string>"}]}
    # Extract the inner text before parsing.
    def _unwrap(response) -> str:
        # Case 1: plain string (direct tool return)
        if isinstance(response, str):
            return response
        # Case 2: list of content blocks (ADK MCP delivery)
        # e.g. [{"type": "text", "text": "..."} ] or [TextContent(type="text", text="...")]
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict) and first.get("type") == "text":
                return first.get("text", "")
            if hasattr(first, "text"):
                return first.text or ""
        # Case 3: {"content": [...]} wrapper dict
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text":
                    return first.get("text", "")
                if hasattr(first, "text"):
                    return first.text or ""
        # Case 4: object with .text attribute (Part, TextContent)
        if hasattr(response, "text"):
            return response.text or ""
        return json.dumps(response) if response else ""

    raw = _unwrap(tool_response)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    if isinstance(parsed, dict) and "error" in parsed:
        logger.warning("[TOOL ERROR] %s → %s: %s", tool_name, parsed.get("error"), parsed.get("message", "")[:120])
        return None

    # ── ingest_data
    if tool_name == "ingest_data" and isinstance(parsed, dict):
        ingested = parsed.get("ingested", 0)
        skipped = parsed.get("skipped", 0)
        errors = parsed.get("errors", [])
        collection = parsed.get("collection", "")

        # Best-effort filename extraction from args
        params = args.get("params", {}) if isinstance(args.get("params"), dict) else {}
        file_path = params.get("file_path") or args.get("file_path", "")
        stem = Path(file_path).stem if file_path else "inline"
        filename = Path(file_path).name if file_path else "inline"

        state["ingestor:connected"] = True
        state["ingestor:last_ingested_file"] = filename
        state["ingestor:last_ingest_summary"] = json.dumps(parsed)

        log: list = state.get("ingestor:session_ingest_log", [])
        log.append({"file": filename, "ingested": ingested, "skipped": skipped, "errors": len(errors)})
        state["ingestor:session_ingest_log"] = log

        try:
            artifact_name = f"ingest_{stem}_{tool_context.invocation_id}.json"
            tool_context.save_artifact(
                filename=artifact_name,
                artifact=types.Part(inline_data=types.Blob(
                    mime_type="application/json",
                    data=json.dumps(parsed, indent=2).encode(),
                )),
            )
            logger.info("[ARTIFACT SAVED] %s", artifact_name)
        except Exception as exc:
            logger.warning("[ARTIFACT SAVE FAILED] %s", exc)

        logger.info("[TOOL POST] ingest_data ingested=%d skipped=%d errors=%d collection=%s",
                    ingested, skipped, len(errors), collection)

    # ── ingest_status
    elif tool_name == "ingest_status" and isinstance(parsed, dict):
        qdrant_ok = parsed.get("qdrant_reachable", False)
        redis_ok = parsed.get("redis_reachable", False)
        state["ingestor:connected"] = qdrant_ok and redis_ok
        logger.info("[TOOL POST] ingest_status qdrant=%s redis=%s collection=%s",
                    qdrant_ok, redis_ok, parsed.get("collection", "?"))

    # ── ingest_configure
    elif tool_name == "ingest_configure" and isinstance(parsed, dict):
        state["ingestor:connected"] = False  # re-init needed; fresh status required
        logger.info("[TOOL POST] ingest_configure applied=%s collection=%s",
                    parsed.get("applied_updates", {}), parsed.get("active_collection", "?"))

    # ── ingest_purge
    elif tool_name == "ingest_purge" and isinstance(parsed, dict):
        if parsed.get("purged"):
            purged_file = parsed.get("filename", "")
            if state.get("ingestor:last_ingested_file") == purged_file:
                del state["ingestor:last_ingested_file"]
            # Remove from version_roots if present
            roots: dict = state.get("ingestor:version_roots", {})
            roots.pop(purged_file, None)
            state["ingestor:version_roots"] = roots
        logger.info("[TOOL POST] ingest_purge purged=%s filename=%s",
                    parsed.get("purged"), parsed.get("filename"))

    # ── ingest_history
    elif tool_name == "ingest_history" and isinstance(parsed, dict):
        filename = parsed.get("filename", "")
        versions = parsed.get("versions", [])
        if filename and versions:
            roots: dict = state.get("ingestor:version_roots", {})
            roots[filename] = [v.get("version_root", "") for v in versions]
            # Cap at 20 filenames
            if len(roots) > _MAX_VERSION_ROOTS_FILES:
                oldest_key = next(iter(roots))
                del roots[oldest_key]
            state["ingestor:version_roots"] = roots
        logger.info("[TOOL POST] ingest_history filename=%s versions=%d", filename, len(versions))

    # ── ingest_audit
    elif tool_name == "ingest_audit" and isinstance(parsed, dict):
        integrity = parsed.get("integrity", "UNKNOWN")
        logger.info("[TOOL POST] ingest_audit %s p.%s → %s",
                    parsed.get("filename"), parsed.get("page_index"), integrity)

    # ── ingest_sync
    elif tool_name == "ingest_sync" and isinstance(parsed, dict):
        logger.info("[TOOL POST] ingest_sync filename=%s pages_reconciled=%d",
                    parsed.get("filename"), parsed.get("pages_reconciled", 0))

    # ── ingest_search
    elif tool_name == "ingest_search" and isinstance(parsed, list):
        logger.info("[TOOL POST] ingest_search result_count=%d", len(parsed))

    return None