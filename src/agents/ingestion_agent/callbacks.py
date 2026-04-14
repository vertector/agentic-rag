"""
callbacks.py — Ingestion Agent Callbacks
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
        ctx_parts.append(f"version_root={state['ingestor:version_root'][:8]}")
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


async def before_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    # ── Security: Purge Gate
    if tool_name == "purge":
        if not state.get("ingestor:purge_confirmed"):
            return {
                "error": "ConfirmationRequired",
                "message": "Destructive operation 'purge' requires explicit confirmation.",
            }
        # Reset the gate for next time
        del state["ingestor:purge_confirmed"]

    # ── Auto-inject category (flat signature — fields are top-level in args)
    if tool_name in ("ingest", "search", "history", "purge"):
        active_cat = state.get("ingestor:active_category")
        if active_cat and not args.get("category"):
            args["category"] = active_cat
            logger.info("[INGESTOR] Auto-injected category=%s into %s", active_cat, tool_name)

    # ── Auto-inject version_root for search (flat signature)
    if tool_name == "search":
        v_root = state.get("ingestor:version_root")
        if v_root and not args.get("version_root"):
            args["version_root"] = v_root
            logger.info("[INGESTOR] Auto-injected version_root=%s... into search", v_root[:8])

    return None


async def after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    def _unwrap(response) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict) and first.get("type") == "text":
                return first.get("text", "")
            if hasattr(first, "text"):
                return first.text or ""
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text":
                    return first.get("text", "")
                if hasattr(first, "text"):
                    return first.text or ""
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

    if tool_name == "status":
        state["ingestor:connected"] = True
        logger.info("[TOOL POST] status qdrant=%s redis=%s collection=%s",
                    parsed.get("qdrant", False), parsed.get("redis", False), parsed.get("collection", ""))

    elif tool_name == "ingest":
        ingested = parsed.get("ingested", 0)
        skipped = parsed.get("skipped", 0)
        errors = parsed.get("errors", [])
        collection = parsed.get("collection", "")

        # Best-effort filename extraction from args (flat signature)
        file_path = args.get("file_path", "")
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
            await tool_context.save_artifact(
                filename=artifact_name,
                artifact=types.Part(inline_data=types.Blob(
                    mime_type="application/json",
                    data=json.dumps(parsed, indent=2).encode(),
                )),
            )
            logger.info("[ARTIFACT SAVED] %s", artifact_name)
        except Exception as exc:
            logger.warning("[ARTIFACT SAVE FAILED] %s", exc)

        logger.info("[TOOL POST] ingest ingested=%d skipped=%d errors=%d collection=%s",
                    ingested, skipped, len(errors), collection)

    elif tool_name == "purge":
        logger.info("[TOOL POST] purge file=%s deleted=%s",
                    parsed.get("filename", ""), parsed.get("deleted", False))

    elif tool_name == "sync":
        logger.info("[TOOL POST] sync fixed=%d collection=%s",
                    parsed.get("fixed", 0), parsed.get("collection", ""))

    return None