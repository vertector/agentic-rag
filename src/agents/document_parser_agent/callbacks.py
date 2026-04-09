"""
callbacks.py — Document Parser Agent Callbacks
All 6 ADK hooks with correct signatures (callback_context, not ctx).
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

logger = logging.getLogger("document_parser_agent.callbacks")

_MAX_SESSION_LOG = 100
_HIGH_WORKER_THRESHOLD = 4
_HIGH_WORKER_ACK_KEY = "parser:high_worker_acknowledged"
_PROMPT_LABEL_ACK_KEY = "parser:prompt_label_acknowledged"
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"})


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    if state.get("pipeline:halt"):
        reason = state.get("pipeline:halt_reason", "Pipeline halted by orchestrator.")
        logger.warning("[PARSER] Blocked by pipeline:halt — %s", reason)
        return types.Content(
            role="model",
            parts=[types.Part(text=json.dumps({"error": "PipelineHalt", "message": reason}))],
        )
    if "parser:session_parse_log" not in state:
        state["parser:session_parse_log"] = []
    warm_status = "warm" if state.get("parser:warm") else "cold (first call ~30s)"
    logger.info("[PARSER START] session=%s pipeline=%s", callback_context.session_id, warm_status)
    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    log: list = state.get("parser:session_parse_log", [])
    if len(log) > _MAX_SESSION_LOG:
        state["parser:session_parse_log"] = log[-_MAX_SESSION_LOG:]
    logger.info(
        "[PARSER END] session=%s total_parses=%d warm=%s",
        callback_context.session_id,
        len(state.get("parser:session_parse_log", [])),
        state.get("parser:warm", False),
    )
    return None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    state = callback_context.state
    ctx_parts = [f"pipeline={'warm' if state.get('parser:warm') else 'cold'}"]
    if state.get("parser:active_category"):
        ctx_parts.append(f"category={state['parser:active_category']}")
    if state.get("parser:last_parsed_file"):
        ctx_parts.append(f"last_file={Path(state['parser:last_parsed_file']).name}")
    ctx_parts.append(f"prior_parses={len(state.get('parser:session_parse_log', []))}")

    if llm_request.contents:
        last_msg = llm_request.contents[-1]
        if last_msg.role == "user" and last_msg.parts:
            last_msg.parts.append(types.Part(text=f"[parser_state: {', '.join(ctx_parts)}]"))
    return None


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not llm_response.content or not llm_response.content.parts:
        return None
    text = " ".join(p.text for p in llm_response.content.parts if p.text)
    if '"escalate": true' in text or '"escalate":true' in text:
        logger.warning("[PARSER ESCALATION] session=%s", callback_context.session_id)
        callback_context.state["parser:escalation_pending"] = True
    return None


def before_tool_callback(
    tool: BaseTool,
    tool_args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    if tool_name == "configure_parser":
        params = tool_args.get("params", {})
        if isinstance(params, dict) and params.get("prompt_label") is not None:
            if not state.get(_PROMPT_LABEL_ACK_KEY):
                return {
                    "confirmation_required": True,
                    "warning": (
                        "Setting `prompt_label` forces ALL layout blocks through one prompt "
                        "and breaks Markdown table extraction. "
                        "Set `parser:prompt_label_acknowledged = True` in session state to proceed."
                    ),
                }
            del state[_PROMPT_LABEL_ACK_KEY]

    if tool_name == "parse_batch":
        params = tool_args.get("params", {}) if isinstance(tool_args.get("params"), dict) else {}
        max_workers = tool_args.get("max_workers") or params.get("max_workers")
        file_paths = tool_args.get("file_paths") or params.get("file_paths") or []

        if max_workers and int(max_workers) > _HIGH_WORKER_THRESHOLD:
            if not state.get(_HIGH_WORKER_ACK_KEY):
                return {
                    "confirmation_required": True,
                    "warning": (
                        f"`max_workers={max_workers}` loads {max_workers} PaddleOCRVL instances "
                        f"(~{int(max_workers) * 1.5:.0f} GB RAM/VRAM estimated). "
                        "Set `parser:high_worker_acknowledged = True` to proceed."
                    ),
                }
            del state[_HIGH_WORKER_ACK_KEY]

        if file_paths:
            bad = [p for p in file_paths if Path(p).suffix.lower() not in SUPPORTED_EXTENSIONS]
            if bad:
                return {"error": "UnsupportedExtension", "message": f"Unsupported: {bad}"}

    if tool_name == "parse_document":
        params = tool_args.get("params", {})
        if isinstance(params, dict):
            path_to_check = params.get("file_path") or params.get("filename")
            if path_to_check and Path(path_to_check).suffix.lower() not in SUPPORTED_EXTENSIONS:
                return {
                    "error": "UnsupportedExtension",
                    "message": f"File type '{Path(path_to_check).suffix}' is not supported.",
                    "supported": sorted(SUPPORTED_EXTENSIONS),
                }
    return None


def after_tool_callback(
    tool: BaseTool,
    tool_args: dict,
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

    if tool_name == "parse_document" and isinstance(parsed, list):
        page_count = len(parsed)
        filename = parsed[0].get("metadata", {}).get("filename", "") if parsed else ""
        stem = Path(filename).stem if filename else "unknown"
        state["parser:warm"] = True
        state["parser:last_parsed_file"] = filename
        state["parser:last_results"] = json.dumps(parsed)
        log: list = state.get("parser:session_parse_log", [])
        log.append({"file": filename, "page_count": page_count, "merkle_roots": [p.get("merkle_root", "") for p in parsed]})
        state["parser:session_parse_log"] = log
        try:
            artifact_name = f"parsed_{stem}_{tool_context.invocation_id}.json"
            tool_context.save_artifact(
                filename=artifact_name,
                artifact=types.Part(inline_data=types.Blob(mime_type="application/json", data=json.dumps(parsed, indent=2).encode())),
            )
            logger.info("[ARTIFACT SAVED] %s — %d pages", artifact_name, page_count)
        except Exception as exc:
            logger.warning("[ARTIFACT SAVE FAILED] %s", exc)
        logger.info("[TOOL POST] parse_document file=%r pages=%d", filename, page_count)

    elif tool_name == "parse_batch" and isinstance(parsed, list):
        state["parser:warm"] = True
        file_paths = tool_args.get("file_paths") or []
        log: list = state.get("parser:session_parse_log", [])
        for i, doc_list in enumerate(parsed):
            fp = file_paths[i] if i < len(file_paths) else f"file_{i}"
            if isinstance(doc_list, list):
                log.append({"file": fp, "page_count": len(doc_list), "merkle_roots": [p.get("merkle_root", "") for p in doc_list]})
        state["parser:session_parse_log"] = log
        try:
            artifact_name = f"batch_{tool_context.invocation_id}.json"
            tool_context.save_artifact(
                filename=artifact_name,
                artifact=types.Part(inline_data=types.Blob(mime_type="application/json", data=json.dumps(parsed, indent=2).encode())),
            )
            logger.info("[ARTIFACT SAVED] %s — %d documents", artifact_name, len(parsed))
        except Exception as exc:
            logger.warning("[ARTIFACT SAVE FAILED] %s", exc)
        logger.info("[TOOL POST] parse_batch documents=%d", len(parsed))

    elif tool_name == "configure_parser" and isinstance(parsed, dict):
        state["parser:warm"] = False
        if "current_settings" in parsed:
            state["parser:active_settings"] = json.dumps(parsed["current_settings"])
        logger.info("[TOOL POST] configure_parser applied=%s warm=False", parsed.get("applied_updates", {}))

    elif tool_name == "get_parser_settings" and isinstance(parsed, dict):
        logger.info("[TOOL POST] get_parser_settings pipeline_version=%s", parsed.get("pipeline_version", "?"))

    return None