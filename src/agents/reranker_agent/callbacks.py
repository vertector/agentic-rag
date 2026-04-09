"""
callbacks.py — Reranker Agent Callbacks
All 6 ADK hooks with correct signatures (callback_context, not ctx).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger("reranker_agent.callbacks")

_MAX_SESSION_SCORES = 50
_SLOW_OPS = frozenset({"cross_encoder_model_name"})
_INGESTOR_REBUILD_OPS = frozenset({"qdrant_url", "redis_host", "redis_port", "embed_model_name"})
_SLOW_OP_ACK_KEY = "reranker:slow_op_acknowledged"


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    if state.get("pipeline:halt"):
        reason = state.get("pipeline:halt_reason", "Pipeline halted by orchestrator.")
        logger.warning("[RERANKER] Blocked by pipeline:halt — %s", reason)
        return types.Content(
            role="model",
            parts=[types.Part(text=json.dumps({"error": "PipelineHalt", "message": reason}))],
        )
    if "reranker:session_scores" not in state:
        state["reranker:session_scores"] = []
    logger.info(
        "[RERANKER START] session=%s category=%s version_root=%s",
        callback_context.session.id,
        state.get("reranker:active_category", "<none>"),
        state.get("reranker:version_root", "<active>"),
    )
    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    scores: list = state.get("reranker:session_scores", [])
    if len(scores) > _MAX_SESSION_SCORES:
        state["reranker:session_scores"] = scores[-_MAX_SESSION_SCORES:]
    logger.info(
        "[RERANKER END] session=%s total_rerank_calls=%d",
        callback_context.session.id,
        len(state.get("reranker:session_scores", [])),
    )
    return None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    state = callback_context.state
    ctx_parts = []
    if state.get("reranker:active_category"):
        ctx_parts.append(f"category={state['reranker:active_category']}")
    if state.get("reranker:version_root"):
        ctx_parts.append(f"version_root={state['reranker:version_root'][:12]}...")
    ctx_parts.append(f"prior_calls={len(state.get('reranker:session_scores', []))}")

    if ctx_parts and llm_request.contents:
        last = llm_request.contents[-1]
        if last.role == "user" and last.parts:
            last.parts.append(types.Part(text=f"[reranker_state: {', '.join(ctx_parts)}]"))
    return None


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not llm_response.content or not llm_response.content.parts:
        return None
    text = " ".join(p.text for p in llm_response.content.parts if p.text)
    if '"escalate": true' in text or '"escalate":true' in text:
        logger.warning("[RERANKER ESCALATION] session=%s", callback_context.session.id)
        callback_context.state["reranker:escalation_pending"] = True
    return None


def before_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    tool_name = tool.name
    state = tool_context.state

    if tool_name == "rerank_configure":
        params = args.get("params", {})
        if isinstance(params, dict):
            needs_slow_ack = bool((_SLOW_OPS | _INGESTOR_REBUILD_OPS) & set(params.keys()))
            if needs_slow_ack and not state.get(_SLOW_OP_ACK_KEY):
                affected = list((_SLOW_OPS | _INGESTOR_REBUILD_OPS) & set(params.keys()))
                return {
                    "confirmation_required": True,
                    "message": (
                        f"Fields {affected} trigger a slow operation or ingestor rebuild. "
                        "Set `reranker:slow_op_acknowledged = True` in session state to proceed."
                    ),
                    "affected_fields": affected,
                }
            if _SLOW_OP_ACK_KEY in state:
                del state[_SLOW_OP_ACK_KEY]

    if tool_name == "rerank_search":
        params = args.get("params", {})
        if isinstance(params, dict):
            top_k = params.get("retrieval_top_k", 50)
            top_n = params.get("rerank_top_n", 5)
            if top_n > top_k:
                corrected_k = max(top_k, top_n * 3)
                params["retrieval_top_k"] = corrected_k
                logger.info("[TOOL AUTO-CORRECT] retrieval_top_k %d → %d", top_k, corrected_k)

    return None


def after_tool_callback(
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

    if tool_name == "rerank_search" and isinstance(parsed, dict):
        query = parsed.get("query", "")
        results = parsed.get("results", [])
        result_count = parsed.get("result_count", 0)
        top_score = results[0].get("final_score", 0.0) if results else 0.0

        state["reranker:last_query"] = query
        state["reranker:last_results"] = json.dumps(results)
        history: list = state.get("reranker:session_scores", [])
        history.append({"query": query, "result_count": result_count, "top_score": round(top_score, 6)})
        state["reranker:session_scores"] = history

        try:
            artifact_name = f"rerank_results_{tool_context.invocation_id}.json"
            tool_context.save_artifact(
                filename=artifact_name,
                artifact=types.Part(inline_data=types.Blob(mime_type="application/json", data=json.dumps(parsed, indent=2).encode())),
            )
            logger.info("[ARTIFACT SAVED] %s — %d results", artifact_name, result_count)
        except Exception as exc:
            logger.warning("[ARTIFACT SAVE FAILED] %s", exc)
        logger.info("[TOOL POST] rerank_search query=%r result_count=%d top_score=%.4f", query[:60], result_count, top_score)

    elif tool_name == "rerank_configure" and isinstance(parsed, dict):
        active = parsed.get("active_settings", {})
        if "alpha" in active:
            state["reranker:active_alpha"] = active["alpha"]
        logger.info("[TOOL POST] rerank_configure applied=%s", parsed.get("applied_updates", {}))

    elif tool_name == "rerank_cache_clear" and isinstance(parsed, dict):
        logger.info("[TOOL POST] rerank_cache_clear cleared=%d", parsed.get("cleared", 0))

    elif tool_name == "rerank_status":
        logger.info("[TOOL POST] rerank_status qdrant=%s redis=%s", parsed.get("qdrant_reachable"), parsed.get("redis_reachable"))

    return None