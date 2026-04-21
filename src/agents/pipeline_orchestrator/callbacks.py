"""
callbacks.py — Pipeline Orchestrator Agent Callbacks
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger("pipeline_orchestrator.callbacks")

_MAX_CONV_LOG = 20


def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state

    if "orchestrator:conversation_log" not in state:
        state["orchestrator:conversation_log"] = []

    logger.info(
        "[ORCHESTRATOR START] session=%s active_file=%s last_intent=%s",
        callback_context.session.id,
        state.get("orchestrator:active_file", "<none>"),
        state.get("orchestrator:last_intent", "<none>"),
    )
    return None


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    log: list = state.get("orchestrator:conversation_log", [])
    if len(log) > _MAX_CONV_LOG:
        state["orchestrator:conversation_log"] = log[-_MAX_CONV_LOG:]
    logger.info("[ORCHESTRATOR END] session=%s", callback_context.session.id)
    return None


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    state = callback_context.state
    # Corrected project root: src/agents/pipeline_orchestrator/callbacks.py -> 4 parents to reach root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    src_dir = project_root / "src"

    # 1. Proactively extract active_file, corpus_id, and version_root from user message
    if llm_request.contents:
        last_msg = llm_request.contents[-1]
        if last_msg.role == "user" and last_msg.parts:
            user_text = " ".join(p.text for p in last_msg.parts if p.text)
            
            # --- Extract active_file ---
            if not state.get("orchestrator:active_file"):
                # Find paths, filenames, or bracketed placeholders
                matches = re.findall(r'([^\s]+(?:/[^\s]+)*\.\w+)|{([^}]+)}', user_text)
                for m_tuple in matches:
                    m = m_tuple[0] or m_tuple[1]
                    m_clean = m.strip('`"\' ')
                    
                    # Try to find a matching dir in src/ for the stem
                    stem = Path(m_clean).stem.lower()
                    # Also handle placeholders like {SAMPLE_PDF} -> try 'sample'
                    clean_stem = stem.replace('_pdf', '').replace('_png', '').replace('_jpg', '').strip('{}')
                    
                    found_dir = None
                    if src_dir.is_dir():
                        for d in src_dir.iterdir():
                            if d.is_dir() and d.name.lower() in (stem, clean_stem):
                                found_dir = d
                                break
                    
                    if found_dir or m_clean.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
                        state["orchestrator:active_file"] = m_clean
                        logger.info("[ORCH] Auto-extracted active_file: %s", m_clean)
                        break

            # --- Extract corpus_id ---
            corpus_matches = re.findall(r"(?:corpus|knowledge base|container)\s+['\"]?([\w-]+)['\"]?", user_text, re.IGNORECASE)
            if corpus_matches:
                state["orchestrator:active_corpus"] = corpus_matches[0]
                logger.info("[ORCH] Auto-extracted active_corpus: %s", corpus_matches[0])

            # --- Extract version_root (64-char hex) ---
            version_matches = re.findall(r"\b([a-fA-F0-9]{64})\b", user_text)
            if version_matches:
                state["orchestrator:active_version"] = version_matches[0]
                logger.info("[ORCH] Auto-extracted active_version: %s", version_matches[0])
            elif "current" in user_text.lower() or "latest" in user_text.lower() or "active version" in user_text.lower():
                # Explicitly clear version pin if user asks for latest
                if "orchestrator:active_version" in state:
                    del state["orchestrator:active_version"]
                logger.info("[ORCH] Cleared active_version pin")

    # 2. Derive parser_output_path from sub-agent output or disk discovery
    if not state.get("orchestrator:parser_output_path"):
        # A. Try to extract from recent document_parser_agent response
        parser_output_raw = state.get("parser:agent_output")
        if parser_output_raw:
            try:
                data = json.loads(parser_output_raw)
                if isinstance(data, dict) and "output_path" in data:
                    state["orchestrator:parser_output_path"] = data["output_path"]
                    logger.info("[ORCH] Found absolute output_path in JSON: %s", data["output_path"])
            except json.JSONDecodeError:
                pass

        # B. Discovery on disk if active_file is known
        if not state.get("orchestrator:parser_output_path"):
            active_file = state.get("orchestrator:active_file")
            if active_file:
                # Resolve placeholder logic
                clean_name = active_file.strip('{}')
                stem = Path(clean_name).stem.lower()
                clean_stem = stem.replace('_pdf', '').replace('_png', '').replace('_jpg', '')
                
                potential_path = None
                if src_dir.is_dir():
                    for d in src_dir.iterdir():
                        if d.is_dir() and d.name.lower() in (stem, clean_stem):
                            p = d / "documents.json"
                            if p.exists():
                                potential_path = p
                                break
                
                if potential_path:
                    state["orchestrator:parser_output_path"] = str(potential_path.resolve())
                    logger.info("[ORCH] Discovered documents.json on disk: %s", potential_path)
                else:
                    # Fallback to derivation from parser:last_parsed_file
                    source_file = state.get("parser:last_parsed_file")
                    if source_file:
                        source_stem = Path(source_file).stem.lower()
                        for d in src_dir.iterdir():
                            if d.is_dir() and d.name.lower() == source_stem:
                                p = d / "documents.json"
                                if p.exists():
                                    state["orchestrator:parser_output_path"] = str(p.resolve())
                                    logger.info("[ORCH] Derived absolute parser_output_path: %s", p)
                                    break

    # 3. Purge confirmation detection
    pending_purge = state.get("orchestrator:pending_purge")
    if pending_purge and llm_request.contents:
        last_msg = llm_request.contents[-1]
        if last_msg.role == "user" and last_msg.parts:
            user_text = " ".join(p.text for p in last_msg.parts if p.text).lower().strip()
            _CONFIRM_TOKENS = {"yes", "confirm", "proceed", "go ahead", "do it", "ok", "sure", "delete it", "yes delete"}
            _CANCEL_TOKENS = {"no", "cancel", "stop", "never mind", "abort"}
            if any(tok in user_text for tok in _CONFIRM_TOKENS):
                state["ingestor:purge_confirmed"] = True
                logger.info("[ORCH] Purge confirmed for '%s'", pending_purge)
            elif any(tok in user_text for tok in _CANCEL_TOKENS):
                del state["orchestrator:pending_purge"]
                logger.info("[ORCH] Purge cancelled for '%s'", pending_purge)

    ctx_parts = []
    if state.get("orchestrator:active_file"):
        ctx_parts.append(f"active_file={state['orchestrator:active_file']}")
    if state.get("orchestrator:parser_output_path"):
        ctx_parts.append(f"parser_output_path={state['orchestrator:parser_output_path']}")
    if state.get("orchestrator:last_intent"):
        ctx_parts.append(f"last_intent={state['orchestrator:last_intent']}")
    if state.get("orchestrator:pipeline_step"):
        ctx_parts.append(f"pipeline_step={state['orchestrator:pipeline_step']}")
    if state.get("orchestrator:pending_purge"):
        ctx_parts.append(f"pending_purge={state['orchestrator:pending_purge']}")

    if ctx_parts and llm_request.contents:
        last_msg = llm_request.contents[-1]
        if last_msg.role == "user" and last_msg.parts:
            last_msg.parts.append(types.Part(text=f"\n\n[orchestrator_state: {', '.join(ctx_parts)}]"))
    return None


def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    if not llm_response.content or not llm_response.content.parts:
        return None

    text = " ".join(p.text for p in llm_response.content.parts if p.text)
    state = callback_context.state

    # Track intent
    for intent in ("PARSE", "INGEST", "RETRIEVE", "PIPELINE", "AMBIGUOUS"):
        if f"intent:{intent}" in text:
            state["orchestrator:last_intent"] = intent
            break

    # After purge confirmed + delegated: clear pending_purge
    if state.get("ingestor:purge_confirmed") is True and state.get("orchestrator:pending_purge"):
        # The condition below was 'not in state', which was logically impossible here.
        # Changed to simply check if the key exists before deleting.
        if "orchestrator:pending_purge" in state:
            del state["orchestrator:pending_purge"]
            logger.info("[ORCH] Cleared orchestrator:pending_purge after successful purge delegation")

    if '"escalate": true' in text or '"escalate":true' in text:
        logger.warning("[ORCHESTRATOR ESCALATION] session=%s", callback_context.session.id)
        state["orchestrator:escalation_pending"] = True

    return None


def before_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    logger.debug("[ORCH TOOL PRE] %s", tool.name)
    return None


def after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    logger.debug("[ORCH TOOL POST] %s", tool.name)
    return None
