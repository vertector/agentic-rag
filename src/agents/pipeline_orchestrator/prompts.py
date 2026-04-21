"""
prompts.py — Pipeline Orchestrator Dynamic Instruction Builder

Prompt split for context caching:

  static_instruction = _STATIC_INSTRUCTION   ← stable XML loaded once at import;
                                               eligible for Gemini prefix caching.
                                               Never put session state here.

  instruction        = build_instruction      ← turn-level callable; returns only
                                               the runtime augments block (or ""
                                               when nothing is active). ADK appends
                                               this after static_instruction each turn.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger("pipeline_orchestrator.prompts")

_AGENT_DIR = pathlib.Path(__file__).parent
_STATIC_PROMPT_PATH = _AGENT_DIR / "system_prompt.xml"

# ---------------------------------------------------------------------------
# _STATIC_INSTRUCTION
#
# Loaded once at import time so the value is a plain str — required by ADK's
# static_instruction parameter. The cached Gemini prefix corresponds exactly
# to this string; any mutation would bust the cache.
# ---------------------------------------------------------------------------

def _load_static_prompt() -> str:
    try:
        text = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
        logger.info("Loaded orchestrator prompt (%d chars)", len(text))
        return text
    except FileNotFoundError:
        logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
        return "<system_prompt><goal>Document pipeline orchestrator.</goal></system_prompt>"


_STATIC_INSTRUCTION: str = _load_static_prompt()


# ---------------------------------------------------------------------------
# build_instruction — dynamic turn-level augments only
#
# Returns the runtime context block that ADK appends after static_instruction.
# Returns "" (empty string) when no session state is active — ADK treats this
# as a no-op and does not append anything to the prompt.
#
# DO NOT return or re-include _STATIC_INSTRUCTION here. With static_instruction
# set on the LlmAgent, the XML is already present in every prompt. Repeating it
# would double the static prefix and break Gemini's prefix-cache hit.
# ---------------------------------------------------------------------------

def build_instruction(context: "InvocationContext") -> str:
    """
    Dynamic instruction builder. Returns only runtime augments when relevant:
      - Active file banner
      - Active corpus/version filters
      - Parser output ready for ingestion
      - Manifest discovery suggestion
      - Pending purge reminder
      - Pipeline step progress
      - Escalation lockout
    """
    state = context.session.state
    augments: list[str] = []

    if state.get("orchestrator:active_file"):
        augments.append(
            f"ACTIVE FILE: '{state['orchestrator:active_file']}' is the current working document. "
            "Resolve ambiguous file references ('it', 'that file', 'this document') to this."
        )

    if state.get("orchestrator:active_corpus"):
        augments.append(
            f"ACTIVE CORPUS: '{state['orchestrator:active_corpus']}' is the selected logical Knowledge Base. "
            "Pass this as corpus_id to all retrieval and ingestion calls."
        )

    if state.get("orchestrator:active_version"):
        augments.append(
            f"PINNED VERSION: '{state['orchestrator:active_version']}' is the selected Merkle root for point-in-time retrieval. "
            "Pass this as version_root to reranker_agent."
        )

    if state.get("orchestrator:parser_output_path"):
        augments.append(
            f"PARSER OUTPUT: '{state['orchestrator:parser_output_path']}' is ready to ingest. "
            "Pass this exact path as file_path to ingestion_agent — not the original PDF path."
        )
    elif state.get("orchestrator:active_file") and not state.get("orchestrator:last_intent") == "PARSE":
        augments.append(
            "MANIFEST DISCOVERY: The active file has no current parser output in state. "
            "If the user wants to ingest it, use `ingestion_agent.find_manifest` first to see if it was parsed before."
        )

    if state.get("orchestrator:pending_purge"):
        augments.append(
            f"PENDING PURGE: User requested deletion of '{state['orchestrator:pending_purge']}'. "
            "If they confirm in this turn, proceed. If they say anything else, cancel and clear this."
        )

    if state.get("orchestrator:pipeline_step"):
        step = state["orchestrator:pipeline_step"]
        augments.append(
            f"PIPELINE IN PROGRESS: Currently on step {step} of a multi-step workflow. "
            "Complete this step before accepting new requests."
        )

    if state.get("orchestrator:escalation_pending"):
        augments.append(
            "⛔  ESCALATION: A sub-agent could not complete the last operation. "
            "Tell the user something went wrong and suggest retrying or contacting support."
        )

    # ── Retrieved Context (for Answer Generation)
    # If the reranker has returned results, inject them so the orchestrator can synthesize an answer.
    rerank_output_raw = state.get("reranker:last_results")
    if rerank_output_raw:
        try:
            results = json.loads(rerank_output_raw)
            if results and isinstance(results, list):
                context_block = ["RETRIEVED CONTEXT (use this to answer the user's question):"]
                for i, res in enumerate(results, 1):
                    citation = res.get("citation", {})
                    content = res.get("content", "")
                    score = res.get("final_score", 0.0)
                    summary = res.get("summary")
                    
                    header = f"Result {i} (File: {citation.get('filename')}, Page: {citation.get('page_index')}, Score: {score:.4f})"
                    context_block.append(f"--- {header} ---")
                    if summary:
                        context_block.append(f"[Semantic Summary]: {summary}")
                    context_block.append(content)
                    context_block.append("")
                
                augments.append("\n".join(context_block))
        except json.JSONDecodeError:
            pass

    if not augments:
        return ""

    runtime_block = "<!-- RUNTIME CONTEXT -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return runtime_block