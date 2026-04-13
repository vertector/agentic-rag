"""
prompts.py — Reranker Agent Dynamic Instruction Builder
=========================================================

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

import pathlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger("reranker_agent.prompts")

_PROMPT_DIR = pathlib.Path(__file__).parent
_STATIC_PROMPT_PATH = _PROMPT_DIR / "system_prompt.xml"

# ---------------------------------------------------------------------------
# _STATIC_INSTRUCTION
#
# Loaded once at import time so the value is a plain str — required by ADK's
# static_instruction parameter. Gemini caches this prefix; any mutation busts
# the cache, so this value must never change after the process starts.
# ---------------------------------------------------------------------------

def _load_static_prompt() -> str:
    try:
        text = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
        logger.info(
            "Loaded system prompt from %s (%d chars)",
            _STATIC_PROMPT_PATH, len(text),
        )
        return text
    except FileNotFoundError:
        logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
        return "<system_prompt><goal>Hybrid document reranker sub-agent.</goal></system_prompt>"


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
    Dynamic instruction builder — called by ADK on every agent turn.

    Augments (appended as a runtime context block when relevant):
      - Degraded-mode warning if BM25 is known unavailable (from state).
      - Active Merkle snapshot banner if version_root is pinned.
      - Alpha override note if CE weight was changed at runtime.
      - Slow-operation acknowledgement if received from the user.
      - Escalation lockout if an escalation is already pending this session.
    """
    state = context.session.state
    augments: list[str] = []

    # ── Degraded mode: BM25 unavailable (set by after_tool_callback on status check)
    if state.get("reranker:bm25_unavailable"):
        augments.append(
            "⚠️  DEGRADED MODE: BM25 sparse leg is unavailable (rank_bm25 not installed). "
            "rerank_search is running single-leg (vector only). Recall may be lower. "
            "Reflect this in pipeline.bm25_active responses."
        )

    # ── Version root pin
    version_root = state.get("reranker:version_root")
    if version_root:
        augments.append(
            f"ACTIVE SNAPSHOT: version_root={version_root[:20]}... is pinned for this session. "
            "Forward it automatically to every rerank_search call unless explicitly overridden."
        )

    # ── Alpha override
    active_alpha = state.get("reranker:active_alpha")
    if active_alpha is not None and active_alpha != 0.7:
        augments.append(
            f"RUNTIME ALPHA: Current CE blend weight is {active_alpha:.2f} (non-default). "
            "Mention this if the orchestrator asks about scoring configuration."
        )

    # ── Slow operation acknowledgement
    if state.get("reranker:slow_op_acknowledged"):
        augments.append(
            "✅  ACKNOWLEDGEMENT RECEIVED: The user/orchestrator has explicitly acknowledged "
            "the latency impact of slow operations (CE model swap or connection changes). "
            "You are now AUTHORIZED to proceed with the requested rerank_configure call."
        )

    # ── Escalation lockout
    if state.get("reranker:escalation_pending"):
        augments.append(
            "⛔  ESCALATION PENDING: An escalation was emitted this session. "
            "Do not attempt further tool calls. Return the escalation payload immediately."
        )

    if not augments:
        return ""

    runtime_block = "<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return runtime_block