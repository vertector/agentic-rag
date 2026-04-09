"""
prompts.py — Reranker Agent Dynamic Instruction Builder
=========================================================

Provides both the static instruction loader (reads system_prompt.xml) and a
dynamic instruction callable for cases where the instruction needs to change
based on session state (e.g. degraded-mode when BM25 is unavailable, or when
a specific Merkle snapshot is pinned for the session).

Usage in agent.py:

    from .prompts import build_instruction

    agent = LlmAgent(
        ...
        instruction=build_instruction,  # callable → dynamic per-turn
    )
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

# Cache the static XML after first read — it doesn't change at runtime.
_STATIC_PROMPT: str | None = None


def _load_static_prompt() -> str:
    """Read and cache system_prompt.xml from disk."""
    global _STATIC_PROMPT
    if _STATIC_PROMPT is None:
        try:
            _STATIC_PROMPT = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info("Loaded system prompt from %s (%d chars)", _STATIC_PROMPT_PATH, len(_STATIC_PROMPT))
        except FileNotFoundError:
            logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
            _STATIC_PROMPT = "<system_prompt><goal>Hybrid document reranker sub-agent.</goal></system_prompt>"
    return _STATIC_PROMPT


def build_instruction(context: "InvocationContext") -> str:
    """
    Dynamic instruction builder — called by ADK on every agent turn.

    Base: always loads system_prompt.xml.
    Augmentations (appended as a runtime context block):
      - Degraded-mode warning if BM25 is known unavailable (from state).
      - Active Merkle snapshot banner if version_root is pinned.
      - Alpha override note if CE weight was changed at runtime.
      - Escalation lockout if an escalation is already pending this session.

    Returns:
        str: The complete instruction string to use for this turn.
    """
    base = _load_static_prompt()
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

    # ── Escalation lockout
    if state.get("reranker:escalation_pending"):
        augments.append(
            "⛔  ESCALATION PENDING: An escalation was emitted this session. "
            "Do not attempt further tool calls. Return the escalation payload immediately."
        )

    if not augments:
        return base

    runtime_block = "\n\n<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return base + runtime_block