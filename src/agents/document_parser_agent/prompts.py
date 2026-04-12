"""
prompts.py — Document Parser Agent Dynamic Instruction Builder
===============================================================

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

logger = logging.getLogger("document_parser_agent.prompts")

_AGENT_DIR = pathlib.Path(__file__).parent
_STATIC_PROMPT_PATH = _AGENT_DIR / "system_prompt.xml"

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
        return (
            "<system_prompt><goal>Document parser sub-agent. "
            "Parse documents via parse_document and parse_batch.</goal></system_prompt>"
        )


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
      - Cold-start warning: if parser:warm is absent or False.
      - Active category: if parser:active_category is set.
      - External VLM reminder: if active_settings shows a non-local backend.
      - Escalation lockout: if parser:escalation_pending is True.
    """
    state = context.session.state
    augments: list[str] = []

    # ── Cold-start warning
    if not state.get("parser:warm"):
        augments.append(
            "⚠️  COLD START: PaddleOCRVL has not been initialised this session. "
            "The first parse_document or parse_batch call will take ~30 seconds "
            "for model loading. Warn the orchestrator proactively before calling."
        )

    # ── Active category
    category = state.get("parser:active_category")
    if category:
        augments.append(
            f"ACTIVE CATEGORY: '{category}' is set for this session. "
            "Tag documents with this category when building log entries."
        )

    # ── External VLM backend check
    active_settings_raw = state.get("parser:active_settings")
    if active_settings_raw:
        try:
            settings = json.loads(active_settings_raw)
            backend = settings.get("vl_rec_backend", "local")
            if backend not in ("local", "native", None):
                url = settings.get("vl_rec_server_url", "<not set>")
                augments.append(
                    f"EXTERNAL VLM BACKEND: backend='{backend}' at {url}. "
                    "If parse_document fails with a connection error, the external "
                    "VLM server may be down. Call get_parser_settings to confirm."
                )
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Escalation lockout
    if state.get("parser:escalation_pending"):
        augments.append(
            "⛔  ESCALATION PENDING: An escalation was emitted this session. "
            "Do not attempt further tool calls. Return the escalation payload immediately."
        )

    if not augments:
        return ""

    runtime_block = "<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return runtime_block