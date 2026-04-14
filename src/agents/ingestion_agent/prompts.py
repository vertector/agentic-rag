"""
prompts.py — Ingestion Agent Dynamic Instruction Builder

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

import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger("ingestion_agent.prompts")

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
        logger.info("Loaded system prompt (%d chars)", len(text))
        return text
    except FileNotFoundError:
        logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
        return "<system_prompt><goal>Ingestion sub-agent.</goal></system_prompt>"


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
      - Parsed document path from orchestrator
      - Disconnected warning (ingestor:connected absent/false)
      - Active version_root banner
      - Escalation lockout
      - Pending purge gate reminder
    """
    state = context.session.state
    augments: list[str] = []

    # ── Parsed path forwarded from orchestrator
    if state.get("orchestrator:parser_output_path"):
        path = state["orchestrator:parser_output_path"]
        augments.append(
            f"PARSED DOCUMENT AVAILABLE: The absolute path to the parsed results is '{path}'. "
            "Use this as the `file_path` for `ingest` calls unless the orchestrator "
            "explicitly specifies a different path."
        )

    # ── Connectivity check
    if not state.get("ingestor:connected"):
        augments.append(
            "⚠️  NOT CONNECTED: Qdrant/Redis connectivity has not been verified this session. "
            "Call `status` before the first ingest call to confirm reachability."
        )

    # ── Pinned version root
    if state.get("ingestor:version_root"):
        vr = state["ingestor:version_root"]
        augments.append(
            f"PINNED VERSION: version_root={vr[:20]}... is active for this session. "
            "Forward it automatically to every search call unless overridden."
        )

    # ── Escalation lockout
    if state.get("ingestor:escalation_pending"):
        augments.append(
            "⛔  ESCALATION PENDING: Do not attempt further tool calls. "
            "Return the escalation payload immediately."
        )

    # ── Purge gate
    if state.get("ingestor:purge_confirmed"):
        augments.append(
            "🗑️  PURGE CONFIRMED: ingestor:purge_confirmed is set. "
            "The next purge call will be allowed through. "
            "Ensure the correct filename is passed."
        )

    if not augments:
        return ""

    runtime_block = "<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return runtime_block