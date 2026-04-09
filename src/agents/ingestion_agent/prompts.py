"""
prompts.py — Ingestion Agent Dynamic Instruction Builder
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
_STATIC_PROMPT: str | None = None


def _load_static_prompt() -> str:
    global _STATIC_PROMPT
    if _STATIC_PROMPT is None:
        try:
            _STATIC_PROMPT = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info("Loaded system prompt (%d chars)", len(_STATIC_PROMPT))
        except FileNotFoundError:
            logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
            _STATIC_PROMPT = "<system_prompt><goal>Ingestion sub-agent.</goal></system_prompt>"
    return _STATIC_PROMPT


def build_instruction(context: "InvocationContext") -> str:
    """
    Dynamic instruction builder. Base = system_prompt.xml.
    Runtime augments appended when relevant:
      - Disconnected warning (ingestor:connected absent/false)
      - Active version_root banner
      - Escalation lockout
      - Pending purge gate reminder
    """
    base = _load_static_prompt()
    state = context.session.state
    augments: list[str] = []

    if not state.get("ingestor:connected"):
        augments.append(
            "⚠️  NOT CONNECTED: Qdrant/Redis connectivity has not been verified this session. "
            "Call `ingest_status` before the first ingest_data call to confirm reachability."
        )

    if state.get("ingestor:version_root"):
        vr = state["ingestor:version_root"]
        augments.append(
            f"PINNED VERSION: version_root={vr[:20]}... is active for this session. "
            "Forward it automatically to every ingest_search call unless overridden."
        )

    if state.get("ingestor:escalation_pending"):
        augments.append(
            "⛔  ESCALATION PENDING: Do not attempt further tool calls. "
            "Return the escalation payload immediately."
        )

    if state.get("ingestor:purge_confirmed"):
        augments.append(
            "🗑️  PURGE CONFIRMED: ingestor:purge_confirmed is set. "
            "The next ingest_purge call will be allowed through. "
            "Ensure the correct filename is passed."
        )

    if not augments:
        return base

    runtime_block = "\n\n<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return base + runtime_block