"""
prompts.py — Pipeline Orchestrator Dynamic Instruction Builder
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

logger = logging.getLogger("pipeline_orchestrator.prompts")

_AGENT_DIR = pathlib.Path(__file__).parent
_STATIC_PROMPT_PATH = _AGENT_DIR / "system_prompt.xml"
_STATIC_PROMPT: str | None = None


def _load_static_prompt() -> str:
    global _STATIC_PROMPT
    if _STATIC_PROMPT is None:
        try:
            _STATIC_PROMPT = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info("Loaded orchestrator prompt (%d chars)", len(_STATIC_PROMPT))
        except FileNotFoundError:
            logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
            _STATIC_PROMPT = "<system_prompt><goal>Document pipeline orchestrator.</goal></system_prompt>"
    return _STATIC_PROMPT


def build_instruction(context: "InvocationContext") -> str:
    """
    Dynamic instruction builder. Appends runtime augments when relevant:
      - Active file banner
      - Pending purge reminder
      - Escalation lockout
      - Pipeline step progress
    """
    base = _load_static_prompt()
    state = context.session.state
    augments: list[str] = []

    if state.get("orchestrator:active_file"):
        augments.append(
            f"ACTIVE FILE: '{state['orchestrator:active_file']}' is the current working document. "
            "Resolve ambiguous file references ('it', 'that file', 'this document') to this."
        )

    if state.get("orchestrator:parser_output_path"):
        augments.append(
            f"PARSER OUTPUT: '{state['orchestrator:parser_output_path']}' is ready to ingest. "
            "Pass this exact path as file_path to ingestion_agent — not the original PDF path."
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

    if not augments:
        return base

    runtime_block = "\n\n<!-- RUNTIME CONTEXT -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return base + runtime_block