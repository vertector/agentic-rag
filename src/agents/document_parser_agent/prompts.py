"""
prompts.py — Document Parser Agent Dynamic Instruction Builder
===============================================================

Provides build_instruction(context) — an ADK-compatible callable that loads
system_prompt.xml as the base and appends runtime augments based on session state.

Runtime augments injected when relevant:
  - Cold-start warning (parser:warm absent)
  - Active category banner
  - Escalation lockout
  - External VLM backend reminder (when active_settings indicates non-local)
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

_STATIC_PROMPT: str | None = None


def _load_static_prompt() -> str:
    global _STATIC_PROMPT
    if _STATIC_PROMPT is None:
        try:
            _STATIC_PROMPT = _STATIC_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info(
                "Loaded system prompt from %s (%d chars)",
                _STATIC_PROMPT_PATH, len(_STATIC_PROMPT),
            )
        except FileNotFoundError:
            logger.error("system_prompt.xml not found at %s", _STATIC_PROMPT_PATH)
            _STATIC_PROMPT = (
                "<system_prompt><goal>Document parser sub-agent. "
                "Parse documents via parse_document and parse_batch.</goal></system_prompt>"
            )
    return _STATIC_PROMPT


def build_instruction(context: "InvocationContext") -> str:
    """
    Dynamic instruction builder — called by ADK on every agent turn.

    Base: loads system_prompt.xml.
    Augments (appended as a runtime context block when relevant):
      - Cold-start warning: if parser:warm is absent or False.
      - Active category: if parser:active_category is set.
      - External VLM reminder: if active_settings shows a non-local backend.
      - Escalation lockout: if parser:escalation_pending is True.
    """
    base = _load_static_prompt()
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
        return base

    runtime_block = "\n\n<!-- RUNTIME CONTEXT (injected by prompts.py) -->\n"
    runtime_block += "\n".join(f"  {a}" for a in augments)
    return base + runtime_block