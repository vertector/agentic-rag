"""
src/agents/document_parser_agent/prompts.py
============================================
Two-layer instruction factory for the document parser agent.

ADK exposes three instruction fields on LlmAgent.  We use two of them:

  static_instruction (str | types.Content)
    ├── Placed in the SYSTEM role at the start of every request.
    ├── Content is IDENTICAL across all turns, so Gemini can prefix-cache it.
    ├── Eligible for both implicit prefix caching and explicit ContextCacheConfig.
    └── Contains: persona, goal, tool_guidance, constraints, examples — the
        stable 11-section XML contract that never changes at runtime.

  instruction (callable → str)       ← the "turn instruction"
    ├── ADK injects the returned string as a USER-role message immediately
    │   before the actual user query (recency bias — model attends to it first).
    ├── Rebuilt on EVERY turn from live session state.
    ├── NOT cached — deliberately small and turn-specific.
    └── Contains: VLM warm-up status, last known config, token budget note.

When static_instruction is absent, instruction falls back to the SYSTEM role
and loses cache eligibility — which is why the split matters.

Public API
-----------
  load_static_instruction() -> str          called once at module import
  build_turn_instruction(ctx) -> str        passed as instruction= callable
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

from google.adk.agents.readonly_context import ReadonlyContext

logger = logging.getLogger(__name__)

# Path to the system prompt XML, relative to this file.
_PROMPT_FILE = pathlib.Path(__file__).parent / "system_prompt.xml"

# Token budget ceiling at which the turn instruction nudges compact output.
TOKEN_BUDGET_WARNING = int(os.environ.get("PARSER_TOKEN_BUDGET_WARNING", "50000"))


# ---------------------------------------------------------------------------
# Static instruction — loaded ONCE, placed in SYSTEM role, cache-eligible
# ---------------------------------------------------------------------------

def load_static_instruction() -> str:
    """
    Load system_prompt.xml and return its content as a plain string.

    Called once at module import time (in agent.py:  static_instruction=load_static_instruction()).
    The result is set-and-forget — it goes to the SYSTEM role in every request
    and never changes while the process is running.

    If you need to pick up a prompt edit without restarting the process, call
    this again and rebuild root_agent — or use `adk web` which hot-reloads on
    file changes automatically.

    Returns:
        str: The full XML system instruction, or a minimal fallback if the
             file is missing (avoids a hard crash on misconfigured deploys).
    """
    try:
        content = _PROMPT_FILE.read_text(encoding="utf-8")
        logger.debug("[PROMPTS] Loaded static_instruction from %s", _PROMPT_FILE)
        return content
    except FileNotFoundError:
        logger.warning(
            "[PROMPTS] system_prompt.xml not found at %s — using fallback.", _PROMPT_FILE
        )
        return _FALLBACK_STATIC
    except OSError as exc:
        logger.error("[PROMPTS] Failed to read system_prompt.xml: %s", exc)
        return _FALLBACK_STATIC


# ---------------------------------------------------------------------------
# Turn instruction — callable, rebuilt every turn, injected as USER role
# ---------------------------------------------------------------------------

def build_turn_instruction(context: ReadonlyContext) -> str:
    """
    Build the per-turn steering instruction from live session state.

    ADK calls this callable on every agent turn and injects the returned
    string as a USER-role message immediately before the actual user query.
    It must stay small — only include content that genuinely varies turn-by-turn.

    The static 11-section XML behavioral contract is NOT included here;
    it lives in static_instruction where it can be cached.

    Args:
        context: ReadonlyContext provided by ADK — gives read-only access
                 to session.state without risking accidental mutations.

    Returns:
        str: A compact XML block with three dynamic signals for this turn.
    """
    state = context.session.state

    return (
        "<turn_context>\n"
        f"  <warm_up_status>{_warm_up_note(state)}</warm_up_status>\n"
        f"  <last_known_parser_config>\n{_config_block(state)}\n"
        "  </last_known_parser_config>\n"
        f"  <token_budget>{_budget_note(state)}</token_budget>\n"
        "</turn_context>"
    )



# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _warm_up_note(state: dict) -> str:
    if state.get("app:parser_warmed_up", False):
        return "VLM is warm — subsequent parse calls respond in 2–10 seconds."
    return (
        "First parse of this session. PaddleOCRVL model loading takes ~30 seconds. "
        "Set user expectations before invoking parser_parse or parser_parse_batch."
    )


def _config_block(state: dict) -> str:
    last_config: dict = state.get("parser_last_config", {})
    if not last_config:
        return "    Unknown — call parser_status to retrieve current settings."
    try:
        return json.dumps(last_config, indent=4)
    except (TypeError, ValueError):
        return "    (config not serialisable)"


def _budget_note(state: dict) -> str:
    total: int = state.get("session_token_total", 0)
    if total >= TOKEN_BUDGET_WARNING:
        return (
            f"Session tokens ({total:,}) exceeded warning threshold "
            f"({TOKEN_BUDGET_WARNING:,}). Use compact ✓/✗ summaries over full JSON."
        )
    return f"Session tokens used: {total:,} / {TOKEN_BUDGET_WARNING:,}."


# ---------------------------------------------------------------------------
# Fallback static instruction (used only when system_prompt.xml is missing)
# ---------------------------------------------------------------------------

_FALLBACK_STATIC = """
<system_instruction>
  <background_information>
    You are the Document Parser Agent. Parse PDF and image files into structured
    Markdown Document objects using PaddleOCRVL. First stage of the RAG pipeline.
  </background_information>
  <goal>
    Parse documents. Surface results and errors. Do not ingest or search.
  </goal>
  <tool_guidance>
    parser_parse        — single file (path or base64)
    parser_parse_batch  — multiple file paths
    parser_configure    — change OCR/VLM settings (requires confirmation)
    parser_status       — check connectivity and active settings
  </tool_guidance>
</system_instruction>
""".strip()
