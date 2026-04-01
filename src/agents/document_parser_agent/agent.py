"""
src/agents/document_parser_agent/agent.py
==========================================
ADK agent entrypoint — defines `root_agent`.

This file is intentionally thin.  All logic lives in sibling modules:
    callbacks.py — the six ADK callback hooks
    prompts.py   — build_instruction() dynamic prompt factory
    system_prompt.xml — 11-section behavioral contract

ADK CLI discovery requirement:
    `adk web` / `adk run` import this module and look for `root_agent`
    at module level.  Do not rename that variable.

Run from src/agents/:
    adk web                              # dev UI — all agents visible
    adk run document_parser_agent "..."  # single-shot CLI
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp import StdioServerParameters

from .callbacks import (
    after_agent_callback,
    after_model_callback,
    after_tool_callback,
    before_agent_callback,
    before_model_callback,
    before_tool_callback,
)
from .prompts import build_turn_instruction, load_static_instruction

# Load .env from this agent's directory (does not override already-set env vars)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — set in .env or export before running
# ---------------------------------------------------------------------------

# Absolute path to the document_parser MCP server
_MCP_SERVER_PATH = os.environ.get(
    "PARSER_MCP_SERVER_PATH",
    # Derive a sensible default relative to this repo's src/ layout:
    # src/agents/document_parser_agent/../../../src/document_parser/server.py
    str(
        (
            __import__("pathlib").Path(__file__).parents[3]
            / "src" / "document_parser" / "server.py"
        ).resolve()
    ),
)

# Python interpreter that has document_parser's dependencies installed
_MCP_PYTHON = os.environ.get(
    "PARSER_MCP_PYTHON",
    "python",  # falls back to PATH python; override in .env for venv isolation
)

# ---------------------------------------------------------------------------
# MCPToolset — synchronous instantiation (required for Cloud Run / Vertex AI)
# ---------------------------------------------------------------------------

_parser_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=_MCP_PYTHON,
            args=[_MCP_SERVER_PATH],
            env={**os.environ},   # forward full env so the MCP server gets its own vars
        ),
        # Generous timeout: first call waits for PaddleOCRVL model load (~30s)
        timeout=120,
    ),
    # Explicit allowlist — new tools added to the MCP server won't auto-appear here
    tool_filter=[
        "parse_document",
        "parse_batch",
        "configure_parser",
        "get_parser_settings",
    ],
)

# ---------------------------------------------------------------------------
# Native FunctionTool — session cache inspector
# ---------------------------------------------------------------------------

def get_parser_cache_stats(tool_context: ToolContext) -> dict:
    """
    List all parse results cached in the current session, with staleness status.

    For each cached file, the fingerprint recorded at parse time (mtime:size)
    is compared against the file's current fingerprint on disk. A mismatch means
    the file has changed since it was last parsed and the cached result should
    not be trusted — the agent should re-parse before ingesting.

    Use this before re-parsing a file to check whether a fresh call is needed,
    or before handing off to the ingestion agent to confirm the output is current.

    Returns:
        dict: {
            "status": "success",
            "cached_parse_count": int,
            "stale_count": int,          # entries whose source file has changed
            "entries": [
                {
                    "file_path":   str,
                    "output_path": str,
                    "page_count":  int,
                    "parsed_at":   str,   # ISO-8601 UTC
                    "fingerprint_at_parse": str,  # "mtime:size" when parsed
                    "fingerprint_now":     str,   # "mtime:size" right now
                    "is_stale":    bool,  # True if file changed after parsing
                }
            ]
        }
    """
    from .callbacks import _file_fingerprint  # local import avoids circular ref

    entries = []
    for key, value in tool_context.state.items():
        if not key.startswith("temp:parsed_"):
            continue
        if not isinstance(value, dict):
            continue

        file_path         = value.get("file_path", "")
        fp_at_parse       = value.get("fingerprint", "unknown")
        fp_now            = _file_fingerprint(file_path) if file_path else "unknown"
        # base64-delivered files have no on-disk path; they can never go stale
        is_stale          = (
            fp_at_parse != fp_now
            and fp_at_parse not in ("base64", "missing", "unknown")
            and fp_now     not in ("missing",)
        )

        entries.append({
            "file_path":             file_path,
            "output_path":           value.get("output_path", ""),
            "page_count":            value.get("page_count"),
            "parsed_at":             value.get("parsed_at", ""),
            "fingerprint_at_parse":  fp_at_parse,
            "fingerprint_now":       fp_now,
            "is_stale":              is_stale,
        })

    stale_count = sum(1 for e in entries if e["is_stale"])

    return {
        "status":             "success",
        "cached_parse_count": len(entries),
        "stale_count":        stale_count,
        "entries":            entries,
    }

# ---------------------------------------------------------------------------
# Context caching configuration
# ---------------------------------------------------------------------------
# Wrapping root_agent in App enables Gemini prefix caching on static_instruction.
# The static XML prompt is ~2–3k tokens; caching saves that prefill cost on
# every turn after the first, reducing both latency (~100–400ms) and token spend.
#
# ContextCacheConfig settings:
#   min_tokens     — don't bother caching if the prefix is tiny (guard for dev)
#   ttl_seconds    — how long the cache entry lives; set to match prompt stability
#   cache_intervals— force a cache refresh every N invocations (staleness guard)
#
# App is the runner-facing wrapper; root_agent is still the ADK CLI entrypoint.
# adk web / adk run discover root_agent; App is only used when constructing Runner.

try:
    from google.adk.apps import App
    from google.adk.agents.context_cache_config import ContextCacheConfig

    _CACHE_CONFIG = ContextCacheConfig(
        min_tokens=1024,       # only cache if static prefix > 1k tokens
        ttl_seconds=3600,      # 1 hour — system_prompt.xml rarely changes in prod
        cache_intervals=10,    # force refresh every 10 invocations as a guard
    )
    _HAS_CONTEXT_CACHE = True
except ImportError:
    # ContextCacheConfig was added in ADK v1.15.0.
    # Gracefully degrade on older installs — agent still works, just without caching.
    _HAS_CONTEXT_CACHE = False
    logger.warning(
        "[AGENT] ContextCacheConfig not available (requires ADK >= 1.15.0). "
        "Upgrade with: pip install google-adk>=1.15.0"
    )

# ---------------------------------------------------------------------------
# root_agent — the sole export that ADK CLI requires
# ---------------------------------------------------------------------------

root_agent = LlmAgent(
    # ── Identity ────────────────────────────────────────────────────────────
    name="document_parser_agent",
    description=(
        "First stage of the document intelligence pipeline. "
        "Parses PDF and image files into structured per-page Document objects "
        "using PaddleOCRVL. "
        "Tools: parse_document, parse_batch, configure_parser, get_parser_settings. "
        "Delegates ingestion to ingestion_agent and search to reranker_agent."
    ),

    # Flash: optimised for pipeline throughput, not reasoning depth.
    # Note: Kimi-2.5 local model can be targeted by wrapping this in LiteLLM
    # or updating the process environment if the user switched models.
    model="gemini-2.5-flash",

    # ── Static instruction — SYSTEM role, cache-eligible ────────────────────
    # Loaded once at import time. Goes to Gemini's system_instruction field.
    # Stable across all turns → eligible for Gemini prefix caching.
    # Contains: persona, goal, tool_guidance, constraints, examples (11 sections).
    static_instruction=load_static_instruction(),

    # ── Turn instruction — USER role, rebuilt every turn ────────────────────
    # ADK injects this as a USER-role message immediately before the user query.
    # Returns only what genuinely varies: warm-up status, last config, token budget.
    # Kept small intentionally — this is never cached and is resent each turn.
    instruction=build_turn_instruction,

    # ── Tools ────────────────────────────────────────────────────────────────
    tools=[
        _parser_mcp,              # MCP: parser_parse, parser_parse_batch,
                                  #      parser_configure, parser_status
        get_parser_cache_stats,   # Native: session cache + staleness inspector
    ],

    # ── Generation config ───────────────────────────────────────────────────
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2048,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            )
        ],
    ),

    # ── output_key ──────────────────────────────────────────────────────────
    output_key="last_parser_response",

    # ── Callbacks ────────────────────────────────────────────────────────────
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# ---------------------------------------------------------------------------
# parser_app — use this instead of root_agent when constructing Runner
# in session_runner.py to get context caching.
#
#   from document_parser_agent.agent import parser_app
#   runner = Runner(agent=parser_app, ...)
#
# adk web / adk run still discover root_agent at module level — they don't
# need parser_app.
# ---------------------------------------------------------------------------

if _HAS_CONTEXT_CACHE:
    parser_app = App(
        name="document_parser_app",
        root_agent=root_agent,
        context_cache_config=_CACHE_CONFIG,
    )
else:
    parser_app = root_agent  # type: ignore[assignment]  — fallback: no caching
