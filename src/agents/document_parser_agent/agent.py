"""
agent.py — Chanoch Clerk Document Parser Agent
================================================

Architecture decision: single LlmAgent (not SequentialAgent).

Rationale:
  DocumentParser.parse() already handles the fixed internal sequence
  (layout detection → VLM inference → page save → chunk assembly) within
  the MCP server. The agent's decision surface is dynamic:
    · Single file vs. batch
    · Local path vs. base64 delivery
    · Default settings vs. per-call overrides
    · Cold vs. warm pipeline
    · Settings inspection before large batch
    · CE model configure → cache clear sequence (analogous to reranker_agent)
  A SequentialAgent would only work for the single-file-no-overrides case.
  A single LlmAgent with tool access handles all paths from one definition.

Model: LiteLlm → ollama_chat/gemma4:e4b-it-q4_K_M (local Ollama).

Toolset:
  · MCPToolset → document_parser_mcp (server.py, stdio)
    Tools: parse_document, parse_batch, configure_parser, get_parser_settings
  · SkillToolset → parse-single, batch-parse

Context optimisation:
  · static_instruction           — stable XML loaded once; Gemini prefix-cache eligible
  · instruction                  — turn-level callable, runtime augments only (not cached)
  · App.context_cache_config     — Gemini 2.0+ token-level caching (no-op for LiteLlm/Ollama)
  · App.events_compaction_config — model-agnostic sliding-window summarisation

Exports:
  app          — for `adk web` / `adk run` discovery (App takes precedence over root_agent)
  root_agent   — secondary alias for backward compat
  runner       — pre-built Runner for programmatic / test use
"""

from __future__ import annotations

import logging
import pathlib

from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset
from mcp import StdioServerParameters

from .callbacks import (
    before_agent_callback,
    after_agent_callback,
    before_model_callback,
    after_model_callback,
    before_tool_callback,
    after_tool_callback,
)
from .prompts import _STATIC_INSTRUCTION, build_instruction
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("document_parser_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

_SRC_ROOT = _AGENT_DIR.parent.parent   # src/agents/document_parser_agent → src
_SERVER_PATH = _SRC_ROOT / "document_parser" / "server.py"

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP Toolset — document_parser_mcp (stdio)
# ─────────────────────────────────────────────────────────────────────────────
# timeout=300: parse_batch with 16 workers + first cold-start model download
# can legitimately take several minutes. 5 minutes is a conservative cap.
# ─────────────────────────────────────────────────────────────────────────────

_parser_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", str(_SERVER_PATH)],
        ),
        timeout=300,
    ),
    tool_filter=[
        "parse_document",
        "parse_batch",
        "configure_parser",
        "get_parser_settings",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# SkillToolset
# ─────────────────────────────────────────────────────────────────────────────

_parse_single_skill = load_skill_from_dir(_SKILLS_DIR / "parse-single")
_batch_parse_skill = load_skill_from_dir(_SKILLS_DIR / "batch-parse")

_skill_toolset = skill_toolset.SkillToolset(
    skills=[_parse_single_skill, _batch_parse_skill]
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent
#
# static_instruction  → stable XML system prompt; prefix-cached by Gemini's
#                       context-caching layer (unchanged between turns).
#                       Never inject session state here.
#
# instruction         → turn-level callable; returns only the runtime augments
#                       block (or "" when nothing active). ADK appends this
#                       after static_instruction each turn.
# ─────────────────────────────────────────────────────────────────────────────

document_parser_agent = LlmAgent(
    name="document_parser_agent",
    description=(
        "PaddleOCRVL 1.5 document parsing sub-agent. Converts PDFs and images "
        "into structured per-page Document objects with markdown, layout-detected "
        "chunks (with bounding boxes), and Merkle integrity roots. Call for: "
        "single-document parsing, batch document parsing, VLM pipeline configuration, "
        "and parser settings inspection. Output is the canonical input format for "
        "ingestion_agent."
    ),
    model="gemini-3.1-flash-lite-preview",  # swap to _model for local Ollama dev
    static_instruction=_STATIC_INSTRUCTION,  # cached — do not put state here
    instruction=build_instruction,           # dynamic state injection per turn
    output_key="parser:agent_output",
    tools=[
        _parser_mcp,
        _skill_toolset,
    ],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# Secondary discovery alias (App takes precedence with adk CLI)
root_agent = document_parser_agent

# ─────────────────────────────────────────────────────────────────────────────
# Services
# ─────────────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# ─────────────────────────────────────────────────────────────────────────────
# Context optimisation
#
# ContextCacheConfig  — Gemini 2.0+ only; caches the stable prompt prefix.
#                       · min_tokens=2048    skip caching for short contexts
#                       · ttl_seconds=900    15-min TTL; aligns with a work session
#                       · cache_intervals=10 hard-refresh every 10 invocations
#                       No-op for LiteLlm/Ollama — safe to leave on.
#
# EventsCompactionConfig — model-agnostic sliding-window summarisation.
#                       parse_batch calls can produce large tool-response events;
#                       compaction at interval=4 keeps the context lean across
#                       multi-document sessions.
#                       · compaction_interval=4  summarise after every 4 turns
#                       · overlap_size=1         carry 1 prior event for continuity
# ─────────────────────────────────────────────────────────────────────────────

_summariser = LlmEventSummarizer(
    llm=Gemini(model="gemini-2.5-flash"),
)

# App is the CLI-discoverable top-level container (ADK v1.14.0+).
# Variable MUST be named `app` for `adk web` / `adk run` auto-discovery.
app = App(
    name="chanoch_clerk_parser",
    root_agent=document_parser_agent,
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=900,
        cache_intervals=10,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=4,  # lower than orchestrator — parse events are large
        overlap_size=1,
        summarizer=_summariser,
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Runner  (programmatic / test use)
#
# The App above is the authoritative entry point for `adk web` / `adk run`.
# Use the Runner below for direct programmatic invocation and unit tests.
# App-level caching and compaction are active only via the App execution path.
# ─────────────────────────────────────────────────────────────────────────────

runner = Runner(
    agent=document_parser_agent,
    app_name="chanoch_clerk_parser",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info(
    "document_parser_agent initialised — model=%s server=%s",
    "gemini-3.1-flash-lite-preview",
    _SERVER_PATH,
)