"""
agent.py — Chanoch Clerk Reranker Agent
=========================================

Architecture decision: single LlmAgent (not SequentialAgent).

Rationale:
  The HybridReranker pipeline has a fixed internal sequence (RRF → CE blend),
  but the *agent's* decision surface is dynamic:
    · A simple query needs only rerank_search.
    · A cold-start session needs rerank_status first.
    · A misconfigured session needs rerank_configure before searching.
    · A CE model swap needs configure → cache_clear → search in sequence.
  A SequentialAgent would hardcode one path through these steps. A single
  LlmAgent with tool access and a strong system prompt handles all paths
  without brittle orchestration logic. The system_prompt.xml `<instructions>`
  section provides the sequential heuristic when a fixed path IS the right one.

Model: LiteLlm → ollama_chat/gemma4:e4b-it-q4_K_M (local Ollama instance).

Toolset:
  · MCPToolset → reranker_mcp FastMCP server (server.py, stdio transport)
    Tools: rerank_search, rerank_configure, rerank_status, rerank_cache_clear
  · SkillToolset → rerank-execution, citation-formatting

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
logger = logging.getLogger("reranker_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

_SRC_ROOT = _AGENT_DIR.parent.parent  # src/agents/reranker_agent → src → project root/src
_SERVER_PATH = _SRC_ROOT / "reranker_pipeline" / "server.py"

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP Toolset — reranker_mcp (stdio)
# ─────────────────────────────────────────────────────────────────────────────

_reranker_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", str(_SERVER_PATH)],
        ),
        timeout=120,  # 120s covers cold-start CE weight download (~300 MB)
    ),
    tool_filter=[
        "rerank_search",
        "rerank_configure",
        "rerank_status",
        "rerank_cache_clear",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# SkillToolset
# ─────────────────────────────────────────────────────────────────────────────

_rerank_execution_skill = load_skill_from_dir(_SKILLS_DIR / "rerank-execution")
_citation_formatting_skill = load_skill_from_dir(_SKILLS_DIR / "citation-formatting")

_skill_toolset = skill_toolset.SkillToolset(
    skills=[_rerank_execution_skill, _citation_formatting_skill]
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

reranker_agent = LlmAgent(
    name="reranker_agent",
    description=(
        "Citation-aware hybrid reranker sub-agent. Executes two-stage RRF + "
        "Cross-Encoder reranking over ingested Qdrant document collections. "
        "Call for: document retrieval, ranked chunk selection, point-in-time "
        "Merkle snapshot queries, reranker diagnostics, and CE configuration."
    ),
    model="gemini-3.1-flash-lite-preview",  # swap to _model for local Ollama dev
    static_instruction=_STATIC_INSTRUCTION,  # cached — do not put state here
    instruction=build_instruction,           # dynamic state injection per turn
    output_key="reranker:agent_output",
    tools=[
        _reranker_mcp,
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
root_agent = reranker_agent

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
#                       rerank_search responses include ranked chunk payloads
#                       with scores, metadata, and bounding boxes. Compaction
#                       at interval=4 prevents context bloat in multi-query
#                       research sessions.
#                       · compaction_interval=4  summarise after every 4 turns
#                       · overlap_size=1         carry 1 prior event for continuity
# ─────────────────────────────────────────────────────────────────────────────

_summariser = LlmEventSummarizer(
    llm=Gemini(model="gemini-2.5-flash"),
)

# App is the CLI-discoverable top-level container (ADK v1.14.0+).
# Variable MUST be named `app` for `adk web` / `adk run` auto-discovery.
app = App(
    name="chanoch_clerk_reranker",
    root_agent=reranker_agent,
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=900,
        cache_intervals=10,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=4,  # lower than orchestrator — ranked chunk payloads are large
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
    agent=reranker_agent,
    app_name="chanoch_clerk_reranker",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info(
    "reranker_agent initialised — model=%s server=%s",
    "gemini-3.1-flash-lite-preview",
    _SERVER_PATH,
)