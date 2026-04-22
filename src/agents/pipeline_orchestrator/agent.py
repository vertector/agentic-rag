"""
agent.py — Chanoch Clerk Pipeline Orchestrator

Primary user-facing agent. Delegates to three sub-agents:
  · document_parser_agent — parse PDFs/images with high-fidelity table extraction
  · ingestion_agent       — ingest, audit, search, manage Corpora, and version history
  · reranker_agent        — hybrid rerank retrieval with point-in-time version pinning

Model: LiteLlm → ollama_chat/gemma4:e4b-it-q4_K_M (dev) / gemini-3.1-flash-lite-preview (prod)
Architecture: LlmAgent with sub_agents (dynamic delegation via LLM routing)

Context optimisation:
  · static_instruction           — stable system prompt prefix, Gemini prompt-cache eligible
  · instruction                  — turn-level callable, injects live session state (not cached)
  · App.context_cache_config     — Gemini 2.0+ token-level caching (no-op for LiteLlm/Ollama)
  · App.events_compaction_config — model-agnostic sliding-window summarisation
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
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset

from agents.document_parser_agent import document_parser_agent
from agents.ingestion_agent import ingestion_agent
from agents.reranker_agent import reranker_agent

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
logger = logging.getLogger("pipeline_orchestrator")

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

# ---------------------------------------------------------------------------
# Toolset
# ---------------------------------------------------------------------------

_intent_routing_skill = load_skill_from_dir(_SKILLS_DIR / "intent-routing")
_skill_toolset = skill_toolset.SkillToolset(skills=[_intent_routing_skill])

# ---------------------------------------------------------------------------
# Agent
#
# static_instruction  → stable XML system prompt; prefix-cached by Gemini's
#                       context-caching layer (unchanged between turns).
#                       Never inject session state here.
#
# instruction         → turn-level callable; called each invocation to build
#                       the dynamic section (active_docs, session state, etc.).
#                       Not cached — changes per turn by design.
# ---------------------------------------------------------------------------

pipeline_orchestrator = LlmAgent(
    name="pipeline_orchestrator",
    description=(
        "Primary user-facing orchestrator for the Chanoch Clerk document "
        "intelligence platform. Routes user requests to document parsing, "
        "ingestion, retrieval, Corpus management, and Merkle integrity "
        "auditing sub-agents."
    ),
    model="gemini-3.1-flash-lite-preview",  # swap to _model for local Ollama dev
    static_instruction=_STATIC_INSTRUCTION,  # cached — do not put state here
    instruction=build_instruction,           # dynamic state injection per turn
    tools=[_skill_toolset],
    sub_agents=[
        document_parser_agent,
        ingestion_agent,
        reranker_agent,
    ],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# adk web / adk run secondary discovery alias (App takes precedence)
root_agent = pipeline_orchestrator

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# ---------------------------------------------------------------------------
# Context optimisation
#
# ContextCacheConfig  — Gemini 2.0+ only; caches the stable prompt prefix so
#                       repeated invocations skip re-encoding it on the server.
#                       · min_tokens=2048    only activate caching above 2 K tokens
#                       · ttl_seconds=900    15-minute TTL; covers a typical work session
#                       · cache_intervals=10 hard-refresh the cache every 10 invocations
#                       Safe no-op when model=LiteLlm/Ollama — leave the config in place.
#
# EventsCompactionConfig — model-agnostic sliding-window summarisation of event history.
#                       Fires every `compaction_interval` completed invocations; retains
#                       `overlap_size` events from the prior window for continuity.
#                       Uses gemini-2.5-flash as the cheap, fast summariser.
#                       · compaction_interval=5  summarise after every 5 turns
#                       · overlap_size=1         carry 1 prior event into each new summary
# ---------------------------------------------------------------------------

_summariser = LlmEventSummarizer(
    llm=Gemini(model="gemini-2.5-flash"),
)

# App is the CLI-discoverable top-level container (ADK Python v1.14.0+).
# Variable MUST be named `app` for `adk web` / `adk run` auto-discovery.
app = App(
    name="chanoch_clerk",
    root_agent=pipeline_orchestrator,
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=900,
        cache_intervals=10,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,
        overlap_size=1,
        summarizer=_summariser,
    ),
)

# ---------------------------------------------------------------------------
# Runner  (programmatic / test use)
#
# The App above is the authoritative entry point for `adk web` / `adk run`.
# Use the Runner below for direct programmatic invocation and unit tests.
# App-level caching and compaction are active only via the App execution path;
# the Runner below bypasses them — suitable for dev and testing.
# ---------------------------------------------------------------------------

runner = Runner(
    agent=pipeline_orchestrator,
    app_name="chanoch_clerk",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info("pipeline_orchestrator initialised (App + Runner)")