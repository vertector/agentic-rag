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
  Gemma 4 e4b-it-q4_K_M is a 4-bit quantised instruction-tuned model — fast
  enough for sub-agent use cases while fitting in commodity VRAM.

Toolset:
  · MCPToolset → reranker_mcp FastMCP server (server.py, stdio transport)
    Tools: rerank_search, rerank_configure, rerank_status, rerank_cache_clear
  · SkillToolset → rerank-execution, citation-formatting

Deployment target: local (InMemorySessionService + InMemoryArtifactService).
  Swap to DatabaseSessionService + GCS/VertexAI ArtifactService for production.

Module exports:
  root_agent   — required by `adk web` / `adk run` for automatic discovery.
  runner       — pre-built Runner for programmatic use (session_runner.py or tests).
"""

from __future__ import annotations

import logging
import pathlib

from google.adk.agents import LlmAgent
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
from .prompts import build_instruction
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("reranker_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

# Path to the reranker_pipeline server.py — resolved relative to the project
# src root so it works regardless of the CWD at invocation.
_SRC_ROOT = _AGENT_DIR.parent.parent  # src/agents/reranker_agent → src → project root/src
_SERVER_PATH = _SRC_ROOT / "reranker_pipeline" / "server.py"

# ─────────────────────────────────────────────────────────────────────────────
# Model — LiteLlm → local Ollama
# ─────────────────────────────────────────────────────────────────────────────
# LiteLlm is the ADK bridge for non-Google models.
# `ollama_chat/` prefix tells LiteLLM to use the Ollama chat completion endpoint.
# api_base must point to the running Ollama instance (default port 11434).
# ─────────────────────────────────────────────────────────────────────────────

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

# ─────────────────────────────────────────────────────────────────────────────
# MCP Toolset — reranker_mcp (stdio)
# ─────────────────────────────────────────────────────────────────────────────
# Spawns server.py as a subprocess via `uv run python`.
# tool_filter restricts ADK to only expose the 4 reranker tools, preventing
# accidental exposure of any future tools added to server.py.
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
# SkillToolset — rerank-execution + citation-formatting
# ─────────────────────────────────────────────────────────────────────────────
# Skills are loaded from the `skills/` directory adjacent to agent.py.
# Each skill directory name must match the `name` field in its SKILL.md.
# ─────────────────────────────────────────────────────────────────────────────

_rerank_execution_skill = load_skill_from_dir(_SKILLS_DIR / "rerank-execution")
_citation_formatting_skill = load_skill_from_dir(_SKILLS_DIR / "citation-formatting")

_skill_toolset = skill_toolset.SkillToolset(
    skills=[_rerank_execution_skill, _citation_formatting_skill]
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent Definition
# ─────────────────────────────────────────────────────────────────────────────

reranker_agent = LlmAgent(
    name="reranker_agent",
    description=(
        "Citation-aware hybrid reranker sub-agent. Executes two-stage RRF + "
        "Cross-Encoder reranking over ingested Qdrant document collections. "
        "Call for: document retrieval, ranked chunk selection, point-in-time "
        "Merkle snapshot queries, reranker diagnostics, and CE configuration."
    ),
    model="gemini-3.1-flash-lite-preview", #_model,

    # Dynamic instruction — personalised per turn from session state.
    # Loads system_prompt.xml as the base and appends runtime augments.
    instruction=build_instruction,
    output_key="reranker:agent_output",
    tools=[
        _reranker_mcp,     # MCP tools: rerank_search, configure, status, cache_clear
        _skill_toolset,    # Skills: rerank-execution, citation-formatting
    ],

    # ── All 6 callbacks wired for production observability
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# Required alias: `adk web` and `adk run` discover the agent via `root_agent`.
# The pipeline_orchestrator imports `reranker_agent` directly as a sub-agent.
root_agent = reranker_agent

# ─────────────────────────────────────────────────────────────────────────────
# Session & Artifact Services + Runner (for local/test use)
# ─────────────────────────────────────────────────────────────────────────────
# InMemory* services are appropriate for local dev and testing.
# Production: swap to a persistent SessionService and a cloud ArtifactService.
# ─────────────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=reranker_agent,
    app_name="chanoch_clerk_reranker",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info(
    "reranker_agent initialised — model=%s server=%s",
    "ollama_chat/gemma4:e4b-it-q4_K_M",
    _SERVER_PATH,
)