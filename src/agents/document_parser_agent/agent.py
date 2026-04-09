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

Deployment: InMemory* for local/dev. Swap to persistent services for production.

Exports:
  root_agent   — for `adk web` / `adk run` discovery
  runner       — pre-built Runner for programmatic use
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

logger = logging.getLogger("document_parser_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

# Resolve server.py relative to src root — same pattern as reranker_agent.
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
# Agent Definition
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
    model='gemini-3.1-flash-lite-preview', #_model,
    instruction=build_instruction,  # dynamic callable
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

# Required for `adk web` / `adk run`
root_agent = document_parser_agent

# ─────────────────────────────────────────────────────────────────────────────
# Services + Runner
# ─────────────────────────────────────────────────────────────────────────────

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=document_parser_agent,
    app_name="chanoch_clerk_parser",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info(
    "document_parser_agent initialised — model=%s server=%s",
    "ollama_chat/gemma4:e4b-it-q4_K_M",
    _SERVER_PATH,
)