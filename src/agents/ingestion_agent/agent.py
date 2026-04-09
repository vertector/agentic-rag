"""
agent.py — Chanoch Clerk Ingestion Agent

Architecture: single LlmAgent.

Tools: 8 MCP tools (ingest_data, ingest_audit, ingest_search, ingest_history,
ingest_purge, ingest_sync, ingest_configure, ingest_status) + 2 Skills.

Model: LiteLlm → ollama_chat/gemma4:e4b-it-q4_K_M.
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
logger = logging.getLogger("ingestion_agent")

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"
_SRC_ROOT = _AGENT_DIR.parent.parent
_SERVER_PATH = _SRC_ROOT / "ingestion_pipeline" / "server.py"

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

# timeout=120: ingest_data with large documents.json + cold embedding model
_ingestion_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", str(_SERVER_PATH)],
        ),
        timeout=120,
    ),
    tool_filter=[
        "ingest_data",
        "ingest_audit",
        "ingest_search",
        "ingest_history",
        "ingest_purge",
        "ingest_sync",
        "ingest_configure",
        "ingest_status",
    ],
)

_ingest_document_skill = load_skill_from_dir(_SKILLS_DIR / "ingest-document")
_audit_recovery_skill = load_skill_from_dir(_SKILLS_DIR / "audit-and-recovery")

_skill_toolset = skill_toolset.SkillToolset(
    skills=[_ingest_document_skill, _audit_recovery_skill]
)

ingestion_agent = LlmAgent(
    name="ingestion_agent",
    description=(
        "Versioned Merkle-tree document ingestion sub-agent. Stores parsed "
        "Document objects (from document_parser_agent) into Qdrant with "
        "Redis-backed integrity proofs. Supports idempotent re-ingestion, "
        "point-in-time version pinning, Merkle integrity auditing, Redis "
        "state recovery, and permanent document purge. Call after parsing, "
        "before reranking."
    ),
    model="gemini-3.1-flash-lite-preview", #_model,
    instruction=build_instruction,
    tools=[_ingestion_mcp, _skill_toolset],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

root_agent = ingestion_agent

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=ingestion_agent,
    app_name="chanoch_clerk_ingestion",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info("ingestion_agent initialised — server=%s", _SERVER_PATH)