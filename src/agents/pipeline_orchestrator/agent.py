"""
agent.py — Chanoch Clerk Pipeline Orchestrator

Primary user-facing agent. Delegates to three sub-agents:
  · document_parser_agent — parse PDFs/images
  · ingestion_agent       — ingest, audit, search, manage
  · reranker_agent        — hybrid rerank retrieval

Model: LiteLlm → ollama_chat/gemma4:e4b-it-q4_K_M
Architecture: LlmAgent with sub_agents (dynamic delegation via LLM routing)
"""

from __future__ import annotations

import logging
import pathlib

from google.adk.agents import LlmAgent
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
from .prompts import build_instruction
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("pipeline_orchestrator")

_AGENT_DIR = pathlib.Path(__file__).parent
_SKILLS_DIR = _AGENT_DIR / "skills"

_model = LiteLlm(
    model="ollama_chat/gemma4:e4b-it-q4_K_M",
    api_base="http://localhost:11434",
)

_intent_routing_skill = load_skill_from_dir(_SKILLS_DIR / "intent-routing")
_skill_toolset = skill_toolset.SkillToolset(skills=[_intent_routing_skill])

pipeline_orchestrator = LlmAgent(
    name="pipeline_orchestrator",
    description=(
        "Primary user-facing orchestrator for the Chanoch Clerk document "
        "intelligence platform. Routes user requests to document parsing, "
        "ingestion, or retrieval sub-agents."
    ),
    model="gemini-3.1-flash-lite-preview", #_model,
    instruction=build_instruction,
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

# adk web / adk run discovery
root_agent = pipeline_orchestrator

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

runner = Runner(
    agent=pipeline_orchestrator,
    app_name="chanoch_clerk",
    session_service=session_service,
    artifact_service=artifact_service,
)

logger.info("pipeline_orchestrator initialised")