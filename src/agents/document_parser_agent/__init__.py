"""
document_parser_agent — Chanoch Clerk document parsing sub-agent.

Public exports:
    document_parser_agent  — the LlmAgent instance (for sub-agent wiring)
    root_agent             — alias required by `adk web` / `adk run`
    runner                 — pre-built Runner for programmatic invocation
"""

from .agent import document_parser_agent, root_agent, runner

__all__ = ["document_parser_agent", "root_agent", "runner"]