"""
session_runner.py — Chanoch Clerk Reranker Agent Demo Loop
===========================================================

Demonstrates the full agentic reranking lifecycle:

  1. Session creation with pre-seeded state (category, optional version_root)
  2. Standard hybrid rerank query
  3. Point-in-time query pinned to a Merkle snapshot
  4. Runtime CE model hot-swap with cache flush
  5. Status check + cache diagnostics
  6. Graceful MCP server teardown via async context manager

Usage:
    uv run python -m src.agents.reranker_agent.session_runner

Environment (set in .env or environment):
    QDRANT_URL           — default: http://localhost:6333
    REDIS_HOST           — default: localhost
    REDIS_PORT           — default: 6379
    COLLECTION_BASE_NAME — default: secure_rag
    EMBED_MODEL_NAME     — default: BAAI/bge-small-en-v1.5
    OLLAMA_HOST          — default: http://localhost:11434 (used by LiteLlm)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

from google.genai import types

from .agent import runner, session_service, artifact_service, reranker_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_runner")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME = "chanoch_clerk_reranker"
USER_ID = "pipeline_orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# Core invocation helper
# ─────────────────────────────────────────────────────────────────────────────

async def invoke(
    session_id: str,
    message: str,
    *,
    print_response: bool = True,
) -> Optional[str]:
    """
    Send a single message to the reranker agent and collect the final response.

    Args:
        session_id: Active session identifier.
        message:    User/orchestrator message text.
        print_response: If True, prints the response to stdout.

    Returns:
        The agent's final response text, or None if no text response was emitted.
    """
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )

    final_response: Optional[str] = None

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text

    if print_response and final_response:
        print(f"\n{'─'*60}")
        print(f"AGENT RESPONSE:\n{final_response}")
        print('─'*60)

    return final_response


# ─────────────────────────────────────────────────────────────────────────────
# Demo loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_demo():
    """
    Full agentic reranking demo — illustrates all primary agent workflows.
    """
    session_id = f"demo_{uuid.uuid4().hex[:8]}"

    # ── 1. Create session with pre-seeded state
    # Pre-seeding category and a pinned version_root so the agent picks them up
    # via state injection in build_instruction().
    initial_state = {
        "reranker:active_category": "legal",
        # version_root: leave None for the first few calls (active version)
    }

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state=initial_state,
    )
    logger.info("Session created: %s", session_id)

    # ── 2. Standard hybrid rerank
    print("\n═══ STEP 1: Standard hybrid rerank (category=legal) ═══")
    await invoke(
        session_id,
        "Rerank for query: 'right to a fair trial and due process'. "
        "Return top 3 results with citation text.",
    )

    # ── 3. Retrieve Merkle snapshot from session state and pin it
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    last_results_raw = session.state.get("reranker:last_results", "[]")
    last_results = json.loads(last_results_raw)

    version_root = None
    if last_results:
        version_root = last_results[0].get("citation", {}).get("version_root")

    if version_root:
        print(f"\n═══ STEP 2: Point-in-time rerank (pinned root={version_root[:12]}...) ═══")
        # Inject version_root into state (simulates orchestrator setting it)
        session.state["reranker:version_root"] = version_root
        await session_service.update_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
            state=session.state,
        )
        await invoke(
            session_id,
            "Rerank for query: 'arbitrary detention and freedom of assembly'. "
            "Use the pinned version_root from session state. Top 5 results.",
        )
    else:
        print("\n[SKIP] No version_root available from step 1 results.")

    # ── 4. Status check
    print("\n═══ STEP 3: Status and diagnostics ═══")
    await invoke(session_id, "Check reranker status — verify Qdrant and Redis connectivity.")

    # ── 5. CE model hot-swap (acknowledge the slow op first)
    print("\n═══ STEP 4: CE model hot-swap (with acknowledgement) ═══")
    # Simulate orchestrator acknowledging the slow operation
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    session.state["reranker:slow_op_acknowledged"] = True
    await session_service.update_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        state=session.state,
    )
    await invoke(
        session_id,
        "Switch the cross-encoder to cross-encoder/ms-marco-MiniLM-L-12-v2 "
        "for higher accuracy, then clear the score cache.",
    )

    # ── 6. Post-swap rerank to confirm new model is active
    print("\n═══ STEP 5: Rerank with new CE model ═══")
    await invoke(
        session_id,
        "Rerank for query: 'right to privacy' with top 3 results. "
        "Include the active CE model name in your response.",
    )

    # ── 7. Print final session score history
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    scores = session.state.get("reranker:session_scores", [])
    print(f"\n═══ SESSION SCORE HISTORY ({len(scores)} calls) ═══")
    for i, entry in enumerate(scores, 1):
        print(
            f"  [{i}] query={entry.get('query', '')[:50]!r} "
            f"result_count={entry.get('result_count')} "
            f"top_score={entry.get('top_score')}"
        )

    # ── 8. List saved artifacts
    try:
        artifacts = await artifact_service.list_artifact_keys(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        print(f"\n═══ ARTIFACTS SAVED ({len(artifacts)}) ═══")
        for key in artifacts:
            print(f"  · {key}")
    except Exception as exc:
        logger.warning("Could not list artifacts: %s", exc)

    logger.info("Demo complete — session=%s", session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_demo())