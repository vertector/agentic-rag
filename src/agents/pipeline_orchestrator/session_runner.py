"""
session_runner.py — Chanoch Clerk Orchestrator Conversational Demo

Simulates a realistic multi-turn user session:
  1. Ambiguous intent → orchestrator asks for clarity
  2. User selects full pipeline → parse → ingest → retrieve
  3. Follow-up retrieval query (no re-parse needed)
  4. Integrity check
  5. Purge attempt → gate blocks → user confirms → purge

Usage:
    cd src
    uv run python -m agents.pipeline_orchestrator.session_runner
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from google.genai import types

from .agent import runner, session_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("session_runner")

APP_NAME = "chanoch_clerk"
USER_ID = "user_001"

SAMPLE_PDF = str(Path(__file__).parent.parent.parent.parent / "data" / "sample.pdf")


async def invoke(session_id: str, message: str) -> Optional[str]:
    print(f"\n👤 {message}")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_response: Optional[str] = None
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    if final_response:
        print(f"🤖 {final_response}")
    return final_response


async def run_demo():
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    logger.info("Session: %s", session_id)

    # ── 1. Ambiguous intent
    # await invoke(session_id, f"Do something with {SAMPLE_PDF}")

    # ── 2. User clarifies — full pipeline
    await invoke(session_id, f"parse the document {SAMPLE_PDF} and ingest it")
    await invoke(session_id, "proceed and ingest the parsed document `documents.json` for the bbs.pdf file")

    # ── 3. Follow-up retrieval (no re-ingest)
    # await invoke(session_id, "What was the percentage gain when using balanced batch sampling?")

    # # ── 4. Integrity check
    # await invoke(session_id, "Can you verify the integrity of that document?")

    # # ── 5. Purge attempt — gate should ask for confirmation
    # await invoke(session_id, "Delete sample.pdf from the system")

    # # ── 6. User confirms
    # await invoke(session_id, "Yes, go ahead")

    logger.info("Demo complete.")


if __name__ == "__main__":
    asyncio.run(run_demo())