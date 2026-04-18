"""
session_runner.py — Chanoch Clerk Ingestion Agent Demo Loop

Steps:
  1. Status check (connectivity)
  2. Ingest manifest.json from parser output
  3. Integrity audit on first page
  4. Version history retrieval
  5. Point-in-time search (pinned to v1 root)
  6. Redis recovery demo (sync)
  7. Purge with gate flow (dry-run only — confirm gate left unset)

Usage:
    cd src
    uv run python -m agents.ingestion_agent.session_runner
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from google.genai import types

from .agent import runner, session_service, artifact_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("session_runner")

APP_NAME = "chanoch_clerk_ingestion"
USER_ID = "pipeline_orchestrator"

# Update to a real manifest.json path before running
SAMPLE_MANIFEST_JSON = str(
    Path(__file__).parent.parent.parent / ".cache" / "snapshots" / "09264004d4be1b9a96a907caed07f954f88a5e7eb942858b4a0772c822c2a337-44136fa355b3678a" / "manifest.json"
)
SAMPLE_FILENAME = "bbs.pdf"


async def invoke(session_id: str, message: str, *, print_response: bool = True) -> Optional[str]:
    content = types.Content(role="user", parts=[types.Part(text=message)])
    final_response: Optional[str] = None
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    if print_response and final_response:
        print(f"\n{'─'*60}\n{final_response[:600]}{'...' if final_response and len(final_response) > 600 else ''}\n{'─'*60}")
    return final_response


async def run_demo():
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={"ingestor:active_category": "research"},
    )
    logger.info("Session: %s", session_id)

    # # ── 1. Status check
    # print("\n═══ STEP 1: Connectivity check ═══")
    # await invoke(session_id, "Check ingestor status — confirm Qdrant and Redis are reachable.")

    # # ── 2. Ingest
    # print("\n═══ STEP 2: Ingest manifest.json ═══")
    # await invoke(session_id, f"Ingest {SAMPLE_MANIFEST_JSON}.")

    # # ── 3. Audit page 1
    # print("\n═══ STEP 3: Integrity audit — page 1 ═══")
    # await invoke(session_id, f"Verify integrity of {SAMPLE_FILENAME} page 1.")


    # ── 4. Search test (User requested)
    print("\n═══ STEP 4: Search test ═══")
    await invoke(session_id, "What was the percentage gain when using balanced batch sampling?")

    # # ── 4. Version history
    # print("\n═══ STEP 4: Version history ═══")
    # response = await invoke(session_id, f"Get version history for {SAMPLE_FILENAME}.")

    # Extract oldest version_root from session state (persisted by after_tool_callback)
    v1_root = None
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    version_roots: dict = session.state.get("ingestor:version_roots", {})
    roots_for_file = version_roots.get(SAMPLE_FILENAME, [])
    if roots_for_file:
        v1_root = roots_for_file[-1]  # oldest version is last in the list

    # # ── 5. Point-in-time search
    # if v1_root:
    #     print(f"\n═══ STEP 5: Point-in-time search (root={v1_root[:12]}...) ═══")
    #     session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    #     session.state["ingestor:version_root"] = v1_root
    #     await invoke(session_id, "Find chunks about 'distribution shifts' using the pinned version root. Top 3 results.")
    # else:
    #     print("\n[SKIP] No version history available for point-in-time search.")

    # # ── 6. Sync demo (non-destructive)
    # print("\n═══ STEP 6: Redis sync (recovery utility) ═══")
    # await invoke(session_id, f"Sync Redis state from Qdrant for {SAMPLE_FILENAME}.")

    # # ── 7. Purge dry-run (gate should block without purge_confirmed)
    # print("\n═══ STEP 7: Purge dry-run (gate should block) ═══")
    # await invoke(session_id, f"Purge all data for {SAMPLE_FILENAME}.")

    # ── Session log
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    log = session.state.get("ingestor:session_ingest_log", [])
    print(f"\n═══ INGEST LOG ({len(log)} calls) ═══")
    for i, entry in enumerate(log, 1):
        print(f"  [{i}] file={entry.get('file')} ingested={entry.get('ingested')} skipped={entry.get('skipped')} errors={entry.get('errors')}")

    try:
        artifacts = await artifact_service.list_artifact_keys(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        print(f"\n═══ ARTIFACTS ({len(artifacts)}) ═══")
        for key in artifacts:
            print(f"  · {key}")
    except Exception as exc:
        logger.warning("Could not list artifacts: %s", exc)

    logger.info("Demo complete — session=%s", session_id)


if __name__ == "__main__":
    asyncio.run(run_demo())
