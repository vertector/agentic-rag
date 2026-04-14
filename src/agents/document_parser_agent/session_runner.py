"""
session_runner.py — Chanoch Clerk Document Parser Agent Demo Loop
=================================================================

Demonstrates the full agentic parsing lifecycle:

  1. Session creation with pre-seeded category
  2. Cold-start single-document parse (expects ~30s on first call)
  3. Settings inspection
  4. Per-call settings override (suppress headers/footers + enable charts)
  5. Batch parse of multiple documents
  6. configure_parser with external VLM backend (with ack flow)
  7. Session log and artifact listing

Usage:
    uv run python -m src.agents.document_parser_agent.session_runner

Environment:
    OLLAMA_HOST      — default: http://localhost:11434
    (all document_parser_mcp env vars are read by server.py directly)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("session_runner")

APP_NAME = "chanoch_clerk_parser"
USER_ID = "pipeline_orchestrator"

# Update these to real paths in your environment before running.
SAMPLE_PDF = str(Path(__file__).parent.parent.parent.parent / "data" / "bbs.pdf")
SAMPLE_BATCH = [
    str(Path(__file__).parent.parent.parent.parent / "data" / "sample.pdf"),
    str(Path(__file__).parent.parent.parent.parent / "data" / "Essential-GraphRAG-sample.pdf"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core invoke helper
# ─────────────────────────────────────────────────────────────────────────────

async def invoke(
    session_id: str,
    message: str,
    *,
    print_response: bool = True,
) -> Optional[str]:
    """Send a single message and collect the final agent response."""
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
        print(f"AGENT RESPONSE:\n{final_response[:500]}{'...' if final_response and len(final_response) > 500 else ''}")
        print("─"*60)

    return final_response


# ─────────────────────────────────────────────────────────────────────────────
# Demo loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_demo():
    session_id = f"demo_{uuid.uuid4().hex[:8]}"

    # ── 1. Create session with pre-seeded state
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state={"parser:active_category": "research"},
    )
    logger.info("Session created: %s", session_id)

    # # ── 2. Cold-start single parse
    print("\n═══ STEP 1: Cold-start single document parse ═══")
    print("(Expect ~30s for PaddleOCRVL initialisation on first call)")
    await invoke(
        session_id,
        f"Parse {SAMPLE_PDF}.",
    )

    # ── 3. Settings inspection
    # print("\n═══ STEP 2: Inspect current parser settings ═══")
    # await invoke(session_id, "Show me the current parser settings.")

    # ── 4. Per-call override: suppress header/footer + enable charts
    # print("\n═══ STEP 3: Parse with per-call settings override ═══")
    # await invoke(
    #     session_id,
    #     f"Parse {SAMPLE_PDF}. For this call only: suppress 'header' and 'footer' and 'footnote' "
    #     f"labels from markdown output, and enable chart recognition.",
    # )

    # # ── 5. Batch parse
    # print("\n═══ STEP 4: Batch parse (2 documents) ═══")
    # batch_list = json.dumps(SAMPLE_BATCH)
    # await invoke(
    #     session_id,
    #     f"Parse these documents in parallel: {batch_list}. "
    #     "Use auto worker count. Do not include page images.",
    # )

    # # ── 6. Configure external VLM backend (with ack flow)
    # print("\n═══ STEP 5: Configure external vLLM backend ═══")
    # # Step 6a: request without ack → expect confirmation_required response
    # await invoke(
    #     session_id,
    #     "Switch the VLM backend to vllm-server at http://localhost:8000 "
    #     "with model PaddlePaddle/PaddleOCR-VL-1.5. No api key needed.",
    # )

    # # ── 6b: simulate orchestrator providing ack
    # # (in a real orchestrator this would be injected after the confirmation response)
    # session = await session_service.get_session(
    #     app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    # )
    # # This specific change doesn't hit the high-worker gate — it hits the
    # # vl_rec_backend guard only if we add one. For now just show the settings call.
    # print("\n[Simulating orchestrator settings confirmation via direct parse]")
    # await invoke(
    #     session_id,
    #     "Actually, keep the local backend. Just confirm current settings are intact.",
    # )

    # ── 7. Session parse log
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    log = session.state.get("parser:session_parse_log", [])
    print(f"\n═══ SESSION PARSE LOG ({len(log)} calls) ═══")
    for i, entry in enumerate(log, 1):
        print(
            f"  [{i}] file={Path(entry.get('file', '')).name} "
            f"pages={entry.get('page_count')} "
            f"roots={[r[:8]+'...' for r in entry.get('merkle_roots', [])[:2]]}"
        )

    # ── 8. Artifact listing
    try:
        artifacts = await artifact_service.list_artifact_keys(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        print(f"\n═══ ARTIFACTS ({len(artifacts)}) ═══")
        for key in artifacts:
            print(f"  · {key}")
    except Exception as exc:
        logger.warning("Could not list artifacts: %s", exc)

    logger.info("Demo complete — session=%s", session_id)


if __name__ == "__main__":
    asyncio.run(run_demo())