import asyncio
import logging
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser
from agents.ingestion_agent.agent import runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ingestion")

async def run_test():
    parser = DocumentParser()
    data_path = Path("data/bbs.pdf")
    
    logger.info(f"1. Parsing {data_path} to ensure we have a manifest.json to ingest...")
    docs, manifest_path = parser.parse(str(data_path))
    
    
    logger.info(f"manifest.json generated at: {manifest_path}")
    
    logger.info("2. Triggering Ingestion Agent...")
    import uuid
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    
    from agents.ingestion_agent.agent import session_service
    await session_service.create_session(
        app_name="chanoch_clerk_ingestion",
        user_id="test_user",
        session_id=session_id,
    )
    
    # We ask the ingestion agent to ingest by FILENAME (discovery flow)
    # and then perform an audit (verification flow)
    prompt = (
        "Ingest 'bbs.pdf' into a corpus called 'test_corpus'. "
        "After storing, please verify the integrity of page 1 to make sure everything is correct."
    )
    
    from google.genai import types
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    print("\n[USER] " + prompt)
    final_response = None
    async for event in runner.run_async(user_id="test_user", session_id=session_id, new_message=content):
        # Handle progress messages (terse mid-pipeline notes)
        if hasattr(event, "content") and event.content and hasattr(event.content, "parts") and event.content.parts:
            text = event.content.parts[0].text
            if text and not (hasattr(event, "is_final_response") and event.is_final_response()):
                 print(f"[PROGRESS] {text}")

        if hasattr(event, "is_final_response") and event.is_final_response() and getattr(event, "content", None) and getattr(event.content, "parts", None):
            final_response = event.content.parts[0].text
            print(f"\n[AGENT FINAL RESPONSE]\n{final_response}\n")
        
if __name__ == "__main__":
    asyncio.run(run_test())
