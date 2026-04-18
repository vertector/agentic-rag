
import asyncio
import sys
import os
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor

async def test_direct_search():
    """
    Directly use the ingestor to search without Agent overhead.
    This avoids 429 errors from the LLM.
    """
    # The session runner uses 'secure_rag' as base name
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        collection_base_name="secure_rag"
    )
    
    # We don't need to run setup() if collection exists, but it's safe to do so
    await ingestor.setup()

    query = "What was the percentage gain when using balanced batch sampling?"
    print(f"\n[1] Querying Ingestor directly: '{query}'")
    
    # Search in the 'research' category as set in session_runner
    results = await ingestor.secure_search(query, category="research", limit=3)
    
    if not results:
        print("    [ERROR] No results found in Qdrant.")
        return

    print(f"    Found {len(results)} hits.")
    
    for i, hit in enumerate(results):
        payload = hit.payload
        content = payload.get("content", "")
        context = payload.get("context", "")
        chunk_type = payload.get("grounding", {}).get("chunk_type", "unknown")
        
        print(f"\n--- Hit {i+1} (Type: {chunk_type}) ---")
        print(f"Context: {context}")
        print(f"Content:\n{content}")

if __name__ == "__main__":
    asyncio.run(test_direct_search())
