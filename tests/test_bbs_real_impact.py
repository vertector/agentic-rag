
import asyncio
import sys
import json
import uuid
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor
from shared.schemas import Document

async def test_bbs_real_impact():
    """
    Test the retrieval impact on the REAL bbs.pdf manifest.
    Forces a clean ingestion to ensure Hybrid Context is applied.
    """
    # Use a completely fresh collection for this specific run
    unique_run = uuid.uuid4().hex[:6]
    collection_name = f"bbs_impact_{unique_run}"
    
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        collection_base_name=collection_name
    )
    await ingestor.setup()

    # Path to the real manifest
    manifest_path = project_root / "src" / ".cache" / "snapshots" / "09264004d4be1b9a96a907caed07f954f88a5e7eb942858b4a0772c822c2a337-44136fa355b3678a" / "manifest.json"
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        return

    print(f"\n[1] Loading real BBS manifest: {manifest_path.name}")
    with open(manifest_path, "r") as f:
        data = json.load(f)
        documents = [Document.model_validate(d) for d in data]

    print(f"    Loaded {len(documents)} pages. Forcing ingestion with Hybrid Context...")
    
    # We clear Redis keys for these files to bypass idempotency check
    for doc in documents:
        # State key format from ingestion_pipeline.py
        model_id = ingestor.model_id
        from ingestion_pipeline.ingestion_pipeline import _encode_filename
        encoded_fn = _encode_filename(doc.metadata.filename)
        redis_key = f"state:{model_id}:doc:{encoded_fn}:page:{doc.metadata.page_index}"
        await ingestor.redis.delete(redis_key)
        
        # Ingest
        doc.metadata.category = "bbs_real"
        await ingestor.process_document(doc)

    # Perform the query
    query = "What was the percentage gain when using balanced batch sampling?"
    print(f"\n[2] Querying: '{query}'")
    
    # Search specifically in the BBS category we just ingested
    results = await ingestor.secure_search(query, category="bbs_real", limit=3)
    
    if not results:
        print("    [ERROR] No results found.")
        # Try a broader search (without category) just in case
        print("    Trying search without category filter...")
        results = await ingestor.secure_search(query, limit=3)

    if not results:
        print("    [CRITICAL] Still no results found.")
        return

    print(f"    Found {len(results)} hits.")
    
    for i, hit in enumerate(results):
        payload = hit.payload
        content = payload.get("content", "")
        context = payload.get("context", "")
        ctype = payload.get("grounding", {}).get("chunk_type", "unknown")
        score = hit.score

        print(f"\n--- Hit {i+1} (Type: {ctype}, Score: {score:.4f}) ---")
        print(f"Context: {context}")
        # Print more content to see if the table is there
        print(f"Content Preview:\n{content[:1000]}...")

    # Success check
    top_hit_content = results[0].payload.get("content", "").lower()
    if "balanced batch sampling" in top_hit_content:
        print("\n[SUCCESS] Retrieval hit relevant BBS content using specific query!")
    else:
        print("\n[INFO] Top hit did not contain 'balanced batch sampling'. Inspect results manually.")

if __name__ == "__main__":
    asyncio.run(test_bbs_real_impact())
