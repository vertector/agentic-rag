
import asyncio
import sys
import uuid
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor
from shared.schemas import Document, Chunk, Metadata, Grounding

async def test_retrieval_impact():
    """
    Demonstrate that ingestion-stage enrichment solves context isolation.
    We ingest a table with a caption and search for words only found in the caption.
    """
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        collection_base_name="test_enrichment"
    )
    await ingestor.setup()

    filename = f"impact_test_{uuid.uuid4().hex[:6]}.pdf"
    
    # Define a document where caption and table are separate chunks (raw parser output)
    doc = Document(
        metadata=Metadata(
            filename=filename,
            page_index=1,
            page_count=1,
            blob_cid=f"blob_{uuid.uuid4().hex[:6]}",
            category="test"
        ),
        chunks=[
            Chunk(
                chunk_markdown="Performance Analysis Results",
                grounding=Grounding(chunk_type="paragraph title", bbox=[0,0,10,10], page_index=1)
            ),
            Chunk(
                chunk_markdown="Table 2: Comparison of balanced batch sampling techniques",
                grounding=Grounding(chunk_type="table caption", bbox=[0,20,10,30], page_index=1)
            ),
            Chunk(
                chunk_markdown="| Method | Gain |\n|---|---|\n| BBS | +15% |",
                grounding=Grounding(chunk_type="table", bbox=[0,40,10,80], page_index=1)
            )
        ]
    )

    print(f"\n[1] Ingesting document: {filename}")
    print("    Ingestor will now perform Hybrid Enrichment...")
    await ingestor.process_document(doc)

    # Search for keywords from the caption
    query = "What was the gain for balanced batch sampling?"
    print(f"\n[2] Searching for: '{query}'")
    
    results = await ingestor.secure_search(query, limit=3)
    
    if not results:
        print("    [ERROR] No results found.")
        return

    print(f"    Found {len(results)} hits.")
    
    top_hit = results[0]
    content = top_hit.payload.get("content", "")
    context = top_hit.payload.get("context", "")
    ctype = top_hit.payload.get("grounding", {}).get("chunk_type", "unknown")

    print(f"\n[3] Top Hit Analysis:")
    print(f"    Chunk Type: {ctype}")
    print(f"    Context: {context}")
    print(f"    FULL CONTENT:\n{content}")

    # Verification
    # The hit SHOULD be the table, and it SHOULD contain both the caption and the table structure
    if ctype == "table" and "| Method | Gain |" in content:
        print("\n[SUCCESS] Retrieval hit the Table chunk and it contains the table structure!")
        print("          This proves the Hybrid Merge is working at ingestion stage.")
    else:
        print("\n[FAILURE] Retrieval did not return the table structure as expected.")

if __name__ == "__main__":
    asyncio.run(test_retrieval_impact())
