
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

async def test_end_to_end():
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        collection_base_name="e2e_impact_test"
    )
    await ingestor.setup()

    # 1. Prepare raw layout chunks (isolated)
    doc = Document(
        metadata=Metadata(
            filename="bbs_sample.pdf",
            page_index=3,
            page_count=10,
            blob_cid="blob_bbs_123",
            category="research"
        ),
        chunks=[
            Chunk(
                chunk_markdown="### 4.2 Sampling Strategies",
                grounding=Grounding(chunk_type="paragraph title", bbox=[10,10,100,20], page_index=3)
            ),
            Chunk(
                chunk_markdown="Table 2: Comparison of balanced batch sampling (BBS) vs standard sampling.",
                grounding=Grounding(chunk_type="table caption", bbox=[10,30,100,40], page_index=3)
            ),
            Chunk(
                chunk_markdown="| Metric | Standard | BBS |\n|---|---|---|\n| Accuracy | 92.1% | 94.5% |\n| Gain | - | +2.4% |",
                grounding=Grounding(chunk_type="table", bbox=[10,50,100,100], page_index=3)
            )
        ]
    )

    print("\n[1] Ingesting BBS Sample with Hybrid Context...")
    await ingestor.process_document(doc)

    # 2. Search using keywords from the CAPTION and the HEADER
    # Neither "percentage gain" nor "balanced batch sampling" are in the table block itself.
    # "percentage gain" is inferred from query, "balanced batch sampling" is in the caption.
    query = "What was the percentage gain when using balanced batch sampling?"
    print(f"\n[2] Querying: '{query}'")
    
    results = await ingestor.secure_search(query, category="research", limit=1)
    
    if not results:
        print("    [ERROR] No results found.")
        return

    hit = results[0]
    print(f"\n[3] TOP HIT FOUND:")
    print(f"    Type: {hit.payload.get('grounding', {}).get('chunk_type')}")
    print(f"    Content:\n{hit.payload.get('content')}")
    
    if "Gain" in hit.payload.get('content') and "balanced batch sampling" in hit.payload.get('content'):
        print("\n[SUCCESS] The enriched table chunk was retrieved correctly!")
    else:
        print("\n[FAILURE] Did not retrieve the expected enriched chunk.")

if __name__ == "__main__":
    asyncio.run(test_end_to_end())
