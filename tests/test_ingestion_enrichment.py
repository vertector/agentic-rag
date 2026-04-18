
import asyncio
import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor
from shared.schemas import Document, Chunk, Metadata, Grounding

def get_mock_document():
    return Document(
        metadata=Metadata(
            filename="test_doc.pdf",
            page_index=1,
            page_count=1,
            blob_cid="mock_blob_123"
        ),
        chunks=[
            Chunk(
                chunk_markdown="Chapter 1: The Beginning",
                grounding=Grounding(chunk_type="paragraph title", bbox=[0,0,10,10], page_index=1)
            ),
            Chunk(
                chunk_markdown="Once upon a time...",
                grounding=Grounding(chunk_type="text", bbox=[0,20,10,30], page_index=1)
            ),
            Chunk(
                chunk_markdown="Table 1: Characters",
                grounding=Grounding(chunk_type="table caption", bbox=[0,40,10,50], page_index=1)
            ),
            Chunk(
                chunk_markdown="| Name | Role |\n|---|---|\n| Alice | Hero |",
                grounding=Grounding(chunk_type="table", bbox=[0,60,10,100], page_index=1)
            )
        ]
    )

async def test_enrichment_logic():
    ingestor = AsyncMerkleQdrantIngestor(qdrant_url="http://localhost:6333")
    doc = get_mock_document()
    
    print("\n--- Testing Ingestor-Stage Enrichment ---")
    enriched = ingestor._enrich_chunks(doc.chunks)
    
    print(f"Original chunks: {len(doc.chunks)}")
    print(f"Enriched chunks: {len(enriched)}")
    
    for i, chunk in enumerate(enriched):
        print(f"\nChunk {i} ({chunk.grounding.chunk_type}):")
        print(f"  Context: {chunk.context}")
        print(f"  Markdown Preview: {repr(chunk.chunk_markdown[:80])}...")

    # Assertions
    # 1. Title tracking
    assert "Header: Chapter 1: The Beginning" in enriched[1].context
    # 2. Table Merging (Caption + Table -> 1 chunk)
    table_chunk = next(c for c in enriched if c.grounding.chunk_type == "table")
    assert "Caption: Table 1: Characters" in table_chunk.context
    assert "Table 1: Characters" in table_chunk.chunk_markdown
    
    print("\n[SUCCESS] Ingestor enrichment logic verified.")

if __name__ == "__main__":
    asyncio.run(test_enrichment_logic())
