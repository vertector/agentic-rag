
import asyncio
import logging
import uuid
from shared.schemas import Document, Metadata, Chunk, Grounding
from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor

async def test_hierarchical_context_persistence():
    logging.basicConfig(level=logging.INFO)
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        collection_base_name="test_persistence"
    )
    await ingestor.setup()
    
    filename = "persistence_test.pdf"
    doc_cid = f"blob_{uuid.uuid4().hex[:6]}"
    
    # Page 1: Establish hierarchy
    page1 = Document(
        metadata=Metadata(filename=filename, page_index=1, page_count=2, blob_cid=doc_cid),
        chunks=[
            Chunk(
                chunk_markdown="# 1 Introduction",
                grounding=Grounding(chunk_type="document_title", bbox=[0,0,10,10], page_index=1)
            ),
            Chunk(
                chunk_markdown="## 1.1 Methodology",
                grounding=Grounding(chunk_type="paragraph_title", bbox=[0,20,10,30], page_index=1)
            ),
            Chunk(
                chunk_markdown="We use a new sampling approach.",
                grounding=Grounding(chunk_type="text", bbox=[0,40,10,50], page_index=1)
            )
        ]
    )
    
    # Page 2: Continuation (no headers)
    page2 = Document(
        metadata=Metadata(filename=filename, page_index=2, page_count=2, blob_cid=doc_cid),
        chunks=[
            Chunk(
                chunk_markdown="| Result | Score |\n|---|---|\n| BBS | 0.95 |",
                grounding=Grounding(chunk_type="table", bbox=[0,0,10,50], page_index=2)
            ),
            Chunk(
                chunk_markdown="This concludes the results.",
                grounding=Grounding(chunk_type="text", bbox=[0,60,10,70], page_index=2)
            )
        ]
    )
    
    print("\n[1] Ingesting Page 1...")
    success1, state1 = await ingestor.process_document(page1)
    print(f"    Final State P1: {state1}")
    
    print("\n[2] Ingesting Page 2 with State from P1...")
    success2, state2 = await ingestor.process_document(page2, initial_header_state=state1)
    print(f"    Final State P2: {state2}")
    
    # 3. Verify retrieval
    print("\n[3] Verifying breadcrumbs on Page 2 via search...")
    # Search for something in the table on Page 2
    hits = await ingestor.secure_search(query="BBS result score", limit=5)
    
    found_table = False
    for hit in hits:
        if hit.payload.get("grounding", {}).get("page_index") == 2:
            ctx = hit.payload.get("context", "")
            print(f"    Hit on Page 2 | Context: {ctx}")
            if "1 Introduction > 1.1 Methodology" in ctx:
                found_table = True
                print("    [SUCCESS] Multi-level context persisted to Page 2!")
    
    assert found_table, "Table on Page 2 did not inherit context from Page 1"
    print("\n[SUCCESS] Task 7.1: Cross-page persistence verified.")

    # --- 4. Verify Level Reset ---
    print("\n[4] Verifying Level Reset (New Document Title)...")
    page3 = Document(
        metadata=Metadata(filename=filename, page_index=3, page_count=3, blob_cid=doc_cid),
        chunks=[
            Chunk(
                chunk_markdown="# 2 Conclusion",
                grounding=Grounding(chunk_type="document_title", bbox=[0,0,10,10], page_index=3)
            ),
            Chunk(
                chunk_markdown="The study is complete.",
                grounding=Grounding(chunk_type="text", bbox=[0,20,10,30], page_index=3)
            )
        ]
    )
    
    # Pass state2 (Introduction > Methodology) to Page 3
    success3, state3 = await ingestor.process_document(page3, initial_header_state=state2)
    print(f"    Final State P3: {state3}")
    
    # Breadcrumb should ONLY contain '2 Conclusion'
    assert state3 == ["2 Conclusion"], f"HeaderStack did not reset! Got: {state3}"
    print("    [SUCCESS] Level 0 reset verified.")

    print("\n[ALL TESTS PASSED]")

if __name__ == "__main__":
    asyncio.run(test_hierarchical_context_persistence())
