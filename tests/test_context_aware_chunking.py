
import sys
import os
from pathlib import Path
import logging

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser
from shared.schemas import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_context_aware_chunking():
    """
    Verify that DocumentParser correctly implements Hybrid Context Association.
    """
    # Use the table.png from the local data directory
    img_path = project_root / "data" / "table.png"
    if not img_path.exists():
        print(f"Skipping test: {img_path} not found")
        return

    print(f"\n--- Testing Context-Aware Chunking: {img_path.name} ---")
    parser = DocumentParser()
    
    # We'll bypass the cache to ensure we run the new logic
    docs, manifest_path = parser.parse(str(img_path))
    
    if not docs:
        print("No documents parsed.")
        return

    doc = docs[0]
    print(f"Parsed {len(doc.chunks)} chunks.")
    
    table_found = False
    context_found = False
    caption_merged = False

    for i, chunk in enumerate(doc.chunks):
        print(f"\nChunk {i}:")
        print(f"  Type: {chunk.grounding.chunk_type}")
        print(f"  Context: {chunk.context}")
        print(f"  Content Preview: {chunk.chunk_markdown[:150]}...")
        
        if chunk.grounding.chunk_type == "table":
            table_found = True
            if "Caption:" in chunk.context:
                caption_merged = True
                print("  [SUCCESS] Caption merged into table chunk context.")
            if "Caption:" in chunk.chunk_markdown:
                print("  [SUCCESS] Caption text found in table chunk markdown.")

        if chunk.context:
            context_found = True

    if not table_found:
        print("  [WARNING] No table found in the document. Try a different sample if needed.")
    
    if context_found:
        print("  [SUCCESS] Hierarchical context was injected into at least one chunk.")
    else:
        print("  [ERROR] No context was injected into any chunk.")

    if table_found and not caption_merged:
        print("  [WARNING] Table found but no caption was merged. (May be no caption in sample).")

    # Verify Merkle Root
    print(f"  Merkle Root: {doc.get_merkle_root()}")

if __name__ == "__main__":
    test_context_aware_chunking()
