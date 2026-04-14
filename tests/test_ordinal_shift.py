import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser
from shared.blob_store import BlobStore
import hashlib

def test_ordinal_shift_integrity():
    """
    Verify that moving content to a different page number (ordinal shift)
    results in a different Merkle root, ensuring downstream re-embedding.
    """
    data_dir = project_root / "data"
    bbs_path = data_dir / "bbs.pdf"
    
    if not bbs_path.exists():
        print(f"Error: {bbs_path} not found")
        return

    parser = DocumentParser()
    
    print("--- 1. Parsing original document ---")
    docs_orig, _ = parser.parse(str(bbs_path))
    page2_orig = docs_orig[1]
    root_orig = page2_orig.get_merkle_root()
    print(f"  Page 2 Original Index: {page2_orig.metadata.page_index}")
    print(f"  Page 2 Merkle Root:     {root_orig}")

    print("\n--- 2. Simulating Ordinal Shift (same content, different index) ---")
    # We manually create a Document object representing the same content but at index 3
    from shared.schemas import Document, Metadata
    
    # Clone metadata but change index
    meta_shifted = page2_orig.metadata.model_copy(update={"page_index": 3})
    
    # Rebuild chunks with new index (this mimics what the new parser logic does)
    chunks_shifted = []
    for c in page2_orig.chunks:
        c_shifted = c.model_copy(deep=True)
        c_shifted.grounding.page_index = 3
        chunks_shifted.append(c_shifted)
        
    doc_shifted = Document(
        markdown=page2_orig.markdown,
        chunks=chunks_shifted,
        metadata=meta_shifted
    )
    
    root_shifted = doc_shifted.get_merkle_root()
    print(f"  Page 3 Shifted Index:   {doc_shifted.metadata.page_index}")
    print(f"  Page 3 Merkle Root:     {root_shifted}")

    print("\n--- Verification ---")
    if root_orig != root_shifted:
        print("SUCCESS: Merkle roots differ! The system correctly identifies the ordinal shift.")
        print("This ensures the vector store will re-index this content at its new location.")
    else:
        print("FAILURE: Merkle roots are identical. Downstream divergence risk (Flaw 6) remains.")

if __name__ == "__main__":
    test_ordinal_shift_integrity()
