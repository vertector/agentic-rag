import sys
import os
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser
from shared.blob_store import BlobStore

def test_page_identity():
    """
    Verify that bbscopy.pdf (page 1 of bbs.pdf) is correctly identified as
    identical to bbs.pdf's first page using visual stability hashing,
    even if their full-file CIDs (CAS) differ.
    """
    data_dir = project_root / "data"
    bbs_path = data_dir / "bbs.pdf"
    bbscopy_path = data_dir / "bbscopy.pdf"
    
    if not bbs_path.exists() or not bbscopy_path.exists():
        print(f"Error: Required files not found in {data_dir}")
        return

    # Initialize components
    parser = DocumentParser()
    store = BlobStore()
    
    print("--- 1. Testing Page-Level Identity (Incremental Caching) ---")
    print(f"Computing page hashes for {bbs_path.name}...")
    bbs_hashes = parser._get_page_hashes(bbs_path)
    print(f"  Found {len(bbs_hashes)} pages.")
    
    print(f"Computing page hashes for {bbscopy_path.name}...")
    bbscopy_hashes = parser._get_page_hashes(bbscopy_path)
    print(f"  Found {len(bbscopy_hashes)} pages.")

    print("\nComparison:")
    print(f"  bbs.pdf Page 1 hash:     {bbs_hashes[0]}")
    print(f"  bbscopy.pdf Page 1 hash: {bbscopy_hashes[0]}")
    
    if bbs_hashes[0] == bbscopy_hashes[0]:
        print("\nRESULT: SUCCESS! Page 1 matches visually.")
        print("This means the system will correctly deduplicate these pages in its incremental cache.")
    else:
        print("\nRESULT: FAILURE! Page hashes do not match.")

    print("\n--- 2. Testing File-Level Identity (CAS / BlobStore) ---")
    bbs_cid = store.put_file(bbs_path)
    bbscopy_cid = store.put_file(bbscopy_path)
    
    print(f"  bbs.pdf CID:     {bbs_cid}")
    print(f"  bbscopy.pdf CID: {bbscopy_cid}")
    
    if bbs_cid != bbscopy_cid:
        print("\nRESULT: Confirmed! Full-file CIDs are different, as expected for binary documents.")
        print("The BlobStore correctly treats them as unique blobs while the Parser logic finds the shared content.")
    else:
        print("\nRESULT: SURPRISE! Full-file CIDs are identical. This means the files are binary-equivalent.")

if __name__ == "__main__":
    test_page_identity()
