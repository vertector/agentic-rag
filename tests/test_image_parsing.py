import sys
from pathlib import Path
import logging

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_image_parsing():
    """
    Verify that DocumentParser correctly handles standard image files (PNG, JPG).
    Tests:
    1. Parsing a PNG file.
    2. Parsing a JPG file.
    3. Verifying Snapshot and Merkle root logic for images.
    """
    data_dir = project_root / "data"
    image_files = [
        data_dir / "receipt.jpg",
        data_dir / "table.png"
    ]
    
    parser = DocumentParser()
    
    for img_path in image_files:
        if not img_path.exists():
            print(f"Skipping {img_path.name} (not found)")
            continue
            
        print(f"\n--- Testing Image: {img_path.name} ---")
        docs, manifest_path = parser.parse(str(img_path))
        
        print(f"  Pages: {len(docs)}")
        print(f"  Snapshot Manifest: {manifest_path.name}")
        
        if len(docs) > 0:
            doc = docs[0]
            print(f"  Markdown Length: {len(doc.markdown)}")
            print(f"  Chunks: {len(doc.chunks)}")
            print(f"  Merkle Root: {doc.get_merkle_root()}")
            
            # Check if sidecar .md was created
            snapshot_dir = manifest_path.parent
            md_files = list(snapshot_dir.glob("*.md"))
            print(f"  Sidecar MD files: {[f.name for f in md_files]}")
            
            if len(md_files) > 0:
                with open(md_files[0], "r", encoding="utf-8") as f:
                    content = f.read()
                    print(f"  MD file content length: {len(content)}")
                    if len(content) == 0:
                        print("  [ERROR] Sidecar MD file is empty!")
                    else:
                        print("  [SUCCESS] Sidecar MD file is populated.")

if __name__ == "__main__":
    test_image_parsing()
