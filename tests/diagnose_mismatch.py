import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import fitz # PyMuPDF
import hashlib

def diagnose_page_mismatch():
    data_dir = project_root / "data"
    bbs_path = data_dir / "bbs.pdf"
    bbscopy_path = data_dir / "bbscopy.pdf"
    
    if not bbs_path.exists() or not bbscopy_path.exists():
        print(f"Error: Required files not found in {data_dir}")
        return

    docs = {
        "bbs.pdf": fitz.open(str(bbs_path)),
        "bbscopy.pdf": fitz.open(str(bbscopy_path))
    }

    try:
        pages = {
            "bbs.pdf (P1)": docs["bbs.pdf"][0],
            "bbscopy.pdf (P1)": docs["bbscopy.pdf"][0]
        }

        print(f"{'Source':<20} | {'Width':<10} | {'Height':<10} | {'Colorspace':<12} | {'Alpha':<6}")
        print("-" * 70)

        for name, page in pages.items():
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            print(f"{name:<20} | {pix.width:<10} | {pix.height:<10} | {pix.colorspace.name:<12} | {pix.alpha:<6}")
            
            # Additional metadata
            print(f"  Rect:      {page.rect}")
            print(f"  CropBox:   {page.cropbox}")
            print(f"  Samples len: {len(pix.samples)}")
            print(f"  Hash:      {hashlib.sha256(pix.samples).hexdigest()}")
            print()

    finally:
        for d in docs.values():
            d.close()

if __name__ == "__main__":
    diagnose_page_mismatch()
