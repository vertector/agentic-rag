import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import fitz # PyMuPDF
import hashlib

def diagnose_components():
    data_dir = project_root / "data"
    bbs_path = data_dir / "bbs.pdf"
    bbscopy_path = data_dir / "bbscopy.pdf"
    
    docs = {
        "bbs.pdf": fitz.open(str(bbs_path)),
        "bbscopy.pdf": fitz.open(str(bbscopy_path))
    }

    try:
        pages = {
            "bbs.pdf (P1)": docs["bbs.pdf"][0],
            "bbscopy.pdf (P1)": docs["bbscopy.pdf"][0]
        }

        results = {}
        for name, page in pages.items():
            # 1. Geometry
            geom = f"{page.rect.width:.2f}:{page.rect.height:.2f}"
            geom_hash = hashlib.sha256(geom.encode()).hexdigest()
            
            # 2. Text
            text_raw = page.get_text("text").strip()
            text_hash = hashlib.sha256(text_raw.encode("utf-8")).hexdigest()
            
            # 3. Visuals (with quantization)
            max_dim = max(page.rect.width, page.rect.height, 1)
            z = 256 / max_dim
            pix = page.get_pixmap(
                matrix=fitz.Matrix(z, z), 
                colorspace=fitz.csGRAY, 
                alpha=False,
                annots=False
            )
            quantized = bytes([s >> 4 for s in pix.samples])
            visual_hash = hashlib.sha256(quantized).hexdigest()
            
            # 4. Binary check of text
            text_chars = len(text_raw)
            
            results[name] = {
                "geom": geom,
                "geom_hash": geom_hash[:16],
                "text_hash": text_hash[:16],
                "text_chars": text_chars,
                "visual_chars": len(quantized),
                "visual_hash": visual_hash[:16],
                "pix_dim": (pix.width, pix.height)
            }

        print(f"{'Metric':<20} | {'bbs.pdf (P1)':<20} | {'bbscopy.pdf (P1)':<20}")
        print("-" * 70)
        for metric in ["geom", "geom_hash", "text_hash", "text_chars", "visual_hash", "visual_chars", "pix_dim"]:
            val1 = results["bbs.pdf (P1)"][metric]
            val2 = results["bbscopy.pdf (P1)"][metric]
            match = "MATCH" if val1 == val2 else "DIFF"
            print(f"{metric:<20} | {str(val1):<20} | {str(val2):<20} | {match}")

    finally:
        for d in docs.values():
            d.close()

if __name__ == "__main__":
    diagnose_components()
