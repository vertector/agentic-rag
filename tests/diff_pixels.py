import sys
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import fitz # PyMuPDF
import numpy as np

def diff_pixels():
    data_dir = project_root / "data"
    bbs_path = data_dir / "bbs.pdf"
    bbscopy_path = data_dir / "bbscopy.pdf"
    
    doc1 = fitz.open(str(bbs_path))
    doc2 = fitz.open(str(bbscopy_path))

    pix1 = doc1[0].get_pixmap(matrix=fitz.Matrix(1, 1))
    pix2 = doc2[0].get_pixmap(matrix=fitz.Matrix(1, 1))

    img1 = np.frombuffer(pix1.samples, dtype=np.uint8)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8)

    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    
    print(f"Total bytes compared: {len(img1)}")
    print(f"Number of differing bytes: {np.count_nonzero(diff)}")
    print(f"Max difference in a single byte: {np.max(diff)}")
    print(f"Mean difference: {np.mean(diff):.4f}")

    # Try Grayscale normalization
    pix1_g = doc1[0].get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
    pix2_g = doc2[0].get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
    
    img1_g = np.frombuffer(pix1_g.samples, dtype=np.uint8)
    img2_g = np.frombuffer(pix2_g.samples, dtype=np.uint8)
    diff_g = np.abs(img1_g.astype(np.int16) - img2_g.astype(np.int16))
    
    print("\nGrayscale Comparison:")
    print(f"Total bytes (Gray): {len(img1_g)}")
    print(f"Differing bytes (Gray): {np.count_nonzero(diff_g)}")
    print(f"Max difference (Gray): {np.max(diff_g)}")

    doc1.close()
    doc2.close()

if __name__ == "__main__":
    diff_pixels()
