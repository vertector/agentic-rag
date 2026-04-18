
import sys
from pathlib import Path
import json

# Add project root and src to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from document_parser.document_parser import DocumentParser
from shared.schemas import PipelineSettings

def test_build_chunks_comprehensive():
    """
    Comprehensive unit test for _build_chunks logic using various PaddleOCR labels.
    """
    mock_json = {
        "parsing_res_list": [
            {"block_label": "document title", "block_content": "Deep Learning Report", "block_bbox": [10, 10, 100, 20]},
            {"block_label": "paragraph title", "block_content": "1. Introduction", "block_bbox": [10, 30, 100, 40]},
            {"block_label": "text", "block_content": "This is basic text.", "block_bbox": [10, 50, 100, 60]},
            {"block_label": "table caption", "block_content": "Table 1: Performance", "block_bbox": [10, 70, 100, 80]},
            {"block_label": "table", "block_content": "| Model | Accuracy |\n|-------|----------|\n| V1    | 95%      |", "block_bbox": [10, 90, 100, 130]},
            {"block_label": "vision_footnote", "block_content": "Footnote about vision.", "block_bbox": [10, 140, 100, 150]},
            {"block_label": "figure caption", "block_content": "Figure A: Architecture", "block_bbox": [10, 160, 100, 170]},
            {"block_label": "figure", "block_content": "[BINARY_IMAGE_DATA]", "block_bbox": [10, 180, 100, 250]},
            {"block_label": "page number", "block_content": "Page 1", "block_bbox": [10, 260, 100, 270]},
        ],
        "layout_det_res": {
            "boxes": [
                {"score": 0.99} for _ in range(9)
            ]
        }
    }

    parser = DocumentParser(PipelineSettings(markdown_ignore_labels=["page number"]))
    chunks = parser._build_chunks(mock_json, page_index=1)

    print(f"Generated {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"\nChunk {i} ({c.grounding.chunk_type}):")
        print(f"  Context: {c.context}")
        print(f"  Markdown: {repr(c.chunk_markdown[:100])}...")

    # --- Assertions ---
    
    # 1. Labels with spaces should be normalized and track context
    # Chunk 0: document title (normalized to document_title)
    assert chunks[0].grounding.chunk_type == "document title"
    
    # 2. paragraph title should update current_header
    # Chunk 1: paragraph title
    assert chunks[1].grounding.chunk_type == "paragraph title"
    
    # 3. text should have context from titles
    assert "Header: 1. Introduction" in chunks[2].context
    
    # 4. table caption should be merged into table
    # Chunk 3 is now the table (because caption was merged)
    assert chunks[3].grounding.chunk_type == "table"
    assert "Caption: Table 1: Performance" in chunks[3].context
    assert "Table 1: Performance" in chunks[3].chunk_markdown
    
    # 5. vision_footnote should exist and have context
    assert chunks[4].grounding.chunk_type == "vision_footnote"
    assert "Header: 1. Introduction" in chunks[4].context
    
    # 6. figure caption should be merged into figure
    assert chunks[5].grounding.chunk_type == "figure"
    assert "Caption: Figure A: Architecture" in chunks[5].context
    
    # 7. ignore labels should work
    for c in chunks:
        assert c.grounding.chunk_type != "page number"

    print("\n[SUCCESS] Comprehensive unit test for _build_chunks logic passed!")

if __name__ == "__main__":
    test_build_chunks_comprehensive()
