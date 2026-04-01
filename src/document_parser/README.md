# Document Parser Module

The `document_parser` module is the core extraction engine for the RAG system. It is designed to take raw files (like PDFs and images) and convert them into rich, structurally-aware Markdown utilizing **PaddleOCR-VL** and sophisticated layout detection pipelines.

## Features

- **Multi-Modal Parsing**: Built around PaddleOCR-VL to extract not just text, but to understand charts, graphs, tables, and complex document layouts.
- **VLM Backend Support**: Supports plugging in remote or local Vision-Language Model servers for accelerated parsing (`mlx-vlm-server`, `vllm-server`, `sglang-server`, `fastdeploy-server`, or `local`).
- **Layout Intelligence**: Automatically detects document structure (headers, footers, tables, paragraphs, images, etc).
- **Dewarping & Orientation**: Auto-corrects skewed/curved documents and properly rotates pages before extraction.
- **Seal & Stamp Recognition**: Explicitly highlights and extracts organizational seals and stamps.
- **Batch Processing**: Provides parallelized worker execution for processing directories full of documents at once.

## Usage (Python API)

You can import and use the parser directly in any Python script or Jupyter Notebook:

```python
import sys
# Ensure src is in your python path
sys.path.insert(0, "./src")

from document_parser import DocumentParser
from shared.schemas import PipelineSettings

# Configure the parser (e.g., pointing it to a local MLX VLM server)
settings = PipelineSettings(
    use_ocr_for_image_block=True,
    use_chart_recognition=True,
    vl_rec_backend="mlx-vlm-server", 
    vl_rec_server_url="http://localhost:8111/",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5"
)

# Instantiate and parse
parser = DocumentParser(settings)
documents = parser.parse("data/sample.pdf")

# Output is a list of Document schemas (one per page)
print(documents[0].markdown)
```

## Model Context Protocol (MCP) Server

This module also ships with a fully compatible MCP Server (`server.py`) powered by `FastMCP`. This allows external MCP Clients (like Claude Desktop or the MCP Inspector) to test, configure, and parse documents without writing any Python.

### Starting the Server (via Inspector)
You can boot the server and test it interactively using the official MCP Inspector. From the project root, run:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python src/document_parser/server.py
```

### Available MCP Tools

Once the server is running, it exposes the following tools to clients:

1. **`parse_document`**: Parse a single document or image. Accepts a local file path or base64 encoded bytes. 
2. **`parse_batch`**: Parse multiple files concurrently using Python multiprocessing.
3. **`configure_parser`**: Dynamically overwrite the server's global `PipelineSettings` (e.g., swapping the VLM backend at runtime).
4. **`get_parser_settings`**: Fetch the server's current configuration state.

## Output Schema
The parser outputs data adhering to the `shared.schemas.Document` model. A typical page extraction yields:
- `markdown`: The full stitched markdown of the entire page.
- `chunks`: Block-level extractions including bounding box coordinates (`grounding`), making it perfect for citation-aware RAG pipelines.
- `metadata`: Contains pagination, file origins, and optionally base64 image captures of the page.
- `merkle_root`: A deterministic hash fingerprint of the page content.
