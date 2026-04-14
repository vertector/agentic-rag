# Chanoch Clerk: Document Parser Module

The `document_parser` module is the core extraction engine for the Chanoch Clerk RAG system. It is designed to take raw files (like PDFs and images) and convert them into rich, structurally-aware Markdown utilizing **PaddleOCR-VL** and sophisticated layout detection pipelines.

## Features

- **Git-like Content Identity**: Uses Content-Addressed Storage (CAS) and Merkle roots to ensure documents and chunks have immutable, verifiable identities.
- **3-Tier Optimization Engine**: Implements a high-performance caching strategy that separates semantic identity (Snapshots) from visual inference (OCR Deduplication).
- **Robust Page Identity**: Normalized binarized hashing ensures identical pages are deduplicated across different files, even with rendering noise.
- **Context-Aware RAG Chunks**: Chunks are cryptographically pinned to their document CID and page index, preventing vector store divergence when content shifts position.
- **Atomic Persistence**: Every write operation uses a transactional "Atomic Rename" strategy to avoid data corruption during process crashes.
- **Concurrency & Locking**: Thread-safe VLM access and cross-process directory locking for safely shared OCR result caches.

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

### Optimization Tiers (The "Git" Lifecycle)

1. **Tier 1: Snapshot Manifest (The "Commit")**: If the exact document content and settings match a previous run, results are returned instantly from a versioned manifest.
2. **Tier 2: Global OCR Cache (The "Library")**: If individual pages match previously parsed pages visually (even in different files), the VLM inference is skipped and the cached OCR results are re-assembled into the current document's context.
3. **Tier 3: Live Inference**: New content is processed via VLM and then committed to both the OCR cache and the document snapshot.

## Testing & Verification

A comprehensive test suite is available in the root `/tests` directory:

- **Identity Stability**: `uv run python tests/test_page_identity.py` (Visual vs Binary identity).
- **Semantic Integrity**: `uv run python tests/test_ordinal_shift.py` (Merkle root behavior on content moves).
- **Blob Storage**: `uv run python tests/test_blob_store.py` (CAS checks).
- **Image Support**: `uv run python tests/test_image_parsing.py` (Multi-format validation).

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
- `chunks`: Block-level extractions with stable CIDs (including `doc_cid` and `page_index` context).
- `metadata`: Contains `blob_cid` (CAS link), pagination, and optional sidecar image captures.
- `merkle_root`: A verifiable DAG root ensuring content-and-context integrity.
