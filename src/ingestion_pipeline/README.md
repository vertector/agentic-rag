# Ingestion Pipeline Module

The `ingestion_pipeline` module is the secure, versioned storage engine for the RAG system. It takes parsed `Document` objects from the `document_parser` module and ingests them into a **Qdrant** vector store, anchoring every snapshot with a cryptographic **Merkle root** stored in **Redis**. This design guarantees end-to-end data integrity — any tampering with stored vectors can be detected at any time.

## Features

- **Merkle-Tree Integrity Proofs**: Every document page is fingerprinted by a deterministic binary Merkle tree built from the SHA-256 hashes of its ordered chunk contents. The root hash is stored in Redis as a trust anchor.
- **Versioned Snapshots**: Subsequent ingests of the same page create a new, timestamped snapshot. The previous snapshot is *soft-deleted* (`is_active=False`) rather than removed, enabling full point-in-time queries and audit trails.
- **Idempotent Ingestion**: If a page's content is unchanged since the last ingest, the operation is a no-op — no duplicate data is written.
- **Semantic Search with Version Pinning**: Query against the currently active version of all documents, or pin a search to any specific historical `version_root`.
- **Dual Embedding Engine Support**: Automatically falls back from `fastembed` to `sentence-transformers` if one is unavailable.
- **Redis Recovery**: If Redis state is lost (e.g. after a restart without persistence), the `sync` tool re-seeds Redis from Qdrant's own root-anchor records.

## Architecture

```
Document (from parser)
        │
        ▼
   Merkle Root Hash ──► Redis (trust anchor / idempotency check)
        │
        ▼
   Chunk Embeddings ──► Qdrant (vector store)
        │
        ▼
   Root Anchor Point (zero-vector, structural record)
```

Any time you call `audit`, the system scrolls Qdrant, reconstructs the Merkle tree, and compares it against the Redis anchor — a mismatch means tampered data.

## Usage (Python API)

```python
import asyncio
import sys
sys.path.insert(0, "./src")

from ingestion_pipeline import AsyncMerkleQdrantIngestor
from shared.schemas import Document, Metadata, Chunk, Grounding

async def main():
    ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url="http://localhost:6333",
        redis_host="localhost",
        redis_port=6379,
        collection_base_name="secure_rag",
        model_name="BAAI/bge-small-en-v1.5",  # or any compatible model
    )
    await ingestor.setup()  # creates collection + payload indexes if absent

    # Ingest a document (output of DocumentParser)
    doc = Document(
        metadata=Metadata(filename="report.pdf", page_index=0, page_count=5, category="research"),
        chunks=[
            Chunk(
                chunk_id="abc123",
                chunk_markdown="Introduction to balanced batch sampling...",
                grounding=Grounding(chunk_type="text", bbox=[0, 0, 100, 50], page_index=0, score=0.95),
            )
        ],
    )
    await ingestor.process_document(doc)

    # Semantic search
    results = await ingestor.secure_search(query="balanced sampling", limit=5)

    # Verify integrity
    is_valid = await ingestor.verify_integrity("report.pdf", page_index=0)
    print("Integrity OK:", is_valid)

asyncio.run(main())
```

## Required Services

Before running the server or any ingestion, make sure both services are running locally:

- **Qdrant** (default: `http://localhost:6333`)
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

- **Redis** (default: `localhost:6379`)
  ```bash
  docker run -p 6379:6379 redis:latest
  ```

> **Durability Warning:** Redis is used as the trust anchor for integrity proofs. If Redis restarts without persistence (AOF/RDB), all integrity checks will fail. Enable AOF (`appendonly yes`) or configure RDB snapshots in production. Use `ingest_sync` to recover if Redis state is lost.

## Model Context Protocol (MCP) Server

This module ships with a fully compatible MCP server (`server.py`) powered by `FastMCP`. External MCP clients (like Claude Desktop or the MCP Inspector) can call the ingestion tools directly without writing any Python.

### Starting the Server (via Inspector)

```bash
npx @modelcontextprotocol/inspector .venv/bin/python src/ingestion_pipeline/server.py
```

### Environment Variables

The server reads its configuration from environment variables with sane defaults. You can override them in your shell or `claude_desktop_config.json`:

| Variable              | Default                | Description                          |
| :-------------------- | :--------------------- | :----------------------------------- |
| `QDRANT_URL`          | `http://localhost:6333`| Qdrant instance URL                  |
| `REDIS_HOST`          | `localhost`            | Redis hostname                       |
| `REDIS_PORT`          | `6379`                 | Redis port                           |
| `COLLECTION_BASE_NAME`| `secure_rag`           | Prefix for the Qdrant collection name|
| `EMBED_MODEL_NAME`    | `BAAI/bge-small-en-v1.5`| Embedding model (fastembed or ST)  |

The final Qdrant collection name is derived automatically as `{COLLECTION_BASE_NAME}_{model-id}` (e.g. `secure_rag_baai-bge-small-en-v1.5`).

### Available MCP Tools

Once the server is running, it exposes the following tools to any MCP client:

#### `ingest`
Ingest document pages into the Qdrant vector store. Accepts either a path to a `manifest.json` file (the output of `document_parser`) or an inline JSON array of `Document` objects.

```json
{ "file_path": "sample/documents.json" }
```

#### `search`
Semantic RAG search over ingested document chunks. Returns scored results with metadata and Merkle version context.

```json
{ "query": "balanced batch sampling", "limit": 5, "category": "research" }
```

Pin a search to a specific historical version using `version_root` (obtained from `history`):
```json
{ "query": "introduction", "version_root": "a9b3c4d5..." }
```

#### `audit`
Mathematically verifies that stored Qdrant vectors for a page match the trusted Merkle root in Redis. Re-derives the Merkle tree from scratch.

```json
{ "filename": "report.pdf", "page_index": 0 }
```

#### `history`
Retrieves the full version audit trail for a document — all Merkle root anchors ever recorded, sorted newest-first.

```json
{ "filename": "report.pdf" }
```

#### `purge`
Permanently hard-deletes **all** Qdrant vectors and Redis state for a document (every page, every version). This is **irreversible** and requires an explicit `confirm: true` flag.

```json
{ "filename": "report.pdf", "confirm": true }
```

#### `sync`
Recovery utility. Re-seeds Redis active-root keys from Qdrant's stored root-anchor records. Use this when Redis data has been lost and integrity checks are failing.

```json
{ "filename": "report.pdf" }
```

#### `configure`
Dynamically updates the Qdrant/Redis connection settings and re-initialises the ingestor at runtime. All fields are optional — only those supplied are changed.

```json
{ "model_name": "BAAI/bge-large-en-v1.5" }
```

#### `status`
Returns the active ingestor configuration and pings both Qdrant and Redis to confirm reachability.

```json
{}
```

## Output Schema

Search results from `search` return the following shape for each matching chunk:

```json
{
  "score": 0.921456,
  "content": "The chunk markdown text...",
  "metadata": {
    "filename": "report.pdf",
    "page_index": 0,
    "page_count": 5,
    "category": "research"
  },
  "chunk_hash": "sha256-hex-fingerprint",
  "version_root": "merkle-root-hex",
  "timestamp": "2026-03-29T08:00:00+00:00"
}
```

## End-to-End Usage with Document Parser

The typical pipeline connects the two MCP servers in sequence:

1. **Parse** a document using `document_parser` → `parse_document` → saves `documents.json`
2. **Ingest** the parsed output using `ingestion_pipeline` → `ingest` with `file_path` pointing to the saved JSON
3. **Search** across ingested content with `search`
4. **Audit** at any time with `audit` to verify data has not been tampered with
