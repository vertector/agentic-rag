# ADR-001: Merkle-Tree Integrity and Multi-Agent Orchestration

## Status
Accepted

## Date
2026-04-14

## Context
The "Chanoch Clerk" project requires a high-throughput, low-latency RAG pipeline that can handle complex documents (PDFs, images) with strong guarantees of data integrity and structural awareness. 

Key requirements:
- **Verifiable Identity**: Every document, page, and chunk must have a deterministic, immutable identity based on its content.
- **Idempotency**: Re-ingesting the same content should be a no-op, avoiding duplicate vectors.
- **Data Integrity**: We need to detect if vectors in the database have been tampered with or corrupted.
- **High-Precision Retrieval**: Standard cosine similarity is often insufficient for complex technical documents.
- **Scalability**: The system should support high-volume parsing and ingestion.

## Decision
We decided on a multi-agent architecture powered by **Google ADK** and **FastMCP**, with a storage layer that utilizes **Merkle Trees** for content identity and integrity.

### 1. Multi-Agent Orchestration
We use a hierarchical agent structure:
- **Pipeline Orchestrator**: The primary user interface, responsible for routing intents (PARSE, INGEST, RETRIEVE) and managing multi-step flows.
- **Document Parser Agent**: Wraps the `document_parser` module, using PaddleOCR-VL and layout detection to extract structured Markdown.
- **Ingestion Agent**: Wraps the `ingestion_pipeline` module, managing storage in Qdrant and integrity proofs in Redis.
- **Reranker Agent**: Wraps the `reranker_pipeline` module, implementing a two-stage hybrid reranking pipeline.

### 2. Merkle-Tree Content Identity
Every document is processed into a Merkle Tree:
- **Chunk Hash**: SHA-256(content + metadata + bounding box).
- **Page Hash**: SHA-256(sorted list of chunk hashes).
- **Document Hash**: SHA-256(metadata + sorted list of page hashes).

The Merkle root for each page/version is stored in **Redis** as a trust anchor. Any changes to the chunks or their order will change the root hash, allowing for cryptographic verification of the stored data.

### 3. Two-Stage Retrieval
To achieve high precision, we implement:
- **Stage 1 (Recall)**: Hybrid search combining Qdrant dense vector search with BM25 sparse keyword search using Reciprocal Rank Fusion (RRF).
- **Stage 2 (Precision)**: Cross-Encoder reranking of the top-K candidates to provide a final ranked list.

## Alternatives Considered

### Single-Agent / Monolithic Script
- **Pros**: Simpler to build initially.
- **Cons**: Harder to maintain, difficult to swap components (e.g., changing the OCR engine), and lacks the "agentic" flexibility for complex user intents.
- **Rejected**: The agentic approach allows for better intent routing and autonomous multi-step execution.

### ChromaDB / FAISS without Merkle Roots
- **Pros**: Lower overhead, easier setup.
- **Cons**: No native support for content-based deduplication or verifiable integrity proofs.
- **Rejected**: Merkle trees are essential for our requirement of verifiable identity and idempotent ingestion at scale.

## Consequences
- **Integrity**: We can audit the database at any time to ensure vectors match the original content.
- **Efficiency**: Git-style hashing avoids re-embedding unchanged content, saving API costs and compute.
- **Complexity**: Requires running both Qdrant and Redis, plus an LLM (Ollama or Gemini) for orchestration.
- **Latency**: The two-stage reranking adds latency, mitigated by LRU caching of scores.
