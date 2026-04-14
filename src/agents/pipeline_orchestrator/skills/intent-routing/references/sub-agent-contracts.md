# Sub-Agent Contracts Reference

## document_parser_agent

**description field (used for routing):**
"PaddleOCRVL 1.5 document parsing sub-agent. Converts PDFs and images into
structured per-page Document objects with markdown, layout-detected chunks
(with bounding boxes), and Merkle integrity roots."

**Key inputs:**
- `file_path`: absolute path to PDF/image
- `file_content_base64` + `filename`: for in-memory delivery
- Per-call overrides: `use_chart_recognition`, `markdown_ignore_labels`, etc.

**Output (success):** list of per-page Document dicts with `markdown`, `chunks`, `metadata`, `merkle_root`.
**Output (error):** `{"error": "...", "message": "...", "suggestion": "..."}`

**State written:** `parser:last_parsed_file`, `parser:last_results`, `parser:warm`

---

## ingestion_agent

**description field:**
"Versioned Merkle-tree document ingestion sub-agent. Stores parsed Document
objects into Qdrant with Redis-backed integrity proofs."

**Key inputs (ingest):**
- `file_path`: path to manifest.json from parser
- `documents`: inline array (max 500)

**Key inputs (search):**
- `query`: 1–1000 chars
- `category`, `version_root`, `limit` (1–50)

**Output (ingest):** `{ingested, skipped, total_pages, errors[], collection}`
**Output (search):** list of `{score, content, metadata, chunk_hash, version_root, timestamp}`
**Output (history):** `{filename, version_count, versions[{version_root, timestamp, page_index, chunk_count}]}`
**Output (audit):** `{filename, page_index, integrity: "PASSED"|"FAILED", detail}`

**State written:** `ingestor:last_ingested_file`, `ingestor:session_ingest_log`,
`ingestor:version_roots`, `ingestor:connected`

---

## reranker_agent

**description field:**
"Citation-aware hybrid reranker sub-agent. Executes two-stage RRF + Cross-Encoder
reranking over ingested Qdrant document collections."

**Key inputs (rerank_search):**
- `query`: 1–2000 chars
- `retrieval_top_k` (default 50, max 200), `rerank_top_n` (default 5, max 50)
- `category`, `version_root`, `collection_name`, `include_citations_text`

**Output (success):**
```json
{
  "query": "...",
  "result_count": 5,
  "results": [
    {
      "content": "...",
      "final_score": 0.923,
      "ce_score": 0.911,
      "rrf_score": 0.0164,
      "retrieval_sources": ["vector", "sparse"],
      "citation": {
        "filename": "...", "page_index": 1, "chunk_index": 2,
        "chunk_hash": "...", "version_root": "...", "bbox": [...]
      }
    }
  ],
  "pipeline": {"alpha": 0.7, "ce_model": "...", "bm25_active": true}
}
```

**State written:** `reranker:last_query`, `reranker:last_results`, `reranker:session_scores`

---

## Parser output path convention

Parser writes to: `{document_stem}/manifest.json`
Example: parsing `/data/contract_v3.pdf` → output at `contract_v3/manifest.json`
Pass this path as `file_path` to ingestion_agent's `ingest`.