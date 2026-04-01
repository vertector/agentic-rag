# Qdrant Document Store

A high-throughput, low-latency vector database module built on [Qdrant](https://qdrant.tech),
purpose-built for the `Document / Chunk / Grounding / Metadata` schema.

Key capabilities:

- **Git-style Merkle hashing** — only re-embeds and re-upserts chunks that actually changed
- **Robust idempotency** — safe to call `upsert_documents` on every pipeline run
- **Async-first** — full `asyncio` support with parallel batch workers
- **gRPC transport** — lower latency than REST for high-volume workloads
- **Advanced filtering** — 15 indexed payload fields, rich compound filter helpers
- **Batch search** — N queries in a single round-trip

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Start Qdrant](#start-qdrant)
4. [Module files](#module-files)
5. [Core concepts](#core-concepts)
6. [QuickStart — five minutes to first search](#quickstart--five-minutes-to-first-search)
7. [Creating the store](#creating-the-store)
8. [Upserting documents](#upserting-documents)
9. [How idempotency works](#how-idempotency-works)
10. [Searching](#searching)
11. [Filtering](#filtering)
12. [Deleting](#deleting)
13. [Async usage](#async-usage)
14. [Full API reference](#full-api-reference)
15. [Running the tests](#running-the-tests)
16. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| Python | 3.10+ | 3.12 recommended |
| uv | any recent | `pip install uv` if absent |
| Docker (optional) | 24+ | Only needed for a persistent server |

---

## Installation

### Using uv (recommended)

```bash
# Inside your project directory
uv add qdrant-client pydantic
```

### Using pip

```bash
pip install qdrant-client pydantic
```

> **Note:** `qdrant-client` bundles the async client and optional gRPC transport.
> To enable gRPC (strongly recommended for production) install the extras:
> ```bash
> uv add "qdrant-client[fastembed]"   # fastembed for local embeddings
> # or just the grpc extra
> pip install "qdrant-client[grpc]"
> ```

---

## Start Qdrant

### Option A — Local in-memory (zero setup, tests only)

Pass `_in_memory=True` to `QdrantDocumentStore`. No server required. Data is
lost when the process exits.

```python
store = QdrantDocumentStore(
    collection_name="my_docs",
    vector_size=1024,
    _in_memory=True,
)
```

### Option B — Docker Compose (recommended)

A `docker-compose.yml` is included in the project. Start it with:

```bash
docker compose up -d
```

Stop it (data is preserved in the named volume):

```bash
docker compose down
```

Wipe everything including stored vectors:

```bash
docker compose down -v
```

The compose file exposes both ports and adds a health check so dependent
services wait until Qdrant is ready:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST  — always required
      - "6334:6334"   # gRPC  — only needed when prefer_grpc=True
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:6333/healthz || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Option C — Docker CLI (quick one-off)

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

- Port `6333` — REST API (always required)
- Port `6334` — gRPC API (only needed if you set `prefer_grpc=True`; safe to omit for local dev)
- Data is persisted to `./qdrant_data` on your host

### Option D — Qdrant Cloud

Sign up at <https://cloud.qdrant.io>, create a cluster, and copy the URL and API key:

```python
store = QdrantDocumentStore(
    collection_name="my_docs",
    vector_size=1024,
    url="https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io",
    api_key="<your-api-key>",
)
```

---

## Module files

```
qdrant_store.py          # Core module — copy this into your project
ingest.py                # CLI script: load JSON → embed → upsert
docker-compose.yml       # Qdrant service with health check
tests_qdrant_store.py    # Full pytest test suite (24 tests)
usage_example.py         # Runnable end-to-end walkthrough (in-memory)
README.md                # This file
```

Copy `qdrant_store.py`, `ingest.py`, and `docker-compose.yml` into your project.

---

## Core concepts

### The data schema

The module is built around four Pydantic models that live inside `qdrant_store.py`
(they are also importable from your own schema module — just swap the imports at
the top of `qdrant_store.py`).

```
Document                    ← one page of a PDF / one logical unit
├── doc_id: UUID
├── markdown: str
├── metadata: Metadata
│   ├── filename: str       ← source file path
│   ├── page_index: int
│   ├── page_count: int
│   └── page_image_base64: str
└── chunks: List[Chunk]     ← each chunk becomes one Qdrant point
    └── Chunk
        ├── chunk_id: UUID  ← becomes the Qdrant point ID
        ├── chunk_markdown: str
        └── grounding: Grounding
            ├── chunk_type: str   (e.g. "abstract", "text", "footer")
            ├── bbox: [x1,y1,x2,y2]
            ├── page_index: int
            └── score: float
```

### Git-style Merkle hash tree

Every time you call `upsert_documents`, the module computes three levels of
SHA-256 hashes before touching the database:

```
chunk_hash  ←  SHA-256(chunk_id + markdown + bbox + chunk_type + score)
page_hash   ←  SHA-256(sorted chunk_hashes for all chunks on the page)
doc_hash    ←  SHA-256(doc_id + filename + page_index + page_hash)
```

All three hashes are stored as indexed payload fields on every Qdrant point.
This means:

- **Chunk level** — if only one sentence changed, only that chunk is re-upserted
- **Page level** — `page_hash` changes when any chunk on that page changes
- **Document level** — `doc_hash` changes when any page changes

Before writing anything, the module calls `retrieve()` to fetch the stored
`content_hash` for all candidate chunk IDs in a single round-trip, then skips
chunks whose hash is identical.

---

## QuickStart — five minutes to first search

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Run the bundled ingest script against your documents
uv run python ingest.py --docs ./bbs/documents.json

# 3. Or run the full usage example (uses in-memory Qdrant, no server needed)
uv run python usage_example.py
```

The ingest script accepts flags for every tunable parameter:

```bash
uv run python ingest.py \
  --docs ./bbs/documents.json \
  --collection research_papers \
  --batch-size 4 \       # lower if you hit OOM during embedding
  --force                # re-embed everything, ignoring stored hashes
```

---

## Creating the store

```python
from qdrant_store import QdrantDocumentStore

store = QdrantDocumentStore(
    collection_name="research_papers",  # created automatically if absent
    vector_size=1024,                   # Qwen3-Embedding-0.6B → 1024 dims
    url="http://localhost:6333",        # Qdrant REST port (always available)
    api_key=None,                       # set for Qdrant Cloud
    # prefer_grpc=True,                 # uncomment only after exposing port 6334
    batch_size=256,                     # chunks per upsert batch
    parallel=4,                         # concurrent async batch workers
)

# Create the collection + all payload indexes (idempotent — safe to call on every startup)
store.ensure_collection()
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | — | Qdrant collection name. Created if it doesn't exist. |
| `vector_size` | `int` | — | Embedding dimensionality. Must match your model (e.g. 1536 for `text-embedding-3-small`). |
| `url` | `str` | `http://localhost:6333` | Qdrant server URL. |
| `api_key` | `str \| None` | `None` | API key for Qdrant Cloud. |
| `prefer_grpc` | `bool` | `False` | Use gRPC transport (port 6334) for lower latency. Requires `-p 6334:6334` in Docker. Leave `False` for local dev; enable for production once port 6334 is confirmed open. |
| `batch_size` | `int` | `256` | Number of points per upsert batch. Tune based on vector size and network. |
| `parallel` | `int` | `4` | Max concurrent async batch workers. Only affects `aupsert_documents`. |
| `distance` | `Distance` | `COSINE` | Vector distance metric. Options: `COSINE`, `EUCLID`, `DOT`. |
| `on_disk_payload` | `bool` | `False` | Store payload on disk instead of RAM. Reduces memory but increases latency. |
| `_in_memory` | `bool` | `False` | Ephemeral in-process store. For tests only — data lost on exit. |

---

## Upserting documents

The contract is simple: you supply the `Document` list and a flat
`{str(chunk_id): vector}` dict. The store handles hashing, deduplication,
and Qdrant batching internally.

### Why Qwen3-Embedding needs a small `batch_size`

`Qwen/Qwen3-Embedding-0.6B` is a **decoder-only LLM**, not a small BERT
encoder. Its attention matrix grows as `O(seq_len² × batch_size)`. Passing
too many chunks at once causes PyTorch to allocate a single multi-gigabyte
buffer and crash:

```
RuntimeError: Invalid buffer size: 14.41 GiB
```

The fix is to embed in small batches of 4–8 chunks and accumulate the
results. Passing `batch_size=4` to `model.encode()` tells sentence-transformers
to run multiple small forward passes instead of one giant one.

### Choosing the right `batch_size`

| Hardware | Safe `batch_size` | Notes |
|----------|------------------|-------|
| CPU only | `4` | Slow but stable |
| 8 GB RAM / integrated GPU | `4–8` | Watch Activity Monitor / `htop` |
| 16 GB RAM | `8–16` | |
| GPU with 8 GB VRAM | `16–32` | Use `device="cuda"` |
| GPU with 24+ GB VRAM | `64–128` | |

Start at `4` and double until you see a memory error, then step back.

---

### Loading documents from JSON (the real workflow)

```python
import json
from models import Document
from qdrant_store import QdrantDocumentStore

with open("./bbs/documents.json", "r", encoding="utf-8") as f:
    docs = [Document(**d) for d in json.load(f)]

store = QdrantDocumentStore(
    collection_name="research_papers",
    vector_size=1024,          # Qwen3-Embedding-0.6B outputs 1024-dim vectors
    url="http://localhost:6333",
)
store.ensure_collection()
```

### Step 1 — flatten all chunks into a single ordered list

```python
# Each element is (chunk_id_str, text) — order is preserved throughout
chunk_pairs = [
    (str(chunk.chunk_id), chunk.chunk_markdown)
    for doc in docs
    for chunk in doc.chunks
    if chunk.chunk_markdown.strip()      # skip empty / whitespace-only chunks
]

chunk_ids   = [p[0] for p in chunk_pairs]
chunk_texts = [p[1] for p in chunk_pairs]
```

### Step 2 — embed with an explicit `batch_size`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# batch_size controls how many chunks go through the model at once.
# Keep it small (4–8) for Qwen3 on CPU/limited RAM.
vectors = model.encode(
    chunk_texts,
    batch_size=4,          # ← the critical parameter; increase only if RAM allows
    show_progress_bar=True,
    normalize_embeddings=True,
).tolist()
```

If you prefer the LangChain wrapper (matches your existing ChromaDB setup):

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    encode_kwargs={"batch_size": 4, "normalize_embeddings": True},
)
vectors = embedder.embed_documents(chunk_texts)
```

For OpenAI (no local memory issue — batching is handled server-side):

```python
from openai import OpenAI

client = OpenAI()
resp = client.embeddings.create(
    model="text-embedding-3-small",   # 1536-dim; set vector_size=1536 above
    input=chunk_texts,
)
vectors = [item.embedding for item in resp.data]
```

### Step 3 — zip IDs to vectors

`chunk_ids` and `vectors` are in the same order, so a single `zip` is all
that's needed:

```python
embeddings: dict[str, list[float]] = dict(zip(chunk_ids, vectors))
```

### Step 4 — upsert

```python
stats = store.upsert_documents(docs, embeddings)

print(f"Total chunks:      {stats.total_chunks}")
print(f"Upserted:          {stats.upserted}")
print(f"Skipped unchanged: {stats.skipped_unchanged}")
print(f"Errors:            {stats.errors}")
```

---

### Complete example (copy-paste ready)

```python
import json
from models import Document
from qdrant_store import QdrantDocumentStore
from sentence_transformers import SentenceTransformer

# ── 1. Load documents ────────────────────────────────────────────────────────
with open("./bbs/documents.json", "r", encoding="utf-8") as f:
    docs = [Document(**d) for d in json.load(f)]

# ── 2. Set up store ──────────────────────────────────────────────────────────
store = QdrantDocumentStore(
    collection_name="research_papers",
    vector_size=1024,              # Qwen3-Embedding-0.6B → 1024 dims
    url="http://localhost:6333",
)
store.ensure_collection()

# ── 3. Flatten all non-empty chunks ──────────────────────────────────────────
chunk_pairs = [
    (str(chunk.chunk_id), chunk.chunk_markdown)
    for doc in docs
    for chunk in doc.chunks
    if chunk.chunk_markdown.strip()
]
chunk_ids   = [p[0] for p in chunk_pairs]
chunk_texts = [p[1] for p in chunk_pairs]

# ── 4. Embed in small batches (critical for LLM-based embedding models) ───────
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
vectors = model.encode(
    chunk_texts,
    batch_size=4,                  # keep low on CPU / limited RAM
    show_progress_bar=True,
    normalize_embeddings=True,
).tolist()

# ── 5. Build the {chunk_id: vector} dict ─────────────────────────────────────
embeddings = dict(zip(chunk_ids, vectors))

# ── 6. Upsert ────────────────────────────────────────────────────────────────
stats = store.upsert_documents(docs, embeddings)
print(f"Upserted {stats.upserted} / {stats.total_chunks} chunks")
```

---

### The three embedding patterns and when to use each

```python
# ── Pattern A: sentence-transformers (local, memory-controlled) ───────────────
# Use when: running locally, need cost-free embeddings, have a GPU
# Key:      always set batch_size explicitly
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
vectors = model.encode(chunk_texts, batch_size=4, normalize_embeddings=True).tolist()


# ── Pattern B: LangChain HuggingFaceEmbeddings (matches your ChromaDB setup) ──
# Use when: you already use LangChain elsewhere and want one embedder instance
# Key:      pass batch_size inside encode_kwargs
from langchain_huggingface import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    encode_kwargs={"batch_size": 4, "normalize_embeddings": True},
)
vectors = embedder.embed_documents(chunk_texts)


# ── Pattern C: OpenAI API (remote, no local memory issue) ────────────────────
# Use when: you have an OpenAI key and want production-grade embeddings
# Key:      set vector_size=1536 in QdrantDocumentStore
from openai import OpenAI
resp = OpenAI().embeddings.create(model="text-embedding-3-small", input=chunk_texts)
vectors = [item.embedding for item in resp.data]
```

### UpsertStats fields

| Field | Type | Description |
|-------|------|-------------|
| `total_chunks` | `int` | Total chunks seen across all documents |
| `upserted` | `int` | Chunks actually written to Qdrant |
| `skipped_unchanged` | `int` | Chunks skipped because their hash matched |
| `errors` | `list[str]` | Error messages (missing embeddings, failed batches) |
| `changed_ratio` | `float` | `upserted / total_chunks` |

### Force re-upsert (bypass hash check)

```python
# Re-embed and re-write everything regardless of whether it changed
stats = store.upsert_documents([doc], embeddings, force=True)
```

---

## How idempotency works

Calling `upsert_documents` multiple times with the same data is safe and cheap:

```python
# First call — writes 4 chunks
stats = store.upsert_documents([doc], embeddings)
# stats.upserted == 4, stats.skipped_unchanged == 0

# Second call — zero writes, zero re-embeddings needed
stats = store.upsert_documents([doc], embeddings)
# stats.upserted == 0, stats.skipped_unchanged == 4
```

### Document-level short-circuit

For large documents you can check at the document level before even computing
embeddings:

```python
if store.is_doc_unchanged(doc):
    print("Document is identical to what's stored — skip embedding entirely")
else:
    embeddings = embed_all_chunks(doc)
    store.upsert_documents([doc], embeddings)
```

`is_doc_unchanged` scrolls for any point whose `doc_hash` matches. A single hit
means the whole document is current. This avoids all embedding API calls for
unchanged documents.

### Handling a partial edit (e.g., one chunk changed)

```python
from qdrant_store import Chunk

# Only the chunk whose content changed needs a new embedding
old_chunk = doc.chunks[2]
updated_chunk = Chunk(
    chunk_id=old_chunk.chunk_id,          # same ID — will overwrite
    chunk_markdown="Updated text here.",   # changed content
    grounding=old_chunk.grounding,
)

# Update the doc
updated_doc = doc.model_copy(
    update={"chunks": doc.chunks[:2] + [updated_chunk] + doc.chunks[3:]}
)

# Re-embed only the changed chunk
embeddings[str(updated_chunk.chunk_id)] = embed([updated_chunk.chunk_markdown])[0]

stats = store.upsert_documents([updated_doc], embeddings)
# stats.upserted == 1   ← only the changed chunk
# stats.skipped_unchanged == N-1
```

---

## Searching

### Basic nearest-neighbour search

```python
query_vector = embed(["out-of-distribution generalization in medical imaging"])[0]

hits = store.search(query_vector, limit=10)

for hit in hits:
    print(f"Score: {hit.score:.4f}")
    print(f"Type:  {hit.payload['chunk_type']}")
    print(f"Text:  {hit.payload['chunk_markdown'][:120]}")
    print()
```

### Search with a score threshold

```python
hits = store.search(query_vector, limit=20, score_threshold=0.75)
# Only returns points with cosine similarity >= 0.75
```

### Batch search (N queries in one round-trip)

```python
query_vectors = [
    embed(["pathology detection deep learning"])[0],
    embed(["balanced mini-batch sampling"])[0],
    embed(["out-of-distribution generalization"])[0],
]

results = store.search_batch(query_vectors, limit=5)

for i, hits in enumerate(results):
    print(f"Query {i}: {len(hits)} hits, top score {hits[0].score:.4f}")
```

Batch search is significantly more efficient than N individual calls — it uses
Qdrant's `query_batch_points` endpoint, which is a single request.

---

## Filtering

The `QdrantFilters` class (also accessible as `store.filters`) provides
pre-built filters for every indexed payload field.

### Simple filters

```python
from qdrant_store import QdrantFilters

# All chunks from a specific file
hits = store.search(q, filter=QdrantFilters.by_filename("papers/my_paper.pdf"))

# All chunks belonging to one document
hits = store.search(q, filter=QdrantFilters.by_doc_id(doc.doc_id))

# Only abstract chunks
hits = store.search(q, filter=QdrantFilters.by_chunk_type("abstract"))

# Chunks on pages 2–5
hits = store.search(q, filter=QdrantFilters.by_page_range(2, 5))

# High-confidence detections only
hits = store.search(q, filter=QdrantFilters.by_detection_score(min_score=0.9))
```

### Compound filter (AND logic)

```python
flt = QdrantFilters.compound(
    filename="papers/my_paper.pdf",
    chunk_types=["abstract", "text"],   # matches either type (OR within the list)
    page_min=0,
    page_max=3,
    min_score=0.85,
    has_image=False,
)

hits = store.search(query_vector, limit=10, filter=flt)
```

All arguments to `compound()` are optional — only the ones you pass are
included in the filter.

### Compound filter parameters

| Parameter | Type | Matches |
|-----------|------|---------|
| `filename` | `str` | Exact source file path |
| `doc_id` | `str \| UUID` | Exact document ID |
| `chunk_types` | `list[str]` | Any of the given types (OR) |
| `page_min` | `int` | `page_index >= page_min` |
| `page_max` | `int` | `page_index <= page_max` |
| `min_score` | `float` | `detection_score >= min_score` |
| `has_image` | `bool` | Page has / doesn't have a base64 image |

### Hash-based filters (for pipeline use)

```python
# Find all chunks that already have this doc_hash (fast doc-level staleness check)
hits = store.search(q, filter=QdrantFilters.by_doc_hash("abc123..."))

# Find all chunks on a page that has changed
hits = store.search(q, filter=QdrantFilters.by_page_hash("def456..."))
```

### Building custom filters

The `QdrantFilters` methods return standard `qdrant_client.models.Filter` objects.
You can combine them manually for complex logic:

```python
from qdrant_client import models as qmodels

# (filename == "a.pdf" AND chunk_type == "abstract") OR (filename == "b.pdf")
flt = qmodels.Filter(
    should=[
        qmodels.Filter(must=[
            qmodels.FieldCondition(key="filename",   match=qmodels.MatchValue(value="a.pdf")),
            qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value="abstract")),
        ]),
        qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value="b.pdf")),
    ]
)
hits = store.search(query_vector, limit=10, filter=flt)
```

---

## Deleting

```python
# Delete all chunks belonging to a document
store.delete_by_doc_id(doc.doc_id)

# Delete all chunks from a source file
store.delete_by_filename("papers/old_paper.pdf")
```

### Re-index a document (delete + upsert)

```python
# Atomically replace a document's chunks
store.delete_by_doc_id(doc.doc_id)
store.upsert_documents([updated_doc], new_embeddings)
```

Or simply call `upsert_documents` with `force=True` — it will overwrite existing
points and the Qdrant upsert semantics handle removed chunks automatically if
you also delete first.

---

## Async usage

All methods have `a`-prefixed async equivalents. The async client uses parallel
batch workers bounded by the `parallel` parameter.

```python
import asyncio
from qdrant_store import QdrantDocumentStore

async def main():
    store = QdrantDocumentStore(
        collection_name="research_papers",
        vector_size=1536,
        url="http://localhost:6333",
        parallel=8,    # 8 concurrent batch workers
    )
    await store.aensure_collection()

    stats = await store.aupsert_documents([doc], embeddings)
    print(f"Upserted: {stats.upserted}")

    hits = await store.asearch(query_vector, limit=10)

    unchanged = await store.ais_doc_unchanged(doc)

    await store.adelete_by_doc_id(doc.doc_id)
    await store.aclose()

asyncio.run(main())
```

### Async method reference

| Async method | Sync equivalent |
|--------------|----------------|
| `aensure_collection()` | `ensure_collection()` |
| `aupsert_documents(docs, embs, *, force)` | `upsert_documents(...)` |
| `asearch(vec, *, limit, filter, ...)` | `search(...)` |
| `ais_doc_unchanged(doc)` | `is_doc_unchanged(doc)` |
| `adelete_by_doc_id(doc_id)` | `delete_by_doc_id(doc_id)` |
| `aclose()` | `close()` |

---

## Full API reference

### `QdrantDocumentStore`

#### `ensure_collection() / aensure_collection()`
Creates the Qdrant collection if it does not exist and creates all 15 payload
indexes. Safe to call on every application startup.

#### `upsert_documents(documents, embeddings, *, force=False) → UpsertStats`
#### `aupsert_documents(documents, embeddings, *, force=False) → UpsertStats`

| Argument | Type | Description |
|----------|------|-------------|
| `documents` | `Sequence[Document]` | Pages / documents to ingest |
| `embeddings` | `dict[str, list[float]]` | `{str(chunk_id): vector}` for every chunk |
| `force` | `bool` | Skip hash check and upsert all (default `False`) |

#### `search(query_vector, *, limit, filter, score_threshold, with_payload, with_vectors)`
#### `asearch(...)`
Returns `list[ScoredPoint]`. Each point has `.score` and `.payload` (all indexed fields).

#### `search_batch(query_vectors, *, limit, filters, score_threshold)`
Single round-trip batch search. `filters` must be the same length as
`query_vectors` if provided (`None` entries mean no filter for that query).
Returns `list[list[ScoredPoint]]`.

#### `is_doc_unchanged(doc) → bool`
#### `ais_doc_unchanged(doc) → bool`
Returns `True` if the database already contains a point with a matching
`doc_hash`. Useful for skipping embedding calls entirely.

#### `delete_by_doc_id(doc_id)`
#### `adelete_by_doc_id(doc_id)`
Removes all Qdrant points whose `doc_id` payload field matches.

#### `delete_by_filename(filename)`
Removes all Qdrant points whose `filename` payload field matches.

#### `scroll_all(filter=None, batch_size=1000) → Iterable[Record]`
Lazy generator that pages through all matching points. Useful for bulk
exports or re-indexing pipelines.

```python
for record in store.scroll_all(filter=QdrantFilters.by_filename("old.pdf")):
    print(record.id, record.payload["chunk_type"])
```

#### `delete_collection()`
Drops the entire collection. Irreversible.

#### `close() / aclose()`
Closes the underlying client connections. Call at application shutdown.

### Standalone hash functions

```python
from qdrant_store import chunk_content_hash, page_content_hash, document_content_hash

h = chunk_content_hash(chunk)       # str (64-char hex)
h = page_content_hash(chunks)       # str
h = document_content_hash(doc)      # str
```

### `QdrantFilters` (also `store.filters`)

```python
QdrantFilters.by_filename(filename)
QdrantFilters.by_doc_id(doc_id)
QdrantFilters.by_chunk_type(chunk_type)
QdrantFilters.by_page_range(page_min, page_max)
QdrantFilters.by_detection_score(min_score)
QdrantFilters.by_doc_hash(doc_hash)
QdrantFilters.by_page_hash(page_hash)
QdrantFilters.by_chunk_type_and_filename(chunk_type, filename)
QdrantFilters.compound(*, filename, doc_id, chunk_types, page_min, page_max, min_score, has_image)
```

---

## Running the tests

The test suite runs against an in-process Qdrant instance — no Docker required.

```bash
# Install test dependencies
uv add --dev pytest

# Run all 24 tests
uv run pytest tests_qdrant_store.py -v
```

Expected result: `24 passed`.

The single warning about payload indexes having no effect in local mode is
expected — indexes are a server-side feature and are created correctly when
running against a real Qdrant instance.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'qdrant_client'`

```bash
uv add qdrant-client
```

### `ConnectionRefusedError` / `Failed to connect to localhost:6333`

Qdrant is not running. Start it with Docker:

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or use `_in_memory=True` for local development.

### `_InactiveRpcError: failed to connect to all addresses … port 6334: Connection refused`

gRPC is trying to connect on port 6334, which isn't open. This happens when
`prefer_grpc=True` is set — either explicitly in your code or from an older
version of this module that had it as the default.

**Check your code first.** If you have `prefer_grpc=True` in your
`QdrantDocumentStore(...)` call, remove it:

```python
# ❌ causes the error if port 6334 isn't exposed
store = QdrantDocumentStore(
    collection_name="research_papers",
    vector_size=1024,
    url="http://localhost:6333",
    prefer_grpc=True,             # ← remove this line
)

# ✅ REST on port 6333, works with any Qdrant setup
store = QdrantDocumentStore(
    collection_name="research_papers",
    vector_size=1024,
    url="http://localhost:6333",
)
```

**If you want gRPC** for production throughput, use Docker Compose which
exposes both ports, then opt in explicitly:

```bash
docker compose up -d    # exposes 6333 and 6334
```

```python
store = QdrantDocumentStore(
    ...,
    prefer_grpc=True,   # safe once docker compose is running
)
```

### `RuntimeError: Invalid buffer size` (OOM during embedding)

This is a memory error from PyTorch, not from Qdrant. It means the embedding
model tried to allocate an attention buffer larger than available RAM/VRAM.
`Qwen3-Embedding-0.6B` is a decoder-only LLM whose attention cost scales as
`O(seq_len² × batch_size)` — it is not a small BERT encoder.

Fix: lower `batch_size` in your `model.encode()` call:

```python
# sentence-transformers
vectors = model.encode(chunk_texts, batch_size=4, normalize_embeddings=True).tolist()

# LangChain wrapper
embedder = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    encode_kwargs={"batch_size": 4, "normalize_embeddings": True},
)
```

Start at `batch_size=4` and double until it crashes, then step back one level.

### `ValueError: vector_size mismatch`

The `vector_size` you passed to `QdrantDocumentStore` does not match the
dimensionality of your embedding model. Common correct values:

| Model | `vector_size` |
|-------|--------------|
| `Qwen/Qwen3-Embedding-0.6B` | `1024` |
| `text-embedding-3-small` (OpenAI) | `1536` |
| `text-embedding-3-large` (OpenAI) | `3072` |
| `all-MiniLM-L6-v2` | `384` |
| `BAAI/bge-large-en-v1.5` | `1024` |

Delete the collection and recreate it with the correct size:

```python
store.delete_collection()
store.ensure_collection()
```

### Upsert is slower than expected

- Increase `batch_size` (try 512 or 1024 for large vectors)
- Enable gRPC (`prefer_grpc=True`)
- For async pipelines, increase `parallel` (try 8–16 for high-core machines)
- Check that `indexing_threshold` isn't set too low — the default 20,000 means
  HNSW indexing only starts after 20k points, which is correct for throughput

### All chunks are re-upserted on every run

The `chunk_id` UUIDs are being regenerated on each parse run. The hash
comparison requires stable IDs. Ensure your parser produces deterministic
UUIDs (e.g. `uuid5(NAMESPACE, filename + str(page_index) + str(chunk_index))`).