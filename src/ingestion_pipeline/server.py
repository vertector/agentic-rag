"""
ingestion_pipeline_mcp — FastMCP server wrapping AsyncMerkleQdrantIngestor.

Transport:  stdio  (registered in claude_desktop_config.json)

Architecture notes
------------------
· The ingestor is fully async (qdrant-client, redis.asyncio) — no executor wrapping needed.
· A lifespan context manager calls ingestor.setup() once at server startup so every tool
  call gets a ready-to-use, connection-verified ingestor instance.
· Configuration is loaded from environment variables with sane defaults; a
  `ingestion_configure` tool allows runtime overrides (restarts the ingestor).
· Document data can be delivered two ways:
    1. file_path  — path to a documents.json produced by document_parser_mcp (fastest)
    2. documents  — inline JSON array (same schema) for direct LLM use
  Both paths normalise to List[Document] before hitting the ingestor.

Import layout
-------------
Place this file in the same directory as ingestion_pipeline.py (and schemas.py /
shared/schemas.py).  Adjust the import lines marked ← if your package layout differs.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Ensure the parent directory (src root) is in sys.path so that absolute imports work from MCP.
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor, IngestorError
from shared.schemas import Document, Metadata, Chunk, Grounding, PipelineSettings  # noqa: F401
from shared.env_loader import load_env

# Load environment variables (api keys, connection URLs) if present
load_env()


# ---------------------------------------------------------------------------
# Constants / env-var config
# ---------------------------------------------------------------------------

SERVER_NAME = "ingestion_pipeline_mcp"

QDRANT_URL          = os.getenv("QDRANT_URL",           "http://localhost:6333")
REDIS_HOST          = os.getenv("REDIS_HOST",           "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT",       "6379"))
COLLECTION_BASE     = os.getenv("COLLECTION_BASE_NAME", "secure_rag")
EMBED_MODEL_NAME    = os.getenv("EMBED_MODEL_NAME",     "BAAI/bge-small-en-v1.5")

MAX_INLINE_DOCS     = 500    # guard against accidentally huge inline payloads
MAX_SEARCH_LIMIT    = 50     # cap for secure_search results

logger = logging.getLogger(SERVER_NAME)


# ---------------------------------------------------------------------------
# Lifespan — start ingestor once, share across all tool calls
# ---------------------------------------------------------------------------

_ingestor: Optional[AsyncMerkleQdrantIngestor] = None


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """
    Initialise and tear-down the ingestor for the server's lifetime.

    Called once when FastMCP starts.  setup() verifies Qdrant + Redis
    connectivity and creates the collection + payload indexes if absent.
    """
    global _ingestor
    logger.info(
        "Connecting to Qdrant=%s  Redis=%s:%d  model=%s",
        QDRANT_URL, REDIS_HOST, REDIS_PORT, EMBED_MODEL_NAME,
    )
    _ingestor = AsyncMerkleQdrantIngestor(
        qdrant_url=QDRANT_URL,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        collection_base_name=COLLECTION_BASE,
        model_name=EMBED_MODEL_NAME,
    )
    try:
        await _ingestor.setup()
        logger.info("Ingestor ready — collection: %s", _ingestor.collection_name)
    except IngestorError as exc:
        # Log and continue; individual tool calls will surface errors
        logger.error("Ingestor setup failed: %s", exc)

    yield {"ingestor": _ingestor}

    # Cleanup
    try:
        await _ingestor.qdrant.close()
        await _ingestor.redis.aclose()
    except Exception:
        pass


mcp = FastMCP(SERVER_NAME, lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_ingestor(ctx: Context) -> AsyncMerkleQdrantIngestor:
    """Retrieve the lifespan-managed ingestor; raises if not ready."""
    global _ingestor
    if _ingestor is None:
        raise IngestorError(
            "Ingestor is not initialised. Check that Qdrant and Redis are reachable "
            "and that the server started without errors."
        )
    return _ingestor


def _load_documents_from_file(file_path: str) -> List[Document]:
    """
    Read a documents.json file (output of document_parser_mcp) and
    deserialise each entry as a Document.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: JSON is invalid or doesn't match the Document schema.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"documents.json not found at '{file_path}'. "
            "Run document_parser_parse first and pass the output path here."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{file_path}': {exc}") from exc

    try:
        return [Document.model_validate(d) for d in raw]
    except Exception as exc:
        raise ValueError(
            f"One or more entries in '{file_path}' don't match the Document schema: {exc}"
        ) from exc


def _load_documents_from_inline(raw: List[dict]) -> List[Document]:
    """Deserialise an inline list of Document dicts."""
    try:
        return [Document.model_validate(d) for d in raw]
    except Exception as exc:
        raise ValueError(f"Inline document data failed schema validation: {exc}") from exc


def _error(kind: str, message: str, suggestion: str = "") -> str:
    """Return a consistent JSON error envelope."""
    payload: Dict[str, Any] = {"error": kind, "message": message}
    if suggestion:
        payload["suggestion"] = suggestion
    return json.dumps(payload, indent=2)


def _resolve_filename(v: str) -> str:
    """
    If the provided string is a path to a .json file (e.g. documents.json), 
    attempt to extract its internal metadata.filename. Always returns the 
    base filename (e.g. 'doc.pdf').
    """
    p = Path(v)
    if p.exists() and p.suffix.lower() == ".json":
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    v = data[0].get("metadata", {}).get("filename", v)
        except Exception:
            pass
    return Path(v).name


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class IngestInput(BaseModel):
    """Input model for ingest_data."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # --- document delivery (exactly one required) ---
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a documents.json file produced by document_parser_mcp "
            "(e.g. 'report/documents.json'). Each entry in the array is ingested "
            "as one document page. Mutually exclusive with `documents`."
        ),
    )
    documents: Optional[List[dict]] = Field(
        default=None,
        max_length=MAX_INLINE_DOCS,
        description=(
            "Inline array of Document objects (same schema as documents.json). "
            "Use when passing data directly without touching disk. "
            f"Maximum {MAX_INLINE_DOCS} documents per call. "
            "Mutually exclusive with `file_path`."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map file_name -> file_path for compatibility
            if "file_name" in data and "file_path" not in data:
                data["file_path"] = data.pop("file_name")

            # Resolve to absolute path if provided
            if data.get("file_path"):
                data["file_path"] = str(Path(data["file_path"]).resolve())
        return data

    @model_validator(mode="after")
    def _check_delivery(self) -> "IngestInput":
        has_path = self.file_path is not None
        has_docs = self.documents is not None
        if not has_path and not has_docs:
            raise ValueError("Provide either `file_path` or `documents`.")
        if has_path and has_docs:
            raise ValueError("`file_path` and `documents` are mutually exclusive.")
        return self


class VerifyIntegrityInput(BaseModel):
    """Input model for ingestion_verify_integrity."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(
        ...,
        min_length=1,
        description=(
            "Original document filename as stored during ingestion "
            "(e.g. 'report.pdf'). Automatically converted to basename."
        ),
    )

    @field_validator("filename", mode="after")
    @classmethod
    def _extract_basename(cls, v: str) -> str:
        return _resolve_filename(v)

    page_index: int = Field(
        ...,
        ge=1,
        description="1-indexed page number to verify (e.g. 1 for the first page).",
    )


class SearchInput(BaseModel):
    """Input model for ingestion_search."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Natural-language query for semantic search "
            "(e.g. 'right to a fair trial', 'Q3 revenue breakdown')."
        ),
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional document category filter (e.g. 'legal', 'medical', 'finance'). "
            "Matches the `category` field set during ingestion."
        ),
    )
    version_root: Optional[str] = Field(
        default=None,
        description=(
            "Merkle root hash to pin the search to a specific historical snapshot. "
            "Omit to query the current active version. "
            "Obtain valid roots from `ingestion_get_history`."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description=f"Maximum number of results to return (1–{MAX_SEARCH_LIMIT}, default 5).",
    )


class HistoryInput(BaseModel):
    """Input model for ingestion_get_history."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(
        ...,
        min_length=1,
        description="Document filename to retrieve the version audit trail for. Automatically converted to basename.",
    )

    @field_validator("filename", mode="after")
    @classmethod
    def _extract_basename(cls, v: str) -> str:
        return _resolve_filename(v)



class PurgeInput(BaseModel):
    """Input model for ingestion_purge_document."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(
        ...,
        min_length=1,
        description=(
            "Document filename to permanently delete (all versions, all pages). "
            "Automatically converted to basename. This operation is irreversible."
        ),
    )

    @field_validator("filename", mode="after")
    @classmethod
    def _extract_basename(cls, v: str) -> str:
        return _resolve_filename(v)

    confirm: bool = Field(
        ...,
        description=(
            "Must be set to `true` to confirm the destructive delete. "
            "This is a safety gate — pass false only to dry-run the call."
        ),
    )


class ReconcileInput(BaseModel):
    """Input model for ingestion_reconcile_redis."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    filename: str = Field(
        ...,
        min_length=1,
        description=(
            "Document filename to recover Redis state for. "
            "Use after Redis data loss to re-seed active-root keys from Qdrant. "
            "Automatically converted to basename."
        ),
    )

    @field_validator("filename", mode="after")
    @classmethod
    def _extract_basename(cls, v: str) -> str:
        return _resolve_filename(v)



class ConfigureInput(BaseModel):
    """Input model for ingestion_configure."""

    model_config = ConfigDict(extra="forbid")

    qdrant_url: Optional[str] = Field(
        default=None,
        description="Qdrant base URL (e.g. 'http://localhost:6333').",
    )
    redis_host: Optional[str] = Field(
        default=None,
        description="Redis hostname or IP (default: 'localhost').",
    )
    redis_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Redis port (default: 6379).",
    )
    collection_base_name: Optional[str] = Field(
        default=None,
        description=(
            "Base name for the Qdrant collection. The embedding model ID is "
            "appended automatically (e.g. 'secure_rag' → 'secure_rag_baai-bge-small-en-v1.5')."
        ),
    )
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Embedding model name used for all vector encodings "
            "(e.g. 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2'). "
            "Changing this resets the ingestor and will target a different collection."
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="ingest_data",
    annotations={
        "title": "Ingest Document Pages into Qdrant",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_data(params: IngestInput, ctx: Context) -> str:
    """
    Ingest one or more document pages into the Qdrant vector store.

    Each page is fingerprinted with a Merkle root derived from its ordered
    chunk content hashes.  Subsequent ingests of the same page are
    idempotent if content is unchanged; changed content creates a new versioned
    snapshot (previous snapshot soft-deleted, not removed).

    Accepts either:
      · `file_path`  — path to a documents.json from document_parser_mcp
      · `documents`  — inline array of Document dicts

    Args:
        params (IngestInput): Validated input. See field descriptions.

    Returns:
        str: JSON summary of the ingestion run.

        Success:
        {
            "ingested": int,          # pages actually written (new/changed)
            "skipped": int,           # pages unchanged (idempotent no-op)
            "errors": [               # per-page errors, if any
                {"page_index": int, "filename": str, "error": str}
            ],
            "collection": str         # target Qdrant collection name
        }
    """
    ingestor = _get_ingestor(ctx)
    logger.info("Loading document data...")

    # --- Resolve documents ------------------------------------------------
    try:
        if params.file_path is not None:
            documents = _load_documents_from_file(params.file_path)
        else:
            documents = _load_documents_from_inline(params.documents)  # type: ignore[arg-type]
    except (FileNotFoundError, ValueError) as exc:
        return _error(
            type(exc).__name__,
            str(exc),
            "Ensure the parser has run successfully and the output path is correct.",
        )

    total = len(documents)
    logger.info(f"Loaded {total} page(s). Starting ingestion...")

    ingested = 0
    skipped  = 0
    errors: List[dict] = []

    for i, doc in enumerate(documents):
        progress = 0.05 + 0.90 * (i / max(total, 1))
        logger.info(f"Ingesting page {i + 1}/{total}: {doc.metadata.filename} p.{doc.metadata.page_index}")
        # Detect no-chunk pages before hitting the ingestor
        if not doc.chunks:
            skipped += 1
            logger.info(
                f"Skipped empty page: {doc.metadata.filename} p.{doc.metadata.page_index}"
            )
            continue

        try:
            # process_document is idempotent — returns True if actually written, False if skipped
            if await ingestor.process_document(doc):
                ingested += 1
            else:
                skipped += 1
        except IngestorError as exc:
            errors.append({
                "page_index": doc.metadata.page_index,
                "filename": doc.metadata.filename,
                "error": str(exc),
            })
            logger.error(
                f"Failed to ingest {doc.metadata.filename} p.{doc.metadata.page_index}: {exc}"
            )
        except Exception as exc:
            errors.append({
                "page_index": doc.metadata.page_index,
                "filename": doc.metadata.filename,
                "error": f"{type(exc).__name__}: {exc}",
            })
            logger.error(f"Unexpected error on page {i}: {exc}")

    logger.info("Ingestion complete.")

    result = {
        "ingested": ingested,
        "skipped":  skipped,
        "total_pages": total,
        "errors": errors,
        "collection": ingestor.collection_name,
    }
    if errors:
        result["warning"] = (
            f"{len(errors)} page(s) failed. Check the `errors` list for details."
        )
    return json.dumps(result, indent=2)


@mcp.tool(
    name="ingest_audit",
    annotations={
        "title": "Verify Merkle Integrity for a Document Page",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_audit(
    params: VerifyIntegrityInput, ctx: Context
) -> str:
    """
    Mathematically verify that the stored Qdrant vectors for a page match
    the trusted Merkle root in Redis.

    Re-derives the Merkle root by scrolling all leaf chunks from Qdrant,
    sorting by chunk_index, and rebuilding the tree.  Compares the result
    against the root stored in Redis at ingestion time.

    Args:
        params (VerifyIntegrityInput): filename + page_index to audit.

    Returns:
        str: JSON result.

        {
            "filename": str,
            "page_index": int,
            "integrity": "PASSED" | "FAILED",
            "detail": str          # human-readable explanation
        }
    """
    ingestor = _get_ingestor(ctx)
    logger.info("Running integrity audit...")

    try:
        passed = await ingestor.verify_integrity(params.filename, params.page_index)
    except Exception as exc:
        return _error(
            type(exc).__name__,
            str(exc),
            "Check Qdrant and Redis connectivity.",
        )

    logger.info("Done.")
    return json.dumps({
        "filename": params.filename,
        "page_index": params.page_index,
        "integrity": "PASSED" if passed else "FAILED",
        "detail": (
            "Merkle root reconstructed from Qdrant matches the trusted root in Redis."
            if passed else
            "Merkle root mismatch — data in Qdrant may have been tampered with "
            "or Redis state is out of sync. Run `ingest_sync` to recover."
        ),
    }, indent=2)


@mcp.tool(
    name="ingest_search",
    annotations={
        "title": "Semantic RAG Search with Optional Version Pinning",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_search(params: SearchInput, ctx: Context) -> str:
    """
    Semantic search over ingested document chunks.

    Omit `version_root` to query the current active version of all documents.
    Pass a specific `version_root` (from `ingest_history`) to perform
    a point-in-time search against any historical snapshot.

    Optionally filter results by document `category`.

    Args:
        params (SearchInput): query, optional category/version_root/limit.

    Returns:
        str: JSON array of matching chunks.

        [
            {
                "score": float,
                "content": str,                    # chunk markdown text
                "metadata": {
                    "filename": str,
                    "page_index": int,
                    "page_count": int,
                    "category": str
                },
                "chunk_hash": str,                 # SHA-256 content fingerprint
                "version_root": str,               # Merkle root of the snapshot
                "timestamp": str                   # ISO-8601 ingestion time (UTC)
            }
        ]
    """
    ingestor = _get_ingestor(ctx)
    logger.info("Embedding query...")

    try:
        hits = await ingestor.secure_search(
            query=params.query,
            category=params.category,
            version_root=params.version_root,
            limit=params.limit,
        )
    except IngestorError as exc:
        return _error(
            "IngestorError",
            str(exc),
            "Verify Qdrant is reachable and the collection has been set up.",
        )
    except Exception as exc:
        return _error(type(exc).__name__, str(exc))

    logger.info("Done.")

    results = [
        {
            "score": round(float(h.score), 6),
            "content": h.payload.get("content", ""),
            "metadata": h.payload.get("metadata", {}),
            "chunk_hash": h.payload.get("chunk_hash", ""),
            "version_root": h.payload.get("version_root", ""),
            "timestamp": h.payload.get("timestamp", ""),
        }
        for h in hits
    ]
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool(
    name="ingest_history",
    annotations={
        "title": "Get Document Version Audit Trail",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_history(params: HistoryInput, ctx: Context) -> str:
    """
    Retrieve the full version history (audit trail) for a document.

    Returns all Merkle root anchors ever stored for the document, sorted
    newest-first.  Use the `version_root` values with `ingest_search` to
    query any historical snapshot.

    Args:
        params (HistoryInput): filename to look up.

    Returns:
        str: JSON array of version records.

        [
            {
                "version_root": str,    # Merkle root hash (use for point-in-time search)
                "timestamp": str,       # ISO-8601 ingestion time (UTC), newest first
                "page_index": int,
                "chunk_count": int
            }
        ]
    """
    ingestor = _get_ingestor(ctx)
    logger.info("Fetching audit trail...")

    try:
        history = await ingestor.get_document_history(params.filename)
    except Exception as exc:
        return _error(
            type(exc).__name__,
            str(exc),
            "Verify the filename matches exactly what was used during ingestion.",
        )

    logger.info("Done.")

    if not history:
        return json.dumps({
            "filename": params.filename,
            "versions": [],
            "message": "No history found. The document may not have been ingested yet.",
        }, indent=2)

    return json.dumps({
        "filename": params.filename,
        "version_count": len(history),
        "versions": history,
    }, indent=2)


@mcp.tool(
    name="ingest_purge",
    annotations={
        "title": "Hard-Delete All Data for a Document",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def ingest_purge(params: PurgeInput, ctx: Context) -> str:
    """
    Permanently delete ALL Qdrant vectors and Redis state for a document
    (every page, every version, every root anchor).

    This operation is IRREVERSIBLE.  The `confirm` field must be set to `true`
    or the call will be aborted with an informational message.

    Args:
        params (PurgeInput): filename + confirm=true.

    Returns:
        str: JSON confirmation or abort message.

        {
            "purged": bool,
            "filename": str,
            "message": str
        }
    """
    if not params.confirm:
        return json.dumps({
            "purged": False,
            "filename": params.filename,
            "message": (
                "Purge aborted — `confirm` was false. "
                "Set `confirm: true` to proceed with permanent deletion."
            ),
        }, indent=2)

    ingestor = _get_ingestor(ctx)
    logger.info(f"Purging {params.filename}...")

    try:
        await ingestor.purge_document(params.filename)
    except IngestorError as exc:
        return _error(
            "IngestorError",
            str(exc),
            "Qdrant data may have been partially deleted. Check logs and retry.",
        )
    except Exception as exc:
        return _error(type(exc).__name__, str(exc))

    logger.info("Done.")
    return json.dumps({
        "purged": True,
        "filename": params.filename,
        "message": "All Qdrant vectors and Redis state deleted for this document.",
    }, indent=2)


@mcp.tool(
    name="ingest_sync",
    annotations={
        "title": "Re-seed Redis State from Qdrant",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_sync(params: ReconcileInput, ctx: Context) -> str:
    """
    Recovery utility: re-seeds Redis active-root keys from Qdrant root anchors.

    Use this when Redis data has been lost (eviction, flush, restart without
    persistence) and integrity checks are failing.  For each page in the
    document's Qdrant history, the most recent root anchor (by timestamp) is
    written back to Redis as the active root.

    Args:
        params (ReconcileInput): filename to reconcile.

    Returns:
        str: JSON summary.

        {
            "filename": str,
            "pages_reconciled": int,
            "message": str
        }
    """
    ingestor = _get_ingestor(ctx)
    logger.info(f"Reconciling Redis state for {params.filename}...")

    try:
        count = await ingestor.reconcile_redis_from_qdrant(params.filename)
    except Exception as exc:
        return _error(
            type(exc).__name__,
            str(exc),
            "Verify both Qdrant and Redis are reachable.",
        )

    logger.info("Done.")
    return json.dumps({
        "filename": params.filename,
        "pages_reconciled": count,
        "message": (
            f"Reconciled {count} page(s). Integrity checks should now pass."
            if count > 0 else
            "No history found in Qdrant for this document — nothing to reconcile."
        ),
    }, indent=2)


@mcp.tool(
    name="ingest_configure",
    annotations={
        "title": "Update Ingestor Connection Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_configure(params: ConfigureInput, ctx: Context) -> str:
    """
    Update the Qdrant / Redis connection settings and re-initialise the ingestor.

    Only fields you supply are changed; unspecified fields keep their current
    values.  The ingestor is torn down and re-created, which verifies new
    connectivity.  Any in-flight ingestion calls should complete before calling
    this tool.

    Args:
        params (ConfigureInput): Fields to update — see field descriptions.

    Returns:
        str: JSON confirmation with the applied settings.
    """
    global _ingestor

    updates = {k: v for k, v in params.model_dump().items() if v is not None}
    if not updates:
        current = {
            "qdrant_url": QDRANT_URL,
            "redis_host": REDIS_HOST,
            "redis_port": REDIS_PORT,
            "collection_base_name": COLLECTION_BASE,
            "model_name": EMBED_MODEL_NAME,
        }
        return json.dumps({
            "message": "No changes — all fields were None.",
            "current_settings": current,
        }, indent=2)

    # Merge updates onto current values
    current_ingestor = _get_ingestor(ctx)
    new_cfg = {
        "qdrant_url":          QDRANT_URL,
        "redis_host":          current_ingestor.redis.connection_pool.connection_kwargs.get("host", REDIS_HOST),
        "redis_port":          current_ingestor.redis.connection_pool.connection_kwargs.get("port", REDIS_PORT),
        "collection_base_name": COLLECTION_BASE,
        "model_name":          current_ingestor.model_name,
    }
    new_cfg.update(updates)

    try:
        # Tear down existing connections
        await current_ingestor.qdrant.close()
        await current_ingestor.redis.aclose()
    except Exception:
        pass

    try:
        new_ingestor = AsyncMerkleQdrantIngestor(
            qdrant_url=new_cfg["qdrant_url"],
            redis_host=new_cfg["redis_host"],
            redis_port=new_cfg["redis_port"],
            collection_base_name=new_cfg["collection_base_name"],
            model_name=new_cfg["model_name"],
        )
        await new_ingestor.setup()
    except Exception as exc:
        return _error(
            type(exc).__name__,
            f"Re-initialisation failed: {exc}",
            "Revert to previous settings or check service connectivity.",
        )

    _ingestor = new_ingestor
    # Update lifespan state so subsequent tool calls get the new instance
    ctx.request_context.lifespan_state["ingestor"] = new_ingestor

    return json.dumps({
        "message": "Ingestor re-initialised with new settings.",
        "applied_updates": updates,
        "active_collection": new_ingestor.collection_name,
    }, indent=2)


@mcp.tool(
    name="ingest_status",
    annotations={
        "title": "Get Ingestor Status and Active Settings",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def ingest_status(ctx: Context) -> str:
    """
    Return the active ingestor configuration and connectivity state.

    Pings both Qdrant and Redis to confirm reachability, and returns the
    active collection name, embedding model, and engine type.

    Returns:
        str: JSON status object.

        {
            "collection": str,
            "model_name": str,
            "embedding_engine": str,           # "fastembed" | "st"
            "qdrant_reachable": bool,
            "redis_reachable": bool,
            "qdrant_url": str,
            "redis_host": str,
            "redis_port": int
        }
    """
    ingestor = _get_ingestor(ctx)

    qdrant_ok = False
    redis_ok  = False

    try:
        await ingestor.qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    try:
        await ingestor.redis.ping()
        redis_ok = True
    except Exception:
        pass

    conn_kwargs = ingestor.redis.connection_pool.connection_kwargs

    return json.dumps({
        "collection":        ingestor.collection_name,
        "model_name":        ingestor.model_name,
        "embedding_engine":  getattr(ingestor, "engine_type", "unknown"),
        "qdrant_reachable":  qdrant_ok,
        "redis_reachable":   redis_ok,
        "qdrant_url":        QDRANT_URL,
        "redis_host":        conn_kwargs.get("host", REDIS_HOST),
        "redis_port":        conn_kwargs.get("port", REDIS_PORT),
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mcp.run()   # stdio transport — Claude Desktop communicates over stdin/stdout