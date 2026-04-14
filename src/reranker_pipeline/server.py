"""
reranker_mcp — FastMCP server wrapping HybridReranker.
=======================================================

Transport:  stdio  (registered in claude_desktop_config.json)

Tools:
    parser_*   — document_parser_mcp
    (none)     — ingestion_pipeline_mcp
    rerank_*   — this server

Architecture:
    Lifespan bootstraps:
        1. AsyncMerkleQdrantIngestor (shared with ingestion_pipeline_mcp, but
           this server owns its own instance — no cross-process state)
        2. HybridReranker wrapping the ingestor

    All cross-encoder inference runs inside asyncio.to_thread (already handled
    by HybridReranker._cross_encoder_stage) so the FastMCP event loop is never
    blocked.

    Configuration is read from environment variables at startup; a
    rerank_configure tool allows hot-swapping alpha / CE model at runtime
    (rebuilds HybridReranker but reuses the existing ingestor connection).

Import layout:
    Place server.py alongside reranker_pipeline.py, ingestion_pipeline.py,
    and schemas.py.  Adjust the three import lines marked ← if your layout
    differs.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field

# Ensure the parent directory (src root) is in sys.path so that absolute imports work from MCP.
import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ← Adjust these lines if your package layout differs
from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor, IngestorError
from reranker_pipeline.reranker_pipeline import HybridReranker, RankedResult, format_citation
from shared.env_loader import load_env

# Load environment variables (api keys, connection URLs) if present
load_env()


# ---------------------------------------------------------------------------
# Constants / env-var config
# ---------------------------------------------------------------------------

SERVER_NAME = "reranker_mcp"

QDRANT_URL          = os.getenv("QDRANT_URL",           "http://localhost:6333")
REDIS_HOST          = os.getenv("REDIS_HOST",           "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT",       "6379"))
COLLECTION_BASE     = os.getenv("COLLECTION_BASE_NAME", "secure_rag")
EMBED_MODEL_NAME    = os.getenv("EMBED_MODEL_NAME",     "BAAI/bge-small-en-v1.5")

CE_MODEL_NAME       = os.getenv("CE_MODEL_NAME",        "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ALPHA        = float(os.getenv("RERANK_ALPHA",   "0.7"))
CACHE_SIZE          = int(os.getenv("RERANK_CACHE_SIZE","4096"))

# Safety caps — prevent accidentally huge requests from stalling the CE
MAX_RETRIEVAL_TOP_K = 200
MAX_RERANK_TOP_N    = 50

logger = logging.getLogger(SERVER_NAME)


# ---------------------------------------------------------------------------
# Lifespan — boot ingestor + reranker once, share across all tool calls
# ---------------------------------------------------------------------------

_ingestor: Optional[AsyncMerkleQdrantIngestor] = None
_reranker: Optional[HybridReranker] = None


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """
    Start-up:
        1. Connect ingestor → verifies Qdrant + Redis, creates collection/indexes.
        2. Load HybridReranker → downloads cross-encoder weights (~90 MB, cached).
    Tear-down:
        Closes Qdrant + Redis connections.
    """
    global _ingestor, _reranker

    logger.info(
        "Connecting — Qdrant=%s  Redis=%s:%d  embed=%s  CE=%s  α=%.2f",
        QDRANT_URL, REDIS_HOST, REDIS_PORT, EMBED_MODEL_NAME, CE_MODEL_NAME, RERANK_ALPHA,
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
        logger.error("Ingestor setup failed (tools will surface this): %s", exc)

    try:
        _reranker = HybridReranker(
            ingestor=_ingestor,
            cross_encoder_model_name=CE_MODEL_NAME,
            alpha=RERANK_ALPHA,
            cache_size=CACHE_SIZE,
        )
        logger.info("HybridReranker ready.")
    except Exception as exc:
        logger.error("Reranker init failed (tools will surface this): %s", exc)

    yield {"ingestor": _ingestor, "reranker": _reranker}

    # Cleanup
    if _ingestor:
        try:
            await _ingestor.qdrant.close()
            await _ingestor.redis.aclose()
        except Exception:
            pass


mcp = FastMCP(SERVER_NAME, lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_reranker(ctx: Context) -> HybridReranker:
    """Retrieve the lifespan-managed reranker; raises descriptively if not ready."""
    global _reranker
    if _reranker is None:
        raise RuntimeError(
            "HybridReranker is not initialised. "
            "Verify that sentence-transformers is installed and that Qdrant/Redis "
            "are reachable. Check server startup logs for details."
        )
    return _reranker


def _get_ingestor(ctx: Context) -> AsyncMerkleQdrantIngestor:
    global _ingestor
    if _ingestor is None:
        raise IngestorError(
            "Ingestor is not initialised. Check Qdrant/Redis connectivity."
        )
    return _ingestor


def _error(kind: str, message: str, suggestion: str = "") -> str:
    """Consistent JSON error envelope across all tools."""
    payload: Dict[str, Any] = {"error": kind, "message": message}
    if suggestion:
        payload["suggestion"] = suggestion
    return json.dumps(payload, indent=2)


def _serialise_result(result: RankedResult) -> dict:
    """
    Convert a RankedResult (+ CitationEnvelope dataclass) to a JSON-safe dict.

    Scores are rounded to 6 decimal places to keep payload size manageable
    while preserving enough precision for ranking comparisons.
    """
    return {
        "content":           result.content,
        "final_score":       round(result.final_score, 6),
        "ce_score":          round(result.ce_score, 6),
        "rrf_score":         round(result.rrf_score, 6),
        "retrieval_sources": result.retrieval_sources,
        "citation": {
            "filename":     result.citation.filename,
            "page_index":   result.citation.page_index,
            "page_count":   result.citation.page_count,
            "chunk_index":  result.citation.chunk_index,
            "chunk_hash":   result.citation.chunk_hash,
            "version_root": result.citation.version_root,
            "category":     result.citation.category,
            "bbox":         result.citation.bbox,
            "timestamp":    result.citation.timestamp,
        },
    }


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class RerankSearchInput(BaseModel):
    """Input model for rerank_search."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "Natural-language query used to retrieve and rank document chunks. "
            "Example: 'balanced batching'."
        ),
    )
    retrieval_top_k: int = Field(
        default=50,
        ge=1,
        le=MAX_RETRIEVAL_TOP_K,
        description=(
            f"Number of candidate chunks fetched per retrieval leg (vector + BM25) "
            f"before cross-encoder reranking. Higher = better recall, slower scoring. "
            f"Recommended: 30–100. Maximum: {MAX_RETRIEVAL_TOP_K}."
        ),
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=MAX_RERANK_TOP_N,
        description=(
            f"Final number of results returned after reranking, sorted by final_score "
            f"descending. Recommended: 3–10 for LLM context windows. "
            f"Maximum: {MAX_RERANK_TOP_N}."
        ),
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional document category filter applied at the Qdrant retrieval stage. "
            "Must match a `category` value set during ingestion "
            "(e.g. 'research', 'medical', 'finance'). Omit to search all categories."
        ),
    )
    corpus_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional Logical Knowledge Base ID to strictly limit the search space. "
            "When set, retrieval is scoped entirely to documents within this Corpus."
        ),
    )
    version_root: Optional[str] = Field(
        default=None,
        description=(
            "Merkle root hash to pin the search to a specific historical document snapshot. "
            "Omit to query the current active version. "
            "Obtain valid roots from history."
        ),
    )
    collection_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional Qdrant collection override to run the search against. "
            "If omitted, it inherently uses the default pipeline collection "
            "(e.g. 'secure_rag_baai-bge-small-en-v1.5')."
        ),
    )
    include_citations_text: bool = Field(
        default=False,
        description=(
            "If true, include a pre-formatted citation string for each result "
            "(suitable for direct insertion into LLM context). "
            "Example: '[1] report.pdf · p.0 · chunk 2 · score 0.923'."
        ),
    )

    # Note: the rerank_top_n <= retrieval_top_k constraint is enforced inside
    # the rerank_search tool rather than here, to avoid MCP schema serialisation
    # issues with cross-field Pydantic validators.


class RerankConfigureInput(BaseModel):
    """Input model for rerank_configure."""

    model_config = ConfigDict(extra="forbid")

    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for the cross-encoder score in the final blend: "
            "final = α·CE_norm + (1-α)·RRF_norm. "
            "0.0 = pure RRF, 1.0 = pure cross-encoder. Default: 0.7."
        ),
    )
    cross_encoder_model_name: Optional[str] = Field(
        default=None,
        description=(
            "HuggingFace cross-encoder model to load. "
            "Changing this forces a model reload (~60–300 MB download on first use). "
            "Examples: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast), "
            "'cross-encoder/ms-marco-MiniLM-L-12-v2' (more accurate)."
        ),
    )
    cache_size: Optional[int] = Field(
        default=None,
        ge=64,
        le=65536,
        description=(
            "Maximum number of (query, chunk) score pairs held in the in-process "
            "LRU cache. Larger values reduce CE re-scoring on repeated queries. "
            "Default: 4096."
        ),
    )
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Override the Qdrant URL (e.g. 'http://localhost:6333').",
    )
    redis_host: Optional[str] = Field(
        default=None,
        description="Override the Redis hostname.",
    )
    redis_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Override the Redis port.",
    )
    embed_model_name: Optional[str] = Field(
        default=None,
        description=(
            "Override the dense embedding model used by the ingestor "
            "(e.g. 'BAAI/bge-small-en-v1.5'). Changing this targets a different "
            "Qdrant collection."
        ),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="rerank_search",
    annotations={
        "title": "Hybrid Rerank Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def rerank_search(params: RerankSearchInput, ctx: Context) -> str:
    """
    Execute a two-stage hybrid retrieval + reranking pipeline over ingested documents.

    Stage 1 — Broad recall via Reciprocal Rank Fusion (RRF):
        · Vector leg  : semantic search via the ingestor (Merkle version-aware)
        · Sparse leg  : BM25 re-rank over the vector candidates' raw content
        · RRF fusion  : score = Σ 1/(60 + rank_i) across both legs

    Stage 2 — Precision reranking via Cross-Encoder:
        · Scores every (query, chunk) pair with full-attention cross-encoder
        · Final score = α·CE_norm + (1-α)·RRF_norm  (default α=0.7)

    Results are sorted by final_score descending and include full citation
    provenance: filename, page_index, bbox, chunk_hash, version_root.

    Args:
        params (RerankSearchInput): Validated input — see field descriptions.

    Returns:
        str: JSON object with results array and pipeline diagnostics.

        {
            "query": str,
            "results": [
                {
                    "content": str,               // chunk markdown text
                    "final_score": float,         // primary ranking signal
                    "ce_score":    float,         // raw cross-encoder logit (normalised)
                    "rrf_score":   float,         // raw RRF score (normalised)
                    "retrieval_sources": [str],   // ["vector"] | ["sparse"] | ["vector","sparse"]
                    "citation": {
                        "filename":     str,
                        "page_index":   int,
                        "page_count":   int,
                        "chunk_index":  int,
                        "chunk_hash":   str,      // SHA-256 content fingerprint
                        "version_root": str,      // Merkle snapshot this chunk belongs to
                        "category":     str,
                        "bbox":         [int, int, int, int],  // PDF coords [x0,y0,x1,y1]
                        "timestamp":    str       // ISO-8601 ingestion time (UTC)
                    },
                    "citation_text": str          // only present when include_citations_text=true
                }
            ],
            "result_count": int,
            "pipeline": {
                "retrieval_top_k": int,
                "rerank_top_n": int,
                "alpha": float,
                "ce_model": str,
                "bm25_active": bool,
                "version_root": str | null,
                "category": str | null
            }
        }
    """
    reranker = _get_reranker(ctx)

    # Cross-field validation
    if params.rerank_top_n > params.retrieval_top_k:
        return _error(
            "ValidationError",
            f"`rerank_top_n` ({params.rerank_top_n}) cannot exceed `retrieval_top_k` ({params.retrieval_top_k}).",
            "Lower rerank_top_n or raise retrieval_top_k.",
        )

    logger.info("Fetching vector candidates...")

    try:
        results: List[RankedResult] = await reranker.rerank(
            query=params.query,
            retrieval_top_k=params.retrieval_top_k,
            rerank_top_n=params.rerank_top_n,
            category=params.category,
            corpus_id=params.corpus_id,
            version_root=params.version_root,
            collection_name=params.collection_name,
        )
    except IngestorError as exc:
        return _error(
            "IngestorError",
            str(exc),
            "Check that Qdrant is reachable and the collection has been seeded via ingest.",
        )
    except Exception as exc:
        logger.error(f"rerank_search unexpected error: {type(exc).__name__}: {exc}")
        return _error(
            type(exc).__name__,
            str(exc),
            "Check server logs for the full traceback.",
        )

    logger.info("Serialising results...")

    serialised: List[dict] = []
    for i, r in enumerate(results):
        item = _serialise_result(r)
        if params.include_citations_text:
            item["citation_text"] = format_citation(r, index=i + 1)
        serialised.append(item)

    stats = reranker.cache_stats()

    response = {
        "query":        params.query,
        "results":      serialised,
        "result_count": len(serialised),
        "pipeline": {
            "retrieval_top_k": params.retrieval_top_k,
            "rerank_top_n":    params.rerank_top_n,
            "alpha":           reranker.alpha,
            "ce_model":        stats["ce_model"],
            "bm25_active":     stats["bm25_available"],
            "corpus_id":       params.corpus_id,
            "version_root":    params.version_root,
            "category":        params.category,
        },
    }

    if not serialised:
        response["warning"] = (
            "No results returned. The collection may be empty, or no chunks match "
            "the query / category / version_root combination. "
            "Run ingest first, then retry."
        )

    logger.info("Done.")
    return json.dumps(response, indent=2, ensure_ascii=False)


@mcp.tool(
    name="rerank_configure",
    annotations={
        "title": "Update Reranker Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def rerank_configure(
    ctx: Context,
    alpha: Optional[float] = None,
    cross_encoder_model_name: Optional[str] = None,
    cache_size: Optional[int] = None,
    qdrant_url: Optional[str] = None,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    embed_model_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Hot-swap reranker settings at runtime.

    Changing `alpha` or `cache_size` rebuilds HybridReranker in-place
    (no model reload — fast).

    Changing `cross_encoder_model_name` triggers a full CE model reload
    (~60–300 MB download on first use, then cached by HuggingFace).

    Changing `qdrant_url`, `redis_host`, `redis_port`, or `embed_model_name`
    also tears down and re-initialises the underlying ingestor (re-verifies
    connectivity and may target a different Qdrant collection).

    Only fields you supply are changed; unspecified fields retain their
    current values.

    Returns:
        str: JSON confirmation with applied settings.
    """
    global _ingestor, _reranker

    updates = {}
    # 1. Collect named args
    if alpha is not None: updates["alpha"] = alpha
    if cross_encoder_model_name is not None: updates["cross_encoder_model_name"] = cross_encoder_model_name
    if cache_size is not None: updates["cache_size"] = cache_size
    if qdrant_url is not None: updates["qdrant_url"] = qdrant_url
    if redis_host is not None: updates["redis_host"] = redis_host
    if redis_port is not None: updates["redis_port"] = redis_port
    if embed_model_name is not None: updates["embed_model_name"] = embed_model_name

    # 2. Collect from kwargs (handles potential 'params' nesting or odd model behavior)
    for k, v in kwargs.items():
        if v is not None and k not in updates:
            updates[k] = v
    
    # 3. Special case: if 'params' was passed as a dict in kwargs
    if "params" in kwargs and isinstance(kwargs["params"], dict):
        for k, v in kwargs["params"].items():
            if v is not None and k not in updates:
                updates[k] = v

    logger.info("[TOOL] rerank_configure consolidated updates: %s", updates)
    
    if not updates:
        rr = _get_reranker(ctx)
        stats = rr.cache_stats()
        return json.dumps({
            "message": "No changes — all fields were None.",
            "current_settings": {
                "alpha":      stats["alpha"],
                "ce_model":   stats["ce_model"],
                "cache_size": stats["ce_cache_size"],
                "rrf_k":      stats["rrf_k"],
                "bm25_active": stats["bm25_available"],
            },
        }, indent=2)

    # --- Determine whether an ingestor rebuild is needed ---
    ingestor_fields = {"qdrant_url", "redis_host", "redis_port", "embed_model_name"}
    needs_ingestor_rebuild = bool(ingestor_fields & set(updates.keys()))

    # Merge updates onto current values
    current_ingestor = _get_ingestor(ctx)
    current_reranker = _get_reranker(ctx)
    current_stats    = current_reranker.cache_stats()

    new_cfg = {
        "qdrant_url":        QDRANT_URL,
        "redis_host":        REDIS_HOST,
        "redis_port":        REDIS_PORT,
        "embed_model_name":  current_ingestor.model_name,
        "alpha":             current_reranker.alpha,
        "ce_model":          current_stats["ce_model"],
        "cache_size":        current_stats["ce_cache_size"],
    }
    new_cfg.update({
        "alpha":            updates.get("alpha",                   new_cfg["alpha"]),
        "ce_model":         updates.get("cross_encoder_model_name", new_cfg["ce_model"]),
        "cache_size":       updates.get("cache_size",              new_cfg["cache_size"]),
        "qdrant_url":       updates.get("qdrant_url",              new_cfg["qdrant_url"]),
        "redis_host":       updates.get("redis_host",              new_cfg["redis_host"]),
        "redis_port":       updates.get("redis_port",              new_cfg["redis_port"]),
        "embed_model_name": updates.get("embed_model_name",        new_cfg["embed_model_name"]),
    })

    # --- Rebuild ingestor if connection params changed ---
    new_ingestor = current_ingestor
    if needs_ingestor_rebuild:
        try:
            await current_ingestor.qdrant.close()
            await current_ingestor.redis.aclose()
        except Exception:
            pass
        try:
            new_ingestor = AsyncMerkleQdrantIngestor(
                qdrant_url=new_cfg["qdrant_url"],
                redis_host=new_cfg["redis_host"],
                redis_port=new_cfg["redis_port"],
                collection_base_name=COLLECTION_BASE,
                model_name=new_cfg["embed_model_name"],
            )
            await new_ingestor.setup()
        except Exception as exc:
            return _error(
                type(exc).__name__,
                f"Ingestor re-initialisation failed: {exc}",
                "Revert to previous connection settings.",
            )
        _ingestor = new_ingestor

    # --- Rebuild reranker ---
    try:
        new_reranker = HybridReranker(
            ingestor=new_ingestor,
            cross_encoder_model_name=new_cfg["ce_model"],
            alpha=new_cfg["alpha"],
            cache_size=new_cfg["cache_size"],
        )
    except Exception as exc:
        return _error(
            type(exc).__name__,
            f"HybridReranker rebuild failed: {exc}",
            "Check the CE model name and that sentence-transformers is installed.",
        )

    _reranker = new_reranker

    return json.dumps({
        "message": "Reranker updated successfully.",
        "applied_updates": updates,
        "active_settings": {
            "alpha":             new_reranker.alpha,
            "ce_model":          new_cfg["ce_model"],
            "cache_size":        new_cfg["cache_size"],
            "qdrant_url":        new_cfg["qdrant_url"],
            "redis_host":        new_cfg["redis_host"],
            "redis_port":        new_cfg["redis_port"],
            "embed_model_name":  new_cfg["embed_model_name"],
            "collection":        new_ingestor.collection_name,
        },
    }, indent=2)


@mcp.tool(
    name="rerank_status",
    annotations={
        "title": "Reranker Status and Cache Diagnostics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def rerank_status(ctx: Context) -> str:
    """
    Return active configuration, scoring parameters, and connectivity status.

    Pings Qdrant and Redis to confirm reachability.  Reports CE model,
    alpha blend weight, RRF k constant, BM25 availability, and current
    in-process cache utilisation.

    Returns:
        str: JSON status object.

        {
            "ce_model":         str,      // active cross-encoder model name
            "alpha":            float,    // CE weight in final blend
            "rrf_k":            int,      // RRF smoothing constant (fixed: 60)
            "bm25_active":      bool,     // true if rank_bm25 is installed
            "ce_cache_entries": int,      // current LRU cache utilisation
            "collection":       str,      // active Qdrant collection
            "embed_model":      str,      // dense embedding model
            "qdrant_reachable": bool,
            "redis_reachable":  bool,
            "qdrant_url":       str,
            "redis_host":       str,
            "redis_port":       int
        }
    """
    reranker = _get_reranker(ctx)
    ingestor = _get_ingestor(ctx)
    stats    = reranker.cache_stats()

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

    conn = ingestor.redis.connection_pool.connection_kwargs

    return json.dumps({
        "ce_model":          stats["ce_model"],
        "alpha":             stats["alpha"],
        "rrf_k":             stats["rrf_k"],
        "bm25_active":       stats["bm25_available"],
        "ce_cache_entries":  stats["ce_cache_size"],
        "collection":        ingestor.collection_name,
        "embed_model":       ingestor.model_name,
        "qdrant_reachable":  qdrant_ok,
        "redis_reachable":   redis_ok,
        "qdrant_url":        QDRANT_URL,
        "redis_host":        conn.get("host", REDIS_HOST),
        "redis_port":        conn.get("port", REDIS_PORT),
    }, indent=2)


@mcp.tool(
    name="rerank_cache_clear",
    annotations={
        "title": "Clear the Cross-Encoder Score Cache",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def rerank_cache_clear(ctx: Context) -> str:
    """
    Evict all entries from the in-process cross-encoder score cache.

    Use this after changing the CE model (via rerank_configure) to prevent
    stale scores from the previous model bleeding into new results.

    Returns:
        str: JSON confirmation with the number of entries cleared.
    """
    reranker = _get_reranker(ctx)
    old_size = reranker._cache.size
    reranker._cache._cache.clear()
    reranker._cache._access_order.clear()
    return json.dumps({
        "cleared": old_size,
        "message": f"Evicted {old_size} cached (query, chunk) score pair(s).",
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