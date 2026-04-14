"""
HybridReranker — Citation-Aware Hybrid Reranking Pipeline
==========================================================

Architecture (two-stage):

  Stage 1 — Broad recall via Reciprocal Rank Fusion (RRF)
    · Vector leg  : AsyncMerkleQdrantIngestor.secure_search()
                    (already handles version pinning + is_active filtering)
    · Sparse leg  : rank-1 BM25 over the vector candidates' raw content
                    (no external index — payloads carry the text already)
    · Fusion      : RRF with k=60 (industry standard) over both ranked lists
                    Score = Σ 1/(k + rank_i)  — position-based, score-agnostic

  Stage 2 — Precision reranking via Cross-Encoder
    · Model       : cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, high quality)
                    Scores each (query, chunk_text) pair with full attention
    · Fusion      : weighted blend  final = α·CE_norm + (1-α)·RRF_norm
                    α defaults to 0.7 (CE-dominant, tunable)

  Output — RankedResult with CitationEnvelope
    Every result carries the full provenance chain needed to produce a
    verifiable in-text citation:
      · filename, page_index, chunk_index  →  human-readable location
      · bbox                               →  PDF coordinate anchor
      · version_root + chunk_hash          →  Merkle integrity link
      · ce_score, rrf_score, final_score   →  scoring transparency

Design decisions:
  · Async-first  : cross-encoder inference runs in asyncio.to_thread so it
                   never blocks the event loop.
  · Score cache  : SHA-256 keyed in-process LRU cache for (query, chunk) pairs
                   avoids re-scoring identical chunks across calls in the same
                   process lifetime.
  · Batch CE     : all (query, chunk) pairs are scored in a single model.predict
                   call — one forward pass per batch is far cheaper than N calls.
  · Normalisation: both CE and RRF scores are min-max normalised before blending
                   so alpha is meaningful regardless of raw score ranges.
  · BM25 in-proc : rank_bm25 is lightweight and runs on the already-fetched
                   payload text — avoids a second round-trip to any index.
  · Pluggable model: swap cross_encoder_model_name at construction time for a
                   larger model (e.g. ms-marco-MiniLM-L-12-v2) when accuracy
                   matters more than latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logging.getLogger("HybridReranker").warning(
        "rank_bm25 not installed — sparse leg disabled. "
        "Install with: pip install rank-bm25"
    )

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

from ingestion_pipeline.ingestion_pipeline import AsyncMerkleQdrantIngestor, IngestorError

logger = logging.getLogger("HybridReranker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class CitationEnvelope:
    """
    Full provenance record for a single retrieved chunk.

    Fields are chosen to support every common citation rendering pattern:
      · Inline text citation  → filename + page_index
      · PDF deep-link         → filename + page_index + bbox
      · Audit / integrity     → version_root + chunk_hash + blob_cid
      · Sequential context    → chunk_index (position in original document)
      · Multitenancy          → corpus_id (Logical Knowledge Base container)
    """
    filename: str
    page_index: int         # 1-indexed
    page_count: int
    chunk_index: int
    chunk_hash: str
    version_root: str
    category: str
    bbox: List[int]         # [x0, y0, x1, y1] in PDF coordinate space
    timestamp: str          # ingestion timestamp of this snapshot
    blob_cid: str           # SHA-256 hash of the original source document
    corpus_id: Optional[str] = None # Knowledge Base container ID


@dataclass
class RankedResult:
    """
    A single reranked chunk, fully decorated for downstream consumption.

    Scores:
      rrf_score   — raw RRF fusion score (higher = more consensus across legs)
      ce_score    — raw cross-encoder logit (higher = more relevant to query)
      final_score — normalised weighted blend; use this for final ordering
    """
    content: str
    citation: CitationEnvelope
    rrf_score: float
    ce_score: float
    final_score: float
    retrieval_sources: List[str]    # which legs returned this chunk: ["vector", "sparse"]


# ---------------------------------------------------------------------------
# Internal candidate (pre-output)
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    point_id: str
    content: str
    payload: dict
    rrf_score: float = 0.0
    ce_score: float = 0.0
    final_score: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Score cache
# ---------------------------------------------------------------------------

class _ScoreCache:
    """
    In-process LRU cache for cross-encoder (query, chunk) scores.

    SHA-256 keys keep memory bounded and avoid storing raw text.
    maxsize=4096 covers ~4k unique query×chunk pairs (~few MB footprint).
    """
    def __init__(self, maxsize: int = 4096):
        self._cache: Dict[str, float] = {}
        self._access_order: List[str] = []
        self._maxsize = maxsize

    def _key(self, query: str, content: str) -> str:
        raw = f"{query}\x00{content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, content: str) -> Optional[float]:
        k = self._key(query, content)
        if k in self._cache:
            self._access_order.remove(k)
            self._access_order.append(k)
            return self._cache[k]
        return None

    def set(self, query: str, content: str, score: float) -> None:
        k = self._key(query, content)
        if k not in self._cache and len(self._cache) >= self._maxsize:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[k] = score
        if k not in self._access_order:
            self._access_order.append(k)

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Core reranker
# ---------------------------------------------------------------------------

class HybridReranker:
    """
    Citation-aware hybrid reranker integrating directly with
    AsyncMerkleQdrantIngestor.

    Usage:
        reranker = HybridReranker(ingestor)
        results = await reranker.rerank(
            query="right to a fair trial",
            retrieval_top_k=50,     # candidates fetched per leg
            rerank_top_n=5,         # final results returned
            category="legal",       # optional metadata filter
            version_root=None,      # None = active; str = point-in-time
        )
        for r in results:
            print(r.content)
            print(r.citation.filename, r.citation.page_index)
            print(f"score: {r.final_score:.4f}")
    """

    DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RRF_K = 60          # industry-standard smoothing constant
    CE_BATCH_SIZE = 64  # pairs per cross-encoder forward pass

    def __init__(
        self,
        ingestor: AsyncMerkleQdrantIngestor,
        cross_encoder_model_name: str = DEFAULT_CE_MODEL,
        alpha: float = 0.7,         # weight for CE score in final blend (0–1)
        cache_size: int = 4096,
    ):
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers is required for the cross-encoder. "
                "Install with: pip install sentence-transformers"
            )

        self.ingestor = ingestor
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._cache = _ScoreCache(maxsize=cache_size)
        self._ce_model_name = cross_encoder_model_name

        logger.info("Loading cross-encoder: %s", cross_encoder_model_name)
        self._ce: CrossEncoder = CrossEncoder(cross_encoder_model_name, max_length=512)
        logger.info("HybridReranker ready (alpha=%.2f, RRF k=%d)", self.alpha, self.RRF_K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        retrieval_top_k: int = 50,
        rerank_top_n: int = 5,
        category: Optional[str] = None,
        corpus_id: Optional[str] = None,
        version_root: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[RankedResult]:
        """
        Full hybrid reranking pipeline.

        Parameters
        ----------
        query           : The user's natural-language query.
        retrieval_top_k : Candidates to fetch per retrieval leg (vector + sparse).
                          Higher = better recall, slower cross-encoder stage.
                          Recommended: 30–100.
        rerank_top_n    : Final results to return after reranking.
                          Recommended: 3–10 for LLM context windows.
        category        : Optional Qdrant metadata filter (e.g. "legal").
        version_root    : Pin to a specific Merkle snapshot. None = active version.

        Returns
        -------
        List[RankedResult] sorted by final_score descending.
        """
        if not query.strip():
            return []

        # Stage 1a: Vector retrieval (via ingestor — handles all Merkle filters and Corpus isolation)
        vector_candidates = await self._vector_leg(
            query, retrieval_top_k, category, corpus_id, version_root, collection_name
        )

        if not vector_candidates:
            logger.warning("Vector leg returned no results for query: %r", query[:80])
            return []

        # Stage 1b: Sparse (BM25) retrieval over the same candidate pool
        sparse_candidates = self._sparse_leg(query, vector_candidates)

        # Stage 1c: RRF fusion
        fused = self._rrf_fuse(
            ranked_lists=[vector_candidates, sparse_candidates]
        )

        # Stage 2: Cross-encoder precision reranking
        fused_with_ce = await self._cross_encoder_stage(query, fused)

        # Stage 3: Final score blend + build output
        results = self._build_results(fused_with_ce, top_n=rerank_top_n)

        logger.info(
            "rerank() → %d results from %d fused candidates (query: %r...)",
            len(results), len(fused), query[:60],
        )
        return results

    # ------------------------------------------------------------------
    # Stage 1a: Vector leg
    # ------------------------------------------------------------------

    async def _vector_leg(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        corpus_id: Optional[str],
        version_root: Optional[str],
        collection_name: Optional[str],
    ) -> List[_Candidate]:
        """
        Delegates to ingestor.secure_search() which enforces:
          · is_active / version_root filtering (Merkle version pinning)
          · metadata.category filtering
          · is_merkle_leaf=True (never returns root anchor points)
        """
        try:
            points = await self.ingestor.secure_search(
                query=query,
                category=category,
                corpus_id=corpus_id,
                version_root=version_root,
                limit=top_k,
                collection_name=collection_name,
            )
        except IngestorError as exc:
            logger.error("Vector leg failed: %s", exc)
            return []

        candidates: List[_Candidate] = []
        for pt in points:
            p = pt.payload or {}
            cand = _Candidate(
                point_id=str(pt.id),
                content=p.get("content", ""),
                payload=p,
                retrieval_sources=["vector"],
            )
            candidates.append(cand)

        return candidates

    # ------------------------------------------------------------------
    # Stage 1b: Sparse BM25 leg
    # ------------------------------------------------------------------

    def _sparse_leg(
        self,
        query: str,
        candidates: List[_Candidate],
    ) -> List[_Candidate]:
        """
        Runs BM25Okapi over the candidate pool fetched by the vector leg.

        This avoids maintaining a separate BM25 index — the payloads already
        carry the raw text. The sparse leg re-ranks the same candidate set,
        so RRF fusion rewards chunks that score well on BOTH semantic and
        keyword axes.

        Returns the same candidates re-ordered by BM25 score.
        """
        if not HAS_BM25 or not candidates:
            return candidates

        tokenized_corpus = [c.content.lower().split() for c in candidates]
        tokenized_query = query.lower().split()

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
        except Exception as exc:
            logger.warning("BM25 scoring failed, skipping sparse leg: %s", exc)
            return candidates

        # Sort by BM25 score; clone candidates so we don't mutate the vector list
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        sparse_ranked: List[_Candidate] = []
        for score, cand in scored:
            if score > 0:   # skip zero-score candidates (no term overlap)
                clone = _Candidate(
                    point_id=cand.point_id,
                    content=cand.content,
                    payload=cand.payload,
                    retrieval_sources=["sparse"],
                )
                sparse_ranked.append(clone)

        return sparse_ranked

    # ------------------------------------------------------------------
    # Stage 1c: RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        ranked_lists: List[List[_Candidate]],
    ) -> List[_Candidate]:
        """
        Reciprocal Rank Fusion across N ranked lists.

        Formula:  RRF(d) = Σ_i  1 / (k + rank_i(d))
        where rank_i is 1-based and k=60 dampens the impact of very high
        ranks in any single list (prevents one outlier dominating).

        Deduplication: if a candidate appears in multiple lists, its scores
        are summed and its retrieval_sources list is merged.
        """
        rrf_scores: Dict[str, float] = {}
        merged_candidates: Dict[str, _Candidate] = {}

        for ranked_list in ranked_lists:
            for rank_idx, cand in enumerate(ranked_list, start=1):
                pid = cand.point_id
                score = 1.0 / (self.RRF_K + rank_idx)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + score

                if pid not in merged_candidates:
                    merged_candidates[pid] = _Candidate(
                        point_id=pid,
                        content=cand.content,
                        payload=cand.payload,
                        retrieval_sources=list(cand.retrieval_sources),
                    )
                else:
                    existing = merged_candidates[pid]
                    for src in cand.retrieval_sources:
                        if src not in existing.retrieval_sources:
                            existing.retrieval_sources.append(src)

        for pid, cand in merged_candidates.items():
            cand.rrf_score = rrf_scores[pid]

        fused = sorted(
            merged_candidates.values(),
            key=lambda c: c.rrf_score,
            reverse=True,
        )

        logger.debug("RRF fused %d unique candidates from %d lists", len(fused), len(ranked_lists))
        return fused

    # ------------------------------------------------------------------
    # Stage 2: Cross-encoder scoring
    # ------------------------------------------------------------------

    async def _cross_encoder_stage(
        self,
        query: str,
        candidates: List[_Candidate],
    ) -> List[_Candidate]:
        """
        Scores every (query, chunk_content) pair with the cross-encoder.

        Runs in asyncio.to_thread so the event loop stays unblocked.
        Results from the in-process cache are injected before inference
        so only unseen pairs pay the compute cost.

        All pairs in a batch are scored in a single model.predict() call
        for maximum GPU/CPU utilisation.
        """
        if not candidates:
            return candidates

        cache_hits = 0
        pairs_to_score: List[tuple] = []       # (idx, query, content)

        # Populate from cache first
        for idx, cand in enumerate(candidates):
            cached = self._cache.get(query, cand.content)
            if cached is not None:
                cand.ce_score = cached
                cache_hits += 1
            else:
                pairs_to_score.append((idx, query, cand.content))

        if cache_hits:
            logger.debug("CE cache: %d hits, %d misses", cache_hits, len(pairs_to_score))

        # Batch-score uncached pairs
        if pairs_to_score:
            all_pairs = [[q, c] for (_, q, c) in pairs_to_score]
            try:
                scores: np.ndarray = await asyncio.to_thread(
                    self._ce.predict, all_pairs, batch_size=self.CE_BATCH_SIZE
                )
            except Exception as exc:
                logger.error("Cross-encoder inference failed: %s", exc)
                # Degrade gracefully: use RRF score only
                for idx, _, _ in pairs_to_score:
                    candidates[idx].ce_score = candidates[idx].rrf_score
                return candidates

            for (idx, q, content), score in zip(pairs_to_score, scores):
                s = float(score)
                candidates[idx].ce_score = s
                self._cache.set(q, content, s)

        return candidates

    # ------------------------------------------------------------------
    # Stage 3: Final blend + RankedResult assembly
    # ------------------------------------------------------------------

    def _build_results(
        self,
        candidates: List[_Candidate],
        top_n: int,
    ) -> List[RankedResult]:
        """
        Normalises CE and RRF scores to [0, 1] then blends:
            final = alpha * CE_norm + (1 - alpha) * RRF_norm

        Assembles the full CitationEnvelope from Qdrant payload fields
        that were written by the ingestor.
        """
        if not candidates:
            return []

        # Min-max normalise CE scores
        ce_scores = np.array([c.ce_score for c in candidates], dtype=float)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_scores)

        # Min-max normalise RRF scores
        rrf_scores = np.array([c.rrf_score for c in candidates], dtype=float)
        rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
        if rrf_max > rrf_min:
            rrf_norm = (rrf_scores - rrf_min) / (rrf_max - rrf_min)
        else:
            rrf_norm = np.ones_like(rrf_scores)

        # Weighted blend
        final_scores = self.alpha * ce_norm + (1.0 - self.alpha) * rrf_norm

        # Sort by final score
        ranked_indices = np.argsort(final_scores)[::-1][:top_n]

        results: List[RankedResult] = []
        for i in ranked_indices:
            cand = candidates[i]
            p = cand.payload
            meta = p.get("metadata", {})

            # Standardise bbox extraction: prefer grounding.bbox, fallback to root or helper
            bbox = p.get("grounding", {}).get("bbox", [])
            if not bbox:
                bbox = p.get("bbox", [])
            if not bbox:
                bbox = _extract_bbox(p)

            citation = CitationEnvelope(
                filename=meta.get("filename", "unknown"),
                page_index=meta.get("page_index", 0),
                page_count=meta.get("page_count", 0),
                chunk_index=p.get("chunk_index", 0),
                chunk_hash=p.get("chunk_hash", ""),
                version_root=p.get("version_root", ""),
                category=meta.get("category", "general"),
                bbox=bbox,
                timestamp=p.get("timestamp", ""),
                blob_cid=meta.get("blob_cid", ""),
                corpus_id=meta.get("corpus_id"),
            )

            results.append(
                RankedResult(
                    content=cand.content,
                    citation=citation,
                    rrf_score=float(rrf_scores[i]),
                    ce_score=float(ce_scores[i]),
                    final_score=float(final_scores[i]),
                    retrieval_sources=cand.retrieval_sources,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        return {
            "ce_cache_size": self._cache.size,
            "ce_model": self._ce_model_name,
            "alpha": self.alpha,
            "rrf_k": self.RRF_K,
            "bm25_available": HAS_BM25,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_bbox(payload: dict) -> List[int]:
    """
    Best-effort bbox extraction from various payload shapes.
    The ingestor stores grounding inside the chunk payload; this handles
    cases where the bbox was flattened or stored at a different path.
    """
    for key in ("bbox", "bounding_box"):
        if key in payload:
            val = payload[key]
            if isinstance(val, list):
                return val
    return []


def format_citation(result: RankedResult, index: int = 1) -> str:
    """
    Renders a RankedResult as a human-readable inline citation string.

    Example output:
        [1] legal_statute_v1.pdf · p.0 · chunk 1 · score 0.923
            Merkle: a3f4b1c2... (2026-04-14T12:00:00+00:00)
    """
    c = result.citation
    lines = [
        f"[{index}] {c.filename} · p.{c.page_index} · chunk {c.chunk_index} "
        f"· score {result.final_score:.3f}",
        f"      CE={result.ce_score:.3f} RRF={result.rrf_score:.4f} "
        f"sources={result.retrieval_sources}",
        f"      Merkle: {c.version_root[:12]}... hash: {c.chunk_hash[:12]}... "
        f"({c.timestamp})",
    ]
    if c.bbox:
        lines.append(f"      bbox: {c.bbox}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

async def usage_example():
    """
    End-to-end demonstration of the hybrid reranker on top of the
    AsyncMerkleQdrantIngestor pipeline.
    """
    from ingestion_pipeline import AsyncMerkleQdrantIngestor
    from shared.schemas import Document, Metadata, Chunk, Grounding

    # 1. Boot ingestor + reranker
    ingestor = AsyncMerkleQdrantIngestor(qdrant_url="http://localhost:6333")
    await ingestor.setup()

    reranker = HybridReranker(ingestor=ingestor, alpha=0.7)

    # 2. Seed some data (skip if already ingested)
    filename = "legal_statute_v1.pdf"
    doc = Document(
        metadata=Metadata(filename=filename, page_index=0, page_count=3, category="legal"),
        chunks=[
            Chunk(
                chunk_markdown="Article 1: Freedom of speech is guaranteed to all citizens.",
                grounding=Grounding(bbox=[0, 0, 200, 50], page_index=0, score=1.0),
            ),
            Chunk(
                chunk_markdown="Article 2: Every accused person has the right to a fair trial.",
                grounding=Grounding(bbox=[0, 50, 200, 100], page_index=0, score=1.0),
            ),
            Chunk(
                chunk_markdown="Article 3: No person shall be subject to arbitrary detention.",
                grounding=Grounding(bbox=[0, 100, 200, 150], page_index=0, score=1.0),
            ),
            Chunk(
                chunk_markdown="Article 4: The right to privacy is inviolable.",
                grounding=Grounding(bbox=[0, 150, 200, 200], page_index=0, score=1.0),
            ),
            Chunk(
                chunk_markdown="Article 5: Freedom of assembly shall not be restricted without due process.",
                grounding=Grounding(bbox=[0, 200, 200, 250], page_index=0, score=1.0),
            ),
        ],
    )
    await ingestor.process_document(doc)

    # 3. Run hybrid reranking
    print("\n--- Hybrid Reranker Results ---")
    results = await reranker.rerank(
        query="right to fair trial and due process",
        retrieval_top_k=20,
        rerank_top_n=3,
        category="legal",
    )

    for i, result in enumerate(results, start=1):
        print(f"\n{format_citation(result, index=i)}")
        print(f"  Content: {result.content}")

    # 4. Point-in-time reranking (pinned to a specific Merkle snapshot)
    history = await ingestor.get_document_history(filename)
    if history:
        pinned_root = history[0]["version_root"]
        print(f"\n--- Point-in-time rerank (root: {pinned_root[:12]}...) ---")
        pinned_results = await reranker.rerank(
            query="right to fair trial and due process",
            retrieval_top_k=20,
            rerank_top_n=3,
            version_root=pinned_root,
        )
        for i, result in enumerate(pinned_results, start=1):
            print(f"\n{format_citation(result, index=i)}")

    # 5. Cache diagnostics
    print(f"\n--- Reranker stats ---")
    print(reranker.cache_stats())


if __name__ == "__main__":
    asyncio.run(usage_example())