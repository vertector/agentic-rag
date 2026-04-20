"""
AsyncMerkleQdrantIngestor — Production-Ready Build
"""

import asyncio
import hashlib
import json
import uuid
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import quote, unquote

import redis.asyncio as redis_async
import numpy as np
import litellm
from qdrant_client import AsyncQdrantClient, models
from shared.schemas import Document, Chunk, Metadata, Grounding
from shared.utils import HeaderStack

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SecureMerkleIngestor")
logging.getLogger("fastembed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Attempt to import embedding engines
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class IngestorError(RuntimeError):
    """Raised when a storage operation fails after logging the cause."""


# ---------------------------------------------------------------------------
# 1. Deterministic Merkle Math
# ---------------------------------------------------------------------------

def build_merkle_tree(hashes: List[str]) -> str:
    """
    Constructs a deterministic binary Merkle tree from a list of leaf hashes.

    IMPORTANT: hashes are NOT sorted — order encodes document sequence.
    Odd-length levels duplicate the last node (Bitcoin-style).
    """
    if not hashes:
        return hashlib.sha256(b"empty_node").hexdigest()

    current_level = list(hashes)  # defensive copy, order strictly preserved

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
            combined = (left + right).encode("utf-8")
            next_level.append(hashlib.sha256(combined).hexdigest())
        current_level = next_level

    return current_level[0]


# ---------------------------------------------------------------------------
# 2. Redis key helpers  (FIX-7)
# ---------------------------------------------------------------------------

def _encode_filename(filename: str) -> str:
    """URL-encode filename so colons / slashes cannot break Redis key structure."""
    return quote(filename, safe="")


def _decode_filename(encoded: str) -> str:
    return unquote(encoded)


# ---------------------------------------------------------------------------
# 3. Timestamp helper  (FIX-1)
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return current UTC time as a timezone-aware ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 4. Paginated scroll helper  (FIX-2, FIX-3)
# ---------------------------------------------------------------------------

async def _scroll_all(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    scroll_filter: models.Filter,
    with_payload,
    page_size: int = 1000,
) -> list:
    """
    Cursor-based full-collection scroll — never truncates regardless of
    total result count.  Returns a flat list of ScoredPoint / Record objects.

    Uses the next_page_offset token returned by each call; stops when
    Qdrant signals there are no more pages (offset is None).
    """
    all_points: list = []
    offset = None

    while True:
        try:
            batch, next_offset = await qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=page_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error("Qdrant scroll failed (offset=%s): %s", offset, exc)
            raise IngestorError(f"Qdrant scroll error: {exc}") from exc

        all_points.extend(batch)

        if next_offset is None:
            break
        offset = next_offset

    return all_points


# ---------------------------------------------------------------------------
# 5. Core Ingestor
# ---------------------------------------------------------------------------

class AsyncMerkleQdrantIngestor:
    """
    Versioned RAG ingestor with Merkle-tree integrity proofs.

    Every document page is fingerprinted by a Merkle root derived from the
    ordered sequence of its chunk content hashes.  Each new version is
    upserted atomically; the previous version is soft-deleted (is_active=False)
    rather than removed, enabling point-in-time queries and full audit trails.
    """

    # Payload fields used in filters — all get keyword/bool indexes in setup()
    _INDEXED_KEYWORD_FIELDS = [
        "version_root",
        "metadata.filename",
        "metadata.category",
        "metadata.blob_cid",
        "metadata.corpus_id",
        "filename",          # top-level on root-anchor points
    ]
    _INDEXED_BOOL_FIELDS = [
        "is_merkle_leaf",
        "is_active",
        "is_merkle_root",
    ]

    def __init__(
        self,
        qdrant_url: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        collection_base_name: str = "secure_rag",
        model_name: Optional[str] = None,
    ):
        self.qdrant_url = qdrant_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.collection_base_name = collection_base_name
        self.model_name = model_name or "BAAI/bge-small-en-v1.5"

        self.qdrant = AsyncQdrantClient(url=qdrant_url, timeout=30.0)
        self.redis = redis_async.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        self._init_encoder()

        self.model_id = self.model_name.replace("/", "-").lower()
        self.collection_name = f"{collection_base_name}_{self.model_id}"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self):
        """
        Initialise collection, payload indexes, and verify connectivity.

        FIX-6: payload indexes are created for every field used in filter
               conditions so scans are index-backed rather than full-table.
        FIX-8: connectivity to both Qdrant and Redis is verified; a
               prominent warning is logged when Redis persistence is off.
        """
        # --- Qdrant connectivity ---
        try:
            test_embedding = await asyncio.to_thread(self._embed, ["init_check"])
        except Exception as exc:
            raise IngestorError(f"Embedding engine failed during setup: {exc}") from exc

        vector_size = len(test_embedding[0])

        try:
            if not await self.qdrant.collection_exists(self.collection_name):
                logger.info(
                    "Creating collection: %s (dim=%d)", self.collection_name, vector_size
                )
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )

            # Create payload indexes (idempotent — Qdrant ignores duplicates)
            for field in self._INDEXED_KEYWORD_FIELDS:
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=False,   # non-blocking; indexes build in background
                )
            for field in self._INDEXED_BOOL_FIELDS:
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.BOOL,
                    wait=False,
                )

            logger.info(
                "Payload indexes ensured for collection: %s", self.collection_name
            )
        except IngestorError:
            raise
        except Exception as exc:
            raise IngestorError(f"Qdrant setup failed: {exc}") from exc

        # --- Redis connectivity + persistence warning (FIX-8) ---
        try:
            await self.redis.ping()
        except Exception as exc:
            raise IngestorError(f"Redis is not reachable: {exc}") from exc

        try:
            config = await self.redis.config_get("appendonly")
            aof_on = config.get("appendonly", "no").lower() == "yes"
            if not aof_on:
                config_rdb = await self.redis.config_get("save")
                rdb_on = bool(config_rdb.get("save", "").strip())
                if not rdb_on:
                    logger.warning(
                        "DURABILITY WARNING: Redis has neither AOF nor RDB persistence "
                        "enabled.  Loss of Redis data will invalidate all integrity "
                        "checks.  Enable AOF with 'appendonly yes' or configure RDB saves."
                    )
        except Exception:
            # CONFIG GET may be disabled on managed Redis — non-fatal
            logger.debug("Could not read Redis persistence config (may be managed Redis).")

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _init_encoder(self):
        if HAS_FASTEMBED:
            try:
                self.encoder = TextEmbedding(model_name=self.model_name)
                self.engine_type = "fastembed"
                logger.info("Loaded FastEmbed for %s", self.model_name)
                return
            except Exception as exc:
                logger.warning("FastEmbed load failed, trying SentenceTransformers: %s", exc)

        if HAS_SENTENCE_TRANSFORMERS:
            self.encoder = SentenceTransformer(self.model_name)
            self.engine_type = "st"
            logger.info("Loaded SentenceTransformers for %s", self.model_name)
        else:
            raise ImportError(
                "No embedding engine available. "
                "Install 'fastembed' or 'sentence-transformers'."
            )

    def _embed(self, texts: List[str]) -> List[np.ndarray]:
        if self.engine_type == "fastembed":
            return [np.array(e) for e in self.encoder.embed(texts)]
        return self.encoder.encode(texts)

    # ------------------------------------------------------------------
    # Internal: deactivate previous version
    # ------------------------------------------------------------------

    async def _deactivate_previous_version(
        self, filename: str, page_index: int, old_root: str
    ):
        """
        Soft-deletes all leaf points belonging to a prior snapshot by setting
        is_active=False.  This preserves point-in-time queryability while
        excluding stale chunks from active searches.
        """
        logger.info(
            "Deactivating snapshot %s... for %s (page %d)",
            old_root[:8], filename, page_index,
        )
        try:
            await self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": False},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="version_root",
                            match=models.MatchValue(value=old_root),
                        ),
                        models.FieldCondition(
                            key="metadata.filename",
                            match=models.MatchValue(value=filename),
                        ),
                        models.FieldCondition(
                            key="metadata.page_index",
                            match=models.MatchValue(value=page_index),
                        ),
                    ]
                ),
                wait=True,
            )
        except Exception as exc:
            # Non-fatal: the new version is already live; stale points remain
            # active temporarily but will be overridden on the next ingest.
            logger.error(
                "Failed to deactivate old snapshot %s...: %s  "
                "(stale points may appear in active searches until next ingest)",
                old_root[:8], exc,
            )

    async def _check_version_exists(self, filename: str, page_index: int, version_root: str) -> bool:
        """
        Efficiently check if a specific document state (Merkle root) already
        exists in the Qdrant object store.
        """
        try:
            # We only need one leaf point to confirm the snapshot exists
            results, _ = await self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="version_root", match=models.MatchValue(value=version_root)
                        ),
                        models.FieldCondition(
                            key="metadata.filename", match=models.MatchValue(value=filename)
                        ),
                        models.FieldCondition(
                            key="metadata.page_index", match=models.MatchValue(value=page_index)
                        ),
                        models.FieldCondition(
                            key="is_merkle_leaf", match=models.MatchValue(value=True)
                        ),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except Exception as exc:
            logger.debug("Existence check failed for version %s: %s", version_root[:8], exc)
            return False

    async def _activate_version(self, filename: str, page_index: int, version_root: str):
        """
        Re-activates all leaf points for a specific snapshot (Object Checkout).
        """
        logger.info(
            "Checking out existing snapshot %s... for %s (p.%d)",
            version_root[:8], filename, page_index,
        )
        try:
            await self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": True},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="version_root", match=models.MatchValue(value=version_root)
                        ),
                        models.FieldCondition(
                            key="metadata.filename", match=models.MatchValue(value=filename)
                        ),
                        models.FieldCondition(
                            key="metadata.page_index", match=models.MatchValue(value=page_index)
                        ),
                    ]
                ),
                wait=True,
            )
        except Exception as exc:
            logger.error("Failed to activate snapshot %s...: %s", version_root[:8], exc)
            raise IngestorError(f"Version activation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal: Hybrid Context Enrichment
    # ------------------------------------------------------------------

    async def _summarize_chunk(self, chunk: Chunk) -> Optional[str]:
        """
        Uses LLM (Gemini 3.1 Flash Lite Preview) to generate a semantic summary of 
        complex data blocks (tables, charts).
        
        Includes a Redis-backed caching layer based on the chunk's structural hash.
        """
        label = (chunk.grounding.chunk_type or "unknown").lower()
        content = chunk.chunk_markdown.strip()
        
        # Only summarize data-heavy blocks
        if label not in ("table", "chart", "figure", "image") and "<table>" not in content.lower():
            return None

        # --- Cache Lookup ---
        structural_hash = chunk.get_structural_hash()
        cache_key = f"cache:summary:{structural_hash}"
        logger.debug("Checking cache for %s. Key: %s", label, cache_key)
        try:
            cached_summary = await self.redis.get(cache_key)
            if cached_summary:
                logger.info("Cache hit for %s chunk summary.", label)
                return cached_summary
        except Exception as exc:
            logger.warning("Redis cache lookup failed: %s", exc)

        prompt = f"""
        Summarize the following {label} from a document. 
        Focus on the key insights, trends, and data points that would be useful for semantic search.
        Keep the summary concise but informative (max 3 sentences).
        
        Context: {chunk.context}
        
        Content:
        {content}
        """
        
        try:
            # Use the specified preview model with retries for rate limits
            import sys
            from contextlib import redirect_stdout

            # Redirect any library-level prints to stderr during the LLM call
            with redirect_stdout(sys.stderr):
                response = await litellm.acompletion(
                    model="gemini/gemini-3.1-flash-lite-preview",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0,
                    num_retries=3, # Handle RESOURCE_EXHAUSTED / ServiceUnavailable
                )
            summary = response.choices[0].message.content.strip()
            
            # --- Cache Update ---
            if summary:
                try:
                    # Cache for 30 days
                    await self.redis.set(cache_key, summary, ex=2592000)
                except Exception as exc:
                    logger.warning("Redis cache update failed: %s", exc)
                    
            return summary
        except Exception as exc:
            logger.warning("Summarization failed for %s chunk: %s", label, exc)
            return None

    def _enrich_chunks(
        self, 
        chunks: List[Chunk], 
        header_stack: Optional[HeaderStack] = None,
        use_deep_context: bool = True
    ) -> tuple[List[Chunk], HeaderStack]:
        """
        Enrich a sequence of structural chunks with hierarchical context.
        Returns: (enriched_chunks, updated_header_stack)
        """
        def strip_html_table(text: str) -> str:
            """Simple regex to strip HTML tags from a table string to reveal text content."""
            if "<table>" not in text.lower() and "<table" not in text.lower():
                return text
            # Replace tags with spaces to avoid merging words
            return re.sub(r'<[^>]+>', ' ', text).strip()

        # Robust Label Categories (aligned with DocumentParser)
        HEADER_LABELS = {
            "document_title", "title", "section_title", "section_header", 
            "paragraph_title"
        }
        CAPTION_LABELS = {
            "table_caption", "figure_caption", "caption", "figure_title",
            "table_title", "chart_title", "image_caption"
        }
        DATA_LABELS = {
            "table", "figure", "image", "chart", "algorithm", "formula", 
            "seal", "text", "plain_text", "paragraph", "list",
            "abstract", "table_of_contents", "references", "footnotes", 
            "vision_footnote", "aside_text", "reference_content"
        }
        
        enriched_chunks = []
        stack = header_stack or HeaderStack()
        
        pending_caption = ""
        pending_caption_bbox = [0, 0, 0, 0]
        pending_caption_score = 1.0

        for chunk in chunks:
            raw_label = chunk.grounding.chunk_type or "unknown"
            label = raw_label.lower().replace(" ", "_")
            content = chunk.chunk_markdown.strip()
            if not content: continue

            # --- State Tracking ---
            if label in HEADER_LABELS:
                level = HeaderStack.get_level(label, content)
                stack.push(level, content)
                # Headers stay standalone
            
            if label in CAPTION_LABELS:
                if pending_caption:
                    # Flush previous unmerged caption
                    breadcrumb = stack.format_breadcrumb() if use_deep_context else ""
                    # Context Injection
                    full_md = f"[Context: {breadcrumb}]\n\n{pending_caption}" if breadcrumb else pending_caption
                    enriched_chunks.append(Chunk(
                        chunk_markdown=full_md,
                        context=breadcrumb,
                        grounding=Grounding(
                            chunk_type="caption",
                            bbox=pending_caption_bbox,
                            page_index=chunk.grounding.page_index,
                            score=pending_caption_score
                        )
                    ))
                pending_caption = content
                pending_caption_bbox = chunk.grounding.bbox
                pending_caption_score = chunk.grounding.score
                continue # Buffer for potential merge
            
            # --- Context Injection & Merging ---
            chunk_markdown = content
            breadcrumb = stack.format_breadcrumb() if use_deep_context else ""
            context_parts = []
            if breadcrumb:
                context_parts.append(breadcrumb)
            
            # We merge captions specifically into 'data' blocks
            is_table = label == "table" or "<table>" in chunk_markdown.lower()
            if pending_caption and (label in DATA_LABELS or is_table):
                chunk_markdown = f"{pending_caption}\n\n{chunk_markdown}"
                context_parts.append(f"Caption: {pending_caption}")
                pending_caption = "" 
            elif pending_caption:
                # Flush pending caption
                standalone_caption_md = f"[Context: {breadcrumb}]\n\n{pending_caption}" if breadcrumb else pending_caption
                enriched_chunks.append(Chunk(
                    chunk_markdown=standalone_caption_md,
                    context=breadcrumb,
                    grounding=Grounding(
                        chunk_type="caption",
                        bbox=pending_caption_bbox,
                        page_index=chunk.grounding.page_index,
                        score=pending_caption_score
                    )
                ))
                pending_caption = ""

            # Table Content Flattening (Improved Retrieval for HTML Tables)
            if is_table:
                flattened = strip_html_table(chunk_markdown)
                if len(flattened) < len(chunk_markdown):
                     chunk_markdown = f"{flattened}\n\n{chunk_markdown}"

            # Final Context String
            context_str = " | ".join(context_parts) if context_parts else ""
            
            # Dynamic Context Injection into Markdown
            if breadcrumb and label not in HEADER_LABELS:
                full_markdown = f"[Context: {context_str}]\n\n{chunk_markdown}" if context_str else chunk_markdown
            else:
                full_markdown = chunk_markdown

            enriched_chunks.append(Chunk(
                chunk_markdown=full_markdown,
                context=context_str,
                grounding=Grounding(
                    chunk_type=raw_label,
                    bbox=chunk.grounding.bbox,
                    page_index=chunk.grounding.page_index,
                    score=chunk.grounding.score
                )
            ))
            
        # Final cleanup for pending captions at the end of the sequence
        if pending_caption:
             breadcrumb = stack.format_breadcrumb() if use_deep_context else ""
             full_md = f"[Context: {breadcrumb}]\n\n{pending_caption}" if breadcrumb else pending_caption
             enriched_chunks.append(Chunk(
                chunk_markdown=full_md,
                context=breadcrumb,
                grounding=Grounding(
                    chunk_type="caption",
                    bbox=pending_caption_bbox,
                    page_index=chunks[0].grounding.page_index if chunks else 1,
                    score=pending_caption_score
                )
            ))
            
        return enriched_chunks, stack

    # ------------------------------------------------------------------
    # process_document — main write path
    # ------------------------------------------------------------------

    async def process_document(
        self, 
        doc: Document, 
        initial_header_state: Optional[List[str]] = None,
        use_deep_context: bool = True
    ) -> tuple[bool, List[str]]:
        """
        Ingest a document with Merkle snapshot versioning.
        Enriches chunks with hierarchical context and LLM summaries before ingestion.

        Returns: (bool actually_written, List[str] final_header_state)
        """
        if not doc.chunks:
            logger.warning(
                "Document %s p.%d has no chunks — skipped.",
                doc.metadata.filename, doc.metadata.page_index,
            )
            return False, (initial_header_state or [])

        # --- Context Enrichment (Hybrid Approach) ---
        initial_stack = HeaderStack(initial_state=initial_header_state)
        enriched_chunks, updated_stack = self._enrich_chunks(
            doc.chunks, 
            header_stack=initial_stack,
            use_deep_context=use_deep_context
        )
        final_state = updated_stack.get_state()
        
        # --- LLM Summarization (Parallel) ---
        summaries = await asyncio.gather(*[self._summarize_chunk(c) for c in enriched_chunks])
        for chunk, summary in zip(enriched_chunks, summaries):
            chunk.summary = summary

        # Recalculate root based on enriched content (now including summaries)
        doc_cid = doc.metadata.blob_cid
        leaf_hashes = [c.get_content_hash(doc_cid) for c in enriched_chunks]
        root_hash = build_merkle_tree(leaf_hashes)

        encoded_fn = _encode_filename(doc.metadata.filename)
        redis_key = (
            f"state:{self.model_id}:doc:{encoded_fn}:page:{doc.metadata.page_index}"
        )

        # 1. Idempotency check
        try:
            previous_root = await self.redis.get(redis_key)
        except Exception as exc:
            raise IngestorError(f"Redis GET failed for key {redis_key}: {exc}") from exc

        if previous_root == root_hash:
            logger.info(
                "Synced (no change): %s p.%d (root: %s...)",
                doc.metadata.filename, doc.metadata.page_index, root_hash[:8],
            )
            return False, final_state

        # --- Restoration Path (Git-style checkout) ---
        if await self._check_version_exists(
            doc.metadata.filename, doc.metadata.page_index, root_hash
        ):
            logger.info(
                "Restoring existing version (Object Checkout): %s p.%d",
                doc.metadata.filename, doc.metadata.page_index,
            )
            await self._activate_version(
                doc.metadata.filename, doc.metadata.page_index, root_hash
            )
            if previous_root:
                await self._deactivate_previous_version(
                    doc.metadata.filename, doc.metadata.page_index, previous_root
                )
            try:
                await self.redis.set(redis_key, root_hash)
            except Exception as exc:
                raise IngestorError(f"Redis checkout update failed: {exc}") from exc
            return True, final_state

        # --- Standard Ingestion Path ---
        logger.info(
            "New snapshot for %s p.%d — root: %s...",
            doc.metadata.filename, doc.metadata.page_index, root_hash[:8],
        )

        # 2. Batch embeddings
        # Priority: Summary -> Markdown
        texts = [c.summary if c.summary else c.chunk_markdown for c in enriched_chunks]
        embeddings: List[np.ndarray] = []
        embed_batch_size = 100
        try:
            for i in range(0, len(texts), embed_batch_size):
                batch_embeds = await asyncio.to_thread(
                    self._embed, texts[i : i + embed_batch_size]
                )
                embeddings.extend(batch_embeds)
                await asyncio.sleep(0)
        except Exception as exc:
            raise IngestorError(f"Embedding failed: {exc}") from exc

        # 3. Build Qdrant point structs
        now_iso = _utcnow_iso()
        points: List[models.PointStruct] = []

        for idx, (chunk, vector) in enumerate(zip(enriched_chunks, embeddings)):
            chunk_hash = chunk.get_content_hash(doc_cid)
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{self.model_id}:{chunk_hash}:{root_hash}",
                )
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "content": chunk.chunk_markdown,
                        "summary": chunk.summary,
                        "context": chunk.context,
                        "metadata": doc.metadata.model_dump(exclude={"page_image_base64"}),
                        "grounding": chunk.grounding.model_dump() if chunk.grounding else {},
                        "chunk_hash": chunk_hash,
                        "chunk_index": idx,
                        "version_root": root_hash,
                        "timestamp": now_iso,
                        "is_merkle_leaf": True,
                        "is_active": True,
                    },
                )
            )

        root_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"root:{root_hash}"))
        points.append(
            models.PointStruct(
                id=root_point_id,
                vector=[0.0] * len(embeddings[0]),
                payload={
                    "is_merkle_root": True,
                    "version_root": root_hash,
                    "filename": doc.metadata.filename,
                    "page_index": doc.metadata.page_index,
                    "chunk_count": len(enriched_chunks),
                    "timestamp": now_iso,
                },
            )
        )

        # 4. Upsert to Qdrant
        upsert_batch_size = 200
        try:
            for i in range(0, len(points), upsert_batch_size):
                batch = points[i : i + upsert_batch_size]
                is_last = (i + upsert_batch_size) >= len(points)
                await self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=is_last,
                )
        except Exception as exc:
            raise IngestorError(f"Qdrant upsert failed: {exc}") from exc

        # 5. Soft-delete previous version
        if previous_root:
            await self._deactivate_previous_version(
                doc.metadata.filename, doc.metadata.page_index, previous_root
            )

        # 6. Commit new state to Redis
        try:
            await self.redis.set(redis_key, root_hash)
        except Exception as exc:
            raise IngestorError(f"Redis SET failed for key {redis_key}: {exc}") from exc

        logger.info("Snapshot %s... committed successfully.", root_hash[:8])
        return True, final_state

    # ------------------------------------------------------------------
    # verify_integrity  (FIX-2: paginated scroll)
    # ------------------------------------------------------------------

    async def verify_integrity(self, filename: str, page_index: int) -> bool:
        """
        Mathematically re-derives the Merkle root from Qdrant and compares it
        to the trusted root stored in Redis.

        FIX-2: Uses paginated scroll (_scroll_all) instead of limit=10_000 so
        documents with >10 000 chunks are verified correctly.
        """
        encoded_fn = _encode_filename(filename)
        redis_key = f"state:{self.model_id}:doc:{encoded_fn}:page:{page_index}"

        try:
            expected_root = await self.redis.get(redis_key)
        except Exception as exc:
            logger.error("Redis GET failed during integrity check: %s", exc)
            return False

        if not expected_root:
            logger.error(
                "Integrity check failed: no tracked state for %s p.%d",
                filename, page_index,
            )
            return False

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="version_root",
                    match=models.MatchValue(value=expected_root),
                ),
                models.FieldCondition(
                    key="is_merkle_leaf",
                    match=models.MatchValue(value=True),
                ),
            ]
        )

        try:
            all_points = await _scroll_all(
                qdrant=self.qdrant,
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                with_payload=["chunk_hash", "chunk_index"],
            )
        except IngestorError:
            logger.critical("Integrity check aborted — scroll failed for %s p.%d", filename, page_index)
            return False

        if not all_points:
            logger.critical(
                "Integrity failure: no leaf vectors found in DB for root %s...",
                expected_root[:8],
            )
            return False

        # Restore original chunk order before rebuilding the tree
        sorted_points = sorted(all_points, key=lambda p: p.payload.get("chunk_index", 0))
        actual_hashes = [p.payload["chunk_hash"] for p in sorted_points]
        reconstructed_root = build_merkle_tree(actual_hashes)

        if reconstructed_root != expected_root:
            logger.critical(
                "INTEGRITY FAILURE for %s p.%d: expected root %s..., got %s...",
                filename, page_index, expected_root[:8], reconstructed_root[:8],
            )
            return False

        logger.info("Integrity verified for %s p.%d (%d chunks)", filename, page_index, len(sorted_points))
        return True

    # ------------------------------------------------------------------
    # get_document_history  (FIX-3: paginated scroll)
    # ------------------------------------------------------------------

    async def get_document_history(self, filename: str) -> List[dict]:
        """
        Retrieves the full audit trail for a document (all historical Merkle roots).

        FIX-3: Uses paginated scroll (_scroll_all) instead of limit=100 so
        documents with >100 versions are returned completely.
        """
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="is_merkle_root",
                    match=models.MatchValue(value=True),
                ),
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename),
                ),
            ]
        )

        try:
            all_points = await _scroll_all(
                qdrant=self.qdrant,
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
            )
        except IngestorError as exc:
            logger.error("get_document_history scroll failed: %s", exc)
            return []

        history = [
            {
                "version_root": p.payload.get("version_root"),
                "timestamp": p.payload.get("timestamp"),
                "page_index": p.payload.get("page_index"),
                "chunk_count": p.payload.get("chunk_count"),
            }
            for p in all_points
        ]
        return sorted(history, key=lambda x: x.get("timestamp") or "", reverse=True)

    # ------------------------------------------------------------------
    # secure_search
    # ------------------------------------------------------------------

    async def secure_search(
        self,
        query: str,
        category: Optional[str] = None,
        corpus_id: Optional[str] = None,
        version_root: Optional[str] = None,
        limit: int = 5,
        collection_name: Optional[str] = None,
    ):
        """
        Semantic RAG search with point-in-time version pinning and Corpus isolation.

        Omit `version_root` to query the current active version.
        Pass a specific `version_root` to query any historical snapshot.
        Pass `corpus_id` to strictly limit the search space to a specific Knowledge Base.
        """
        try:
            query_vector = await asyncio.to_thread(self._embed, [query])
        except Exception as exc:
            raise IngestorError(f"Query embedding failed: {exc}") from exc

        must_filters = [
            models.FieldCondition(
                key="is_merkle_leaf", match=models.MatchValue(value=True)
            )
        ]

        if category:
            must_filters.append(
                models.FieldCondition(
                    key="metadata.category", match=models.MatchValue(value=category)
                )
            )

        if corpus_id:
            must_filters.append(
                models.FieldCondition(
                    key="metadata.corpus_id", match=models.MatchValue(value=corpus_id)
                )
            )

        if version_root:
            must_filters.append(
                models.FieldCondition(
                    key="version_root", match=models.MatchValue(value=version_root)
                )
            )
        else:
            must_filters.append(
                models.FieldCondition(
                    key="is_active", match=models.MatchValue(value=True)
                )
            )

        if category:
            must_filters.append(
                models.FieldCondition(
                    key="metadata.category", match=models.MatchValue(value=category)
                )
            )

        try:
            response = await self.qdrant.query_points(
                collection_name=collection_name or self.collection_name,
                query=query_vector[0].tolist(),
                query_filter=models.Filter(must=must_filters),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise IngestorError(f"Qdrant query failed: {exc}") from exc

        return response.points

    # ------------------------------------------------------------------
    # purge_document  (FIX-4: compound OR filter for both key paths)
    # ------------------------------------------------------------------

    async def purge_document(self, filename: str):
        """
        Hard-deletes ALL data for a given document (all pages, all versions,
        all root anchors) from both Qdrant and Redis.

        FIX-4: The original filter only matched points whose payload uses the
        "metadata.filename" path (leaf points), silently leaving root-anchor
        points (which store the filename under the top-level key "filename")
        behind.

        The fixed filter uses a `should` (OR) clause covering both paths so
        that a single Qdrant delete operation removes every point associated
        with the document.
        """
        logger.info("Purging all data for document: %s", filename)

        # OR filter: matches leaf points (metadata.filename) AND root anchors (filename)
        doc_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.filename",
                    match=models.MatchValue(value=filename),
                ),
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename),
                ),
            ]
        )

        try:
            await self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=doc_filter,
                wait=True,
            )
            logger.info("Qdrant points purged for: %s", filename)
        except Exception as exc:
            raise IngestorError(f"Qdrant delete failed for {filename}: {exc}") from exc

        # Remove all Redis state keys for this document (all pages)
        encoded_fn = _encode_filename(filename)
        pattern = f"state:{self.model_id}:doc:{encoded_fn}:page:*"
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                cursor = 0
                keys_to_delete: List[str] = []
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor=cursor, match=pattern, count=100
                    )
                    keys_to_delete.extend(keys)
                    if cursor == 0:
                        break
                if keys_to_delete:
                    pipe.delete(*keys_to_delete)
                    await pipe.execute()
                    logger.info(
                        "Redis keys purged for %s (%d keys)", filename, len(keys_to_delete)
                    )
                else:
                    logger.info("No Redis keys found for %s", filename)
        except Exception as exc:
            raise IngestorError(
                f"Redis purge failed for {filename} (Qdrant already purged): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # reconcile_redis_from_qdrant  (bonus: Redis recovery utility)
    # ------------------------------------------------------------------

    async def reconcile_redis_from_qdrant(self, filename: str) -> int:
        """
        Recovery utility: re-seeds Redis active-root state from Qdrant.

        Use this if Redis data is lost. For each page in the document's
        history, the most recent root anchor (by timestamp) is written
        back to Redis as the active root.

        Returns the number of pages reconciled.
        """
        logger.info("Reconciling Redis state from Qdrant for: %s", filename)
        history = await self.get_document_history(filename)

        if not history:
            logger.warning("No history found in Qdrant for %s — nothing to reconcile.", filename)
            return 0

        # Group by page_index and pick the latest root per page
        page_latest: dict = {}
        for entry in history:
            page = entry.get("page_index")
            if page is None:
                continue
            existing = page_latest.get(page)
            if existing is None or (entry.get("timestamp") or "") > (existing.get("timestamp") or ""):
                page_latest[page] = entry

        reconciled = 0
        for page_index, entry in page_latest.items():
            root = entry.get("version_root")
            if not root:
                continue
            encoded_fn = _encode_filename(filename)
            redis_key = f"state:{self.model_id}:doc:{encoded_fn}:page:{page_index}"
            try:
                await self.redis.set(redis_key, root)
                logger.info("Reconciled page %d → root %s...", page_index, root[:8])
                reconciled += 1
            except Exception as exc:
                logger.error("Failed to reconcile page %d: %s", page_index, exc)

        logger.info("Reconciliation complete: %d page(s) for %s", reconciled, filename)
        return reconciled


# ---------------------------------------------------------------------------
# 7. Usage Example
# ---------------------------------------------------------------------------

async def usage_example():
    """
    Full lifecycle: ingest → verify → update → history → point-in-time search → purge.
    """
    ingestor = AsyncMerkleQdrantIngestor(qdrant_url="http://localhost:6333")
    await ingestor.setup()

    filename = "legal_statute_v1.pdf"

    # 1. Ingest Version 1
    doc_v1 = Document(
        metadata=Metadata(filename=filename, page_index=0, page_count=1, category="legal"),
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
    print("\n[Step 1] Ingesting Version 1 ...")
    await ingestor.process_document(doc_v1)

    # 2. Integrity audit
    is_valid = await ingestor.verify_integrity(filename, 0)
    print(f"[Step 2] Integrity Audit: {'PASSED' if is_valid else 'FAILED'}")

    # 3. Update (Version 2)
    doc_v2 = Document(
        metadata=Metadata(filename=filename, page_index=0, page_count=1, category="legal"),
        chunks=[
            Chunk(
                chunk_markdown="Article 1: Freedom of speech is guaranteed.",
                grounding=Grounding(bbox=[0, 0, 10, 10], page_index=0, score=1.0),
            ),
            Chunk(
                chunk_markdown="Article 2: Every accused person has the right to a fair trial. (Updated with Clause A)",
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
    print("\n[Step 3] Ingesting Version 2 ...")
    await ingestor.process_document(doc_v2)

    # 4. Audit trail
    history = await ingestor.get_document_history(filename)
    print(f"\n[Step 4] Audit Trail for {filename}:")
    for h in history:
        print(f"  root: {(h['version_root'] or '')[:12]}...  ts: {h['timestamp']}")

    # 5. Point-in-time search comparison
    print("\n[Step 5] Search Comparison:")

    results_now = await ingestor.secure_search("trial rights")
    if results_now:
        print(f"  Active result: {results_now[0].payload.get('content', '')[:80]}")

    if len(history) >= 2:
        v1_root = history[-1]["version_root"]  # oldest
        results_old = await ingestor.secure_search("trial rights", version_root=v1_root)
        if results_old:
            print(
                f"  Point-in-time ({v1_root[:8]}): "
                f"{results_old[0].payload.get('content', '')[:80]}"
            )

    # 6. Cleanup
    # await ingestor.purge_document(filename)


if __name__ == "__main__":
    asyncio.run(usage_example())
