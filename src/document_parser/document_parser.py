"""
Document parser using PaddleOCRVL with Redis-backed 3-tier optimization.

Optimizations:
1. CAS Symlinking: Results stored in .cache/parsed/<hash>/, symlinked from src/
2. LRU Pruning: Redis-backed tracking and automated cleanup of old versions.
3. Incremental Parsing: Page-level hashing/caching via fitz (PyMuPDF).
"""

import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import redis
import fitz  # PyMuPDF
from PIL import Image

from shared.schemas import Chunk, Document, Grounding, Metadata, PipelineSettings
from shared.env_loader import load_env
from shared.utils import validate_path, sanitize_stem, resolve_placeholders
from shared.blob_store import get_blob_store

# Hydrate environment from .env
load_env()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers for ProcessPoolExecutor workers.
# ---------------------------------------------------------------------------

_worker_parser: Optional["DocumentParser"] = None


def _init_worker(settings_dict: dict) -> None:
    """Create one DocumentParser per worker process (called once at startup)."""
    global _worker_parser
    _worker_parser = DocumentParser(PipelineSettings(**settings_dict))


def _worker_parse(input_path: str) -> dict:
    """
    Parse a single document inside a worker process.
    Returns a dict with 'documents' (JSON-safe) and 'output_path'.
    """
    assert _worker_parser is not None, "Worker not initialised — _init_worker not called"
    docs, path = _worker_parser.parse(input_path)
    return {
        "documents": [doc.model_dump(mode="json") for doc in docs],
        "output_path": str(path)
    }


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------


# Fields that affect parse output and must be included in the cache key.
# Infrastructure fields (vl_rec_backend, vl_rec_server_url, vl_rec_api_key) are
# excluded — they determine how the VLM is reached, not what it produces, so
# switching backends pointing at the same model reuses cached results correctly.
_CACHE_KEY_SETTINGS = (
    "use_ocr_for_image_block",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_chart_recognition",
    "use_layout_detection",
    "use_seal_recognition",
    "format_block_content",
    "merge_layout_blocks",
    "merge_tables",
    "relevel_titles",
    "markdown_ignore_labels",
    "pipeline_version",
    "parser_logic_version",
    "layout_threshold",
    "layout_nms",
    "layout_unclip_ratio",
    "layout_merge_bboxes_mode",
    "layout_shape_mode",
    "temperature",
    "top_p",
    "max_new_tokens",
    "repetition_penalty",
    "prompt_label",
    "vl_rec_api_model_name",
    # min_pixels, max_pixels and vlm_extra_args are intentionally excluded:
    # min_pixels/max_pixels are silently nulled by PipelineSettings validators
    # for backends that don't support them (e.g. mlx-vlm-server), producing
    # different hashes for functionally identical parses. vlm_extra_args is an
    # arbitrary pass-through dict whose serialisation is not guaranteed stable.
)

# Baseline defaults for delta-hashing. Computed once at import time from a
# default PipelineSettings instance so it stays in sync with the schema.
_PIPELINE_DEFAULTS: dict = PipelineSettings().model_dump()


class DirectoryLock:
    """Simple cross-process directory lock using atomic mkdir."""
    def __init__(self, lock_path: Path, timeout: float = 120.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._locked = False

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                self.lock_path.mkdir(exist_ok=False)
                self._locked = True
                return self
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    # Check if stale (older than 10 mins)
                    try:
                        if time.time() - self.lock_path.stat().st_mtime > 600:
                            logger.warning("Clearing stale lock: %s", self.lock_path)
                            shutil.rmtree(self.lock_path, ignore_errors=True)
                            continue
                    except Exception:
                        pass
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._locked:
            try:
                shutil.rmtree(self.lock_path, ignore_errors=True)
            except Exception:
                pass
            self._locked = False


class DocumentParser:
    """
    Production-ready document parser using PaddleOCRVL with 3-tier optimization.
    """

    __slots__ = ("_settings", "_pipeline", "_pipeline_lock", "_redis", "_cache_root", "_lru_key", "_ocr_lock_path")

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        self._settings = settings or PipelineSettings()
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

        # Cache and LRU tracking setup
        project_src = Path(__file__).resolve().parent.parent
        self._cache_root = (project_src / ".cache").resolve()
        (self._cache_root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self._cache_root / "ocr_results").mkdir(parents=True, exist_ok=True)
        
        # Cleanup old cache dirs if they exist
        shutil.rmtree(self._cache_root / "pages", ignore_errors=True)
        shutil.rmtree(self._cache_root / "parsed", ignore_errors=True)
        
        # Shared locks
        self._ocr_lock_path = self._cache_root / "ocr_results.lock"
        self._lru_key = "parser:lru"

        # Initialise Redis connection for caching
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        try:
            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=2.0,
            )
            self._redis.ping()
            logger.info("Connected to Redis at %s:%d for parsing cache.", redis_host, redis_port)
        except Exception as exc:
            logger.warning("Redis parsing cache unavailable: %s", exc)
            self._redis = None

    def _ocr_lock(self):
        """Cross-process lock for the OCR result cache."""
        return DirectoryLock(self._ocr_lock_path)

    def _get_ocr_cache_path(self, visual_hash: str) -> Path:
        """Returns the path to the raw OCR result for a given visual content."""
        # Include model name to ensure model upgrades bust the cache (Flaw 4)
        model_name = sanitize_stem(self._settings.vl_rec_api_model_name)
        return self._cache_root / "ocr_results" / model_name / visual_hash

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

    def _get_settings_hash(self) -> str:
        """
        Return a short hash representing only the output-affecting settings that
        differ from PipelineSettings defaults.

        Hashing the full settings snapshot causes spurious cache misses when the
        LLM agent sends redundant fields (fields it wasn't asked to change but
        includes anyway, e.g. relevel_titles=True matching the default). By
        hashing only the delta from defaults, two parses with the same meaningful
        overrides produce the same cache key regardless of which extra fields the
        LLM happened to include in the tool call.
        """
        current = self._settings.model_dump()
        defaults = _PIPELINE_DEFAULTS
        delta = {
            k: current[k]
            for k in _CACHE_KEY_SETTINGS
            if k in current and (k == "parser_logic_version" or current[k] != defaults.get(k))
        }
        blob = json.dumps(delta, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _get_pipeline(self):
        """Thread-safe singleton for VLM pipeline."""
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    from paddleocr import PaddleOCRVL
                    self._pipeline = PaddleOCRVL(**self._settings.to_init_kwargs())
        return self._pipeline

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_page_hashes(self, file_path: Path) -> List[str]:
        """Generate stable hashes for each page using normalized rendering and text analysis."""
        hashes = []
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            doc = fitz.open(str(file_path))
            for page in doc:
                # 1. Structural Metadata (Geometry)
                geom = f"{page.rect.width:.2f}:{page.rect.height:.2f}"
                
                # 2. Text Content (Content Stability)
                # We strip to avoid trailing invisible whitespace differences
                text = page.get_text("text").strip().encode("utf-8")
                
                # 3. Normalized Visual Rendering (Visual Stability)
                # We normalize to a small 64px thumbnail and Binarize the results.
                # Thresholding at 128 removes all anti-aliasing gray levels, leaving only 
                # the core visual structure (Black or White). This is extremely stable.
                max_dim = max(page.rect.width, page.rect.height, 1)
                z = 64 / max_dim 
                
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(z, z), 
                    colorspace=fitz.csGRAY, 
                    alpha=False,
                    annots=False
                )
                
                # 4. Binarization (High Stability)
                # Any pixel > 127 becomes White (255), everything else Black (0).
                # This completely eliminates sub-pixel rendering and font-rendering noise.
                binarized = bytes([255 if s > 127 else 0 for s in pix.samples])
                
                # Combine factors for a robust "Git-like" content identity
                combined = hashlib.sha256()
                combined.update(geom.encode("utf-8"))
                combined.update(b"|")
                combined.update(text)
                combined.update(b"|")
                combined.update(binarized)
                
                hashes.append(combined.hexdigest())
            doc.close()
        else:
            # Single page image
            with open(file_path, "rb") as f:
                hashes.append(hashlib.sha256(f.read()).hexdigest())
        return hashes

    def _update_lru(self, file_hash: str):
        """Update last accessed timestamp for a hash in Redis."""
        if self._redis:
            try:
                self._redis.zadd(self._lru_key, {file_hash: time.time()})
            except Exception as exc:
                logger.debug("LRU update failed: %s", exc)

    def prune_cache(self, max_size_gb: float = 10.0):
        """Delete least recently used cache entries until total size is within limit."""
        if not self._redis:
            logger.warning("Redis unavailable — pruning skipped.")
            return

        try:
            current_size = sum(f.stat().st_size for f in self._cache_root.rglob('*') if f.is_file())
            max_size = max_size_gb * 1024**3

            if current_size <= max_size:
                return

            logger.info("Cache pruning: %.2f GB > %.2f GB limit.", current_size/1024**3, max_size_gb)

            while current_size > max_size:
                oldest = self._redis.zpopmin(self._lru_key)
                if not oldest: break
                
                oldest_hash = oldest[0][0]
                target = self._cache_root / "parsed" / oldest_hash
                if target.exists():
                    dir_size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
                    shutil.rmtree(target)
                    current_size -= dir_size
                    logger.info("Pruned old cache entry: %s", oldest_hash)

            logger.info("Pruning complete. New size: %.2f GB", current_size/1024**3)
        except Exception as exc:
            logger.error("Pruning failed: %s", exc)

    @staticmethod
    def _image_to_base64(img) -> str:
        """Convert PIL Image/numpy array to base64."""
        if not hasattr(img, "save"): img = Image.fromarray(img)
        with BytesIO() as buf:
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _save_page(res, temp_dir: str) -> None:
        """Write page results to temp dir."""
        res.save_to_json(save_path=temp_dir)
        res.save_to_markdown(save_path=temp_dir)

    def _build_chunks(self, json_data: dict, page_index: int) -> List[Chunk]:
        """
        Centrally build context-aware chunks using a hybrid approach:
        1. Hierarchical Context: Track current 'title' to provide breadcrumbs.
        2. Caption Pairing: Merge captions into subsequent tables/figures.
        """
        ignored = set(self._settings.markdown_ignore_labels or [])
        chunks = []
        
        # Robust Label Categories based on PaddleOCR-VL / PP-DocLayoutV2
        HEADER_LABELS = {
            "document_title", "title", "section_title", "section_header", 
            "paragraph_title"
        }
        CAPTION_LABELS = {
            "table_caption", "figure_caption", "caption", "figure_title",
            "table_title", "chart_title", "image_caption", "vision_footnote"
        }
        DATA_LABELS = {
            "table", "figure", "image", "chart", "algorithm", "formula", 
            "seal", "text", "plain_text", "paragraph", "list",
            "abstract", "table_of_contents", "references", "footnotes", 
            "aside_text", "reference_content"
        }

        parsing_res = json_data.get("parsing_res_list", [])
        layout_boxes = json_data.get("layout_det_res", {}).get("boxes", [])

        for i, (block, layout_item) in enumerate(zip(parsing_res, layout_boxes or [{}] * len(parsing_res))):
            raw_label = block.get("block_label", "unknown")
            if raw_label in ignored: continue
            
            # Normalize label: spaces to underscores, lowercase
            label = raw_label.lower().replace(" ", "_")
            
            content = block.get("block_content", "").strip()
            if not content: continue
            
            bbox = block.get("block_bbox", [0, 0, 0, 0])
            score = layout_item.get("score", 1.0) if isinstance(layout_item, dict) else 1.0

            # DocumentParser now produces "Pristine Chunks" without merging.
            # Every block has its own identity and bounding box for the visualizer.
            # IngestionPipeline will handle dynamic context injection and semantic merging.
            chunks.append(Chunk(
                chunk_markdown=content,
                context="",
                grounding=Grounding(
                    chunk_type=raw_label, # Keep original label for metadata
                    bbox=bbox,
                    page_index=page_index,
                    score=score
                )
            ))
            
        return chunks

    def _process_page(self, page_output: dict, json_path: Path, md_path: Path, blob_cid: Optional[str] = None) -> Document:
        """Build Document from saved page files."""
        with open(json_path, "r", encoding="utf-8") as f: json_data = json.load(f)
        with open(md_path, "r", encoding="utf-8") as f: md_data = f.read()

        page_index = (json_data.get("page_index") or 0) + 1
        chunks = self._build_chunks(json_data, page_index)

        output_img = page_output.get("doc_preprocessor_res", {}).get("output_img")
        metadata = Metadata(
            filename=json_data.get("input_path", ""),
            page_image_base64=self._image_to_base64(output_img) if output_img is not None else "",
            page_index=page_index,
            page_count=json_data.get("page_count") or 1,
            blob_cid=blob_cid,
        )
        return Document(markdown=md_data, chunks=chunks, metadata=metadata)

    def parse(self, input_path: str, **kwargs) -> Tuple[List[Document], Path]:
        """Extract structured markdown/chunks with Snapshot-based versioning and OCR deduplication."""
        from shared.utils import atomic_json_dump
        
        # 1. Identity & Context
        resolved_path = resolve_placeholders(input_path)
        document_path = validate_path(resolved_path)
        if not document_path.exists(): raise FileNotFoundError(f"No file at {document_path}")

        blob_cid = get_blob_store().put_file(document_path)
        settings_hash = self._get_settings_hash()
        
        # Flaw 5: Snapshot-based immutability. Each parse run is a unique commit.
        snapshot_id = f"{blob_cid}-{settings_hash}"
        snapshot_dir = (self._cache_root / "snapshots" / snapshot_id).resolve()
        manifest_path = snapshot_dir / "manifest.json"
        
        self._update_lru(snapshot_id)
        
        safe_stem = sanitize_stem(document_path.stem)
        project_src = Path(__file__).resolve().parent.parent
        symlink_path = project_src / f"{safe_stem}-{settings_hash}"

        # --- Tier 1: Snapshot Cache Check ---
        if manifest_path.exists():
            logger.info("Snapshot hit: %s", snapshot_id[:8])
            self._ensure_symlink(symlink_path, snapshot_dir)
            with open(manifest_path, "r", encoding="utf-8") as f:
                docs = [Document.model_validate(d) for d in json.load(f)]
                return docs, manifest_path

        # --- Tier 2: OCR Cache Check & Inference ---
        visual_hashes = self._get_page_hashes(document_path)
        pages: List[Optional[Document]] = [None] * len(visual_hashes)
        missing_indices = []

        for i, vh in enumerate(visual_hashes):
            ocr_path = self._get_ocr_cache_path(vh) / "ocr_result.json"
            if ocr_path.exists():
                # Dedup hit: Reuse raw OCR output but re-contextualise for THIS document/page
                with open(ocr_path, "r", encoding="utf-8") as f:
                    raw_ocr = json.load(f)
                    pages[i] = self._reassemble_page(raw_ocr, blob_cid, i + 1, len(visual_hashes))
            else:
                missing_indices.append(i)

        if missing_indices:
            # Inference required for subset
            logger.info("OCR partial miss: %d/%d pages.", len(missing_indices), len(visual_hashes))
            raw_results = self._infer_missing_pages(document_path, missing_indices)
            
            # Atomic commit to OCR Cache
            with self._ocr_lock():
                for idx, raw in zip(missing_indices, raw_results):
                    vh = visual_hashes[idx]
                    ocr_dir = self._get_ocr_cache_path(vh)
                    ocr_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json_dump(ocr_dir / "ocr_result.json", raw)
                    pages[idx] = self._reassemble_page(raw, blob_cid, idx + 1, len(visual_hashes))

        # --- Tier 3: Snapshot Assembly ---
        documents = [p for p in pages if p is not None]
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(manifest_path, [d.model_dump(mode="json") for d in documents])
        
        # Export sidecar markdown files for human/repo consumption
        for d in documents:
            md_name = f"{safe_stem}_p{d.metadata.page_index}.md"
            from shared.utils import atomic_write
            atomic_write(snapshot_dir / md_name, d.markdown)
            
        self._ensure_symlink(symlink_path, snapshot_dir)
        return documents, manifest_path

    def _reassemble_page(self, raw_ocr: dict, blob_cid: str, page_index: int, page_count: int) -> Document:
        """Convert raw OCR/VLM data into a semantically valid Document object for a specific ordinal position."""
        # This solves Flaw 7: Ordinal position is injected at reassembly, not baked into the OCR cache.
        chunks = self._build_chunks(raw_ocr, page_index)

        metadata = Metadata(
            filename=raw_ocr.get("input_path", "unknown"),
            page_index=page_index,
            page_count=page_count,
            page_image_base64=raw_ocr.get("page_image", ""),
            blob_cid=blob_cid
        )
        return Document(markdown=raw_ocr.get("markdown", ""), chunks=chunks, metadata=metadata)

    def _infer_missing_pages(self, document_path: Path, indices: List[int]) -> List[dict]:
        """Perform VLM inference on a subset of pages."""
        pipeline = self._get_pipeline()
        raw_results = []
        
        with tempfile.NamedTemporaryFile(suffix=document_path.suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            if document_path.suffix.lower() == ".pdf":
                src = fitz.open(str(document_path))
                dst = fitz.open()
                dst.insert_pdf(src)
                dst.select(indices)
                dst.save(str(tmp_path))
                src.close()
                dst.close()
            else:
                shutil.copy(document_path, tmp_path)

        try:
            preds = list(pipeline.predict(input=str(tmp_path)))
            # We must use the library's official serialization (save_to_json) 
            # as PredictResult contains non-serializable PaddleOCRVLBlock objects.
            with tempfile.TemporaryDirectory() as td:
                for p in preds:
                    p.save_to_json(save_path=td)
                    p.save_to_markdown(save_path=td)
                
                stem = tmp_path.stem
                json_files = sorted(Path(td).glob(f"**/{stem}*.json"))
                md_files = sorted(Path(td).glob(f"**/{stem}*.md"))
                
                for p, j_path, m_path in zip(preds, json_files, md_files):
                    with open(j_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    with open(m_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    
                    raw_results.append({
                        "parsing_res_list": data.get("parsing_res_list", []),
                        "markdown": md_content,
                        "page_image": self._image_to_base64(p.get("doc_preprocessor_res", {}).get("output_img")) if p.get("doc_preprocessor_res") else "",
                        "input_path": document_path.name
                    })
        finally:
            if tmp_path.exists(): os.unlink(tmp_path)
            
        return raw_results

    def _ensure_symlink(self, link: Path, target: Path):
        """Safely ensure link points to target, handling existing files/dirs/links."""
        target_abs = target.resolve()
        
        if link.is_symlink():
            try:
                if link.resolve() == target_abs:
                    return
            except OSError:
                pass # Broken link
            link.unlink()
        elif link.exists():
            if link.is_dir():
                shutil.rmtree(link)
            else:
                link.unlink()
        
        # Ensure parent exists
        link.parent.mkdir(parents=True, exist_ok=True)
        # Ensure target is a concrete directory
        target.mkdir(parents=True, exist_ok=True)
        
        link.symlink_to(target, target_is_directory=True)

    def _persist_to_cache(self, folder: Path, docs: List[Document], doc_path: Path, link: Path):
        """Persist full document results and update symlink. Idempotent: no-op if already written."""
        from shared.utils import atomic_write, atomic_json_dump
        folder.mkdir(parents=True, exist_ok=True)
        output_file = folder / "documents.json"
        if not output_file.exists():
            atomic_json_dump(output_file, [d.model_dump(mode="json") for d in docs])
            for d in docs:
                md_name = f"{doc_path.stem}_{d.metadata.page_index}.md"
                atomic_write(folder / md_name, d.markdown)
        self._ensure_symlink(link, folder)

    def parse_batch(self, input_paths: List[str], max_workers: Optional[int] = None) -> List[dict]:
        """Parallel batch parsing with cached returns."""
        if not input_paths: return []
        n_workers = max_workers or min(len(input_paths), os.cpu_count() or 1)
        settings = self._settings.model_dump()
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(settings,)) as pool:
            results: List[dict] = list(pool.map(_worker_parse, input_paths))
        
        for res in results:
            res["documents"] = [Document.model_validate(d) for d in res["documents"]]
            res["output_path"] = Path(res["output_path"])
        return results