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


class DocumentParser:
    """
    Production-ready document parser using PaddleOCRVL with 3-tier optimization.
    """

    __slots__ = ("_settings", "_pipeline", "_pipeline_lock", "_redis", "_cache_root", "_lru_key")

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        self._settings = settings or PipelineSettings()
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

        # Cache and LRU tracking setup
        project_src = Path(__file__).resolve().parent.parent
        self._cache_root = (project_src / ".cache").resolve()
        (self._cache_root / "parsed").mkdir(parents=True, exist_ok=True)
        (self._cache_root / "pages").mkdir(parents=True, exist_ok=True)
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

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

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
        """Generate stable hashes for each page using rendering for visual stability."""
        hashes = []
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            doc = fitz.open(str(file_path))
            for page in doc:
                # Rendering to low-res pixmap is more "visual" and stable than raw stream
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                hashes.append(hashlib.sha256(pix.samples).hexdigest())
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

    def _process_page(self, page_output: dict, json_path: Path, md_path: Path) -> Document:
        """Build Document from saved page files."""
        with open(json_path, "r", encoding="utf-8") as f: json_data = json.load(f)
        with open(md_path, "r", encoding="utf-8") as f: md_data = f.read()

        page_index = (json_data.get("page_index") or 0) + 1
        chunks = [
            Chunk(
                chunk_markdown=data.get("block_content", ""),
                grounding=Grounding(
                    chunk_type=data.get("block_label", "unknown"),
                    bbox=data.get("block_bbox", [0, 0, 0, 0]),
                    score=item.get("score", 0.0),
                    page_index=page_index,
                ),
            )
            for data, item in zip(json_data.get("parsing_res_list", []), json_data.get("layout_det_res", {}).get("boxes", []))
        ]

        output_img = page_output.get("doc_preprocessor_res", {}).get("output_img")
        metadata = Metadata(
            filename=json_data.get("input_path", ""),
            page_image_base64=self._image_to_base64(output_img) if output_img is not None else "",
            page_index=page_index,
            page_count=json_data.get("page_count") or 1,
        )
        return Document(markdown=md_data, chunks=chunks, metadata=metadata)

    def parse(self, input_path: str, **kwargs) -> Tuple[List[Document], Path]:
        """Extract structured markdown/chunks with 3-tier optimization."""
        # Resolve placeholders and validate path
        resolved_path = resolve_placeholders(input_path)
        document_path = validate_path(resolved_path)
        if not document_path.exists(): raise FileNotFoundError(f"No file at {document_path}")

        file_hash = self._get_file_hash(document_path)
        self._update_lru(file_hash)

        cache_folder = (self._cache_root / "parsed" / file_hash).resolve()
        output_file = cache_folder / "documents.json"
        
        safe_stem = sanitize_stem(document_path.stem)
        project_src = Path(__file__).resolve().parent.parent
        # DO NOT .resolve() symlink_path here, as it would follow existing links
        symlink_path = project_src / safe_stem

        # --- Phase 1: Full Cache Check ---
        if output_file.exists():
            logger.info("Cache hit (Full): %s", file_hash[:8])
            self._ensure_symlink(symlink_path, cache_folder)
            with open(output_file, "r", encoding="utf-8") as f:
                return [Document.model_validate(d) for d in json.load(f)], output_file

        # --- Phase 3: Incremental Check ---
        page_hashes = self._get_page_hashes(document_path)
        pages: List[Optional[Document]] = [None] * len(page_hashes)
        missing_indices = []

        for i, ph in enumerate(page_hashes):
            p_path = (self._cache_root / "pages" / ph / "page.json").resolve()
            if p_path.exists():
                with open(p_path, "r", encoding="utf-8") as f:
                    doc = Document.model_validate(json.load(f))
                    # Normalize metadata for current document context
                    doc.metadata.filename = document_path.name
                    doc.metadata.page_index = i + 1
                    doc.metadata.page_count = len(page_hashes)
                    pages[i] = doc
            else:
                missing_indices.append(i)

        if not missing_indices:
            logger.info("Incremental hit: All pages cached.")
            documents = [p for p in pages if p is not None]
            self._persist_to_cache(cache_folder, documents, document_path, symlink_path)
            return documents, output_file

        # --- Partial Miss: Process Subset ---
        pipeline = self._get_pipeline()
        logger.info("Incremental parse: %d/%d pages missing.", len(missing_indices), len(page_hashes))

        with tempfile.NamedTemporaryFile(suffix=document_path.suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            if document_path.suffix.lower() == ".pdf":
                src = fitz.open(str(document_path))
                dst = fitz.open()
                dst.insert_pdf(src)
                dst.select(missing_indices)
                dst.save(str(tmp_path))
                src.close()
                dst.close()
            else:
                shutil.copy(document_path, tmp_path)

        try:
            raw_output = list(pipeline.predict(input=str(tmp_path)))
            n_workers = min(len(raw_output), os.cpu_count() or 4)
            with tempfile.TemporaryDirectory() as td:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for res in raw_output: ex.submit(self._save_page, res, td)
                
                stem = tmp_path.stem
                jsons = sorted(Path(td).glob(f"**/{stem}*.json"))
                mds = sorted(Path(td).glob(f"**/{stem}*.md"))
                
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    new_docs = [fut.result() for fut in [ex.submit(self._process_page, po, j, m) for po, j, m in zip(raw_output, jsons, mds)]]

            # Cache new pages and reassemble
            for idx, doc in zip(missing_indices, new_docs):
                ph = page_hashes[idx]
                p_dir = (self._cache_root / "pages" / ph).resolve()
                p_dir.mkdir(parents=True, exist_ok=True)
                
                doc.metadata.filename = document_path.name
                doc.metadata.page_index = idx + 1
                doc.metadata.page_count = len(page_hashes)
                
                with open(p_dir / "page.json", "w", encoding="utf-8") as f:
                    json.dump(doc.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
                pages[idx] = doc

        finally:
            if tmp_path.exists(): os.unlink(tmp_path)

        documents = [p for p in pages if p is not None]
        self._persist_to_cache(cache_folder, documents, document_path, symlink_path)
        return documents, output_file

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
        """Persist full document results and update symlink."""
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "documents.json", "w", encoding="utf-8") as f:
            json.dump([d.model_dump(mode="json") for d in docs], f, indent=2, ensure_ascii=False)
        for d in docs:
            md_name = f"{doc_path.stem}_{d.metadata.page_index}.md"
            with open(folder / md_name, "w", encoding="utf-8") as f:
                f.write(d.markdown)
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
