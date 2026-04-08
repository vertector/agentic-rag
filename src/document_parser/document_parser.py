"""
Document parser using PaddleOCRVL.

Concurrency model
-----------------
1. ThreadPoolExecutor — parallel page saves inside parse()
   Rationale: save_to_json / save_to_markdown are I/O-bound and independent
   per page. I/O releases the GIL, so threads actually run concurrently.

2. ThreadPoolExecutor — parallel per-page processing inside parse()
   Rationale: json.load uses a C extension (GIL released), PIL image ops
   release the GIL. Pydantic construction is fast; thread overhead is
   dominated by the I/O + C-ext work that precedes it.

3. ProcessPoolExecutor — parse_batch() for document-level parallelism.
   Rationale: each worker process owns one PaddleOCRVL instance, so the
   model is loaded once per process, not once per document. The pipeline
   is never pickled — only a plain settings dict crosses the boundary.

NOT async: PaddleOCR has no async API. Wrapping predict() in
asyncio.run_in_executor is just threading with extra boilerplate.

Thread safety
-------------
- _get_pipeline uses double-checked locking (threading.Lock).
- _process_page reads only immutable per-call inputs — no shared state.
- _save_page writes to uniquely named files in a per-call temp dir — no races.
"""

import base64
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from PIL import Image

from shared.schemas import Chunk, Document, Grounding, Metadata, PipelineSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers for ProcessPoolExecutor workers.
# Must be at module level to be picklable.
# ---------------------------------------------------------------------------

_worker_parser: Optional["DocumentParser"] = None


def _init_worker(settings_dict: dict) -> None:
    """Create one DocumentParser per worker process (called once at startup)."""
    global _worker_parser
    _worker_parser = DocumentParser(PipelineSettings(**settings_dict))


def _worker_parse(input_path: str) -> List[dict]:
    """
    Parse a single document inside a worker process.

    Returns JSON-safe dicts so no Document objects cross the process boundary
    (Pydantic models with UUID fields are not reliably picklable in all envs).
    """
    assert _worker_parser is not None, "Worker not initialised — _init_worker not called"
    return [doc.model_dump(mode="json") for doc in _worker_parser.parse(input_path)]


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------


class DocumentParser:
    """
    Production-ready document parser using PaddleOCRVL.

    Usage:
        >>> parser = DocumentParser()
        >>> pages = parser.parse("report.pdf")
        # Output written to ./report/documents.json

        >>> # Batch — launches N worker processes, each loads the model once.
        >>> all_pages = parser.parse_batch(["a.pdf", "b.pdf", "c.pdf"])

        >>> # Custom settings
        >>> settings = PipelineSettings(use_chart_recognition=False)
        >>> parser = DocumentParser(settings=settings)
    """

    __slots__ = ("_settings", "_pipeline", "_pipeline_lock")

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        self._settings = settings or PipelineSettings()
        self._pipeline = None
        # Lock guards lazy pipeline initialisation against concurrent callers.
        self._pipeline_lock = threading.Lock()

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

    def _get_pipeline(self):
        """
        Thread-safe singleton — ensures we only load the heavy VLM once.

        The outer `if` is a fast-path that skips lock acquisition once the
        pipeline is ready. The inner `if` prevents a double-init when two
        threads both pass the outer check before either acquires the lock.
        """
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    from paddleocr import PaddleOCRVL

                    self._pipeline = PaddleOCRVL(
                        **self._settings.to_init_kwargs(),
                    )
        return self._pipeline

    @staticmethod
    def _image_to_base64(img) -> str:
        """Convert a PIL Image or numpy array to a base64-encoded PNG string."""
        if not hasattr(img, "save"):
            img = Image.fromarray(img)
        with BytesIO() as buf:
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _save_page(res, temp_dir: str) -> None:
        """
        Write one page's JSON and markdown to temp_dir (called in a thread).

        PaddleOCR names output files after the input stem + page index,
        so concurrent writes to the same temp_dir do not collide.
        """
        res.save_to_json(save_path=temp_dir)
        res.save_to_markdown(save_path=temp_dir)

    def _process_page(
        self,
        page_output: dict,
        json_path: Path,
        md_path: Path,
    ) -> Document:
        """
        Build a Document from one page's saved files (called in a thread).

        Thread-safe: reads only immutable per-call arguments. json.load and
        PIL image ops both release the GIL, giving real concurrency gains.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        with open(md_path, "r", encoding="utf-8") as f:
            md_data = f.read()

        page_index = (json_data.get("page_index") or 0) + 1
        chunks_data = json_data.get("parsing_res_list", [])
        layout_boxes = json_data.get("layout_det_res", {}).get("boxes", [])

        chunks: List[Chunk] = [
            Chunk(
                chunk_markdown=data.get("block_content", ""),
                grounding=Grounding(
                    chunk_type=data.get("block_label", "unknown"),
                    bbox=data.get("block_bbox", [0, 0, 0, 0]),
                    score=item.get("score", 0.0),
                    page_index=page_index,
                ),
            )
            for data, item in zip(chunks_data, layout_boxes)
        ]

        output_img = page_output.get("doc_preprocessor_res", {}).get("output_img")

        metadata = Metadata(
            filename=json_data.get("input_path", ""),
            page_image_base64=(
                self._image_to_base64(output_img) if output_img is not None else ""
            ),
            page_index=page_index,
            page_count=json_data.get("page_count") or 1,
        )

        return Document(markdown=md_data, chunks=chunks, metadata=metadata)

    def parse(self, input_path: str, **kwargs) -> List[Document]:
        """
        Extract structured markdown and chunks from a document.

        Args:
            input_path: Path to the document (PDF, image, …).
            **kwargs: Overrides for inference (e.g. temperature, max_new_tokens).

        Returns:
            List of Document objects, one per page, in page order.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        document_path = Path(input_path)
        if not document_path.exists():
            raise FileNotFoundError(f"No file found at {document_path}")

        pipeline = self._get_pipeline()

        # Merge inference settings with any per-call overrides
        predict_args = {**self._settings.to_predict_kwargs(), **kwargs}

        # Run extraction
        raw_output = pipeline.predict(
            input=str(document_path),
            # **predict_args,
        )

        if document_path.suffix.lower() == ".pdf":
            output: List[dict] = pipeline.restructure_pages(
                list(raw_output),
                merge_tables=self._settings.merge_tables,
                relevel_titles=self._settings.relevel_titles,
            )
        else:
            output = list(raw_output)

        # Cap workers at page count — no benefit in more threads than pages.
        n_workers = min(len(output), os.cpu_count() or 4)

        with tempfile.TemporaryDirectory() as temp_dir:

            # --- Stage 1: save all pages in parallel (I/O-bound) ---
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                save_futures = [
                    executor.submit(self._save_page, res, temp_dir) for res in output
                ]
                # Iterate with as_completed to propagate exceptions immediately
                # (fail-fast) rather than waiting for all pages to finish.
                for fut in as_completed(save_futures):
                    fut.result()

            stem = document_path.stem
            json_files = sorted(Path(temp_dir).glob(f"**/{stem}*.json"))
            md_files = sorted(Path(temp_dir).glob(f"**/{stem}*.md"))

            # --- Stage 2: process all pages in parallel ---
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit in page order and collect in the same order — do NOT
                # use as_completed here, which would scramble the page sequence.
                page_futures = [
                    executor.submit(self._process_page, page_out, j, m)
                    for page_out, j, m in zip(output, json_files, md_files)
                ]
                documents: List[Document] = [fut.result() for fut in page_futures]

        # --- Stage 3: persist (single sequential write) ---
        output_folder = Path(document_path.stem)
        output_folder.mkdir(parents=True, exist_ok=True)

        with open(output_folder / "documents.json", "w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump(mode="json") for doc in documents],
                f,
                indent=2,
                ensure_ascii=False,
            )

        # --- Stage 3b: write standalone markdown files ---
        # JSON escapes newlines (\n → literal \\n), which makes the "markdown"
        # field inside documents.json unreadable when copy-pasted into a
        # viewer. These standalone .md files preserve real newlines and render
        # correctly in any markdown viewer.
        for doc in documents:
            md_filename = f"{document_path.stem}_{doc.metadata.page_index}.md"
            with open(output_folder / md_filename, "w", encoding="utf-8") as f:
                f.write(doc.markdown)

        return documents

    def parse_batch(
        self,
        input_paths: List[str],
        max_workers: Optional[int] = None,
    ) -> List[List[Document]]:
        """
        Parse multiple documents in parallel using separate worker processes.

        Each worker process calls _init_worker() once on startup, creating its
        own PaddleOCRVL pipeline. The pipeline is never pickled — only a plain
        settings dict and file path strings cross the process boundary.

        Use this when processing N independent documents. For a single document
        with many pages, parse() already parallelises at the page level.

        Args:
            input_paths: Paths to the documents to parse.
            max_workers: Worker process count. Defaults to
                         min(len(input_paths), os.cpu_count()).

        Returns:
            List of per-document Document lists, in the same order as input_paths.
        """
        if not input_paths:
            return []

        n_workers = max_workers or min(len(input_paths), os.cpu_count() or 1)
        settings_dict = self._settings.model_dump()

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(settings_dict,),
        ) as pool:
            # map() preserves submission order — results align with input_paths.
            serialized: List[List[dict]] = list(pool.map(_worker_parse, input_paths))

        # Reconstruct Document objects. model_validate handles UUID coercion
        # from the string representation produced by model_dump(mode="json").
        return [
            [Document.model_validate(d) for d in docs]
            for docs in serialized
        ]
