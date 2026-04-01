"""
document_parser_mcp — FastMCP server wrapping DocumentParser + PaddleOCRVL.

Transport:  stdio  (registered in claude_desktop_config.json)
Threading:  DocumentParser.parse() is CPU/IO-bound and has no async API.
            All blocking calls are dispatched through asyncio.run_in_executor
            so the MCP event loop stays responsive between tool calls.

File delivery:
  Clients may supply a local file path (fastest, no overhead) OR
  base64-encoded file bytes + a filename (for in-memory / remote workflows).
  When base64 is given a NamedTemporaryFile is created, parsed, then removed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys

# Ensure the parent directory (rag_engine root) is in sys.path so that
# absolute imports like 'shared' and 'document_parser' resolve properly when 
# this script is executed directly via MCP.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from shared.schemas import PipelineSettings, Document
from document_parser.document_parser import DocumentParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_NAME = "document_parser_mcp"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MAX_BATCH_SIZE = 16          # safety cap for parse_batch
MAX_BASE64_MB  = 50          # reject payloads larger than this

logger = logging.getLogger(SERVER_NAME)

# ---------------------------------------------------------------------------
# Server + shared parser instance
# ---------------------------------------------------------------------------

mcp = FastMCP(SERVER_NAME)

# Shared parser — lazily initialises PaddleOCRVL on first call.
# Settings can be reconfigured via configure_parser.
_parser: DocumentParser = DocumentParser()


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class ParseDocumentInput(BaseModel):
    """Input model for parse_document."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # --- file delivery (exactly one must be provided) ---
    file_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute or relative path to the document on the server's local filesystem. "
            "Supported formats: PDF, PNG, JPEG, TIFF, BMP, WEBP. "
            "Example: '/home/user/docs/report.pdf'"
        ),
    )
    file_content_base64: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded bytes of the document. Use when the file is not already "
            "on disk (e.g. uploaded via the client). Must be paired with `filename`."
        ),
    )
    filename: Optional[str] = Field(
        default=None,
        description=(
            "Original filename including extension (e.g. 'invoice.pdf'). "
            "Required when `file_content_base64` is provided so the parser can "
            "determine the file type."
        ),
    )

    # --- pipeline flags (all optional, override server defaults) ---
    use_ocr_for_image_block: Optional[bool] = Field(
        default=None, description="Apply OCR to image blocks (default: True)."
    )
    use_doc_orientation_classify: Optional[bool] = Field(
        default=None, description="Auto-correct document orientation (default: True)."
    )
    use_doc_unwarping: Optional[bool] = Field(
        default=None, description="Dewarp skewed/curved documents (default: False)."
    )
    use_chart_recognition: Optional[bool] = Field(
        default=None, description="Extract charts and graphs (default: True)."
    )
    use_layout_detection: Optional[bool] = Field(
        default=None, description="Detect document layout structure (default: True)."
    )
    use_seal_recognition: Optional[bool] = Field(
        default=None, description="Detect seals and stamps (default: True)."
    )
    format_block_content: Optional[bool] = Field(
        default=None, description="Format extracted block content (default: True)."
    )
    merge_layout_blocks: Optional[bool] = Field(
        default=None, description="Merge adjacent same-type layout blocks (default: True)."
    )
    markdown_ignore_labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Layout labels to omit from Markdown output, e.g. ['header', 'footer']. "
            "Default: [] (nothing ignored)."
        ),
    )
    pipeline_version: Optional[str] = Field(
        default=None, description="PaddleOCR-VL pipeline version ('v1' or 'v1.5')."
    )
    layout_threshold: Optional[float] = Field(
        default=None, description="Score threshold for layout detection (e.g. 0.45)."
    )
    layout_nms: Optional[bool] = Field(
        default=None, description="Use NMS in layout detection."
    )
    layout_unclip_ratio: Optional[float] = Field(
        default=None, description="Expansion coefficient for layout boxes."
    )
    layout_merge_bboxes_mode: Optional[str] = Field(
        default=None, description="Method for filtering overlapping boxes ('overlap')."
    )
    layout_shape_mode: Optional[str] = Field(
        default=None, description="Mode for handling layout shapes ('auto')."
    )
    prompt_label: Optional[str] = Field(
        default=None, description="Custom prompt label for the VLM inference."
    )
    repetition_penalty: Optional[float] = Field(
        default=None, description="Repetition penalty for VLM sampling."
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature for VLM sampling."
    )
    top_p: Optional[float] = Field(
        default=None, description="Top-p sampling for VLM inference."
    )
    min_pixels: Optional[int] = Field(
        default=None, description="Minimum pixels for VLM preprocessing."
    )
    max_pixels: Optional[int] = Field(
        default=None, description="Maximum pixels for VLM preprocessing."
    )
    max_new_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens generated by the VLM per block."
    )
    use_queues: Optional[bool] = Field(
        default=None, description="Use internal queues for processing."
    )
    include_page_images: bool = Field(
        default=False,
        description=(
            "Include base64-encoded page images in the response. "
            "Disabled by default to keep response size manageable."
        ),
    )

    @model_validator(mode="after")
    def _check_delivery_mode(self) -> "ParseDocumentInput":
        has_path   = self.file_path is not None
        has_b64    = self.file_content_base64 is not None
        has_fname  = self.filename is not None

        if not has_path and not has_b64:
            raise ValueError("Provide either `file_path` or `file_content_base64` + `filename`.")
        if has_path and has_b64:
            raise ValueError("`file_path` and `file_content_base64` are mutually exclusive.")
        if has_b64 and not has_fname:
            raise ValueError("`filename` is required when `file_content_base64` is provided.")
        return self

    @field_validator("file_path")
    @classmethod
    def _validate_extension_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        ext = Path(v).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return v

    @field_validator("filename")
    @classmethod
    def _validate_extension_filename(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        ext = Path(v).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return v


class ParseBatchInput(BaseModel):
    """Input model for parse_batch."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    file_paths: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=(
            "List of absolute/relative paths to documents on the server filesystem. "
            f"Maximum {MAX_BATCH_SIZE} documents per call."
        ),
    )
    max_workers: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description=(
            "Number of parallel worker processes. Defaults to min(len(file_paths), cpu_count). "
            "Each worker loads a separate PaddleOCRVL instance."
        ),
    )
    include_page_images: bool = Field(
        default=False,
        description="Include base64-encoded page images in the response (default: False).",
    )

    @field_validator("file_paths")
    @classmethod
    def _validate_paths(cls, v: List[str]) -> List[str]:
        for path in v:
            ext = Path(path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type '{ext}' in path '{path}'. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
        return v


class ConfigureParserInput(BaseModel):
    """Input model for configure_parser."""

    model_config = ConfigDict(extra="forbid")

    vl_rec_api_model_name: Optional[str] = Field(
        default=None,
        description="Vision-language model name (e.g. 'PaddlePaddle/PaddleOCR-VL-1.5').",
    )
    vl_rec_backend: Optional[str] = Field(
        default=None,
        description=(
            "Inference backend: 'vllm-server', 'mlx-vlm-server', 'sglang-server', "
            "'fastdeploy-server', or 'local'. Defaults to platform-appropriate backend."
        ),
    )
    vl_rec_server_url: Optional[str] = Field(
        default=None,
        description="Base URL of the VLM server (/v1 OpenAI-compatible endpoint).",
    )
    vl_rec_api_key: Optional[str] = Field(
        default=None,
        description="API key for the VLM server (if authentication is required).",
    )
    use_ocr_for_image_block: Optional[bool] = Field(default=None, description="OCR on image blocks.")
    use_doc_orientation_classify: Optional[bool] = Field(default=None, description="Orientation correction.")
    use_doc_unwarping: Optional[bool] = Field(default=None, description="Dewarp documents.")
    use_chart_recognition: Optional[bool] = Field(default=None, description="Chart recognition.")
    use_layout_detection: Optional[bool] = Field(default=None, description="Layout detection.")
    use_seal_recognition: Optional[bool] = Field(default=None, description="Seal/stamp recognition.")
    format_block_content: Optional[bool] = Field(default=None, description="Format block content.")
    merge_layout_blocks: Optional[bool] = Field(default=None, description="Merge adjacent blocks.")
    markdown_ignore_labels: Optional[List[str]] = Field(
        default=None, description="Labels to exclude from Markdown output."
    )
    pipeline_version: Optional[str] = Field(
        default=None, description="Pipeline version ('v1' or 'v1.5')."
    )
    layout_threshold: Optional[float] = Field(default=None)
    layout_nms: Optional[bool] = Field(default=None)
    layout_unclip_ratio: Optional[float] = Field(default=None)
    layout_merge_bboxes_mode: Optional[str] = Field(default=None)
    layout_shape_mode: Optional[str] = Field(default=None)
    prompt_label: Optional[str] = Field(default=None)
    repetition_penalty: Optional[float] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    min_pixels: Optional[int] = Field(default=None)
    max_pixels: Optional[int] = Field(default=None)
    max_new_tokens: Optional[int] = Field(default=None)
    use_queues: Optional[bool] = Field(default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_settings_override(params: ParseDocumentInput) -> Optional[PipelineSettings]:
    """
    Build a per-call PipelineSettings only when the caller supplied overrides.
    Returns None if no overrides were given (parser uses its current settings).
    """
    override_fields = {
        "use_ocr_for_image_block",
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_chart_recognition",
        "use_layout_detection",
        "use_seal_recognition",
        "format_block_content",
        "merge_layout_blocks",
        "markdown_ignore_labels",
        "pipeline_version",
        "layout_threshold",
        "layout_nms",
        "layout_unclip_ratio",
        "layout_merge_bboxes_mode",
        "layout_shape_mode",
        "prompt_label",
        "repetition_penalty",
        "temperature",
        "top_p",
        "min_pixels",
        "max_pixels",
        "max_new_tokens",
        "use_queues",
    }
    overrides = {
        f: getattr(params, f)
        for f in override_fields
        if getattr(params, f) is not None
    }
    if not overrides:
        return None

    base = _parser.settings.model_dump()
    base.update(overrides)
    return PipelineSettings(**base)


def _serialise_documents(
    documents: List[Document],
    include_images: bool,
) -> List[dict]:
    """Convert Document objects to JSON-safe dicts, optionally stripping images."""
    result = []
    for doc in documents:
        d = doc.model_dump(mode="json")
        if not include_images:
            d.get("metadata", {}).pop("page_image_base64", None)
        result.append(d)
    return result


def _write_base64_to_tempfile(b64_content: str, filename: str) -> str:
    """
    Decode base64 content and write to a NamedTemporaryFile.

    Returns the temp file path. Caller is responsible for cleanup.
    Raises ValueError if the payload exceeds MAX_BASE64_MB.
    """
    raw = base64.b64decode(b64_content)
    mb = len(raw) / (1024 * 1024)
    if mb > MAX_BASE64_MB:
        raise ValueError(
            f"Payload is {mb:.1f} MB — exceeds the {MAX_BASE64_MB} MB limit. "
            "Save the file to disk and use `file_path` instead."
        )

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        return tmp.name


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="parse_document",
    annotations={
        "title": "Parse a Single Document",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parse_document(
    ctx: Context,
    file_path: Optional[str] = None,
    file_content_base64: Optional[str] = None,
    filename: Optional[str] = None,
    use_ocr_for_image_block: Optional[bool] = None,
    use_doc_orientation_classify: Optional[bool] = None,
    use_doc_unwarping: Optional[bool] = None,
    use_chart_recognition: Optional[bool] = None,
    use_layout_detection: Optional[bool] = None,
    use_seal_recognition: Optional[bool] = None,
    format_block_content: Optional[bool] = None,
    merge_layout_blocks: Optional[bool] = None,
    markdown_ignore_labels: Optional[List[str]] = None,
    pipeline_version: Optional[str] = None,
    layout_threshold: Optional[float] = None,
    layout_nms: Optional[bool] = None,
    layout_unclip_ratio: Optional[float] = None,
    layout_merge_bboxes_mode: Optional[str] = None,
    layout_shape_mode: Optional[str] = None,
    prompt_label: Optional[str] = None,
    repetition_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    use_queues: Optional[bool] = None,
    include_page_images: bool = False,
) -> str:
    """
    Parse a single document (PDF or image) using PaddleOCRVL.

    Provide EITHER file_path (local path) OR file_content_base64 + filename (not both).
    Returns a JSON array of per-page Document objects with markdown, chunks, and metadata.

    Args:
        file_path: Absolute or relative path to the document on the server filesystem.
                   Supported: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp
        file_content_base64: Base64-encoded file bytes. Only when file is NOT on disk.
                             Must be paired with `filename`.
        filename: Original filename with extension (e.g. 'report.pdf'). Required with base64.
        include_page_images: Include base64 page images in response (default: False).
        use_ocr_for_image_block: Apply OCR to image blocks.
        use_chart_recognition: Extract charts and graphs.
        use_layout_detection: Detect document layout structure.
    """
    logger.info("Preparing file...")

    tmp_path: Optional[str] = None
    parse_path: str

    try:
        # --- Validate delivery mode (was Pydantic model_validator) -------------
        if file_path is None and file_content_base64 is None:
            return json.dumps({
                "error": "InvalidInput",
                "message": "Provide either `file_path` or `file_content_base64` + `filename`.",
            })
        if file_path is not None and file_content_base64 is not None:
            return json.dumps({
                "error": "InvalidInput",
                "message": "`file_path` and `file_content_base64` are mutually exclusive.",
            })
        if file_content_base64 is not None and not filename:
            return json.dumps({
                "error": "InvalidInput",
                "message": "`filename` is required when `file_content_base64` is provided.",
            })

        # --- Resolve input file ------------------------------------------------
        if file_path is not None:
            parse_path = file_path
            if not Path(parse_path).exists():
                return json.dumps({
                    "error": "FileNotFoundError",
                    "message": f"No file at '{parse_path}'. Check the path and try again.",
                })
        else:
            logger.info("Decoding base64 content...")
            try:
                tmp_path = _write_base64_to_tempfile(
                    file_content_base64,  # type: ignore[arg-type]
                    filename,             # type: ignore[arg-type]
                )
            except (ValueError, Exception) as exc:
                return json.dumps({
                    "error": "Base64DecodeError",
                    "message": str(exc),
                })
            parse_path = tmp_path

        # --- Build per-call settings (if any overrides) ------------------------
        # Reconstruct Pydantic model for _build_settings_override reuse
        params = ParseDocumentInput(
            file_path=file_path,
            file_content_base64=file_content_base64,
            filename=filename,
            use_ocr_for_image_block=use_ocr_for_image_block,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
            use_layout_detection=use_layout_detection,
            use_seal_recognition=use_seal_recognition,
            format_block_content=format_block_content,
            merge_layout_blocks=merge_layout_blocks,
            markdown_ignore_labels=markdown_ignore_labels,
            pipeline_version=pipeline_version,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            prompt_label=prompt_label,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            use_queues=use_queues,
            include_page_images=include_page_images,
        )
        settings_override = _build_settings_override(params)
        active_parser = (
            DocumentParser(settings_override) if settings_override else _parser
        )

        # --- Run parse in thread (blocking call, preserves event loop) ---------
        logger.info("Loading pipeline (first call may take ~30s)...")

        loop = asyncio.get_running_loop()
        documents: List[Document] = await loop.run_in_executor(
            None,
            active_parser.parse,
            parse_path,
        )

        logger.info("Serialising results...")
        serialised = _serialise_documents(documents, include_page_images)

        # Attach Merkle roots (content fingerprints) to each page
        for doc_obj, doc_dict in zip(documents, serialised):
            doc_dict["merkle_root"] = doc_obj.get_merkle_root()

        logger.info("Done.")
        return json.dumps(serialised, indent=2, ensure_ascii=False)

    except FileNotFoundError as exc:
        logger.error(f"parse_document FileNotFoundError: {exc}")
        return json.dumps({
            "error": "FileNotFoundError",
            "message": str(exc),
            "suggestion": "Verify the path is correct and the file exists on the server.",
        })
    except PermissionError as exc:
        logger.error(f"parse_document PermissionError: {exc}")
        return json.dumps({
            "error": "PermissionError",
            "message": str(exc),
            "suggestion": "Check that the server process has read access to the file.",
        })
    except Exception as exc:
        logger.error(f"parse_document unexpected error: {type(exc).__name__}: {exc}")
        return json.dumps({
            "error": type(exc).__name__,
            "message": str(exc),
            "suggestion": (
                "Check server logs for the full traceback. "
                "Common causes: unsupported file format, corrupted file, "
                "VLM server unreachable."
            ),
        })
    finally:
        # Always clean up the temp file even on exceptions.
        if tmp_path and Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@mcp.tool(
    name="parse_batch",
    annotations={
        "title": "Parse Multiple Documents in Parallel",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parse_batch(
    ctx: Context,
    file_paths: List[str],
    max_workers: Optional[int] = None,
    include_page_images: bool = False,
) -> str:
    """
    Parse multiple documents in parallel using separate worker processes.

    Args:
        file_paths: List of absolute/relative paths to documents (1–16 files).
        max_workers: Number of parallel workers (default: auto based on CPU count).
        include_page_images: Include base64 page images in response (default: False).

    Returns:
        str: JSON array of arrays — outer index matches input order, inner array
             contains per-page Document dicts. Returns a JSON error on failure.
    """
    if not file_paths:
        return json.dumps({"error": "InvalidInput", "message": "`file_paths` must not be empty."})
    if len(file_paths) > MAX_BATCH_SIZE:
        return json.dumps({"error": "InvalidInput", "message": f"Maximum {MAX_BATCH_SIZE} files per batch."})

    logger.info(f"Queuing {len(file_paths)} document(s)...")

    # Validate all paths before launching workers
    missing = [p for p in file_paths if not Path(p).exists()]
    if missing:
        return json.dumps({
            "error": "FileNotFoundError",
            "message": f"{len(missing)} file(s) not found.",
            "missing_paths": missing,
            "suggestion": "Correct the paths and retry.",
        })

    try:
        logger.info("Launching worker processes...")
        loop = asyncio.get_running_loop()

        all_docs: List[List[Document]] = await loop.run_in_executor(
            None,
            lambda: _parser.parse_batch(
                file_paths,
                max_workers=max_workers,
            ),
        )

        logger.info("Serialising results...")
        result = [
            _serialise_documents(docs, include_page_images)
            for docs in all_docs
        ]

        logger.info("Done.")
        return json.dumps(result, indent=2, ensure_ascii=False)

    except FileNotFoundError as exc:
        logger.error(f"parse_batch FileNotFoundError: {exc}")
        return json.dumps({"error": "FileNotFoundError", "message": str(exc)})
    except Exception as exc:
        logger.error(f"parse_batch error: {type(exc).__name__}: {exc}")
        return json.dumps({
            "error": type(exc).__name__,
            "message": str(exc),
            "suggestion": (
                "Check server logs. Common causes: unsupported format, corrupted files, "
                "insufficient system memory for the requested worker count."
            ),
        })


@mcp.tool(
    name="configure_parser",
    annotations={
        "title": "Update Server-Level Parser Settings",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def configure_parser(params: ConfigureParserInput) -> str:
    """
    Update the server-level PipelineSettings used by all subsequent parse calls.

    Only the fields you supply are changed; unspecified fields keep their
    current values. Changing settings resets the lazy-loaded PaddleOCRVL
    pipeline (it will be re-initialised on the next parse call).

    Note: settings supplied directly to `parse_document` override
    these server-level values for that single call only.

    Args:
        params (ConfigureParserInput): Fields to update — see field descriptions.

    Returns:
        str: JSON object confirming the applied settings.
    """
    global _parser

    current = _parser.settings.model_dump()
    updates = {k: v for k, v in params.model_dump().items() if v is not None}

    if not updates:
        return json.dumps({
            "message": "No changes — all fields were None.",
            "current_settings": current,
        })

    current.update(updates)
    try:
        new_settings = PipelineSettings(**current)
        _parser = DocumentParser(new_settings)
        return json.dumps({
            "message": "Settings updated. Pipeline will be re-initialised on next parse call.",
            "applied_updates": updates,
            "current_settings": new_settings.model_dump(),
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "error": type(exc).__name__,
            "message": str(exc),
            "suggestion": "Check field values against PipelineSettings constraints.",
        })


@mcp.tool(
    name="get_parser_settings",
    annotations={
        "title": "Get Current Parser Settings",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_parser_settings() -> str:
    """
    Return the current server-level PipelineSettings as a JSON object.

    Useful for inspecting what settings will be applied to the next
    parse call before invoking `parse_document`.

    Returns:
        str: JSON object of current PipelineSettings fields.
    """
    return json.dumps(_parser.settings.model_dump(), indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mcp.run()  # stdio transport — Claude Desktop communicates over stdin/stdout