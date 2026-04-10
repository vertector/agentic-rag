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
from typing import Any, Dict, List, Optional

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
from shared.env_loader import load_env

# Load environment variables (api keys, server URLs) if present
load_env()

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

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore", populate_by_name=True)

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
    file_name: Optional[str] = Field(
        default=None,
        description="Legacy alias for file_path. Handled automatically.",
    )
    vl_rec_api_model_name: Optional[str] = Field(
        default=None,
        description="Vision-language model name (e.g. 'PaddlePaddle/PaddleOCR-VL-1.5').",
    )
    vl_rec_backend: Optional[str] = Field(
        default=None,
        description="Optional backend override for this specific parse call.",
    )
    vl_rec_server_url: Optional[str] = Field(
        default=None,
        description="Base URL of the VLM server (/v1 OpenAI-compatible endpoint).",
    )
    vl_rec_api_key: Optional[str] = Field(
        default=None,
        description="API key for the VLM server if authentication is required.",
    )
    use_doc_preprocessor: Optional[bool] = Field(
        default=None,
        description="Legacy flag, safely ignored or passed as extra arg.",
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
        default=None, description="Extract charts and graphs (default: False)."
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
    merge_tables: Optional[bool] = Field(
        default=None, description="Merge table content across page boundaries (default: True)."
    )
    relevel_titles: Optional[bool] = Field(
        default=None, description="Automatically adjust paragraph title levels (default: True)."
    )
    markdown_ignore_labels: List[str] = Field(
        default_factory=list,
        description=(
            "Layout labels to omit from Markdown output, e.g. ['header', 'footer']. "
            "Default: [] (nothing ignored)."
        ),
    )
    vlm_extra_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generic bucket for backend-specific MLLM parameters."
    )
    pipeline_version: Optional[str] = Field(
        default=None, description="PaddleOCR-VL pipeline version ('v1' or 'v1.5')."
    )

    # --- layout tuning ---
    layout_threshold: Optional[float] = Field(
        default=None, description="Score threshold for layout detection proposals (default: 0.3)."
    )
    layout_nms: Optional[bool] = Field(
        default=None, description="Enable Non-Maximum Suppression for layout boxes (default: True)."
    )
    layout_unclip_ratio: Optional[float] = Field(
        default=None, description="Expansion coefficient for detected layout boxes."
    )
    layout_merge_bboxes_mode: Optional[str] = Field(
        default=None, description="Strategy for merging overlapping bounding boxes ('union' or 'large')."
    )
    layout_shape_mode: Optional[str] = Field(
        default=None, description="Geometric shape of layout boxes: 'auto', 'rectangle', etc."
    )

    # --- VLM inference tuning ---
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature for VLM (default: 0.0 = deterministic)."
    )
    top_p: Optional[float] = Field(
        default=None, description="Top-p nucleus sampling threshold (default: 1.0)."
    )
    max_new_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens generated by the VLM per block (default: 4096)."
    )
    repetition_penalty: Optional[float] = Field(
        default=None, description="Repetition penalty for VLM sampling (default: 1.0)."
    )
    prompt_label: Optional[str] = Field(
        default=None,
        description=(
            "Global VLM task prompt override. Leave None (default) to let PaddleOCRVL "
            "route each block automatically. WARNING: setting this forces ALL blocks through "
            "one prompt, breaking table structure recognition."
        ),
    )

    # --- pixel limits ---
    min_pixels: Optional[int] = Field(
        default=None, description="Minimum total pixels for VLM input image (default: 147,384)."
    )
    max_pixels: Optional[int] = Field(
        default=None, description="Maximum total pixels for VLM input image (default: 8,699,840)."
    )

    # --- response options ---
    include_page_images: bool = Field(
        default=False,
        description=(
            "Include base64-encoded page images in the response. "
            "Disabled by default to keep response size manageable."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: dict) -> dict:
        file_name_val = data.get("file_name")
        file_path_val = data.get("file_path")
        
        # Only assign file_name to file_path if file_path wasn't explicitly provided
        # and base64 isn't being used (if base64 is used, file_name is just the filename)
        if file_name_val and not file_path_val and not data.get("file_content_base64"):
            file_path_val = file_name_val
            
        if file_path_val:
            # Map resolving the absolute path via pathlib as requested
            data["file_path"] = str(Path(file_path_val).resolve())
            
        return data

    @model_validator(mode="after")
    def _check_delivery_mode(self) -> "ParseDocumentInput":
        has_path  = self.file_path is not None
        has_b64   = self.file_content_base64 is not None
        has_fname = self.filename is not None

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

    # --- backend ---
    vl_rec_api_model_name: Optional[str] = Field(
        default=None,
        description="Vision-language model name (e.g. 'PaddlePaddle/PaddleOCR-VL-1.5').",
    )
    vl_rec_backend: Optional[str] = Field(
        default=None,
        description=(
            "Inference backend: 'vllm-server', 'mlx-vlm-server', 'sglang-server', "
            "'fastdeploy-server', 'local', or 'native'. Defaults to platform-appropriate backend."
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

    # --- core flags ---
    use_ocr_for_image_block: Optional[bool] = Field(default=None, description="OCR on image blocks.")
    use_doc_orientation_classify: Optional[bool] = Field(default=None, description="Orientation correction.")
    use_doc_unwarping: Optional[bool] = Field(default=None, description="Dewarp documents.")
    use_chart_recognition: Optional[bool] = Field(default=None, description="Chart recognition.")
    use_layout_detection: Optional[bool] = Field(default=None, description="Layout detection.")
    use_seal_recognition: Optional[bool] = Field(default=None, description="Seal/stamp recognition.")
    format_block_content: Optional[bool] = Field(default=None, description="Format block content.")
    merge_layout_blocks: Optional[bool] = Field(default=None, description="Merge adjacent blocks.")
    merge_tables: Optional[bool] = Field(default=None, description="Merge tables across page boundaries.")
    relevel_titles: Optional[bool] = Field(default=None, description="Auto-adjust paragraph title levels.")
    markdown_ignore_labels: Optional[List[str]] = Field(
        default=None, description="Labels to exclude from Markdown output."
    )
    pipeline_version: Optional[str] = Field(
        default=None, description="Pipeline version ('v1' or 'v1.5')."
    )
    use_doc_preprocessor: Optional[bool] = Field(
        default=None, description="Legacy flag for document preprocessor."
    )

    # --- layout tuning ---
    layout_threshold: Optional[float] = Field(default=None, description="Layout detection score threshold.")
    layout_nms: Optional[bool] = Field(default=None, description="Enable NMS for layout boxes.")
    layout_unclip_ratio: Optional[float] = Field(default=None, description="Expansion ratio for layout boxes.")
    layout_merge_bboxes_mode: Optional[str] = Field(default=None, description="Bounding box merge strategy.")
    layout_shape_mode: Optional[str] = Field(default=None, description="Layout box shape mode.")

    # --- VLM inference tuning ---
    temperature: Optional[float] = Field(default=None, description="VLM sampling temperature.")
    top_p: Optional[float] = Field(default=None, description="VLM top-p sampling threshold.")
    max_new_tokens: Optional[int] = Field(default=None, description="Max tokens generated per block.")
    repetition_penalty: Optional[float] = Field(default=None, description="VLM repetition penalty.")
    prompt_label: Optional[str] = Field(default=None, description="Global VLM task prompt override.")

    # --- pixel limits ---
    min_pixels: Optional[int] = Field(default=None, description="Minimum pixels for VLM input image.")
    max_pixels: Optional[int] = Field(default=None, description="Maximum pixels for VLM input image.")

    # --- extra args ---
    vlm_extra_args: Optional[Dict[str, Any]] = Field(
        default=None, description="Generic bucket for backend-specific parameters not covered above."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All PipelineSettings fields that can be overridden on a per-call basis.
_OVERRIDE_FIELDS = {
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
    "min_pixels",
    "max_pixels",
    "vlm_extra_args",
    "vl_rec_api_model_name",
    "vl_rec_backend",
    "vl_rec_server_url",
    "vl_rec_api_key",
    "use_doc_preprocessor",
}


def _build_settings_override(params: ParseDocumentInput) -> Optional[PipelineSettings]:
    """
    Build a per-call PipelineSettings only when the caller supplied overrides.
    Returns None if no overrides were given (parser uses its current settings).
    """
    overrides = {
        f: getattr(params, f)
        for f in _OVERRIDE_FIELDS
        if getattr(params, f, None) is not None
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
    params: ParseDocumentInput,
) -> str:
    """
    Parse a single document (PDF or image) using PaddleOCRVL.

    Provide EITHER file_path (local path) OR file_content_base64 + filename (not both).
    Returns a JSON array of per-page Document objects with markdown, chunks, and metadata.

    Args:
        params (ParseDocumentInput): Encapsulated JSON object containing all settings and delivery methods.
    """
    logger.info("Preparing file...")

    tmp_path: Optional[str] = None
    parse_path: str

    try:
        # --- Resolve input file -----------------------------------------------
        if params.file_path is not None:
            parse_path = params.file_path
            if not Path(parse_path).exists():
                return json.dumps({
                    "error": "FileNotFoundError",
                    "message": f"No file at '{parse_path}'. Check the path and try again.",
                })
        else:
            logger.info("Decoding base64 content...")
            try:
                tmp_path = _write_base64_to_tempfile(
                    params.file_content_base64,  # type: ignore[arg-type]
                    params.filename,             # type: ignore[arg-type]
                )
            except (ValueError, Exception) as exc:
                return json.dumps({
                    "error": "Base64DecodeError",
                    "message": str(exc),
                })
            parse_path = tmp_path

        # --- Build per-call settings (if any overrides) ----------------------
        settings_override = _build_settings_override(params)
        active_parser = (
            DocumentParser(settings_override) if settings_override else _parser
        )

        # --- Run parse in thread (blocking call, preserves event loop) -------
        logger.info("Loading pipeline (first call may take ~30s)...")

        loop = asyncio.get_running_loop()
        documents: List[Document] = await loop.run_in_executor(
            None,
            active_parser.parse,
            parse_path,
        )

        logger.info("Serialising results...")
        serialised = _serialise_documents(documents, params.include_page_images)

        # Attach Merkle roots (content fingerprints) to each page
        for doc_obj, doc_dict in zip(documents, serialised):
            doc_dict["merkle_root"] = doc_obj.get_merkle_root()

        # Calculate absolute output path to return to orchestrator
        project_src = Path(__file__).resolve().parent.parent
        output_folder = (project_src / Path(parse_path).stem).resolve()
        output_path = str(output_folder / "documents.json")

        logger.info(f"Done. Output saved to {output_path}")
        return json.dumps({
            "documents": serialised,
            "output_path": output_path
        }, indent=2, ensure_ascii=False)

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