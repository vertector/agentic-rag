import os
import sys
import hashlib
import json
import uuid
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BackendType = Literal[
    "vllm-server",
    "mlx-vlm-server",
    "sglang-server",
    "fastdeploy-server",
    "local",
    "native",
]

def _default_backend() -> BackendType:
    return "mlx-vlm-server" if sys.platform == "darwin" else "vllm-server"

def _default_server_url(backend: BackendType) -> str:
    return {
        "vllm-server":       "http://paddleocr_vl_merkle:8080/v1",
        "mlx-vlm-server":    "http://localhost:8080/v1",
        "sglang-server":     "http://localhost:8080/v1",
        "fastdeploy-server": "http://localhost:8080/v1",
        "local":             "",
        "native":            "",
    }[backend]

# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(strict=False, arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

class Grounding(_Base):
    chunk_type: str = Field(
        default="unknown",
        description="Layout block type (e.g. 'paragraph', 'table', 'image')",
    )
    bbox: List[int] = Field(
        description="Bounding box [x1, y1, x2, y2] in PDF coordinate space",
    )
    page_index: int = Field(
        ge=0,
        description="Zero-indexed page number where this chunk appears",
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Layout detection confidence score",
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, v: Any) -> List[int]:
        """Coerce floats (PaddleOCR), reject wrong length or negatives."""
        if len(v) != 4:
            raise ValueError(f"bbox must have exactly 4 coordinates, got {len(v)}")
        coerced = [int(c) for c in v]
        if any(c < 0 for c in coerced):
            raise ValueError(f"bbox coordinates must be non-negative, got {coerced}")
        return coerced


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(_Base):
    chunk_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this chunk",
    )
    chunk_markdown: str = Field(
        default="",
        description="Markdown-formatted content of the chunk",
    )
    grounding: Grounding = Field(
        description="Spatial grounding information for this chunk",
    )

    def get_content_hash(self) -> str:
        """
        Deterministic SHA-256 leaf hash for Merkle tree construction.

        Intentionally excludes:
          · chunk_id   — random UUID, not content
          · score      — detection confidence, not content
          · page_image_base64 — rendering artifact, not content
        """
        payload = {
            "m": self.chunk_markdown,
            "t": self.grounding.chunk_type,
            "b": self.grounding.bbox,
            "p": self.grounding.page_index,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class Metadata(_Base):
    filename: str = Field(
        description="Original input file path",
    )
    page_index: int = Field(
        ge=0,
        description="Zero-indexed page number",
    )
    page_count: int = Field(
        ge=1,
        description="Total number of pages in the document",
    )
    category: str = Field(
        default="general",
        description=(
            "Document domain used as a Qdrant filter key. "
            "Examples: 'legal', 'medical', 'finance'. "
            "Has no effect on Merkle hashing — purely a routing label."
        ),
    )
    page_image_base64: str = Field(
        default="",
        description=(
            "Base64-encoded PNG of the page. Populated by the parser; "
            "excluded from Merkle hashing and not written to Qdrant."
        ),
    )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def build_merkle_tree(hashes: List[str]) -> str:
    """
    Deterministic binary Merkle tree from an ordered list of leaf hashes.

    Order is strictly preserved — it encodes document sequence.
    Odd-length levels duplicate the last node (Bitcoin-style).
    """
    if not hashes:
        return hashlib.sha256(b"empty_node").hexdigest()

    current_level = list(hashes)
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
            next_level.append(
                hashlib.sha256((left + right).encode("utf-8")).hexdigest()
            )
        current_level = next_level

    return current_level[0]


class Document(_Base):
    doc_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this document page",
    )
    markdown: str = Field(
        default="",
        description="Full markdown content of the page",
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="Content chunks extracted from this page",
    )
    metadata: Metadata = Field(
        description="Page metadata",
    )

    def get_merkle_root(self) -> str:
        """
        Compute the Merkle root for this document's current chunk sequence.

        Falls back to a hash of the raw markdown if no chunks are present.
        Note: page_image_base64 is deliberately excluded from hashing.
        """
        if not self.chunks:
            return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()
        return build_merkle_tree([c.get_content_hash() for c in self.chunks])


class PipelineSettings(_Base):
    """PaddleOCRVL pipeline configuration settings."""

    use_ocr_for_image_block: bool = Field(
        default=False,
        description="Apply OCR to image blocks to extract text",
    )
    use_doc_orientation_classify: bool = Field(
        default=False,
        description="Automatically classify and correct document orientation",
    )
    use_doc_unwarping: bool = Field(
        default=False,
        description="Apply document unwarping for skewed/curved documents",
    )
    use_chart_recognition: bool = Field(
        default=False,
        description="Enable chart/graph recognition and extraction",
    )
    use_layout_detection: bool = Field(
        default=False,
        description="Enable layout detection for document structure analysis",
    )
    use_seal_recognition: bool = Field(
        default=False,
        description="Enable seal/stamp recognition in documents",
    )
    format_block_content: bool = Field(
        default=False,
        description="Format extracted block content with proper structure",
    )
    merge_layout_blocks: bool = Field(
        default=True,
        description="Merge adjacent layout blocks of the same type",
    )
    markdown_ignore_labels: List[str] = Field(
        default_factory=list,
        description="Layout labels to ignore when generating markdown",
    )
    pipeline_version: Literal["v1", "v1.5"] = Field(
        default="v1.5",
        description="Version of the PaddleOCR-VL pipeline to use",
    )
    vl_rec_api_model_name: str = Field(
        default="PaddlePaddle/PaddleOCR-VL-1.5",
        description="Vision-language model name for OCR recognition",
    )

    # --- Layout detection tuning ---
    layout_threshold: float = Field(
        default=0.45,
        description="Score threshold for the layout detection model",
    )
    layout_nms: bool = Field(
        default=True,
        description="Whether to use Non-Maximum Suppression in layout detection",
    )
    layout_unclip_ratio: float = Field(
        default=1.5,
        description="Expansion coefficient for layout detection boxes",
    )
    layout_merge_bboxes_mode: str = Field(
        default="overlap",
        description="Method for filtering/merging overlapping boxes",
    )
    layout_shape_mode: str = Field(
        default="auto",
        description="Mode for handling layout shapes (e.g. 'auto')",
    )

    # --- VLM Inference tuning ---
    prompt_label: Optional[str] = Field(
        default=None,
        description="Custom prompt label for the VLM inference",
    )
    repetition_penalty: float = Field(
        default=1.2,
        description="Repetition penalty for VLM sampling",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for VLM sampling (0.0 for greedy)",
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p sampling for VLM inference",
    )
    min_pixels: int = Field(
        default=3136,
        description="Minimum pixels for image preprocessing for the VLM",
    )
    max_pixels: int = Field(
        default=3136,
        description="Maximum pixels for image preprocessing for the VLM",
    )
    max_new_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens generated by the VLM",
    )

    # --- General processing ---
    use_queues: bool = Field(
        default=False,
        description="Whether to use queues for internal pipeline parallelism",
    )

    vl_rec_backend: BackendType = Field(
        default_factory=lambda: os.getenv("VLM_BACKEND", _default_backend()),
        description=(
            "Inference backend for the VLM. "
            "Supported: 'vllm-server', 'mlx-vlm-server', 'sglang-server', 'fastdeploy-server', 'local', 'native'. "
            "Defaults to 'mlx-vlm-server' on macOS, 'vllm-server' elsewhere."
        ),
    )
    vl_rec_server_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("VLM_SERVER_URL"),
        description=(
            "Base URL of the VLM inference server (/v1 OpenAI-compatible endpoint). "
            "Resolved automatically from vl_rec_backend if not set explicitly."
        ),
    )
    vl_rec_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("VLM_API_KEY"),
        description="API key for the VLM server, if authentication is required.",
    )

    @model_validator(mode="after")
    def _resolve_server_url(self) -> "PipelineSettings":
        if self.vl_rec_server_url is None and self.vl_rec_backend not in ("local", "native"):
            object.__setattr__(
                self, "vl_rec_server_url", _default_server_url(self.vl_rec_backend)
            )
        return self

    def to_init_kwargs(self) -> dict:
        """Parameters for PaddleOCRVL.__init__()."""
        # Excludes only internal auth fields when backend is local.
        # Includes vl_rec_api_model_name (passed as positional in DocumentParser,
        # but present in the init signature).
        exclude = set()
        if self.vl_rec_backend in ("local", "native"):
            exclude |= {"vl_rec_server_url", "vl_rec_api_key"}

        kwargs = self.model_dump(exclude=exclude)

        # Map 'local' alias to the PaddleX 'native' backend name
        if kwargs.get("vl_rec_backend") == "local":
            kwargs["vl_rec_backend"] = "native"

        # These are only valid for predict() signature in paddleocr_vl.py,
        # not present in __init__ signature.
        predict_only = {
            "layout_shape_mode",
            "prompt_label",
            "repetition_penalty",
            "temperature",
            "top_p",
            "min_pixels",
            "max_pixels",
            "max_new_tokens",
        }
        for k in predict_only:
            kwargs.pop(k, None)

        return kwargs

    def to_predict_kwargs(self) -> dict:
        """Parameters for PaddleOCRVL.predict()."""
        # These are present in predict() signature in paddleocr_vl.py.
        predict_fields = {
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_layout_detection",
            "use_chart_recognition",
            "use_seal_recognition",
            "use_ocr_for_image_block",
            "layout_threshold",
            "layout_nms",
            "layout_unclip_ratio",
            "layout_merge_bboxes_mode",
            "layout_shape_mode",
            "use_queues",
            "prompt_label",
            "format_block_content",
            "repetition_penalty",
            "temperature",
            "top_p",
            "min_pixels",
            "max_pixels",
            "max_new_tokens",
            "merge_layout_blocks",
            "markdown_ignore_labels",
        }
        kwargs = self.model_dump(include=predict_fields)
        return kwargs

    def to_pipeline_kwargs(self) -> dict:
        """
        Deprecated. Use to_init_kwargs() and to_predict_kwargs() instead.
        Maintained for backward compatibility.
        """
        exclude = {"vl_rec_api_model_name"}
        if self.vl_rec_backend in ("local", "native"):
            exclude |= {"vl_rec_server_url", "vl_rec_api_key"}
        
        kwargs = self.model_dump(exclude=exclude)
        if kwargs.get("vl_rec_backend") == "local":
            kwargs["vl_rec_backend"] = "native"
        return kwargs
        


# """
# Pydantic models for document parsing.

# This module defines the data models for parsed documents with validators,
# Field descriptions, and strict type hints.
# """

# from typing import Any, List
# from uuid import UUID, uuid4

# from pydantic import BaseModel, ConfigDict, Field, field_validator


# class Grounding(BaseModel):
#     """Spatial grounding information for a document chunk."""

#     model_config = ConfigDict(strict=False)

#     chunk_type: str = Field(
#         default="unknown",
#         description="Layout block type (e.g., 'paragraph', 'table', 'image')",
#     )
#     bbox: List[int] = Field(
#         description="Bounding box coordinates [x1, y1, x2, y2]",
#     )
#     page_index: int = Field(
#         ge=0,
#         description="Zero-indexed page number where this chunk appears",
#     )
#     score: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="Detection confidence score from layout analysis",
#     )

#     @field_validator("bbox")
#     @classmethod
#     def validate_bbox(cls, v: List[int]) -> List[int]:
#         """Validate that bbox has exactly 4 coordinates."""
#         if len(v) != 4:
#             raise ValueError("bbox must have exactly 4 coordinates [x1, y1, x2, y2]")
#         if not all(coord >= 0 for coord in v):
#             raise ValueError("bbox coordinates must be non-negative")
#         return v


# class Chunk(BaseModel):
#     """A parsed content chunk from a document."""

#     model_config = ConfigDict(strict=False)

#     chunk_id: UUID = Field(
#         default_factory=uuid4,
#         description="Unique identifier for this chunk",
#     )
#     chunk_markdown: str = Field(
#         default="",
#         description="Markdown-formatted content of the chunk",
#     )
#     grounding: Grounding = Field(
#         description="Spatial grounding information for this chunk",
#     )


# class Metadata(BaseModel):
#     """Document page metadata."""

#     model_config = ConfigDict(strict=False, arbitrary_types_allowed=True)

#     filename: str = Field(
#         description="Original input file path",
#     )
#     page_image_base64: str = Field(
#         default="",
#         description="Base64 encoded PNG image of the page",
#     )
#     page_index: int = Field(
#         ge=0,
#         description="Zero-indexed page number",
#     )
#     page_count: int = Field(
#         ge=1,
#         description="Total number of pages in the document",
#     )


# class Document(BaseModel):
#     """A parsed document page with chunks and metadata."""

#     model_config = ConfigDict(strict=False)

#     doc_id: UUID = Field(
#         default_factory=uuid4,
#         description="Unique identifier for this document page",
#     )
#     markdown: str = Field(
#         default="",
#         description="Full markdown content of the page",
#     )
#     chunks: List[Chunk] = Field(
#         default_factory=list,
#         description="List of content chunks extracted from the page",
#     )
#     metadata: Metadata = Field(
#         description="Page metadata including filename and page info",
#     )


# class PipelineSettings(BaseModel):
#     """PaddleOCRVL pipeline configuration settings."""

#     model_config = ConfigDict(strict=False)

#     use_ocr_for_image_block: bool = Field(
#         default=True,
#         description="Apply OCR to image blocks to extract text",
#     )
#     use_doc_orientation_classify: bool = Field(
#         default=True,
#         description="Automatically classify and correct document orientation",
#     )
#     use_doc_unwarping: bool = Field(
#         default=False,
#         description="Apply document unwarping for skewed/curved documents",
#     )
#     use_chart_recognition: bool = Field(
#         default=True,
#         description="Enable chart/graph recognition and extraction",
#     )
#     use_layout_detection: bool = Field(
#         default=True,
#         description="Enable layout detection for document structure analysis",
#     )
#     use_seal_recognition: bool = Field(
#         default=True,
#         description="Enable seal/stamp recognition in documents",
#     )
#     format_block_content: bool = Field(
#         default=True,
#         description="Format extracted block content with proper structure",
#     )
#     merge_layout_blocks: bool = Field(
#         default=True,
#         description="Merge adjacent layout blocks of the same type",
#     )
#     markdown_ignore_labels: List[str] = Field(
#         default_factory=list,
#         description="Layout labels to ignore when generating markdown",
#     )
#     vl_rec_api_model_name: str = Field(
#         default="PaddlePaddle/PaddleOCR-VL-1.5",
#         description="Vision-language model name for OCR recognition",
#     )

#     def to_pipeline_kwargs(self) -> dict:
#         """Convert settings to kwargs for PaddleOCRVL pipeline."""
#         return {
#             "use_ocr_for_image_block": self.use_ocr_for_image_block,
#             "use_doc_orientation_classify": self.use_doc_orientation_classify,
#             "use_doc_unwarping": self.use_doc_unwarping,
#             "use_chart_recognition": self.use_chart_recognition,
#             "use_layout_detection": self.use_layout_detection,
#             "use_seal_recognition": self.use_seal_recognition,
#             "format_block_content": self.format_block_content,
#             "merge_layout_blocks": self.merge_layout_blocks,
#             "markdown_ignore_labels": self.markdown_ignore_labels,
#         }
