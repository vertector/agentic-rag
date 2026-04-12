import os
import sys
from pathlib import Path
from shared.env_loader import load_env

# Hydrate environment from .env at the shared layer
load_env()

import hashlib
import json
import uuid
from typing import Any, Dict, List, Literal, Optional
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
        ge=1,
        description="1-indexed page number where this chunk appears",
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
        description="Original input file path. Automatically converted to base filename.",
    )

    @field_validator("filename", mode="after")
    @classmethod
    def _extract_basename(cls, v: str) -> str:
        return Path(v).name

    page_index: int = Field(
        ge=1,
        description="1-indexed page number",
    )
    page_count: int = Field(
        ge=1,
        description="Total number of pages in the document",
    )
    category: str = Field(
        default="general",
        description=(
            "Document domain used as a Qdrant filter key. "
            "Examples: 'research', 'medical', 'finance'. "
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


# ---------------------------------------------------------------------------
# PipelineSettings
# ---------------------------------------------------------------------------

class PipelineSettings(_Base):
    """
    PaddleOCRVL v1.5 pipeline configuration.

    Parameters are split between to_init_kwargs() (constructor) and
    to_predict_kwargs() (per-call inference) — do not mix them up.

    Three-fix summary for table parsing quality
    -------------------------------------------
    FIX 1 — prompt_label removed from to_predict_kwargs()
        PaddleOCRVL routes each detected block to its task-specific VLM prompt:
          table region   → "Table Recognition:"   ← correct header alignment
          formula region → "Formula Recognition:"
          other text     → "OCR:"
        Passing prompt_label="text" globally overrides this routing and forces
        ALL blocks — including tables — through the generic "OCR:" prompt,
        which produces flat text without structural markdown table formatting.
        The field is retained here for explicit override when needed (e.g. debug
        runs forcing a single task type), but it is excluded from to_predict_kwargs()
        unless explicitly set to a non-None value AND the caller sets
        override_prompt_label=True.

    FIX 2 — layout_threshold lowered from 0.4 → 0.3
        The PaddleOCRVL YAML default is 0.3. At 0.4, layout detection drops
        lower-confidence proposals — which is fine for high-contrast text but
        problematic for tables with light borders, borderless cells, or merged
        header rows. Missed table proposals cause the region to be detected as
        multiple separate text blocks instead of one table block, preventing
        the VLM from receiving the full table structure as a unit.

    FIX 3 — max_pixels raised from 2,822,400 → 8,699,840
        Medical and academic PDFs are typically 300 DPI.
        A4 at 300 DPI = 2480×3508 = 8,699,840 pixels.
        With the old limit the image was downscaled to ~1412×1998 (0.57×),
        shrinking table cell text from ~8-10px to ~4-5px — below the VLM's
        reliable recognition threshold. No downscaling = full fidelity
        for small numerals, parentheses, and decimal points in table cells.
        The PaddleOCR-VL-0.9B model is lightweight enough to handle this
        on both Apple Silicon (mlx-vlm-server) and GPU (vllm-server).
    """

    # ── Core flags ────────────────────────────────────────────────────────────

    use_ocr_for_image_block: bool = Field(
        default=True,
        description="Apply OCR to image blocks to extract text",
    )
    use_doc_orientation_classify: bool = Field(
        default=True,
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
        default=True,
        description="Enable layout detection for document structure analysis",
    )
    use_seal_recognition: bool = Field(
        default=True,
        description="Enable seal/stamp recognition in documents",
    )
    format_block_content: bool = Field(
        default=True,
        description="Format extracted block content with proper structure",
    )
    merge_layout_blocks: bool = Field(
        default=True,
        description="Merge adjacent layout blocks of the same type",
    )
    merge_tables: bool = Field(
        default=True,
        description="Merge table content across page boundaries",
    )
    relevel_titles: bool = Field(
        default=True,
        description="Automatically adjust paragraph title levels for structure",
    )
    markdown_ignore_labels: List[str] = Field(
        default_factory=list,
        description="Layout labels to ignore when generating markdown",
    )
    pipeline_version: Literal["v1", "v1.5"] = Field(
        default="v1.5",
        description="Version of the PaddleOCR-VL pipeline to use",
    )

    # ── Backend config ────────────────────────────────────────────────────────

    vl_rec_api_model_name: str = Field(
        default="PaddlePaddle/PaddleOCR-VL-1.5",
        description="Vision-language model name for OCR recognition",
    )
    vl_rec_backend: BackendType = Field(
        default_factory=lambda: os.getenv("VLM_BACKEND", _default_backend()),
        description="Inference backend for the VLM.",
    )
    vl_rec_server_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("VLM_SERVER_URL"),
        description="Base URL of the VLM inference server.",
    )
    vl_rec_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("VLM_API_KEY"),
        description="API key if required by the server.",
    )

    # ── Layout tuning ─────────────────────────────────────────────────────────

    layout_threshold: Optional[float] = Field(
        default=0.3,
        # FIX 2: was 0.4 — see class docstring.
        description=(
            "Score threshold for layout detection proposals. "
            "Proposals below this score are discarded. "
            "YAML default and recommended value: 0.3. "
            "Lower values increase recall (fewer missed table regions) at the "
            "cost of more false positives on noisy backgrounds."
        ),
    )
    layout_nms: Optional[bool] = Field(
        default=True,
        description="Enable Non-Maximum Suppression for layout boxes",
    )
    layout_unclip_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Expansion ratio for detected layout boxes before passing to sub-models. "
            "None = use YAML default [1.0, 1.0] (no expansion). "
            "Set to 1.05 if the VLM consistently clips characters at region edges."
        ),
    )
    layout_merge_bboxes_mode: Optional[str] = Field(
        default=None,
        description=(
            "Strategy for merging overlapping bounding boxes: 'union' or 'large'. "
            "None = use per-class YAML defaults (tables → 'union')."
        ),
    )
    layout_shape_mode: Optional[str] = Field(
        default="auto",
        description="Geometric shape of layout boxes: 'auto', 'rectangle', etc.",
    )

    # ── VLM inference tuning ──────────────────────────────────────────────────

    temperature: Optional[float] = Field(
        default=0.0,
        description="Sampling temperature. 0.0 = fully deterministic (recommended for OCR).",
    )
    top_p: Optional[float] = Field(
        default=1.0,
        description="Top-p nucleus sampling threshold.",
    )
    max_new_tokens: Optional[int] = Field(
        default=4096,
        description=(
            "Maximum VLM output tokens per block. "
            "4096 is sufficient for most tables; increase for very large tables "
            "with many rows (>50)."
        ),
    )
    repetition_penalty: Optional[float] = Field(
        default=1.0,
        description=(
            "Penalty for repeating token sequences. "
            "1.0 = no penalty. Raise to 1.05–1.1 if the VLM repeats table rows."
        ),
    )

    # ── FIX 1: prompt_label — excluded from to_predict_kwargs() by default ───
    # See class docstring. This field exists for explicit debug override only.
    # Normal use: leave as None and let PaddleOCRVL route per block type.
    prompt_label: Optional[str] = Field(
        default=None,
        description=(
            "Global VLM task prompt override. "
            "None (default) = let PaddleOCRVL route each block to its task-specific "
            "prompt automatically: table→'Table Recognition:', formula→'Formula "
            "Recognition:', other→'OCR:'. "
            "Set to a specific value ONLY for debugging a single task type. "
            "WARNING: setting this forces ALL blocks through one prompt, "
            "breaking table structure recognition."
        ),
    )

    # ── FIX 3: pixel limits ───────────────────────────────────────────────────

    min_pixels: Optional[int] = Field(
        default=147_384,
        description=(
            "Minimum total pixels for VLM input image. "
            "Images smaller than this are upscaled. "
            "Default 147,384 ≈ 384² (PaddleOCR-VL internal minimum tile size)."
        ),
    )
    max_pixels: Optional[int] = Field(
        default=8_699_840,
        # FIX 3: was 2,822,400 — see class docstring.
        description=(
            "Maximum total pixels for VLM input image. "
            "Images larger than this are downscaled before VLM inference. "
            "Default 8,699,840 = 2480×3508 (A4 at 300 DPI, no downscaling). "
            "This prevents loss of small text in table cells on high-DPI documents. "
            "The PaddleOCR-VL-0.9B model handles this comfortably on both "
            "Apple Silicon (mlx-vlm-server) and GPU (vllm-server)."
        ),
    )

    vlm_extra_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generic bucket for backend-specific parameters not covered above.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _resolve_server_url(self) -> "PipelineSettings":
        if self.vl_rec_server_url is None and self.vl_rec_backend not in ("local", "native"):
            object.__setattr__(
                self, "vl_rec_server_url", _default_server_url(self.vl_rec_backend)
            )
        return self

    def _get_mapped_backend(self) -> str:
        """Map 'local' alias to 'native' for PaddleOCR consistency."""
        return "native" if self.vl_rec_backend == "local" else self.vl_rec_backend

    # ── Constructor kwargs ────────────────────────────────────────────────────

    def to_init_kwargs(self) -> dict:
        """Parameters forwarded to PaddleOCRVL.__init__()."""
        init_keys = {
            "use_ocr_for_image_block", "use_doc_orientation_classify", "use_doc_unwarping",
            "use_chart_recognition", "use_layout_detection", "use_seal_recognition",
            "format_block_content", "merge_layout_blocks", "markdown_ignore_labels",
            "pipeline_version", "vl_rec_server_url", "vl_rec_api_key", "vl_rec_api_model_name",
            "layout_threshold", "layout_nms", "layout_unclip_ratio", "layout_merge_bboxes_mode",
        }
        full_dump = self.model_dump()
        kwargs = {k: v for k, v in full_dump.items() if k in init_keys and v is not None}
        kwargs["vl_rec_backend"] = self._get_mapped_backend()

        if kwargs["vl_rec_backend"] == "native":
            kwargs.pop("vl_rec_server_url", None)
            kwargs.pop("vl_rec_api_key", None)

        return kwargs

    # ── Per-call inference kwargs ─────────────────────────────────────────────

    def to_predict_kwargs(self) -> dict:
        """
        Parameters forwarded to PaddleOCRVL.predict().

        prompt_label is intentionally excluded from the default set (FIX 1).
        It is only included when explicitly set to a non-None value, which
        the caller does for single-task debug runs — never in normal operation.
        """
        predict_keys = {
            "temperature", "top_p", "max_new_tokens", "repetition_penalty",
            "layout_threshold", "layout_nms",
            "layout_unclip_ratio", "layout_merge_bboxes_mode",
            "layout_shape_mode", "min_pixels", "max_pixels",
            # prompt_label intentionally excluded here — see FIX 1 in class docstring.
            # It is added below only when the caller explicitly set it.
        }

        full_dump = self.model_dump()
        kwargs = {k: v for k, v in full_dump.items() if k in predict_keys and v is not None}

        # Only pass prompt_label if explicitly overridden (not None = not default)
        if self.prompt_label is not None:
            kwargs["prompt_label"] = self.prompt_label

        if self.vlm_extra_args:
            kwargs.update(self.vlm_extra_args)

        return kwargs

    def to_pipeline_kwargs(self) -> dict:
        """Backwards compatibility alias for to_init_kwargs()."""
        return self.to_init_kwargs()
        