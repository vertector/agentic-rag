# Pipeline Settings Reference — PaddleOCRVL 1.5

## Default Values

| Field | Default | Notes |
|---|---|---|
| `vl_rec_api_model_name` | `"PaddlePaddle/PaddleOCR-VL-1.5"` | HF model name |
| `vl_rec_backend` | platform-auto | `"local"` on CPU; `"vllm-server"` on GPU |
| `vl_rec_server_url` | `null` | Required for external backends |
| `vl_rec_api_key` | `null` | For authenticated VLM endpoints |
| `use_ocr_for_image_block` | `True` | Apply OCR to image regions |
| `use_doc_orientation_classify` | `True` | Auto-correct rotated pages |
| `use_doc_unwarping` | `False` | Dewarp curved/scanned docs (slow) |
| `use_chart_recognition` | `False` | Extract charts as structured data |
| `use_layout_detection` | `True` | Detect layout regions |
| `use_seal_recognition` | `True` | Detect seals/stamps |
| `format_block_content` | `True` | Post-process block markdown |
| `merge_layout_blocks` | `True` | Merge adjacent same-type blocks |
| `merge_tables` | `True` | Merge tables across page boundaries |
| `relevel_titles` | `True` | Auto-adjust heading levels |
| `markdown_ignore_labels` | `[]` | Common: `["header", "footer", "page_number"]` |
| `pipeline_version` | `"v1.5"` | Use `"v1"` for older PaddleOCR |
| `layout_threshold` | `0.3` | Raise to reduce false positives |
| `layout_nms` | `True` | Non-maximum suppression for boxes |
| `layout_unclip_ratio` | `null` | Expand layout boxes (>1 = larger). null = use YAML default [1.0, 1.0] |
| `layout_merge_bboxes_mode` | `null` | `"union"` or `"large"`. null = use per-class YAML defaults |
| `layout_shape_mode` | `"auto"` | `"auto"`, `"rectangle"` |
| `temperature` | `0.0` | 0 = deterministic VLM |
| `top_p` | `1.0` | Nucleus sampling (irrelevant at temp=0) |
| `max_new_tokens` | `4096` | Max tokens per block |
| `repetition_penalty` | `1.0` | >1 reduces repetition |
| `prompt_label` | `null` | ⚠️ DANGEROUS — see Gotchas |
| `min_pixels` | `147384` | ~384×384 pixels |
| `max_pixels` | `8699840` | ~2944×2944 pixels |
| `vlm_extra_args` | `{}` | Pass-through to backend |

## Common Override Patterns

### Suppress headers/footers from markdown output
```json
{"markdown_ignore_labels": ["header", "footer", "page_number"]}
```

### High-accuracy mode (slower)
```json
{
  "temperature": 0.0,
  "max_new_tokens": 8192,
  "layout_threshold": 0.5,
  "use_doc_unwarping": true
}
```

### Charts enabled (disable for pure text docs — saves time)
```json
{"use_chart_recognition": true}
```

### External vLLM server
```json
{
  "vl_rec_backend": "vllm-server",
  "vl_rec_server_url": "http://localhost:8000",
  "vl_rec_api_model_name": "PaddlePaddle/PaddleOCR-VL-1.5"
}
```

### Scanned/low-quality document
```json
{
  "use_doc_unwarping": true,
  "use_doc_orientation_classify": true,
  "layout_threshold": 0.25,
  "max_new_tokens": 8192
}
```

## Supported Extensions
`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`

## MAX_BASE64_MB
50 MB decoded payload limit. Files larger than 50 MB must use `file_path`.