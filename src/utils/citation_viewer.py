"""
Citation viewer for RAG responses — Google AI Mode-style highlighting.

Renders the AI answer with clickable citation badges. When a citation is
clicked, the corresponding document page is shown with the cited chunk's
bounding box highlighted, animated into view.

Designed to work inside Jupyter notebooks via IPython.display.HTML.
"""

import base64
import html
import re
import uuid as _uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

from shared.schemas import Document


# ── colour palette (same as interactive_layout) ─────────────────────
_PALETTE = [
    "#4285F4", "#EA4335", "#FBBC04", "#34A853",
    "#FF6D01", "#46BDC6", "#7B61FF", "#E8710A",
    "#1A73E8", "#D93025", "#0D652D", "#9334E6",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _image_dimensions_from_b64(b64: str) -> tuple:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    return img.size


# ── public API ──────────────────────────────────────────────────────

def display_cited_response(
    response: Dict[str, Any],
    documents: List[Document],
    width: int = 920,
) -> None:
    """
    Render a RAG response with interactive, highlighted citations.

    Inline citations like [1], [2] become clickable badges. Clicking one
    reveals the source document page with the cited chunk's bounding box
    highlighted — similar to Google AI Mode.

    Args:
        response: The dict returned by a LangChain RAG chain, containing
                  ``response["answer"].content`` (str) and
                  ``response["context"]`` (list of LangChain Documents).
        documents: The original list of :class:`Document` objects (used
                   to retrieve the full page image and bounding boxes).
        width: CSS pixel width of the rendered widget.
    """
    from IPython.display import HTML, display as ipy_display

    wid = f"cv_{_uuid.uuid4().hex[:10]}"

    answer_content = response["answer"].content
    context_docs = response["context"]

    # ── extract references ──────────────────────────────────────────
    refs_match = re.search(r"## References\s*(.*)", answer_content, re.DOTALL)
    if refs_match:
        answer_text = answer_content[: refs_match.start()].strip()
        refs_text = refs_match.group(1)
    else:
        answer_text = answer_content.strip()
        refs_text = ""

    # Parse citation metadata from references section
    cite_pattern = (
        r"\*\*\[(\d+)\]\*\*\s*Source:\s*([^,]+),\s*Page:\s*(\d+)"
        r",\s*Chunk ID:\s*([a-f0-9\-]+)"
    )
    raw_cites = re.findall(cite_pattern, refs_text)

    # Build page-image lookup from Document list:  doc_id -> { page_idx -> base64 }
    page_images: Dict[str, Dict[int, str]] = {}
    # Build chunk lookup:  chunk_id -> { bbox, page_index, chunk_type, score }
    chunk_lookup: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        did = str(doc.doc_id)
        page_images.setdefault(did, {})[doc.metadata.page_index] = (
            doc.metadata.page_image_base64
        )
        for ch in doc.chunks:
            chunk_lookup[str(ch.chunk_id)] = {
                "bbox": ch.grounding.bbox,
                "page_index": ch.grounding.page_index,
                "chunk_type": ch.grounding.chunk_type,
                "score": ch.grounding.score,
                "preview": (ch.chunk_markdown or "")[:300],
            }

    # ── build citation objects ──────────────────────────────────────
    citations = []
    for cite_num, source, page, chunk_id in raw_cites:
        # find doc_id from context docs
        doc_id = None
        for cd in context_docs:
            if cd.metadata.get("chunk_id") == chunk_id:
                doc_id = cd.metadata.get("doc_id")
                break

        page_idx = int(page)
        img_b64 = ""
        if doc_id and doc_id in page_images and page_idx in page_images[doc_id]:
            img_b64 = page_images[doc_id][page_idx]

        bbox = chunk_lookup.get(chunk_id, {}).get("bbox", [0, 0, 0, 0])
        chunk_type = chunk_lookup.get(chunk_id, {}).get("chunk_type", "unknown")
        score = chunk_lookup.get(chunk_id, {}).get("score", 0)
        preview = chunk_lookup.get(chunk_id, {}).get("preview", "")

        citations.append({
            "num": int(cite_num),
            "source": source.strip(),
            "page": page_idx,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "score": score,
            "preview": preview,
            "bbox": bbox,
            "img_b64": img_b64,
        })

    # ── render answer with clickable badges ─────────────────────────
    def _badge_replacer(m):
        n = int(m.group(1))
        color = _PALETTE[(n - 1) % len(_PALETTE)]
        return (
            f'<span class="{wid}-badge" data-cite="{n}" '
            f'style="display:inline-block;background:{color};color:#fff;'
            f"font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;"
            f"cursor:pointer;margin:0 1px;vertical-align:super;line-height:1;"
            f'transition:transform .12s,box-shadow .12s;" '
            f'onmouseenter="this.style.transform=\'scale(1.18)\';'
            f"this.style.boxShadow='0 0 6px {_hex_to_rgba(color, 0.6)}'\" "
            f'onmouseleave="this.style.transform=\'scale(1)\';'
            f"this.style.boxShadow='none'\" "
            f"onclick=\"(function(el){{var p=document.getElementById('{wid}-panel');"
            f"var items=p.querySelectorAll('.{wid}-cite-panel');"
            f"items.forEach(function(it){{it.style.display='none'}});"
            f"var t=document.getElementById('{wid}-cite-'+el.dataset.cite);"
            f"if(t){{t.style.display='block';p.style.display='block';"
            f"p.scrollIntoView({{behavior:'smooth',block:'nearest'}})}}"
            f"}})(this)\">"
            f"[{n}]</span>"
        )

    styled_answer = re.sub(r"\[(\d+)\]", _badge_replacer, html.escape(answer_text))
    # Preserve paragraph breaks
    styled_answer = styled_answer.replace("\n\n", "</p><p>").replace("\n", "<br>")
    styled_answer = f"<p>{styled_answer}</p>"

    # Only keep citations that are actually referenced inline
    inline_nums = set(int(x) for x in re.findall(r"\[(\d+)\]", answer_text))
    citations = [c for c in citations if c["num"] in inline_nums]

    # ── build citation panels (one per citation) ────────────────────
    cite_panels_html = []
    for c in citations:
        n = c["num"]
        color = _PALETTE[(n - 1) % len(_PALETTE)]

        # Compute bbox overlay as percentages
        overlay_html = ""
        if c["img_b64"]:
            iw, ih = _image_dimensions_from_b64(c["img_b64"])
            x1, y1, x2, y2 = c["bbox"]
            lp = x1 / iw * 100
            tp = y1 / ih * 100
            wp = (x2 - x1) / iw * 100
            hp = (y2 - y1) / ih * 100

            overlay_html = f"""
            <div style="position:relative;border-radius:6px;overflow:hidden;
                        border:1px solid #333;margin-top:10px;">
              <img src="data:image/png;base64,{c['img_b64']}"
                   style="display:block;width:100%;height:auto;" />
              <!-- highlight overlay on the cited chunk -->
              <div style="
                position:absolute;
                left:{lp:.4f}%;top:{tp:.4f}%;
                width:{wp:.4f}%;height:{hp:.4f}%;
                border:2.5px solid {color};
                border-radius:4px;
                background:{_hex_to_rgba(color, 0.18)};
                box-shadow:0 0 14px {_hex_to_rgba(color, 0.4)};
                animation:{wid}-pulse 1.8s ease-in-out 2;
                z-index:2;
              "></div>
            </div>"""

        preview_esc = html.escape(c["preview"]).replace("\n", "<br>")

        cite_panels_html.append(f"""
        <div id="{wid}-cite-{n}" class="{wid}-cite-panel"
             style="display:none;animation:{wid}-fadeIn .25s ease-out;">
          <!-- header -->
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="background:{color};color:#fff;font-weight:700;
                           padding:2px 10px;border-radius:12px;font-size:13px;">
                [{n}]
              </span>
              <span style="color:#aaa;font-size:12px;">
                {html.escape(c['source'])} &middot; Page {c['page'] + 1}
                &middot; <em>{html.escape(c['chunk_type'])}</em>
                ({c['score']:.0%})
              </span>
            </div>
            <span style="cursor:pointer;color:#888;font-size:18px;padding:0 4px;"
                  onclick="this.parentElement.parentElement.style.display='none';"
                  title="Close">✕</span>
          </div>
          <!-- markdown preview -->
          <div style="background:#1a1a2e;border-radius:6px;padding:10px 14px;
                      font-size:12.5px;color:#c8c8d4;line-height:1.55;
                      max-height:120px;overflow-y:auto;border:1px solid #2a2a4a;
                      margin-bottom:4px;">
            {preview_esc}
          </div>
          <!-- page image with highlight -->
          {overlay_html}
        </div>""")

    # ── assemble full widget ────────────────────────────────────────
    full_html = f"""
    <style>
      @keyframes {wid}-fadeIn {{
        from {{ opacity:0; transform:translateY(8px); }}
        to   {{ opacity:1; transform:translateY(0); }}
      }}
      @keyframes {wid}-pulse {{
        0%,100% {{ box-shadow:0 0 18px {_hex_to_rgba('#4285F4', 0.55)}; }}
        50%     {{ box-shadow:0 0 30px {_hex_to_rgba('#4285F4', 0.85)},
                              0 0 60px {_hex_to_rgba('#4285F4', 0.3)}; }}
      }}
    </style>

    <div id="{wid}" style="
      width:{width}px;
      font-family:'Inter','Segoe UI',system-ui,sans-serif;
      background:#0d1117;
      border-radius:14px;
      overflow:hidden;
      box-shadow:0 6px 32px rgba(0,0,0,0.5);
      margin:16px 0;
      color:#e6e6ef;
    ">
      <!-- header -->
      <div style="
        padding:14px 20px;
        background:linear-gradient(135deg,#161b22,#0f3460);
        display:flex;align-items:center;gap:10px;
      ">
        <span style="font-size:18px;">✦</span>
        <span style="font-weight:600;font-size:15px;">AI Response</span>
        <span style="margin-left:auto;font-size:12px;color:#6a7a9a;">
          {len(citations)} source{"s" if len(citations) != 1 else ""} cited
        </span>
      </div>

      <!-- answer body -->
      <div style="padding:18px 22px;font-size:14.5px;line-height:1.75;">
        {styled_answer}
      </div>

      <!-- citation panels container -->
      <div id="{wid}-panel" style="
        display:none;
        padding:0 22px 18px 22px;
        border-top:1px solid #1e2a3a;
      ">
        <div style="font-size:12px;color:#5a6a8a;padding:12px 0 8px 0;
                    text-transform:uppercase;letter-spacing:.5px;font-weight:600;">
          Source reference
        </div>
        {"".join(cite_panels_html)}
      </div>

      <!-- references list -->
      <div style="padding:10px 22px 14px;background:#0a0e14;
                  border-top:1px solid #1a2030;">
        <details style="color:#6a7a9a;font-size:12px;">
          <summary style="cursor:pointer;user-select:none;padding:4px 0;
                         font-weight:600;letter-spacing:.3px;">
            References ({len(citations)})
          </summary>
          <div style="margin-top:8px;">
            {"".join(
              f'<div style="padding:3px 0;color:#5a6a8a;">'
              f'<span style="color:{_PALETTE[(c["num"]-1) % len(_PALETTE)]};">'
              f'[{c["num"]}]</span> '
              f'{html.escape(c["source"])} · Page {c["page"]+1} · '
              f'<em>{html.escape(c["chunk_type"])}</em></div>'
              for c in citations
            )}
          </div>
        </details>
      </div>
    </div>
    """

    ipy_display(HTML(full_html))
