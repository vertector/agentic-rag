"""
Interactive layout visualization for document parsing results.

Renders PaddleOCR layout detection as an interactive HTML/JS widget
inside Jupyter notebooks. Regions are hoverable, clickable, and
selectable with rich tooltips — no external JS dependencies required.
"""

import base64
import html
import uuid as _uuid
from io import BytesIO
from typing import List

from PIL import Image

from shared.schemas import Document


# ── colour palette ──────────────────────────────────────────────────
_PALETTE = [
    "#4285F4",  # blue
    "#EA4335",  # red
    "#FBBC04",  # yellow
    "#34A853",  # green
    "#FF6D01",  # orange
    "#46BDC6",  # teal
    "#7B61FF",  # purple
    "#E8710A",  # deep-orange
    "#1A73E8",  # indigo
    "#D93025",  # crimson
    "#0D652D",  # forest
    "#9334E6",  # violet
]


def _color_for_index(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _image_dimensions(base64_string: str) -> tuple:
    """Return (width, height) of a base64-encoded image."""
    img = Image.open(BytesIO(base64.b64decode(base64_string)))
    return img.size


def display_layout_interactive(
    document: Document,
    min_confidence: float = 0.5,
    width: int = 900,
) -> None:
    """
    Display an interactive layout visualization in a Jupyter notebook.

    Each detected chunk is rendered as a semi-transparent, hoverable,
    clickable region overlaid on the page image.  A tooltip shows
    chunk type, confidence score, and a markdown preview.

    Args:
        document: Document object with chunks and page image.
        min_confidence: Hide chunks below this confidence.
        width: CSS pixel width of the rendered widget.
    """
    from IPython.display import HTML, display as ipy_display

    widget_id = f"ilv_{_uuid.uuid4().hex[:10]}"
    img_w, img_h = _image_dimensions(document.metadata.page_image_base64)

    # ── build chunk data ────────────────────────────────────────────
    labels = sorted(set(c.grounding.chunk_type for c in document.chunks))
    label_color = {lab: _color_for_index(i) for i, lab in enumerate(labels)}

    regions_html = []
    for chunk in document.chunks:
        if chunk.grounding.score < min_confidence:
            continue

        g = chunk.grounding
        x1, y1, x2, y2 = g.bbox
        # percentages relative to natural image size
        left_pct = x1 / img_w * 100
        top_pct = y1 / img_h * 100
        w_pct = (x2 - x1) / img_w * 100
        h_pct = (y2 - y1) / img_h * 100

        color = label_color[g.chunk_type]
        bg = _hex_to_rgba(color, 0.15)
        border = _hex_to_rgba(color, 0.6)
        hover_bg = _hex_to_rgba(color, 0.30)
        selected_bg = _hex_to_rgba(color, 0.40)

        preview = html.escape(
            (chunk.chunk_markdown or "").replace("\n", " ")[:220]
        )
        ctype = html.escape(g.chunk_type)
        cid = str(chunk.chunk_id)
        score = f"{g.score:.2f}"

        regions_html.append(f"""
        <div class="{widget_id}-region"
             data-cid="{cid}"
             data-type="{ctype}"
             data-score="{score}"
             data-preview="{preview}"
             style="
               position:absolute;
               left:{left_pct:.4f}%;top:{top_pct:.4f}%;
               width:{w_pct:.4f}%;height:{h_pct:.4f}%;
               background:transparent;
               border:2px solid transparent;
               border-radius:3px;
               cursor:pointer;
               transition: background .15s, border-color .15s, box-shadow .15s;
               z-index:2;
             "
             onmouseenter="
               if(!this.classList.contains('selected')){{
                 this.style.background='{hover_bg}';
                 this.style.borderColor='{border}';
                 this.style.boxShadow='0 0 8px {_hex_to_rgba(color, 0.5)}';
               }}
               var tt=document.getElementById('{widget_id}-tooltip');
               tt.innerHTML='<strong>'+this.dataset.type+'</strong> &middot; score '+this.dataset.score
                 +'<br><span style=\\'font-size:11px;color:#ccc;\\'>'+this.dataset.cid+'</span>'
                 +(this.dataset.preview?'<hr style=\\'margin:6px 0;border-color:#444\\'/><span style=\\'font-size:12px;line-height:1.4;\\'>'+this.dataset.preview+'</span>':'');
               tt.style.display='block';
             "
             onmouseleave="
               if(!this.classList.contains('selected')){{
                 this.style.background='transparent';
                 this.style.borderColor='transparent';
                 this.style.boxShadow='none';
               }}
               document.getElementById('{widget_id}-tooltip').style.display='none';
             "
             onmousemove="
               var tt=document.getElementById('{widget_id}-tooltip');
               var box=document.getElementById('{widget_id}').getBoundingClientRect();
               var x=event.clientX-box.left+14, y=event.clientY-box.top+14;
               if(x+tt.offsetWidth>box.width) x=x-tt.offsetWidth-28;
               if(y+tt.offsetHeight>box.height) y=y-tt.offsetHeight-28;
               tt.style.left=x+'px';tt.style.top=y+'px';
             "
             onclick="
               this.classList.toggle('selected');
               if(this.classList.contains('selected')){{
                 this.style.background='{selected_bg}';
                 this.style.borderColor='{border}';
                 this.style.borderWidth='3px';
                 this.style.boxShadow='0 0 12px {_hex_to_rgba(color, 0.7)}';
               }} else {{
                 this.style.background='transparent';
                 this.style.borderColor='transparent';
                 this.style.borderWidth='2px';
                 this.style.boxShadow='none';
               }}
             "
        ></div>""")

    # ── legend ──────────────────────────────────────────────────────
    legend_items = "".join(
        f"""<span style="display:inline-flex;align-items:center;margin-right:16px;margin-bottom:6px;">
              <span style="width:14px;height:14px;border-radius:3px;background:{c};
                           display:inline-block;margin-right:6px;border:1px solid {_hex_to_rgba(c, 0.8)};"></span>
              <span style="font-size:13px;color:#e0e0e0;">{html.escape(l)}</span>
            </span>"""
        for l, c in label_color.items()
    )

    page_info = f"Page {document.metadata.page_index} / {document.metadata.page_count}"
    chunk_count = sum(1 for c in document.chunks if c.grounding.score >= min_confidence)

    full_html = f"""
    <div id="{widget_id}" style="
        position:relative;
        width:{width}px;
        background:#1a1a2e;
        border-radius:12px;
        overflow:hidden;
        box-shadow:0 4px 24px rgba(0,0,0,0.4);
        font-family:'Inter','Segoe UI',system-ui,sans-serif;
        margin:12px 0;
    ">
      <!-- header bar -->
      <div style="
        display:flex;justify-content:space-between;align-items:center;
        padding:10px 16px;
        background:linear-gradient(135deg,#16213e,#0f3460);
        color:#e0e0e0;font-size:13px;
      ">
        <span><strong>Layout Detection</strong> &mdash; {page_info}</span>
        <span style="opacity:.7;">{chunk_count} regions (≥ {min_confidence} confidence)</span>
      </div>

      <!-- image + overlay container -->
      <div style="position:relative;width:100%;">
        <img src="data:image/png;base64,{document.metadata.page_image_base64}"
             style="display:block;width:100%;height:auto;" />
        {"".join(regions_html)}

        <!-- tooltip -->
        <div id="{widget_id}-tooltip" style="
          display:none;position:absolute;z-index:10;
          max-width:360px;padding:10px 14px;
          background:rgba(22,33,62,0.95);
          border:1px solid #3a4a7a;border-radius:8px;
          color:#f0f0f0;font-size:13px;line-height:1.5;
          pointer-events:none;
          backdrop-filter:blur(6px);
          box-shadow:0 4px 16px rgba(0,0,0,0.5);
        "></div>
      </div>

      <!-- legend -->
      <div style="padding:10px 16px;background:#16213e;display:flex;flex-wrap:wrap;align-items:center;">
        <span style="font-size:12px;color:#8899bb;margin-right:12px;">Legend:</span>
        {legend_items}
      </div>
    </div>
    """

    ipy_display(HTML(full_html))


def display_layout_interactive_batch(
    documents: List[Document],
    min_confidence: float = 0.5,
    width: int = 900,
) -> None:
    """
    Display interactive layout visualizations for a list of documents
    in a horizontally scrollable strip.

    Args:
        documents: List of Document objects to visualize.
        min_confidence: Hide chunks below this confidence.
        width: CSS pixel width of each rendered widget.
    """
    from IPython.display import HTML, display as ipy_display

    container_id = f"batch_{_uuid.uuid4().hex[:10]}"

    # Collect each widget's inner HTML without calling ipy_display per page.
    # We re-use the existing function's logic by monkey-patching display — 
    # simpler: just inline the container and call the single-doc function
    # inside a flex wrapper by capturing its output.
    widgets_html = []
    for document in documents:
        widget_id = f"ilv_{_uuid.uuid4().hex[:10]}"
        img_w, img_h = _image_dimensions(document.metadata.page_image_base64)

        labels = sorted(set(c.grounding.chunk_type for c in document.chunks))
        label_color = {lab: _color_for_index(i) for i, lab in enumerate(labels)}

        regions_html = []
        for chunk in document.chunks:
            if chunk.grounding.score < min_confidence:
                continue

            g = chunk.grounding
            x1, y1, x2, y2 = g.bbox
            left_pct   = x1 / img_w * 100
            top_pct    = y1 / img_h * 100
            w_pct      = (x2 - x1) / img_w * 100
            h_pct      = (y2 - y1) / img_h * 100

            color       = label_color[g.chunk_type]
            bg          = _hex_to_rgba(color, 0.15)
            border      = _hex_to_rgba(color, 0.6)
            hover_bg    = _hex_to_rgba(color, 0.30)
            selected_bg = _hex_to_rgba(color, 0.40)

            preview = html.escape((chunk.chunk_markdown or "").replace("\n", " ")[:220])
            ctype   = html.escape(g.chunk_type)
            cid     = str(chunk.chunk_id)
            score   = f"{g.score:.2f}"

            regions_html.append(f"""
            <div class="{widget_id}-region"
                 data-cid="{cid}" data-type="{ctype}"
                 data-score="{score}" data-preview="{preview}"
                 style="position:absolute;left:{left_pct:.4f}%;top:{top_pct:.4f}%;
                        width:{w_pct:.4f}%;height:{h_pct:.4f}%;
                        background:transparent;border:2px solid transparent;
                        border-radius:3px;cursor:pointer;z-index:2;
                        transition:background .15s,border-color .15s,box-shadow .15s;"
                 onmouseenter="
                   if(!this.classList.contains('selected')){{
                     this.style.background='{hover_bg}';
                     this.style.borderColor='{border}';
                     this.style.boxShadow='0 0 8px {_hex_to_rgba(color, 0.5)}';
                   }}
                   var tt=document.getElementById('{widget_id}-tooltip');
                   tt.innerHTML='<strong>'+this.dataset.type+'</strong> &middot; score '+this.dataset.score
                     +'<br><span style=\\'font-size:11px;color:#ccc;\\'>'+this.dataset.cid+'</span>'
                     +(this.dataset.preview?'<hr style=\\'margin:6px 0;border-color:#444\\'/><span style=\\'font-size:12px;line-height:1.4;\\'>'+this.dataset.preview+'</span>':'');
                   tt.style.display='block';"
                 onmouseleave="
                   if(!this.classList.contains('selected')){{
                     this.style.background='transparent';
                     this.style.borderColor='transparent';
                     this.style.boxShadow='none';
                   }}
                   document.getElementById('{widget_id}-tooltip').style.display='none';"
                 onmousemove="
                   var tt=document.getElementById('{widget_id}-tooltip');
                   var box=document.getElementById('{widget_id}').getBoundingClientRect();
                   var x=event.clientX-box.left+14,y=event.clientY-box.top+14;
                   if(x+tt.offsetWidth>box.width)x=x-tt.offsetWidth-28;
                   if(y+tt.offsetHeight>box.height)y=y-tt.offsetHeight-28;
                   tt.style.left=x+'px';tt.style.top=y+'px';"
                 onclick="
                   this.classList.toggle('selected');
                   if(this.classList.contains('selected')){{
                     this.style.background='{selected_bg}';
                     this.style.borderColor='{border}';
                     this.style.borderWidth='3px';
                     this.style.boxShadow='0 0 12px {_hex_to_rgba(color, 0.7)}';
                   }} else {{
                     this.style.background='transparent';
                     this.style.borderColor='transparent';
                     this.style.borderWidth='2px';
                     this.style.boxShadow='none';
                   }}"
            ></div>""")

        legend_items = "".join(
            f"""<span style="display:inline-flex;align-items:center;margin-right:12px;margin-bottom:4px;">
                  <span style="width:12px;height:12px;border-radius:3px;background:{c};
                               display:inline-block;margin-right:5px;"></span>
                  <span style="font-size:12px;color:#e0e0e0;">{html.escape(l)}</span>
                </span>"""
            for l, c in label_color.items()
        )

        page_info   = f"Page {document.metadata.page_index} / {document.metadata.page_count}"
        chunk_count = sum(1 for c in document.chunks if c.grounding.score >= min_confidence)

        widgets_html.append(f"""
        <div id="{widget_id}" style="
            flex:0 0 auto;
            width:{width}px;
            background:#1a1a2e;
            border-radius:12px;
            overflow:hidden;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);
            font-family:'Inter','Segoe UI',system-ui,sans-serif;
        ">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      padding:10px 16px;
                      background:linear-gradient(135deg,#16213e,#0f3460);
                      color:#e0e0e0;font-size:13px;">
            <span><strong>Layout Detection</strong> &mdash; {page_info}</span>
            <span style="opacity:.7;">{chunk_count} regions (≥ {min_confidence})</span>
          </div>
          <div style="position:relative;width:100%;">
            <img src="data:image/png;base64,{document.metadata.page_image_base64}"
                 style="display:block;width:100%;height:auto;" />
            {"".join(regions_html)}
            <div id="{widget_id}-tooltip" style="
              display:none;position:absolute;z-index:10;
              max-width:320px;padding:10px 14px;
              background:rgba(22,33,62,0.95);
              border:1px solid #3a4a7a;border-radius:8px;
              color:#f0f0f0;font-size:13px;line-height:1.5;
              pointer-events:none;backdrop-filter:blur(6px);
              box-shadow:0 4px 16px rgba(0,0,0,0.5);
            "></div>
          </div>
          <div style="padding:8px 16px;background:#16213e;display:flex;flex-wrap:wrap;align-items:center;">
            <span style="font-size:12px;color:#8899bb;margin-right:10px;">Legend:</span>
            {legend_items}
          </div>
        </div>""")

    ipy_display(HTML(f"""
    <div id="{container_id}" style="
        display:flex;
        flex-direction:row;
        gap:16px;
        overflow-x:auto;
        padding:12px 4px;
        scrollbar-width:thin;
        scrollbar-color:#3a4a7a #16213e;
    ">
      {"".join(widgets_html)}
    </div>
    """))