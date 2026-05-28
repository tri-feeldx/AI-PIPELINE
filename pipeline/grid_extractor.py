"""Grid extractor — reads exact grid axis positions from PDF vector text.

Grid labels (A, B, C... / 1, 2, 3...) are printed at large font size near
the drawing border. Their pixel positions on the page directly encode the
real structural grid coordinates (after scale correction).

Scale formula:
  real_mm = pdf_pts × (25.4 / 72) × scale_denominator
  e.g. at 1:100 →  real_mm = pdf_pts × 0.3528 × 100 = pdf_pts × 35.28

Output: grid dict with x_axes and y_axes in real mm, zero-based.
"""

import re
from typing import NamedTuple

import fitz


PT_TO_MM = 25.4 / 72.0  # 1 PDF point in millimetres


class GridAxis(NamedTuple):
    label: str
    pdf_pos: float   # position on PDF page (x for vertical grid lines, y for horizontal)
    real_mm: float   # position in real-world mm (zero-based)


def _extract_texts(page: fitz.Page) -> list[tuple[str, float, float, float]]:
    """Return [(text, x, y, size)] for all text spans."""
    items = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    items.append((t, span["origin"][0], span["origin"][1], span["size"]))
    return items


def extract_scale(page: fitz.Page) -> int | None:
    """Find scale ratio from titleblock text, e.g. 'SCALE: 1 : 100' → 100."""
    texts = _extract_texts(page)
    all_text = " ".join(t for t, *_ in texts)
    # Must have the word SCALE (or NTS) to avoid matching grid labels "1 2 3"
    m = re.search(r"SCALE\s*[:\s]+\s*1\s*[:/\s]+\s*(\d{2,4})\b", all_text, re.I)
    if m:
        return int(m.group(1))
    return None


def extract_grid(page: fitz.Page, scale: int = 100) -> dict:
    """Extract structural grid axes from a plan page.

    Returns:
        {
          "x_axes": [{"label": "1", "pdf_pos": 533.0, "real_mm": 0.0}, ...],
          "y_axes": [{"label": "A", "pdf_pos": 416.0, "real_mm": 0.0}, ...],
          "scale": 100,
          "pt_to_mm": 35.28,
        }
    """
    texts = _extract_texts(page)
    w, h = page.rect.width, page.rect.height
    pt_to_mm = PT_TO_MM * scale

    # Grid labels are: single character, large font (>14pt), near the drawing border
    # Numbers (1,2,3...) appear near the TOP or BOTTOM border → X-axis grid
    # Letters (A,B,C...) appear near the LEFT or RIGHT border → Y-axis grid
    x_candidates: list[tuple[str, float]] = []  # (label, x_pos)
    y_candidates: list[tuple[str, float]] = []  # (label, y_pos)

    for text, x, y, sz in texts:
        if sz < 12:  # skip small text
            continue
        clean = text.strip().upper()
        if len(clean) != 1:
            continue

        # Grid number labels: near top border (y < 15% of height)
        if clean.isdigit() and y < h * 0.15:
            x_candidates.append((clean, x))

        # Grid letter labels: near right border (x > 65% of width) or left border
        elif clean.isalpha() and (x > w * 0.65 or x < w * 0.1):
            y_candidates.append((clean, y))

    # Deduplicate by label (keep first occurrence per label)
    seen_x: dict[str, float] = {}
    for label, pos in sorted(x_candidates, key=lambda a: a[1]):
        if label not in seen_x:
            seen_x[label] = pos

    seen_y: dict[str, float] = {}
    for label, pos in sorted(y_candidates, key=lambda a: a[1]):
        if label not in seen_y:
            seen_y[label] = pos

    # Sort by position
    sorted_x = sorted(seen_x.items(), key=lambda a: a[1])  # by x pos
    sorted_y = sorted(seen_y.items(), key=lambda a: a[1])  # by y pos

    # Build zero-based real mm coordinates
    def _to_real(items: list[tuple[str, float]], base_pos: float) -> list[dict]:
        return [
            {
                "label": label,
                "pdf_pos": round(pos, 2),
                "real_mm": round((pos - base_pos) * pt_to_mm, 1),
            }
            for label, pos in items
        ]

    x_base = sorted_x[0][1] if sorted_x else 0
    y_base = sorted_y[0][1] if sorted_y else 0

    return {
        "x_axes": _to_real(sorted_x, x_base),
        "y_axes": _to_real(sorted_y, y_base),
        "scale": scale,
        "pt_to_mm": round(pt_to_mm, 4),
        "page_size_pts": (round(w, 1), round(h, 1)),
    }


def extract_grids_from_pdf(pdf_path: str, plan_page_indices: list[int] | None = None) -> dict:
    """Extract the best grid from the entire PDF.

    Only uses pages at 1:100 scale (structural plan scale) to avoid
    contamination from detail drawings at other scales.

    plan_page_indices: 0-based page indices to restrict to (optional).
    """
    doc = fitz.open(pdf_path)
    master_scale = 100  # structural plans are always 1:100

    # Count how many times each label appears across 1:100 pages
    label_count_x: dict[str, int] = {}
    label_count_y: dict[str, int] = {}
    all_x: dict[str, dict] = {}
    all_y: dict[str, dict] = {}
    source_pages: list[int] = []

    for i in range(doc.page_count):
        if plan_page_indices is not None and i not in plan_page_indices:
            continue

        page = doc[i]
        page_scale = extract_scale(page)
        # Only use 1:100 pages (skip detail pages at 1:10, 1:20 etc.)
        if page_scale is not None and page_scale != master_scale:
            continue

        g = extract_grid(page, master_scale)
        if not g["x_axes"] and not g["y_axes"]:
            continue
        source_pages.append(i + 1)

        for ax in g["x_axes"]:
            lbl = ax["label"]
            label_count_x[lbl] = label_count_x.get(lbl, 0) + 1
            if lbl not in all_x:
                all_x[lbl] = ax

        for ax in g["y_axes"]:
            lbl = ax["label"]
            label_count_y[lbl] = label_count_y.get(lbl, 0) + 1
            if lbl not in all_y:
                all_y[lbl] = ax

    # Only keep labels that appear on at least 2 pages (filter one-off noise)
    # Exception: if very few pages processed, keep all
    min_count = 2 if len(source_pages) >= 4 else 1
    all_x = {k: v for k, v in all_x.items() if label_count_x.get(k, 0) >= min_count}
    all_y = {k: v for k, v in all_y.items() if label_count_y.get(k, 0) >= min_count}

    # Sort and re-zero-base
    def _sort_labels(axes: dict[str, dict]) -> list[dict]:
        items = sorted(axes.values(), key=lambda a: a["pdf_pos"])
        if not items:
            return []
        base = items[0]["real_mm"]
        for a in items:
            a["real_mm"] = round(a["real_mm"] - base, 1)
        return items

    doc.close()

    return {
        "x_axes": _sort_labels(all_x),
        "y_axes": _sort_labels(all_y),
        "scale": master_scale,
        "pt_to_mm": round(PT_TO_MM * master_scale, 4),
        "source_pages": source_pages,
    }
