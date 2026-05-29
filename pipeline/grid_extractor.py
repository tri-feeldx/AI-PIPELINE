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


def _is_valid_grid_axis(sorted_by_pos: list[tuple[str, float]]) -> bool:
    """Universal structural grid validity — two physical principles:

    A) Monotonic order: column/row labels in position order must also be in
       value order (column 3 is always right of column 2).
    B) Spacing uniformity: structural bays are approximately equal
       (coefficient of variation of gaps must be < 0.60).

    Rejects page reference numbers, detail callout numbers, and other
    non-grid text that passes font-size and border-proximity filters.
    """
    if len(sorted_by_pos) < 2:
        return False

    labels    = [lbl for lbl, _ in sorted_by_pos]
    positions = [pos for _, pos in sorted_by_pos]

    # Principle A: value order matches position order
    if all(lbl.isdigit() for lbl in labels):
        vals = [int(lbl) for lbl in labels]
        n    = len(vals)
        inversions = sum(
            1 for i in range(n) for j in range(i + 1, n) if vals[i] > vals[j]
        )
        total_pairs = n * (n - 1) / 2
        if inversions / total_pairs > 0.30:
            return False   # >30% inversions → reference numbers, not grid
    elif all(lbl.isalpha() and len(lbl) == 1 for lbl in labels):
        for i in range(len(labels) - 1):
            if ord(labels[i + 1]) - ord(labels[i]) > 3:
                return False   # alphabet gap > 3 → not a sequential grid

    # Principle B: spacing uniformity (CV < 0.60)
    if len(positions) >= 3:
        gaps     = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap > 0:
            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            cv = (variance ** 0.5) / mean_gap
            if cv > 0.60:
                return False   # irregular spacing → not a grid axis

    return True


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
    """Find scale ratio from titleblock text, e.g. 'SCALE: 1 : 100' or '1:80' → int."""
    texts   = _extract_texts(page)
    all_text = " ".join(t for t, *_ in texts)

    # Primary: explicit SCALE keyword
    m = re.search(r"SCALE\s*[:\s]+\s*1\s*[:/\s]+\s*(\d{2,4})\b", all_text, re.I)
    if m:
        return int(m.group(1))

    # Secondary: bare ratio "1 : 80", "1:80", "1/80" (no keyword needed)
    # Require word boundary before the 1 and after the denominator
    for m in re.finditer(r"(?<![:\d])\b1\s*[:/]\s*(\d{2,4})\b(?![\d:])", all_text):
        val = int(m.group(1))
        if 10 <= val <= 500:      # reasonable structural drawing scale
            return val
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

    # Some PDFs (Bluebeam-combined) have content OUTSIDE the declared MediaBox.
    # Expand content bounds to include all actual text before computing borders.
    if texts:
        content_h = max(h, max(y for _, _, y, _ in texts) + 10)
        content_w = max(w, max(x for _, x, _, _ in texts) + 10)
    else:
        content_h, content_w = h, w

    # Grid labels: large font (>12pt), near the drawing border.
    # Support single-char (1, A) AND multi-char (AA, AB, 10, 11) labels.
    # Numbers OR letters near TOP/BOTTOM  → X-axis (vertical grid lines).
    # Letters near LEFT/RIGHT (but NOT top/bottom) → Y-axis (horizontal grid lines).
    x_candidates: list[tuple[str, float]] = []  # (label, x_pos)
    y_candidates: list[tuple[str, float]] = []  # (label, y_pos)

    for text, x, y, sz in texts:
        if sz < 12:  # skip small text
            continue
        clean = text.strip().upper()
        # Allow 1–3 character grid labels; must be all-digits or all-letters
        if len(clean) == 0 or len(clean) > 3:
            continue
        if not (clean.isdigit() or clean.isalpha()):
            continue

        near_top_bot = (y < content_h * 0.15 or y > content_h * 0.85)
        near_side    = (x < content_w * 0.15 or x > content_w * 0.65)

        if clean.isdigit() and near_top_bot:
            # Numbers near top/bottom → X-axis (standard convention)
            x_candidates.append((clean, x))
        elif clean.isalpha():
            if near_side and not near_top_bot:
                # Single/multi-char letters at left/right only → Y-axis
                y_candidates.append((clean, y))
            elif near_top_bot and len(clean) > 1:
                # Multi-char letters (AA, AB, BA...) at top/bottom edge → X-axis
                # Skip single-char to avoid section marks (A-A, B-B cutlines)
                x_candidates.append((clean, x))

    # Deduplicate by label (keep first occurrence per label)
    seen_x: dict[str, float] = {}
    for label, pos in sorted(x_candidates, key=lambda a: a[1]):
        if label not in seen_x:
            seen_x[label] = pos

    seen_y: dict[str, float] = {}
    for label, pos in sorted(y_candidates, key=lambda a: a[1]):
        if label not in seen_y:
            seen_y[label] = pos

    # Sort by position, then validate as genuine structural grid axes
    sorted_x = sorted(seen_x.items(), key=lambda a: a[1])
    sorted_y = sorted(seen_y.items(), key=lambda a: a[1])
    if not _is_valid_grid_axis(sorted_x):
        sorted_x = []   # rejects page ref numbers / detail callouts
    if not _is_valid_grid_axis(sorted_y):
        sorted_y = []

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


def extract_grids_from_pdf(
    pdf_path: str,
    plan_page_indices: list[int] | None = None,
    dominant_scale: int | None = None,
) -> dict:
    """Extract the best grid from the entire PDF.

    Uses the dominant structural plan scale (auto-detected if not provided).
    Accepts scales 50–250 so that 1:80 and 1:200 drawings are handled.
    Detail pages at 1:5, 1:10, 1:20 are skipped automatically.

    plan_page_indices: 0-based page indices to restrict to (optional).
    dominant_scale: override auto-detected scale (optional).
    """
    from collections import Counter

    doc = fitz.open(pdf_path)

    # Auto-detect dominant scale if not provided
    if dominant_scale is None:
        detected: list[int] = []
        for i in range(doc.page_count):
            s = extract_scale(doc[i])
            if s is not None and 50 <= s <= 250:
                detected.append(s)
        dominant_scale = Counter(detected).most_common(1)[0][0] if detected else 100

    master_scale = dominant_scale

    # Count how many times each label appears across plan-scale pages
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
        # Skip detail pages (< 50) and very small-scale overview pages (> 250)
        # Also skip pages at scales very different from master (e.g. master=100, skip 20)
        if page_scale is not None:
            if page_scale < 50 or page_scale > 250:
                continue
            # Accept pages within 30% of master scale
            if abs(page_scale - master_scale) / master_scale > 0.30:
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

    # Sort, validate merged axis (re-run after multi-page merge), and re-zero-base
    def _sort_labels(axes: dict[str, dict]) -> list[dict]:
        items = sorted(axes.values(), key=lambda a: a["pdf_pos"])
        if not items:
            return []
        # Re-validate the merged set: page-level validation may pass on
        # small subsets, but the merged labels must also be globally valid.
        pairs = [(a["label"], a["pdf_pos"]) for a in items]
        if not _is_valid_grid_axis(pairs):
            return []
        base = items[0]["real_mm"]
        for a in items:
            a["real_mm"] = round(a["real_mm"] - base, 1)
        return items

    doc.close()

    return {
        "x_axes":  _sort_labels(all_x),
        "y_axes":  _sort_labels(all_y),
        "scale": master_scale,
        "pt_to_mm": round(PT_TO_MM * master_scale, 4),
        "source_pages": source_pages,
    }
