"""Structural element detector — reads columns, beams, slabs from PDF vector paths.

Uses PyMuPDF drawing paths and text positions to detect structural elements
with exact coordinates. No AI required for this stage.

Column detection:
  Filled ~17×17 pt light-gray squares at grid intersections (mark column positions).

Beam detection:
  Beam section labels (e.g. "40b CH", "310UB40") placed at beam midpoints.
  Position of label text → which grid bay → which grid line pair the beam spans.

Slab detection:
  Large filled gray polygons covering panel areas between grid lines.
"""

import re
from typing import NamedTuple

import fitz


# ── Colour constants ───────────────────────────────────────────────────────────

# Column symbol fill colours in PDF
_COLUMN_FILLS = [
    (0.94, 0.94, 0.94),   # light gray (filled UC/SHS columns)
    (1.0,  1.0,  1.0),    # white (hollow section)
]
_COLUMN_MIN_SZ = 8    # min side length in pts
_COLUMN_MAX_SZ = 30   # max side length in pts

# Beam label pattern: "40b CH", "310UB40", "UB15a", "36b UB", etc.
_BEAM_LABEL_RE = re.compile(
    r"\b(?:\d+[a-zA-Z]+\s*[a-zA-Z]+|\d+\s*[A-Z]{2,3}\s*\d*|[A-Z]+\d+[a-zA-Z]*)\b"
)
_SECTION_RE = re.compile(
    r"(?:(\d+[a-zA-Z]+\s*[A-Z]+)|(\d+\s*(?:UB|UC|CH|SH|SHS|RHS|CHS|PFC|EA|UA)\s*\d*))"
    r"|(?:((?:UB|UC|CH|SH|SHS|RHS|CHS|PFC)\s*\d+[a-zA-Z]*))",
    re.I
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colour_close(c1, c2, tol=0.05) -> bool:
    if c1 is None or c2 is None:
        return False
    return all(abs(a - b) < tol for a, b in zip(c1, c2))


def _nearest_grid(pos: float, axes: list[dict], key="pdf_pos") -> dict | None:
    """Return the axis entry whose pdf_pos is closest to pos."""
    if not axes:
        return None
    return min(axes, key=lambda a: abs(a[key] - pos))


def _grid_mm(axis_entry: dict | None) -> float:
    if axis_entry is None:
        return 0.0
    return axis_entry.get("real_mm", 0.0)


# ── Column detection ──────────────────────────────────────────────────────────

def detect_columns(page: fitz.Page, grid: dict) -> list[dict]:
    """Find column symbols on a plan page.

    Column symbols are small filled squares at grid intersections.
    Returns list of {grid_ref, x_mm, y_mm, x_label, y_label}.
    """
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    if not x_axes or not y_axes:
        return []

    columns = []
    seen = set()

    for path in page.get_drawings():
        # Must be a filled rectangle (type 'f')
        if path.get("type") != "f":
            continue
        rect = path.get("rect")
        if rect is None:
            continue

        w_pts = rect.width
        h_pts = rect.height

        # Column symbol: approximately square, 8-30 pts
        if not (_COLUMN_MIN_SZ <= w_pts <= _COLUMN_MAX_SZ and
                _COLUMN_MIN_SZ <= h_pts <= _COLUMN_MAX_SZ):
            continue

        # Aspect ratio roughly square
        if max(w_pts, h_pts) / max(min(w_pts, h_pts), 0.1) > 2.5:
            continue

        fill = path.get("fill")
        is_col_colour = (
            fill is not None and
            (any(_colour_close(fill, c) for c in _COLUMN_FILLS) or
             all(0.85 < v <= 1.0 for v in fill[:3]))  # any light gray/white
        )
        if not is_col_colour:
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # Match to nearest grid intersection
        x_ax = _nearest_grid(cx, x_axes)
        y_ax = _nearest_grid(cy, y_axes)

        if x_ax is None or y_ax is None:
            continue

        # Snap tolerance: must be within 1.5× the symbol size of the grid line
        if abs(cx - x_ax["pdf_pos"]) > w_pts * 2:
            continue
        if abs(cy - y_ax["pdf_pos"]) > h_pts * 2:
            continue

        key = (x_ax["label"], y_ax["label"])
        if key in seen:
            continue
        seen.add(key)

        columns.append({
            "grid_ref": f"{y_ax['label']}/{x_ax['label']}",
            "x_label": x_ax["label"],
            "y_label": y_ax["label"],
            "x_mm": _grid_mm(x_ax),
            "y_mm": _grid_mm(y_ax),
            "symbol_size_pts": round((w_pts + h_pts) / 2, 1),
        })

    return columns


# ── Beam detection ────────────────────────────────────────────────────────────

def _find_section_label(text: str) -> str | None:
    """Extract a structural section label from a text string."""
    m = _SECTION_RE.search(text)
    if m:
        return m.group(0).strip()
    return None


def detect_beams(page: fitz.Page, grid: dict) -> list[dict]:
    """Find beams by locating section label text and mapping to grid bays.

    Beam labels (e.g. "40b CH") are placed at the midpoint of the beam on plan.
    Their position tells us which grid bay they occupy and their orientation.
    """
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    if len(x_axes) < 2 or len(y_axes) < 2:
        return []

    # Get all text with positions
    all_texts = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    all_texts.append((t, span["origin"][0], span["origin"][1], span["size"]))

    beams = []

    # Group nearby small text spans that form beam labels
    # Beam labels are small (6-12pt) and appear between grid lines
    label_candidates = [(t, x, y) for t, x, y, sz in all_texts if 5 < sz < 13]

    processed_positions = set()

    for text, lx, ly, *_ in [(t, x, y) for t, x, y in label_candidates]:
        section = _find_section_label(text)
        if not section:
            continue

        pos_key = (round(lx, 0), round(ly, 0))
        if pos_key in processed_positions:
            continue
        processed_positions.add(pos_key)

        # Determine which bays (X and Y) this label is in
        x_ax_left  = max((a for a in x_axes if a["pdf_pos"] <= lx), key=lambda a: a["pdf_pos"], default=None)
        x_ax_right = min((a for a in x_axes if a["pdf_pos"] >= lx), key=lambda a: a["pdf_pos"], default=None)
        y_ax_top   = max((a for a in y_axes if a["pdf_pos"] <= ly), key=lambda a: a["pdf_pos"], default=None)
        y_ax_bot   = min((a for a in y_axes if a["pdf_pos"] >= ly), key=lambda a: a["pdf_pos"], default=None)

        if x_ax_left is None or x_ax_right is None:
            continue
        if y_ax_top is None or y_ax_bot is None:
            continue

        # Determine beam orientation from its position within the bay
        bay_w = x_ax_right["pdf_pos"] - x_ax_left["pdf_pos"]
        bay_h = y_ax_bot["pdf_pos"] - y_ax_top["pdf_pos"]

        if bay_w < 1 and bay_h < 1:
            continue

        rel_x = (lx - x_ax_left["pdf_pos"]) / max(bay_w, 1)  # 0=left, 1=right
        rel_y = (ly - y_ax_top["pdf_pos"])  / max(bay_h, 1)   # 0=top, 1=bottom

        # Beam spanning X direction (left↔right): label near top or bottom of bay
        # Beam spanning Y direction (top↔bottom): label near left or right of bay
        EDGE_ZONE = 0.35  # within 35% of edge = spanning in that direction

        if rel_y < EDGE_ZONE or rel_y > (1 - EDGE_ZONE):
            # Horizontal beam (spans left→right in this bay)
            y_ax = y_ax_top if rel_y < 0.5 else y_ax_bot
            beams.append({
                "grid_ref": f"{y_ax['label']}/{x_ax_left['label']}-{x_ax_right['label']}",
                "section_label": section,
                "from_x_mm": _grid_mm(x_ax_left),
                "from_y_mm": _grid_mm(y_ax),
                "to_x_mm":   _grid_mm(x_ax_right),
                "to_y_mm":   _grid_mm(y_ax),
                "orientation": "x",
                "label_pos_pts": (round(lx, 1), round(ly, 1)),
            })
        else:
            # Vertical beam (spans top→bottom in this bay)
            x_ax = x_ax_left if rel_x < 0.5 else x_ax_right
            beams.append({
                "grid_ref": f"{y_ax_top['label']}-{y_ax_bot['label']}/{x_ax['label']}",
                "section_label": section,
                "from_x_mm": _grid_mm(x_ax),
                "from_y_mm": _grid_mm(y_ax_top),
                "to_x_mm":   _grid_mm(x_ax),
                "to_y_mm":   _grid_mm(y_ax_bot),
                "orientation": "y",
                "label_pos_pts": (round(lx, 1), round(ly, 1)),
            })

    return beams


# ── Slab detection ────────────────────────────────────────────────────────────

def detect_slabs(page: fitz.Page, grid: dict) -> list[dict]:
    """Detect slab panels as large filled polygons covering grid bays.

    A slab panel covers at least one full grid bay. We look for filled paths
    whose bounding box aligns with grid lines.
    """
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    if len(x_axes) < 2 or len(y_axes) < 2:
        return []

    # Minimum panel area: at least 50% of one grid bay
    if len(x_axes) >= 2 and len(y_axes) >= 2:
        bay_w = abs(x_axes[1]["pdf_pos"] - x_axes[0]["pdf_pos"])
        bay_h = abs(y_axes[1]["pdf_pos"] - y_axes[0]["pdf_pos"])
        min_area = bay_w * bay_h * 0.4
    else:
        min_area = 1000

    slabs = []
    seen_rects = set()

    for path in page.get_drawings():
        if path.get("type") not in ("f", "fs"):
            continue
        rect = path.get("rect")
        if rect is None or rect.is_empty:
            continue

        area = rect.width * rect.height
        if area < min_area:
            continue

        fill = path.get("fill")
        if fill is None:
            continue

        # Slab fill: medium gray (RC slab) or hatched areas
        if not (0.6 < fill[0] < 0.98 and 0.6 < fill[1] < 0.98 and 0.6 < fill[2] < 0.98):
            continue

        rect_key = (round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1))
        if rect_key in seen_rects:
            continue
        seen_rects.add(rect_key)

        x_from = _nearest_grid(rect.x0, x_axes)
        x_to   = _nearest_grid(rect.x1, x_axes)
        y_from = _nearest_grid(rect.y0, y_axes)
        y_to   = _nearest_grid(rect.y1, y_axes)

        if None in (x_from, x_to, y_from, y_to):
            continue

        grid_ref = (
            f"{y_from['label']}-{y_to['label']}/"
            f"{x_from['label']}-{x_to['label']}"
        )

        slabs.append({
            "grid_ref": grid_ref,
            "section_label": "RC SLAB",
            "from_x_mm": _grid_mm(x_from),
            "from_y_mm": _grid_mm(y_from),
            "to_x_mm":   _grid_mm(x_to),
            "to_y_mm":   _grid_mm(y_to),
            "rect_pts": rect_key,
        })

    return slabs


# ── Main dispatcher ───────────────────────────────────────────────────────────

def detect_elements(page: fitz.Page, page_type: str, grid: dict) -> dict:
    """Run all detectors appropriate for this page type."""
    result: dict = {
        "columns": [],
        "beams":   [],
        "slabs":   [],
    }

    plan_types = {"floor_plan", "roof_plan", "foundation_plan"}
    if page_type not in plan_types:
        return result

    result["columns"] = detect_columns(page, grid)
    result["beams"]   = detect_beams(page, grid)
    result["slabs"]   = detect_slabs(page, grid)

    return result
