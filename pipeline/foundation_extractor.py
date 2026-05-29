"""Foundation plan extractor — reads footing types, dimensions, and positions.

Handles Australian structural drawing conventions (AS3600, AS2159):
  - Bored piers / driven piles  (P1, P2 … Pn)
  - Capping beams               (CB1, CB2 …)
  - Raft / pad / strip footings (RF1, F1, …)
  - Ground / strap beams        (DM1, GB1, …)

Column-based schedule parsing for Australian pile schedules:
  MARK | PILE SIZE (DIA.) | SOCKET LENGTH | COMPRESSION | …
  P1   | 750              | 0.6           | 850         | …

Off-grid pile detection: finds ALL pile annotations on the page,
not just those at grid intersections.
"""

import re
from typing import Optional
import fitz

PT_TO_MM = 25.4 / 72.0

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Pile / footing type labels
_PILE_MARK_RE    = re.compile(r'^P(\d+[A-Za-z]?)$',  re.IGNORECASE)   # P1, P2a
# Narrow: require explicit prefix before digits; reject bare single letters followed by
# numbers in dimension context (e.g. F400 in "F'c=400", M24 bolts, P800 pressure).
# Negative lookbehind: not after =, ×, x (dimension context)
# Negative lookahead: not before kN, MPa, mm, kg (force/unit context)
_FOOTING_TYPE_RE = re.compile(
    r'(?<![=×xX\d])'
    r'\b(?:PF|PC|MC|M[DĐ]|MB|SF|RF|CB|F|P|M)(\d+[A-Za-z]?)\b'
    r'(?!\s*(?:kN|MPa|mm|kg|m\b))',
    re.IGNORECASE,
)

# Pile diameter with explicit prefix: Ø750, Φ600
_PILE_DIA_PREFIX_RE = re.compile(r'[ØΦφ](\d{2,4})|(?<![A-Za-z])D(\d{2,4})(?![A-Za-z])')

# Inline plan dimensions: 1200x1200, 1200×1200x500
_DIMS_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*[xX×*]\s*(\d+(?:\.\d+)?)'
    r'(?:\s*[xX×*]\s*(\d+(?:\.\d+)?))?'
)

# Pile length / socket length: L=20000, L=20m, 19.0 (bare float in schedule row)
_PILE_LEN_RE  = re.compile(r'\bL\s*=\s*(\d+(?:\.\d+)?)\s*(m|mm)?\b', re.IGNORECASE)

# Ground / capping beam labels: DM1, DG1, GB1, CB1
_GBEAM_RE = re.compile(
    r'\b(?:DM|DG|GB|ĐM|DĐ|GM|CB)\s*-?\s*(\d+[A-Za-z]?)\b',
    re.IGNORECASE,
)

# Drawing-area margin base (pts past rightmost grid line → schedule territory).
# Actual margin is computed dynamically as max(60, 8% of grid span).
_DRAWING_X_MARGIN_BASE_PTS = 60


# ── Text utilities ─────────────────────────────────────────────────────────────

def _all_text(page: fitz.Page) -> list[tuple[str, float, float, float]]:
    """Return [(text, x, y, font_size)] for every text span on the page."""
    items: list[tuple[str, float, float, float]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    items.append((
                        t,
                        span["origin"][0],
                        span["origin"][1],
                        span["size"],
                    ))
    return items


def _group_rows(
    spans: list[tuple[str, float, float, float]],
    tolerance: float = 4.0,
) -> dict[float, list[tuple[str, float]]]:
    """Group text spans into horizontal rows (same Y ± tolerance)."""
    rows: dict[float, list[tuple[str, float]]] = {}
    for t, x, y, _ in spans:
        key = round(y / tolerance) * tolerance   # symmetric rounding; was int()*int() which broke on floats
        rows.setdefault(key, []).append((t, x))
    return rows


# ── Schedule parsers ───────────────────────────────────────────────────────────

def _extract_pile_dia_from_row(row_text: str) -> float:
    """Extract pile diameter from a schedule row.

    Handles both:
      - Prefixed:  Ø750  Φ600
      - Plain:     750   600   (Australian pile schedule column value)
    Returns diameter in mm, or 0.0 if not found.
    """
    # Prefixed first (unambiguous)
    m = _PILE_DIA_PREFIX_RE.search(row_text)
    if m:
        return float(m.group(1) or m.group(2))

    # Column-based: first integer token in typical pile diameter range.
    # Range 250–2000mm covers: micropiles (250-300), standard (400-1200), caissons (1500-2000).
    for tok in row_text.split():
        try:
            v = float(tok)
            if 250 <= v <= 2000 and v == int(v):
                return v
        except ValueError:
            pass
    return 0.0


def _extract_socket_len_from_row(row_text: str) -> float:
    """Extract socket/pile length from a schedule row. Returns mm."""
    # Explicit pattern: L=20m, L=20000
    m = _PILE_LEN_RE.search(row_text)
    if m:
        length = float(m.group(1))
        unit   = (m.group(2) or "m").lower()
        return length * 1000 if unit == "m" else length

    # Schedule: first small float token after the mark label = socket/pile length in metres.
    # Range 0.05–80m: covers shallow AU piles (~5-25m) and deep SE Asia piles (40-60m+).
    tokens = row_text.split()
    for tok in tokens:
        if _FOOTING_TYPE_RE.match(tok):
            continue
        try:
            v = float(tok)
            if 0.05 <= v <= 80.0 and tok not in ("0", "1"):
                return v * 1000   # metres → mm
        except ValueError:
            pass
    return 0.0


def parse_footing_schedule(page: fitz.Page) -> dict[str, dict]:
    """Parse footing / pile schedule from the page.

    Returns {LABEL → spec_dict} with keys:
        ftype, width_mm, depth_mm, height_mm,
        pile_dia_mm, pile_len_mm, pile_count
    """
    spans = _all_text(page)
    rows  = _group_rows(spans)
    schedule: dict[str, dict] = {}

    for y_key in sorted(rows):
        row_items = sorted(rows[y_key], key=lambda r: r[1])
        row_text  = " ".join(t for t, _ in row_items)

        m = _FOOTING_TYPE_RE.search(row_text)
        if not m:
            continue

        label = m.group(0).upper().replace(" ", "")
        if label in schedule:
            continue                             # keep first (header) row only

        ftype = _label_to_ftype(label)

        entry: dict = {
            "label":       label,
            "ftype":       ftype,
            "width_mm":    0.0,
            "depth_mm":    0.0,
            "height_mm":   0.0,
            "pile_dia_mm": 0.0,
            "pile_len_mm": 0.0,
            "pile_count":  0,
        }

        if ftype in ("pile_cap", "bored_pier"):
            # Two possible schedule formats:
            # A) Pile-centric:  "P1 | 750 | 0.6 | 850 | ..."
            #    Signature: first large number (300-1050) is followed by a
            #    small float (0.05-25) = socket length in metres.
            # B) Cap-centric:   "PC1 | 1400 | 1400 | 1100 | 65"
            #    Signature: 3 large numbers of similar magnitude, no small float.
            tokens = row_text.split()
            is_pile_format = False
            pile_dia_candidate = 0.0
            for idx, tok in enumerate(tokens):
                try:
                    v = float(tok)
                    if 200 <= v <= 2000 and v == int(v):
                        # Check if next token is a small float (socket length in m)
                        for nxt in tokens[idx + 1:idx + 4]:
                            try:
                                nv = float(nxt)
                                if 0.05 <= nv <= 80.0:
                                    is_pile_format = True
                                    pile_dia_candidate = v
                                    break
                            except ValueError:
                                continue
                        if is_pile_format:
                            break
                except ValueError:
                    continue

            # Also honour explicit Ø/Φ prefix (unambiguous)
            m_dia = _PILE_DIA_PREFIX_RE.search(row_text)
            if m_dia:
                is_pile_format = True
                pile_dia_candidate = float(m_dia.group(1) or m_dia.group(2))

            if is_pile_format:
                # Format A — pile-diameter + socket length row
                entry["pile_dia_mm"] = pile_dia_candidate
                entry["pile_len_mm"] = _extract_socket_len_from_row(row_text)
                d = pile_dia_candidate
                if d > 0:
                    entry["width_mm"]  = round(d * 2.0)
                    entry["depth_mm"]  = round(d * 2.0)
                    entry["height_mm"] = round(d * 0.93)
            else:
                # Format B — cap W × L × D (pile spec is elsewhere / geotech report)
                large_nums = [float(t) for t, _ in row_items
                              if _is_plain_number(t) and float(t) >= 300]
                if len(large_nums) >= 3:
                    entry["width_mm"]  = large_nums[0]
                    entry["depth_mm"]  = large_nums[1]
                    entry["height_mm"] = large_nums[2]
                elif len(large_nums) == 2:
                    entry["width_mm"]  = large_nums[0]
                    entry["depth_mm"]  = large_nums[0]
                    entry["height_mm"] = large_nums[1]
                entry["pile_len_mm"] = _extract_socket_len_from_row(row_text)
                # pile_dia stays 0 — will be estimated in ruby_generator

        elif ftype == "pad_footing":
            # Australian pad footing schedule: WxDxH inline (e.g. 1200x1200x450)
            # or separate columns: WIDTH | LENGTH | DEPTH
            dims = _parse_dims(row_text)
            if dims:
                entry["width_mm"]  = dims["width_mm"]
                entry["depth_mm"]  = dims["depth_mm"]
                entry["height_mm"] = dims["height_mm"] if dims["height_mm"] > 0 else 450.0
            else:
                # Fallback: first 3 numbers ≥ 300mm in row → W, D, H
                nums = [float(t) for t, _ in row_items
                        if _is_plain_number(t) and float(t) >= 300]
                if len(nums) >= 3:
                    entry["width_mm"]  = nums[0]
                    entry["depth_mm"]  = nums[1]
                    entry["height_mm"] = nums[2]
                elif len(nums) == 2:
                    entry["width_mm"]  = nums[0]
                    entry["depth_mm"]  = nums[0]   # square if only one plan dim
                    entry["height_mm"] = nums[1]

        elif ftype == "strip_footing":
            dims = _parse_dims(row_text)
            if dims:
                entry["width_mm"]  = dims["width_mm"]
                entry["height_mm"] = dims["depth_mm"] if dims["depth_mm"] > 0 else 400.0

        elif ftype == "capping_beam":
            # CB1: 750 WIDTH × 600 DEPTH
            nums = [float(t) for t, _ in row_items
                    if _is_plain_number(t) and 100 <= float(t) <= 2000]
            if len(nums) >= 2:
                entry["width_mm"]  = nums[0]
                entry["height_mm"] = nums[1]

        elif ftype == "raft":
            # RF1: LENGTH × WIDTH × DEPTH (thin rafts can be 100-300mm)
            nums = [float(t) for t, _ in row_items
                    if _is_plain_number(t) and float(t) >= 100]
            if len(nums) >= 3:
                entry["width_mm"]  = nums[1]   # plan width
                entry["depth_mm"]  = nums[0]   # plan length (stored as depth)
                entry["height_mm"] = nums[2]   # slab thickness

        elif ftype in ("pad_footing", "strip_footing"):
            dims = _parse_dims(row_text)
            entry.update(dims)

        schedule[label] = entry

    return schedule


def _is_plain_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ── Annotation detection ───────────────────────────────────────────────────────

def _compute_snap_radius(x_axes: list, y_axes: list) -> float:
    """Compute snap-to-grid radius from actual grid spacing.

    Uses 25% of the smallest detected bay width/height.
    Works for any scale: 1:50, 1:100, 1:200, etc.
    Falls back to 600mm when no grid is available.
    """
    spacings = []
    for axes in [x_axes, y_axes]:
        for i in range(len(axes) - 1):
            gap = axes[i + 1]["real_mm"] - axes[i]["real_mm"]
            if gap > 0:
                spacings.append(gap)
    return min(spacings) * 0.25 if spacings else 600.0


def find_all_pile_annotations(
    page: fitz.Page,
    grid: dict,
    snap_radius_mm: float | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Find ALL pile / footing annotations on the drawing (including off-grid).

    Returns:
        on_grid  : {grid_ref → label}   snapped to nearest intersection
        off_grid : list of {id, grid_ref, x_mm, y_mm, label}
    """
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    pt_to_mm = grid.get("pt_to_mm", PT_TO_MM * 100)

    # Compute snap radius adaptively from grid spacing (works for any scale)
    if snap_radius_mm is None:
        snap_radius_mm = _compute_snap_radius(x_axes, y_axes)

    spans = _all_text(page)
    page_w = page.rect.width

    # X threshold: annotations farther right than this are in the schedule/title block.
    # Dynamic margin: 8% of grid span, minimum 60pts.
    if x_axes:
        grid_span_pts = x_axes[-1]["pdf_pos"] - x_axes[0]["pdf_pos"]
        margin = max(_DRAWING_X_MARGIN_BASE_PTS, grid_span_pts * 0.08)
        x_max_drawing = x_axes[-1]["pdf_pos"] + margin
    else:
        x_max_drawing = page_w * 0.62

    # Grid origin in PDF pts (zero-based reference)
    gx0 = x_axes[0]["pdf_pos"] if x_axes else 0.0
    gy0 = y_axes[0]["pdf_pos"] if y_axes else 0.0

    on_grid: dict[str, str] = {}
    off_grid: list[dict] = []
    seen_positions: set[tuple] = set()

    for t, x, y, sz in spans:
        m = _FOOTING_TYPE_RE.match(t.strip())
        if not m:
            continue
        label = t.strip().upper().replace(" ", "")

        # Skip schedule / title-block area (to the right of drawing)
        if x > x_max_drawing:
            continue

        pos_key = (round(x), round(y))
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        # Convert to real_mm (zero-based from grid origin)
        rx = (x - gx0) * pt_to_mm
        ry = (y - gy0) * pt_to_mm

        # Try to snap to nearest grid intersection
        nearest_ref, nearest_dist = _nearest_grid_intersection(rx, ry, x_axes, y_axes)

        if nearest_dist <= snap_radius_mm and nearest_ref:
            # On-grid: keep closest label (in case of ties, first found wins)
            if nearest_ref not in on_grid:
                on_grid[nearest_ref] = label
        else:
            # Off-grid: store with absolute real_mm coords
            off_grid.append({
                "id":       f"FDN-OFF-{label}-{round(rx)}-{round(ry)}",
                "grid_ref": f"OFF/{label}@{round(rx)},{round(ry)}",
                "x_mm":     round(rx),
                "y_mm":     round(ry),
                "label":    label,
            })

    return on_grid, off_grid


def _nearest_grid_intersection(
    rx: float, ry: float,
    x_axes: list[dict],
    y_axes: list[dict],
) -> tuple[Optional[str], float]:
    """Return (grid_ref, distance_mm) of the nearest grid intersection."""
    if not x_axes or not y_axes:
        return None, float("inf")

    best_xa = min(x_axes, key=lambda a: abs(a["real_mm"] - rx))
    best_ya = min(y_axes, key=lambda a: abs(a["real_mm"] - ry))

    dist = ((rx - best_xa["real_mm"]) ** 2 + (ry - best_ya["real_mm"]) ** 2) ** 0.5
    ref  = f"{best_ya['label']}/{best_xa['label']}"
    return ref, dist


def find_pile_spec_global(page: fitz.Page) -> dict:
    """Extract a page-wide pile spec from note text (e.g. 'Ø600 BORED PIERS')."""
    all_str = " ".join(t for t, *_ in _all_text(page))

    spec: dict = {}
    m = _PILE_DIA_PREFIX_RE.search(all_str)
    if m:
        dia = float(m.group(1) or m.group(2))
        if dia < 30:
            dia *= 10
        spec["pile_dia_mm"] = dia

    m = _PILE_LEN_RE.search(all_str)
    if m:
        length = float(m.group(1))
        unit   = (m.group(2) or "m").lower()
        spec["pile_len_mm"] = length * 1000 if unit == "m" else length

    return spec


# ── Raft detection ─────────────────────────────────────────────────────────────

def detect_raft_foundations(
    page: fitz.Page,
    grid: dict,
    schedule: dict[str, dict],
) -> list[dict]:
    """Find raft foundation rectangles from vector drawings.

    Matches large gray-filled rectangles (area > 50 000 pts²) to RF entries
    in the schedule by dimension similarity.
    """
    x_axes  = grid.get("x_axes", [])
    y_axes  = grid.get("y_axes", [])
    pt_mm   = grid.get("pt_to_mm", PT_TO_MM * 100)
    gx0     = x_axes[0]["pdf_pos"] if x_axes else 0.0
    gy0     = y_axes[0]["pdf_pos"] if y_axes else 0.0

    rafts: list[dict] = []
    seen: set[tuple] = set()

    for path in page.get_drawings():
        rect = path.get("rect")
        if rect is None or rect.is_empty:
            continue
        if rect.width * rect.height < 20_000:
            continue
        # Both plan dimensions must be > 1500mm real (no grid lines / border lines)
        if rect.width * pt_mm < 1500 or rect.height * pt_mm < 1500:
            continue
        # Aspect ratio ≤ 4:1 (rafts are not strip-shaped)
        longer  = max(rect.width, rect.height)
        shorter = min(rect.width, rect.height)
        if shorter < 1 or longer / shorter > 4:
            continue
        fill = path.get("fill")
        if fill is None:
            continue
        # Gray fill (RF concrete hatching or outline)
        if not (0.80 <= fill[0] <= 0.99 and
                0.80 <= fill[1] <= 0.99 and
                0.80 <= fill[2] <= 0.99):
            continue

        key = (round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1))
        if key in seen:
            continue
        seen.add(key)

        w_mm = rect.width  * pt_mm
        h_mm = rect.height * pt_mm
        x_from = (rect.x0 - gx0) * pt_mm
        y_from = (rect.y0 - gy0) * pt_mm
        x_to   = (rect.x1 - gx0) * pt_mm
        y_to   = (rect.y1 - gy0) * pt_mm

        # Match against RF schedule entries
        matched_label = None
        matched_thickness = 1000.0
        for label, spec in schedule.items():
            if spec["ftype"] != "raft":
                continue
            sw = spec.get("width_mm", 0)
            sl = spec.get("depth_mm", 0)   # stored as depth = plan length
            if sw <= 0 or sl <= 0:
                continue
            # Either orientation
            dim_match = (
                (abs(w_mm - sw) / sw < 0.12 and abs(h_mm - sl) / sl < 0.12) or
                (abs(w_mm - sl) / sl < 0.12 and abs(h_mm - sw) / sw < 0.12)
            )
            if dim_match:
                matched_label     = label
                matched_thickness = spec.get("height_mm", 1000.0)
                break

        if matched_label is None:
            # No schedule match but large rectangle → add with defaults
            matched_label     = "RF?"
            matched_thickness = 1000.0

        rafts.append({
            "id":           f"RAFT-{matched_label}",
            "grid_ref":     f"RAFT/{matched_label}",
            "label":        matched_label,
            "ftype":        "raft",
            "x_from_mm":    round(x_from),
            "y_from_mm":    round(y_from),
            "x_to_mm":      round(x_to),
            "y_to_mm":      round(y_to),
            "x_mm":         round((x_from + x_to) / 2),
            "y_mm":         round((y_from + y_to) / 2),
            "width_mm":     round(w_mm),
            "depth_mm":     round(h_mm),
            "height_mm":    matched_thickness,
            "material":     "concrete",
        })

    return rafts


# ── Ground / capping beam detection ───────────────────────────────────────────

def detect_ground_beams(page: fitz.Page, grid: dict) -> list[dict]:
    """Find ground-beam and capping-beam labels (DM, GB, CB) between grid lines."""
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    if len(x_axes) < 2 or len(y_axes) < 2:
        return []

    spans = _all_text(page)
    beams: list[dict] = []
    seen:  set[tuple] = set()

    for t, lx, ly, _ in spans:
        m = _GBEAM_RE.search(t)
        if not m:
            continue

        key = (round(lx), round(ly))
        if key in seen:
            continue
        seen.add(key)

        label = m.group(0).upper().replace(" ", "")

        x_left  = max((a for a in x_axes if a["pdf_pos"] <= lx),
                      key=lambda a: a["pdf_pos"], default=None)
        x_right = min((a for a in x_axes if a["pdf_pos"] >= lx),
                      key=lambda a: a["pdf_pos"], default=None)
        y_top   = max((a for a in y_axes if a["pdf_pos"] <= ly),
                      key=lambda a: a["pdf_pos"], default=None)
        y_bot   = min((a for a in y_axes if a["pdf_pos"] >= ly),
                      key=lambda a: a["pdf_pos"], default=None)

        if None in (x_left, x_right, y_top, y_bot):
            continue

        bay_w = x_right["pdf_pos"] - x_left["pdf_pos"]
        bay_h = y_bot["pdf_pos"]   - y_top["pdf_pos"]
        if bay_w < 1 and bay_h < 1:
            continue

        rel_x = (lx - x_left["pdf_pos"]) / max(bay_w, 1)
        rel_y = (ly - y_top["pdf_pos"])  / max(bay_h, 1)

        nearby_row = " ".join(
            t2 for t2, x2, y2, _ in spans
            if abs(x2 - lx) < 120 and abs(y2 - ly) < 15
        )
        dims   = _parse_dims(nearby_row)
        w_mm   = dims.get("width_mm",  300.0)
        h_mm   = dims.get("depth_mm",  600.0)

        EDGE = 0.35
        if rel_y < EDGE or rel_y > (1 - EDGE):
            y_ax = y_top if rel_y < 0.5 else y_bot
            beams.append({
                "id":           f"GB-{label}",
                "grid_ref":     f"{y_ax['label']}/{x_left['label']}-{x_right['label']}",
                "section_label": label,
                "from_x_mm":    x_left["real_mm"],
                "from_y_mm":    y_ax["real_mm"],
                "to_x_mm":      x_right["real_mm"],
                "to_y_mm":      y_ax["real_mm"],
                "width_mm":     w_mm,
                "height_mm":    h_mm,
                "orientation":  "x",
            })
        else:
            x_ax = x_left if rel_x < 0.5 else x_right
            beams.append({
                "id":           f"GB-{label}",
                "grid_ref":     f"{y_top['label']}-{y_bot['label']}/{x_ax['label']}",
                "section_label": label,
                "from_x_mm":    x_ax["real_mm"],
                "from_y_mm":    y_top["real_mm"],
                "to_x_mm":      x_ax["real_mm"],
                "to_y_mm":      y_bot["real_mm"],
                "width_mm":     w_mm,
                "height_mm":    h_mm,
                "orientation":  "y",
            })

    return beams


# ── Main entry point ───────────────────────────────────────────────────────────

def extract_foundations(page: fitz.Page, grid: dict, global_schedule: dict | None = None) -> dict:
    """Extract all foundation elements from a foundation-plan page.

    Returns:
        footings           : list[dict]  pile caps, pad footings, off-grid piles
        ground_beams       : list[dict]  DM / GB / CB beams
        rafts              : list[dict]  raft slab elements
        schedule           : dict        parsed schedule table
        pile_spec          : dict        global pile spec from notes
        has_foundation_plan: bool
    """
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    pt_mm  = grid.get("pt_to_mm", PT_TO_MM * 100)

    # Parse schedule from this page, then merge with pre-built global schedule.
    # Global schedule wins for entries where page-level data has zero dims.
    page_schedule = parse_footing_schedule(page)
    schedule = dict(global_schedule) if global_schedule else {}
    for mark, spec in page_schedule.items():
        existing = schedule.get(mark, {})
        if (spec.get("width_mm", 0) > existing.get("width_mm", 0)
                or spec.get("pile_dia_mm", 0) > existing.get("pile_dia_mm", 0)):
            schedule[mark] = spec

    pile_spec     = find_pile_spec_global(page)
    on_grid, off_grid_anns = find_all_pile_annotations(page, grid)
    ground_beams  = detect_ground_beams(page, grid)
    rafts         = detect_raft_foundations(page, grid, schedule)

    has_piles = (
        any(v.get("ftype") in ("pile_cap", "bored_pier") or v.get("pile_dia_mm", 0) > 0
            for v in schedule.values())
        or pile_spec.get("pile_dia_mm", 0) > 0
    )

    # Single schedule type → apply to all intersections
    single_label: Optional[str] = None
    pile_labels = [k for k, v in schedule.items()
                   if v.get("ftype") in ("pile_cap", "bored_pier", "pad_footing")]
    if len(pile_labels) == 1:
        single_label = pile_labels[0]

    footings: list[dict] = []

    # Foundation element types that belong in footings list
    _FOOTING_FTYPES = {"pile_cap", "bored_pier", "pad_footing", "strip_footing"}

    # ── On-grid footings ────────────────────────────────────────────────────────
    for ya in y_axes:
        for xa in x_axes:
            grid_ref = f"{ya['label']}/{xa['label']}"
            label    = on_grid.get(grid_ref) or single_label
            spec     = _resolve_spec(label, schedule, pile_spec, has_piles)
            if spec is None:
                continue
            if spec.get("ftype") not in _FOOTING_FTYPES:
                continue

            footings.append({
                "id":          f"FDN-{ya['label']}{xa['label']}",
                "grid_ref":    grid_ref,
                "x_mm":        xa["real_mm"],
                "y_mm":        ya["real_mm"],
                "label":       label or "AUTO",
                "ftype":       spec["ftype"],
                "width_mm":    spec["width_mm"],
                "depth_mm":    spec["depth_mm"],
                "height_mm":   spec["height_mm"],
                "pile_dia_mm": spec.get("pile_dia_mm", pile_spec.get("pile_dia_mm", 0)),
                "pile_len_mm": spec.get("pile_len_mm", pile_spec.get("pile_len_mm", 0)),
                "pile_count":  spec.get("pile_count", 1),
                "material":    "concrete",
            })

    # ── Off-grid footings ───────────────────────────────────────────────────────
    for ann in off_grid_anns:
        label = ann["label"]
        spec  = _resolve_spec(label, schedule, pile_spec, has_piles)
        if spec is None:
            spec = _default_spec(has_piles, pile_spec)
        if spec.get("ftype") not in _FOOTING_FTYPES:
            continue   # CB → capping_beam, RF → raft; handled separately

        footings.append({
            "id":          ann["id"],
            "grid_ref":    ann["grid_ref"],
            "x_mm":        ann["x_mm"],
            "y_mm":        ann["y_mm"],
            "label":       label,
            "ftype":       spec["ftype"],
            "width_mm":    spec["width_mm"],
            "depth_mm":    spec["depth_mm"],
            "height_mm":   spec["height_mm"],
            "pile_dia_mm": spec.get("pile_dia_mm", pile_spec.get("pile_dia_mm", 0)),
            "pile_len_mm": spec.get("pile_len_mm", pile_spec.get("pile_len_mm", 0)),
            "pile_count":  spec.get("pile_count", 1),
            "material":    "concrete",
        })

    # ── Fallback: no annotations found → generate from grid ────────────────────
    # Only auto-fill ALL grid intersections when schedule has ≥3 entries
    # (meaning it is a real schedule, not noise). With fewer entries the
    # grid-fill produces massive false-positive counts; signal the caller
    # to try Vision AI instead.
    if not footings and x_axes and y_axes and (schedule or pile_spec):
        if len(schedule) >= 3:
            default = _default_spec(has_piles, pile_spec)
            for ya in y_axes:
                for xa in x_axes:
                    footings.append({
                        "id":          f"FDN-{ya['label']}{xa['label']}",
                        "grid_ref":    f"{ya['label']}/{xa['label']}",
                        "x_mm":        xa["real_mm"],
                        "y_mm":        ya["real_mm"],
                        "label":       "AUTO",
                        "ftype":       default["ftype"],
                        "width_mm":    default["width_mm"],
                        "depth_mm":    default["depth_mm"],
                        "height_mm":   default["height_mm"],
                        "pile_dia_mm": default.get("pile_dia_mm", 0),
                        "pile_len_mm": default.get("pile_len_mm", 0),
                        "pile_count":  default.get("pile_count", 1),
                        "material":    "concrete",
                    })

    # ── Lift pits ───────────────────────────────────────────────────────────────
    lift_pits = _detect_lift_pits(page, grid, pt_mm)

    # ── Planter box slab flag ───────────────────────────────────────────────────
    has_planter = bool(re.search(
        r'\bLANDSCAPE\s+SLAB\b|\bPLANTER\s+BOX\b', page.get_text(), re.I
    ))

    return {
        "footings":            footings,
        "ground_beams":        ground_beams,
        "rafts":               rafts,
        "lift_pits":           lift_pits,
        "has_planter_slab":    has_planter,
        "schedule":            schedule,
        "pile_spec":           pile_spec,
        "has_foundation_plan": bool(footings or ground_beams or rafts or schedule),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

_LIFT_PIT_RE = re.compile(
    r'\b(?:LIFT\s*PIT|ELEVATOR\s*PIT|LP)\b.*?(\d{3,4})\s*(?:DEEP|mm)',
    re.I | re.DOTALL,
)
_LIFT_PIT_LABEL_RE = re.compile(r'\bLP\s*[-/]?\s*(\d+)', re.I)


def _detect_lift_pits(page: fitz.Page, grid: dict, pt_mm: float) -> list[dict]:
    """Detect lift/elevator pit annotations on foundation plan.

    Returns list of {id, grid_ref, x_mm, y_mm, depth_mm, width_mm}.
    Lift pits are below the core raft — typically 1200-1500mm deeper.
    """
    text = page.get_text()
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    spans  = _all_text(page)

    # Build a coord lookup: label string → (x_pdf, y_pdf)
    label_coords: dict[str, tuple[float, float]] = {}
    for t, x, y, _ in spans:
        lm = _LIFT_PIT_LABEL_RE.match(t.strip())
        if lm:
            key = f"LP{lm.group(1)}"
            label_coords.setdefault(key, (x, y))

    pits: list[dict] = []

    for m in _LIFT_PIT_RE.finditer(text):
        depth_mm = int(m.group(1))
        if depth_mm < 500 or depth_mm > 3000:
            continue

        pit_id = f"LP-{len(pits)+1:02d}"

        # Try to find real position from label span coordinates
        lbl_key = f"LP{len(pits)+1}"
        if lbl_key in label_coords:
            px, py = label_coords[lbl_key]
            gx0 = x_axes[0]["pdf_pos"] if x_axes else 0.0
            gy0 = y_axes[0]["pdf_pos"] if y_axes else 0.0
            rx = (px - gx0) * pt_mm
            ry = (py - gy0) * pt_mm
            nearest_ref, dist = _nearest_grid_intersection(rx, ry, x_axes, y_axes)
            grid_ref = nearest_ref if (nearest_ref and dist < 1500) else "off_grid"
            x_mm, y_mm = round(rx), round(ry)
            needs_review = False
        else:
            grid_ref, x_mm, y_mm, needs_review = "off_grid", 0.0, 0.0, True

        pits.append({
            "id":           pit_id,
            "ftype":        "lift_pit",
            "depth_mm":     depth_mm,
            "width_mm":     1500,
            "height_mm":    depth_mm,
            "material":     "concrete",
            "grid_ref":     grid_ref,
            "x_mm":         x_mm,
            "y_mm":         y_mm,
            "needs_review": needs_review,
            "note":         "Lift pit — verify position from drawing",
        })

    return pits


def _label_to_ftype(label: str) -> str:
    lu = label.upper()
    if _PILE_MARK_RE.match(lu):                                   # P1, P2 …
        return "pile_cap"
    if any(lu.startswith(p) for p in ("PC", "MC", "MD", "MĐ")):
        return "pile_cap"
    if lu.startswith("CB"):
        return "capping_beam"
    if lu.startswith("RF"):
        return "raft"
    if lu.startswith("MB") or lu.startswith("SF"):
        return "strip_footing"
    if lu.startswith("PF"):
        return "pad_footing"
    return "pad_footing"


def _resolve_spec(
    label: Optional[str],
    schedule: dict[str, dict],
    pile_spec: dict,
    has_piles: bool,
) -> Optional[dict]:
    if label and label in schedule:
        return schedule[label].copy()
    if label:
        return _infer_from_label(label, pile_spec)
    return None


def _infer_from_label(label: str, pile_spec: dict) -> dict:
    lu = label.upper()
    if _PILE_MARK_RE.match(lu) or any(lu.startswith(p) for p in ("PC", "MC", "MD", "MĐ")):
        d = pile_spec.get("pile_dia_mm", 600)
        return {
            "ftype":       "pile_cap",
            "width_mm":    round(d * 2.0),
            "depth_mm":    round(d * 2.0),
            "height_mm":   round(d * 0.93),
            "pile_dia_mm": d,
            "pile_len_mm": pile_spec.get("pile_len_mm", 20000),
            "pile_count":  1,
        }
    if lu.startswith("MB") or lu.startswith("SF"):
        return {"ftype": "strip_footing", "width_mm": 800, "depth_mm": 0, "height_mm": 400,
                "pile_dia_mm": 0, "pile_len_mm": 0, "pile_count": 0}
    return {"ftype": "pad_footing", "width_mm": 1200, "depth_mm": 1200, "height_mm": 500,
            "pile_dia_mm": pile_spec.get("pile_dia_mm", 0),
            "pile_len_mm": pile_spec.get("pile_len_mm", 0), "pile_count": 0}


def _default_spec(has_piles: bool, pile_spec: dict) -> dict:
    if has_piles:
        d = pile_spec.get("pile_dia_mm", 600)
        return {
            "ftype": "pile_cap",
            "width_mm":    round(d * 2.0),
            "depth_mm":    round(d * 2.0),
            "height_mm":   round(d * 0.93),
            "pile_dia_mm": d,
            "pile_len_mm": pile_spec.get("pile_len_mm", 20000),
            "pile_count":  1,
        }
    return {"ftype": "pad_footing", "width_mm": 1200, "depth_mm": 1200, "height_mm": 500,
            "pile_dia_mm": 0, "pile_len_mm": 0, "pile_count": 0}


def _parse_dims(text: str) -> dict:
    m = _DIMS_RE.search(text)
    if not m:
        return {}
    w, d = float(m.group(1)), float(m.group(2))
    h = float(m.group(3)) if m.group(3) else 0.0
    if w < 20: w *= 1000
    if d < 20: d *= 1000
    if 0 < h < 20: h *= 1000
    return {"width_mm": w, "depth_mm": d, "height_mm": h if h > 0 else 500.0}
