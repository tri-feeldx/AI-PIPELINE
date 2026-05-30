"""Vision extractor — Gemini 2.5 Flash via Vertex AI.

Called ONLY when vector extraction quality gate fails.
Targeted prompts fill specific gaps: grid, schedule, or foundation positions.

Environment variables required:
  GOOGLE_CLOUD_PROJECT  — GCP project ID
  VERTEX_LOCATION       — e.g. us-central1 (default)
  GEMINI_MODEL          — e.g. gemini-2.5-flash (default)
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time

import fitz

logger = logging.getLogger(__name__)

_DPI = 150   # full-page render DPI
_TILE_DPI = 200  # higher DPI for tiles (smaller area, more detail per px)

# ── Lazy Gemini client ─────────────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        project  = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT env var not set — required for Vertex AI"
            )
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def _model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _page_to_png(page: fitz.Page, dpi: int = _DPI) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _call_gemini(image_bytes: bytes, prompt: str) -> str:
    """Send image + prompt to Gemini, return raw text response."""
    from google.genai import types
    client = _get_client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=_model(),
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            # No response_mime_type="application/json" — JSON mode has a lower internal
            # output limit (~2579 chars). Plain text mode returns 2.5× more data.
            # _parse_json() already strips ```json ... ``` fences.
            max_output_tokens=32768,
        ),
    )
    return response.text


def _call_gemini_with_retry(image_bytes: bytes, prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini with exponential backoff on 429 rate-limit errors."""
    delays = [15, 45, 120]  # Vertex AI QPM window is ~60s; wait ≥60s to clear quota
    for attempt in range(max_retries):
        try:
            return _call_gemini(image_bytes, prompt)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < max_retries - 1:
                    sleep_s = delays[attempt] + random.uniform(0, 4)
                    logger.warning(
                        "Vision AI 429 rate limit — retry %d/%d in %.0fs",
                        attempt + 1, max_retries - 1, sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    logger.error("Vision AI quota exhausted after %d retries", max_retries)
                    return None
            else:
                raise
    return None


def _page_to_tiles(
    page: fitz.Page,
    dpi: int = _TILE_DPI,
    cols: int = 2,
    rows: int = 2,
    overlap: float = 0.10,
):
    """Yield (png_bytes, x0_frac, y0_frac, w_frac, h_frac) for each tile.

    Splits the page into a cols×rows grid with `overlap` fraction of overlap
    on each side so annotations near tile borders are fully visible in at
    least one tile. Coordinates are fractions of the full page dimensions.
    """
    pw = page.rect.width
    ph = page.rect.height
    tile_w = 1.0 / cols + overlap
    tile_h = 1.0 / rows + overlap
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for r in range(rows):
        for c in range(cols):
            x0 = max(0.0, c / cols - overlap / 2)
            y0 = max(0.0, r / rows - overlap / 2)
            x1 = min(1.0, x0 + tile_w)
            y1 = min(1.0, y0 + tile_h)
            clip = fitz.Rect(x0 * pw, y0 * ph, x1 * pw, y1 * ph)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            yield pix.tobytes("png"), x0, y0, (x1 - x0), (y1 - y0)


def _parse_json(text: str) -> dict | list | None:
    """Parse JSON from model output, stripping markdown fences if present.

    If the full response is truncated mid-JSON (Gemini output limit exceeded),
    attempts to salvage all complete objects from an array before the cut.
    """
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Normalise Python/JS literals that json.loads rejects
    text = re.sub(r'\bNone\b',     'null',  text)
    text = re.sub(r'\bNaN\b',      '0',     text)
    text = re.sub(r'\bInfinity\b', '0',     text)
    text = re.sub(r'\bTrue\b',     'true',  text)
    text = re.sub(r'\bFalse\b',    'false', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error from Gemini: %s | text[:200]=%s", e, text[:200])

    # Partial-JSON recovery for truncated array responses.
    # Strategy: walk through the text collecting complete JSON objects one by one.
    try:
        decoder = json.JSONDecoder()
        results = []
        s = text.strip()
        # Skip leading '[' whitespace
        i = s.find("[")
        if i == -1:
            return None
        i += 1  # move past '['
        while i < len(s):
            # Skip whitespace and commas
            while i < len(s) and s[i] in " \t\n\r,":
                i += 1
            if i >= len(s) or s[i] == "]":
                break
            if s[i] != "{":
                break
            try:
                obj, end_idx = decoder.raw_decode(s, i)
                results.append(obj)
                i = end_idx
            except json.JSONDecodeError:
                break   # truncation point — stop here
        if results:
            logger.warning("Partial-JSON recovery: salvaged %d complete objects", len(results))
            return results
    except Exception:
        pass

    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_grid_vision(page: fitz.Page) -> dict | None:
    """Extract structural grid from page image using Gemini.

    Returns dict with same schema as grid_extractor.extract_grid():
      {x_axes: [{label, x_percent}], y_axes: [{label, y_percent}], scale: int}
    x_percent/y_percent are 0.0–1.0 fractions of page dimensions.
    Caller converts to real_mm using page size + scale.
    Returns None on failure.
    """
    prompt = """You are reading a structural engineering drawing.

Find ALL visible structural grid axis labels in this image.
Grid labels are typically printed at large font size near the drawing border:
- Numbers (1, 2, 3 ... or 01, 02 ...) near the top OR bottom edge → X-axis (vertical grid lines)
- Letters (A, B, C ... or AA, AB, BA ...) near the left OR right edge → Y-axis (horizontal grid lines)
- Multi-character labels like AA, AB, BB are valid

Also find the drawing scale (e.g. "1:100", "SCALE 1:80", "1/100").

Return ONLY valid JSON with NO explanation:
{
  "x_axes": [{"label": "1", "x_percent": 0.12}, {"label": "2", "x_percent": 0.28}],
  "y_axes": [{"label": "A", "y_percent": 0.15}, {"label": "B", "y_percent": 0.45}],
  "scale": 100
}

x_percent = horizontal position from LEFT edge (0.0=left, 1.0=right)
y_percent = vertical position from TOP edge (0.0=top, 1.0=bottom)
scale = denominator only (e.g. 100 for 1:100, 80 for 1:80)
If scale not found, use 100."""

    png = _page_to_png(page)
    raw = _call_gemini_with_retry(png, prompt)
    if raw is None:
        return None
    data = _parse_json(raw)
    if not isinstance(data, dict):
        return None
    if not data.get("x_axes") and not data.get("y_axes"):
        logger.warning("extract_grid_vision: empty axes returned")
        return None
    return data


def extract_schedule_vision(page: fitz.Page) -> list[dict] | None:
    """Extract pile/footing schedule table from page image using Gemini.

    Returns list of schedule entries matching foundation_extractor schema.
    Returns None on failure.
    """
    prompt = """You are reading a structural engineering drawing (may be Australian or Vietnamese).
Find the pile/footing schedule table in this image (if present).

The table may use English OR Vietnamese headers:
English: MARK | PILE SIZE/DIA | SOCKET LENGTH | NO. OF PILES | CAP SIZE (WxDxH) | etc.
Vietnamese: MÁC / KÝ HIỆU | ĐƯỜNG KÍNH CỌC / ĐK | CHIỀU SÂU / SỐ LƯỢNG | KÍCH THƯỚC ĐÀI | etc.

Dimension formats you may see:
- English: 1500x1500x700, Ø750, 750mm
- Vietnamese: BxH=1200x700 (cap dimensions), ĐK600 or Φ600 (pile diameter), 1200×700

Parse every row and classify each mark:
- P1, P2, PC1, PC2, MC1, MD1 → "pile_cap"
- F1, F2, F3, PF1 → "pad_footing"
- CB1, CB2, DM1, DG1 → "capping_beam"
- RF1, RF2, RS1 → "raft"
- MB1, SF1 → "strip_footing"

Return ONLY valid JSON array (empty array [] if no schedule found):
[
  {
    "mark": "P1",
    "ftype": "pile_cap",
    "pile_dia_mm": 750,
    "socket_m": 3.5,
    "pile_count": 2,
    "width_mm": 1500,
    "depth_mm": 1500,
    "height_mm": 700
  }
]

Rules:
- pile_dia_mm: diameter in mm (convert from m if needed, 0 if N/A)
- socket_m: socket/embedment depth in metres (0 if N/A)
- pile_count: number of piles in group (1 if single pile or pad footing)
- width_mm / depth_mm: plan dimensions of cap/footing in mm
- height_mm: thickness/depth of cap/footing in mm
- Use 0 for any value not found in the table"""

    png = _page_to_png(page)
    raw = _call_gemini_with_retry(png, prompt)
    if raw is None:
        return None
    data = _parse_json(raw)
    return data if isinstance(data, list) else None


def _build_synthetic_grid(vision_grid_data: dict, page: fitz.Page) -> dict:
    """Convert extract_grid_vision() result (x_percent/y_percent) to real_mm grid dict.

    Vision AI returns grid label positions as fractions of page dimensions.
    We convert to pdf_pts then to real_mm using the scale found (or 100).
    The result has the same schema as grid_extractor.extract_grid().
    """
    from pipeline.grid_extractor import PT_TO_MM

    pw = page.rect.width
    ph = page.rect.height
    scale = int(vision_grid_data.get("scale", 100))
    pt_to_mm = PT_TO_MM * scale

    raw_x = vision_grid_data.get("x_axes", [])
    raw_y = vision_grid_data.get("y_axes", [])

    def _to_axes(items: list[dict], key: str, page_dim: float, is_y: bool) -> list[dict]:
        converted = []
        for ax in items:
            pct = float(ax.get(key, 0.5))
            pdf_pos = pct * page_dim
            converted.append({"label": str(ax.get("label", "?")), "pdf_pos": round(pdf_pos, 2)})
        converted.sort(key=lambda a: a["pdf_pos"])
        if not converted:
            return []
        base = converted[0]["pdf_pos"]
        for a in converted:
            a["real_mm"] = round((a["pdf_pos"] - base) * pt_to_mm, 1)
        return converted

    x_axes = _to_axes(raw_x, "x_percent", pw, False)
    y_axes = _to_axes(raw_y, "y_percent", ph, True)

    return {
        "x_axes": x_axes,
        "y_axes": y_axes,
        "scale": scale,
        "pt_to_mm": round(pt_to_mm, 4),
        "source": "vision_synthetic",
    }


def extract_foundations_vision(
    page: fitz.Page,
    grid: dict,
    schedule: dict,
) -> list[dict] | None:
    """Extract foundation positions from page image using Gemini.

    grid: dict with x_axes/y_axes lists (label + real_mm)
    schedule: dict of {mark → spec} from parse_footing_schedule
    Returns list of foundation dicts (same schema as foundation_extractor output).
    Returns None on failure.
    """
    # Build a synthetic Y-axis from Vision AI whenever the vector extractor failed
    # to find Y-axis grid labels (common for AU drawings where both axes use numbers).
    # Without Y-axis snapping, all off-grid positions use raw y_percent * scale which
    # is too imprecise at 1:250 scale (1% error = ~21 m real-world offset).
    if not grid.get("y_axes"):
        logger.info("extract_foundations_vision: Y-axis missing — calling extract_grid_vision for Y")
        vision_grid_data = extract_grid_vision(page)
        if vision_grid_data and (vision_grid_data.get("x_axes") or vision_grid_data.get("y_axes")):
            vision_synth = _build_synthetic_grid(vision_grid_data, page)
            # Merge: keep existing vector X-axis if present, take Vision Y-axis
            merged = dict(grid)
            if not merged.get("x_axes") and vision_synth.get("x_axes"):
                merged["x_axes"] = vision_synth["x_axes"]
            if vision_synth.get("y_axes"):
                merged["y_axes"] = vision_synth["y_axes"]
            if not merged.get("pt_to_mm"):
                merged["pt_to_mm"] = vision_synth.get("pt_to_mm", 35.28)
            grid = merged
            logger.info(
                "extract_foundations_vision: grid after merge — %d x-axes, %d y-axes",
                len(grid.get("x_axes", [])), len(grid.get("y_axes", [])),
            )
        else:
            logger.warning("extract_foundations_vision: Vision AI grid also failed — Y positions will be approximate")

    x_labels = [a["label"] for a in grid.get("x_axes", [])]
    y_labels = [a["label"] for a in grid.get("y_axes", [])]
    known_marks = list(schedule.keys()) if schedule else []

    prompt = f"""You are a structural BIM engineer reading an Australian foundation plan drawing.

The structural grid visible in this drawing has:
- X-axis grid lines labelled (left→right): {x_labels if x_labels else '(detect from drawing)'}
- Y-axis grid lines labelled (top→bottom): {y_labels if y_labels else '(detect from drawing)'}

Known foundation types from schedule: {known_marks if known_marks else 'detect from drawing (P1,P2,PC1,F1,F2,CB1,RF1 etc.)'}

━━━ CRITICAL DISTINCTION — READ THIS FIRST ━━━
There are TWO types of mark text visible on this drawing. Only count TYPE 1:

TYPE 1 — PLAN ANNOTATIONS (COUNT THESE):
  Mark labels placed directly ON pile cap symbols or footing outlines within
  the main plan drawing area. Each sits at a specific XY position on the plan.
  Looks like: a small "PG1" label on top of a square/rectangle cap symbol.

TYPE 2 — SCHEDULE TABLE ENTRIES (IGNORE COMPLETELY):
  A table (usually in a corner) listing mark types with dimension columns:
    MARK | CAP SIZE | PILE DIA | SOCKET | NO. PILES
    PG1  | 1500×700 |   750    |  12m   |     2
    PG2  | 1200×600 |   600    |  10m   |     1
  These are SPECIFICATIONS, not placed positions. Do NOT count table rows.

ALSO IGNORE (do not count anything from these areas):
  - Keyplan / locator diagram (small overview map, usually top-right corner)
  - Title block (bottom strip with drawing number, date, revision)
  - Section callout bubbles (circles with numbers like ①②③)
  - Dimension lines and leader arrows
  - North arrow and scale bar

━━━ TASK ━━━
Find every foundation annotation PLACED ON THE PLAN DRAWING (not in tables).
For EACH placed foundation:
  1. Read its exact mark label (e.g. P1, PC2, F1, PG3)
  2. Identify which grid intersection it sits ON.
     grid_ref format = "Y_label/X_label" e.g. "A/1", "B/3"
  3. ONLY if not on a grid intersection: record x_percent, y_percent (0.0–1.0 from top-left)

ACCURACY RULES:
- Foundation on grid intersection A×1 → grid_ref="A/1", is_on_grid=true
- Foundation off-grid → grid_ref="off_grid", is_on_grid=false, give x_percent/y_percent
- Do NOT guess grid_ref — if unsure, set is_on_grid=false
- Each physical occurrence of a mark = one entry (P1 at 4 corners = 4 entries)

Return ONLY valid JSON array, no explanation:
[
  {{
    "label": "P1",
    "grid_ref": "A/1",
    "x_percent": 0.15,
    "y_percent": 0.22,
    "is_on_grid": true,
    "confidence": "high"
  }},
  {{
    "label": "P1",
    "grid_ref": "A/2",
    "x_percent": 0.28,
    "y_percent": 0.22,
    "is_on_grid": true,
    "confidence": "high"
  }},
  {{
    "label": "P3",
    "grid_ref": "off_grid",
    "x_percent": 0.05,
    "y_percent": 0.50,
    "is_on_grid": false,
    "confidence": "medium"
  }}
]

confidence: "high" = clearly on grid, "medium" = estimated, "low" = uncertain"""

    # Use 2×2 tiled images for better resolution on large A0/A1 drawings.
    # Each tile is sent at higher DPI; results are merged and deduplicated.
    all_raw: list[dict] = []
    seen_grid_refs: set[str] = set()

    for tile_png, tx0, ty0, tw, th in _page_to_tiles(page):
        raw = _call_gemini_with_retry(tile_png, prompt)
        if raw is None:
            logger.warning("extract_foundations_vision: tile (%.2f,%.2f) got no response", tx0, ty0)
            continue
        data = _parse_json(raw)
        if not isinstance(data, list):
            continue

        # Re-map tile-local x_percent/y_percent back to full-page fractions
        for item in data:
            item["x_percent"] = tx0 + float(item.get("x_percent", 0.5)) * tw
            item["y_percent"] = ty0 + float(item.get("y_percent", 0.5)) * th

            gref  = item.get("grid_ref", "off_grid")
            label = str(item.get("label", "?")).upper()
            xp    = float(item["x_percent"])
            yp    = float(item["y_percent"])

            if gref != "off_grid":
                # Grid-ref dedup: exact match (same intersection in overlapping tile)
                if gref in seen_grid_refs:
                    continue
                seen_grid_refs.add(gref)
            else:
                # Off-grid dedup: position proximity only (no label match).
                # True tile-overlap duplicates remap to virtually identical page
                # positions (Gemini accuracy ≈ 0.3% of page). Threshold 0.5%
                # catches duplicates without removing real foundations at 1-2m spacing.
                if any(
                    abs(xp - e["x_percent"]) < 0.005
                    and abs(yp - e["y_percent"]) < 0.005
                    for e in all_raw
                    if e.get("grid_ref", "off_grid") == "off_grid"
                ):
                    continue
            all_raw.append(item)

    if not all_raw:
        return None

    # Schedule-guided filter: if we have ≥3 known marks from the schedule,
    # remove any Vision AI result whose label doesn't match a known mark type.
    # This is a safety net against schedule-table rows that slipped past the prompt.
    if schedule and len(schedule) >= 3:
        known_upper = {k.upper() for k in schedule}
        before = len(all_raw)
        all_raw = [f for f in all_raw if str(f.get("label", "")).upper() in known_upper]
        removed = before - len(all_raw)
        if removed:
            logger.info(
                "Schedule filter: dropped %d unrecognised marks (kept %d/%d)",
                removed, len(all_raw), before,
            )

    if not all_raw:
        logger.warning("extract_foundations_vision: all results filtered by schedule — returning None")
        return None

    return _convert_vision_fdns_to_model(all_raw, grid, schedule, page)


def _convert_vision_fdns_to_model(
    vision_fdns: list[dict],
    grid: dict,
    schedule: dict,
    page: fitz.Page,
) -> list[dict]:
    """Convert Gemini foundation positions (x_percent, y_percent) to real_mm coords."""
    pw = page.rect.width
    ph = page.rect.height
    pt_to_mm = grid.get("pt_to_mm", 35.28)

    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])

    x_base = x_axes[0]["pdf_pos"] if x_axes else 0.0
    y_base = y_axes[0]["pdf_pos"] if y_axes else 0.0

    # Column connection marks follow the pattern [BLD]-CC[N] (e.g. A-CC01, D-CC03).
    # They appear next to pile caps in foundation plan drawings but are column marks,
    # NOT foundation elements — exclude them from the foundation list.
    _COL_MARK_RE = re.compile(r'^[A-Z]+-CC\d+$', re.I)

    result = []
    for i, vf in enumerate(vision_fdns):
        label = str(vf.get("label", "?")).upper()
        if _COL_MARK_RE.match(label):
            continue   # column connection mark, not a foundation
        grid_ref = vf.get("grid_ref", "off_grid")
        xp = float(vf.get("x_percent", 0.5))
        yp = float(vf.get("y_percent", 0.5))

        # Convert percent → pdf_pts → real_mm
        x_pdf = xp * pw
        y_pdf = yp * ph
        x_mm = round((x_pdf - x_base) * pt_to_mm, 1)
        y_mm = round((y_pdf - y_base) * pt_to_mm, 1)

        # Snap to grid intersection if on_grid
        if vf.get("is_on_grid") and "/" in grid_ref:
            parts = grid_ref.split("/")
            y_lbl, x_lbl = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", "")
            for xa in x_axes:
                if xa["label"] == x_lbl:
                    x_mm = xa["real_mm"]
                    break
            for ya in y_axes:
                if ya["label"] == y_lbl:
                    y_mm = ya["real_mm"]
                    break

        # Look up spec from schedule
        spec = schedule.get(label, {})
        ftype = spec.get("ftype", "pile_cap")

        confidence = vf.get("confidence", "medium")
        result.append({
            "id": f"AI-{label}-{i:03d}",
            "ftype": ftype,
            "label": label,
            "grid_ref": grid_ref,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "pile_dia_mm":  spec.get("pile_dia_mm", 0),
            "pile_len_mm":  spec.get("pile_len_mm", 0),
            "pile_count":   spec.get("pile_count", 1),
            "width_mm":     spec.get("width_mm", 1500),
            "depth_mm":     spec.get("depth_mm", 1500),
            "height_mm":    spec.get("height_mm", 700),
            "material":     "concrete",
            "source":       "vision_ai",
            "needs_review": confidence == "low",
        })

    return result
