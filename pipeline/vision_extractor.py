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
import re

import fitz

logger = logging.getLogger(__name__)

_DPI = 150          # page render DPI — 150 is good balance of quality vs token cost
_MAX_RETRIES = 2    # retry once on JSON parse error

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
            response_mime_type="application/json",
        ),
    )
    return response.text


def _parse_json(text: str) -> dict | list | None:
    """Parse JSON from model output, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error from Gemini: %s | text[:200]=%s", e, text[:200])
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
    for attempt in range(_MAX_RETRIES):
        try:
            raw = _call_gemini(png, prompt)
            data = _parse_json(raw)
            if not isinstance(data, dict):
                continue
            if not data.get("x_axes") and not data.get("y_axes"):
                logger.warning("extract_grid_vision: empty axes returned (attempt %d)", attempt)
                continue
            return data
        except Exception as e:
            logger.warning("extract_grid_vision attempt %d failed: %s", attempt, e)
    return None


def extract_schedule_vision(page: fitz.Page) -> list[dict] | None:
    """Extract pile/footing schedule table from page image using Gemini.

    Returns list of schedule entries matching foundation_extractor schema.
    Returns None on failure.
    """
    prompt = """You are reading an Australian structural engineering drawing.
Find the pile/footing schedule table in this image (if present).

The table typically has columns like:
MARK | PILE SIZE/DIA | SOCKET LENGTH | NO. OF PILES | CAP SIZE (WxDxH) | etc.
OR for pad footings:
MARK | WIDTH | LENGTH | DEPTH | REINFORCEMENT

Parse every row and classify each mark:
- P1, P2, PC1, PC2 → "pile_cap"
- F1, F2, F3 → "pad_footing"
- CB1, CB2 → "capping_beam"
- RF1, RF2, RS1 → "raft"

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
    for attempt in range(_MAX_RETRIES):
        try:
            raw = _call_gemini(png, prompt)
            data = _parse_json(raw)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("extract_schedule_vision attempt %d failed: %s", attempt, e)
    return None


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
    x_labels = [a["label"] for a in grid.get("x_axes", [])]
    y_labels = [a["label"] for a in grid.get("y_axes", [])]
    known_marks = list(schedule.keys()) if schedule else []

    prompt = f"""You are a structural BIM engineer reading an Australian foundation plan drawing.

The structural grid visible in this drawing has:
- X-axis grid lines labelled (left→right): {x_labels if x_labels else '(detect from drawing)'}
- Y-axis grid lines labelled (top→bottom): {y_labels if y_labels else '(detect from drawing)'}

Known foundation types from schedule: {known_marks if known_marks else 'detect from drawing (P1,P2,PC1,F1,F2,CB1,RF1 etc.)'}

TASK: Find every foundation annotation in the drawing.
For EACH foundation:
  1. Read its exact mark label (e.g. P1, PC2, F1)
  2. Identify which grid intersection it sits ON — this is critical for accuracy.
     grid_ref format = "Y_label/X_label" e.g. "A/1", "B/3", "AA/5"
  3. ONLY if not on a grid intersection: record x_percent, y_percent (0.0–1.0 from top-left)

ACCURACY RULES:
- A foundation sitting on grid line intersection A×1 → grid_ref="A/1", is_on_grid=true
- A foundation between grid lines (off-grid) → grid_ref="off_grid", is_on_grid=false, give x_percent/y_percent
- Do NOT guess grid_ref — if unsure, set is_on_grid=false
- Count repeated marks (e.g. P1 appears at every exterior corner)

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

    png = _page_to_png(page)
    for attempt in range(_MAX_RETRIES):
        try:
            raw = _call_gemini(png, prompt)
            data = _parse_json(raw)
            if isinstance(data, list):
                return _convert_vision_fdns_to_model(data, grid, schedule, page)
        except Exception as e:
            logger.warning("extract_foundations_vision attempt %d failed: %s", attempt, e)
    return None


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

    result = []
    for i, vf in enumerate(vision_fdns):
        label = str(vf.get("label", "?")).upper()
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
