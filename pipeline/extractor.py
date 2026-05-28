"""Stage 3: Extract structural elements from each classified page using Gemini Vision via Vertex AI.

Output: data/jobs/{job_id}/stage3_raw_extractions.json
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types


# ── Prompts ──────────────────────────────────────────────────────────────────

GRID_PROMPT = """You are a structural engineer reading a structural plan drawing.
Identify ALL grid lines (axes) visible on this drawing.
Grid lines are reference lines labeled with letters (A, B, C...) or numbers (1, 2, 3...).
Look for dimension annotations between grid lines to get spacing in mm.

Output ONLY valid JSON:
{
  "x_axes": [{"label": "1", "cumulative_mm": 0}, {"label": "2", "cumulative_mm": 6000}],
  "y_axes": [{"label": "A", "cumulative_mm": 0}, {"label": "B", "cumulative_mm": 5400}],
  "unit_detected": "mm",
  "dimension_confidence": 0.85,
  "notes": "any notes"
}
x_axes = horizontal numbered lines, y_axes = vertical lettered lines.
First axis always starts at cumulative_mm = 0. Return null axes arrays if no grid found."""


PLAN_ELEMENTS_PROMPT = """You are a structural engineer reading a structural marking/framing plan.
Extract ALL structural elements visible on this drawing.

For COLUMNS: solid filled squares/circles at grid intersections, labels like 'UC', 'SHS', 'CHS', 'CH'.
For BEAMS: lines between grid points labeled with section sizes: 'UB', 'UC', 'PFC', 'RHS', 'CH', 'SH'.
For SLABS: hatched/bounded areas labeled 'RC SLAB', 'PT SLAB', 'TOPPING'.

Output ONLY valid JSON array:
[
  {"element_type": "column", "grid_ref": "A1", "section_label": "150UC37", "level_name": "Level 1", "notes": ""},
  {"element_type": "beam", "grid_ref": "A1-A2", "section_label": "310UB40", "level_name": "Roof", "notes": ""},
  {"element_type": "slab", "grid_ref": "A1-B2", "section_label": "150RC", "level_name": "Level 1", "notes": ""}
]
Return [] if no elements found."""


FOUNDATION_PROMPT = """You are a structural engineer reading a foundation plan.
Extract ALL foundation elements (pad footings, strip footings, pile caps, ground beams, raft slabs).

Output ONLY valid JSON array:
[
  {"element_type": "pad_footing", "grid_ref": "A1", "plan_size_label": "1200x1200", "depth_label": "500", "notes": ""},
  {"element_type": "strip_footing", "grid_ref": "A1-A2", "width_label": "600", "depth_label": "300", "notes": ""}
]
Return [] if no foundation elements found."""


SCHEDULE_PROMPT = """You are a structural engineer reading a structural schedule or legend table.
Parse ALL rows in this schedule/table and output a lookup table.

Output ONLY valid JSON array:
[
  {"label": "40b CH", "element_type": "beam", "width_mm": 40, "height_mm": 40, "material": "steel", "notes": ""},
  {"label": "310UB40", "element_type": "beam", "width_mm": 165, "height_mm": 310, "material": "steel", "notes": ""}
]
Return [] if no schedule data found."""


ELEVATION_PROMPT = """You are a structural engineer reading a structural elevation or section drawing.
Extract all level/floor heights and any visible structural elements.

Output ONLY valid JSON:
{
  "levels": [
    {"name": "Ground Floor", "elevation_mm": 0},
    {"name": "Level 1", "elevation_mm": 3600},
    {"name": "Roof", "elevation_mm": 7200}
  ],
  "elements": [
    {"element_type": "column", "grid_ref": "A", "section_label": "150UC37", "notes": ""}
  ],
  "notes": ""
}"""


# ── Dispatch ─────────────────────────────────────────────────────────────────

PLAN_TYPES = {"floor_plan", "roof_plan", "foundation_plan"}
SCHEDULE_TYPES = {"schedule"}
ELEVATION_TYPES = {"elevation", "section"}
SKIP_TYPES = {"cover", "notes", "unknown"}


def _call_gemini(client: genai.Client, model_name: str, image_path: str, prompt: str) -> dict:
    """Single Gemini Vision call. Returns {data, raw, error}."""
    try:
        image_bytes = Path(image_path).read_bytes()
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw = response.text
        data = json.loads(raw)
        return {"data": data, "raw": raw, "error": None}
    except json.JSONDecodeError as e:
        return {"data": None, "raw": getattr(response, "text", ""), "error": f"JSON parse: {e}"}
    except Exception as e:
        return {"data": None, "raw": "", "error": str(e)}


def extract_page(page_info: dict, client: genai.Client, model_name: str) -> dict:
    """Run appropriate Gemini extraction for one classified page."""
    drawing_type = page_info.get("drawing_type", "unknown")
    image_path = page_info["file_path"]

    extraction = {
        "page_num": page_info["page_num"],
        "drawing_type": drawing_type,
        "level_name": page_info.get("level_name"),
        "scale_ratio": page_info.get("scale_ratio"),
        "grid": None,
        "elements": [],
        "schedules": [],
        "levels": [],
        "raw_gemini_responses": {},
        "extraction_warnings": [],
    }

    if drawing_type in SKIP_TYPES:
        extraction["extraction_warnings"].append(f"Skipped: {drawing_type} page")
        return extraction

    if drawing_type in PLAN_TYPES:
        r = _call_gemini(client, model_name, image_path, GRID_PROMPT)
        extraction["grid"] = r["data"]
        extraction["raw_gemini_responses"]["grid"] = r["raw"]
        if r["error"]:
            extraction["extraction_warnings"].append(f"Grid: {r['error']}")

    if drawing_type == "foundation_plan":
        r = _call_gemini(client, model_name, image_path, FOUNDATION_PROMPT)
        extraction["elements"] = r["data"] or []
        extraction["raw_gemini_responses"]["foundation"] = r["raw"]
        if r["error"]:
            extraction["extraction_warnings"].append(f"Foundation: {r['error']}")
    elif drawing_type in {"floor_plan", "roof_plan"}:
        r = _call_gemini(client, model_name, image_path, PLAN_ELEMENTS_PROMPT)
        extraction["elements"] = r["data"] or []
        extraction["raw_gemini_responses"]["plan"] = r["raw"]
        if r["error"]:
            extraction["extraction_warnings"].append(f"Plan: {r['error']}")
    elif drawing_type in SCHEDULE_TYPES:
        r = _call_gemini(client, model_name, image_path, SCHEDULE_PROMPT)
        extraction["schedules"] = r["data"] or []
        extraction["raw_gemini_responses"]["schedule"] = r["raw"]
        if r["error"]:
            extraction["extraction_warnings"].append(f"Schedule: {r['error']}")
    elif drawing_type in ELEVATION_TYPES:
        r = _call_gemini(client, model_name, image_path, ELEVATION_PROMPT)
        data = r["data"] or {}
        extraction["levels"] = data.get("levels", [])
        extraction["elements"] = data.get("elements", [])
        extraction["raw_gemini_responses"]["elevation"] = r["raw"]
        if r["error"]:
            extraction["extraction_warnings"].append(f"Elevation: {r['error']}")

    return extraction


def extract_all_pages(
    classification: dict,
    job_dir: str,
    model_name: str,
    project: str,
    location: str,
    workers: int = 6,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Extract structural elements from all classified pages in parallel.

    workers: concurrent Gemini API calls per batch (default 6).
    Note: plan pages make 2 Gemini calls each (grid + elements),
    so effective concurrency is workers × 2 at peak.
    """
    client = genai.Client(vertexai=True, project=project, location=location)
    job_dir = Path(job_dir)
    pages = classification["pages"]
    total = len(pages)

    completed_count = 0
    lock = threading.Lock()
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_page, page_info, client, model_name): page_info["page_num"]
            for page_info in pages
        }
        for future in as_completed(futures):
            page_num = futures[future]
            results[page_num] = future.result()

            with lock:
                completed_count += 1
                done = completed_count

            if progress_cb:
                progress_cb(done, total)

    # Restore page order
    extractions = [results[p["page_num"]] for p in pages]

    counts = {"columns": 0, "beams": 0, "slabs": 0, "foundations": 0, "schedules": 0, "levels": 0}
    for e in extractions:
        for elem in e.get("elements", []):
            t = (elem.get("element_type") or "").lower()
            if "column" in t:
                counts["columns"] += 1
            elif "beam" in t:
                counts["beams"] += 1
            elif "slab" in t:
                counts["slabs"] += 1
            elif any(x in t for x in ["footing", "foundation", "pile"]):
                counts["foundations"] += 1
        counts["schedules"] += len(e.get("schedules", []))
        counts["levels"] += len(e.get("levels", []))

    result = {
        "stage": 3,
        "stage_name": "Element Extractor",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_name,
        "workers": workers,
        "total_pages": total,
        "extraction_summary": counts,
        "pages": extractions,
    }

    (job_dir / "stage3_raw_extractions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
