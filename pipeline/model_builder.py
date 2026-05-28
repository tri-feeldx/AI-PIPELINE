"""Model builder — assembles the full structural model from vector-extracted data.

Processes every plan page of the PDF using:
  - grid_extractor  → exact grid coordinates
  - element_detector → columns, beams, slabs at exact positions
  - level_extractor  → floor elevation heights

Writes stage2_classification.json, stage3_vector_extractions.json,
and stage4_unified_model.json.
"""

import json
import time
from pathlib import Path

import fitz

from pipeline.page_classifier import classify_all_pages
from pipeline.grid_extractor import extract_grids_from_pdf, extract_grid, PT_TO_MM
from pipeline.element_detector import detect_elements
from pipeline.level_extractor import extract_levels_from_pdf


# ── Section dimension lookup (built-in common Australian sections) ─────────────

_SECTION_DIMS: dict[str, dict] = {
    # CH (cold-formed channel)
    "40b CH":  {"width_mm": 40,  "height_mm": 40,  "material": "steel"},
    "35c CH":  {"width_mm": 35,  "height_mm": 35,  "material": "steel"},
    "30b SH":  {"width_mm": 30,  "height_mm": 30,  "material": "steel"},
    "36b UB":  {"width_mm": 171, "height_mm": 360, "material": "steel"},
    "310UB40": {"width_mm": 165, "height_mm": 310, "material": "steel"},
    "250UB37": {"width_mm": 146, "height_mm": 256, "material": "steel"},
    "200UB25": {"width_mm": 133, "height_mm": 203, "material": "steel"},
    "150UC37": {"width_mm": 154, "height_mm": 162, "material": "steel"},
    "100UC15": {"width_mm": 100, "height_mm": 97,  "material": "steel"},
    "200RC":   {"width_mm": 300, "height_mm": 300, "material": "concrete"},
    "150RC":   {"width_mm": 200, "height_mm": 200, "material": "concrete"},
}

_DEFAULT_BEAM_DIMS   = {"width_mm": 100, "height_mm": 200, "material": "steel"}
_DEFAULT_COLUMN_DIMS = {"width_mm": 150, "height_mm": 150, "material": "steel"}


def _section_dims(label: str) -> dict:
    label = (label or "").strip()
    if label in _SECTION_DIMS:
        return _SECTION_DIMS[label]
    # Pattern heuristics
    lu = label.upper()
    if "UB" in lu or "UC" in lu:
        return {"width_mm": 150, "height_mm": 200, "material": "steel"}
    if any(x in lu for x in ("CH", "SH", "PFC", "SHS", "RHS", "CHS")):
        return {"width_mm": 100, "height_mm": 100, "material": "steel"}
    if any(x in lu for x in ("RC", "CONC", "PT")):
        return {"width_mm": 250, "height_mm": 250, "material": "concrete"}
    return _DEFAULT_BEAM_DIMS


def build_model(pdf_path: str, job_dir: str, progress_cb=None) -> dict:
    """Full vector-based pipeline — no AI calls.

    Stages:
      1. Classify pages from text
      2. Extract grid (vector)
      3. Extract elements per page (vector)
      4. Extract level heights (text)
      5. Assemble unified model
    """
    job_dir = Path(job_dir)
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    # ── Stage 2a: Page classification ────────────────────────────────────────
    classifications = classify_all_pages(pdf_path)
    _write_json(job_dir / "stage2_classification.json", {
        "stage": 2,
        "stage_name": "Page Classifier (vector)",
        "timestamp": _ts(),
        "total_pages": total_pages,
        "type_summary": {
            t: sum(1 for c in classifications if c["drawing_type"] == t)
            for t in {c["drawing_type"] for c in classifications}
        },
        "pages": classifications,
    })
    if progress_cb:
        progress_cb("classify", 1.0)

    # ── Stage 2b: Grid extraction ─────────────────────────────────────────────
    grid = extract_grids_from_pdf(pdf_path)
    if progress_cb:
        progress_cb("grid", 1.0)

    # ── Stage 3: Element detection per plan page ──────────────────────────────
    plan_types = {"floor_plan", "roof_plan", "foundation_plan"}
    page_extractions = []
    all_columns: dict[str, dict] = {}   # key = "Y/X" grid ref
    all_beams:   list[dict] = []
    all_slabs:   list[dict] = []

    for idx, cls in enumerate(classifications):
        page_num  = cls["page_num"]
        page_type = cls["drawing_type"]
        page      = doc[page_num - 1]

        if page_type in plan_types:
            # Extract grid for THIS page to use for element detection
            # (some pages may use different scales or partial grids)
            page_grid = extract_grid(page, grid.get("scale", 100))
            if not page_grid["x_axes"]:
                page_grid = grid  # fall back to global grid

            elements = detect_elements(page, page_type, page_grid)
        else:
            elements = {"columns": [], "beams": [], "slabs": []}

        page_extractions.append({
            "page_num": page_num,
            "drawing_type": page_type,
            "columns_found": len(elements["columns"]),
            "beams_found":   len(elements["beams"]),
            "slabs_found":   len(elements["slabs"]),
            "raw": elements,
        })

        # Merge into global collections (deduplicate by grid ref)
        for col in elements["columns"]:
            ref = col["grid_ref"]
            if ref not in all_columns:
                all_columns[ref] = {**col, "source_page": page_num}

        for beam in elements["beams"]:
            all_beams.append({**beam, "source_page": page_num})

        for slab in elements["slabs"]:
            all_slabs.append({**slab, "source_page": page_num})

        if progress_cb:
            progress_cb("extract", (idx + 1) / total_pages)

    doc.close()

    _write_json(job_dir / "stage3_vector_extractions.json", {
        "stage": 3,
        "stage_name": "Vector Element Extractor",
        "timestamp": _ts(),
        "grid": grid,
        "pages": page_extractions,
    })

    # ── Stage 4a: Level heights ───────────────────────────────────────────────
    levels = extract_levels_from_pdf(pdf_path)
    if progress_cb:
        progress_cb("levels", 1.0)

    # ── Stage 4b: Generate pile positions from grid ───────────────────────────
    foundations = _generate_piles_from_grid(grid)

    # ── Stage 4c: Assign section sizes and level spans ───────────────────────
    bottom_lv = levels[0]  if levels else {"name": "GROUND FLOOR", "elevation_mm": 0}
    top_lv    = levels[-1] if levels else {"name": "ROOF",         "elevation_mm": 18000}

    cols_final = []
    for col in all_columns.values():
        cols_final.append({
            "id":            f"COL-{len(cols_final)+1:04d}",
            "grid_ref":      col["grid_ref"],
            "x_mm":          col["x_mm"],
            "y_mm":          col["y_mm"],
            "from_level":    bottom_lv["name"],
            "from_elev_mm":  bottom_lv["elevation_mm"],
            "to_level":      top_lv["name"],
            "to_elev_mm":    top_lv["elevation_mm"],
            "width_mm":      150,
            "depth_mm":      150,
            "material":      "steel",
            "source_page":   col.get("source_page"),
        })

    beams_final = []
    seen_beams = set()
    for beam in all_beams:
        key = (round(beam["from_x_mm"]), round(beam["from_y_mm"]),
               round(beam["to_x_mm"]),   round(beam["to_y_mm"]))
        if key in seen_beams:
            continue
        seen_beams.add(key)
        dims = _section_dims(beam.get("section_label", ""))
        beams_final.append({
            "id":          f"BEAM-{len(beams_final)+1:04d}",
            "grid_ref":    beam["grid_ref"],
            "section_label": beam.get("section_label", ""),
            "from_x_mm":   beam["from_x_mm"],
            "from_y_mm":   beam["from_y_mm"],
            "to_x_mm":     beam["to_x_mm"],
            "to_y_mm":     beam["to_y_mm"],
            "level":       top_lv["name"],
            "elev_mm":     top_lv["elevation_mm"],
            "width_mm":    dims["width_mm"],
            "height_mm":   dims["height_mm"],
            "material":    dims["material"],
            "source_page": beam.get("source_page"),
        })

    slabs_final = []
    seen_slabs = set()
    for slab in all_slabs:
        key = (round(slab["from_x_mm"]), round(slab["from_y_mm"]),
               round(slab["to_x_mm"]),   round(slab["to_y_mm"]))
        if key in seen_slabs:
            continue
        seen_slabs.add(key)
        slabs_final.append({
            "id":          f"SLAB-{len(slabs_final)+1:04d}",
            "grid_ref":    slab["grid_ref"],
            "section_label": slab.get("section_label", "RC SLAB"),
            "from_x_mm":   slab["from_x_mm"],
            "from_y_mm":   slab["from_y_mm"],
            "to_x_mm":     slab["to_x_mm"],
            "to_y_mm":     slab["to_y_mm"],
            "level":       top_lv["name"],
            "elev_mm":     top_lv["elevation_mm"],
            "thickness_mm": 150,
            "material":    "concrete",
            "source_page": slab.get("source_page"),
        })

    unified = {
        "stage": 4,
        "stage_name": "Model Builder (vector pipeline)",
        "timestamp": _ts(),
        "source_pdf": pdf_path,
        "grid_system": {
            "x_axes": grid["x_axes"],
            "y_axes": grid["y_axes"],
            "scale": grid.get("scale", 100),
            "unit": "mm",
        },
        "levels": levels,
        "columns":     cols_final,
        "beams":       beams_final,
        "slabs":       slabs_final,
        "foundations": foundations,
        "summary_counts": {
            "columns":     len(cols_final),
            "beams":       len(beams_final),
            "slabs":       len(slabs_final),
            "foundations": len(foundations),
            "levels":      len(levels),
        },
    }

    _write_json(job_dir / "stage4_unified_model.json", unified)
    if progress_cb:
        progress_cb("model", 1.0)

    return unified


def _generate_piles_from_grid(grid: dict) -> list[dict]:
    """Generate pile cap positions at every grid intersection."""
    x_axes = grid.get("x_axes", [])
    y_axes = grid.get("y_axes", [])
    piles = []
    for ya in y_axes:
        for xa in x_axes:
            is_perimeter = (
                xa == x_axes[0] or xa == x_axes[-1]
                or ya == y_axes[0] or ya == y_axes[-1]
            )
            dia = 750 if is_perimeter else 750
            piles.append({
                "id":      f"PILE-{ya['label']}{xa['label']}",
                "type":    "pile",
                "grid_ref": f"{ya['label']}/{xa['label']}",
                "x_mm":    xa["real_mm"],
                "y_mm":    ya["real_mm"],
                "width_mm": dia,
                "depth_mm": 500,
                "material": "concrete",
                "perimeter": is_perimeter,
            })
    return piles


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
