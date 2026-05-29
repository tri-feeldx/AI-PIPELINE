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
from pipeline.foundation_extractor import extract_foundations, parse_footing_schedule
from pipeline.quality_gate import assess_quality
from pipeline.schedule_extractor import extract_all_schedules, extract_concrete_defaults


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

    # ── Stage 2b: Member schedule extraction (columns, beams) ────────────────
    # Uses find_tables() on schedule/detail pages — free, fast, no AI needed.
    # Produces exact column/beam dimensions from structural schedules.
    member_schedules  = extract_all_schedules(pdf_path, classifications)
    col_schedule      = member_schedules.get("columns", {})
    beam_schedule     = member_schedules.get("beams", {})
    concrete_defaults = extract_concrete_defaults(pdf_path, classifications)

    # ── Stage 2c: Global foundation schedule pre-scan ─────────────────────────
    # Parse schedule from ALL pages (schedule, detail, foundation_plan).
    # This captures pile cap / pad footing tables that sit on dedicated
    # schedule pages (e.g. ST-003-31 "Foundation Sections and Details")
    # which are not processed by the per-page foundation extractor.
    global_schedule: dict = {}
    _sched_types = {"schedule", "detail", "foundation_plan"}
    for cls in classifications:
        if cls["drawing_type"] in _sched_types:
            pg = doc[cls["page_num"] - 1]
            sched = parse_footing_schedule(pg)
            for mark, spec in sched.items():
                # Keep entry with most complete data
                existing = global_schedule.get(mark, {})
                if (spec.get("width_mm", 0) > existing.get("width_mm", 0)
                        or spec.get("pile_dia_mm", 0) > existing.get("pile_dia_mm", 0)):
                    global_schedule[mark] = spec

    # ── Stage 2c: Grid extraction (scale-aware) ───────────────────────────────
    dominant_scale = _detect_dominant_scale(classifications)
    grid = extract_grids_from_pdf(pdf_path, dominant_scale=dominant_scale)
    if progress_cb:
        progress_cb("grid", 1.0)

    # ── Stage 3: Element detection per plan page ──────────────────────────────
    plan_types = {"floor_plan", "roof_plan", "foundation_plan"}
    page_extractions = []
    all_columns: dict[str, dict] = {}   # key = "Y/X" grid ref
    all_beams:   list[dict] = []
    all_slabs:   list[dict] = []

    # Foundation extraction results — collect ALL foundation_plan pages,
    # then merge with X-offset so multi-building PDFs include every building.
    all_fdn_results: list[dict] = []

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

            # Detect if this page uses Post-Tensioned construction
            page_text_upper = page.get_text().upper()
            page_is_pt = (
                "POST-TENSION" in page_text_upper
                or "POST TENSION" in page_text_upper
            )

            elements = detect_elements(page, page_type, page_grid)

            # Run dedicated foundation extractor on foundation plan pages.
            # Keep the result with the MOST footings (Footing Plan at 1:100
            # beats Footing Details at 1:20 which appears first in page order).
            if page_type == "foundation_plan":
                active_grid = page_grid if page_grid["x_axes"] else grid
                fdn_result = extract_foundations(page, active_grid, global_schedule)

                # Quality gate — try Vision AI if vector result is poor
                if fdn_result.get("has_foundation_plan"):
                    quality = assess_quality(fdn_result, active_grid, classifications)
                    if quality.needs_vision:
                        fdn_result = _vision_fallback(
                            page, active_grid, fdn_result, quality
                        )
                    # Collect every foundation page (multi-building support)
                    all_fdn_results.append(fdn_result)
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
            all_beams.append({
                **beam,
                "source_page": page_num,
                "is_pt": page_is_pt,
                "fc_mpa": concrete_defaults.get("beam", 40),
            })

        for slab in elements["slabs"]:
            all_slabs.append({
                **slab,
                "source_page": page_num,
                "is_pt": page_is_pt,
                "fc_mpa": concrete_defaults.get("slab", 40),
            })

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

    # ── Stage 4b: Foundation elements ────────────────────────────────────────
    # Merge ALL foundation pages (handles multi-building PDFs).
    # Falls back to grid-intersection generation only when no foundation
    # plan page was found at all.
    foundation_extraction = _merge_foundation_pages(all_fdn_results)

    ai_used = {
        "grid":        foundation_extraction.get("_vision_grid", False),
        "schedule":    foundation_extraction.get("_vision_schedule", False),
        "foundations": foundation_extraction.get("_vision_foundations", False),
    }
    if foundation_extraction.get("footings"):
        foundations   = foundation_extraction["footings"]
        ground_beams  = foundation_extraction.get("ground_beams", [])
        rafts         = foundation_extraction.get("rafts", [])
    else:
        foundations  = _generate_piles_from_grid(grid)
        ground_beams = []
        rafts        = []

    # ── Stage 4c: Assign section sizes and level spans ───────────────────────
    bottom_lv = levels[0]  if levels else {"name": "GROUND FLOOR", "elevation_mm": 0}
    top_lv    = levels[-1] if levels else {"name": "ROOF",         "elevation_mm": 18000}

    cols_final = []
    for col in all_columns.values():
        # Look up actual dimensions from column schedule if available
        col_mark = col.get("column_mark", col.get("section_label", ""))
        sched_spec = col_schedule.get(col_mark, {})
        col_w = sched_spec.get("width_mm", 150)
        col_d = sched_spec.get("depth_mm", 150)
        col_mat = sched_spec.get("material", "steel")
        from_lv_name = sched_spec.get("from_level") or bottom_lv["name"]
        to_lv_name   = sched_spec.get("to_level")   or top_lv["name"]

        # Resolve level elevation from levels list
        def _elev(name, default):
            for lv in levels:
                if lv["name"] == name:
                    return lv["elevation_mm"]
            return default

        cols_final.append({
            "id":            f"COL-{len(cols_final)+1:04d}",
            "grid_ref":      col["grid_ref"],
            "column_mark":   col_mark,
            "x_mm":          col["x_mm"],
            "y_mm":          col["y_mm"],
            "from_level":    from_lv_name,
            "from_elev_mm":  _elev(from_lv_name, bottom_lv["elevation_mm"]),
            "to_level":      to_lv_name,
            "to_elev_mm":    _elev(to_lv_name, top_lv["elevation_mm"]),
            "width_mm":      col_w,
            "depth_mm":      col_d,
            "material":      col_mat,
            "fc_mpa":        sched_spec.get("fc_mpa", concrete_defaults.get("column", 50)),
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
        section_lbl = beam.get("section_label", "")
        # Prefer beam schedule lookup, fall back to section dims table
        bsched = beam_schedule.get(section_lbl, {})
        dims = {
            "width_mm":  bsched.get("width_mm")  or _section_dims(section_lbl)["width_mm"],
            "height_mm": bsched.get("depth_mm")  or _section_dims(section_lbl)["height_mm"],
            "material":  bsched.get("material")  or _section_dims(section_lbl)["material"],
        }
        beams_final.append({
            "id":          f"BEAM-{len(beams_final)+1:04d}",
            "grid_ref":    beam["grid_ref"],
            "section_label": section_lbl,
            "from_x_mm":   beam["from_x_mm"],
            "from_y_mm":   beam["from_y_mm"],
            "to_x_mm":     beam["to_x_mm"],
            "to_y_mm":     beam["to_y_mm"],
            "level":       top_lv["name"],
            "elev_mm":     top_lv["elevation_mm"],
            "width_mm":    dims["width_mm"],
            "height_mm":   dims["height_mm"],
            "material":    dims["material"],
            "is_pt":       beam.get("is_pt", False),
            "fc_mpa":      beam.get("fc_mpa", 40),
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
            "is_pt":       slab.get("is_pt", False),
            "fc_mpa":      slab.get("fc_mpa", 40),
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
        "columns":      cols_final,
        "beams":        beams_final,
        "slabs":        slabs_final,
        "foundations":          foundations,
        "ground_beams":         ground_beams,
        "rafts":                rafts,
        "lift_pits":            foundation_extraction.get("lift_pits", []),
        "has_planter_slab":     foundation_extraction.get("has_planter_slab", False),
        "foundation_schedule":  foundation_extraction.get("schedule", {}),
        "foundation_pile_spec": foundation_extraction.get("pile_spec", {}),
        "column_schedule":      col_schedule,
        "beam_schedule":        beam_schedule,
        "ai_used": ai_used,
        "summary_counts": {
            "columns":      len(cols_final),
            "beams":        len(beams_final),
            "slabs":        len(slabs_final),
            "foundations":  len(foundations),
            "ground_beams": len(ground_beams),
            "rafts":        len(rafts),
            "levels":       len(levels),
        },
    }

    _write_json(job_dir / "stage4_unified_model.json", unified)
    if progress_cb:
        progress_cb("model", 1.0)

    return unified


def _merge_foundation_pages(results: list[dict]) -> dict:
    """Merge foundations from all foundation_plan pages.

    For multi-building PDFs each page is a separate building.
    Normalises each building to its own (0,0) origin then places
    buildings side-by-side with a 5 m gap so they don't overlap in SketchUp.
    Single-building PDFs (1 result) are returned unchanged.
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    all_footings:   list[dict] = []
    all_gbeams:     list[dict] = []
    all_rafts:      list[dict] = []
    all_lift_pits:  list[dict] = []
    merged_schedule: dict = {}
    has_planter = any(r.get("has_planter_slab", False) for r in results)

    x_cursor = 0.0

    for bldg_idx, r in enumerate(results):
        all_lift_pits.extend(r.get("lift_pits", []))
        footings = r.get("footings", [])
        if not footings:
            merged_schedule.update(r.get("schedule", {}))
            continue

        xs = [f.get("x_mm", 0) for f in footings]
        ys = [f.get("y_mm", 0) for f in footings]
        x_min, x_max = min(xs), max(xs)
        y_min = min(ys)
        bldg_width = max(x_max - x_min, 10_000)  # floor at 10 m

        for f in footings:
            nf = dict(f)
            nf["x_mm"] = round(f["x_mm"] - x_min + x_cursor, 1)
            nf["y_mm"] = round(f["y_mm"] - y_min, 1)
            nf["building_idx"] = bldg_idx
            all_footings.append(nf)

        for gb in r.get("ground_beams", []):
            ngb = dict(gb)
            ngb["from_x_mm"] = round(gb.get("from_x_mm", 0) - x_min + x_cursor, 1)
            ngb["to_x_mm"]   = round(gb.get("to_x_mm",   0) - x_min + x_cursor, 1)
            ngb["from_y_mm"] = round(gb.get("from_y_mm", 0) - y_min, 1)
            ngb["to_y_mm"]   = round(gb.get("to_y_mm",   0) - y_min, 1)
            all_gbeams.append(ngb)

        for rf in r.get("rafts", []):
            nrf = dict(rf)
            nrf["x_from_mm"] = round(rf.get("x_from_mm", 0) - x_min + x_cursor, 1)
            nrf["x_to_mm"]   = round(rf.get("x_to_mm",   0) - x_min + x_cursor, 1)
            nrf["y_from_mm"] = round(rf.get("y_from_mm", 0) - y_min, 1)
            nrf["y_to_mm"]   = round(rf.get("y_to_mm",   0) - y_min, 1)
            all_rafts.append(nrf)

        merged_schedule.update(r.get("schedule", {}))
        x_cursor += bldg_width + 5_000   # 5 m gap between buildings

    return {
        "footings":        all_footings,
        "ground_beams":    all_gbeams,
        "rafts":           all_rafts,
        "lift_pits":       all_lift_pits,
        "has_planter_slab": has_planter,
        "schedule":        merged_schedule,
        "pile_spec":       results[0].get("pile_spec", {}),
        "has_foundation_plan": True,
        "_vision_foundations": any(r.get("_vision_foundations") for r in results),
        "_vision_schedule":    any(r.get("_vision_schedule")    for r in results),
        "_vision_grid":        any(r.get("_vision_grid")        for r in results),
    }


def _vision_fallback(
    page: fitz.Page,
    grid: dict,
    fdn_result: dict,
    quality,
) -> dict:
    """Run targeted Vision AI calls to fill specific extraction gaps.

    Only imports vision_extractor when actually needed (keeps startup fast
    and avoids Vertex AI auth errors when AI is not configured).
    """
    import logging
    log = logging.getLogger(__name__)

    try:
        from pipeline.vision_extractor import (
            extract_grid_vision,
            extract_schedule_vision,
            extract_foundations_vision,
        )
    except ImportError as e:
        log.warning("vision_extractor import failed (%s) — skipping AI fallback", e)
        return fdn_result

    improved = dict(fdn_result)
    failed = quality.failed_checks

    # Fix grid first (other fixes depend on it)
    if "grid" in failed:
        log.info("Vision fallback: extracting grid from image")
        vision_grid = extract_grid_vision(page)
        if vision_grid:
            grid = _apply_vision_grid(vision_grid, page, grid)
            improved["_vision_grid"] = True

    # Fix schedule (dims=0 or schedule check failed)
    if "schedule" in failed or "dims" in failed:
        log.info("Vision fallback: extracting schedule from image")
        vision_sched = extract_schedule_vision(page)
        if vision_sched:
            improved["schedule"] = _merge_schedules(
                improved.get("schedule", {}), vision_sched
            )
            improved["_vision_schedule"] = True

    # Fix foundation positions (missing, bad coords, or bad coverage)
    if any(c in failed for c in ("foundations", "coords", "coverage")):
        log.info("Vision fallback: extracting foundation positions from image")
        vision_fdns = extract_foundations_vision(
            page, grid, improved.get("schedule", {})
        )
        if vision_fdns:
            existing = improved.get("footings", [])
            # Prefer vision results if they're more complete
            if len(vision_fdns) >= len(existing):
                improved["footings"] = vision_fdns
            else:
                improved["footings"] = existing + [
                    f for f in vision_fdns
                    if f["grid_ref"] not in {e.get("grid_ref") for e in existing}
                ]
            improved["_vision_foundations"] = True

    return improved


def _apply_vision_grid(vision_grid: dict, page: fitz.Page, fallback_grid: dict) -> dict:
    """Convert Gemini grid (x_percent/y_percent) to real_mm coords."""
    from pipeline.grid_extractor import PT_TO_MM

    pw = page.rect.width
    ph = page.rect.height
    scale = vision_grid.get("scale", fallback_grid.get("scale", 100))
    pt_to_mm = PT_TO_MM * scale

    def _build_axes(items, dim, coord_key):
        sorted_items = sorted(items, key=lambda a: a.get(coord_key, 0))
        base_pdf = sorted_items[0].get(coord_key, 0) * dim if sorted_items else 0.0
        axes = []
        for a in sorted_items:
            pdf_pos = a.get(coord_key, 0) * dim
            axes.append({
                "label":   str(a.get("label", "?")),
                "pdf_pos": round(pdf_pos, 2),
                "real_mm": round((pdf_pos - base_pdf) * pt_to_mm, 1),
            })
        return axes

    x_axes = _build_axes(vision_grid.get("x_axes", []), pw, "x_percent")
    y_axes = _build_axes(vision_grid.get("y_axes", []), ph, "y_percent")

    return {
        **fallback_grid,
        "x_axes":   x_axes or fallback_grid.get("x_axes", []),
        "y_axes":   y_axes or fallback_grid.get("y_axes", []),
        "scale":    scale,
        "pt_to_mm": round(pt_to_mm, 4),
        "source":   "vision_ai",
    }


def _merge_schedules(vector_sched: dict, vision_list: list[dict]) -> dict:
    """Merge vector schedule with vision AI schedule, preferring vision for zero-dim entries."""
    merged = dict(vector_sched)
    for entry in vision_list:
        mark = str(entry.get("mark", "")).upper()
        if not mark:
            continue
        existing = merged.get(mark, {})
        # Use vision data if existing dims are zero
        if existing.get("width_mm", 0) == 0 and existing.get("pile_dia_mm", 0) == 0:
            merged[mark] = {
                "ftype":        entry.get("ftype", "pile_cap"),
                "pile_dia_mm":  entry.get("pile_dia_mm", 0),
                "pile_len_mm":  entry.get("socket_m", 0) * 1000,
                "pile_count":   entry.get("pile_count", 1),
                "width_mm":     entry.get("width_mm", 0),
                "depth_mm":     entry.get("depth_mm", 0),
                "height_mm":    entry.get("height_mm", 0),
                "source":       "vision_ai",
            }
        elif not existing:
            merged[mark] = {
                "ftype":        entry.get("ftype", "pile_cap"),
                "pile_dia_mm":  entry.get("pile_dia_mm", 0),
                "pile_len_mm":  entry.get("socket_m", 0) * 1000,
                "pile_count":   entry.get("pile_count", 1),
                "width_mm":     entry.get("width_mm", 0),
                "depth_mm":     entry.get("depth_mm", 0),
                "height_mm":    entry.get("height_mm", 0),
                "source":       "vision_ai",
            }
    return merged


def _detect_dominant_scale(classifications: list[dict]) -> int:
    """Return the dominant structural plan scale.

    Prefers scales in the structural plan range (50–150) over site plan
    scales (>200) or large-scale details (<30).
    """
    from collections import Counter
    # Prefer typical structural plan scales (1:50–1:150)
    struct = [
        c["scale_ratio"] for c in classifications
        if c.get("scale_ratio") and 50 <= c["scale_ratio"] <= 150
    ]
    if struct:
        return Counter(struct).most_common(1)[0][0]
    # Fallback: any non-extreme scale
    any_s = [
        c["scale_ratio"] for c in classifications
        if c.get("scale_ratio") and 30 <= c["scale_ratio"] <= 250
    ]
    return Counter(any_s).most_common(1)[0][0] if any_s else 100


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
