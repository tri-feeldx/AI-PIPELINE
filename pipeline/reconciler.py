"""Stage 4: Reconcile raw extractions into a unified structural model.

Pure Python — no AI calls. Merges data across pages, builds grid coordinates,
deduplicates elements, resolves conflicts.

Output: data/jobs/{job_id}/stage4_unified_model.json
"""

import json
import re
import time
from pathlib import Path


def _safe_num(value, default=0):
    """Return numeric value, falling back to default for None/non-numeric."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Grid ──────────────────────────────────────────────────────────────────────

def _merge_grids(extractions: list) -> dict:
    """Union grid axes from ALL plan pages (not just the best one).

    For each unique axis label, keeps the cumulative_mm from the highest-confidence page.
    This captures axes that appear on some pages but not others.
    """
    # {label → {"cumulative_mm": float, "conf": float}}
    all_x: dict = {}
    all_y: dict = {}
    source_pages: list = []

    for e in extractions:
        g = e.get("grid")
        if not g:
            continue
        conf = _safe_num(g.get("dimension_confidence"), 0.5)
        source_pages.append(e["page_num"])

        for ax in (g.get("x_axes") or []):
            label = str(ax.get("label", "")).strip()
            mm = ax.get("cumulative_mm")
            if label and mm is not None:
                mm_f = _safe_num(mm, 0)
                if label not in all_x or conf > all_x[label]["conf"]:
                    all_x[label] = {"cumulative_mm": mm_f, "conf": conf}

        for ax in (g.get("y_axes") or []):
            label = str(ax.get("label", "")).strip()
            mm = ax.get("cumulative_mm")
            if label and mm is not None:
                mm_f = _safe_num(mm, 0)
                if label not in all_y or conf > all_y[label]["conf"]:
                    all_y[label] = {"cumulative_mm": mm_f, "conf": conf}

    # Build sorted axis lists
    x_axes = sorted(
        [{"label": k, "cumulative_mm": v["cumulative_mm"]} for k, v in all_x.items()],
        key=lambda a: a["cumulative_mm"],
    )
    y_axes = sorted(
        [{"label": k, "cumulative_mm": v["cumulative_mm"]} for k, v in all_y.items()],
        key=lambda a: a["cumulative_mm"],
    )

    # Zero-base both axes
    for axes in (x_axes, y_axes):
        if axes:
            base = axes[0]["cumulative_mm"]
            for a in axes:
                a["cumulative_mm"] = round(a["cumulative_mm"] - base, 1)

    return {
        "x_axes": x_axes,
        "y_axes": y_axes,
        "source_pages": source_pages,
        "unit": "mm",
    }


def _resolve_grid_ref(grid_ref: str, grid: dict) -> dict | None:
    """Convert grid ref to XY mm coordinates.

    Supports multiple formats:
    - Australian/NZ  : "B/1", "B.5/1", "B/1-2" (Y/X with slash)
    - Concatenated   : "A1", "1A", "B1-B2"
    - Span           : "A1-B2", "B/1-2"
    """
    if not grid_ref:
        return None

    ref = grid_ref.strip()

    x_labels = {
        str(a["label"]): _safe_num(a.get("cumulative_mm"), 0)
        for a in (grid.get("x_axes") or [])
        if a.get("label") is not None
    }
    y_labels = {
        str(a["label"]): _safe_num(a.get("cumulative_mm"), 0)
        for a in (grid.get("y_axes") or [])
        if a.get("label") is not None
    }

    def _lookup_x(label: str) -> float | None:
        """Look up X coordinate, interpolating for intermediate grids like '1.5'."""
        if label in x_labels:
            return x_labels[label]
        # Try integer part e.g. "1.5" → between "1" and "2"
        try:
            v = float(label)
            lo = str(int(v))
            hi = str(int(v) + 1)
            if lo in x_labels and hi in x_labels:
                frac = v - int(v)
                return x_labels[lo] + frac * (x_labels[hi] - x_labels[lo])
        except ValueError:
            pass
        return None

    def _lookup_y(label: str) -> float | None:
        """Look up Y coordinate, interpolating for intermediate grids like 'B.5'."""
        if label in y_labels:
            return y_labels[label]
        # Try float-ish label e.g. "B.5"
        parts = label.split(".")
        if len(parts) == 2:
            base, frac_str = parts[0], parts[1]
            if base in y_labels:
                # Find next Y label after base
                keys = sorted(y_labels.keys())
                idx = keys.index(base) if base in keys else -1
                if 0 <= idx < len(keys) - 1:
                    nxt = keys[idx + 1]
                    try:
                        frac = int(frac_str) / 10
                        return y_labels[base] + frac * (y_labels[nxt] - y_labels[base])
                    except ValueError:
                        pass
        return None

    # ── Slab range: "B-C/2-4" or "1-4/C-D" (panel covering Y-range / X-range) ──
    if "/" in ref:
        slash_idx = ref.index("/")
        left  = ref[:slash_idx].strip()
        right = ref[slash_idx + 1:].strip()

        # Try both orientations: left=Y-range, right=X-range AND left=X-range, right=Y-range
        def _try_range(y_part: str, x_part: str):
            y1_s, _, y2_s = y_part.partition("-")
            x1_s, _, x2_s = x_part.partition("-")
            y1 = _lookup_y(y1_s.strip()) if "-" in y_part else _lookup_y(y_part.strip())
            y2 = _lookup_y(y2_s.strip()) if "-" in y_part else None
            x1 = _lookup_x(x1_s.strip()) if "-" in x_part else _lookup_x(x_part.strip())
            x2 = _lookup_x(x2_s.strip()) if "-" in x_part else None

            fy = y1
            ty = y2 if y2 is not None else y1
            fx = x1
            tx = x2 if x2 is not None else x1

            if all(v is not None for v in (fy, ty, fx, tx)):
                return {
                    "x_mm": (fx + tx) / 2,
                    "y_mm": (fy + ty) / 2,
                    "from_x_mm": fx, "from_y_mm": fy,
                    "to_x_mm": tx, "to_y_mm": ty,
                }
            return None

        result = _try_range(left, right)
        if result:
            return result
        # Try reversed orientation (X-range / Y-range)
        result = _try_range(right, left)
        if result:
            return result

    # ── Australian point: "B/1" or "B.5/2" ───────────────────────────────────
    if "/" in ref:
        slash_idx = ref.index("/")
        y_part = ref[:slash_idx].strip()
        x_part = ref[slash_idx + 1:].strip()
        y_mm = _lookup_y(y_part)
        x_mm = _lookup_x(x_part)
        if y_mm is not None and x_mm is not None:
            return {"x_mm": x_mm, "y_mm": y_mm}

    # ── Span with dash: "A1-B2" or "B/1-B/3" ────────────────────────────────
    if "-" in ref:
        dash_parts = ref.split("-")
        pt1 = _resolve_grid_ref(dash_parts[0].strip(), grid)
        pt2 = _resolve_grid_ref(dash_parts[-1].strip(), grid)
        if pt1 and pt2:
            return {
                "x_mm": (pt1["x_mm"] + pt2["x_mm"]) / 2,
                "y_mm": (pt1["y_mm"] + pt2["y_mm"]) / 2,
                "from_x_mm": pt1["x_mm"], "from_y_mm": pt1["y_mm"],
                "to_x_mm": pt2["x_mm"], "to_y_mm": pt2["y_mm"],
            }

    # ── Concatenated: "A1", "1A" ─────────────────────────────────────────────
    for split in range(1, len(ref)):
        a, b = ref[:split], ref[split:]
        if a in x_labels and b in y_labels:
            return {"x_mm": x_labels[a], "y_mm": y_labels[b]}
        if a in y_labels and b in x_labels:
            return {"x_mm": x_labels[b], "y_mm": y_labels[a]}

    # ── Descriptive refs (Gemini hallucinations) ──────────────────────────────
    # "Between Y-X1 & Y-X2" → midpoint  e.g. "Between E-1 & E-2"
    m = re.search(
        r"between\s+([A-Za-z\.]+)[/-](\d+)\s+(?:&|and)\s+([A-Za-z\.]+)[/-](\d+)",
        ref, re.IGNORECASE,
    )
    if m:
        pt1 = _resolve_grid_ref(f"{m.group(1)}/{m.group(2)}", grid)
        pt2 = _resolve_grid_ref(f"{m.group(3)}/{m.group(4)}", grid)
        if pt1 and pt2:
            return {
                "x_mm": (pt1["x_mm"] + pt2["x_mm"]) / 2,
                "y_mm": (pt1["y_mm"] + pt2["y_mm"]) / 2,
            }

    # "North/South/East/West of Y/X" → use the grid point itself
    m2 = re.search(
        r"(?:north|south|east|west)\s+of\s+([A-Za-z\.]+)[/-](\d+)",
        ref, re.IGNORECASE,
    )
    if m2:
        return _resolve_grid_ref(f"{m2.group(1)}/{m2.group(2)}", grid)

    return None


# ── Levels ────────────────────────────────────────────────────────────────────

def _normalise_level_name(name: str) -> str:
    """Normalise level name for comparison: strip, lowercase, collapse spaces."""
    return " ".join(name.lower().split()).replace("level 0", "level ").replace("floor 0", "floor ")


def _merge_levels(extractions: list) -> list:
    """Collect unique levels, deduplicate by elevation proximity, sort by elevation."""
    raw = []
    for e in extractions:
        for lv in (e.get("levels") or []):
            name = (lv.get("name") or "").strip()
            elev = _safe_num(lv.get("elevation_mm"), None)
            if name:
                raw.append({"name": name, "elevation_mm": elev})

    if not raw:
        return [
            {"name": "Foundation", "elevation_mm": -600},
            {"name": "Ground Floor", "elevation_mm": 0},
            {"name": "Roof", "elevation_mm": 4000},
        ]

    # Cluster by (normalised name OR elevation within 200mm tolerance)
    clusters: list[dict] = []
    for lv in raw:
        norm = _normalise_level_name(lv["name"])
        elev = lv["elevation_mm"]
        matched = None
        for cluster in clusters:
            # Same normalised name → same level
            if norm == _normalise_level_name(cluster["name"]):
                matched = cluster
                break
            # Elevation within 200mm → same level (different pages, slight variation)
            if elev is not None and cluster["elevation_mm"] is not None:
                if abs(elev - cluster["elevation_mm"]) < 200:
                    matched = cluster
                    break
        if matched:
            # Keep best elevation (non-None wins; prefer smaller absolute value on tie)
            if matched["elevation_mm"] is None and elev is not None:
                matched["elevation_mm"] = elev
        else:
            clusters.append({"name": lv["name"], "elevation_mm": elev})

    # Replace any remaining None elevations with estimated values
    known = [c["elevation_mm"] for c in clusters if c["elevation_mm"] is not None]
    for i, c in enumerate(clusters):
        if c["elevation_mm"] is None:
            c["elevation_mm"] = (i * 3600) if not known else (max(known) + 3600 * (i + 1))

    # Sort by elevation
    clusters.sort(key=lambda c: _safe_num(c["elevation_mm"], 0))
    return clusters


# ── Section lookup ────────────────────────────────────────────────────────────

def _build_section_lookup(extractions: list) -> dict:
    """Build {label → dimensions} from all schedule pages."""
    lookup = {}
    for e in extractions:
        for s in (e.get("schedules") or []):
            label = (s.get("label") or "").strip()
            if label:
                lookup[label] = {
                    "width_mm": _safe_num(s.get("width_mm"), None),
                    "height_mm": _safe_num(s.get("height_mm"), None),
                    "element_type": s.get("element_type"),
                    "material": s.get("material", "steel"),
                    "notes": s.get("notes", ""),
                }
    return lookup


def _get_section_dims(label: str, lookup: dict) -> dict:
    """Look up section dimensions, return defaults if not found."""
    if label and label in lookup:
        entry = lookup[label]
        w = entry.get("width_mm") or 150
        h = entry.get("height_mm") or 200
        return {"width_mm": w, "height_mm": h, "material": entry.get("material", "steel")}

    label_up = (label or "").upper()
    if "UC" in label_up or "UB" in label_up:
        return {"width_mm": 150, "height_mm": 200, "material": "steel"}
    if any(x in label_up for x in ("CH", "SH", "PFC")):
        return {"width_mm": 100, "height_mm": 100, "material": "steel"}
    if any(x in label_up for x in ("SHS", "RHS", "CHS")):
        return {"width_mm": 100, "height_mm": 100, "material": "steel"}
    if any(x in label_up for x in ("RC", "CONC", "PT")):
        return {"width_mm": 300, "height_mm": 300, "material": "concrete"}
    return {"width_mm": 150, "height_mm": 150, "material": "steel"}


# ── Element merging ───────────────────────────────────────────────────────────

def _merge_elements(extractions: list, grid: dict, levels: list, section_lookup: dict) -> dict:
    """Merge all elements from all pages into typed lists with coordinates."""
    columns: dict[str, dict] = {}   # keyed "grid_ref|level" to dedup
    beams: list[dict] = []
    slabs: list[dict] = []
    foundations: list[dict] = []
    merge_log: list[str] = []

    bottom_level = levels[0] if levels else {"name": "Foundation", "elevation_mm": -600}
    top_level = levels[-1] if levels else {"name": "Roof", "elevation_mm": 4000}

    for e in extractions:
        page_num = e["page_num"]
        for elem in (e.get("elements") or []):
            etype = ((elem.get("element_type") or "")).lower()
            grid_ref = ((elem.get("grid_ref") or "")).strip()
            section = ((elem.get("section_label") or "")).strip()
            level_name = ((elem.get("level_name") or "")).strip()
            coords = _resolve_grid_ref(grid_ref, grid)
            dims = _get_section_dims(section, section_lookup)

            if "column" in etype:
                key = f"{grid_ref}|{level_name or 'all'}"
                if key not in columns:
                    columns[key] = {
                        "id": f"COL-{len(columns) + 1:03d}",
                        "grid_ref": grid_ref,
                        "section_label": section,
                        "from_level": bottom_level["name"],
                        "to_level": top_level["name"],
                        "coords": coords,
                        "width_mm": dims["width_mm"],
                        "depth_mm": dims["height_mm"],
                        "material": dims["material"],
                        "source_page": page_num,
                    }
                    merge_log.append(f"Column {key} ← page {page_num}")

            elif "beam" in etype:
                beams.append({
                    "id": f"BEAM-{len(beams) + 1:03d}",
                    "grid_ref": grid_ref,
                    "section_label": section,
                    "level": level_name or top_level["name"],
                    "coords": coords,
                    "width_mm": dims["width_mm"],
                    "height_mm": dims["height_mm"],
                    "material": dims["material"],
                    "source_page": page_num,
                    "notes": (elem.get("notes") or ""),
                })
                merge_log.append(f"Beam {grid_ref} ← page {page_num}")

            elif any(x in etype for x in ("slab", "topping", "suspended")):
                slabs.append({
                    "id": f"SLAB-{len(slabs) + 1:03d}",
                    "grid_ref": grid_ref,
                    "section_label": section,
                    "level": level_name or top_level["name"],
                    "coords": coords,
                    "thickness_mm": dims["height_mm"],
                    "material": dims["material"],
                    "source_page": page_num,
                })

            elif any(x in etype for x in ("footing", "foundation", "pile", "raft", "strip", "pad")):
                plan_size = (elem.get("plan_size_label") or "")
                # Parse diameter from "750 DIA" or width from "1200x1200"
                dia = 750
                m_dia = re.search(r"(\d+)\s*(?:dia|diam)", plan_size, re.IGNORECASE)
                if m_dia:
                    dia = int(m_dia.group(1))
                elif "x" in str(plan_size).lower():
                    try:
                        dia = int(str(plan_size).split("x")[0].strip())
                    except ValueError:
                        pass

                # If ref is descriptive ("Perimeter"/"Internal"), mark for grid-gen
                descriptive = not coords and grid_ref.lower() in (
                    "perimeter", "internal", "under rf1", "general"
                )
                foundations.append({
                    "id": f"FOUND-{len(foundations) + 1:03d}",
                    "type": etype,
                    "grid_ref": grid_ref,
                    "section_label": section,
                    "coords": coords,
                    "width_mm": dia,
                    "depth_mm": int(_safe_num(elem.get("depth_label"), 500)),
                    "material": "concrete",
                    "source_page": page_num,
                    "descriptive": descriptive,
                    "notes": (elem.get("notes") or ""),
                })

    # ── Generate pile positions from grid for descriptive foundations ─────────
    has_perimeter = any(f.get("grid_ref", "").lower() == "perimeter" for f in foundations)
    has_internal  = any(f.get("grid_ref", "").lower() == "internal"  for f in foundations)

    if (has_perimeter or has_internal) and grid.get("x_axes") and grid.get("y_axes"):
        x_axes = grid["x_axes"]
        y_axes = grid["y_axes"]

        # Get perimeter/internal pile dimensions from found entries
        perim_w  = next((f["width_mm"] for f in foundations if f.get("grid_ref","").lower()=="perimeter"), 750)
        intern_w = next((f["width_mm"] for f in foundations if f.get("grid_ref","").lower()=="internal"),  750)

        # Remove descriptive entries (will be replaced by grid-generated ones)
        foundations = [f for f in foundations if not f.get("descriptive")]

        for yi, yax in enumerate(y_axes):
            for xi, xax in enumerate(x_axes):
                is_perimeter = (
                    yi == 0 or yi == len(y_axes) - 1
                    or xi == 0 or xi == len(x_axes) - 1
                )
                w = perim_w if is_perimeter else intern_w
                foundations.append({
                    "id": f"PILE-{yax['label']}{xax['label']}",
                    "type": "pile",
                    "grid_ref": f"{yax['label']}/{xax['label']}",
                    "section_label": f"{w} DIA",
                    "coords": {"x_mm": xax["cumulative_mm"], "y_mm": yax["cumulative_mm"]},
                    "width_mm": w,
                    "depth_mm": 500,
                    "material": "concrete",
                    "source_page": "grid_generated",
                    "descriptive": False,
                })
        merge_log.append(
            f"Generated {len(x_axes)*len(y_axes)} pile positions from {len(x_axes)}x{len(y_axes)} grid"
        )

    return {
        "columns": list(columns.values()),
        "beams": beams,
        "slabs": slabs,
        "foundations": foundations,
        "merge_log": merge_log,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def reconcile(extractions_result: dict, job_dir: str) -> dict:
    """Build unified model from stage3 raw extractions."""
    job_dir = Path(job_dir)
    pages = extractions_result["pages"]

    grid = _merge_grids(pages)
    levels = _merge_levels(pages)
    section_lookup = _build_section_lookup(pages)
    elements = _merge_elements(pages, grid, levels, section_lookup)

    conflicts_resolved = []
    for col in elements["columns"]:
        if col["coords"] is None:
            conflicts_resolved.append(
                f"Column {col['grid_ref']}: grid ref unresolved, placed at origin"
            )
            col["coords"] = {"x_mm": 0, "y_mm": 0}

    unified = {
        "stage": 4,
        "stage_name": "Reconciler",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_info": {
            "source_pages": len(pages),
            "grid_source_page": grid.get("source_page"),
        },
        "grid_system": {
            "x_axes": grid.get("x_axes", []),
            "y_axes": grid.get("y_axes", []),
            "unit": grid.get("unit", "mm"),
        },
        "levels": levels,
        "section_lookup": section_lookup,
        "columns": elements["columns"],
        "beams": elements["beams"],
        "slabs": elements["slabs"],
        "foundations": elements["foundations"],
        "summary_counts": {
            "columns": len(elements["columns"]),
            "beams": len(elements["beams"]),
            "slabs": len(elements["slabs"]),
            "foundations": len(elements["foundations"]),
            "levels": len(levels),
        },
        "merge_log": elements["merge_log"],
        "conflicts_resolved": conflicts_resolved,
    }

    (job_dir / "stage4_unified_model.json").write_text(json.dumps(unified, indent=2), encoding="utf-8")
    return unified
