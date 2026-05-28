"""Stage 5: Generate SketchUp Ruby (.rb) from unified model.

Uses exact mm coordinates from the vector pipeline.
All units in mm, converted to inches for SketchUp (1mm = 0.0393701 in).

Output: data/jobs/{job_id}/sketchup_model.rb
        data/jobs/{job_id}/stage5_generation_report.json
"""

import json
import math
import time
from pathlib import Path

MM_TO_IN = 0.0393701  # 1 mm in SketchUp inches


def _in(mm) -> float:
    try:
        return round(float(mm) * MM_TO_IN, 6)
    except (TypeError, ValueError):
        return 0.0


def _level_z(level_name: str, levels: list[dict]) -> float:
    for lv in levels:
        if lv["name"] == level_name:
            return _in(lv["elevation_mm"])
    return 0.0


def generate_ruby(model: dict, job_dir: str) -> dict:
    """Generate .rb and report. Returns report dict."""
    job_dir = Path(job_dir)
    levels      = model.get("levels", [])
    columns     = model.get("columns", [])
    beams       = model.get("beams", [])
    slabs       = model.get("slabs", [])
    foundations = model.get("foundations", [])

    skipped  = []
    warnings = []
    lines    = []
    counts   = {"columns": 0, "beams": 0, "slabs": 0, "foundations": 0}

    bottom_lv = levels[0]  if levels else {"name": "GROUND FLOOR", "elevation_mm": 0}
    top_lv    = levels[-1] if levels else {"name": "ROOF",         "elevation_mm": 18000}

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# =============================================================",
        "# auto_pipeline — PDF Structural → SketchUp 3D (vector build)",
        f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"# Columns: {len(columns)}  Beams: {len(beams)}  "
        f"Slabs: {len(slabs)}  Foundations: {len(foundations)}",
        "# Run via SketchUp: Extensions > Ruby Console → load 'path/to/file.rb'",
        "# =============================================================",
        "",
        "model = Sketchup.active_model",
        "model.start_operation('auto_pipeline structural model', true)",
        "ents = model.active_entities",
        "",
    ]

    # ── Layers ────────────────────────────────────────────────────────────────
    lines += [
        "lyr_found = model.layers.add('Foundations')",
        "lyr_col   = model.layers.add('Columns')",
        "lyr_beam  = model.layers.add('Beams')",
        "lyr_slab  = model.layers.add('Slabs')",
        "",
    ]

    # ── Materials ─────────────────────────────────────────────────────────────
    lines += [
        "mat_conc  = model.materials.add('Concrete');  mat_conc.color  = Sketchup::Color.new(190,185,175)",
        "mat_steel = model.materials.add('Steel');     mat_steel.color = Sketchup::Color.new(90,130,175)",
        "mat_slab  = model.materials.add('Slab');      mat_slab.color  = Sketchup::Color.new(215,205,190)",
        "",
    ]

    # ── Helper functions ──────────────────────────────────────────────────────
    lines += [
        "# Axis-aligned box",
        "def ap_box(ents, x, y, z, w, d, h, lyr, mat)",
        "  g = ents.add_group; g.layer = lyr",
        "  f = g.entities.add_face(",
        "    [x,y,z],[x+w,y,z],[x+w,y+d,z],[x,y+d,z])",
        "  f.pushpull(h)",
        "  g.entities.each{|e| e.material=mat if e.is_a?(Sketchup::Face)}",
        "  g",
        "end",
        "",
        "# Beam between two 3D points (any angle, uses Transform)",
        "def ap_beam(ents, x1,y1,z1, x2,y2,z2, bw,bh, lyr, mat)",
        "  dx=x2-x1; dy=y2-y1",
        "  len=Math.sqrt(dx*dx+dy*dy); return if len<1e-6",
        "  ang=Math.atan2(dy,dx)",
        "  g=ents.add_group; g.layer=lyr",
        "  ge=g.entities",
        "  f=ge.add_face([0,-bw/2.0,z1-bh],[len,-bw/2.0,z1-bh],",
        "                [len, bw/2.0,z1-bh],[0,  bw/2.0,z1-bh])",
        "  f.pushpull(bh)",
        "  g.entities.each{|e| e.material=mat if e.is_a?(Sketchup::Face)}",
        "  tr=Geom::Transformation.rotation(Geom::Point3d.new(0,0,0),",
        "       Geom::Vector3d.new(0,0,1),ang)*",
        "     Geom::Transformation.translation(Geom::Vector3d.new(x1,y1,0))",
        "  g.transform!(tr); g",
        "end",
        "",
        "# Circular cylinder (piles / round columns)",
        "def ap_cylinder(ents, cx,cy,z_bot, radius, height, lyr, mat)",
        "  g=ents.add_group; g.layer=lyr",
        "  ge=g.entities",
        "  n=16  # segments",
        "  pts=(0...n).map{|i| a=2*Math::PI*i/n; Geom::Point3d.new(cx+radius*Math.cos(a),cy+radius*Math.sin(a),z_bot)}",
        "  f=ge.add_face(pts)",
        "  f.pushpull(height)",
        "  g.entities.each{|e| e.material=mat if e.is_a?(Sketchup::Face)}",
        "  g",
        "end",
        "",
    ]

    # ── Foundations (piles) ───────────────────────────────────────────────────
    lines += ["# ===== FOUNDATIONS (PILES) ====="]
    z_pile_top = _in(bottom_lv.get("elevation_mm", 0))
    z_pile_bot = z_pile_top - _in(500)  # pile goes 500mm below ground

    for f in foundations:
        x = _in(f.get("x_mm", 0))
        y = _in(f.get("y_mm", 0))
        r = _in(f.get("width_mm", 750) / 2)
        lines.append(
            f"ap_cylinder(ents, {x:.6f},{y:.6f},{z_pile_bot:.6f},{r:.6f},"
            f"{_in(500):.6f},lyr_found,mat_conc)"
            f"  # {f.get('id','')} @ {f.get('grid_ref','')}"
        )
        counts["foundations"] += 1
    lines.append("")

    # ── Columns ───────────────────────────────────────────────────────────────
    lines += ["# ===== COLUMNS ====="]
    for col in columns:
        x   = _in(col.get("x_mm", 0))
        y   = _in(col.get("y_mm", 0))
        w   = _in(col.get("width_mm", 150))
        d   = _in(col.get("depth_mm", 150))
        z_b = _in(col.get("from_elev_mm", 0))
        z_t = _in(col.get("to_elev_mm", 18000))
        h   = z_t - z_b
        if h <= 0:
            h = _in(18000)
            warnings.append(f"{col.get('id','?')}: zero height, defaulted 18m")

        mat = "mat_steel" if col.get("material", "steel") == "steel" else "mat_conc"
        lines.append(
            f"ap_box(ents, {x-w/2:.6f},{y-d/2:.6f},{z_b:.6f},"
            f"{w:.6f},{d:.6f},{h:.6f},lyr_col,{mat})"
            f"  # {col.get('id','')} @ {col.get('grid_ref','')}"
        )
        counts["columns"] += 1
    lines.append("")

    # ── Beams ──────────────────────────────────────────────────────────────────
    lines += ["# ===== BEAMS ====="]
    for beam in beams:
        x1 = _in(beam.get("from_x_mm", 0))
        y1 = _in(beam.get("from_y_mm", 0))
        x2 = _in(beam.get("to_x_mm", 0))
        y2 = _in(beam.get("to_y_mm", 0))
        z  = _in(beam.get("elev_mm", top_lv.get("elevation_mm", 18000)))
        bw = _in(beam.get("width_mm", 100))
        bh = _in(beam.get("height_mm", 200))

        span = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if span < 0.01:
            skipped.append(f"{beam.get('id','?')}: zero span")
            continue

        mat = "mat_steel" if beam.get("material", "steel") == "steel" else "mat_conc"
        lines.append(
            f"ap_beam(ents, {x1:.6f},{y1:.6f},{z:.6f},{x2:.6f},{y2:.6f},{z:.6f},"
            f"{bw:.6f},{bh:.6f},lyr_beam,{mat})"
            f"  # {beam.get('id','')} {beam.get('section_label','')} @ {beam.get('grid_ref','')}"
        )
        counts["beams"] += 1
    lines.append("")

    # ── Slabs ──────────────────────────────────────────────────────────────────
    lines += ["# ===== SLABS ====="]
    for slab in slabs:
        x1 = _in(slab.get("from_x_mm", 0))
        y1 = _in(slab.get("from_y_mm", 0))
        x2 = _in(slab.get("to_x_mm", 0))
        y2 = _in(slab.get("to_y_mm", 0))
        z  = _in(slab.get("elev_mm", top_lv.get("elevation_mm", 18000)))
        t  = _in(slab.get("thickness_mm", 150))
        sw = abs(x2 - x1)
        sd = abs(y2 - y1)

        if sw < 0.01 or sd < 0.01:
            skipped.append(f"{slab.get('id','?')}: zero area")
            continue

        lines.append(
            f"ap_box(ents, {min(x1,x2):.6f},{min(y1,y2):.6f},{z:.6f},"
            f"{sw:.6f},{sd:.6f},{t:.6f},lyr_slab,mat_slab)"
            f"  # {slab.get('id','')} @ {slab.get('grid_ref','')}"
        )
        counts["slabs"] += 1
    lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "model.commit_operation",
        "puts 'auto_pipeline vector build: model loaded.'",
        "puts \"Foundations:#{" + str(counts["foundations"]) + "} Columns:#{" +
        str(counts["columns"]) + "} Beams:#{" + str(counts["beams"]) +
        "} Slabs:#{" + str(counts["slabs"]) + "}\"",
    ]

    ruby_code = "\n".join(lines)
    (job_dir / "sketchup_model.rb").write_text(ruby_code, encoding="utf-8")

    report = {
        "stage": 5,
        "stage_name": "Ruby Generator",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elements_generated": counts,
        "ruby_line_count": len(lines),
        "warnings": warnings,
        "skipped": skipped,
    }
    (job_dir / "stage5_generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
