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


def _bim_attrs(var: str, attrs: dict) -> list[str]:
    """Return Ruby lines that set BIM attributes on a SketchUp group variable."""
    if not attrs:
        return []
    lines = [f"if {var}"]
    for k, v in attrs.items():
        if isinstance(v, str):
            lines.append(f"  {var}.set_attribute('BIM','{k}',{repr(v)})")
        elif isinstance(v, bool):
            lines.append(f"  {var}.set_attribute('BIM','{k}',{'true' if v else 'false'})")
        else:
            lines.append(f"  {var}.set_attribute('BIM','{k}',{v})")
    lines.append("end")
    return lines


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


def _pile_offsets(pile_count: int, dia_mm: float) -> list[tuple[float, float]]:
    """Return (dx_mm, dy_mm) offsets from cap centre for each pile in a group.

    Uses standard 2.5D centre-to-centre spacing (LOD 300 simplified arrangement).
    """
    s = dia_mm * 2.5   # spacing
    h = s * 0.5        # half-spacing
    r3 = s * 0.577     # s/√3  (triangle geometry)

    if pile_count <= 0:
        return [(0, 0)]   # single centred pile as default

    layouts: dict[int, list[tuple[float, float]]] = {
        1: [(0, 0)],
        2: [(-h, 0), (h, 0)],
        3: [(0, r3), (-h, -r3 * 0.5), (h, -r3 * 0.5)],
        4: [(-h, -h), (h, -h), (h, h), (-h, h)],
        5: [(0, 0), (-h, -h), (h, -h), (h, h), (-h, h)],
        6: [(-s, -h), (0, -h), (s, -h), (-s, h), (0, h), (s, h)],
    }
    if pile_count in layouts:
        return layouts[pile_count]

    # Generic grid layout for larger counts
    cols = max(1, math.ceil(math.sqrt(pile_count)))
    rows = math.ceil(pile_count / cols)
    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= pile_count:
                break
            dx = (c - (cols - 1) / 2) * s
            dy = (r - (rows - 1) / 2) * s
            positions.append((dx, dy))
    return positions


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
    counts   = {"columns": 0, "beams": 0, "slabs": 0, "foundations": 0, "ground_beams": 0}

    bottom_lv = levels[0]  if levels else {"name": "GROUND FLOOR", "elevation_mm": 0}
    top_lv    = levels[-1] if levels else {"name": "ROOF",         "elevation_mm": 18000}

    ground_beams  = model.get("ground_beams", [])
    rafts         = model.get("rafts", [])
    lift_pits     = model.get("lift_pits", [])
    has_planter   = model.get("has_planter_slab", False)

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# =============================================================",
        "# auto_pipeline — PDF Structural → SketchUp 3D (vector build)",
        f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"# Columns: {len(columns)}  Beams: {len(beams)}  Slabs: {len(slabs)}  "
        f"Foundations: {len(foundations)}  GroundBeams: {len(ground_beams)}  "
        f"Rafts: {len(rafts)}",
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
        "lyr_found    = model.layers.add('Foundations')",
        "lyr_pit      = model.layers.add('LiftPits')",
        "lyr_pile     = model.layers.add('Piles')",
        "lyr_gbeam    = model.layers.add('GroundBeams')",
        "lyr_raft     = model.layers.add('Rafts')",
        "lyr_col      = model.layers.add('Columns')",
        "lyr_beam     = model.layers.add('Beams-RC')",
        "lyr_beam_pt  = model.layers.add('Beams-PT')",
        "lyr_slab     = model.layers.add('Slabs-RC')",
        "lyr_slab_pt  = model.layers.add('Slabs-PT')",
        "",
    ]

    # ── Materials ─────────────────────────────────────────────────────────────
    lines += [
        "mat_conc  = model.materials.add('Concrete-RC');  mat_conc.color  = Sketchup::Color.new(190,185,175)",
        "mat_pt    = model.materials.add('Concrete-PT');  mat_pt.color    = Sketchup::Color.new(100,160,220)",
        "mat_pile  = model.materials.add('Pile');         mat_pile.color  = Sketchup::Color.new(160,155,145)",
        "mat_steel = model.materials.add('Steel');        mat_steel.color = Sketchup::Color.new(90,130,175)",
        "mat_slab  = model.materials.add('Slab-RC');      mat_slab.color  = Sketchup::Color.new(215,205,190)",
        "mat_slab_pt = model.materials.add('Slab-PT');    mat_slab_pt.color = Sketchup::Color.new(160,200,240)",
        "mat_pit     = model.materials.add('LiftPit');    mat_pit.color     = Sketchup::Color.new(80,80,80)",
        "",
    ]

    # ── Helper functions ──────────────────────────────────────────────────────
    lines += [
        "# Axis-aligned box: origin corner (x,y,z), dimensions (w,d,h)",
        "def ap_box(ents, x, y, z, w, d, h, lyr, mat)",
        "  return nil if w.abs < 1e-6 || d.abs < 1e-6 || h.abs < 1e-6",
        "  g = ents.add_group; g.layer = lyr",
        "  f = g.entities.add_face(",
        "    [x,y,z],[x+w,y,z],[x+w,y+d,z],[x,y+d,z])",
        "  f.pushpull(h)",
        "  g.entities.each{|e| e.material=mat if e.is_a?(Sketchup::Face)}",
        "  g",
        "end",
        "",
        "# Beam between two XY points at elevation z (any plan angle)",
        "def ap_beam(ents, x1,y1,z1, x2,y2,z2, bw,bh, lyr, mat)",
        "  dx=x2-x1; dy=y2-y1",
        "  len=Math.sqrt(dx*dx+dy*dy); return if len<1e-6",
        "  return if bw.abs < 1e-6 || bh.abs < 1e-6",
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
        "# Circular cylinder: centre (cx,cy), bottom at z_bot, extruding upward by height",
        "def ap_cylinder(ents, cx,cy,z_bot, radius, height, lyr, mat)",
        "  return nil if radius < 1e-6 || height.abs < 1e-6",
        "  g=ents.add_group; g.layer=lyr",
        "  ge=g.entities",
        "  n=16",
        "  pts=(0...n).map{|i| a=2*Math::PI*i/n",
        "    Geom::Point3d.new(cx+radius*Math.cos(a),cy+radius*Math.sin(a),z_bot)}",
        "  f=ge.add_face(pts)",
        "  f.pushpull(height)",
        "  g.entities.each{|e| e.material=mat if e.is_a?(Sketchup::Face)}",
        "  g",
        "end",
        "",
    ]

    # ── Foundations ───────────────────────────────────────────────────────────
    lines += ["# ===== FOUNDATIONS ====="]

    # Ground floor elevation (top of pile cap / top of pad footing)
    z_ground = _in(bottom_lv.get("elevation_mm", 0))
    # Pile caps sit slightly below ground slab (300 mm setdown)
    CAP_SETDOWN_MM = 300

    for f in foundations:
        ftype      = f.get("ftype", "pile")   # pile_cap | pad_footing | strip_footing | pile
        x_c        = _in(f.get("x_mm", 0))
        y_c        = _in(f.get("y_mm", 0))
        cap_w      = _in(f.get("width_mm",  1500))
        cap_d      = _in(f.get("depth_mm",  1500))
        cap_h      = _in(f.get("height_mm", 700))
        pile_dia   = f.get("pile_dia_mm", 0)
        pile_len   = f.get("pile_len_mm", 0)
        pile_count = f.get("pile_count",  1)
        gref       = f.get("grid_ref", "")
        fid        = f.get("id", "")

        # For pile caps: estimate pile diameter from cap size when not in schedule.
        # Cap is typically 2×pile_dia; CFA standard sizes: 450/500/600/650/750mm.
        if ftype in ("pile_cap", "pile") and pile_dia == 0 and cap_w > 0:
            _est = (f.get("width_mm", 0) + f.get("depth_mm", 0)) / 2 * 0.45
            _standards = [450, 500, 600, 650, 750, 900, 1050, 1200]
            pile_dia = min(_standards, key=lambda s: abs(s - _est)) if _est > 200 else 600
            pile_len = f.get("pile_len_mm", 0) or 8000  # 8m default if missing
            warnings.append(
                f"{fid}: pile_dia estimated {pile_dia}mm from cap "
                f"(refer to geotech report for actual spec)"
            )

        # Guard: use sensible fallbacks if dimensions are zero (failed extraction)
        if cap_w < 1e-6:
            cap_w = _in(1500)
            warnings.append(f"{fid}: width_mm=0, defaulted to 1500mm")
        if cap_d < 1e-6:
            cap_d = _in(1500)
            warnings.append(f"{fid}: depth_mm=0, defaulted to 1500mm")
        if cap_h < 1e-6:
            cap_h = _in(700)
            warnings.append(f"{fid}: height_mm=0, defaulted to 700mm")

        # Z: top of cap/footing = ground level - setdown
        z_cap_top = z_ground - _in(CAP_SETDOWN_MM)
        z_cap_bot = z_cap_top - cap_h

        fc_fdn = f.get("fc_mpa", 50)
        fdn_label = f.get("label", fid)

        if ftype in ("pile_cap", "pile"):
            # Pile cap box
            lines.append(
                f"_g = ap_box(ents, {x_c-cap_w/2:.6f},{y_c-cap_d/2:.6f},{z_cap_bot:.6f},"
                f"{cap_w:.6f},{cap_d:.6f},{cap_h:.6f},lyr_found,mat_conc)"
                f"  # {fid} cap @ {gref}"
            )
            lines += _bim_attrs("_g", {
                "element_type": "PileCap",
                "mark": fdn_label,
                "fc_mpa": fc_fdn,
                "width_mm": round(cap_w / MM_TO_IN),
                "depth_mm": round(cap_d / MM_TO_IN),
                "height_mm": round(cap_h / MM_TO_IN),
            })
            counts["foundations"] += 1

            # Piles below cap
            if pile_dia > 0 and pile_len > 0:
                pile_r   = _in(pile_dia / 2)
                pile_h_i = _in(pile_len)
                z_pile_top = z_cap_bot
                z_pile_bot_coord = z_pile_top - pile_h_i

                offsets = _pile_offsets(pile_count, pile_dia)
                for (odx_mm, ody_mm) in offsets:
                    px = x_c + _in(odx_mm)
                    py = y_c + _in(ody_mm)
                    lines.append(
                        f"ap_cylinder(ents, {px:.6f},{py:.6f},{z_pile_bot_coord:.6f},"
                        f"{pile_r:.6f},{pile_h_i:.6f},lyr_pile,mat_pile)"
                        f"  # pile Ø{pile_dia:.0f} @ {gref}"
                    )

        elif ftype == "pad_footing":
            # Simple rectangular pad footing
            lines.append(
                f"ap_box(ents, {x_c-cap_w/2:.6f},{y_c-cap_d/2:.6f},{z_cap_bot:.6f},"
                f"{cap_w:.6f},{cap_d:.6f},{cap_h:.6f},lyr_found,mat_conc)"
                f"  # {fid} @ {gref}"
            )
            counts["foundations"] += 1

        elif ftype == "strip_footing":
            # Strip footing is handled as a ground beam spanning two columns;
            # emit as a rectangular beam at foundation level.
            # (from/to coords may be set if extracted — otherwise use cap width as a square pad)
            to_x = _in(f.get("to_x_mm", f.get("x_mm", 0)))
            to_y = _in(f.get("to_y_mm", f.get("y_mm", 0)))
            if abs(to_x - x_c) > 1e-4 or abs(to_y - y_c) > 1e-4:
                lines.append(
                    f"ap_beam(ents, {x_c:.6f},{y_c:.6f},{z_cap_top:.6f},"
                    f"{to_x:.6f},{to_y:.6f},{z_cap_top:.6f},"
                    f"{cap_w:.6f},{cap_h:.6f},lyr_found,mat_conc)"
                    f"  # {fid} strip @ {gref}"
                )
            else:
                lines.append(
                    f"ap_box(ents, {x_c-cap_w/2:.6f},{y_c-cap_d/2:.6f},{z_cap_bot:.6f},"
                    f"{cap_w:.6f},{cap_d:.6f},{cap_h:.6f},lyr_found,mat_conc)"
                    f"  # {fid} strip-pad @ {gref}"
                )
            counts["foundations"] += 1

        else:
            # Legacy / unknown type: cylindrical pile (backwards-compatible)
            r = _in(f.get("width_mm", 750) / 2)
            depth_i = _in(f.get("depth_mm", 500))
            lines.append(
                f"ap_cylinder(ents, {x_c:.6f},{y_c:.6f},{z_cap_bot:.6f},"
                f"{r:.6f},{depth_i:.6f},lyr_found,mat_conc)"
                f"  # {fid} @ {gref}"
            )
            counts["foundations"] += 1

    lines.append("")

    # ── Ground beams ───────────────────────────────────────────────────────────
    lines += ["# ===== GROUND BEAMS ====="]
    z_gbeam = z_ground - _in(CAP_SETDOWN_MM)   # top of ground beams = top of pile caps

    for gb in ground_beams:
        x1 = _in(gb.get("from_x_mm", 0))
        y1 = _in(gb.get("from_y_mm", 0))
        x2 = _in(gb.get("to_x_mm",   0))
        y2 = _in(gb.get("to_y_mm",   0))
        bw = _in(gb.get("width_mm",  300))
        bh = _in(gb.get("height_mm", 600))
        span = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if span < 0.01:
            skipped.append(f"{gb.get('id','?')}: zero span ground beam")
            continue
        lines.append(
            f"ap_beam(ents, {x1:.6f},{y1:.6f},{z_gbeam:.6f},"
            f"{x2:.6f},{y2:.6f},{z_gbeam:.6f},"
            f"{bw:.6f},{bh:.6f},lyr_gbeam,mat_conc)"
            f"  # {gb.get('id','')} {gb.get('section_label','')} @ {gb.get('grid_ref','')}"
        )
    lines.append("")

    # ── Raft foundations ───────────────────────────────────────────────────────
    lines += ["# ===== RAFT FOUNDATIONS ====="]
    for rf in rafts:
        x1   = _in(rf.get("x_from_mm", rf.get("x_mm", 0)))
        y1   = _in(rf.get("y_from_mm", rf.get("y_mm", 0)))
        x2   = _in(rf.get("x_to_mm",   rf.get("x_mm", 0)))
        y2   = _in(rf.get("y_to_mm",   rf.get("y_mm", 0)))
        t_rf = _in(rf.get("height_mm", 1000))
        sw   = abs(x2 - x1)
        sd   = abs(y2 - y1)
        if sw < 0.01 or sd < 0.01:
            skipped.append(f"{rf.get('id','?')}: zero area raft")
            continue
        z_rf_top = z_ground - _in(CAP_SETDOWN_MM)
        z_rf_bot = z_rf_top - t_rf
        lines.append(
            f"ap_box(ents, {min(x1,x2):.6f},{min(y1,y2):.6f},{z_rf_bot:.6f},"
            f"{sw:.6f},{sd:.6f},{t_rf:.6f},lyr_raft,mat_conc)"
            f"  # {rf.get('id','')} {rf.get('label','')} raft"
        )
    lines.append("")

    # ── Lift pits ──────────────────────────────────────────────────────────────
    lines += ["# ===== LIFT PITS ====="]
    z_pit_ref = z_ground - _in(CAP_SETDOWN_MM)  # top of pit = foundation level
    for lp in lift_pits:
        lp_x = _in(lp.get("x_mm", 0))
        lp_y = _in(lp.get("y_mm", 0))
        lp_w = _in(lp.get("width_mm", 1500))
        lp_d = _in(lp.get("width_mm", 1500))   # typically square
        lp_h = _in(lp.get("depth_mm", 1200))
        if lp_w < 1e-6 or lp_h < 1e-6:
            continue
        z_pit_bot = z_pit_ref - lp_h
        lid = lp.get("id", "")
        lines.append(
            f"_g = ap_box(ents, {lp_x-lp_w/2:.6f},{lp_y-lp_d/2:.6f},{z_pit_bot:.6f},"
            f"{lp_w:.6f},{lp_d:.6f},{lp_h:.6f},lyr_pit,mat_pit)"
            f"  # {lid} lift pit"
        )
        lines += _bim_attrs("_g", {
            "element_type": "LiftPit",
            "mark": lid,
            "depth_mm": lp.get("depth_mm", 1200),
            "note": lp.get("note", ""),
        })
    if has_planter:
        lines.append("# NOTE: Planter/landscape slabs detected (min 300mm thick) — verify slab thickness")
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

        if w < 1e-6: w = _in(150)
        if d < 1e-6: d = _in(150)
        mat = "mat_steel" if col.get("material", "steel") == "steel" else "mat_conc"
        cid = col.get('id', '')
        lines.append(
            f"_g = ap_box(ents, {x-w/2:.6f},{y-d/2:.6f},{z_b:.6f},"
            f"{w:.6f},{d:.6f},{h:.6f},lyr_col,{mat})"
            f"  # {cid} @ {col.get('grid_ref','')}"
        )
        lines += _bim_attrs("_g", {
            "element_type": "Column",
            "mark": col.get("column_mark", cid),
            "fc_mpa": col.get("fc_mpa", 50),
            "material": col.get("material", "steel"),
            "width_mm": round(w / MM_TO_IN),
            "depth_mm": round(d / MM_TO_IN),
        })
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

        if bw < 1e-6: bw = _in(100)
        if bh < 1e-6: bh = _in(200)
        is_pt   = beam.get("is_pt", False)
        fc_mpa  = beam.get("fc_mpa", 40)
        mat_var = "mat_pt" if is_pt else ("mat_steel" if beam.get("material", "steel") == "steel" else "mat_conc")
        lyr_var = "lyr_beam_pt" if is_pt else "lyr_beam"
        bid = beam.get('id', '')
        lines.append(
            f"_g = ap_beam(ents, {x1:.6f},{y1:.6f},{z:.6f},{x2:.6f},{y2:.6f},{z:.6f},"
            f"{bw:.6f},{bh:.6f},{lyr_var},{mat_var})"
            f"  # {bid} {beam.get('section_label','')} @ {beam.get('grid_ref','')}"
        )
        lines += _bim_attrs("_g", {
            "element_type": "Beam",
            "mark": bid,
            "is_pt": is_pt,
            "fc_mpa": fc_mpa,
            "material": beam.get("material", "steel"),
            "width_mm": round(bw / MM_TO_IN),
            "depth_mm": round(bh / MM_TO_IN),
        })
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

        is_pt_slab = slab.get("is_pt", False)
        fc_slab    = slab.get("fc_mpa", 40)
        mat_slab_var = "mat_slab_pt" if is_pt_slab else "mat_slab"
        lyr_slab_var = "lyr_slab_pt" if is_pt_slab else "lyr_slab"
        sid = slab.get('id', '')
        lines.append(
            f"_g = ap_box(ents, {min(x1,x2):.6f},{min(y1,y2):.6f},{z:.6f},"
            f"{sw:.6f},{sd:.6f},{t:.6f},{lyr_slab_var},{mat_slab_var})"
            f"  # {sid} @ {slab.get('grid_ref','')}"
        )
        lines += _bim_attrs("_g", {
            "element_type": "Slab",
            "mark": sid,
            "is_pt": is_pt_slab,
            "fc_mpa": fc_slab,
            "thickness_mm": round(t / MM_TO_IN),
        })
        counts["slabs"] += 1
    lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "model.commit_operation",
        "puts 'auto_pipeline vector build complete.'",
        "puts \"Foundations:#{" + str(counts["foundations"]) +
        "} GroundBeams:#{" + str(len(ground_beams)) +
        "} Rafts:#{" + str(len(rafts)) +
        "} Columns:#{" + str(counts["columns"]) +
        "} Beams:#{" + str(counts["beams"]) +
        "} Slabs:#{" + str(counts["slabs"]) + "}\"",
    ]

    ruby_code = "\n".join(lines)
    (job_dir / "sketchup_model.rb").write_text(ruby_code, encoding="utf-8")

    report = {
        "stage": 5,
        "stage_name": "Ruby Generator",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elements_generated": {**counts, "ground_beams": len(ground_beams), "rafts": len(rafts), "lift_pits": len(lift_pits)},
        "ruby_line_count": len(lines),
        "warnings": warnings,
        "skipped": skipped,
    }
    (job_dir / "stage5_generation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
