"""Structural PDF → SketchUp 3D — Streamlit App (Vector Pipeline)

Upload any structural PDF, get a .rb file ready to run in SketchUp.
Uses direct PDF vector data extraction — no AI vision required.
Each stage writes its own JSON proof file to data/jobs/{job_id}/.
"""

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")   # absolute path — works regardless of CWD

import streamlit as st

JOBS_DIR = Path("data/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structural PDF → SketchUp 3D",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️  Structural PDF → SketchUp 3D")
st.caption(
    "Upload any CAD-generated structural PDF. "
    "Reads vector geometry directly — no AI required. "
    "Generates SketchUp Ruby script with foundations, columns, beams, slabs."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Settings")
    dpi_input = st.slider("Preview DPI (for page thumbnails)", 72, 200, 120, 24)
    st.divider()
    st.markdown("**Per-stage output files:**")
    st.code(
        "stage2_classification.json\n"
        "stage3_vector_extractions.json\n"
        "stage4_unified_model.json\n"
        "stage5_generation_report.json\n"
        "sketchup_model.rb",
        language=None,
    )
    st.caption("All files saved in `data/jobs/{job_id}/`")
    st.divider()
    st.markdown("**How to use the .rb file:**")
    st.markdown(
        "In SketchUp: open Ruby Console `Window → Ruby Console`, "
        "then type:\n```\nload 'C:/path/to/sketchup_model.rb'\n```"
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload structural PDF (CAD-generated)",
    type=["pdf"],
    help="Must be a vector PDF from structural engineering software (not scanned).",
)

if not uploaded:
    st.info("👆  Upload a structural PDF to begin.")
    st.stop()

st.divider()
run_btn = st.button("▶  Run Vector Pipeline", type="primary")

if not run_btn:
    st.stop()

# ── Import pipeline modules here (NOT at top level) ───────────────────────────
# Streamlit hot-reload clears sys.modules mid-load when imports are at module
# level — moving them inside the run handler avoids KeyError: 'pipeline.*'.
from pipeline.model_builder import build_model
from pipeline.ruby_generator import generate_ruby

# ── Job setup ─────────────────────────────────────────────────────────────────
job_id = str(uuid.uuid4())[:8]
job_dir = JOBS_DIR / job_id
job_dir.mkdir(parents=True, exist_ok=True)

pdf_path = job_dir / uploaded.name
pdf_path.write_bytes(uploaded.read())

st.success(f"Job `{job_id}` — {pdf_path.stat().st_size // 1024} KB uploaded")

# ── Per-job log file ──────────────────────────────────────────────────────────
import logging as _logging
_log_path = job_dir / "pipeline.log"
_file_handler = _logging.FileHandler(_log_path, encoding="utf-8")
_file_handler.setLevel(_logging.DEBUG)
_file_handler.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_root_logger = _logging.getLogger()
_root_logger.addHandler(_file_handler)
_root_logger.setLevel(_logging.DEBUG)

# ── Pipeline ──────────────────────────────────────────────────────────────────
stage_status = {
    "classify": st.empty(),
    "grid":     st.empty(),
    "extract":  st.empty(),
    "levels":   st.empty(),
    "model":    st.empty(),
}

prog_bars = {
    "classify": st.progress(0, text="Classifying pages…"),
    "grid":     st.progress(0, text="Extracting grid…"),
    "extract":  st.progress(0, text="Detecting structural elements…"),
    "levels":   st.progress(0, text="Extracting floor levels…"),
    "model":    st.progress(0, text="Assembling 3D model…"),
}

stage_labels = {
    "classify": "📐  Stage 1: Page Classification",
    "grid":     "📏  Stage 2: Grid & Scale Extraction",
    "extract":  "🔍  Stage 3: Element Detection (vector)",
    "levels":   "📊  Stage 4: Floor Level Heights",
    "model":    "🏗️  Stage 5: Assemble 3D Model",
}

def progress_cb(stage: str, pct: float):
    bar = prog_bars.get(stage)
    if bar:
        label = stage_labels.get(stage, stage)
        if pct >= 1.0:
            bar.progress(1.0, text=f"✅  {label}")
        else:
            bar.progress(max(0.05, pct), text=f"{label}…  {int(pct*100)}%")

try:
    unified = build_model(str(pdf_path), str(job_dir), progress_cb=progress_cb)
except Exception as e:
    st.error(f"Pipeline failed: {e}")
    st.exception(e)
    st.stop()

# ── Stage 5: Ruby generation ──────────────────────────────────────────────────
st.subheader("⚙️  Generating Ruby Script")
prog_rb = st.progress(0, text="Writing .rb file…")

try:
    report = generate_ruby(unified, str(job_dir))
    prog_rb.progress(1.0, text="✅  sketchup_model.rb ready")
except Exception as e:
    st.error(f"Ruby generation failed: {e}")
    st.stop()

# ── Stage JSON viewers ────────────────────────────────────────────────────────
st.divider()

col_left, col_right = st.columns(2)

with col_left:
    with st.expander("📄  stage2_classification.json"):
        cls_path = job_dir / "stage2_classification.json"
        if cls_path.exists():
            cls_data = json.loads(cls_path.read_text(encoding="utf-8"))
            st.json({
                "type_summary": cls_data.get("type_summary", {}),
                "pages": [{"page": p["page_num"], "type": p["drawing_type"],
                           "scale": p.get("scale_ratio"), "title": (p.get("drawing_title") or "")[:40]}
                          for p in cls_data.get("pages", [])],
            })

    with st.expander("📄  stage3_vector_extractions.json"):
        ext_path = job_dir / "stage3_vector_extractions.json"
        if ext_path.exists():
            ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
            g = ext_data.get("grid", {})
            st.json({
                "grid": {
                    "x_axes": len(g.get("x_axes", [])),
                    "y_axes": len(g.get("y_axes", [])),
                    "scale":  f"1:{g.get('scale', 100)}",
                    "x_labels": [a["label"] for a in g.get("x_axes", [])],
                    "y_labels": [a["label"] for a in g.get("y_axes", [])],
                },
                "pages": ext_data.get("pages", []),
            })

with col_right:
    with st.expander("📄  stage4_unified_model.json"):
        ai = unified.get("ai_used", {})
        if any(ai.values()):
            st.info("🤖 Vision AI used: " + ", ".join(k for k, v in ai.items() if v))
        c = unified["summary_counts"]
        g = unified["grid_system"]
        st.json({
            "summary": c,
            "grid_mm": {
                "x": {a["label"]: round(a["real_mm"]) for a in g["x_axes"]},
                "y": {a["label"]: round(a["real_mm"]) for a in g["y_axes"]},
            },
            "levels": [(l["name"], l["elevation_mm"]) for l in unified["levels"]],
        })

    with st.expander("📄  stage5_generation_report.json"):
        st.json(report)

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
st.success("✅  Pipeline complete!")

# Close and remove the per-job log handler
_root_logger.removeHandler(_file_handler)
_file_handler.close()

rb_bytes = (job_dir / "sketchup_model.rb").read_bytes()
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️  Download sketchup_model.rb",
        data=rb_bytes,
        file_name="sketchup_model.rb",
        mime="text/plain",
        type="primary",
    )
with col_dl2:
    if _log_path.exists():
        st.download_button(
            "⬇️  Download pipeline.log",
            data=_log_path.read_bytes(),
            file_name=f"pipeline_{job_id}.log",
            mime="text/plain",
        )

eg = report["elements_generated"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Foundations", eg.get("foundations", 0))
c2.metric("Rafts",       eg.get("rafts", 0))
c3.metric("Gnd Beams",   eg.get("ground_beams", 0))
c4.metric("Columns",     eg.get("columns", 0))
c5.metric("Beams",       eg.get("beams", 0))
c6.metric("Slabs",       eg.get("slabs", 0))

# Foundation type breakdown
fdn_schedule = unified.get("foundation_schedule", {})
if fdn_schedule:
    with st.expander("🔩  Foundation schedule extracted from PDF"):
        rows = []
        for lbl, spec in sorted(fdn_schedule.items()):
            rows.append({
                "Type": lbl,
                "Category": spec.get("ftype", "?"),
                "Dia (mm)": int(spec.get("pile_dia_mm", 0)) or "—",
                "Socket (m)": round(spec.get("pile_len_mm", 0) / 1000, 1) or "—",
                "Cap W×D (mm)": (
                    f"{int(spec.get('width_mm',0))}×{int(spec.get('depth_mm',0))}"
                    if spec.get("width_mm") else "—"
                ),
                "Cap H (mm)": int(spec.get("height_mm", 0)) or "—",
            })
        st.dataframe(rows, use_container_width=True)

st.caption(
    f"Job: `data/jobs/{job_id}/`  |  "
    f"Grid: {len(unified['grid_system']['x_axes'])}×{len(unified['grid_system']['y_axes'])}  |  "
    f"Scale: 1:{unified['grid_system']['scale']}  |  "
    f"Ruby lines: {report['ruby_line_count']}  |  "
    f"Warnings: {len(report['warnings'])}"
)

if report["warnings"]:
    with st.expander(f"⚠️  {len(report['warnings'])} warnings"):
        for w in report["warnings"]:
            st.warning(w)
