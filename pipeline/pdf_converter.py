"""Stage 1: Convert each PDF page to a PNG image.

Output: data/jobs/{job_id}/pages/page_01.png ... page_N.png
        data/jobs/{job_id}/stage1_manifest.json
"""

import json
import os
import time
from pathlib import Path

import fitz  # PyMuPDF


def convert_pdf_to_images(pdf_path: str, job_dir: str, dpi: int = 150) -> dict:
    """Convert all pages of a PDF to PNG images.

    Returns the stage1_manifest dict (also written to disk).
    """
    pdf_path = Path(pdf_path)
    job_dir = Path(job_dir)
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    for i in range(doc.page_count):
        page = doc[i]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = pages_dir / f"page_{i + 1:02d}.png"
        pix.save(str(out_path))

        pages.append({
            "page_num": i + 1,
            "file_path": str(out_path),
            "width_px": pix.width,
            "height_px": pix.height,
            "file_size_bytes": out_path.stat().st_size,
        })

    doc.close()

    manifest = {
        "stage": 1,
        "stage_name": "PDF → Images",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_pdf": str(pdf_path),
        "dpi": dpi,
        "page_count": len(pages),
        "pages": pages,
    }

    manifest_path = job_dir / "stage1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
