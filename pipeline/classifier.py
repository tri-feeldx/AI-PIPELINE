"""Stage 2: Classify each drawing page using Gemini Vision via Vertex AI (google-genai SDK).

Uses ThreadPoolExecutor for parallel processing (default 6 workers).
Output: data/jobs/{job_id}/stage2_classification.json
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types


CLASSIFY_PROMPT = """You are an expert structural engineer reviewing engineering drawings.
Analyze this drawing image and classify it.

Output ONLY valid JSON (no markdown, no explanation):
{
  "drawing_type": "<one of: floor_plan, roof_plan, foundation_plan, elevation, section, detail, schedule, cover, notes, unknown>",
  "level_name": "<floor/level name if visible, e.g. 'Ground Floor', 'Level 1', 'Roof', null if not clear>",
  "scale_ratio": <scale denominator as integer, e.g. 100 for 1:100, null if not found>,
  "material_type": "<one of: steel, concrete, mixed, unknown>",
  "drawing_title": "<title text from title block if visible, else null>",
  "has_grid_lines": <true or false>,
  "confidence": <0.0 to 1.0, how confident you are in this classification>
}"""


def _classify_one(page_info: dict, client: genai.Client, model_name: str) -> dict:
    """Classify a single page. Thread-safe — client is shared read-only."""
    image_bytes = Path(page_info["file_path"]).read_bytes()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                CLASSIFY_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw = response.text
        result = json.loads(raw)
    except json.JSONDecodeError:
        raw = getattr(response, "text", "")
        result = {
            "drawing_type": "unknown",
            "level_name": None,
            "scale_ratio": None,
            "material_type": "unknown",
            "drawing_title": None,
            "has_grid_lines": False,
            "confidence": 0.0,
        }
    except Exception as e:
        raw = ""
        result = {
            "drawing_type": "unknown",
            "level_name": None,
            "scale_ratio": None,
            "material_type": "unknown",
            "drawing_title": None,
            "has_grid_lines": False,
            "confidence": 0.0,
            "error": str(e),
        }

    result["raw_gemini_response"] = raw
    return {
        "page_num": page_info["page_num"],
        "file_path": page_info["file_path"],
        **result,
    }


def classify_all_pages(
    manifest: dict,
    job_dir: str,
    model_name: str,
    project: str,
    location: str,
    workers: int = 6,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Classify all pages in parallel.

    workers: number of concurrent Gemini API calls (default 6).
    Returns stage2_classification dict (also written to disk).
    """
    client = genai.Client(vertexai=True, project=project, location=location)
    job_dir = Path(job_dir)
    pages = manifest["pages"]
    total = len(pages)

    # completed_count is shared across threads — use a lock
    completed_count = 0
    lock = threading.Lock()

    # results dict keyed by page_num to preserve order
    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_classify_one, page_info, client, model_name): page_info["page_num"]
            for page_info in pages
        }
        for future in as_completed(futures):
            page_num = futures[future]
            classification = future.result()
            results[page_num] = classification

            with lock:
                completed_count += 1
                done = completed_count

            if progress_cb:
                progress_cb(done, total)

    # Sort back into page order
    classifications = [results[p["page_num"]] for p in pages]

    result = {
        "stage": 2,
        "stage_name": "Page Classifier",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_name,
        "workers": workers,
        "total_pages": total,
        "type_summary": {
            t: sum(1 for c in classifications if c.get("drawing_type") == t)
            for t in {c.get("drawing_type", "unknown") for c in classifications}
        },
        "pages": classifications,
    }

    (job_dir / "stage2_classification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
