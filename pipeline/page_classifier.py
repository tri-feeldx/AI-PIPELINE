"""Page classifier — text-based, no AI.

Strategy (priority order):
1. Drawing number pattern (most reliable for CAD-generated structural PDFs):
   e.g. TTW-00-DR-ST-21041 = Level 04 plan, TTW-RF-DR-ST-70111 = roof plan
2. Drawing title text in titleblock area
3. Keywords in main drawing area (not titleblock)
"""

import re
from pathlib import Path

import fitz


# ── Scale detection ─────────────────────────────────────────────────────────

_SCALE_RE = re.compile(r"SCALE\s*[:\s]+\s*1\s*[:/\s]+\s*(\d{2,4})\b", re.I)


def _extract_spans(page: fitz.Page) -> list[tuple[str, float, float, float]]:
    """Return [(text, x, y, size)] for all text spans on the page."""
    items = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    items.append((t, span["origin"][0], span["origin"][1], span["size"]))
    return items


def _scale_from_text(text: str) -> int | None:
    m = _SCALE_RE.search(text)
    return int(m.group(1)) if m else None


# ── Drawing-number based classification ─────────────────────────────────────
# Drawing numbers encode the drawing type in their suffix/prefix.
# e.g. drawing no ending in:
#   70xxx = steelwork plan  →  floor_plan / roof_plan
#   21xxx = level outline plan → floor_plan
#   10001 = footing details → foundation_plan
#   12xxx, 13xxx = insitu wall details → section/detail
#   00001 = notes sheet
#   ST-700xx = member schedule → schedule
#   91xxx = loading diagram → detail

_DWG_NUM_RE = re.compile(r"[A-Z]{2,6}-[A-Z]{2}-[A-Z]{2}-[A-Z]{2,4}-(\d{5})\b")

_DWG_TYPE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^0000[01]$"),        "cover"),
    (re.compile(r"^0000[2-9]$"),       "notes"),
    (re.compile(r"^1[01]\d{3}$"),      "foundation_plan"),  # 10xxx-11xxx
    (re.compile(r"^1[2-5]\d{3}$"),     "detail"),           # 12xxx-15xxx
    (re.compile(r"^2\d{4}$"),          "floor_plan"),        # 2xxxx = level plans
    (re.compile(r"^3\d{4}$"),          "floor_plan"),        # 3xxxx = seismic plans
    (re.compile(r"^70[01]\d{2}$"),     "roof_plan"),        # 700xx-701xx = steelwork/roof
    (re.compile(r"^70[2-9]\d{2}$"),    "elevation"),        # 702xx+ = steelwork elevs
    (re.compile(r"^7[1-9]\d{3}$"),     "detail"),           # 71xxx+ = steelwork details
    (re.compile(r"^9[01]\d{3}$"),      "detail"),           # loading diagrams
    (re.compile(r"^7000\d$"),          "schedule"),         # 70000x = schedules
]


def _type_from_dwg_number(num_str: str) -> str | None:
    for pattern, dtype in _DWG_TYPE_MAP:
        if pattern.match(num_str):
            return dtype
    return None


# ── Keyword-based classification ─────────────────────────────────────────────

_KW: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bFOOTING\s+PLAN\b|\bFOUNDATION\s+PLAN\b|\bPILE\s+LAYOUT\b", re.I), "foundation_plan"),
    (re.compile(r"\bROOF\s+(STEEL|FRAMING|MARKING|PLAN)\b", re.I),                      "roof_plan"),
    (re.compile(r"\bLEVEL\s*\d+\s+(?:OUTLINE|FLOOR|MARKING|FRAMING|SLAB)\b", re.I),    "floor_plan"),
    (re.compile(r"\bSTEELWORK\s+MARKING\s+PLAN\b|\bSTEEL.*MARKING\s+PLAN\b", re.I),    "floor_plan"),
    (re.compile(r"\bMEMBER\s+SCHEDULE\b|\bSTEELWORK\s+.*SCHEDULE\b", re.I),            "schedule"),
    (re.compile(r"\bCOLUMN\s+SCHEDULE\b|\bBEAM\s+SCHEDULE\b", re.I),                  "schedule"),
    (re.compile(r"\bSTEELWORK\s+ELEVATION\b|\bSTRUCTURAL\s+ELEVATION\b", re.I),       "elevation"),
    (re.compile(r"\bSECTION\s+\w+|\bTYPICAL\s+SECTION\b", re.I),                       "section"),
    (re.compile(r"\bLOADING\s+DIAGRAM\b", re.I),                                        "detail"),
    (re.compile(r"\bDETAIL\b|\bTYPICAL\b|\bCONNECTION\b", re.I),                       "detail"),
    (re.compile(r"\bNOTES\s+SHEET\b|\bGENERAL\s+NOTES\b|\bSPECIFICATION\b", re.I),    "notes"),
    (re.compile(r"\bCOVER\s+SHEET\b|\bDRAWING\s+LIST\b", re.I),                        "cover"),
]


def classify_page(page: fitz.Page, page_num: int) -> dict:
    """Classify a page using drawing number suffix, then keywords."""
    spans = _extract_spans(page)
    w, h = page.rect.width, page.rect.height

    # Scale extraction (from full text)
    all_text = " ".join(t for t, *_ in spans)
    scale_ratio = _scale_from_text(all_text)

    # ── 1. Drawing number from titleblock (bottom-left: x<30%, y>75% of page) ─
    # e.g. "70201" at (326,1352) on a 2384x1684 page
    drawing_type = "unknown"
    drawing_title = None

    titleblock_nums = [
        t for t, x, y, sz in spans
        if y > h * 0.75 and x < w * 0.35 and re.fullmatch(r"\d{5}", t)
    ]
    if titleblock_nums:
        dtype = _type_from_dwg_number(titleblock_nums[0])
        if dtype:
            drawing_type = dtype

    # ── 2. Full drawing number (e.g. TTW-RF-DR-ST-70201) ───────────────────
    if drawing_type == "unknown":
        m = _DWG_NUM_RE.search(all_text)
        if m:
            dtype = _type_from_dwg_number(m.group(1))
            if dtype:
                drawing_type = dtype

    # ── 3. Main drawing area keywords (exclude right-margin notes x>80%) ────
    if drawing_type == "unknown":
        main_area_text = " ".join(
            t for t, x, y, sz in spans
            if x < w * 0.80  # exclude right margin (tender notes etc.)
        )
        # Remove titleblock boilerplate
        clean = re.sub(
            r"\b(?:conjunction|architectural|contract|documents|notify"
            r"|specifications|requirements|copyright|revision|drawn|checked"
            r"|approved|project|drawing|tender\s+notes)\b",
            " ", main_area_text, flags=re.I
        )
        for pattern, dtype in _KW:
            if pattern.search(clean):
                drawing_type = dtype
                break

    # ── Drawing title: look for medium text in bottom-center titleblock ──────
    # Bottom 25% of page, middle horizontal band (x: 30-80%)
    title_candidates = sorted(
        [
            (t, sz) for t, x, y, sz in spans
            if y > h * 0.75 and w * 0.3 < x < w * 0.82 and sz > 9
            and len(t) > 6
            and not t.replace(".", "").replace(":", "").replace(" ", "").isdigit()
        ],
        key=lambda ts: -ts[1],
    )
    if title_candidates:
        drawing_title = title_candidates[0][0]

    # Level name
    level_name = None
    search_for_level = drawing_title or ""
    m2 = re.search(r"(LEVEL\s*0*(\d+)|GROUND\s*FLOOR|ROOF|MEZZANINE)", search_for_level, re.I)
    if not m2:
        m2 = re.search(r"(LEVEL\s*0*(\d+)|GROUND\s*FLOOR|ROOF|MEZZANINE)", all_text, re.I)
    if m2:
        level_name = m2.group(0).upper().strip()

    return {
        "page_num":      page_num,
        "drawing_type":  drawing_type,
        "drawing_title": drawing_title,
        "level_name":    level_name,
        "scale_ratio":   scale_ratio,
        "has_grid_lines": _has_grid_labels(spans, w, h),
        "confidence":    1.0,
    }


def _has_grid_labels(spans, w, h) -> bool:
    labels = [
        t for t, x, y, sz in spans
        if sz > 12 and len(t) == 1 and (t.isdigit() or t.isalpha())
        and (y < h * 0.12 or x > w * 0.7)
    ]
    return len(labels) >= 2


def classify_all_pages(pdf_path: str) -> list[dict]:
    """Classify all pages of a PDF. Returns list of classification dicts."""
    doc = fitz.open(pdf_path)
    results = [classify_page(doc[i], i + 1) for i in range(doc.page_count)]
    doc.close()
    return results
