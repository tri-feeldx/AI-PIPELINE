"""Schedule extractor — reads structural member schedules from PDF tables.

Uses PyMuPDF find_tables() (vector-native, zero cost) as primary method.
Falls back to Vision AI only when no tables are found on a page.

Extracts:
  column_schedule  : {mark → {width_mm, depth_mm, from_level, to_level, material, fc_mpa}}
  beam_schedule    : {mark → {width_mm, depth_mm, material}}

Column schedule format (Australian standard):
  MARK | BASE LEVEL | TOP LEVEL | TYPE | VERT REINF | CONCRETE STRENGTH | ...
  A-CC01 | Carpark-B2 | Carpark | 350 x 900mm | - | 50 | ...

Steel section format: "310UB40", "200UC25", "150UC37" → looked up in section table.
"""

from __future__ import annotations

import logging
import re

import fitz

logger = logging.getLogger(__name__)

# ── Australian steel section dimensions (W×D in mm) ───────────────────────────
# Source: OneSteel / InfraBuild section tables
_STEEL_SECTIONS: dict[str, tuple[int, int]] = {
    # UB (Universal Beam) — width × depth
    "610UB125": (229, 612), "610UB113": (228, 607), "610UB101": (228, 602),
    "530UB92":  (209, 533), "530UB82":  (209, 528),
    "460UB82":  (191, 460), "460UB74":  (190, 457), "460UB67":  (190, 454),
    "410UB60":  (178, 406), "410UB54":  (178, 403),
    "360UB57":  (172, 359), "360UB51":  (171, 356), "360UB45":  (171, 352),
    "310UB46":  (166, 307), "310UB40":  (165, 304), "310UB32":  (149, 298),
    "250UB37":  (146, 256), "250UB31":  (146, 252), "250UB25":  (124, 248),
    "200UB30":  (134, 203), "200UB25":  (133, 203), "200UB18":  (133, 198),
    "180UB22":  (90,  179), "180UB18":  (90,  175), "180UB16":  (90,  173),
    "150UB18":  (75,  155), "150UB14":  (75,  150),
    # UC (Universal Column) — width × depth
    "310UC158": (311, 327), "310UC137": (309, 320), "310UC118": (307, 315),
    "310UC97":  (305, 308), "250UC89":  (260, 260), "250UC73":  (254, 254),
    "200UC60":  (205, 210), "200UC52":  (206, 206), "200UC46":  (203, 203),
    "150UC37":  (154, 162), "150UC30":  (153, 158), "150UC23":  (152, 152),
    "100UC15":  (100,  97),
    # RHS (Rectangular Hollow Section) — width × depth
    "200X100X6RHS": (100, 200), "150X100X6RHS": (100, 150),
    "200X100X5RHS": (100, 200), "150X75X5RHS":  (75, 150),
    # SHS (Square Hollow Section)
    "150X150X6SHS": (150, 150), "100X100X6SHS": (100, 100),
    # CHS (Circular Hollow Section) — treat as square for simplicity
    "168.3X6CHS": (168, 168), "114.3X6CHS": (114, 114),
}

_DIM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
    re.IGNORECASE,
)
_CONCRETE_MPa_RE = re.compile(r"\b(\d{2,3})\s*(?:MPa|mpa)?\b")


def _parse_section_type(type_str: str) -> dict:
    """Parse a section type string into {width_mm, depth_mm, material}."""
    if not type_str:
        return {}
    s = str(type_str).strip()

    # Explicit WxD format: "350 x 900mm", "220x1000mm", "350X900"
    m = _DIM_RE.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return {
            "width_mm": int(min(a, b)),
            "depth_mm": int(max(a, b)),
            "material": "concrete",  # WxD format is always RC
        }

    # Steel section lookup (try exact then fuzzy)
    su = s.upper().replace(" ", "")
    if su in _STEEL_SECTIONS:
        w, d = _STEEL_SECTIONS[su]
        return {"width_mm": w, "depth_mm": d, "material": "steel"}

    # Partial match: first key that starts with the cleaned string
    for key, (w, d) in _STEEL_SECTIONS.items():
        if key.startswith(su[:6]):
            return {"width_mm": w, "depth_mm": d, "material": "steel"}

    return {}


def _parse_column_schedule(df) -> dict[str, dict]:
    """Parse a column schedule DataFrame into {mark → spec}.

    Expects header row containing: MARK, TYPE, BASE LEVEL, TOP LEVEL,
    CONCRETE STRENGTH (or similar).
    """
    import pandas as pd

    # Find the header row (first row containing 'MARK' or 'TYPE')
    header_idx = None
    for idx, row in df.iterrows():
        vals = [str(v).upper() for v in row.values if v and str(v).strip()]
        if any("MARK" in v for v in vals):
            header_idx = idx
            break

    if header_idx is None:
        return {}

    # Use found row as column headers, reset below it
    df.columns = [str(v).strip() if v else f"col{i}"
                  for i, v in enumerate(df.iloc[header_idx])]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)

    # Normalise column names
    col_map: dict[str, str] = {}
    for c in df.columns:
        cu = str(c).upper()
        if "MARK" in cu:
            col_map["mark"] = c
        elif "TYPE" in cu and "mark" not in col_map.get("type", ""):
            col_map["type"] = c
        elif "BASE" in cu or ("FROM" in cu and "LEVEL" in cu):
            col_map["from_level"] = c
        elif "TOP" in cu or ("TO" in cu and "LEVEL" in cu):
            col_map["to_level"] = c
        elif "CONCRETE" in cu or "STRENGTH" in cu or "MPa" in cu.upper():
            col_map["fc"] = c

    if "mark" not in col_map or "type" not in col_map:
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        mark = str(row.get(col_map["mark"], "")).strip()
        if not mark or mark.lower() in ("nan", "none", "-", ""):
            continue

        type_str = str(row.get(col_map["type"], "")).strip()
        spec = _parse_section_type(type_str)
        if not spec:
            continue

        fc_str = str(row.get(col_map.get("fc", ""), "")).strip()
        fc_m = _CONCRETE_MPa_RE.search(fc_str)
        if fc_m:
            spec["fc_mpa"] = int(fc_m.group(1))
            spec["material"] = "concrete"

        spec["from_level"] = str(row.get(col_map.get("from_level", ""), "")).strip()
        spec["to_level"]   = str(row.get(col_map.get("to_level", ""), "")).strip()

        # Same mark can appear multiple times (different level spans); keep all spans
        if mark not in result:
            result[mark] = spec
        else:
            # Extend the span: keep earliest from_level, latest to_level
            existing = result[mark]
            existing.setdefault("spans", []).append({
                "from": spec["from_level"],
                "to":   spec["to_level"],
            })

    return result


def _parse_beam_schedule(df) -> dict[str, dict]:
    """Parse a beam schedule DataFrame into {mark → spec}."""
    # Find header row
    header_idx = None
    for idx, row in df.iterrows():
        vals = [str(v).upper() for v in row.values if v and str(v).strip()]
        if any("MARK" in v for v in vals):
            header_idx = idx
            break
    if header_idx is None:
        return {}

    df.columns = [str(v).strip() if v else f"col{i}"
                  for i, v in enumerate(df.iloc[header_idx])]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)

    col_map: dict[str, str] = {}
    for c in df.columns:
        cu = str(c).upper()
        if "MARK" in cu:
            col_map["mark"] = c
        elif "SIZE" in cu or "TYPE" in cu or "SECTION" in cu:
            col_map["type"] = c
        elif "WIDTH" in cu or "W" == cu:
            col_map["width"] = c
        elif "DEPTH" in cu or "D" == cu:
            col_map["depth"] = c

    if "mark" not in col_map:
        return {}

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        mark = str(row.get(col_map["mark"], "")).strip()
        if not mark or mark.lower() in ("nan", "none", "-", ""):
            continue

        spec: dict = {}
        if "type" in col_map:
            spec = _parse_section_type(str(row.get(col_map["type"], "")))
        if not spec and "width" in col_map and "depth" in col_map:
            try:
                spec = {
                    "width_mm": int(float(str(row[col_map["width"]]))),
                    "depth_mm": int(float(str(row[col_map["depth"]]))),
                    "material": "concrete",
                }
            except (ValueError, TypeError):
                pass

        if spec:
            result[mark] = spec

    return result


def _is_column_schedule(df) -> bool:
    """Heuristic: does this DataFrame look like a column schedule?"""
    for _, row in df.iterrows():
        vals = [str(v).upper() for v in row.values if v and str(v).strip()]
        text = " ".join(vals)
        if "MARK" in text and ("TYPE" in text or "LEVEL" in text or "COLUMN" in text):
            return True
    return False


def _is_beam_schedule(df) -> bool:
    for _, row in df.iterrows():
        vals = [str(v).upper() for v in row.values if v and str(v).strip()]
        text = " ".join(vals)
        if "MARK" in text and ("BEAM" in text or "SIZE" in text or "SECTION" in text):
            return True
    return False


def _tables_from_page(page: fitz.Page) -> list:
    """Return list of pandas DataFrames from page tables."""
    try:
        tabs = page.find_tables()
        if tabs.tables:
            dfs = []
            for t in tabs.tables:
                try:
                    dfs.append(t.to_pandas())
                except Exception:
                    pass
            return dfs
    except Exception as e:
        logger.warning("find_tables() failed on page: %s", e)
    return []


def extract_all_schedules(
    pdf_path: str,
    classifications: list[dict],
) -> dict:
    """Extract column and beam schedules from all schedule/detail pages.

    Returns:
        {
          "columns": {mark → spec},
          "beams":   {mark → spec},
        }
    """
    doc = fitz.open(pdf_path)
    col_schedule:  dict[str, dict] = {}
    beam_schedule: dict[str, dict] = {}

    _target_types = {"schedule", "detail", "foundation_plan", "floor_plan"}

    for cls in classifications:
        if cls["drawing_type"] not in _target_types:
            continue

        page = doc[cls["page_num"] - 1]
        dfs = _tables_from_page(page)

        for df in dfs:
            if df.empty or df.shape[0] < 2:
                continue

            if _is_column_schedule(df):
                parsed = _parse_column_schedule(df.copy())
                for mark, spec in parsed.items():
                    if mark not in col_schedule:
                        col_schedule[mark] = spec
                    elif spec.get("width_mm", 0) > col_schedule[mark].get("width_mm", 0):
                        col_schedule[mark] = spec

            elif _is_beam_schedule(df):
                parsed = _parse_beam_schedule(df.copy())
                for mark, spec in parsed.items():
                    if mark not in beam_schedule:
                        beam_schedule[mark] = spec

    doc.close()

    logger.info(
        "extract_all_schedules: found %d column marks, %d beam marks",
        len(col_schedule), len(beam_schedule),
    )
    return {"columns": col_schedule, "beams": beam_schedule}
