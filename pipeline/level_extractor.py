"""Level height extractor.

Tries to find floor elevations from PDF text (RL annotations).
Falls back to standard 3600mm floor-to-floor height when not found.
"""

import re
import fitz

PT_TO_MM = 25.4 / 72.0

_RL_RE = re.compile(r"RL\s*[+\-]?\s*([\d]+\.?\d*)", re.I)
_LEVEL_RL_RE = re.compile(
    r"(LEVEL\s*\d+|GROUND\s*FL(?:OOR)?|ROOF|MEZZANINE)"
    r"\s*(?:RL|FFL|EL|ELEV)\s*[+\-]?\s*([\d]+\.?\d*)",
    re.I,
)


def extract_levels_from_pdf(pdf_path: str) -> list[dict]:
    """Extract floor levels. Returns list sorted by elevation_mm."""
    doc = fitz.open(pdf_path)
    all_levels: dict[str, int] = {}

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()

        # Only try pages that look like elevation drawings
        if not re.search(r"\b(elevation|section)\b", text, re.I):
            continue

        for m in _LEVEL_RL_RE.finditer(text):
            name = re.sub(r"\s+", " ", m.group(1)).strip().upper()
            name = re.sub(r"LEVEL\s+0*(\d+)", r"LEVEL \1", name)
            try:
                val = float(m.group(2))
                elev_mm = round(val * 1000) if val < 100 else round(val)
                if name not in all_levels:
                    all_levels[name] = elev_mm
            except ValueError:
                pass

    doc.close()

    if len(all_levels) >= 2:
        items = sorted(
            [{"name": k, "elevation_mm": v} for k, v in all_levels.items()],
            key=lambda l: l["elevation_mm"],
        )
        # Sanity check
        heights = [items[i+1]["elevation_mm"] - items[i]["elevation_mm"]
                   for i in range(len(items)-1)]
        if all(2000 <= h <= 7000 for h in heights):
            return items

    # Fallback: find level names from the PDF and assign 3600mm spacing
    level_names = _find_level_names(pdf_path)
    return _assign_default_heights(level_names)


def _find_level_names(pdf_path: str) -> list[str]:
    """Collect unique level names from elevation pages, ordered by occurrence."""
    doc = fitz.open(pdf_path)
    seen: dict[str, int] = {}  # name → count

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()
        if not re.search(r"\b(elevation|section)\b", text, re.I):
            continue
        for m in re.finditer(
            r"\b(LEVEL\s*\d+|GROUND\s*FL(?:OOR)?|ROOF|MEZZANINE)\b",
            text, re.I
        ):
            raw = m.group(0)
            # Skip "Level 18, 25 Martin Place" style address patterns
            after = text[m.end():m.end()+20]
            if re.match(r"\s*,\s*\d", after):  # followed by ", number" = address
                continue
            # Skip level numbers > 10 (unlikely to be real floors)
            num_m = re.search(r"\d+", raw)
            if num_m and int(num_m.group()) > 10:
                continue
            name = re.sub(r"\s+", " ", raw).strip().upper()
            name = re.sub(r"LEVEL\s+0*(\d+)", r"LEVEL \1", name)
            seen[name] = seen.get(name, 0) + 1

    doc.close()

    # Keep names that appear at least twice (once per elevation side)
    names = [k for k, v in seen.items() if v >= 2]

    # Sort: ground first, then numbered levels, roof last
    def _sort_key(n):
        if "GROUND" in n: return 0
        m = re.search(r"\d+", n)
        if m: return int(m.group())
        if "ROOF" in n: return 999
        return 500

    names.sort(key=_sort_key)
    return names


def _assign_default_heights(names: list[str]) -> list[dict]:
    """Assign 3600mm floor-to-floor spacing to level names."""
    if not names:
        return [
            {"name": "LEVEL 1", "elevation_mm": 0},
            {"name": "LEVEL 2", "elevation_mm": 3600},
            {"name": "LEVEL 3", "elevation_mm": 7200},
            {"name": "LEVEL 4", "elevation_mm": 10800},
            {"name": "LEVEL 5", "elevation_mm": 14400},
            {"name": "ROOF",    "elevation_mm": 18000},
        ]

    levels = []
    floor_height = 3600  # standard Australian commercial floor-to-floor
    for i, name in enumerate(names):
        levels.append({"name": name, "elevation_mm": i * floor_height})
    return levels
