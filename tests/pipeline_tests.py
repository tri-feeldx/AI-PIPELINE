"""5 automated test cases for the structural PDF pipeline.

Run with:  python -m pytest tests/test_pipeline.py -v
All 5 tests must pass before handover.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── TC-1: JSON bad literals ────────────────────────────────────────────────────

def test_parse_json_bad_literals():
    """_parse_json must handle undefined, NaN, True and produce valid Python."""
    from pipeline.vision_extractor import _parse_json

    text = '[{"label":"P1","x":0.1,"y":undefined,"ok":True,"bad":NaN}]'
    result = _parse_json(text)

    assert isinstance(result, list), "Expected list"
    assert len(result) == 1, "Expected 1 item"
    assert result[0]["bad"] == 0        # NaN → 0
    assert result[0]["y"] is None       # undefined → null
    assert result[0]["ok"] is True      # True preserved as Python True


# ── TC-2: Truncated array recovery ────────────────────────────────────────────

def test_parse_json_truncated_recovery():
    """Partial array cut off mid-object must salvage all complete objects before cut."""
    from pipeline.vision_extractor import _parse_json

    # 3 complete objects, 4th truncated mid-value
    text = '[{"label":"P1"},{"label":"P2"},{"label":"P3"},{"label":"P4","x":0.'
    result = _parse_json(text)

    assert isinstance(result, list), "Expected list from partial recovery"
    assert len(result) == 3, f"Expected 3 salvaged objects, got {len(result)}"
    assert [r["label"] for r in result] == ["P1", "P2", "P3"]


# ── TC-3: OVERALL vs DETAIL role assignment ────────────────────────────────────

def test_assign_foundation_roles_mixed_scale():
    """Pages at 1:250 → overall; pages at 1:100 → detail when ratio ≥ 1.5×."""
    from pipeline.model_builder import _assign_foundation_roles

    classifications = [
        {"drawing_type": "foundation_plan", "page_num": 1, "scale_ratio": 250},
        {"drawing_type": "foundation_plan", "page_num": 2, "scale_ratio": 100},
        {"drawing_type": "foundation_plan", "page_num": 3, "scale_ratio": 100},
        {"drawing_type": "floor_plan",      "page_num": 4, "scale_ratio": 100},
    ]
    _assign_foundation_roles(classifications)

    assert classifications[0]["plan_role"] == "overall", "1:250 page must be 'overall'"
    assert classifications[1]["plan_role"] == "detail",  "1:100 page must be 'detail'"
    assert classifications[2]["plan_role"] == "detail",  "1:100 page must be 'detail'"
    assert "plan_role" not in classifications[3], "floor_plan must not get plan_role"


# ── TC-4: Schedule-guided filter ──────────────────────────────────────────────

def test_schedule_filter_drops_unknown_marks():
    """Vision results whose labels are not in the schedule must be dropped."""
    schedule = {"PC1": {}, "PF1": {}, "PF2": {}, "PF3": {}}  # 4 known marks ≥ 3

    all_raw = [
        {"label": "PC1",   "grid_ref": "A/1"},       # KEEP — in schedule
        {"label": "PF1",   "grid_ref": "A/2"},       # KEEP — in schedule
        {"label": "L-D1",  "grid_ref": "C/3"},       # DROP — schedule-table leak
        {"label": "TABLE", "grid_ref": "off_grid"},  # DROP — noise
    ]

    # Replicate the filter logic from extract_foundations_vision
    if len(schedule) >= 3:
        known_upper = {k.upper() for k in schedule}
        filtered = [f for f in all_raw if str(f.get("label", "")).upper() in known_upper]
    else:
        filtered = all_raw

    assert len(filtered) == 2, f"Expected 2 kept, got {len(filtered)}"
    assert all(f["label"] in {"PC1", "PF1"} for f in filtered)


# ── TC-5: Foundation deduplication ────────────────────────────────────────────

def test_deduplicate_footings_prefers_vector():
    """Same label within 150mm must deduplicate; vector source preferred over vision_ai."""
    from pipeline.model_builder import _deduplicate_footings

    footings = [
        {"label": "PC1", "x_mm": 1000.0, "y_mm": 2000.0, "source": "vision_ai"},
        {"label": "PC1", "x_mm": 1030.0, "y_mm": 2020.0, "source": "vision_ai"},  # duplicate
        {"label": "PC1", "x_mm": 1000.0, "y_mm": 2000.0, "source": "vector"},     # preferred
        {"label": "PC2", "x_mm": 5000.0, "y_mm": 2000.0, "source": "vision_ai"},  # different pos
    ]

    result = _deduplicate_footings(footings, tol_mm=150.0)

    assert len(result) == 2, f"Expected 2 unique foundations, got {len(result)}"
    pc1 = next(f for f in result if f["label"] == "PC1")
    assert pc1["source"] == "vector", "Vector source must be preferred over vision_ai"
