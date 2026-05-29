"""Quality gate — assess vector extraction results and decide if Vision AI is needed.

Checks:
  - Grid: X≥2 AND Y≥2 axes detected
  - Foundations: >0 found on foundation plan pages
  - Coords: no extreme negative/large coords (raw page coords leaked)
  - Dims: <30% of foundations have zero width/depth/height
  - Schedule: >70% of schedule entries have non-zero dimensions
  - Coverage: found_fdns / (x_axes * y_axes) — too low → suspicious
  - Per-building: if building count detected, each should have foundations
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


_COORD_LIMIT_MM = 30_000   # coords > 30m from origin = wrong
_DIM_ZERO_THRESH = 0.30    # >30% zero-dim foundations → bad
_SCHEDULE_OK_THRESH = 0.70  # >70% schedule entries need non-zero dims
_COVERAGE_MIN = 0.05        # found / (nx*ny) > 5% minimum


@dataclass
class QualityReport:
    grid_ok: bool
    foundations_ok: bool
    coords_ok: bool
    dims_ok: bool
    schedule_ok: bool
    coverage_ratio: float
    building_count: int
    fdn_per_building: dict = field(default_factory=dict)
    failed_checks: list = field(default_factory=list)

    @property
    def needs_vision(self) -> bool:
        return bool(self.failed_checks)

    def __str__(self) -> str:
        status = "OK" if not self.needs_vision else f"NEEDS_AI({','.join(self.failed_checks)})"
        return (f"Quality[{status}] grid={self.grid_ok} fdns={self.foundations_ok} "
                f"coords={self.coords_ok} dims={self.dims_ok} sched={self.schedule_ok} "
                f"cov={self.coverage_ratio:.2f} bldgs={self.building_count}")


def assess_quality(
    fdn_result: dict,
    grid: dict,
    classifications: list[dict] | None = None,
) -> QualityReport:
    """Assess extraction quality and return a QualityReport.

    fdn_result: output of extract_foundations() — has keys: footings, schedule, has_foundation_plan
    grid: output of extract_grids_from_pdf() — has x_axes, y_axes
    classifications: page classification list (to check building count)
    """
    failed: list[str] = []

    # ── Grid check ────────────────────────────────────────────────────────────
    x_count = len(grid.get("x_axes", []))
    y_count = len(grid.get("y_axes", []))
    grid_ok = x_count >= 2 and y_count >= 2
    if not grid_ok:
        failed.append("grid")

    # ── Foundation presence ───────────────────────────────────────────────────
    footings = fdn_result.get("footings", [])
    foundations_ok = len(footings) > 0
    if not foundations_ok and fdn_result.get("has_foundation_plan", False):
        # foundation plan page was found but nothing extracted → definitely needs AI
        failed.append("foundations")

    # ── Coordinate sanity ─────────────────────────────────────────────────────
    coords_ok = True
    if footings:
        bad_coords = sum(
            1 for f in footings
            if abs(f.get("x_mm", 0)) > _COORD_LIMIT_MM
            or abs(f.get("y_mm", 0)) > _COORD_LIMIT_MM
            or f.get("x_mm", 0) < -5000   # clearly wrong origin
        )
        if bad_coords / len(footings) > 0.30:
            coords_ok = False
            failed.append("coords")

    # ── Dimension sanity ──────────────────────────────────────────────────────
    dims_ok = True
    if footings:
        zero_dims = sum(
            1 for f in footings
            if f.get("width_mm", 0) == 0 or f.get("depth_mm", 0) == 0
        )
        if zero_dims / len(footings) > _DIM_ZERO_THRESH:
            dims_ok = False
            failed.append("dims")

    # ── Schedule sanity ───────────────────────────────────────────────────────
    schedule = fdn_result.get("schedule", {})
    schedule_ok = True
    if schedule:
        ok_entries = sum(
            1 for s in schedule.values()
            if (s.get("pile_dia_mm", 0) > 0
                or s.get("width_mm", 0) > 0
                or s.get("height_mm", 0) > 0)
        )
        if ok_entries / len(schedule) < _SCHEDULE_OK_THRESH:
            schedule_ok = False
            failed.append("schedule")

    # ── Coverage ratio ────────────────────────────────────────────────────────
    max_possible = max(x_count * y_count, 1)
    coverage = len(footings) / max_possible
    if foundations_ok and coverage < _COVERAGE_MIN and x_count >= 3 and y_count >= 3:
        failed.append("coverage")

    # ── Building count (heuristic from titles) ────────────────────────────────
    building_count = _detect_building_count(classifications or [])
    fdn_per_building: dict[str, int] = {}
    if building_count > 1:
        buildings = _assign_buildings(footings, building_count)
        fdn_per_building = buildings
        empty_buildings = [b for b, cnt in buildings.items() if cnt == 0]
        if empty_buildings:
            failed.append(f"missing_bldg({','.join(empty_buildings)})")

    return QualityReport(
        grid_ok=grid_ok,
        foundations_ok=foundations_ok,
        coords_ok=coords_ok,
        dims_ok=dims_ok,
        schedule_ok=schedule_ok,
        coverage_ratio=round(coverage, 3),
        building_count=building_count,
        fdn_per_building=fdn_per_building,
        failed_checks=failed,
    )


def _detect_building_count(classifications: list[dict]) -> int:
    """Estimate number of buildings from page titles."""
    building_labels: set[str] = set()
    pattern = re.compile(
        r"\b(?:BUILDING|BLOCK|WING|STAGE|BLDG)\s*([A-Z0-9]+)\b", re.I
    )
    for cls in classifications:
        title = (cls.get("drawing_title") or "") + " " + (cls.get("notes") or "")
        for m in pattern.finditer(title):
            building_labels.add(m.group(1).upper())
    return max(1, len(building_labels))


def _assign_buildings(footings: list[dict], building_count: int) -> dict[str, int]:
    """Heuristic: split footings by X-coordinate percentile bands."""
    if not footings or building_count <= 1:
        return {}

    xs = sorted(f.get("x_mm", 0) for f in footings)
    if not xs:
        return {}

    x_min, x_max = xs[0], xs[-1]
    span = max(x_max - x_min, 1)
    band_size = span / building_count

    counts: dict[str, int] = {}
    labels = [chr(ord("A") + i) for i in range(building_count)]
    for lbl in labels:
        counts[lbl] = 0

    for f in footings:
        band_idx = min(int((f.get("x_mm", 0) - x_min) / band_size), building_count - 1)
        counts[labels[band_idx]] += 1

    return counts
