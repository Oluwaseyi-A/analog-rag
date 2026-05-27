"""MCP tool: query simulation measurements via SQL filters."""

import os
import sys
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.database.sql_store import SQLStore


def _get_sql() -> SQLStore:
    db_path = os.environ.get("DB_PATH", "/app/data/measurements.db")
    return SQLStore(db_path)


def query_measurements(
    gbp_min: Optional[float] = None,
    gbp_max: Optional[float] = None,
    phase_min: Optional[float] = None,
    phase_max: Optional[float] = None,
    dcgain_min: Optional[float] = None,
    power_max: Optional[float] = None,
    sr_min: Optional[float] = None,
    cmrr_min: Optional[float] = None,
    stable_only: bool = True,
    topology: Optional[str] = None,
    scenario: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query the measurements SQLite table with performance constraints.

    All frequency/magnitude values use SI base units (Hz, dB, W, etc.).

    Returns list of dicts: circuit_id + key performance metrics.
    """
    sql = _get_sql()
    conditions = []

    if stable_only:
        conditions.append("stable = 1")
    if gbp_min is not None:
        conditions.append(f"gbp >= {float(gbp_min)}")
    if gbp_max is not None:
        conditions.append(f"gbp <= {float(gbp_max)}")
    if phase_min is not None:
        conditions.append(f"phase_in_deg >= {float(phase_min)}")
    if phase_max is not None:
        conditions.append(f"phase_in_deg <= {float(phase_max)}")
    if dcgain_min is not None:
        conditions.append(f"dcgain >= {float(dcgain_min)}")
    if power_max is not None:
        conditions.append(f"power <= {float(power_max)}")
    if sr_min is not None:
        conditions.append(f"sr >= {float(sr_min)}")
    if cmrr_min is not None:
        conditions.append(f"cmrrdc <= {float(cmrr_min)}")
    if topology is not None:
        conditions.append(f"topology_name = '{topology}'")
    if scenario is not None:
        conditions.append(f"scenario_name = '{scenario}'")

    where = " AND ".join(conditions) if conditions else ""
    rows = sql.query(sql_filter=where, limit=limit)

    return [_format_row(r) for r in rows]


def get_measurement(circuit_id: str) -> Optional[dict]:
    """Get all measurements for a specific circuit_id."""
    sql = _get_sql()
    row = sql.get_raw(circuit_id)
    if row is None:
        return {"error": f"circuit_id {circuit_id!r} not found"}
    return row


def list_measurement_fields() -> list[str]:
    """Return the list of available measurement columns."""
    return [
        "circuit_id", "topology_name", "scenario_name", "sample_id",
        "dcgain (dB)", "gbp (Hz)", "phase_in_deg (degrees)",
        "sr (V/us)", "power (W)", "area (um^2)", "cmrrdc (dB)",
        "dcpsrn (dB)", "dcpsrp (dB)", "foml (MHz·pF/mA)",
        "foms (MHz·pF/mA·V)", "settling_time (s)", "d_settle",
        "vos25 (V)", "stable (0/1)",
    ]


def _format_row(row: dict) -> dict:
    return {
        "circuit_id": row.get("circuit_id"),
        "topology_name": row.get("topology_name"),
        "scenario_name": row.get("scenario_name"),
        "dcgain_dB": row.get("dcgain"),
        "gbp_Hz": row.get("gbp"),
        "phase_deg": row.get("phase_in_deg"),
        "sr_vus": row.get("sr"),
        "power_W": row.get("power"),
        "area_um2": row.get("area"),
        "cmrrdc_dB": row.get("cmrrdc"),
        "foml": row.get("foml"),
        "foms": row.get("foms"),
        "stable": row.get("stable"),
    }
