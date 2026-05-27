"""SQLite store for simulation measurements."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    circuit_id       TEXT PRIMARY KEY,
    topology_name    TEXT NOT NULL,
    scenario_name    TEXT NOT NULL,
    sample_id        TEXT NOT NULL,
    dcgain           REAL,
    gbp              REAL,
    phase_in_deg     REAL,
    sr               REAL,
    power            REAL,
    area             REAL,
    cmrrdc           REAL,
    dcpsrn           REAL,
    dcpsrp           REAL,
    foml             REAL,
    foms             REAL,
    settling_time    REAL,
    d_settle         REAL,
    vos25            REAL,
    stable           INTEGER,
    raw_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_topology ON measurements(topology_name);
CREATE INDEX IF NOT EXISTS idx_gbp ON measurements(gbp);
CREATE INDEX IF NOT EXISTS idx_phase ON measurements(phase_in_deg);
CREATE INDEX IF NOT EXISTS idx_stable ON measurements(stable);
"""


def _result_to_row(circuit_id: str, result: dict) -> dict:
    return {
        "circuit_id": circuit_id,
        "topology_name": result.get("topology_name", ""),
        "scenario_name": result.get("scenario_name", ""),
        "sample_id": result.get("sample_id", ""),
        "dcgain": result.get("measured__dcgain"),
        "gbp": result.get("measured__gbp"),
        "phase_in_deg": result.get("measured__phase_in_deg"),
        "sr": result.get("measured__SR"),
        "power": result.get("measured__power"),
        "area": result.get("measured__area"),
        "cmrrdc": result.get("measured__cmrrdc"),
        "dcpsrn": result.get("measured__dcpsrn"),
        "dcpsrp": result.get("measured__dcpsrp"),
        "foml": result.get("measured__foml"),
        "foms": result.get("measured__foms"),
        "settling_time": result.get("measured__settlingTime"),
        "d_settle": result.get("measured__d_settle"),
        "vos25": result.get("measured__vos25"),
        "stable": int(result.get("measured__stable", 0)),
        "raw_json": json.dumps(result),
    }


class SQLStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def upsert(self, circuit_id: str, result_row: dict):
        row = _result_to_row(circuit_id, result_row)
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "circuit_id")
        sql = (
            f"INSERT INTO measurements ({col_str}) VALUES ({placeholders}) "
            f"ON CONFLICT(circuit_id) DO UPDATE SET {updates}"
        )
        with self._conn() as conn:
            conn.execute(sql, [row[c] for c in cols])

    def upsert_batch(self, records: list[tuple[str, dict]]):
        for circuit_id, result_row in records:
            self.upsert(circuit_id, result_row)

    def query(self, sql_filter: str = "", limit: int = 100) -> list[dict]:
        """Execute a WHERE clause filter and return matching rows as dicts."""
        where = f"WHERE {sql_filter}" if sql_filter else ""
        sql = f"SELECT * FROM measurements {where} LIMIT {limit}"
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get(self, circuit_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM measurements WHERE circuit_id = ?", (circuit_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("raw_json"):
            d["raw"] = json.loads(d["raw_json"])
        return d

    def get_raw(self, circuit_id: str) -> Optional[dict]:
        row = self.get(circuit_id)
        if row and row.get("raw"):
            return row["raw"]
        return row

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]

    def list_circuit_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT circuit_id FROM measurements ORDER BY circuit_id").fetchall()
        return [r[0] for r in rows]

    def list_topologies(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT topology_name FROM measurements ORDER BY topology_name").fetchall()
        return [r[0] for r in rows]

    def query_performance(
        self,
        gbp_min: Optional[float] = None,
        phase_min: Optional[float] = None,
        dcgain_min: Optional[float] = None,
        stable_only: bool = True,
        topology: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        conditions = []
        if stable_only:
            conditions.append("stable = 1")
        if gbp_min is not None:
            conditions.append(f"gbp >= {gbp_min}")
        if phase_min is not None:
            conditions.append(f"phase_in_deg >= {phase_min}")
        if dcgain_min is not None:
            conditions.append(f"dcgain >= {dcgain_min}")
        if topology is not None:
            conditions.append(f"topology_name = '{topology}'")
        where = " AND ".join(conditions)
        return self.query(where, limit=limit)
