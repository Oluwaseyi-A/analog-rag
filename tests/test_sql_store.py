"""Unit tests for SQL store."""

import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from src.database.sql_store import SQLStore

SAMPLE_RESULT = {
    "topology_name": "Fan_SMC_Pin_3",
    "scenario_name": "nominal_tt",
    "sample_id": "test_sample_001",
    "measured__dcgain": 98.0,
    "measured__gbp": 713494.0,
    "measured__phase_in_deg": 74.1,
    "measured__SR": 0.42,
    "measured__power": 9.96e-05,
    "measured__area": 133.99,
    "measured__cmrrdc": -64.0,
    "measured__dcpsrn": -70.5,
    "measured__dcpsrp": -69.3,
    "measured__foml": 42.3,
    "measured__foms": 71601293.0,
    "measured__settlingTime": 1.097e-05,
    "measured__d_settle": 0.003,
    "measured__vos25": -0.001,
    "measured__stable": 1,
}


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLStore(Path(tmpdir) / "test.db")
        yield store


def test_upsert_and_get(db):
    cid = "Fan_SMC_Pin_3/nominal_tt/test_sample_001"
    db.upsert(cid, SAMPLE_RESULT)
    row = db.get(cid)
    assert row is not None
    assert row["topology_name"] == "Fan_SMC_Pin_3"
    assert abs(row["dcgain"] - 98.0) < 0.01


def test_count(db):
    db.upsert("topo/sc/s1", {**SAMPLE_RESULT, "sample_id": "s1"})
    db.upsert("topo/sc/s2", {**SAMPLE_RESULT, "sample_id": "s2"})
    assert db.count() == 2


def test_query_filter(db):
    db.upsert("topo1/sc/s1", {**SAMPLE_RESULT, "measured__gbp": 2e6, "measured__phase_in_deg": 70.0})
    db.upsert("topo2/sc/s2", {**SAMPLE_RESULT, "measured__gbp": 0.5e6, "measured__phase_in_deg": 70.0})
    results = db.query("gbp >= 1e6", limit=10)
    assert len(results) == 1
    assert results[0]["gbp"] >= 1e6


def test_query_performance(db):
    db.upsert("t1/sc/s1", {**SAMPLE_RESULT, "measured__gbp": 2e6, "measured__phase_in_deg": 65.0})
    db.upsert("t2/sc/s2", {**SAMPLE_RESULT, "measured__gbp": 0.5e6, "measured__phase_in_deg": 55.0})
    results = db.query_performance(gbp_min=1e6, phase_min=60.0)
    assert all(r["gbp"] >= 1e6 for r in results)
    assert all(r["phase_in_deg"] >= 60.0 for r in results)


def test_upsert_idempotent(db):
    cid = "topo/sc/s1"
    db.upsert(cid, SAMPLE_RESULT)
    db.upsert(cid, {**SAMPLE_RESULT, "measured__dcgain": 99.0})
    assert db.count() == 1
    row = db.get(cid)
    assert abs(row["dcgain"] - 99.0) < 0.01


def test_list_topologies(db):
    db.upsert("A/sc/s1", {**SAMPLE_RESULT, "topology_name": "A"})
    db.upsert("B/sc/s1", {**SAMPLE_RESULT, "topology_name": "B"})
    topos = db.list_topologies()
    assert "A" in topos and "B" in topos
