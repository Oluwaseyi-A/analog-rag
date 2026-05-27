"""Unit tests for graph builder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
torch = pytest.importorskip("torch", reason="torch not installed — skip in non-Docker env")
from src.ingestion.netlist_parser import parse_and_canonicalize
from src.ingestion.graph_builder import build_graph, build_homogeneous_graph

FAN_SMC_SNIPPET = """\
.subckt fan_smc_pin_3 gnda vdda vinn vinp vout
xm11 VOUT net050 VDDA VDDA sky130_fd_pr__pfet_01v8 l=0.4026457506 w=4.855850139 m=10
xm9 net063 VINP net31 net31 sky130_fd_pr__pfet_01v8 l=0.2789243531 w=2.231427981 m=18
xm8 DM_2 VINN net31 net31 sky130_fd_pr__pfet_01v8 l=0.2789243531 w=2.231427981 m=18
xm23 VOUT net049 GNDA GNDA sky130_fd_pr__nfet_01v8 l=0.2097779069 w=0.6218585292 m=10
I0 net013 GNDA 3.4186871u
C0 net050 VOUT 16.487924p
.ends fan_smc_pin_3
"""


@pytest.fixture
def parsed():
    return parse_and_canonicalize(FAN_SMC_SNIPPET)


def test_hetero_graph_node_types(parsed):
    g = build_graph(parsed)
    assert "device" in g.node_types
    assert "net" in g.node_types
    assert g["device"].x.shape[1] == 21
    assert g["net"].x.shape[1] == 3


def test_hetero_graph_has_edges(parsed):
    g = build_graph(parsed)
    edge_index = g["device", "connects_to", "net"].edge_index
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0


def test_homogeneous_graph_shape(parsed):
    g = build_homogeneous_graph(parsed)
    assert g.x.shape[1] == 24  # 21 device + 3 net features
    assert g.edge_index.shape[0] == 2
    total_nodes = g.n_devices + g.n_nets
    assert g.x.shape[0] == total_nodes


def test_device_features_pmos(parsed):
    g = build_graph(parsed)
    dev_x = g["device"].x
    devices = parsed["devices"]
    pmos_devs = [i for i, d in enumerate(devices) if d["model_class"] == "pmos_lv"]
    assert len(pmos_devs) > 0
    for i in pmos_devs:
        assert dev_x[i, 18] == 1.0  # is_pmos
        assert dev_x[i, 19] == 0.0  # is_nmos


def test_net_feature_supply(parsed):
    g = build_graph(parsed)
    net_x = g["net"].x
    net_names = g["net_names"]
    supply_idx = [i for i, n in enumerate(net_names) if n in ("VDD", "GND", "GNDA", "VDDA")]
    for i in supply_idx:
        assert net_x[i, 0] == 1.0  # is_supply
