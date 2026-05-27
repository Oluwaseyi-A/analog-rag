"""Unit tests for embedder (offline — no API calls)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
torch = pytest.importorskip("torch", reason="torch not installed — skip in non-Docker env")
from src.ingestion.embedder import GINEncoder, embed_graph, build_behavior_embedding, build_behavior_vector
from src.ingestion.netlist_parser import parse_and_canonicalize
from src.ingestion.graph_builder import build_homogeneous_graph

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

SAMPLE_RESULT = {
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


def test_gin_encoder_output_shape():
    model = GINEncoder(in_channels=24, hidden_dim=256, out_dim=256)
    model.eval()
    x = torch.randn(10, 24)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    from torch_geometric.data import Data, Batch
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 256)


def test_embed_graph_returns_list():
    parsed = parse_and_canonicalize(FAN_SMC_SNIPPET)
    g = build_homogeneous_graph(parsed)
    emb = embed_graph(g)
    assert isinstance(emb, list)
    assert len(emb) == 256
    assert all(isinstance(v, float) for v in emb)


def test_embed_graph_normalized():
    parsed = parse_and_canonicalize(FAN_SMC_SNIPPET)
    g = build_homogeneous_graph(parsed)
    emb = embed_graph(g)
    import math
    norm = math.sqrt(sum(v**2 for v in emb))
    assert abs(norm - 1.0) < 1e-3


def test_behavior_vector_length():
    vec = build_behavior_vector(SAMPLE_RESULT, dim=64)
    assert len(vec) == 64
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_behavior_embedding_equals_vector():
    vec = build_behavior_embedding(SAMPLE_RESULT, dim=64)
    assert len(vec) == 64


def test_behavior_vector_stable_flag():
    result_stable = {**SAMPLE_RESULT, "measured__stable": 1}
    result_unstable = {**SAMPLE_RESULT, "measured__stable": 0}
    v_stable = build_behavior_vector(result_stable, dim=64)
    v_unstable = build_behavior_vector(result_unstable, dim=64)
    assert v_stable[14] == 1.0   # stable field is at index 14
    assert v_unstable[14] == 0.0
