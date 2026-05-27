"""Build PyTorch Geometric heterogeneous graphs from parsed netlists."""

import math
import torch
from torch_geometric.data import HeteroData, Data


TERMINAL_TYPE_MAP = {
    "gate": 0,
    "drain": 1,
    "source": 2,
    "bulk": 3,
    "p": 4,
    "n": 5,
}

DEVICE_TYPE_MAP = {
    "mosfet": 0,
    "capacitor": 1,
    "current_source": 2,
    "resistor": 3,
}

MODEL_CLASS_MAP = {
    "pmos_lv": 0, "pmos_hvt": 1, "pmos_lvt": 2,
    "nmos_lv": 3, "nmos_hvt": 4, "nmos_lvt": 5,
    "cap": 6, "cap_mim": 6,
    "isrc": 7,
    "res": 8, "res_poly": 8,
    "mos_unknown": 9, "cap_unknown": 6,
}

SUPPLY_NETS = {"vdd", "vdda", "vss", "gnda", "0", "gnd"}
PORT_PREFIXES = ("vinp", "vinn", "vout", "vcm", "vin", "vip", "vbias")


def _is_supply(net: str) -> bool:
    return net.lower() in SUPPLY_NETS or net.upper() in {"VDD", "GND", "GNDA", "VDDA", "VSS"}


def _is_port(net: str, ports: list) -> bool:
    ln = net.lower()
    return ln in {p.lower() for p in ports} or any(ln.startswith(pfx) for pfx in PORT_PREFIXES)


def _log_norm(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0,1] on log scale."""
    if value <= 0 or lo <= 0:
        return 0.0
    log_lo, log_hi = math.log(lo), math.log(hi)
    log_val = math.log(max(lo, min(hi, value)))
    return (log_val - log_lo) / (log_hi - log_lo)


def build_graph(parsed_dict: dict) -> HeteroData:
    """Build a heterogeneous PyG graph from a parsed netlist dict.

    Node types: 'device', 'net'
    Edge types: ('device', 'connects_to', 'net') and reverse

    Device features (per device):
        [0]   device_type one-hot (4 dims)
        [4]   model_class one-hot (10 dims)
        [14]  w_norm (log-scaled, 0–1) — MOSFETs only
        [15]  l_norm (log-scaled, 0–1)
        [16]  m_norm (log-scaled, 0–1)
        [17]  wl_ratio (clipped to [0, 100], normalized)
        [18]  is_pmos (binary)
        [19]  is_nmos (binary)
        [20]  is_mirror (binary — placeholder, set by traversal)
    Total: 21 features

    Net features (per net):
        [0]  is_supply (binary)
        [1]  is_port (binary)
        [2]  fanout_norm (log-scaled fanout / log(max_fanout))
    Total: 3 features

    Edge features:
        [0]  terminal_type one-hot (6 dims)
    """
    devices_list = parsed_dict["devices"]
    ports = parsed_dict.get("ports", [])

    # --- Index nets ---
    all_nets = set()
    for d in devices_list:
        all_nets.update(d["terminals"].values())
    # Canonical order: supply/port first, then internal
    supply_ports = [n for n in all_nets if _is_supply(n) or _is_port(n, ports)]
    internal = sorted(n for n in all_nets if n not in supply_ports)
    sorted_nets = supply_ports + internal
    net_idx = {n: i for i, n in enumerate(sorted_nets)}

    # Fanout counts
    fanout = {n: 0 for n in all_nets}
    for d in devices_list:
        for t, n in d["terminals"].items():
            fanout[n] = fanout.get(n, 0) + 1
    max_fanout = max(fanout.values()) if fanout else 1

    # --- Build device node features ---
    dev_feats = []
    for dev in devices_list:
        feat = [0.0] * 21
        # device_type one-hot [0..3]
        dt_idx = DEVICE_TYPE_MAP.get(dev["device_type"], 0)
        feat[dt_idx] = 1.0
        # model_class one-hot [4..13]
        mc_idx = MODEL_CLASS_MAP.get(dev["model_class"], 9)
        feat[4 + mc_idx] = 1.0
        if dev["device_type"] == "mosfet":
            params = dev.get("params", {})
            w = params.get("w", 1.0)
            l = params.get("l", 0.15)
            m = params.get("m", 1.0)
            feat[14] = _log_norm(w, 0.1, 20.0)
            feat[15] = _log_norm(l, 0.1, 2.0)
            feat[16] = _log_norm(m, 1, 200)
            feat[17] = min(w / max(l, 1e-9), 100.0) / 100.0
            feat[18] = 1.0 if "pmos" in dev["model_class"] else 0.0
            feat[19] = 1.0 if "nmos" in dev["model_class"] else 0.0
        dev_feats.append(feat)

    # --- Build net node features ---
    net_feats = []
    for net in sorted_nets:
        fo = fanout.get(net, 0)
        feat = [
            1.0 if _is_supply(net) else 0.0,
            1.0 if _is_port(net, ports) else 0.0,
            math.log1p(fo) / math.log1p(max_fanout),
        ]
        net_feats.append(feat)

    # --- Build edges ---
    dev_to_net_src, dev_to_net_dst, edge_feats = [], [], []
    for dev_i, dev in enumerate(devices_list):
        for term, net in dev["terminals"].items():
            net_i = net_idx.get(net)
            if net_i is None:
                continue
            term_type = TERMINAL_TYPE_MAP.get(term, 0)
            term_feat = [0.0] * 6
            term_feat[term_type] = 1.0
            dev_to_net_src.append(dev_i)
            dev_to_net_dst.append(net_i)
            edge_feats.append(term_feat)

    data = HeteroData()

    data["device"].x = torch.tensor(dev_feats, dtype=torch.float) if dev_feats else torch.zeros(0, 21)
    data["net"].x = torch.tensor(net_feats, dtype=torch.float) if net_feats else torch.zeros(0, 3)

    if dev_to_net_src:
        edge_index = torch.tensor([dev_to_net_src, dev_to_net_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        data["device", "connects_to", "net"].edge_index = edge_index
        data["device", "connects_to", "net"].edge_attr = edge_attr
        # reverse edges
        data["net", "rev_connects_to", "device"].edge_index = edge_index.flip(0)
        data["net", "rev_connects_to", "device"].edge_attr = edge_attr
    else:
        data["device", "connects_to", "net"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["net", "rev_connects_to", "device"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    data["net_names"] = sorted_nets
    data["device_names"] = [d["name"] for d in devices_list]

    return data


def build_homogeneous_graph(parsed_dict: dict) -> Data:
    """Build a simpler homogeneous PyG graph for GIN encoder.

    All nodes (devices + nets) in a single node set.
    Device features padded to match net feature width.
    """
    hetero = build_graph(parsed_dict)
    n_dev = hetero["device"].x.shape[0]
    n_net = hetero["net"].x.shape[0]

    # Pad device features from 21 → 24 by appending 3 zeros; net features 3 → 24 by prepending 21 zeros
    dev_x = hetero["device"].x  # [n_dev, 21]
    net_x = hetero["net"].x     # [n_net, 3]

    dev_padded = torch.cat([dev_x, torch.zeros(n_dev, 3)], dim=1)  # [n_dev, 24]
    net_padded = torch.cat([torch.zeros(n_net, 21), net_x], dim=1)  # [n_net, 24]
    node_x = torch.cat([dev_padded, net_padded], dim=0)  # [n_dev+n_net, 24]

    # Remap edges: net indices shift by n_dev
    edge_index_raw = hetero["device", "connects_to", "net"].edge_index
    if edge_index_raw.shape[1] > 0:
        src = edge_index_raw[0]           # device indices
        dst = edge_index_raw[1] + n_dev   # net indices offset
        edge_index = torch.stack([src, dst], dim=0)
        # Add reverse
        rev_edge_index = torch.stack([dst, src], dim=0)
        full_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    else:
        full_edge_index = torch.zeros(2, 0, dtype=torch.long)

    data = Data(x=node_x, edge_index=full_edge_index)
    data.n_devices = n_dev
    data.n_nets = n_net
    return data
