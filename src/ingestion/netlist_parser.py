"""SPICE netlist parser and canonicalizer for analog circuit netlists."""

import re
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Device:
    name: str           # e.g. "xm11"
    device_type: str    # "mosfet", "capacitor", "current_source", "resistor", "voltage_source"
    model: str          # raw model name from SPICE
    model_class: str    # normalized class, e.g. "pmos_lv"
    terminals: dict     # {"drain": net, "gate": net, "source": net, "bulk": net} or {"p": net, "n": net}
    params: dict        # raw params: {"w": float, "l": float, "m": int}
    params_binned: dict # log-binned params for canonical form


@dataclass
class ParsedNetlist:
    raw_text: str
    subckt_name: str
    ports: list         # ordered port list from .subckt
    devices: list       # List[Device]
    nets: set           # all net names
    canonical_text: str = ""
    net_map: dict = field(default_factory=dict)   # original_net -> canonical_net


# Sky130 PDK model to class mapping
PDK_MODEL_MAP = {
    "sky130_fd_pr__pfet_01v8": "pmos_lv",
    "sky130_fd_pr__pfet_01v8_hvt": "pmos_hvt",
    "sky130_fd_pr__pfet_01v8_lvt": "pmos_lvt",
    "sky130_fd_pr__nfet_01v8": "nmos_lv",
    "sky130_fd_pr__nfet_01v8_hvt": "nmos_hvt",
    "sky130_fd_pr__nfet_01v8_lvt": "nmos_lvt",
    "sky130_fd_pr__cap_mim_m3_1": "cap_mim",
    "sky130_fd_pr__cap_mim_m3_2": "cap_mim",
    "sky130_fd_pr__res_generic_nd": "res_poly",
}

SUPPLY_NETS = {"vdd", "vdda", "vss", "gnda", "0", "gnd", "avdd", "avss"}
PORT_NET_PATTERNS = ["vinp", "vinn", "vout", "vcm", "vbias", "vin", "vip"]


def _parse_value(s: str) -> float:
    """Convert SPICE value string to float (handles suffixes p, n, u, m, k, meg, g)."""
    s = s.strip().lower()
    if not s:
        return 0.0
    suffixes = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
                "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
    for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            try:
                return float(s[:-len(suffix)]) * mult
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _log_bin(value: float, lo: float, hi: float, n_bins: int = 16) -> int:
    """Map value into log-spaced bin index [0, n_bins-1]."""
    if value <= 0 or lo <= 0:
        return 0
    log_lo, log_hi = math.log(lo), math.log(hi)
    log_val = math.log(max(lo, min(hi, value)))
    idx = int((log_val - log_lo) / (log_hi - log_lo) * (n_bins - 1) + 0.5)
    return max(0, min(n_bins - 1, idx))


def _parse_mosfet_line(tokens: list) -> tuple[dict, dict]:
    """Parse MOSFET instance line tokens → (terminals, params).

    Format: Xname drain gate source bulk model [params...]
    """
    terminals = {
        "drain": tokens[1].lower(),
        "gate": tokens[2].lower(),
        "source": tokens[3].lower(),
        "bulk": tokens[4].lower(),
    }
    model = tokens[5]
    params = {}
    for tok in tokens[6:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            params[k.lower()] = _parse_value(v)
    return terminals, model, params


def _parse_capacitor_line(tokens: list) -> tuple[dict, dict]:
    """Parse capacitor instance: Cname node+ node- value"""
    terminals = {"p": tokens[1].lower(), "n": tokens[2].lower()}
    params = {"c": _parse_value(tokens[3]) if len(tokens) > 3 else 0.0}
    return terminals, params


def _parse_current_source_line(tokens: list) -> tuple[dict, dict]:
    """Parse current source: Iname node+ node- value"""
    terminals = {"p": tokens[1].lower(), "n": tokens[2].lower()}
    params = {"i": _parse_value(tokens[3]) if len(tokens) > 3 else 0.0}
    return terminals, params


def _parse_resistor_line(tokens: list) -> tuple[dict, dict]:
    """Parse resistor: Rname node+ node- value"""
    terminals = {"p": tokens[1].lower(), "n": tokens[2].lower()}
    params = {"r": _parse_value(tokens[3]) if len(tokens) > 3 else 0.0}
    return terminals, params


def parse_spice_subckt(netlist_text: str, pdk_map: dict = None) -> ParsedNetlist:
    """Parse a SPICE subcircuit definition into a ParsedNetlist."""
    if pdk_map is None:
        pdk_map = PDK_MODEL_MAP

    lines = []
    current = ""
    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            if current:
                lines.append(current)
                current = ""
            continue
        if stripped.startswith("+"):
            current += " " + stripped[1:].strip()
        else:
            if current:
                lines.append(current)
            current = stripped
    if current:
        lines.append(current)

    subckt_name = ""
    ports = []
    devices = []
    all_nets: set = set()

    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        first = tokens[0].lower()

        if first == ".subckt":
            subckt_name = tokens[1]
            ports = [p.lower() for p in tokens[2:]]
            all_nets.update(ports)
            continue

        if first == ".ends":
            break

        name = tokens[0].lower()
        if name.startswith("xm") or name.startswith("m"):
            if len(tokens) < 6:
                continue
            terminals, model, params = _parse_mosfet_line(tokens)
            model_class = pdk_map.get(model, "mos_unknown")
            all_nets.update(terminals.values())
            w = params.get("w", 1.0)
            l = params.get("l", 0.15)
            m = int(params.get("m", 1))
            params_binned = {
                "w_bin": _log_bin(w, 0.1, 20.0),
                "l_bin": _log_bin(l, 0.1, 2.0),
                "m_bin": _log_bin(m, 1, 200),
                "wl_ratio": round(w / max(l, 1e-9), 2),
            }
            devices.append(Device(name=name, device_type="mosfet", model=model,
                                   model_class=model_class, terminals=terminals,
                                   params=params, params_binned=params_binned))

        elif name.startswith("c"):
            terminals, params = _parse_capacitor_line(tokens)
            all_nets.update(terminals.values())
            devices.append(Device(name=name, device_type="capacitor", model="cap",
                                   model_class="cap", terminals=terminals,
                                   params=params, params_binned={}))

        elif name.startswith("i"):
            terminals, params = _parse_current_source_line(tokens)
            all_nets.update(terminals.values())
            devices.append(Device(name=name, device_type="current_source", model="isrc",
                                   model_class="isrc", terminals=terminals,
                                   params=params, params_binned={}))

        elif name.startswith("r"):
            terminals, params = _parse_resistor_line(tokens)
            all_nets.update(terminals.values())
            devices.append(Device(name=name, device_type="resistor", model="res",
                                   model_class="res", terminals=terminals,
                                   params=params, params_binned={}))

    parsed = ParsedNetlist(
        raw_text=netlist_text,
        subckt_name=subckt_name,
        ports=ports,
        devices=devices,
        nets=all_nets,
    )
    return parsed


def canonicalize(parsed: ParsedNetlist, pdk_map: dict = None) -> ParsedNetlist:
    """BFS rename nets from supply/port anchors to deterministic integer labels."""
    if pdk_map is None:
        pdk_map = PDK_MODEL_MAP

    supply_nets = SUPPLY_NETS
    port_nets = set(parsed.ports)

    # Build net adjacency: net -> set of devices touching it
    net_to_devices: dict = defaultdict(set)
    for dev in parsed.devices:
        for term, net in dev.terminals.items():
            net_to_devices[net].add(dev.name)

    # BFS starting from supply + port nets (anchors)
    net_map = {}
    anchor_counter = {"supply": 0, "port": 0, "int": 0}

    def assign(net: str, prefix: str, counter_key: str) -> str:
        if net in net_map:
            return net_map[net]
        lnet = net.lower()
        name = f"{prefix}{anchor_counter[counter_key]}"
        net_map[net] = name
        anchor_counter[counter_key] += 1
        return name

    # First pass: assign well-known names
    for net in parsed.nets:
        lnet = net.lower()
        if lnet in supply_nets or lnet in {"0", "gnd", "gnda", "vss"}:
            if lnet in {"vdd", "vdda", "avdd"}:
                net_map[net] = "VDD"
            elif lnet in {"vss", "gnda", "gnd", "avss", "0"}:
                net_map[net] = "GND"
            else:
                net_map[net] = lnet.upper()
        elif lnet in {p.lower() for p in port_nets}:
            # Keep port names semantic
            net_map[net] = net.upper()

    # BFS from anchors to interior nets
    queue = deque(net_map.keys())
    visited = set(net_map.keys())
    internal_idx = 0

    while queue:
        current_net = queue.popleft()
        for dev_name in net_to_devices.get(current_net, []):
            dev = next((d for d in parsed.devices if d.name == dev_name), None)
            if dev is None:
                continue
            for term, net in dev.terminals.items():
                if net not in visited:
                    visited.add(net)
                    net_map[net] = f"N{internal_idx}"
                    internal_idx += 1
                    queue.append(net)

    # Any remaining unvisited nets
    for net in parsed.nets:
        if net not in net_map:
            net_map[net] = f"N{internal_idx}"
            internal_idx += 1

    parsed.net_map = net_map

    # Rebuild canonical text
    lines = [f".subckt {parsed.subckt_name} {' '.join(net_map.get(p, p.upper()) for p in parsed.ports)}"]
    for dev in parsed.devices:
        canonical_terminals = {k: net_map.get(v, v) for k, v in dev.terminals.items()}
        if dev.device_type == "mosfet":
            model_class = pdk_map.get(dev.model, dev.model_class)
            params_str = " ".join(
                f"{k}={_fmt_bin(v)}" for k, v in dev.params_binned.items()
                if k in ("w_bin", "l_bin", "m_bin")
            )
            t = canonical_terminals
            lines.append(
                f"{dev.name} {t['drain']} {t['gate']} {t['source']} {t['bulk']} {model_class} {params_str}"
            )
        elif dev.device_type == "capacitor":
            t = canonical_terminals
            c_val = dev.params.get("c", 0.0)
            lines.append(f"{dev.name} {t['p']} {t['n']} {_fmt_si(c_val)}f")
        elif dev.device_type == "current_source":
            t = canonical_terminals
            i_val = dev.params.get("i", 0.0)
            lines.append(f"{dev.name} {t['p']} {t['n']} {_fmt_si(i_val)}u")
        elif dev.device_type == "resistor":
            t = canonical_terminals
            r_val = dev.params.get("r", 0.0)
            lines.append(f"{dev.name} {t['p']} {t['n']} {_fmt_si(r_val)}k")
    lines.append(f".ends {parsed.subckt_name}")

    parsed.canonical_text = "\n".join(lines)
    return parsed


def _fmt_bin(v) -> str:
    return str(int(v)) if isinstance(v, (int, float)) else str(v)


def _fmt_si(v: float) -> str:
    """Format float to 3 sig figs."""
    if v == 0:
        return "0"
    return f"{v:.3g}"


def generate_llm_header(parsed: ParsedNetlist, topology_descriptions: dict = None) -> str:
    """Generate a one-line LLM description of the circuit topology."""
    n_pmos = sum(1 for d in parsed.devices if "pmos" in d.model_class)
    n_nmos = sum(1 for d in parsed.devices if "nmos" in d.model_class)
    n_caps = sum(1 for d in parsed.devices if d.device_type == "capacitor")
    n_isrc = sum(1 for d in parsed.devices if d.device_type == "current_source")

    if topology_descriptions and parsed.subckt_name in topology_descriptions:
        base = topology_descriptions[parsed.subckt_name]
    else:
        base = parsed.subckt_name.replace("_", " ")

    header = (
        f"* {base}. "
        f"PMOS input differential pair. "
        f"Devices: {n_pmos}P/{n_nmos}N MOSFETs, {n_caps} caps, {n_isrc} current sources. "
        f"Sky130 1.8V LV process."
    )
    return header


def load_netlist_file(path: str | Path) -> str:
    """Read a netlist file, handling CRLF line endings."""
    text = Path(path).read_text(errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_and_canonicalize(netlist_text: str, topology_descriptions: dict = None) -> dict:
    """Full pipeline: parse + canonicalize + generate header. Returns dict for embedding."""
    parsed = parse_spice_subckt(netlist_text)
    parsed = canonicalize(parsed)
    header = generate_llm_header(parsed, topology_descriptions)
    canonical_with_header = header + "\n" + parsed.canonical_text

    return {
        "subckt_name": parsed.subckt_name,
        "ports": parsed.ports,
        "canonical_text": canonical_with_header,
        "raw_text": parsed.raw_text,
        "net_map": parsed.net_map,
        "device_count": len(parsed.devices),
        "devices": [
            {
                "name": d.name,
                "device_type": d.device_type,
                "model": d.model,
                "model_class": d.model_class,
                "terminals": d.terminals,
                "params": d.params,
                "params_binned": d.params_binned,
            }
            for d in parsed.devices
        ],
    }
