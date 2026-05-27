"""Unit tests for netlist parser and canonicalization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from src.ingestion.netlist_parser import (
    parse_spice_subckt, canonicalize, parse_and_canonicalize, _parse_value, _log_bin
)

FAN_SMC_SNIPPET = """\
.subckt fan_smc_pin_3 gnda vdda vinn vinp vout
xm11 VOUT net050 VDDA VDDA sky130_fd_pr__pfet_01v8 l=0.4026457506 w=4.855850139 m=10
xm7 net049 net013 VDDA VDDA sky130_fd_pr__pfet_01v8 l=0.3722953979 w=4.973977458 m=8
xm9 net063 VINP net31 net31 sky130_fd_pr__pfet_01v8 l=0.2789243531 w=2.231427981 m=18
xm8 DM_2 VINN net31 net31 sky130_fd_pr__pfet_01v8 l=0.2789243531 w=2.231427981 m=18
xm23 VOUT net049 GNDA GNDA sky130_fd_pr__nfet_01v8 l=0.2097779069 w=0.6218585292 m=10
I0 net013 GNDA 3.4186871u
C0 net050 VOUT 16.487924p
.ends fan_smc_pin_3
"""


def test_parse_device_count():
    parsed = parse_spice_subckt(FAN_SMC_SNIPPET)
    assert parsed.subckt_name.lower() == "fan_smc_pin_3"
    assert len(parsed.ports) == 5
    mosfets = [d for d in parsed.devices if d.device_type == "mosfet"]
    assert len(mosfets) == 5
    caps = [d for d in parsed.devices if d.device_type == "capacitor"]
    assert len(caps) == 1
    isrc = [d for d in parsed.devices if d.device_type == "current_source"]
    assert len(isrc) == 1


def test_model_class_mapping():
    parsed = parse_spice_subckt(FAN_SMC_SNIPPET)
    pmos_devs = [d for d in parsed.devices if d.model_class == "pmos_lv"]
    nmos_devs = [d for d in parsed.devices if d.model_class == "nmos_lv"]
    assert len(pmos_devs) == 4
    assert len(nmos_devs) == 1


def test_canonicalize_produces_text():
    parsed = parse_spice_subckt(FAN_SMC_SNIPPET)
    canon = canonicalize(parsed)
    assert len(canon.canonical_text) > 50
    assert ".subckt" in canon.canonical_text
    assert ".ends" in canon.canonical_text
    assert "VDD" in canon.canonical_text or "VDDA" in canon.canonical_text
    assert "GND" in canon.canonical_text or "GNDA" in canon.canonical_text


def test_net_map_has_all_nets():
    parsed = parse_spice_subckt(FAN_SMC_SNIPPET)
    canon = canonicalize(parsed)
    for net in parsed.nets:
        assert net in canon.net_map


def test_parse_and_canonicalize_full():
    result = parse_and_canonicalize(FAN_SMC_SNIPPET)
    assert result["subckt_name"].lower() == "fan_smc_pin_3"
    assert result["device_count"] == 7
    assert len(result["canonical_text"]) > 50
    assert result["canonical_text"].startswith("*")  # header comment


def test_parse_value():
    assert abs(_parse_value("3.4186871u") - 3.4186871e-6) < 1e-15
    assert abs(_parse_value("16.487924p") - 16.487924e-12) < 1e-20
    assert abs(_parse_value("1k") - 1000.0) < 1e-10
    assert abs(_parse_value("1meg") - 1e6) < 1.0
    assert _parse_value("0") == 0.0


def test_log_bin_bounds():
    assert _log_bin(0.1, 0.1, 20.0) == 0
    assert _log_bin(20.0, 0.1, 20.0) == 15
    assert 0 <= _log_bin(1.0, 0.1, 20.0) < 16


def test_params_binned():
    parsed = parse_spice_subckt(FAN_SMC_SNIPPET)
    for dev in parsed.devices:
        if dev.device_type == "mosfet":
            assert "w_bin" in dev.params_binned
            assert "l_bin" in dev.params_binned
            assert "m_bin" in dev.params_binned
