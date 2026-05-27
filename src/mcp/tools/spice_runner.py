"""MCP tool: ngspice batch runner and measurement extractor."""

import os
import re
import sys
import json
import math
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.ingestion.netlist_parser import parse_spice_subckt, load_netlist_file
from src.ingestion.graph_builder import build_homogeneous_graph
from src.ingestion.embedder import embed_graph
from src.ingestion.netlist_parser import parse_and_canonicalize

NGSPICE_BIN = os.environ.get("NGSPICE_BIN", "/usr/bin/ngspice")

# Template for AC+DC testbench — mirrors AnalogGym TB_Amplifier_ACDC.cir
ACDC_TEMPLATE = """\
* Auto-generated AC/DC testbench for {subckt_name}
.include {netlist_path}

.param mc_mm_switch=0
.param mc_pr_switch=0
.param supply_voltage={supply_v}
.param VCM_ratio={vcm_ratio}
.param PARAM_CLOAD={cload}

{pdk_include}

V1 vdda 0 'supply_voltage'
V2 gnda 0 0

Vindc opin 0 'supply_voltage*VCM_ratio'
Vin signal_in 0 dc 'supply_voltage*VCM_ratio' ac 1 sin('supply_voltage*VCM_ratio' 100m 500)

Lfb opout opout_dc 1T
Cin opout_dc signal_in 1T

Xop1 gnda vdda opout_dc opin opout  {subckt_name}
Cload1 opout 0 'PARAM_CLOAD'

xop2 gnda vdda cm2 cm1 cm3  {subckt_name}
Cload2 cm3 0 'PARAM_CLOAD'
vcmdc cm0 0 'supply_voltage*VCM_ratio'
vcmac1 cm1 cm0 0 ac=1
vcmac2 cm2 cm3 0 ac=1
.meas ac cmrrdc find vdb(cm3) at=0.1
.meas ac dcgain find vdb(opout) at=0.1
.meas ac gain_bandwidth_product when vdb(opout)=0
.meas ac phase_in_rad find vp(opout) when vdb(opout)=0
.meas ac phase_in_deg param='phase_in_rad*180/3.1416'

vgndapsrr gndpsrr 0 0 ac=1
xop3 gndpsrr vdda psrr3 vcm_psrr psrr_out  {subckt_name}
cload3 psrr_out 0 'PARAM_CLOAD'
vcm_psrr_src vcm_psrr 0 'supply_voltage*VCM_ratio'
.meas ac dcpsrn find vdb(psrr_out) at=0.1

vdda_psrp vdda_p 0 'supply_voltage' ac=1
xop4 gnda vdda_p psrp4 vcm_psrp psrp_out  {subckt_name}
cload4 psrp_out 0 'PARAM_CLOAD'
vcm_psrp_src vcm_psrp 0 'supply_voltage*VCM_ratio'
.meas ac dcpsrp find vdb(psrp_out) at=0.1

.op
.meas op ivdd25 i(V1)
.meas op vout25 v(opout)
.meas op vos25 param='vout25 - supply_voltage*VCM_ratio'

.ac dec 100 1 1000Meg
.end
"""

TRAN_TEMPLATE = """\
* Auto-generated transient testbench for {subckt_name}
.include {netlist_path}

.param mc_mm_switch=0
.param mc_pr_switch=0
.param supply_voltage={supply_v}
.param VCM_ratio={vcm_ratio}
.param PARAM_CLOAD={cload}
.param GBW_ideal={gbw_ideal}
.param STEP_TIME='10/GBW_ideal'
.param TRAN_SIM_TIME='20/GBW_ideal + 1e-6'
.param val0={val0}
.param val1={val1}

{pdk_include}

V1 vdda 0 'supply_voltage'
V2 gnda 0 0

Vinstep instep 0 pulse(val0 val1 1n 1n 1n 'TRAN_SIM_TIME/2' 'TRAN_SIM_TIME')
Lfb opout opout_dc 1T
Cin opout_dc instep 1T
Xop gnda vdda opout_dc instep opout  {subckt_name}
Cload opout 0 'PARAM_CLOAD'

.ic v(opout)='supply_voltage*VCM_ratio'
.tran 'STEP_TIME' 'TRAN_SIM_TIME'
.meas tran SR_rise deriv v(opout) when v(opout)='supply_voltage*VCM_ratio + (val1-val0)*0.5' rise=1
.meas tran SR_fall deriv v(opout) when v(opout)='supply_voltage*VCM_ratio + (val1-val0)*0.5' fall=1
.end
"""


def run_spice_simulation(
    netlist_text: str,
    analysis: str = "acdc",
    tb_params: Optional[dict] = None,
    pdk_root: Optional[str] = None,
) -> dict:
    """Write netlist + testbench to tempdir, run ngspice -b, parse .meas output.

    analysis: 'acdc' | 'tran' | 'both'
    Returns measurement dict matching result_row.json schema (measured__ keys).
    """
    if tb_params is None:
        tb_params = {}

    pdk_root = pdk_root or os.environ.get("PDK_ROOT", "/data/PDK")
    supply_v = tb_params.get("supply_voltage", 1.8)
    vcm_ratio = tb_params.get("VCM_ratio", 0.25)
    cload = tb_params.get("PARAM_CLOAD", "500p")
    gbw_ideal = tb_params.get("GBW_ideal", 50000.0)
    val0 = tb_params.get("val0", 0.3)
    val1 = tb_params.get("val1", 0.5)

    sky130_tt = os.path.join(pdk_root, "sky130_pdk/libs.tech/ngspice/corners/tt.spice")
    pdk_include = f".include {sky130_tt}" if os.path.exists(sky130_tt) else "* PDK not found"

    # Parse to get subckt_name
    parsed = parse_spice_subckt(netlist_text)
    subckt_name = parsed.subckt_name or "dut"

    results = {}

    with tempfile.TemporaryDirectory(prefix="analog_rag_") as tmpdir:
        netlist_path = os.path.join(tmpdir, f"{subckt_name}.sp")
        with open(netlist_path, "w") as f:
            f.write(netlist_text)

        if analysis in ("acdc", "both"):
            tb_text = ACDC_TEMPLATE.format(
                subckt_name=subckt_name,
                netlist_path=netlist_path,
                supply_v=supply_v,
                vcm_ratio=vcm_ratio,
                cload=cload,
                pdk_include=pdk_include,
            )
            acdc_path = os.path.join(tmpdir, "tb_acdc.cir")
            with open(acdc_path, "w") as f:
                f.write(tb_text)
            acdc_results = _run_ngspice(acdc_path, tmpdir)
            results.update(acdc_results)

        if analysis in ("tran", "both"):
            tb_text = TRAN_TEMPLATE.format(
                subckt_name=subckt_name,
                netlist_path=netlist_path,
                supply_v=supply_v,
                vcm_ratio=vcm_ratio,
                cload=cload,
                gbw_ideal=gbw_ideal,
                val0=val0,
                val1=val1,
                pdk_include=pdk_include,
            )
            tran_path = os.path.join(tmpdir, "tb_tran.cir")
            with open(tran_path, "w") as f:
                f.write(tb_text)
            tran_results = _run_ngspice(tran_path, tmpdir)
            results.update(tran_results)

    return _normalize_results(results)


def _run_ngspice(cir_path: str, workdir: str) -> dict:
    """Run ngspice -b on a .cir file and parse .meas output from stdout."""
    try:
        proc = subprocess.run(
            [NGSPICE_BIN, "-b", cir_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=workdir,
        )
        stdout = proc.stdout + proc.stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"error": "ngspice timeout", "return_code": -1}
    except FileNotFoundError:
        return {"error": f"ngspice not found at {NGSPICE_BIN}", "return_code": -1}

    measurements = _parse_meas_output(stdout)
    measurements["_ngspice_return_code"] = return_code
    measurements["_ngspice_stdout"] = stdout[-2000:]
    return measurements


def _parse_meas_output(output: str) -> dict:
    """Parse ngspice .meas output lines into a dict."""
    results = {}
    # Lines like: "dcgain               =  9.80593e+01"
    pattern = re.compile(r"^\s*(\w+)\s*=\s*([0-9eE+\-\.]+|failed)", re.MULTILINE | re.IGNORECASE)
    for match in pattern.finditer(output):
        key = match.group(1).lower()
        val_str = match.group(2).lower()
        if val_str == "failed":
            results[key] = None
        else:
            try:
                results[key] = float(val_str)
            except ValueError:
                results[key] = None
    return results


def _normalize_results(raw: dict) -> dict:
    """Map ngspice measurement names to result_row.json schema (measured__ prefix)."""
    mapping = {
        "dcgain": "measured__dcgain",
        "gain_bandwidth_product": "measured__gbp",
        "phase_in_deg": "measured__phase_in_deg",
        "phase_in_rad": "measured__phase_in_rad",
        "cmrrdc": "measured__cmrrdc",
        "dcpsrn": "measured__dcpsrn",
        "dcpsrp": "measured__dcpsrp",
        "ivdd25": "measured__ivdd25",
        "vout25": "measured__vout25",
        "vos25": "measured__vos25",
        "sr_rise": "measured__sr_rise",
        "sr_fall": "measured__sr_fall",
    }
    out = {}
    for src, dst in mapping.items():
        if src in raw and raw[src] is not None:
            out[dst] = raw[src]

    # Derived
    gbp = out.get("measured__gbp")
    power_raw = raw.get("ivdd25")
    if power_raw is not None:
        out["measured__power"] = abs(power_raw) * 1.8

    stable = 1
    if out.get("measured__dcgain") is None or out.get("measured__gbp") is None:
        stable = 0
    if out.get("measured__phase_in_deg") is not None and abs(out["measured__phase_in_deg"]) < 10:
        stable = 0
    out["measured__stable"] = stable

    out["_raw"] = {k: v for k, v in raw.items() if not k.startswith("_")}
    out["_ngspice_return_code"] = raw.get("_ngspice_return_code", -1)
    return out


def parse_netlist(netlist_text: str) -> dict:
    """Parse and validate a SPICE netlist, returning device list + net list."""
    parsed = parse_and_canonicalize(netlist_text)
    return {
        "subckt_name": parsed["subckt_name"],
        "ports": parsed["ports"],
        "device_count": parsed["device_count"],
        "devices": parsed["devices"],
        "canonical_text": parsed["canonical_text"],
        "valid": parsed["device_count"] > 0,
    }


def build_circuit_graph(netlist_text: str, gin_weights: Optional[str] = None) -> dict:
    """Parse netlist → build PyG graph → embed with GIN encoder."""
    parsed = parse_and_canonicalize(netlist_text)
    homo = build_homogeneous_graph(parsed)
    emb = embed_graph(homo, weights_path=gin_weights)
    return {
        "subckt_name": parsed["subckt_name"],
        "device_count": parsed["device_count"],
        "node_count": homo.x.shape[0] if homo.x is not None else 0,
        "edge_count": homo.edge_index.shape[1] if homo.edge_index is not None else 0,
        "embedding": emb,
        "embedding_dim": len(emb),
    }
