"""SPICE simulation agent: testbench generation, simulation, and comparison."""

import os
import json
import logging
from typing import Any, Optional

import litellm
from litellm import completion

log = logging.getLogger(__name__)

SPICE_SYSTEM_PROMPT = """\
You are an analog circuit simulation agent specializing in SPICE simulation.

Your tasks:
1. Given a SPICE subcircuit netlist, build or adapt an appropriate testbench
2. Run AC/DC and transient analysis via ngspice batch mode
3. Extract measurements matching the result_row.json schema
4. Compare simulated FoMs to the closest KB circuit and summarize differences

You have access to:
- parse_netlist: Validate SPICE and extract device/net info
- run_spice_simulation: Run ngspice -b, return measured performance metrics
- build_circuit_graph: Build GNN graph embedding from netlist
- semantic_search_netlist: Find similar circuits in KB for comparison
- get_circuit_record: Fetch full KB record for comparison

Rules:
- ALWAYS use ngspice batch mode (run_spice_simulation handles this automatically)
- Extract: dcgain, gbp, phase_in_deg, power, stable as minimum required metrics
- If simulation fails (ngspice_return_code != 0), report the error and attempt diagnosis
- Compare simulated results to KB nearest-neighbor and report percentage differences
- Cite KB circuits by circuit_id

Output JSON with keys:
  simulation_results: {measured metrics}
  comparison: {kb_circuit_id, metric_diffs}
  diagnosis: string (if simulation failed)
  synthesis: string (summary for superagent)
"""


def _dispatch_tool(name: str, args: dict) -> Any:
    from src.mcp.tools.spice_runner import run_spice_simulation, parse_netlist, build_circuit_graph
    from src.mcp.tools.vector_search import semantic_search_netlist, get_circuit_record

    dispatch = {
        "run_spice_simulation": run_spice_simulation,
        "parse_netlist": parse_netlist,
        "build_circuit_graph": build_circuit_graph,
        "semantic_search_netlist": semantic_search_netlist,
        "get_circuit_record": get_circuit_record,
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


def _build_tools_spec() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "parse_netlist",
                "description": "Parse and validate a SPICE subcircuit text",
                "parameters": {
                    "type": "object",
                    "properties": {"netlist_text": {"type": "string"}},
                    "required": ["netlist_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_spice_simulation",
                "description": "Run ngspice simulation on the netlist. Returns measured performance metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "netlist_text": {"type": "string"},
                        "analysis": {"type": "string", "enum": ["acdc", "tran", "both"], "default": "acdc"},
                        "tb_params": {"type": "object"},
                    },
                    "required": ["netlist_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_circuit_graph",
                "description": "Build GNN graph embedding from netlist for similarity search",
                "parameters": {
                    "type": "object",
                    "properties": {"netlist_text": {"type": "string"}},
                    "required": ["netlist_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search_netlist",
                "description": "Find similar KB circuits by text similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_circuit_record",
                "description": "Get full KB record for comparison",
                "parameters": {
                    "type": "object",
                    "properties": {"circuit_id": {"type": "string"}},
                    "required": ["circuit_id"],
                },
            },
        },
    ]


def run_spice_agent(
    netlist_text: str,
    analysis: str = "acdc",
    tb_params: Optional[dict] = None,
    model: Optional[str] = None,
    max_tool_calls: int = 10,
) -> dict:
    """Run the SPICE simulation agent.

    Returns dict with simulation_results, comparison, synthesis.
    """
    model = model or os.environ.get("SPICE_AGENT_MODEL", "claude-sonnet-4-6")
    tools = _build_tools_spec()

    user_content = (
        f"Simulate this SPICE netlist and compare it to the nearest KB circuit.\n"
        f"Analysis type: {analysis}\n"
        f"TB params: {json.dumps(tb_params or {})}\n\n"
        f"```spice\n{netlist_text[:3000]}\n```"
    )

    messages = [
        {"role": "system", "content": SPICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    tool_calls_used = 0

    for _ in range(max_tool_calls):
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"synthesis": content, "simulation_results": {}, "comparison": {}}
            result["tool_calls_used"] = tool_calls_used
            return result

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for tc in msg.tool_calls:
            tool_calls_used += 1
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_result = _dispatch_tool(fn_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, default=str),
            })

    return {
        "simulation_results": {},
        "comparison": {},
        "synthesis": "Max tool calls reached.",
        "tool_calls_used": tool_calls_used,
    }


def compare_to_kb(circuit_id: str, simulated_metrics: dict) -> dict:
    """Direct comparison between simulated metrics and a KB circuit's stored measurements."""
    from src.database.sql_store import SQLStore
    import os
    sql = SQLStore(os.environ.get("DB_PATH", "/app/data/measurements.db"))
    kb = sql.get_raw(circuit_id)
    if kb is None:
        return {"error": f"circuit_id {circuit_id!r} not found"}

    key_metrics = ["dcgain", "gbp", "phase_in_deg", "power", "stable"]
    diffs = {}
    for metric in key_metrics:
        sim_key = f"measured__{metric}"
        kb_key = f"measured__{metric}"
        sim_val = simulated_metrics.get(sim_key)
        kb_val = kb.get(kb_key)
        if sim_val is not None and kb_val is not None and kb_val != 0:
            pct_diff = (sim_val - kb_val) / abs(kb_val) * 100
            diffs[metric] = {
                "simulated": sim_val,
                "kb_stored": kb_val,
                "pct_diff": round(pct_diff, 2),
            }

    return {
        "circuit_id": circuit_id,
        "metric_diffs": diffs,
        "within_1pct": all(abs(v["pct_diff"]) < 1.0 for v in diffs.values()),
    }
