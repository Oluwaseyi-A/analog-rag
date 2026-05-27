"""MCP server exposing research agent + SPICE agent tools."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from src.mcp.tools.vector_search import (
    semantic_search_netlist,
    graph_similarity_search,
    get_circuit_record,
    list_topologies,
    rerank_by_relevance,
)
from src.mcp.tools.sql_query import (
    query_measurements,
    get_measurement,
    list_measurement_fields,
)
from src.mcp.tools.spice_runner import (
    run_spice_simulation,
    parse_netlist,
    build_circuit_graph,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

server = Server("analog-rag-mcp")


def _ok(data: Any) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


def _err(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps({"error": msg}))]


# ── Tool definitions ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="semantic_search_netlist",
            description="Embed a query string and search the analog_netlists ChromaDB collection. Returns circuit_id, score, topology, and netlist excerpt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or SPICE-style query about the circuit"},
                    "k": {"type": "integer", "default": 10, "description": "Number of results to return"},
                    "topology_filter": {"type": "string", "description": "Filter to a specific topology name"},
                    "stable_only": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="query_measurements",
            description="Query simulation measurements table with performance constraints. All values in SI base units.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gbp_min": {"type": "number", "description": "Min gain-bandwidth product (Hz)"},
                    "gbp_max": {"type": "number", "description": "Max gain-bandwidth product (Hz)"},
                    "phase_min": {"type": "number", "description": "Min phase margin (degrees)"},
                    "phase_max": {"type": "number", "description": "Max phase margin (degrees)"},
                    "dcgain_min": {"type": "number", "description": "Min DC gain (dB)"},
                    "power_max": {"type": "number", "description": "Max power consumption (W)"},
                    "sr_min": {"type": "number", "description": "Min slew rate (V/us)"},
                    "cmrr_min": {"type": "number", "description": "Max CMRR (dB, negative value — e.g. -60)"},
                    "stable_only": {"type": "boolean", "default": True},
                    "topology": {"type": "string"},
                    "scenario": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                },
            },
        ),
        types.Tool(
            name="graph_similarity_search",
            description="Find topologically similar circuits using GNN graph embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "circuit_id": {"type": "string", "description": "Reference circuit_id (topology/scenario/sample_id)"},
                    "k": {"type": "integer", "default": 10},
                },
                "required": ["circuit_id"],
            },
        ),
        types.Tool(
            name="get_circuit_record",
            description="Fetch all three entities (netlist, measurements, graph metadata) for a circuit_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "circuit_id": {"type": "string"},
                },
                "required": ["circuit_id"],
            },
        ),
        types.Tool(
            name="list_topologies",
            description="List all 17 known analog amplifier topology names with descriptions.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="rerank_by_relevance",
            description="Re-rank a list of candidate circuit_ids by relevance to a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "circuit_ids": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query", "circuit_ids"],
            },
        ),
        types.Tool(
            name="run_spice_simulation",
            description="Run ngspice batch simulation on a SPICE netlist. Returns measured performance metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "netlist_text": {"type": "string", "description": "Complete SPICE subcircuit text"},
                    "analysis": {"type": "string", "enum": ["acdc", "tran", "both"], "default": "acdc"},
                    "tb_params": {
                        "type": "object",
                        "description": "Testbench parameters (supply_voltage, PARAM_CLOAD, VCM_ratio, etc.)",
                    },
                },
                "required": ["netlist_text"],
            },
        ),
        types.Tool(
            name="parse_netlist",
            description="Parse and validate a SPICE subcircuit text. Returns device list and net connectivity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "netlist_text": {"type": "string"},
                },
                "required": ["netlist_text"],
            },
        ),
        types.Tool(
            name="build_circuit_graph",
            description="Parse netlist → build PyG heterogeneous graph → encode with GIN → return embedding.",
            inputSchema={
                "type": "object",
                "properties": {
                    "netlist_text": {"type": "string"},
                },
                "required": ["netlist_text"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "semantic_search_netlist":
            result = semantic_search_netlist(
                query=arguments["query"],
                k=arguments.get("k", 10),
                topology_filter=arguments.get("topology_filter"),
                stable_only=arguments.get("stable_only", True),
            )
        elif name == "query_measurements":
            result = query_measurements(
                gbp_min=arguments.get("gbp_min"),
                gbp_max=arguments.get("gbp_max"),
                phase_min=arguments.get("phase_min"),
                phase_max=arguments.get("phase_max"),
                dcgain_min=arguments.get("dcgain_min"),
                power_max=arguments.get("power_max"),
                sr_min=arguments.get("sr_min"),
                cmrr_min=arguments.get("cmrr_min"),
                stable_only=arguments.get("stable_only", True),
                topology=arguments.get("topology"),
                scenario=arguments.get("scenario"),
                limit=arguments.get("limit", 50),
            )
        elif name == "graph_similarity_search":
            result = graph_similarity_search(
                circuit_id=arguments["circuit_id"],
                k=arguments.get("k", 10),
            )
        elif name == "get_circuit_record":
            result = get_circuit_record(circuit_id=arguments["circuit_id"])
        elif name == "list_topologies":
            result = list_topologies()
        elif name == "rerank_by_relevance":
            result = rerank_by_relevance(
                query=arguments["query"],
                circuit_ids=arguments["circuit_ids"],
                top_k=arguments.get("top_k", 5),
            )
        elif name == "run_spice_simulation":
            result = run_spice_simulation(
                netlist_text=arguments["netlist_text"],
                analysis=arguments.get("analysis", "acdc"),
                tb_params=arguments.get("tb_params"),
            )
        elif name == "parse_netlist":
            result = parse_netlist(netlist_text=arguments["netlist_text"])
        elif name == "build_circuit_graph":
            result = build_circuit_graph(netlist_text=arguments["netlist_text"])
        else:
            return _err(f"Unknown tool: {name}")

        return _ok(result)

    except Exception as e:
        log.exception(f"Tool {name} raised an exception")
        return _err(str(e))


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
