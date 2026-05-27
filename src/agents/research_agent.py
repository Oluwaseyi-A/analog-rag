"""Research agent: retrieval-optimized agent with query rewriting and reranking."""

import os
import json
import logging
from typing import Any, Optional

import litellm
from litellm import completion

log = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are an analog circuit retrieval agent. Your ONLY job is to retrieve the most relevant circuits from the knowledge base.

You have access to these tools:
- semantic_search_netlist: Search by netlist text similarity
- query_measurements: Filter by performance specs (GBP, phase margin, gain, power, etc.)
- graph_similarity_search: Find topologically similar circuits
- get_circuit_record: Fetch full details for a circuit_id
- list_topologies: List all 17 known topology families
- rerank_by_relevance: Re-rank candidates by relevance score

Strategy:
1. Rewrite the user query into the most retrieval-effective form for each tool
2. Issue semantic search first, then SQL filter to narrow by specs
3. Use graph_similarity_search to expand or re-rank candidates
4. Return the top-K most relevant circuit records with supporting evidence
5. NEVER hallucinate circuit details — only return what is in the knowledge base
6. Cite every circuit by its circuit_id (topology_name/scenario/sample_id)

Output format: JSON with keys "circuits" (list) and "retrieval_summary" (string).
Each circuit: {circuit_id, topology_name, score, key_metrics, evidence}
"""


def build_mcp_tools_spec() -> list[dict]:
    """Build LiteLLM tool specs for research agent (non-MCP path — direct function calls)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "semantic_search_netlist",
                "description": "Embed query and search analog_netlists by text similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                        "topology_filter": {"type": "string"},
                        "stable_only": {"type": "boolean", "default": True},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_measurements",
                "description": "SQL filter on simulation measurements table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gbp_min": {"type": "number"},
                        "gbp_max": {"type": "number"},
                        "phase_min": {"type": "number"},
                        "dcgain_min": {"type": "number"},
                        "power_max": {"type": "number"},
                        "stable_only": {"type": "boolean", "default": True},
                        "topology": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_similarity_search",
                "description": "Find topologically similar circuits by GNN graph embedding cosine similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "circuit_id": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["circuit_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_circuit_record",
                "description": "Fetch full circuit record: netlist + measurements + graph metadata",
                "parameters": {
                    "type": "object",
                    "properties": {"circuit_id": {"type": "string"}},
                    "required": ["circuit_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_topologies",
                "description": "List all 17 topology families with descriptions",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _dispatch_tool(name: str, args: dict) -> Any:
    """Call the actual tool function (direct import path, no MCP overhead)."""
    from src.mcp.tools.vector_search import (
        semantic_search_netlist,
        graph_similarity_search,
        get_circuit_record,
        list_topologies,
        rerank_by_relevance,
    )
    from src.mcp.tools.sql_query import query_measurements

    dispatch = {
        "semantic_search_netlist": semantic_search_netlist,
        "query_measurements": query_measurements,
        "graph_similarity_search": graph_similarity_search,
        "get_circuit_record": get_circuit_record,
        "list_topologies": list_topologies,
        "rerank_by_relevance": rerank_by_relevance,
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


def run_research_agent(
    query: str,
    model: Optional[str] = None,
    max_tool_calls: int = 8,
) -> dict:
    """Run the research agent with tool-calling loop.

    Returns dict with 'circuits', 'retrieval_summary', 'tool_calls_used'.
    """
    model = model or os.environ.get("RESEARCH_AGENT_MODEL", "claude-sonnet-4-6")
    tools = build_mcp_tools_spec()
    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"Find the most relevant circuits for this query:\n\n{query}"},
    ]

    tool_calls_used = 0
    all_tool_results = []

    for _ in range(max_tool_calls):
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        if not resp or not resp.choices:
            return {"circuits": [], "retrieval_summary": "LLM returned empty response.", "tool_calls_used": tool_calls_used}
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Final answer
            content = msg.content or ""
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"retrieval_summary": content, "circuits": []}
            result["tool_calls_used"] = tool_calls_used
            result["_raw_tool_results"] = all_tool_results
            return result

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for tc in msg.tool_calls:
            tool_calls_used += 1
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            try:
                tool_result = _dispatch_tool(fn_name, args)
            except Exception as e:
                tool_result = {"error": f"Tool {fn_name} failed: {e}"}
            all_tool_results.append({"tool": fn_name, "args": args, "result": tool_result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, default=str),
            })

    return {
        "circuits": [],
        "retrieval_summary": "Max tool calls reached without final answer.",
        "tool_calls_used": tool_calls_used,
        "_raw_tool_results": all_tool_results,
    }
