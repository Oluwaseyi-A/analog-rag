"""LLM superagent with MCP client: routes queries to research or SPICE agent."""

import os
import json
import logging
from typing import Any, Optional

import litellm
from litellm import completion

log = logging.getLogger(__name__)

SUPERAGENT_SYSTEM_PROMPT = """\
You are an expert analog circuit design assistant powered by a knowledge base of 17 amplifier topologies.

You have access to two specialized agents:
1. **research_agent**: Retrieves circuits from the knowledge base by semantic search, SQL filter, and graph similarity. Use for topology matching, performance comparison queries.
2. **spice_agent**: Simulates a user-provided SPICE netlist and compares to KB circuits. Use for circuit analysis of netlists not in the KB.

Decision rules:
- "Show me a folded-cascode OTA with GBW > 1 MHz" → research_agent (topology + performance query)
- "Find circuits similar to this netlist: ..." → both agents (research + SPICE sim)
- "What is the phase margin of Fan_SMC_Pin_3?" → research_agent
- "Simulate this netlist I designed" → spice_agent
- "Compare my design to KB circuits" → spice_agent

When answering:
- Cite source circuits by circuit_id (topology_name/scenario/sample_id)
- Report measured values from KB, never invent numbers
- Summarize topology differences in plain English
- If no circuits match, say so clearly
"""


def _build_superagent_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "call_research_agent",
                "description": "Retrieve circuits from KB using semantic, SQL, and graph search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Retrieval query for the research agent"},
                        "max_tool_calls": {"type": "integer", "default": 8},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "call_spice_agent",
                "description": "Simulate a SPICE netlist and compare to KB circuits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "netlist_text": {"type": "string", "description": "Complete SPICE subcircuit text"},
                        "analysis": {"type": "string", "enum": ["acdc", "tran", "both"], "default": "acdc"},
                        "tb_params": {"type": "object"},
                    },
                    "required": ["netlist_text"],
                },
            },
        },
    ]


def _dispatch_superagent_tool(name: str, args: dict) -> Any:
    if name == "call_research_agent":
        from src.agents.research_agent import run_research_agent
        return run_research_agent(
            query=args["query"],
            max_tool_calls=args.get("max_tool_calls", 8),
        )
    elif name == "call_spice_agent":
        from src.agents.spice_agent import run_spice_agent
        return run_spice_agent(
            netlist_text=args["netlist_text"],
            analysis=args.get("analysis", "acdc"),
            tb_params=args.get("tb_params"),
        )
    else:
        return {"error": f"Unknown tool: {name}"}


def run_superagent(
    user_query: str,
    model: Optional[str] = None,
    max_rounds: int = 6,
    history: Optional[list] = None,
) -> dict:
    """Run the LLM superagent with research + SPICE tool delegation.

    Returns: {answer: str, citations: list[str], tool_calls_used: int, raw_agent_results: list}
    """
    model = model or os.environ.get("SUPERAGENT_MODEL", "claude-sonnet-4-6")
    tools = _build_superagent_tools()

    messages = [{"role": "system", "content": SUPERAGENT_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    tool_calls_used = 0
    raw_results = []

    for _ in range(max_rounds):
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            citations = _extract_citations(raw_results)
            return {
                "answer": answer,
                "citations": citations,
                "tool_calls_used": tool_calls_used,
                "raw_agent_results": raw_results,
            }

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for tc in msg.tool_calls:
            tool_calls_used += 1
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            try:
                result = _dispatch_superagent_tool(fn_name, args)
            except Exception as e:
                result = {"error": f"Agent {fn_name} failed: {e}"}
            raw_results.append({"agent": fn_name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str)[:8000],
            })

    return {
        "answer": "Max rounds reached without a final answer.",
        "citations": [],
        "tool_calls_used": tool_calls_used,
        "raw_agent_results": raw_results,
    }


def _extract_citations(raw_results: list) -> list[str]:
    """Pull circuit_ids from agent results for citation tracking."""
    cited = []
    for r in raw_results:
        result = r.get("result", {})
        circuits = result.get("circuits", [])
        for c in circuits:
            cid = c.get("circuit_id")
            if cid and cid not in cited:
                cited.append(cid)
        # Also from simulation comparison
        kb_cid = result.get("comparison", {}).get("kb_circuit_id")
        if kb_cid and kb_cid not in cited:
            cited.append(kb_cid)
    return cited
