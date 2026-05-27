"""Streamlit UI: chat interface for the analog circuit RAG system."""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

import streamlit as st
import plotly.express as px
import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/app/data/measurements.db")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

st.set_page_config(
    page_title="Analog Circuit RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached resource inits ─────────────────────────────────────────────────────

@st.cache_resource
def get_sql_store():
    try:
        from src.database.sql_store import SQLStore
        return SQLStore(DB_PATH)
    except Exception:
        return None


@st.cache_resource
def get_vector_store():
    try:
        from src.database.vector_store import VectorStore
        return VectorStore(host=CHROMA_HOST, port=CHROMA_PORT)
    except Exception:
        return None


# ── Helper rendering functions ────────────────────────────────────────────────

def display_retrieved_circuits(raw_results: list, sql):
    all_circuits = []
    for r in raw_results:
        res = r.get("result", {})
        circuits = res.get("circuits", [])
        all_circuits.extend(circuits)

    if not all_circuits:
        st.info("No circuits retrieved yet.")
        return

    for circ in all_circuits[:5]:
        cid = circ.get("circuit_id", "unknown")
        with st.expander(f"📋 {cid}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Topology:** {circ.get('topology_name', 'N/A')}")
                st.markdown(f"**Score:** {circ.get('score', 0):.4f}")
                st.markdown(f"**Evidence:** {circ.get('evidence', 'N/A')}")
            with col2:
                if sql:
                    try:
                        row = sql.get(cid)
                        if row:
                            st.metric("GBP (Hz)", f"{row.get('gbp', 0):.2e}")
                            st.metric("Phase (°)", f"{row.get('phase_in_deg', 0):.1f}")
                            st.metric("DC Gain (dB)", f"{row.get('dcgain', 0):.1f}")
                    except Exception:
                        pass


def display_performance_chart(citations: list, sql):
    if not citations or not sql:
        st.info("No citations to chart.")
        return

    rows = []
    for cid in citations[:5]:
        try:
            row = sql.get(cid)
            if row:
                rows.append({
                    "circuit_id": cid,
                    "GBP (MHz)": (row.get("gbp") or 0) / 1e6,
                    "Phase (°)": row.get("phase_in_deg") or 0,
                    "DC Gain (dB)": row.get("dcgain") or 0,
                    "Power (µW)": (row.get("power") or 0) * 1e6,
                    "FoML": row.get("foml") or 0,
                })
        except Exception:
            pass

    if not rows:
        st.info("No measurement data available for cited circuits.")
        return

    df = pd.DataFrame(rows).set_index("circuit_id")
    fig = px.bar(
        df.reset_index().melt(id_vars="circuit_id"),
        x="variable", y="value", color="circuit_id",
        barmode="group",
        title="Performance Comparison",
        labels={"value": "Value", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "last_raw_results" not in st.session_state:
    st.session_state.last_raw_results = []

sql = get_sql_store()
vs = get_vector_store()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Analog Circuit RAG")
    st.markdown("---")

    st.subheader("Knowledge Base")
    if sql:
        try:
            n_circuits = sql.count()
            topologies = sql.list_topologies()
            st.metric("Circuits", n_circuits)
            st.metric("Topologies", len(topologies))
        except Exception:
            st.warning("SQLite not yet populated — run ingestion first.")
    else:
        st.warning("SQLite store not available.")

    if vs:
        try:
            n_net = vs.count_netlists()
            st.metric("ChromaDB netlists", n_net)
        except Exception:
            st.warning("ChromaDB not reachable.")

    st.markdown("---")

    st.subheader("Topology Browser")
    if sql:
        try:
            topos = sql.list_topologies()
            if topos:
                selected_topo = st.selectbox("Select topology", [""] + topos)
                if selected_topo:
                    rows = sql.query(f"topology_name = '{selected_topo}'", limit=20)
                    if rows:
                        df = pd.DataFrame(rows)[["circuit_id", "scenario_name", "dcgain", "gbp", "phase_in_deg", "stable"]]
                        st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"Topology browser error: {e}")

    st.markdown("---")

    st.subheader("Model Settings")
    model_choice = st.selectbox(
        "Superagent model",
        ["claude-sonnet-4-6", "claude-opus-4-7", "gpt-4o", "gemini/gemini-2.0-flash"],
        index=0,
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.session_state.last_raw_results = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("Analog Circuit Assistant")
st.markdown("Ask about amplifier topologies, performance specs, or paste a SPICE netlist for simulation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about analog circuits or paste a SPICE netlist..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            t0 = time.time()
            try:
                from src.agents.superagent import run_superagent
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ][-10:]
                result = run_superagent(
                    user_query=prompt,
                    model=model_choice,
                    history=history,
                )
                elapsed = time.time() - t0
                answer = result.get("answer", "No answer generated.")
                citations = result.get("citations", [])
                raw_results = result.get("raw_agent_results", [])
                st.session_state.last_citations = citations
                st.session_state.last_raw_results = raw_results
                st.markdown(answer)
                if citations:
                    st.markdown("**Sources:**")
                    for cid in citations:
                        st.markdown(f"- `{cid}`")
                st.caption(f"Responded in {elapsed:.1f}s · {result.get('tool_calls_used', 0)} tool calls")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
                log.exception("Superagent error")

# ── Results panel ─────────────────────────────────────────────────────────────

if st.session_state.last_citations or st.session_state.last_raw_results:
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Retrieved Circuits", "Performance Chart", "Raw Agent Data"])

    with tab1:
        display_retrieved_circuits(st.session_state.last_raw_results, sql)

    with tab2:
        display_performance_chart(st.session_state.last_citations, sql)

    with tab3:
        st.json(st.session_state.last_raw_results)
