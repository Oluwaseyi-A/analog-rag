"""MCP tool implementations: semantic netlist search + graph similarity search."""

import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.ingestion.embedder import embed_text, embed_graph, build_behavior_embedding
from src.database.vector_store import VectorStore
from src.database.sql_store import SQLStore


def _get_vs() -> VectorStore:
    host = os.environ.get("CHROMA_HOST", "chromadb")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    return VectorStore(host=host, port=port)


def _get_sql() -> SQLStore:
    db_path = os.environ.get("DB_PATH", "/app/data/measurements.db")
    return SQLStore(db_path)


def semantic_search_netlist(
    query: str,
    k: int = 10,
    topology_filter: Optional[str] = None,
    stable_only: bool = True,
) -> list[dict]:
    """Embed query string and search analog_netlists collection.

    Returns list of dicts with circuit_id, score, topology, and netlist excerpt.
    """
    emb = embed_text(query)
    vs = _get_vs()

    where = None
    conditions = []
    if topology_filter:
        conditions.append({"topology_name": {"$eq": topology_filter}})
    if stable_only:
        # Filter using SQL store instead — ChromaDB metadata doesn't have stable flag reliably
        pass

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = vs.search_netlists(query_embedding=emb, k=k, where=where)

    # Optionally filter stable circuits via SQL cross-reference
    if stable_only and results:
        sql = _get_sql()
        stable_ids = {r["circuit_id"] for r in sql.query("stable = 1", limit=10000)}
        results = [r for r in results if r["circuit_id"] in stable_ids]

    output = []
    for r in results:
        meta = r.get("metadata", {})
        doc = r.get("document", "")
        excerpt = doc[:300] if doc else ""
        output.append({
            "circuit_id": r["circuit_id"],
            "score": round(r["score"], 4),
            "topology_name": meta.get("topology_name", ""),
            "scenario_name": meta.get("scenario_name", ""),
            "description": meta.get("description", ""),
            "netlist_excerpt": excerpt,
        })
    return output


def graph_similarity_search(
    circuit_id: str,
    k: int = 10,
) -> list[dict]:
    """Find circuits similar to the given circuit_id using GNN graph embeddings.

    Loads the stored graph embedding from ChromaDB and does cosine search.
    """
    vs = _get_vs()
    ref = vs.get_graph(circuit_id)
    if ref is None:
        return [{"error": f"circuit_id {circuit_id!r} not found in graph collection"}]

    ref_emb = ref["embedding"]
    if ref_emb is None:
        return [{"error": f"No graph embedding stored for {circuit_id!r} — run ingestion first"}]
    results = vs.search_graphs(query_embedding=ref_emb, k=k + 1)
    # Exclude self
    results = [r for r in results if r["circuit_id"] != circuit_id][:k]

    output = []
    for r in results:
        meta = r.get("metadata", {})
        output.append({
            "circuit_id": r["circuit_id"],
            "score": round(r["score"], 4),
            "topology_name": meta.get("topology_name", ""),
            "scenario_name": meta.get("scenario_name", ""),
        })
    return output


def get_circuit_record(circuit_id: str) -> dict:
    """Fetch all three entities (netlist, measurements, graph metadata) for a circuit_id."""
    vs = _get_vs()
    sql = _get_sql()

    netlist = vs.get_netlist(circuit_id)
    graph_meta = vs.get_graph(circuit_id)
    measurements = sql.get_raw(circuit_id)

    if netlist is None and graph_meta is None and measurements is None:
        return {"error": f"circuit_id {circuit_id!r} not found in any store"}

    return {
        "circuit_id": circuit_id,
        "netlist": {
            "canonical_text": netlist["document"] if netlist else None,
            "metadata": netlist["metadata"] if netlist else {},
        } if netlist else None,
        "measurements": _clean_measurements(measurements) if measurements else None,
        "graph": {
            "topology_name": graph_meta["metadata"].get("topology_name") if graph_meta else None,
            "device_count": graph_meta["metadata"].get("device_count") if graph_meta else None,
            "pt_path": graph_meta["metadata"].get("pt_path") if graph_meta else None,
        } if graph_meta else None,
    }


def list_topologies(descriptions: dict = None) -> list[dict]:
    """List all known topology names with descriptions."""
    sql = _get_sql()
    topos = sql.list_topologies()

    default_desc = {
        "Alfio_RAFFC_Pin_3": "Recycling amplifier with feed-forward compensation",
        "Fan_SMC_Pin_3": "Single Miller compensation OTA with feed-forward",
        "HoiLee_AFFC_Pin_3": "Active feed-forward compensation amplifier",
        "Leung_DFCFC1_Pin_3": "Damping-factor-control frequency compensation type 1",
        "Leung_DFCFC2_Pin_3": "Damping-factor-control frequency compensation type 2",
        "Leung_NMCF_Pin_3": "Nested Miller compensation with feedforward",
        "Leung_NMCNR_Pin_3": "Nested Miller compensation with null resistor",
        "Peng_ACBC_Pin_3": "Active cascaded bandwidth compensation amplifier",
        "Peng_IAC_Pin_3": "Indirect active compensation amplifier",
        "Peng_TCFC_Pin_3": "Transconductance-cancellation frequency compensation",
        "Qu2017_AZC_Pin_3": "Active zero compensation amplifier (2017)",
        "Qu_LEC_Pin_3": "Low-power embedded compensation amplifier",
        "Ramos_PFC_Pin_3": "Push-pull frequency compensation OTA",
        "Sau_CFCC_Pin_3": "Cascode frequency compensation cascaded",
        "Song_DACFC_Pin_3": "Dual-path active capacitance frequency compensation",
        "Tan_CLIA_Pin_3": "Composite local impedance attenuation amplifier",
        "Yan_AZ_Pin_3": "Active zero compensation with push-pull output",
    }
    desc = descriptions or default_desc

    return [
        {"topology_name": t, "description": desc.get(t, t), "in_kb": t in topos}
        for t in sorted(set(list(default_desc.keys()) + topos))
    ]


def rerank_by_relevance(query: str, circuit_ids: list[str], top_k: int = 5) -> list[dict]:
    """LLM-free reranker: re-score candidates by second-pass semantic similarity.

    Uses behavior + text cosine similarity for a simple linear rerank.
    """
    vs = _get_vs()
    query_emb = embed_text(query)

    scored = []
    for cid in circuit_ids:
        net = vs.get_netlist(cid)
        if net is None or net.get("embedding") is None:
            # Fall back to search result order
            scored.append({"circuit_id": cid, "rerank_score": 0.5})
            continue

        # Recompute similarity via fresh query
        results = vs.search_netlists(query_emb, k=len(circuit_ids) + 5)
        id_to_score = {r["circuit_id"]: r["score"] for r in results}
        scored.append({
            "circuit_id": cid,
            "rerank_score": id_to_score.get(cid, 0.0),
        })

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def _clean_measurements(m: dict) -> dict:
    """Return only the measured__ fields from a raw result_row dict."""
    if m is None:
        return {}
    return {k: v for k, v in m.items() if k.startswith("measured__") or k in ("topology_name", "scenario_name", "sample_id", "circuit_id")}
