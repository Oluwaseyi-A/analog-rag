"""CLI script: walk rag_data/, ingest all runs into ChromaDB + SQLite."""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Allow running as `python -m src.ingestion.ingest`
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ingestion.netlist_parser import load_netlist_file, parse_and_canonicalize
from src.database.sql_store import SQLStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def find_runs(rag_data_path: Path) -> list[dict]:
    """Walk rag_data/runs/<topology>/<scenario>/<sample_id>/ and collect run metadata."""
    runs_dir = rag_data_path / "runs"
    records = []
    for topo_dir in sorted(runs_dir.iterdir()):
        if not topo_dir.is_dir():
            continue
        for scenario_dir in sorted(topo_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            for sample_dir in sorted(scenario_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                result_file = sample_dir / "result_row.json"
                netlist_file = sample_dir / "netlist_resolved.sp"
                if not result_file.exists() or not netlist_file.exists():
                    log.warning(f"Skipping {sample_dir}: missing result_row.json or netlist_resolved.sp")
                    continue
                records.append({
                    "topology_name": topo_dir.name,
                    "scenario_name": scenario_dir.name,
                    "sample_id": sample_dir.name,
                    "sample_dir": sample_dir,
                    "result_file": result_file,
                    "netlist_file": netlist_file,
                    "circuit_id": f"{topo_dir.name}/{scenario_dir.name}/{sample_dir.name}",
                })
    return records


def ingest_run(
    record: dict,
    vs,
    sql: SQLStore,
    cfg: dict,
    topology_descriptions: dict,
    graph_store_path: Path,
    gin_weights: str = None,
    dry_run: bool = False,
) -> bool:
    circuit_id = record["circuit_id"]
    try:
        # 1. Load result_row.json
        with open(record["result_file"]) as f:
            result_row = json.load(f)
        result_row["topology_name"] = record["topology_name"]
        result_row["scenario_name"] = record["scenario_name"]
        result_row["sample_id"] = record["sample_id"]

        # 2. Parse + canonicalize netlist
        netlist_text = load_netlist_file(record["netlist_file"])
        parsed = parse_and_canonicalize(netlist_text, topology_descriptions)
        canonical_text = parsed["canonical_text"]

        if dry_run:
            log.info(f"[DRY RUN] Would ingest: {circuit_id}")
            return True

        # 3. Embed canonical text
        from src.ingestion.embedder import embed_text, embed_graph, build_behavior_embedding
        from src.ingestion.graph_builder import build_homogeneous_graph
        text_emb = embed_text(canonical_text, model=cfg.get("embedding_model", "text-embedding-3-large"))

        # 4. Build + embed graph
        homo_graph = build_homogeneous_graph(parsed)
        graph_emb = embed_graph(homo_graph, weights_path=gin_weights)

        # 5. Build behavior embedding
        beh_emb = build_behavior_embedding(result_row, dim=cfg.get("behavior_dim", 64))

        # 6. Save graph .pt file
        import torch
        graph_store_path.mkdir(parents=True, exist_ok=True)
        pt_path = graph_store_path / f"{circuit_id.replace('/', '__')}.pt"
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(homo_graph, str(pt_path))

        # 7. Metadata for vector store
        base_meta = {
            "topology_name": record["topology_name"],
            "scenario_name": record["scenario_name"],
            "sample_id": record["sample_id"],
            "circuit_id": circuit_id,
            "subckt_name": parsed.get("subckt_name", ""),
            "device_count": parsed.get("device_count", 0),
            "description": topology_descriptions.get(record["topology_name"], record["topology_name"]),
        }

        # 8. Upsert into all three ChromaDB collections
        vs.upsert_netlist(
            circuit_id=circuit_id,
            embedding=text_emb,
            canonical_text=canonical_text,
            metadata={**base_meta, "raw_netlist_path": str(record["netlist_file"])},
        )
        vs.upsert_graph(
            circuit_id=circuit_id,
            embedding=graph_emb,
            metadata={**base_meta, "pt_path": str(pt_path)},
        )
        vs.upsert_behavior(
            circuit_id=circuit_id,
            embedding=beh_emb,
            metadata={
                **base_meta,
                "dcgain": result_row.get("measured__dcgain"),
                "gbp": result_row.get("measured__gbp"),
                "phase_in_deg": result_row.get("measured__phase_in_deg"),
                "sr": result_row.get("measured__SR"),
                "power": result_row.get("measured__power"),
                "stable": result_row.get("measured__stable", 0),
            },
        )

        # 9. Upsert into SQLite
        sql.upsert(circuit_id, result_row)

        return True

    except Exception as e:
        log.error(f"Failed to ingest {circuit_id}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Ingest AnalogGym rag_data into ChromaDB + SQLite")
    parser.add_argument("--config", default="configs/ingest_config.json")
    parser.add_argument("--rag-data", help="Override rag_data path from config")
    parser.add_argument("--chroma-host", help="Override ChromaDB host")
    parser.add_argument("--chroma-port", type=int, help="Override ChromaDB port")
    parser.add_argument("--db-path", help="Override SQLite db path")
    parser.add_argument("--graph-store", help="Override graph .pt store path")
    parser.add_argument("--gin-weights", help="Path to trained GIN weights (.pt)")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no writes")
    parser.add_argument("--topology", help="Ingest only this topology name")
    parser.add_argument("--limit", type=int, help="Ingest at most N runs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    rag_data_path = Path(args.rag_data or cfg["rag_data_path"])
    chroma_host = args.chroma_host or cfg.get("chroma_host", "chromadb")
    chroma_port = args.chroma_port or cfg.get("chroma_port", 8000)
    db_path = args.db_path or cfg.get("db_path", "/app/data/measurements.db")
    graph_store = Path(args.graph_store or cfg.get("graph_store_path", "/app/data/graphs"))
    gin_weights = args.gin_weights or os.environ.get("GIN_WEIGHTS_PATH")
    topology_descriptions = cfg.get("topology_descriptions", {})

    log.info(f"rag_data: {rag_data_path}")
    log.info(f"ChromaDB: {chroma_host}:{chroma_port}")
    log.info(f"SQLite:   {db_path}")

    runs = find_runs(rag_data_path)
    if args.topology:
        runs = [r for r in runs if r["topology_name"] == args.topology]
    if args.limit:
        runs = runs[:args.limit]

    log.info(f"Found {len(runs)} runs to ingest")

    if not args.dry_run:
        from src.database.vector_store import VectorStore
        vs = VectorStore(host=chroma_host, port=chroma_port)
        sql = SQLStore(db_path)
    else:
        vs = None
        sql = None

    ok = 0
    fail = 0
    for record in tqdm(runs, desc="Ingesting"):
        success = ingest_run(
            record=record,
            vs=vs,
            sql=sql,
            cfg=cfg,
            topology_descriptions=topology_descriptions,
            graph_store_path=graph_store,
            gin_weights=gin_weights,
            dry_run=args.dry_run,
        )
        if success:
            ok += 1
        else:
            fail += 1

    if not args.dry_run:
        log.info(f"ChromaDB netlists: {vs.count_netlists()}")
        log.info(f"ChromaDB graphs:   {vs.count_graphs()}")
        log.info(f"ChromaDB behavior: {vs.count_behavior()}")
        log.info(f"SQLite rows:       {sql.count()}")

    log.info(f"Done. Success={ok}, Failures={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
