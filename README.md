# Analog Circuit RAG

An **Agentic Retrieval-Augmented Generation (RAG)** system for analog circuit netlists, built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). Ask questions about amplifier topologies, filter circuits by performance specs, or paste a SPICE netlist for live simulation — all through a single chat interface.

The knowledge base covers **17 multi-stage amplifier topologies** from the [AnalogGym](https://github.com/CODA-Lab/AnalogGym) benchmark suite, simulated on the SkyWater 130nm PDK.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI  :8501                            │
│            chat input · topology browser · performance charts           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ user query
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          LLM Superagent                                 │
│                     (LiteLLM → Claude / GPT)                            │
│                                                                         │
│  Parses intent → decides retrieval strategy → synthesizes final answer  │
│  Cites every circuit by circuit_id  (topology/scenario/sample_id)       │
└──────────────┬──────────────────────────────────┬───────────────────────┘
               │ topology / performance query      │ "simulate this netlist"
               ▼                                   ▼
┌──────────────────────────┐         ┌─────────────────────────────────┐
│     Research Agent       │         │         SPICE Agent             │
│  (retrieval-optimised)   │         │    (simulation-optimised)       │
│                          │         │                                 │
│  1. Rewrite query for    │         │  1. parse_netlist()             │
│     each tool            │         │  2. run_spice_simulation()      │
│  2. semantic_search      │         │     → ngspice -b (batch only)   │
│  3. query_measurements   │         │  3. semantic_search() to find   │
│  4. graph_similarity     │         │     nearest KB circuit          │
│  5. get_circuit_record   │         │  4. compare_to_kb() → FoM diff  │
│  6. rerank_by_relevance  │         │  5. return structured report    │
└─────────────┬────────────┘         └──────────────┬──────────────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │  MCP tool calls
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MCP Server  :8001                              │
│                    (9 tools exposed over stdio)                         │
├──────────────────────────┬──────────────────────────────────────────────┤
│   Research tools         │   Simulation tools                           │
│  ─────────────────────   │  ──────────────────────────────────────────  │
│  semantic_search_netlist │  run_spice_simulation                        │
│  query_measurements      │  parse_netlist                               │
│  graph_similarity_search │  build_circuit_graph                         │
│  get_circuit_record      │                                              │
│  list_topologies         │                                              │
│  rerank_by_relevance     │                                              │
└──────────┬───────────────┴──────────────────────┬───────────────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────────────┐    ┌─────────────────────────────────────┐
│       Knowledge Base         │    │        ngspice  (batch mode)        │
│                              │    │                                     │
│  ChromaDB  (3 collections)   │    │  Auto-generated AC/DC + tran TB     │
│  ┌────────────────────────┐  │    │  .meas output → measured__ fields   │
│  │ analog_netlists        │  │    │  Matches result_row.json schema     │
│  │  text-embedding-3-large│  │    └─────────────────────────────────────┘
│  │  3072-dim cosine ANN   │  │
│  ├────────────────────────┤  │
│  │ analog_graphs          │  │
│  │  GIN encoder 256-dim   │  │
│  │  contrastive trained   │  │
│  ├────────────────────────┤  │
│  │ analog_behavior        │  │
│  │  64-dim feature vector │  │
│  │  (gain, GBW, PM, SR…)  │  │
│  └────────────────────────┘  │
│                              │
│  SQLite  measurements.db     │
│  ─────────────────────────── │
│  dcgain · gbp · phase · SR   │
│  power · area · CMRR · PSRR  │
│  foml · foms · settling_time │
└──────────────────────────────┘
```

---

## How the Agents Connect

### 1 · Superagent (orchestrator)
`src/agents/superagent.py`

The superagent is the only component that talks to the user. It receives the raw query and decides which specialist to call:

| Query type | Routes to |
|---|---|
| "Find an OTA with GBW > 1 MHz" | Research Agent |
| "Show me circuits like Fan_SMC_Pin_3" | Research Agent |
| "Simulate this netlist I designed" | SPICE Agent |
| "Compare my design to the KB" | SPICE Agent → Research Agent |

It runs a tool-calling loop (up to 6 rounds), accumulates citations from both agents, then synthesises a final answer.

### 2 · Research Agent (retrieval)
`src/agents/research_agent.py`

Prompted purely for retrieval — it never tries to answer the user directly. Its loop:

```
user query
    │
    ├─► semantic_search_netlist()   ← rewritten query, k=10
    │       ChromaDB cosine ANN on canonical netlist text
    │
    ├─► query_measurements()        ← extracted numeric constraints
    │       SQLite WHERE clause (gbp, phase, dcgain, power…)
    │
    ├─► graph_similarity_search()   ← top candidate circuit_id
    │       GIN embedding cosine ANN on analog_graphs
    │
    ├─► rerank_by_relevance()       ← merged candidate list
    │       second-pass semantic re-scoring
    │
    └─► get_circuit_record()        ← final circuit_ids
            returns netlist + measurements + graph metadata
```

Returns structured JSON: `{circuits: [...], retrieval_summary: "..."}`.

### 3 · SPICE Agent (simulation)
`src/agents/spice_agent.py`

Handles netlists not in the KB. Its loop:

```
user netlist
    │
    ├─► parse_netlist()             ← validate + extract subckt_name
    │
    ├─► run_spice_simulation()      ← auto-generates AC/DC testbench
    │       ngspice -b (batch, no GUI)
    │       parses .meas output → measured__dcgain, measured__gbp…
    │
    ├─► semantic_search_netlist()   ← find nearest KB circuit
    │
    └─► get_circuit_record()        ← fetch stored measurements
            compare_to_kb() → % diff per FoM
```

Returns: `{simulation_results: {...}, comparison: {...}, synthesis: "..."}`.

### 4 · MCP Server (tool bus)
`src/mcp/server.py`

All tool calls from both agents pass through the MCP server. Running as a separate Docker service on port 8001, it registers 9 tools over stdio and dispatches to the underlying Python functions.

---

## Knowledge Base Design

Every circuit in the KB is stored as **three linked entities**, all keyed by `circuit_id = topology_name/scenario/sample_id`:

| Entity | Store | Embedding | Dim |
|---|---|---|---|
| Canonicalized netlist text | ChromaDB `analog_netlists` | OpenAI `text-embedding-3-large` | 3072 |
| Simulation measurements | SQLite + ChromaDB `analog_behavior` | Normalized feature vector | 64 |
| Circuit graph (PyG) | ChromaDB `analog_graphs` + `.pt` files | GIN encoder (contrastive-trained) | 256 |

**Netlist canonicalization** (`src/ingestion/netlist_parser.py`): raw SPICE has arbitrary node names that destroy embedding similarity for topologically equivalent circuits. Before embedding, BFS renaming from supply/port anchors produces deterministic integer labels (`N0`, `N1`…), PDK model names are replaced with class tokens (`pmos_lv`, `nmos_lv`), and W/L/M are log-binned.

**Graph construction** (`src/ingestion/graph_builder.py`): each circuit becomes a bipartite graph — device nodes (21-dim: type, model class, W/L/M, is_mirror) + net nodes (3-dim: is_supply, is_port, fanout) — connected by typed edges (gate/drain/source/bulk/cap/res).

**GIN training** (`src/ingestion/gnn_train.py`): contrastive InfoNCE loss on augmentation pairs (node permutation, edge dropout, feature noise). After training, circuits of the same topology cluster together in the 256-dim embedding space.

---

## Project Layout

```
analog-rag/
├── src/
│   ├── ingestion/
│   │   ├── netlist_parser.py   # SPICE parse + BFS canonicalization
│   │   ├── graph_builder.py    # PyG hetero/homo graph construction
│   │   ├── embedder.py         # GINEncoder, OpenAI embedder, behavior vector
│   │   ├── ingest.py           # CLI: walk rag_data/, ingest all runs
│   │   └── gnn_train.py        # Contrastive GIN training (InfoNCE)
│   ├── database/
│   │   ├── vector_store.py     # ChromaDB wrapper (3 collections)
│   │   └── sql_store.py        # SQLite measurements store
│   ├── mcp/
│   │   ├── server.py           # MCP server (9 tools over stdio)
│   │   └── tools/
│   │       ├── vector_search.py  # semantic + graph search tools
│   │       ├── sql_query.py      # measurement filter tools
│   │       └── spice_runner.py   # ngspice runner + testbench generator
│   ├── agents/
│   │   ├── superagent.py       # Orchestrator — routes to research/SPICE
│   │   ├── research_agent.py   # Retrieval-optimised agent
│   │   └── spice_agent.py      # Simulation + KB comparison agent
│   ├── ui/
│   │   └── app.py              # Streamlit chat UI
│   └── eval/
│       └── success_criteria.py # SC-R1..R4, SC-S1..S2, SC-I1..I2 evaluator
├── configs/
│   └── ingest_config.json      # Paths, model names, topology catalog
├── tests/                      # Unit tests (14 passing)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── RAG_PLAN.md                 # Full design document
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- AnalogGym data on your machine (mounted read-only)
- API keys in `.env` (copy from `.env.example`)

### 1. Configure

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

### 2. Start services

```bash
docker compose up --build -d
```

Starts ChromaDB (:8000), MCP server (:8001), and Streamlit UI (:8501).

### 3. Ingest the knowledge base (one-time)

```bash
docker compose --profile ingest run --rm ingest
```

Parses all 48 circuit runs, embeds netlists via OpenAI, builds GIN graph embeddings, and loads ChromaDB + SQLite.

### 4. Open the UI

Navigate to **http://localhost:8501**.

### 5. Example queries

```
Show me a three-stage OTA with feed-forward compensation

Find circuits with GBW > 700 kHz and phase margin > 65 degrees

What is the settling time of Fan_SMC_Pin_3 in the nominal corner?

[paste a .subckt block] — simulate this and compare to the KB
```

---

## Optional: Train the GNN

Improves graph similarity search by training the GIN encoder on the KB circuits:

```bash
docker compose --profile train run --rm gnn-train
```

Weights are saved to `/app/data/gin_weights.pt` and picked up automatically on the next query.

## Run Success Criteria Evaluation

```bash
docker compose --profile eval run --rm eval
# Report written to /app/data/eval_report.json
```

Checks SC-R1 (semantic recall@5 ≥ 70%), SC-R2 (SQL precision 100%), SC-R3 (graph topology recall), SC-R4 (entity linking), SC-S1 (simulation accuracy within 1%), SC-S2 (testbench coverage ≥ 14/17), SC-I1/I2 (infra health).

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM routing | LiteLLM (Claude · GPT · Gemini via env var) |
| Text embeddings | OpenAI `text-embedding-3-large` (3072-dim) |
| Vector store | ChromaDB (self-hosted, 3 collections) |
| SQL store | SQLite |
| Graph ML | PyTorch Geometric — GIN encoder |
| SPICE simulator | ngspice ≥ 42, batch mode (`-b`) only |
| Agent protocol | MCP Python SDK |
| UI | Streamlit + Plotly |
| Container | Docker + Docker Compose |
| PDK | SkyWater 130nm (`sky130_fd_pr`) |
