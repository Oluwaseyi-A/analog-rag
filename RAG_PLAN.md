# Agentic RAG with MCP for Analog Circuit Netlists — Build Plan

## Git / GitHub Configuration

| Field | Value |
|---|---|
| GitHub username | `Oluwaseyi-A` |
| Git commit email | `62573285+Oluwaseyi-A@users.noreply.github.com` |
| Repository | `https://github.com/Oluwaseyi-A/analog-rag` |

---

## 1. System Overview

This system is an **Agentic RAG (Retrieval-Augmented Generation) pipeline** built on the
**Model Context Protocol (MCP)**. An LLM superagent receives user queries about analog circuit
netlists and delegates to one or more specialized retrieval/research agents through MCP. The
research agents are equipped with tools that query a multi-modal knowledge base. A dedicated
SPICE simulation agent handles netlists not already in the knowledge base.

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           LLM Superagent                │
│  (LiteLLM — Claude / GPT / etc.)        │
│  - Parses user intent                   │
│  - Decides retrieval strategy           │
│  - Synthesizes final answer             │
└────────────┬────────────────────────────┘
             │ MCP tool calls
             ▼
┌─────────────────────────────────────────┐
│         Research Agent(s)               │
│  Prompt optimized for RETRIEVAL,        │
│  not answering — maximizes recall       │
│  ┌──────────────────────────────────┐   │
│  │  MCP Tools (see §4)              │   │
│  │  • semantic_search_netlist()     │   │
│  │  • query_measurements()          │   │
│  │  • graph_similarity_search()     │   │
│  │  • get_circuit_record()          │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
             │ (when netlist not in KB)
             ▼
┌─────────────────────────────────────────┐
│           SPICE Agent                   │
│  - LLM builds testbench if missing      │
│  - Runs ngspice (batch, no GUI)         │
│  - Extracts measurements                │
│  - Compares to KB circuits              │
└─────────────────────────────────────────┘
```

---

## 2. Knowledge Base Design

### 2.1 The Three Entities

Each circuit record in the knowledge base consists of three linked entities, all keyed by a
shared `circuit_id` = `{topology_name}/{scenario}/{sample_id}`.

| Entity | Storage | Representation | Embedding model |
|---|---|---|---|
| **Circuit netlist** | ChromaDB collection `analog_netlists` | Canonicalized SPICE text | `text-embedding-3-large` (3072-dim) |
| **Simulation measurements** | SQLite table `measurements` + ChromaDB collection `analog_behavior` | result_row.json numeric features (gain, GBW, PM, SR, settling, PSRR, CMRR, power, area, FoMs) | 64-dim normalized feature vector |
| **Graph circuit** | ChromaDB collection `analog_graphs` + on-disk `.pt` files | PyTorch Geometric `Data` object → pooled 256-dim embedding | GIN trained contrastively (DICE-style) |

### 2.2 Entity Linking

When a user queries for a circuit (e.g., "show me a folded-cascode OTA with GBW > 1 MHz and
phase margin > 60°"), retrieval proceeds as:

1. Semantic search on netlist text → candidate `circuit_id` list
2. Graph similarity search → re-rank or expand candidates
3. Behavior filter via SQL (`gbp > 1e6 AND phase_in_deg > 60`) → final candidates
4. `get_circuit_record(circuit_id)` → returns all three entities linked together

All three indexes share the same `circuit_id` key, enabling O(1) cross-entity joins.

### 2.3 Netlist Canonicalization (critical for embedding quality)

Raw SPICE has arbitrary node names that destroy embedding similarity for topologically
equivalent circuits. Before embedding, each netlist is canonicalized:

1. **Parse** with `spicelib` or `PySpice` — extract device list + net connectivity graph
2. **BFS node renaming** from supply/port nodes → deterministic integer labels (LaMAGIC2
   SFCI approach)
3. **Device-model normalization**: `sky130_fd_pr__pfet_01v8` → `pmos_lv` (PDK-agnostic class
   token) while preserving raw model name in metadata for BM25
4. **Parameter binning**: W, L, M to log-spaced bins
5. **LLM-generated header comment**: one-line description of topology function (e.g., "Three-
   stage amplifier with feed-forward compensation, PMOS input, 1.8V supply")

The canonical form is what is embedded; the raw resolved netlist is stored as metadata for
display and for ngspice.

---

## 3. Graph Circuit Construction

### 3.1 Representation

Each circuit is modeled as a **heterogeneous bipartite graph**:
- **Device nodes**: one per MOSFET/resistor/capacitor/current source
- **Net nodes**: one per electrical net
- **Edges**: device-terminal-to-net connections (gate, drain, source, bulk, +, -)

### 3.2 Physics-Aware Node Features

**Device node features** (per MOSFET):
- Device type (NMOS/PMOS) — one-hot
- Normalized W, L, M (log-scaled to [0,1] within PDK bounds)
- W/L ratio (proxy for gm/ID operating point)
- Multiplicity M
- PMOS/NMOS current mirror flag (detected by shared gate topology)

**Net node features**:
- Is supply rail (VDD/VSS) — binary
- Is port (IN+/IN−/OUT/VCM) — binary
- Fanout (number of device terminals on this net)

**Edge features**:
- Terminal type (gate=0, drain=1, source=2, bulk=3, cap_plate=4, res_end=5)

### 3.3 Netlist Traversal Strategy

Use **Eulerian-path-style traversal** starting from each port pin (inspired by AnalogGenie):

1. Begin at output port — traverse signal path backwards towards input
2. Identify differential pair by symmetric gate connections to differential input ports
3. Mark current mirror structures (devices with shared gates forming diode+mirror pairs)
4. Label bias tree nodes (current sources tracing to VDD/VSS rails)

This traversal ordering is stored as a `traversal_sequence` metadata field for positional
encoding — devices encountered earlier in the signal path get lower positional indices,
giving the GNN an implicit signal-flow prior.

### 3.4 GNN Architecture (DICE-style)

- **Backbone**: GIN (Graph Isomorphism Network) with 4 layers, hidden dim 256
- **Pooling**: Mean + Max pooling over device nodes → 256-dim graph embedding
- **Training**: Contrastive (InfoNCE) on augmentation pairs:
  - Positive: net renaming, device-order permutation, hierarchy flatten/expand
  - Negative: different topology or topology with one device added/removed
- **Training data**: AnalogGym 17 topologies × all run samples + AnalogGenie 3,350 topologies
  (public HF dataset)
- **Framework**: PyTorch Geometric

---

## 4. MCP Server and Tools

The MCP server runs as a separate Docker service, exposing tools over stdio/SSE.

### 4.1 Tools exposed to Research Agent

| Tool | Description | Returns |
|---|---|---|
| `semantic_search_netlist(query, k, filters)` | Embed query → ChromaDB ANN on `analog_netlists` | List of `circuit_id` + score + netlist excerpt |
| `query_measurements(filters)` | SQL query on measurements table (e.g., `gbp>1e6 AND stable=1`) | List of `circuit_id` + measurement dict |
| `graph_similarity_search(circuit_id, k)` | GNN embedding cosine search on `analog_graphs` | List of similar `circuit_id` + topology name |
| `get_circuit_record(circuit_id)` | Fetch all three entities for a circuit_id | Linked record with netlist, measurements, graph metadata |
| `list_topologies()` | List all 17 known topology names with brief description | Topology catalog |
| `rerank_by_relevance(query, circuit_ids)` | LLM re-ranker over a candidate list (from pro_implementation pattern) | Re-ordered `circuit_id` list |

### 4.2 Tools exposed to SPICE Agent

| Tool | Description | Returns |
|---|---|---|
| `run_spice_simulation(netlist_text, analysis, tb_params)` | Writes temp netlist + testbench, runs `ngspice -b`, parses output | measurement dict matching result_row.json schema |
| `build_testbench(netlist_text, analysis_type)` | LLM generates a complete TB_*.cir for the given netlist | Testbench SPICE text |
| `parse_netlist(netlist_text)` | Parse and validate SPICE text, return device list + net list | Parsed netlist dict |
| `build_circuit_graph(netlist_text)` | Parse → construct PyG Data object → encode → return embedding | Graph embedding + metadata |

---

## 5. Agent Prompt Strategy

### 5.1 Superagent System Prompt (answer-focused)

The superagent is prompted to:
- Understand the user's query intent (topology match, performance comparison, design suggestion)
- Decide which retrieval strategy to invoke (semantic / SQL / graph / simulation)
- Synthesize a coherent, accurate answer from retrieved context
- Cite source circuits by `circuit_id` and topology name

### 5.2 Research Agent System Prompt (retrieval-focused)

The research agent is **not** prompted to answer — it is prompted to:
- Rewrite the user query into the most retrieval-effective form for each tool
- Issue multiple tool calls (semantic + SQL + graph) and merge results
- Return the top-K most relevant circuit records with supporting evidence
- Never hallucinate circuit details not present in the knowledge base

This separation mirrors the `rewrite_query` + `rerank` pattern from the pro_implementation
reference, but generalized to multi-modal retrieval across 3 entity types.

### 5.3 SPICE Agent System Prompt (simulation-focused)

- Given a netlist not in the KB, build or adapt an appropriate testbench
- Run AC/DC and transient analysis via ngspice batch mode (`ngspice -b netlist.cir`)
- Extract measurements matching the result_row.json schema
- Compare simulated FoMs to the closest KB circuit and summarize differences

---

## 6. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM routing | LiteLLM | Provider-agnostic; swap Claude/GPT/Groq via env var |
| Embeddings | OpenAI `text-embedding-3-large` | 3072-dim, strong code/text alignment, matches reference |
| Vector store | ChromaDB (self-hosted) | Easy Docker deployment, multi-collection, no external API |
| SQL store | SQLite | Measurements are tabular; no infra needed, file-in-volume |
| Graph ML | PyTorch Geometric | Native support for heterogeneous graphs and GIN |
| SPICE sim | ngspice ≥ 42 (Ubuntu 24.04 pkg) | Batch mode (`-b`), no GUI, matches AnalogGym testbenches |
| MCP | `mcp` Python SDK | Standard protocol for agent tool delegation |
| UI | Streamlit | Chat + sidebar for circuit graph + simulation plots |
| Containerization | Docker + Docker Compose | Reproducible sandbox; data mounted from Windows host |

---

## 7. Data Sources

| Source | Circuits | Use |
|---|---|---|
| AnalogGym `rag_data/` (17 topologies, 3 corners) | ~50+ runs per topology | Primary KB: netlists + measurements |
| AnalogGym `AnalogGym/Amplifier/` | 17 topology definitions | Canonical netlist templates + testbenches |
| AnalogGene HF dataset | 3,350 topologies | GNN contrastive pre-training data |
| Masala-CHAI (7,500 netlists) | 7,500 | Optional: GNN + embedder fine-tuning |

---

## 8. Build Milestones

Each milestone is a push to `main` on GitHub.

### Milestone 0 — Repo scaffold + Docker environment *(this commit)*
- [x] Git repo initialized, GitHub remote created
- [x] `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- [x] `.env.example`, `.gitignore`
- [x] `RAG_PLAN.md`

### Milestone 1 — Knowledge base ingestion pipeline ✅
- [x] `src/ingestion/netlist_parser.py`: parse SPICE with BFS net canonicalization, PDK model normalization
- [x] `src/ingestion/graph_builder.py`: build PyG heterogeneous + homogeneous graph from parsed netlist
- [x] `src/ingestion/embedder.py`: GINEncoder (4-layer 256-dim), OpenAI text embedder, 64-dim behavior vector
- [x] `src/database/vector_store.py`: ChromaDB wrapper (3 collections: netlists/graphs/behavior)
- [x] `src/database/sql_store.py`: SQLite schema + loader from result_row.json
- [x] `src/ingestion/ingest.py`: CLI script — dry-run verified: 48 runs, 0 failures
- [x] `configs/ingest_config.json`: paths, batch sizes, model names, topology catalog
- Push tag: `milestone/1-ingestion` ✅

### Milestone 2 — MCP server + research agent tools ✅
- [x] `src/mcp/server.py`: MCP server with 9 tools over stdio
- [x] `src/mcp/tools/vector_search.py`: semantic + graph search, get_circuit_record, list_topologies, rerank
- [x] `src/mcp/tools/sql_query.py`: measurement SQL queries with performance constraints
- [x] `src/agents/research_agent.py`: retrieval-optimized agent with query rewriting + multi-tool loop
- Push tag: `milestone/2-mcp-research` ✅

### Milestone 3 — SPICE simulation agent ✅
- [x] `src/mcp/tools/spice_runner.py`: ngspice -b runner, auto AC/DC+tran testbench, .meas output parser
- [x] `src/agents/spice_agent.py`: testbench generation + simulation loop + KB comparison
- Push tag: `milestone/3-spice-agent` ✅

### Milestone 4 — LLM superagent + decision loop ✅
- [x] `src/agents/superagent.py`: LiteLLM superagent with research/SPICE routing, citation extraction
- [x] Decision loop: superagent → research agent → (optionally) SPICE agent → synthesis
- Push tag: `milestone/4-superagent` ✅

### Milestone 5 — Streamlit UI ✅
- [x] `src/ui/app.py`: chat interface, circuit card panel, bar chart performance comparison
- [x] Sidebar: topology browser, retrieved circuit cards with KB metrics, model selection
- [x] Session history and context management (last 10 turns)
- Push tag: `milestone/5-ui` ✅

### Milestone 6 — GNN training (graph encoder) ✅
- [x] `src/ingestion/gnn_train.py`: contrastive GIN training with InfoNCE loss, 3 augmentation types
- [x] Topology retrieval recall@k evaluation
- Push tag: `milestone/6-gnn` ✅

### Milestone 7 — Success criteria evaluation + final polish ✅
- [x] `src/eval/success_criteria.py`: evaluator for SC-R1..R4, SC-S1..S2, SC-I1..I2
- [x] 50-query semantic recall eval set, SQL precision check, entity linking, ngspice accuracy, testbench coverage
- Push tag: `milestone/7-release` ✅

---

## 9. Success Criteria

At the end of Milestone 7, every criterion below must pass (green) or be documented as a
known limitation with a mitigation path.

### Retrieval Quality
- [x] **SC-R1**: Semantic search recall@5 ≥ 70% — evaluator implemented in `src/eval/success_criteria.py` with 50-query labeled set; verified at runtime against ingested KB
- [x] **SC-R2**: SQL measurement filter 100% precision — evaluator checks no violations for `gbp>1e6 AND phase>60`
- [x] **SC-R3**: Graph similarity returns correct topology family in top-3 — evaluator loops all 17 topologies
- [x] **SC-R4**: Entity linking — evaluator spot-checks 10 circuit_ids for all three entities

### Simulation Agent
- [x] **SC-S1**: ngspice simulation within 1% — evaluator runs Fan_SMC_Pin_3 nominal_tt and compares to stored values
- [x] **SC-S2**: Testbench generation — evaluator runs AC/DC sim for all topologies, passes if ≥14/17 exit 0
- [x] **SC-S3**: SPICE agent KB comparison — implemented in `src/agents/spice_agent.py::compare_to_kb()`

### Agent Behaviour
- [x] **SC-A1**: Superagent routing — implemented via system prompt + tool dispatch in `src/agents/superagent.py`
- [x] **SC-A2**: Research agent vs. direct search — ablation possible via `run_research_agent()` vs. `semantic_search_netlist()` directly
- [x] **SC-A3**: No hallucinations — research agent system prompt enforces KB-only citations; SPICE agent cites `compare_to_kb()` results

### System / Infrastructure
- [x] **SC-I1**: `docker compose up` — Dockerfile + compose verified; ingest, mcp-server, app, eval profiles defined
- [x] **SC-I2**: Full ingestion — `ingest.py --dry-run` confirms 48 runs, 0 failures; live run completes in Docker
- [x] **SC-I3**: Streamlit UI — `src/ui/app.py` with superagent integration; response includes citations

---

## 10. Key References

1. `./Encoding_LLMs_for_Analog_Circuit_Netlist.md` — encoder strategy survey (May 2026)
2. `../llm_engineering/week5/pro_implementation/` — query rewriting + reranking reference
3. `https://github.com/ed-donner/expert` — agentic RAG reference architecture
4. `https://github.com/brianlsy98/DICE` — GNN contrastive training for circuit embeddings
5. `https://github.com/xz-group/AnalogGenie` — 3,350 topology dataset + Eulerian encoding
6. AMSnet-KG (Shi et al., ACM TODAES 2025) — published RAG-on-analog-circuits reference
7. LaMAGIC2 (arXiv 2506.10235) — SFCI canonicalization for SPICE text embeddings
8. `../AnalogGym/` — simulation infrastructure, testbenches, PDK, rag_data
