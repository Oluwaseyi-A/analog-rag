# Encoders for RAG over SPICE/Spectre Analog Netlists: State of the Art (May 2026)

## TL;DR
- **There is no production-grade, off-the-shelf encoder purpose-built for SPICE/Spectre analog netlists** that you can drop in like an OpenAI text embedder; the closest things — DICE, AnalogGenie's encoder, CktGNN, AMSnet-KG — are research artifacts with limited checkpoints and very narrow training corpora (mostly op-amps and discrete textbook circuits).
- **The pragmatic recommendation for May 2026** is a **hybrid pipeline**: use `voyage-code-3` (or `jina-embeddings-v4` with the code LoRA) as a strong off-the-shelf text encoder on *canonicalized* SPICE text for tasks (1) topology similarity and (3) subcircuit/device lookup, fuse with a **graph-structure embedding** (a GraphSAGE/GIN trained contrastively in the style of DICE) for hard topology matches, and **augment any embedding by simulation-derived feature vectors** (AC magnitude/phase, DC operating point, transient signatures) for task (2) functional-behavior matching — because raw netlists do not encode behavior.
- **Custom fine-tuning is genuinely needed if you want above-baseline performance**, but a *minimum viable* path is well-defined: take CodeSage-v2 or voyage-code-3 → continue contrastive pre-training on SPICE pairs (positive = synthesis-equivalent perturbations: node renames, device permutation, hierarchy flatten/expand; negative = unrelated topologies), 1–10k labeled triplets typically suffice for LoRA-scale adaptation.

## Key Findings

### 1. Domain-specific encoders for analog circuit netlists exist but are narrow

The 2023–2026 literature contains four lines of work that produce *embeddings* (as opposed to generators) for transistor-level circuits, plus several generators whose internal representations could be repurposed:

| Model | Year | Type | Domain | Checkpoint | Repo |
|---|---|---|---|---|---|
| **CktGNN** (Dong et al., ICLR'23) | 2023 | Nested GNN VAE | Op-amps only (DAG over 24-subgraph basis) | Not shipped; retrainable | github.com/zehao-dong/CktGNN |
| **DICE** (Lee et al., 2025, arXiv 2502.08949) | 2025 | MPNN, graph contrastive pretraining | Device-level analog + digital, simulation-free | Code only | github.com/brianlsy98/DICE |
| **NetTAG** (Fang et al., DAC'25, arXiv 2504.09260) | 2025 | LLM text-encoder + graph transformer on text-attributed graphs | **Gate-level digital only** | Code only | github.com/hkust-zhiyao/NetTAG |
| **DeepGate2/3/4** (Shi et al.) | 2023–25 | GNN + transformer on AIGs | **Digital AIG only** | Trainable; needs DG2 ckpt | github.com/zyzheng17/DeepGate3 |
| **AnalogGenie** (Gao et al., ICLR'25) | 2025 | Decoder-only transformer (~11.8M params) on Eulerian-circuit sequences | Analog topologies (3,350 designs, 11 classes incl. LDO, BGR, Comparator, PLL, LNA, PA, Mixer, VCO) | Public HF checkpoint + HF dataset | github.com/xz-group/AnalogGenie ; huggingface.co/JianGao666/AnalogGenie |
| **GANA** (Kunal et al., DATE'20) | 2020 | GCN for subcircuit/annotation | Analog transistor netlists | Dataset only | github.com/kkunal1408/GANA_circuit_data |

**Critical observation for your use case**: of these, only **DICE**, **AnalogGenie**, **GANA**, and **CktGNN** actually train on analog/transistor-level circuits. NetTAG and the DeepGate family are explicitly gate-level digital (and-inverter graphs and post-synthesis cell netlists), so despite being the most polished foundation models in EDA, they will not work on your SPICE MOSFET netlists without retraining. **None of the four analog-relevant models ships a turn-key checkpoint optimized for the *retrieval* objective** — AnalogGenie's checkpoint is a *generative* model and its sequence-token representations are not pre-trained for cosine retrieval, though its encoder layers can be used as features. CktGNN's training corpus is exclusively op-amps. DICE is the most general device-level encoder but is described in the paper as a research prototype evaluated on three downstream property-prediction tasks rather than as a calibrated retrieval encoder.

### 2. General-purpose code embedders are a surprisingly strong baseline

Per Voyage AI's official launch blog post (blog.voyageai.com/2024/12/04/voyage-code-3/, December 4 2024): "It outperforms OpenAI-v3-large and CodeSage-large by an average of 13.80% and 16.81% on a suite of 32 code retrieval datasets, respectively." Voyage-code-3 has a 32K-token context and Matryoshka dimensions (256/512/1024/2048). Per Jina AI's paper (arXiv 2506.18902): "Code retrieval capabilities reach 71.59 on CoIR benchmark, though specialized models like voyage-code-3 (77.33) achieve higher scores in this domain" — i.e., voyage-code-3 is the current SOTA for *general code retrieval*, but neither has been evaluated on SPICE/Spectre specifically. CodeSage-v2 (codesage/codesage-large-v2, 1.3B encoder with 2048-dim output, Apache-2.0) is a strong fully open alternative supporting nine programming languages plus the Stack corpus — note SPICE is *not* in its training distribution.

**The empirical question** — how well do these models work on SPICE netlists? — is not directly answered in the published literature. The closest analog is the LaMAGIC/LaMAGIC2 line (LaMAGIC at ICML'24; LaMAGIC2 at arXiv 2506.10235, June 2025), which fine-tunes T5/BART-style language models on canonical SPICE-graph serializations. Per the LaMAGIC2 abstract: "LaMAGIC2 also exhibits better transferability for circuits with more vertices with up to 58.5% improvement" — i.e., changing the input formulation from `O(|V|²)` adjacency strings to an `O(|V|)` "Succinct Float-input Canonical formulation with Identifier" (SFCI) dramatically improves transfer to larger circuits. The takeaway: **canonicalization of the netlist text matters more than the model choice for code embedders**, because raw SPICE has arbitrary node-naming conventions that explode embedding variance for topologically identical circuits.

### 3. Graph-based approaches are technically correct but operationally heavier

A netlist *is* a graph (devices = nodes, nets = edges), and contrastive GNN training is the right inductive bias for topology similarity (task 1). The recent DICE paper makes the case most cleanly: it is "the first self-supervised pretrained graph neural network model for any circuit expressed at the device level" and uses simulation-free augmentations — positive augmentations preserve high-level semantics, negative augmentations modify them — to maximize discrimination across topologies in a contrastive InfoNCE objective. AMSnet-KG (Shi et al., ACM TODAES 2025) takes a different tack: it builds an explicit knowledge graph of AMS circuits with functional and performance annotations and uses Neo4j Cypher queries instead of dense embeddings for retrieval — this is the published RAG-over-analog-netlists architecture as of late 2025 and is worth studying as a reference design.

**For hierarchical netlists** (`.SUBCKT` blocks, Spectre `subckt`/`inline subckt`), DeepGate3 introduced a "window-shifting" approach that partitions netlists into ≤512-gate areas and aggregates with a Pooling Transformer + `[CLS]` token, and this same hierarchy-aware pooling pattern transfers cleanly to analog: embed each `.SUBCKT` independently, then aggregate via a learned pooling head, and additionally embed the flat (expanded) form for fallback retrieval.

### 4. Functional behavior cannot be recovered from the netlist alone

This is the single most important architectural point for your task (2). Two SPICE netlists with identical topology can have wildly different transfer functions depending on device sizing (W/L, multiplier, finger count) and parameter values; conversely, very different topologies can produce indistinguishable AC behavior over a band of interest. **No published embedding of netlist text or graph structure captures behavior in the AC/DC/transient sense.** AnalogCoder-Pro explicitly addresses this by adding *waveform images* as a separate modality fed to an MLLM, motivated by the observation that LLMs reading textual numerical samples misidentify a noisy triangular waveform as a sine wave while MLLMs reading the waveform image correctly identify it.

The practical implication: for functional-behavior retrieval you must compute a simulation-derived feature vector per netlist — for example a fixed-grid AC magnitude/phase sample (e.g., 64 log-spaced frequency points → 128-dim vector), DC operating points of named nets, key FoMs from a testbench (gain, GBW, phase margin, slew, PSRR, noise) — and concatenate or alternate this with the structural embedding in the vector store.

### 5. The 2024–2026 LLM-for-analog ecosystem at a glance

The landscape is moving rapidly. The most relevant artifacts:

- **AnalogCoder** (Lai et al., AAAI'25 Oral) — training-free LLM agent that emits PySpice; ships a *circuit tool library* of reusable modular sub-circuits which is itself a small but high-quality corpus for fine-tuning a retriever.
- **AnalogCoder-Pro** (Lai et al., 2025, arXiv 2508.02518) — multimodal extension; fine-tunes Qwen2.5-Coder via LoRA on rejection-sampled high-quality netlists from AnalogGenie (~390 fine-tuning examples).
- **LaMAGIC** (Chang et al., ICML 2024) / **LaMAGIC2** (Chang et al., arXiv 2506.10235, June 2025) — fine-tuned masked / encoder-decoder language models for power-converter topology generation, with the SFCI canonical formulation we recommend you adopt for *any* SPICE-text embedding pipeline.
- **AnalogGenie** (Gao et al., ICLR'25) — releases a 3,350-circuit dataset and a transformer trained on Eulerian-circuit sequences; the **3,350 SPICE topologies cover 11 circuit classes**, making it currently the largest publicly available analog topology corpus you can index for RAG.
- **Masala-CHAI** (Bhandari et al., 2024–25, arXiv 2411.14299) — 7,500 captioned SPICE netlists extracted from 10 textbooks via a YOLOv8 + GPT-4o pipeline; fine-tuning *GPT-4o* on this corpus yields a 46% Pass@1 improvement when used in an AnalogCoder agent (per the Masala-CHAI Fig. 1 caption: "Pass@1 performance of GPT-4o fine-tuned with Masala CHAI datasets extracted from between 1-10 textbooks"). This is the most useful *training data* for an analog-domain embedder available today.
- **Schemato** (2024, arXiv 2411.13899) — netlist-to-schematic LLM, hits 76% compilation success vs. 63% for best baseline LLMs.
- **GNN-ACLP** (2024, arXiv 2504.10240) — graph-based link prediction across SpiceNetlist, Image2Net, and AnalogGenie datasets with 92–99% cross-dataset accuracy, including a "Netlist Babel Fish" RAG component for format conversion.
- **AMSnet-KG** (Shi et al., ACM TODAES 2025) — knowledge-graph RAG over an annotated AMS netlist dataset, the canonical published "RAG-on-analog-circuits" reference architecture.

### 6. Vector database considerations

The retrieval volume here is modest by RAG standards (thousands to low millions of subcircuits, not billions of documents), so almost any vector database works — FAISS, Qdrant, pgvector, LanceDB, Weaviate. The actual constraints are:
- **Multi-vector storage per netlist**: you'll want at least 3 fields — `text_embedding`, `graph_embedding`, `behavior_embedding` — and either separate indexes with rank fusion (RRF) or a single index of concatenated/projected vectors.
- **Hybrid sparse+dense**: keep a BM25/SPLADE channel over device-type/model-name/`.SUBCKT` tokens, because exact symbol matching ("nch_lvt_25", "vdda_1p2") is critical for design reuse — pure dense retrieval will hallucinate near-misses.
- **Matryoshka-aware**: voyage-code-3's 256-dim mode is materially cheaper at scale with minimal recall loss; design indexes for the lower truncation up front.

## Details

### Recommended preprocessing pipeline for SPICE/Spectre netlists

The biggest leverage point in this entire problem is canonicalization, because raw SPICE is wildly redundant in surface form. Concretely:

1. **Parse** with a real parser (HSpice/Spectre grammars; use `PyEDA`, `spicelib`, the Auto-SPICE parser, or PySpice's `Netlist.py`). Do *not* embed raw text directly.
2. **Hierarchy handling**: keep two views — (a) **flat** (expand all `.SUBCKT` invocations) for behavior comparison, (b) **hierarchical** (per-subckt embedding then pooled) for design-reuse retrieval.
3. **Node-name canonicalization**: rename nets by a deterministic graph traversal (BFS from supplies/ports), so that topologically identical circuits with different net names produce identical canonical strings — this is exactly LaMAGIC2's SFCI insight.
4. **Device-model abstraction**: maintain two tokens per device — the raw model name (`nch_25_mac`) and a normalized class (`nmos_lv`). The first lets BM25 match exact device reuse; the second lets the dense embedder generalize across PDKs.
5. **Parameter binning**: discretize W, L, M, fingers, etc., into log-spaced bins; raw float values blow up embedding variance with no semantic gain.
6. **Comment stripping / re-comment**: strip designer-noise comments, but *add* a generated comment header with subcircuit class label (from GANA-style annotation, or LLM-tagged) and intended function — this is what makes natural-language queries work in a RAG.

### Hybrid embedding architecture (recommended)

For each netlist or subcircuit, store three embeddings:

- **`E_text`** (768–2048 dim): canonicalized SPICE text → `voyage-code-3` (best quality, paid API) or `jina-embeddings-v4` with code adapter (best open weights, 3.09B params after merging) or `codesage-large-v2` (best fully-open free option, Apache-2.0). For Spectre's hierarchical `inline subckt`, prepend a short natural-language header generated by an LLM ("Folded-cascode OTA, NMOS input pair, 1.2 V supply") — this exploits the model's strong NL↔code alignment.
- **`E_graph`** (128–512 dim): device-pin bipartite graph → a small GIN/GraphSAGE encoder fine-tuned contrastively per DICE's recipe (positive augmentations: net renames, hierarchy flatten/expand, symmetric-device swap; negative: cross-topology). Start from DICE's public code; budget on the order of a single GPU-day to retrain on your library.
- **`E_behavior`** (64–256 dim): a simulation-feature vector. Minimum: gain/phase at log-spaced frequencies; better: PCA over a 1k-frequency AC sweep, plus DC op-point of labeled nets, plus a small fixed transient signature.

At query time, rank-fuse the three using Reciprocal Rank Fusion or a learned linear head over ~1k labeled query→target pairs.

### Decision tree for each retrieval task

- **Task 1 — Topology similarity**: graph embedding dominates; rank-fuse text. Most direct off-the-shelf path is to retrain DICE on your library (one GPU-day), then use cosine over `E_graph`. If you want zero-training, use voyage-code-3 on the *canonicalized* netlist — empirically this is decent because canonicalization removes most of the surface variance, but it will not match a graph encoder on permutation-equivalent topologies.
- **Task 2 — Functional-behavior matching**: `E_behavior` dominates; you *must* simulate. The netlist text and graph alone cannot disambiguate behavior. Use AC magnitude/phase + key FoMs as the primary key; treat text/graph as a coarse pre-filter. For frequently-queried netlists, cache simulation results.
- **Task 3 — Subcircuit/device location for design reuse**: hybrid sparse+dense; BM25 over device model names and `.SUBCKT` headers is critical — design reuse is fundamentally about finding "the same `pmos_lv` cascode mirror" by name. Use voyage-code-3 (or CodeSage-large-v2 if open weights required) on the canonical text as the dense channel.

### Off-the-shelf vs custom-trained — the verdict

There is **no off-the-shelf SPICE-native embedder ready for production**. The closest candidate, AnalogGenie's HF checkpoint (huggingface.co/JianGao666/AnalogGenie), is a *generative* decoder-only transformer trained on Eulerian-sequence representations of 3,350 topologies — its hidden states could be pooled into embeddings, but this is not what it was trained or evaluated for, and the corpus is small.

**Minimum viable custom training recipe**:
1. Start from `voyage-code-3` (if you accept the API) or `codesage-large-v2` (open weights).
2. Build 1–10k contrastive triplets via *automatic* augmentation on your own library: positive = canonicalized version of same netlist with renamed nets / reordered devices / flattened hierarchy; negative = random other netlist or topology-perturbed (one device deleted/added — this is DICE's exact recipe).
3. LoRA-fine-tune with InfoNCE for 1–3 epochs. Budget: hours on one A100.
4. Optionally bootstrap labels using a strong LLM (GPT-4o, Claude) to annotate "these two netlists implement the same function" pairs from the AnalogGenie + Masala-CHAI + AMSnet-KG public corpora (≈11k circuits combined) to add cross-topology functional positives.

The marginal benefit of going further (training a from-scratch graph foundation model in the DICE/NetTAG style) is large only if you have >100k netlists; below that scale, fine-tuning a strong generic encoder dominates.

## Recommendations

**Stage 0 — Validate the off-the-shelf baseline first (1–3 days).** Build a small evaluation set of ≈100 hand-labeled query→correct-result pairs from your own library. Compare voyage-code-3, jina-embeddings-v4 (code adapter), and codesage-large-v2 on canonicalized vs. raw SPICE. **Threshold to move on: if top-5 recall on canonicalized text is <60% for topology queries, proceed to Stage 1.**

**Stage 1 — Add canonicalization + hybrid sparse-dense (1 week).** Implement the canonicalization pipeline above. Add BM25 over device/subckt tokens. This typically lifts recall substantially with no training. **Threshold to move on: if functional-behavior queries (task 2) still fail systematically, proceed to Stage 2 — they will, because raw text cannot capture behavior.**

**Stage 2 — Add the simulation-behavior channel (2–4 weeks).** Standardize a per-netlist AC/DC testbench, run it once per indexed netlist, store the resulting feature vector. This is more engineering than ML — your bottleneck will be testbench generation, not the embedder. **Threshold to move on: if topology-similarity queries still rank permutation-equivalent circuits poorly, proceed to Stage 3.**

**Stage 3 — Train a graph encoder à la DICE (1–2 GPU-weeks).** Clone github.com/brianlsy98/DICE; rebuild the contrastive augmentation pipeline on your own corpus + AnalogGenie + Masala-CHAI; train a 128–256-dim GIN/GraphSAGE; fuse with text + behavior at retrieval time. **Threshold to stop**: if RRF-fused recall@10 ≥ 90% and your designers report ≤5% irrelevant suggestions, ship.

**Stage 4 — Fine-tune the text embedder (optional, 1–2 GPU-days).** LoRA on `codesage-large-v2` or `voyage-code-3` (if Voyage's fine-tuning API is enabled) with InfoNCE on auto-generated triplets. Worth it only if Stage 3 + Stage 1 fusion still leaves a measurable gap and you have ≥1k labeled positive pairs.

**Datasets to seed your training corpus immediately**: AnalogGenie (3,350 SPICE topologies, 11 circuit classes), Masala-CHAI (7,500 captioned SPICE netlists from 10 textbooks), AMSnet (semi-automated AMS netlists), AMSnet-KG (annotated KG version), OCB op-amps from CktGNN (10k op-amp DAGs; note OCB ships abstract DAGs over a 24-subgraph basis, not raw SPICE text, so you will need to regenerate text via CktGNN's simulator interface).

**Vector DB**: Qdrant or LanceDB with three named vector fields (`text`, `graph`, `behavior`) and a sparse BM25 channel. Avoid coupling yourself to a hosted-only solution because per-netlist embedding count (text + graph + behavior + per-subckt variants) will be 5–20× the netlist count.

## Caveats

- **Most published numbers are not on SPICE.** Voyage-code-3's 13.80%/16.81% gains are on general code (Python, Java, etc.). Jina-v4's 71.59 CoIR score is similarly out-of-domain. The only meaningful published in-domain numbers come from AnalogGenie (generation Pass@1), DICE (downstream property prediction, not retrieval), and GNN-ACLP (link prediction, 92–99% cross-dataset accuracy). **You will have to build your own SPICE-retrieval benchmark to make data-driven decisions.**
- **The "circuit foundation model" hype is mostly digital.** DeepGate3, NetTAG, and the bulk of recent EDA representation-learning work targets AIGs or post-synthesis gate netlists — not analog. Read claims of "foundation model for circuits" carefully.
- **AnalogGenie's HF model is a generator, not a retriever.** Its embeddings are usable but uncalibrated for cosine-distance retrieval. Treat it as a feature extractor at best, not as a drop-in encoder.
- **Behavior matching is intrinsically simulation-bound.** Any RAG architecture that promises functional retrieval purely from netlist text or graph is making a claim the literature does not support. Budget for simulation infrastructure.
- **PDK-specific device-model names are a hidden trap.** Embedders trained on public data have never seen your foundry's model card names; this is why a sparse channel + device-class normalization step is non-negotiable.
- **Forward-looking claims**: AnalogCoder-Pro and LaMAGIC2 are recent (mid-2025) and their full reproducibility on your data is unverified; the LaMAGIC2 venue is currently arXiv 2506.10235 pending confirmed proceedings attribution, and the 58.5% figure quoted from its abstract is specifically a transferability gain on larger circuits, not a general retrieval improvement. Treat the LoRA-on-Qwen2.5-Coder recipe as plausible but unproven outside the authors' benchmarks.
- The published RAG-over-analog work to date (AMSnet-KG, AnalogCoder's tool library) uses **knowledge-graph + LLM-as-retriever** patterns rather than dense embedding retrieval. This is partly because the corpora are small (hundreds to thousands of circuits) and partly because the natural-language metadata is more discriminative than the netlist text itself. If your library is <5k subcircuits, consider a KG-RAG pattern over an embedding-only RAG.