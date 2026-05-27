# analog-rag — Claude Code session context

## Environment
You are running inside a Docker container (Ubuntu 24.04).
- Python 3.11, ngspice 42 (batch mode only — always pass `-b`, never open GUI)
- AnalogGym data mounted read-only at `/data` (rag_data at `/data/rag_data/`, PDK at `/data/PDK/`)
- ChromaDB running at `http://chromadb:8000`
- Repo source at `/app/src/`

## Git
- Username: `Oluwaseyi-A`
- Email: `62573285+Oluwaseyi-A@users.noreply.github.com`
- Never add `Co-Authored-By` lines to commits.

## Architecture (see RAG_PLAN.md for full detail)
- **LLM routing**: LiteLLM (provider-agnostic)
- **Text embeddings**: OpenAI `text-embedding-3-large`
- **Vector store**: ChromaDB (3 collections: `analog_netlists`, `analog_graphs`, `analog_behavior`)
- **SQL store**: SQLite for simulation measurements
- **Graph ML**: PyTorch Geometric — GIN encoder, DICE-style contrastive training
- **Agent protocol**: MCP Python SDK
- **UI**: Streamlit on port 8501

## Key conventions
- `circuit_id` = `{topology_name}/{scenario}/{sample_id}` — links netlist, measurements, graph
- ngspice must always be called as `ngspice -b <netlist.cir>` (batch, no GUI)
- Push to GitHub at every milestone (see RAG_PLAN.md §8)
