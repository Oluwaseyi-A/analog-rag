"""ChromaDB wrapper for three analog circuit collections."""

import json
import os
from typing import Any, Optional
import chromadb
from chromadb.config import Settings


COLLECTION_NETLISTS = "analog_netlists"
COLLECTION_GRAPHS = "analog_graphs"
COLLECTION_BEHAVIOR = "analog_behavior"


class VectorStore:
    """Thin wrapper around ChromaDB providing typed access to 3 collections."""

    def __init__(
        self,
        host: str = "chromadb",
        port: int = 8000,
        netlist_collection: str = COLLECTION_NETLISTS,
        graph_collection: str = COLLECTION_GRAPHS,
        behavior_collection: str = COLLECTION_BEHAVIOR,
    ):
        self.client = chromadb.HttpClient(host=host, port=port)
        self._netlists = self.client.get_or_create_collection(
            netlist_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._graphs = self.client.get_or_create_collection(
            graph_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._behavior = self.client.get_or_create_collection(
            behavior_collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Netlist collection ────────────────────────────────────────────────────

    def upsert_netlist(
        self,
        circuit_id: str,
        embedding: list[float],
        canonical_text: str,
        metadata: dict,
    ):
        meta = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in metadata.items()}
        meta["canonical_text"] = canonical_text[:2000]
        self._netlists.upsert(
            ids=[circuit_id],
            embeddings=[embedding],
            documents=[canonical_text],
            metadatas=[meta],
        )

    def search_netlists(
        self,
        query_embedding: list[float],
        k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": k, "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        results = self._netlists.query(**kwargs)
        return _unpack_results(results, k)

    # ── Graph collection ──────────────────────────────────────────────────────

    def upsert_graph(
        self,
        circuit_id: str,
        embedding: list[float],
        metadata: dict,
    ):
        meta = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in metadata.items()}
        self._graphs.upsert(
            ids=[circuit_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def search_graphs(
        self,
        query_embedding: list[float],
        k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": k, "include": ["metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        results = self._graphs.query(**kwargs)
        return _unpack_results(results, k)

    # ── Behavior collection ───────────────────────────────────────────────────

    def upsert_behavior(
        self,
        circuit_id: str,
        embedding: list[float],
        metadata: dict,
    ):
        meta = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in metadata.items()}
        self._behavior.upsert(
            ids=[circuit_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def search_behavior(
        self,
        query_embedding: list[float],
        k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": k, "include": ["metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        results = self._behavior.query(**kwargs)
        return _unpack_results(results, k)

    # ── Fetch by ID ───────────────────────────────────────────────────────────

    def get_netlist(self, circuit_id: str) -> Optional[dict]:
        res = self._netlists.get(ids=[circuit_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return {"id": circuit_id, "document": res["documents"][0], "metadata": res["metadatas"][0]}

    def get_graph(self, circuit_id: str) -> Optional[dict]:
        res = self._graphs.get(ids=[circuit_id], include=["metadatas", "embeddings"])
        if not res["ids"]:
            return None
        return {"id": circuit_id, "metadata": res["metadatas"][0], "embedding": res["embeddings"][0]}

    def get_behavior(self, circuit_id: str) -> Optional[dict]:
        res = self._behavior.get(ids=[circuit_id], include=["metadatas", "embeddings"])
        if not res["ids"]:
            return None
        return {"id": circuit_id, "metadata": res["metadatas"][0], "embedding": res["embeddings"][0]}

    def count_netlists(self) -> int:
        return self._netlists.count()

    def count_graphs(self) -> int:
        return self._graphs.count()

    def count_behavior(self) -> int:
        return self._behavior.count()

    def reset(self):
        """Delete and recreate all collections (for testing)."""
        for name in [COLLECTION_NETLISTS, COLLECTION_GRAPHS, COLLECTION_BEHAVIOR]:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self.__init__(
            host=self.client._server_url.split("://")[1].split(":")[0],
        )


def _unpack_results(results: dict, k: int) -> list[dict]:
    # ChromaDB returns None (not a missing key) for fields not in the result set —
    # so we must guard against None explicitly, not just use dict.get defaults.
    ids_outer = results.get("ids") or [[]]
    ids = ids_outer[0] if ids_outer else []

    docs_outer = results.get("documents") or [[None] * len(ids)]
    docs = docs_outer[0] if docs_outer else [None] * len(ids)

    metas_outer = results.get("metadatas") or [[{}] * len(ids)]
    metas = metas_outer[0] if metas_outer else [{}] * len(ids)

    dists_outer = results.get("distances") or [[0.0] * len(ids)]
    dists = dists_outer[0] if dists_outer else [0.0] * len(ids)

    out = []
    for i, cid in enumerate(ids):
        out.append({
            "circuit_id": cid,
            "score": 1.0 - (dists[i] if i < len(dists) else 0.0),
            "document": docs[i] if i < len(docs) else None,
            "metadata": metas[i] if i < len(metas) else {},
        })
    return out
