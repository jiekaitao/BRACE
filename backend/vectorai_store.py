"""VectorAI DB client wrapper — all vector storage and similarity search operations.

Provides persistent vector storage via Actian VectorAI DB (gRPC) for:
- Cross-session person re-identification (CLIP/OSNet embeddings)
- Motion segment similarity search
- Vector-based activity classification

Graceful degradation: if VectorAI is unavailable, all methods return None/empty
and the system falls back to in-memory behaviour.

Internal client adapters:
- _GrpcVectorAIClient: real gRPC client for the VDSS service
- Any mock/stub passed via dependency injection (for tests)
"""

from __future__ import annotations

import json
import os
import time
import uuid as _uuid
from typing import Any, Optional

import numpy as np

# Try importing the generated gRPC stubs
try:
    import grpc
    # Relative imports work inside Docker (/app/vdss_proto) and dev
    try:
        from vdss_proto import vdss_service_pb2, vdss_service_pb2_grpc, vdss_types_pb2
    except ImportError:
        from backend.vdss_proto import vdss_service_pb2, vdss_service_pb2_grpc, vdss_types_pb2
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

# Legacy: also try importing the CortexClient (placeholder SDK from plan)
try:
    from cortex import CortexClient  # type: ignore[import-untyped]
except ImportError:
    CortexClient = None  # type: ignore[assignment,misc]


# Collection definitions
_COLLECTIONS = {
    "person_embeddings": {"dimension": 768, "description": "CLIP person re-ID embeddings"},
    "motion_segments": {"dimension": 42, "description": "SRP motion feature vectors"},
    "activity_templates": {"dimension": 42, "description": "Labeled reference movement templates"},
}


# ---------------------------------------------------------------------------
# gRPC Client Adapter
# ---------------------------------------------------------------------------


class _GrpcVectorAIClient:
    """Thin adapter that maps CortexClient-like calls to real VDSS gRPC RPCs."""

    def __init__(self, host: str, port: int):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._stub = vdss_service_pb2_grpc.VDSSServiceStub(self._channel)

    # --- health ---
    def health(self) -> dict:
        resp = self._stub.HealthCheck(vdss_service_pb2.HealthCheckRequest())
        if resp.status.code != 0:
            raise RuntimeError(resp.status.message)
        return {"status": "ok", "version": resp.version}

    # --- collection management ---
    def create_collection(self, name: str, dimension: int = 0, description: str = "") -> None:
        config = vdss_types_pb2.CollectionConfig(
            dimension=dimension,
            distance_metric=vdss_types_pb2.COSINE,
            index_algorithm=vdss_types_pb2.HNSW,
            index_driver=vdss_types_pb2.FAISS,
        )
        resp = self._stub.CreateCollection(
            vdss_service_pb2.CreateCollectionRequest(
                collection_name=name, config=config,
            )
        )
        # code 0 = success, ignore "already exists"
        if resp.status.code != 0 and "already" not in resp.status.message.lower():
            raise RuntimeError(resp.status.message)

        # Open the collection so it's ready for use
        open_resp = self._stub.OpenCollection(
            vdss_service_pb2.OpenCollectionRequest(collection_name=name)
        )
        if open_resp.status.code != 0 and "already" not in open_resp.status.message.lower():
            raise RuntimeError(open_resp.status.message)

    # --- insert ---
    def insert(
        self,
        collection: str,
        vectors: list[list[float]],
        metadata: list[dict] | None = None,
    ) -> None:
        """Insert vectors with optional JSON metadata payload."""
        vec_ids = []
        vec_objs = []
        payloads = []

        for i, vec_data in enumerate(vectors):
            vec_ids.append(vdss_types_pb2.VectorIdentifier(uuid=str(_uuid.uuid4())))
            vec_objs.append(vdss_types_pb2.Vector(
                data=vec_data, dimension=len(vec_data),
            ))
            meta = metadata[i] if metadata and i < len(metadata) else {}
            payloads.append(vdss_types_pb2.Payload(json=json.dumps(meta)))

        if len(vectors) == 1:
            resp = self._stub.UpsertVector(
                vdss_service_pb2.UpsertVectorRequest(
                    collection_name=collection,
                    vector_id=vec_ids[0],
                    vector=vec_objs[0],
                    payload=payloads[0],
                )
            )
            if resp.status.code != 0:
                raise RuntimeError(resp.status.message)
        else:
            resp = self._stub.BatchUpsert(
                vdss_service_pb2.BatchUpsertRequest(
                    collection_name=collection,
                    vector_ids=vec_ids,
                    vectors=vec_objs,
                    payloads=payloads,
                )
            )
            if resp.status.code != 0:
                raise RuntimeError(resp.status.message)

    # --- search ---
    def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search for nearest vectors, returning dicts with score + metadata."""
        query_vec = vdss_types_pb2.Vector(data=query, dimension=len(query))
        filter_json = json.dumps(filters) if filters else ""

        resp = self._stub.Search(
            vdss_service_pb2.SearchRequest(
                collection_name=collection,
                query=query_vec,
                top_k=top_k,
                filter_json=filter_json,
                with_payload=True,
                with_vector=False,
            )
        )
        if resp.status.code != 0:
            raise RuntimeError(resp.status.message)

        results = []
        for sr in resp.results:
            meta = {}
            if sr.payload and sr.payload.json:
                try:
                    meta = json.loads(sr.payload.json)
                except json.JSONDecodeError:
                    pass
            results.append({
                "id": sr.id.uuid or str(sr.id.u64_id),
                "score": float(sr.score),
                "metadata": meta,
            })
        return results


# ---------------------------------------------------------------------------
# VectorAIStore (public API — unchanged from mock-based version)
# ---------------------------------------------------------------------------


class VectorAIStore:
    """Wrapper around VectorAI DB for all vector operations.

    Supports two client backends:
    1. Native gRPC (_GrpcVectorAIClient) — used when grpc+stubs are available
    2. Injected mock client — used in unit tests

    Usage::

        store = VectorAIStore(host="localhost", port=5555)
        if store.health_check():
            store.store_person_embedding(clip_vec, person_id="S1", session_id="abc")
            match = store.find_person(clip_vec, threshold=0.85)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ):
        self._host = host or os.environ.get("VECTORAI_HOST", "localhost")
        self._port = port or int(os.environ.get("VECTORAI_PORT", "5555"))
        self._client: Any = None
        self._available = False

        # Try native gRPC first, then fall back to CortexClient SDK
        if _GRPC_AVAILABLE:
            try:
                client = _GrpcVectorAIClient(self._host, self._port)
                client.health()  # connectivity check
                self._client = client
                # Create collections (idempotent)
                for name, spec in _COLLECTIONS.items():
                    try:
                        client.create_collection(name, dimension=spec["dimension"])
                    except Exception:
                        pass  # may already exist
                self._available = True
                print(f"[vectorai] Connected via gRPC to {self._host}:{self._port}", flush=True)
                return
            except Exception as e:
                print(f"[vectorai] gRPC connection failed ({e})", flush=True)

        if CortexClient is not None:
            try:
                self._client = CortexClient(host=self._host, port=self._port)
                for name, spec in _COLLECTIONS.items():
                    try:
                        self._client.create_collection(
                            name, dimension=spec["dimension"], description=spec.get("description", ""),
                        )
                    except Exception:
                        pass
                self._available = True
                print(f"[vectorai] Connected via CortexClient to {self._host}:{self._port}", flush=True)
                return
            except Exception as e:
                print(f"[vectorai] CortexClient connection failed ({e})", flush=True)

        print("[vectorai] No VectorAI client available — disabled", flush=True)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if VectorAI is reachable."""
        if not self._available or self._client is None:
            return False
        try:
            self._client.health()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Person embeddings (dim=768, CLIP)
    # ------------------------------------------------------------------

    def store_person_embedding(
        self,
        embedding: np.ndarray,
        person_id: str,
        session_id: str,
        timestamp: float | None = None,
    ) -> None:
        """Store a person appearance embedding for cross-session re-ID."""
        if not self._available:
            return
        try:
            self._client.insert(
                "person_embeddings",
                vectors=[embedding.tolist()],
                metadata=[{
                    "person_id": person_id,
                    "session_id": session_id,
                    "timestamp": timestamp or time.time(),
                }],
            )
        except Exception as e:
            print(f"[vectorai] store_person_embedding failed: {e}", flush=True)

    def find_person(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> dict | None:
        """Search for a matching person by embedding similarity.

        Returns the best match above *threshold* as
        ``{"person_id": str, "session_id": str, "score": float}``
        or ``None`` if no match.
        """
        if not self._available:
            return None
        try:
            results = self._client.search(
                "person_embeddings",
                query=query_embedding.tolist(),
                top_k=top_k,
            )
            if not results:
                return None
            best = results[0]
            if best["score"] >= threshold:
                return {
                    "person_id": best["metadata"]["person_id"],
                    "session_id": best["metadata"].get("session_id", ""),
                    "score": best["score"],
                }
            return None
        except Exception as e:
            print(f"[vectorai] find_person failed: {e}", flush=True)
            return None

    # ------------------------------------------------------------------
    # Motion segments (dim=42, SRP features)
    # ------------------------------------------------------------------

    def store_motion_segment(
        self,
        features: np.ndarray,
        activity_label: str,
        session_id: str,
        person_id: str,
        risk_score: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        """Store a motion segment's feature vector with metadata."""
        if not self._available:
            return
        try:
            self._client.insert(
                "motion_segments",
                vectors=[features.tolist()],
                metadata=[{
                    "activity_label": activity_label,
                    "session_id": session_id,
                    "person_id": person_id,
                    "risk_score": risk_score,
                    "timestamp": timestamp or time.time(),
                }],
            )
        except Exception as e:
            print(f"[vectorai] store_motion_segment failed: {e}", flush=True)

    def find_similar_movements(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Find similar past motion segments by vector similarity.

        Returns list of ``{"score": float, "metadata": dict}`` sorted by
        descending similarity.
        """
        if not self._available:
            return []
        try:
            results = self._client.search(
                "motion_segments",
                query=query_features.tolist(),
                top_k=top_k,
                filters=filters,
            )
            return [
                {"score": r["score"], "metadata": r["metadata"]}
                for r in results
            ]
        except Exception as e:
            print(f"[vectorai] find_similar_movements failed: {e}", flush=True)
            return []

    # ------------------------------------------------------------------
    # Activity templates (dim=42, labeled reference movements)
    # ------------------------------------------------------------------

    def store_activity_template(
        self,
        features: np.ndarray,
        activity_name: str,
        source: str = "labeled_data",
    ) -> None:
        """Store a labeled activity template vector."""
        if not self._available:
            return
        try:
            self._client.insert(
                "activity_templates",
                vectors=[features.tolist()],
                metadata=[{
                    "activity_name": activity_name,
                    "source": source,
                }],
            )
        except Exception as e:
            print(f"[vectorai] store_activity_template failed: {e}", flush=True)

    def classify_activity(
        self,
        features: np.ndarray,
        threshold: float = 0.80,
        top_k: int = 3,
    ) -> tuple[str | None, float]:
        """Classify a motion segment by finding the nearest activity template.

        Returns ``(activity_name, confidence)`` where activity_name is None
        if no template exceeds the threshold.
        """
        if not self._available:
            return None, 0.0
        try:
            results = self._client.search(
                "activity_templates",
                query=features.tolist(),
                top_k=top_k,
            )
            if not results:
                return None, 0.0
            best = results[0]
            score = best["score"]
            if score >= threshold:
                return best["metadata"]["activity_name"], score
            return None, score
        except Exception as e:
            print(f"[vectorai] classify_activity failed: {e}", flush=True)
            return None, 0.0
