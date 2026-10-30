"""E5 + Qdrant dense candidate retrieval."""
from __future__ import annotations
from models import RetrievedDoc

class DenseRetriever:
    """Qdrant-backed dense retrieval using a SentenceTransformer encoder."""

    def __init__(
        self,
        collection: str,
        qdrant_url: str | None = None,
        qdrant_path: str = "./qdrant_local",
        model_name: str = "intfloat/e5-large-v2",
        device: str | None = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use dense retrieval.") from exc
        from qdrant_backend import make_client

        self.client = make_client(local_path=qdrant_path, url=qdrant_url)
        self.collection = collection
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        vector = self.encoder.encode(
            f"query: {query}", normalize_embeddings=True, device=self.device,
            show_progress_bar=False,
        ).tolist()
        if hasattr(self.client, "query_points"):
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
            )
        return [
            RetrievedDoc(
                doc_id=str(hit.id),
                text=(hit.payload or {}).get("text", ""),
                score=float(hit.score),
                source="dense",
                metadata=(hit.payload or {}).get("metadata"),
                score_components={"dense_cosine": float(hit.score)},
            )
            for hit in hits
        ]

    def close(self) -> None:
        self.client.close()
