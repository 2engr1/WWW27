"""Create Qdrant clients with local disk persistence by default."""
from pathlib import Path


def make_client(local_path="./qdrant_local", url=None):
    from qdrant_client import QdrantClient
    if url:
        return QdrantClient(url=url)
    path = Path(local_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
