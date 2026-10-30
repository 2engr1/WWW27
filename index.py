#!/usr/bin/env python3
"""
Standalone index builder for prepared EnterpriseRAG datasets.

This script builds one or both retrieval backends:
1. BM25: a persistent lexical retrieval index stored as a pickle file.
2. Dense: E5 passage embeddings stored in a persistent Qdrant collection.

Expected input:
    A prepared dataset directory produced by prepare.py, containing:
        corpus.jsonl
        dataset.json

Default output:
    DATASET/indexes/
        bm25.pkl
        bm25.json
        qdrant_local/
        dense.json

Examples:
    # Build both BM25 and E5 + Qdrant indexes.
    python index.py index \
        --dataset /path/to/hybrid_data_chunk/20000 \
        --backend both \
        --embed-model /path/to/e5-base-v2 \
        --batch-size 64

    # Build only BM25.
    python index.py index \
        --dataset /path/to/hybrid_data_chunk/20000 \
        --backend bm25

    # Build only Dense (E5 + Qdrant).
    python index.py index \
        --dataset /path/to/hybrid_data_chunk/20000 \
        --backend dense \
        --embed-model /path/to/e5-base-v2

    # Use a remote Qdrant server instead of local persistence.
    python index.py index \
        --dataset /path/to/hybrid_data_chunk/20000 \
        --backend dense \
        --qdrant-url http://127.0.0.1:6333

Dependencies:
    pip install rank-bm25 numpy sentence-transformers qdrant-client

The BM25 index is serialized using the project's retrieval.BM25Retriever
class. Keep the project's src/retrieval.py available when loading that
pickle file in the retrieval stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


def read_jsonl(path: Path):
    """Read a JSONL file and yield one JSON object per non-empty line."""
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            yield record


def write_json(path: Path, value):
    """Write a JSON object using UTF-8 encoding."""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_dataset(dataset_path: str):
    """Load dataset metadata and validate the corpus checksum."""
    dataset_dir = Path(dataset_path).expanduser().resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset directory does not exist: {dataset_dir}"
        )

    metadata_path = dataset_dir / "dataset.json"
    corpus_path = dataset_dir / "corpus.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_fields = ("corpus_sha256", "retrieval_units", "collection")
    missing_fields = [
        field for field in required_fields if field not in metadata
    ]
    if missing_fields:
        raise ValueError(
            f"Invalid dataset.json; missing fields: {missing_fields}"
        )

    actual_checksum = file_sha256(corpus_path)
    if actual_checksum != metadata["corpus_sha256"]:
        raise ValueError(
            "Corpus checksum mismatch. The corpus.jsonl file may have "
            "changed after preprocessing."
        )

    return dataset_dir, metadata


def load_project_modules(src_dir: str | None):
    """
    Add the project's src directory to sys.path and import its retrieval
    and Qdrant backend modules.

    Importing the original BM25Retriever class keeps the serialized BM25
    index compatible with the existing retrieval pipeline.
    """
    if src_dir:
        source_dir = Path(src_dir).expanduser().resolve()
    else:
        source_dir = Path(__file__).resolve().parent / "src"

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Project src directory not found: {source_dir}. "
            "Place index.py in the project root or pass --src-dir."
        )

    source_dir_string = str(source_dir)
    if source_dir_string not in sys.path:
        sys.path.insert(0, source_dir_string)

    try:
        from retrieval import BM25Retriever
    except ImportError as exc:
        raise ImportError(
            "Cannot import BM25Retriever from src/retrieval.py. "
            "Check the project source directory and install rank-bm25/numpy."
        ) from exc

    return BM25Retriever


def build_bm25_index(dataset_dir: Path, metadata: dict, bm25_class):
    """Build or reuse the persistent BM25 index."""
    index_dir = dataset_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "bm25.pkl"
    marker_path = index_dir / "bm25.json"
    corpus_checksum = metadata["corpus_sha256"]

    # Reuse the index only when its marker matches the current corpus.
    if index_path.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))

        if marker.get("corpus_sha256") == corpus_checksum:
            print(f"Reusing BM25 index: {index_path}")
            return index_path

    print("Loading corpus for BM25 indexing...")
    corpus = list(read_jsonl(dataset_dir / "corpus.jsonl"))

    if len(corpus) != int(metadata["retrieval_units"]):
        raise ValueError(
            "Corpus record count does not match dataset.json: "
            f"{len(corpus)} != {metadata['retrieval_units']}"
        )

    print(f"Building BM25 index for {len(corpus):,} retrieval units...")
    retriever = bm25_class(corpus)

    # Write to a temporary file first to avoid leaving a partial pickle.
    temporary_path = index_dir / "bm25.tmp"
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(
                retriever,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_path.replace(index_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    write_json(
        marker_path,
        {
            "corpus_sha256": corpus_checksum,
            "retrieval_units": len(corpus),
            "index_type": "rank_bm25.BM25Okapi",
        },
    )

    print(f"Saved BM25 index: {index_path}")
    return index_path


def build_dense_index(args, dataset_dir: Path, metadata: dict):
    """Encode corpus passages with E5 and store vectors in Qdrant."""
    try:
        from qdrant_client.http.models import (
            Distance,
            PointStruct,
            VectorParams,
        )
        from sentence_transformers import SentenceTransformer
        from qdrant_backend import make_client
    except ImportError as exc:
        raise ImportError(
            "Dense indexing requires sentence-transformers and qdrant-client. "
            "Install them with: pip install sentence-transformers qdrant-client"
        ) from exc

    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")

    index_dir = dataset_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Keep the collection name deterministic for the same dataset/model pair.
    model_hash = hashlib.sha256(
        args.embed_model.encode("utf-8")
    ).hexdigest()[:8]
    collection_name = f"{metadata['collection']}_{model_hash}"

    if args.qdrant_path:
        local_path = str(
            Path(args.qdrant_path).expanduser().resolve()
        )
    else:
        local_path = str(
            (index_dir / "qdrant_local").resolve()
        )

    qdrant_url = args.qdrant_url
    client = make_client(
        local_path=local_path,
        url=qdrant_url,
    )

    spec = {
        "corpus_sha256": metadata["corpus_sha256"],
        "model": args.embed_model,
        "collection": collection_name,
        "backend": "server" if qdrant_url else "local",
        "url": qdrant_url,
        "local_path": None if qdrant_url else local_path,
    }

    marker_path = index_dir / "dense.json"
    pending_path = index_dir / "dense.pending.json"

    try:
        # Reuse a completed collection only when its configuration and
        # exact point count match the current dataset.
        if marker_path.exists():
            old_marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )

            if old_marker.get("spec") != spec:
                raise ValueError(
                    "Dense index configuration changed. Use a separate "
                    "prepared dataset directory or Qdrant collection."
                )

            if client.collection_exists(collection_name):
                point_count = client.count(
                    collection_name=collection_name,
                    exact=True,
                ).count

                if point_count == int(metadata["retrieval_units"]):
                    print(
                        f"Reusing Qdrant collection: {collection_name}"
                    )
                    return collection_name

            raise ValueError(
                "The completed Qdrant index is missing or its point count "
                "has changed. Restore the database or use a new collection."
            )

        # Prevent accidentally writing into an unrelated existing collection.
        if client.collection_exists(collection_name):
            if (
                not pending_path.exists()
                or json.loads(
                    pending_path.read_text(encoding="utf-8")
                ) != spec
            ):
                raise ValueError(
                    "Refusing to modify an existing Qdrant collection "
                    "that is not associated with this indexing task."
                )
        else:
            write_json(pending_path, spec)

        print(f"Loading embedding model: {args.embed_model}")
        model = SentenceTransformer(args.embed_model)
        vector_size = model.get_sentence_embedding_dimension()

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        total_indexed = 0
        batch = []

        def upload_batch(records):
            """Encode and upload one batch of corpus passages."""
            vectors = model.encode(
                [
                    "passage: " + record["text"]
                    for record in records
                ],
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points = [
                PointStruct(
                    id=record["id"],
                    vector=vector.tolist(),
                    payload={
                        "text": record["text"],
                        "metadata": record.get("metadata", {}),
                    },
                )
                for record, vector in zip(records, vectors)
            ]

            client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points,
            )

        print(
            f"Encoding and indexing {metadata['retrieval_units']:,} "
            "retrieval units..."
        )

        for record in read_jsonl(dataset_dir / "corpus.jsonl"):
            batch.append(record)

            if len(batch) >= args.batch_size:
                upload_batch(batch)
                total_indexed += len(batch)
                batch = []

                if total_indexed % 1024 == 0:
                    print(
                        f"Indexed {total_indexed:,} retrieval units",
                        flush=True,
                    )

        if batch:
            upload_batch(batch)
            total_indexed += len(batch)

        expected_count = int(metadata["retrieval_units"])
        actual_count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        if actual_count != expected_count:
            raise ValueError(
                f"Qdrant point count mismatch: "
                f"expected {expected_count}, got {actual_count}"
            )

        write_json(
            marker_path,
            {
                "spec": spec,
                "points": actual_count,
                "vector_size": vector_size,
                "distance": "Cosine",
            },
        )
        pending_path.unlink(missing_ok=True)

        if qdrant_url:
            print(
                f"Saved Qdrant server index: {collection_name}"
            )
        else:
            print(
                f"Saved persistent local Qdrant index: "
                f"{local_path} ({collection_name})"
            )

        return collection_name

    finally:
        client.close()


def run_index(args):
    """Validate the prepared dataset and build the requested indexes."""
    dataset_dir, metadata = load_dataset(args.dataset)

    # Import the project's original BM25 class to preserve pickle compatibility.
    bm25_class = None
    if args.backend in ("bm25", "both"):
        bm25_class = load_project_modules(args.src_dir)

    print(f"Dataset: {dataset_dir}")
    print(f"Retrieval units: {metadata['retrieval_units']:,}")
    print(f"Backend: {args.backend}")

    if args.backend in ("bm25", "both"):
        build_bm25_index(
            dataset_dir=dataset_dir,
            metadata=metadata,
            bm25_class=bm25_class,
        )

    if args.backend in ("dense", "both"):
        # The Qdrant backend helper is also located in the project's src dir.
        if args.src_dir:
            source_dir = Path(args.src_dir).expanduser().resolve()
        else:
            source_dir = Path(__file__).resolve().parent / "src"

        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))

        build_dense_index(
            args=args,
            dataset_dir=dataset_dir,
            metadata=metadata,
        )


def main():
    """Parse command-line arguments and execute index construction."""
    parser = argparse.ArgumentParser(
        description="Build BM25 and/or E5 + Qdrant retrieval indexes."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Build retrieval indexes for a prepared dataset.",
    )
    index_parser.add_argument(
        "--dataset",
        required=True,
        help="Prepared tier directory containing corpus.jsonl and dataset.json.",
    )
    index_parser.add_argument(
        "--backend",
        choices=("bm25", "dense", "both"),
        default="both",
        help="Index backend to build.",
    )
    index_parser.add_argument(
        "--src-dir",
        default=None,
        help="Project src directory; defaults to ./src next to index.py.",
    )
    index_parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Local Qdrant storage path; defaults to DATASET/indexes/qdrant_local.",
    )
    index_parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Optional remote Qdrant server URL; overrides local mode.",
    )
    index_parser.add_argument(
        "--embed-model",
        default="intfloat/e5-large-v2",
        help="SentenceTransformer model name or local model directory.",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding and upload batch size.",
    )

    args = parser.parse_args()

    if args.command == "index":
        if args.batch_size < 1:
            parser.error("--batch-size must be greater than zero")
        run_index(args)


if __name__ == "__main__":
    main()
