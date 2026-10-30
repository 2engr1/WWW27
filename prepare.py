#!/usr/bin/env python3
"""
Standalone EnterpriseRAG data preparation.

Features:
1. Load the original EnterpriseRAG dataset;
2. Validate the manifest, Gold questions, global document order, and dataset tiers;
3. Support document-level and chunk-level retrieval units;
4. Generate stable document IDs and chunk IDs;
5. Store the mapping between chunks and their parent documents;
6. Export queries.jsonl, corpus.jsonl, id_map.jsonl, qrels.tsv, and dataset.json.

Document mode requires only the Python standard library.
Chunk mode additionally requires tiktoken:
    pip install tiktoken

Examples:
    python prepare.py prepare \\
        --data /path/to/EnterpriseRAG \\
        --out /path/to/prepared_data \\
        --unit chunk \\
        --chunk-tokens 400 \\
        --overlap 80

    python prepare.py prepare \\
        --data /path/to/EnterpriseRAG \\
        --out /path/to/prepared_data_document \\
        --unit document
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import uuid


VERSION = "1.0.0"

# Fixed namespace: the same source document ID produces the same UUID across runs.
NS = uuid.UUID("154f289a-fef5-4c99-a2e1-09983f5e6109")


def rows(path: Path):
    """Read a JSONL file line by line; each non-empty line must be a JSON object."""
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: JSON object required")
                yield row


def dump(path: Path, obj):
    """Write an object as a human-readable UTF-8 JSON file."""
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def emit(file_obj, obj):
    """Write one record to a JSONL file."""
    file_obj.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def doc_uuid(doc_id: str) -> str:
    """
    Generate a stable UUID for a source document.

    UUIDv5 generates deterministic IDs:
    the same doc_id and namespace always produce the same UUID.
    """
    return str(uuid.uuid5(NS, str(doc_id)))


def fingerprint(path: Path):
    """Record input file size and modification time to determine whether prepared outputs can be reused."""
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def prepare(args):
    src = Path(args.data).resolve()
    out = Path(args.out).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {src}")
    if out == src or src in out.parents:
        raise ValueError("Output must be outside source dataset")

    # ---------- 1. Load and validate the dataset manifest ----------
    manifest_path = src / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiers = manifest["tiers"]

    labels = [str(t["tier"]) for t in tiers]
    if (
        len(set(labels)) != len(labels)
        or any("/" in label or label in (".", "..") for label in labels)
    ):
        raise ValueError("Invalid tiers")

    sizes = [int(t["documents"]) for t in tiers]
    if not sizes or any(x >= y for x, y in zip(sizes, sizes[1:])):
        raise ValueError("Tiers must be strictly increasing")

    # Input files included in the dataset version signature.
    input_files = [
        "manifest.json",
        "corpus.jsonl",
        "gold.jsonl",
        "order.txt",
    ] + [f"tiers/{label}/doc_ids.txt" for label in labels]

    signature = {
        "version": VERSION,
        "source": str(src),
        "unit": args.unit,
        "chunk_tokens": args.chunk_tokens,
        "overlap": args.overlap,
        "input_files": {
            name: fingerprint(src / name) for name in input_files
        },
        "manifest_sha256": sha(manifest_path),
    }

    # ---------- 2. Check whether completed preprocessing outputs can be reused ----------
    if out.exists():
        complete_marker = out / "prepared.json"
        if not complete_marker.exists():
            raise ValueError(
                "Output exists without completion marker; "
                "use a new output directory"
            )

        old = json.loads(complete_marker.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError(
                "Input/config changed; use a new output directory"
            )

        for relative_path, info in old["files"].items():
            output_file = out / relative_path
            if (
                not output_file.exists()
                or output_file.stat().st_size != info["bytes"]
            ):
                raise ValueError(f"Output missing/changed: {output_file}")

            if args.verify and sha(output_file) != info["sha256"]:
                raise ValueError(f"Output hash mismatch: {output_file}")

        print(f"Reusing prepared dataset: {out} (no conversion)")
        return

    # ---------- 3. Load and validate Gold queries ----------
    gold = list(rows(src / "gold.jsonl"))

    if len(gold) != args.expected_questions:
        raise ValueError(
            f"Expected {args.expected_questions} questions, got {len(gold)}"
        )

    question_ids = [q["question_id"] for q in gold]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question IDs")

    for question in gold:
        if (
            not question.get("expected_doc_ids")
            or not isinstance(question.get("gold_answer"), str)
            or not question["gold_answer"].strip()
        ):
            raise ValueError(
                f"Invalid question: {question.get('question_id')}"
            )

    gold_doc_ids = {
        doc_id
        for question in gold
        for doc_id in question["expected_doc_ids"]
    }

    # ---------- 4. Validate global document order and dataset tiers ----------
    order = (src / "order.txt").read_text(encoding="utf-8").splitlines()

    if len(order) != len(set(order)) or len(order) != sizes[-1]:
        raise ValueError("Invalid global order")

    for label, size in zip(labels, sizes):
        tier_ids = (
            src / "tiers" / label / "doc_ids.txt"
        ).read_text(encoding="utf-8").splitlines()

        # Each tier must be a prefix of the global order to guarantee strictly nested datasets.
        if tier_ids != order[:size] or not gold_doc_ids <= set(tier_ids):
            raise ValueError(
                f"Invalid nested tier/gold coverage: {label}"
            )

    # ---------- 5. Initialize the chunk tokenizer ----------
    encoder = None
    if args.unit == "chunk":
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "Chunk mode requires tiktoken. Install it with: pip install tiktoken"
            ) from exc

        encoder = tiktoken.get_encoding("cl100k_base")

    # Write to a staging directory first, then rename it atomically after all outputs are complete.
    stage = out.with_name(out.name + ".building")
    if stage.exists():
        raise ValueError(
            f"Incomplete previous build at {stage}; "
            "inspect/remove it or choose a new output"
        )

    stage.mkdir(parents=True)

    # ---------- 6. Convert query records and map Gold document IDs to stable UUIDs ----------
    queries = [
        {
            "question_id": q["question_id"],
            "question_type": q["question_type"],
            "query": q["question"],
            "relevant_doc_ids": list(
                dict.fromkeys(doc_uuid(doc_id) for doc_id in q["expected_doc_ids"])
            ),
            "expected_doc_ids": list(dict.fromkeys(q["expected_doc_ids"])),
            "gold_answer": q["gold_answer"],
            "answer_facts": q.get("answer_facts", []),
        }
        for q in gold
    ]

    retrieval_unit_counts = {label: 0 for label in labels}

    # ---------- 7. Create output files for each tier ----------
    with ExitStack() as stack:
        handles = {}

        for label in labels:
            tier_dir = stage / label
            tier_dir.mkdir()

            corpus_file = stack.enter_context(
                (tier_dir / "corpus.jsonl").open("w", encoding="utf-8")
            )
            mapping_file = stack.enter_context(
                (tier_dir / "id_map.jsonl").open("w", encoding="utf-8")
            )
            handles[label] = (corpus_file, mapping_file)

            with (tier_dir / "queries.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for query in queries:
                    emit(f, query)

            shutil.copyfile(src / "gold.jsonl", tier_dir / "gold.jsonl")
            shutil.copyfile(
                src / "tiers" / label / "doc_ids.txt",
                tier_dir / "doc_ids.txt",
            )

            # qrels uses stable parent-document UUIDs instead of source business IDs.
            with (tier_dir / "qrels.tsv").open(
                "w", encoding="utf-8"
            ) as f:
                for query in queries:
                    for relevant_id in query["relevant_doc_ids"]:
                        f.write(
                            f'{query["question_id"]}\t0\t{relevant_id}\t1\n'
                        )

        # ---------- 8. Iterate over the source corpus and create document or chunk retrieval units ----------
        processed_documents = 0

        for index, row in enumerate(rows(src / "corpus.jsonl")):
            if index >= len(order) or row["doc_id"] != order[index]:
                raise ValueError(
                    f"Corpus order mismatch at row {index + 1}"
                )

            text = row["text"]
            original_doc_id = row["doc_id"]
            parent_id = doc_uuid(original_doc_id)

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty text: {original_doc_id}")

            # Document mode: treat each full document as one retrieval unit.
            if encoder is None:
                chunks = [text]

            # Chunk mode: split text into token windows while preserving overlap.
            else:
                tokens = encoder.encode(text, disallowed_special=())
                chunks = []
                start = 0

                while start < len(tokens):
                    end = min(start + args.chunk_tokens, len(tokens))

                    # Adjust the window boundary to ensure valid UTF-8 decoding,
                    # without replacing invalid byte sequences with replacement characters.
                    while True:
                        try:
                            chunk = encoder.decode_bytes(
                                tokens[start:end]
                            ).decode("utf-8")
                            break
                        except UnicodeDecodeError:
                            end -= 1
                            if end <= start:
                                raise ValueError(
                                    "chunk-tokens too small for a Unicode character"
                                )

                    chunks.append(chunk)

                    if end == len(tokens):
                        break

                    start = max(start + 1, end - args.overlap)

                    # Ensure that the next window also starts at a valid UTF-8 boundary.
                    while start < end:
                        try:
                            encoder.decode_bytes(
                                tokens[start:end]
                            ).decode("utf-8")
                            break
                        except UnicodeDecodeError:
                            start += 1

            # ---------- 9. Generate stable retrieval-unit IDs, metadata, and parent-document mappings ----------
            for chunk_index, chunk_text in enumerate(chunks):
                if encoder is None:
                    # Document mode: the retrieval-unit ID is the parent-document ID.
                    unit_id = parent_id
                else:
                    # Chunk mode: derive the ID from the source document ID, chunk index, and chunk-content hash.
                    # The same document, chunking configuration, and chunk content produce the same ID.
                    chunk_hash = hashlib.sha256(
                        chunk_text.encode("utf-8")
                    ).hexdigest()

                    unit_id = str(
                        uuid.uuid5(
                            NS,
                            original_doc_id
                            + "\0"
                            + str(chunk_index)
                            + "\0"
                            + chunk_hash,
                        )
                    )

                metadata = {
                    "parent_id": parent_id,
                    "original_doc_id": original_doc_id,
                    "source": row.get("original_path", ""),
                    "source_type": row.get("source_type"),
                    "title": row.get("title", ""),
                    "chunk_index": chunk_index,
                    "original_metadata": row.get("metadata", {}),
                }

                # corpus.jsonl: retrieval units consumed directly by the BM25 and Dense index builders.
                corpus_record = {
                    "id": unit_id,
                    "text": chunk_text,
                    "metadata": metadata,
                }

                # id_map.jsonl: explicitly records the mapping between retrieval units and parent documents.
                mapping_record = {
                    "id": unit_id,
                    "parent_id": parent_id,
                    "original_doc_id": original_doc_id,
                    "chunk_index": chunk_index,
                }

                # Write the document to each tier whose size includes its position in the global order.
                for label, size in zip(labels, sizes):
                    if index < size:
                        corpus_file, mapping_file = handles[label]
                        emit(corpus_file, corpus_record)
                        emit(mapping_file, mapping_record)
                        retrieval_unit_counts[label] += 1

            processed_documents = index + 1
            if processed_documents % 10000 == 0:
                print(
                    f"Converted {processed_documents:,} documents",
                    flush=True,
                )

        if processed_documents != len(order):
            raise ValueError("Corpus incomplete")

    # ---------- 10. Generate configuration and validation metadata for each tier ----------
    for label, document_count in zip(labels, sizes):
        tier_dir = stage / label
        corpus_hash = sha(tier_dir / "corpus.jsonl")

        dump(
            tier_dir / "dataset.json",
            {
                "version": VERSION,
                "tier": label,
                "documents": document_count,
                "retrieval_units": retrieval_unit_counts[label],
                "unit": args.unit,
                "questions": len(queries),
                "gold_documents": len(gold_doc_ids),
                "collection": (
                    f"enterprise_{label}_{args.unit}_{corpus_hash[:12]}"
                ),
                "corpus_sha256": corpus_hash,
                "queries_sha256": sha(tier_dir / "queries.jsonl"),
                "metric_unit": "unique parent document",
                "chunk_tokens": (
                    args.chunk_tokens if encoder is not None else None
                ),
                "overlap": args.overlap if encoder is not None else None,
            },
        )

    # ---------- 11. Write the completion marker and output-file checksums ----------
    files = {
        str(path.relative_to(stage)): {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        }
        for path in stage.rglob("*")
        if path.is_file()
    }

    dump(
        stage / "prepared.json",
        {
            "signature": signature,
            "files": files,
        },
    )

    # Rename the staging directory to the final output directory after all files are written successfully.
    stage.rename(out)

    print(
        f"Completed: {out}; "
        f"retrieval units per tier: {retrieval_unit_counts}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Standalone EnterpriseRAG data preparation"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Convert source data into retrieval-ready datasets",
    )
    prepare_parser.add_argument("--data", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument(
        "--unit",
        choices=["document", "chunk"],
        default="document",
    )
    prepare_parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=400,
    )
    prepare_parser.add_argument(
        "--overlap",
        type=int,
        default=80,
    )
    prepare_parser.add_argument(
        "--expected-questions",
        type=int,
        default=470,
    )
    prepare_parser.add_argument(
        "--verify",
        action="store_true",
        help="Also verify SHA-256 of previously generated output files",
    )

    args = parser.parse_args()

    if not 0 <= args.overlap < args.chunk_tokens:
        parser.error("Require 0 <= overlap < chunk-tokens")

    prepare(args)


if __name__ == "__main__":
    main()
