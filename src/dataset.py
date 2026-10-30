"""Prepared EnterpriseRAG dataset loading and parent-document mapping."""
from __future__ import annotations
import json
from pathlib import Path
from models import RetrievedDoc
from utils import rows

class RetrievalDataset:
    def __init__(self, dataset_dir):
        self.root = Path(dataset_dir).expanduser().resolve()
        metadata_path = self.root / "dataset.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._queries = None
        self._id_map = None

    def load_queries(self):
        if self._queries is None:
            self._queries = list(rows(self.root / "queries.jsonl"))
        return self._queries

    def load_id_mapping(self):
        if self._id_map is None:
            mapping_path = self.root / "id_map.jsonl"
            self._id_map = {record["id"]: record for record in rows(mapping_path)} if mapping_path.exists() else {}
        return self._id_map

    @staticmethod
    def parent_id(doc):
        metadata = doc.metadata or {}
        return metadata.get("parent_id", doc.doc_id)

    def parent_ranking(self, docs):
        return list(dict.fromkeys(self.parent_id(doc) for doc in docs))

    def parent_ranking_from_records(self, records):
        parents = []
        seen = set()
        for record in records:
            metadata = record.get("metadata") or {}
            parent = metadata.get("parent_id", record.get("doc_id", record.get("id")))
            if parent not in seen:
                seen.add(parent)
                parents.append(parent)
        return parents

    def gold_ids(self, query_record):
        return set(query_record.get("relevant_doc_ids", []))

    def to_retrieved_docs(self, records):
        return [RetrievedDoc(
            doc_id=record["doc_id"],
            text=record.get("text", ""),
            score=float(record.get("score", 0.0)),
            source=record.get("source", "unknown"),
            metadata=record.get("metadata"),
            score_components=record.get("score_components"),
        ) for record in records]
