#!/usr/bin/env python3
"""Compute parent-document retrieval metrics from main.py JSONL results."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import EVAL_KS
from utils import dump, emit, rows


def unique_parent_ids(records):
    """Deduplicate ranked retrieval units by parent document, preserving first rank."""
    result = []
    seen = set()
    for record in records:
        metadata = record.get("metadata") or {}
        parent = metadata.get("parent_id", record.get("doc_id"))
        if parent is not None and parent not in seen:
            seen.add(parent)
            result.append(parent)
    return result


def retrieval_metrics(ranked_parent_ids, gold_ids, k):
    """Calculate Recall@k, MRR@k, and binary nDCG@k."""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("Gold relevant document set cannot be empty")
    ranked = list(dict.fromkeys(ranked_parent_ids))[:k]
    relevance = [1 if item in gold else 0 for item in ranked]
    hits = sum(relevance)
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    reciprocal_rank = next((1.0 / (index + 1) for index, rel in enumerate(relevance) if rel), 0.0)
    return {
        f"recall@{k}": hits / len(gold),
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def evaluate_file(input_path, output_path=None):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve() if output_path else Path(str(input_path) + ".evaluated.jsonl")
    summary_path = Path(str(input_path) + ".summary.json")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == input_path:
        raise ValueError("Evaluated output must not overwrite raw run results")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Evaluation output already exists; choose new output paths or move previous files")

    per_query = []
    metric_totals = {}
    latency_totals = {}
    judge_totals = {}
    modes = set()
    run_configs = []
    for record in rows(input_path):
        modes.add(record.get("mode", "unknown"))
        run_configs.append(record.get("run_config", {}))
        gold = set(record.get("relevant_doc_ids", []))
        ranked_parents = record.get("ranked_parent_ids") or unique_parent_ids(record.get("ranked_docs", []))
        first_stage_parents = record.get("first_stage_parent_ids", [])
        dense_records = record.get("dense_top_candidates", [])
        dense_parents = list(dict.fromkeys(item.get("parent_id", item.get("doc_id")) for item in dense_records))
        scores = {}
        for k in EVAL_KS:
            scores.update(retrieval_metrics(ranked_parents, gold, k))
        scores["first_stage_recall@50"] = len(set(first_stage_parents[:50]) & gold) / len(gold) if gold else 0.0
        scores["dense_candidate_recall@1000"] = len(set(dense_parents) & gold) / len(gold) if gold else 0.0
        context_parents = set(record.get("context_parent_ids", []))
        scores["generation_context_recall@1"] = len(context_parents & gold) / len(gold) if gold else 0.0

        evaluated = dict(record)
        evaluated["metrics"] = scores
        per_query.append(evaluated)
        for key, value in scores.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
        for key, value in (record.get("latency_ms") or {}).items():
            latency_totals[key] = latency_totals.get(key, 0.0) + float(value)
        if record.get("judge"):
            for key in ("correctness", "completeness", "faithfulness"):
                judge_totals[key] = judge_totals.get(key, 0.0) + float(record["judge"][key])

    if not per_query:
        raise ValueError("No result records found")
    count = len(per_query)
    averages = {key: value / count for key, value in metric_totals.items()}
    if judge_totals:
        for key, value in judge_totals.items():
            averages[f"judge_{key}"] = value / count
    avg_latency = {key: value / count for key, value in latency_totals.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for record in per_query:
            emit(handle, record)

    run_config = run_configs[0] if run_configs else {}
    if len(modes) != 1:
        raise ValueError(f"Expected one mode per result file; found {sorted(modes)}")
    summary = {
        "questions": count,
        "mode": next(iter(modes)),
        "averages": averages,
        "avg_latency_ms": avg_latency,
        "evaluated_file": str(output_path),
        "top50_file": run_config.get("top50_output", str(Path(str(input_path) + ".top50.jsonl"))),
        "config": run_config,
        "protocol": {
            "dense_candidates": run_config.get("dense_candidates", 1000),
            "ns_candidates": run_config.get("ns_candidates", 50),
            "saved_ranking": run_config.get("saved_top_k", 50),
            "generation_context": run_config.get("generation_top_k", 1),
            "evaluation_k": list(EVAL_KS),
            "metric_unit": "unique parent document",
        },
    }
    dump(summary_path, summary)
    print(f"Evaluated {count} queries")
    print(f"Per-query metrics: {output_path}")
    print(f"Summary: {summary_path}")
    for key in ("mrr@10", "recall@10", "ndcg@10", "mrr@50", "recall@50", "ndcg@50"):
        if key in averages:
            print(f"{key}: {averages[key]:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw JSONL file generated by main.py")
    parser.add_argument("--out", default=None, help="Optional evaluated JSONL path")
    args = parser.parse_args()
    evaluate_file(args.input, args.out)

if __name__ == "__main__":
    main()
