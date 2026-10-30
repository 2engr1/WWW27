"""Shared retrieval data structures."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str
    metadata: dict | None = None
    score_components: dict | None = None

def harmonic_mean_score(left: float, right: float, epsilon: float = 0.0) -> float:
    """Return the harmonic mean of two non-negative scores."""
    denominator = left + right + epsilon
    return (2.0 * left * right / denominator) if denominator > 0 else 0.0
