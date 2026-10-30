"""Central experiment protocol and mode definitions."""
from dataclasses import dataclass

DENSE_CANDIDATES = 1000
NS_CANDIDATES = 50
SAVE_TOP_K = 50
GENERATION_TOP_K = 1
EVAL_KS = (5, 10, 20, 50)

@dataclass(frozen=True)
class ModeSpec:
    uses_ns: bool
    ns_score_mode: str | None
    reranker: str | None
    description: str

MODES = {
    "dense": ModeSpec(False, None, None, "Dense retrieval only"),
    "dense-rerank": ModeSpec(False, None, "ce", "Dense retrieval followed by conventional Cross-Encoder reranking"),
    "ns": ModeSpec(True, "ns", "ns", "Bidirectional NS constraints followed by NS-Rerank"),
    "ns-ce": ModeSpec(True, "ns", "ce", "Bidirectional NS constraints followed by conventional Cross-Encoder reranking"),
    "ns-retrieval": ModeSpec(True, "ns", None, "Bidirectional NS retrieval without a reranker"),
    "ns-suf": ModeSpec(True, "suf", None, "Sufficiency-only NS retrieval"),
    "ns-nec": ModeSpec(True, "nec", None, "Necessity-only NS retrieval"),
}

NS_SUMMARY_MODES = (
    ("ns-retrieval", "NS-Retrieval"),
    ("ns-ce", "NS+CE"),
    ("ns", "NS"),
    ("ns-nec", "NS-Nec"),
    ("ns-suf", "NS-Suf"),
)
