"""Optional standard Cross-Encoder and NS-Rerank evidence verification."""
from __future__ import annotations

import math

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from models import RetrievedDoc, harmonic_mean_score


def _verify_device(model_wrapper, device: str) -> None:
    """Verify the underlying model is actually resident on the requested device."""
    model = getattr(model_wrapper, "model", model_wrapper)
    parameters_fn = getattr(model, "parameters", None)
    if not callable(parameters_fn):
        return  # permits simple test doubles
    parameters = list(parameters_fn())
    if not parameters:
        return
    expected = str(torch.device(device))
    actual = {str(parameter.device) for parameter in parameters}
    if actual != {expected}:
        raise RuntimeError(f"CrossEncoder device mismatch: expected {expected}, got {sorted(actual)}")


def _sigmoid(value: float) -> float:
    """Stable sigmoid used to put CE logits on a positive, bounded scale."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderReranker:
    """Conventional Cross-Encoder reranking over dense-retrieval candidates."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        _verify_device(self.model, self.device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        top_k: int = 10,
    ) -> list[RetrievedDoc]:
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = [
            RetrievedDoc(
                doc_id=doc.doc_id,
                text=doc.text,
                score=float(score),
                source="cross-encoder",
                metadata=doc.metadata,
                score_components={"cross_encoder": float(score)},
            )
            for doc, score in zip(candidates, np.asarray(scores).reshape(-1))
        ]
        reranked.sort(key=lambda doc: -doc.score)
        return reranked[:top_k]


class NSReranker:
    """Second-stage NS-Rerank: whole-document relevance + best local evidence.

    D(q,d) is Cross-Encoder relevance for the candidate text. E(q,d) is the
    maximum Cross-Encoder relevance over overlapping token windows. Raw CE logits
    are passed through sigmoid before harmonic aggregation so both components are
    positive and on the same bounded scale.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        batch_size: int = 32,
        window_tokens: int = 256,
        window_stride: int = 128,
    ):
        if window_tokens < 1:
            raise ValueError("window_tokens must be positive")
        if window_stride < 1 or window_stride > window_tokens:
            raise ValueError("window_stride must be in [1, window_tokens]")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)
        _verify_device(self.model, self.device)
        # Ensure predict() returns logits; sigmoid is applied exactly once below.
        if hasattr(self.model, "default_activation_function"):
            self.model.default_activation_function = torch.nn.Identity()
        self.batch_size = max(1, int(batch_size))
        self.window_tokens = int(window_tokens)
        self.window_stride = int(window_stride)
        self.tokenizer = self.model.tokenizer

    def _windows(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        if not token_ids:
            return [text or ""]
        if len(token_ids) <= self.window_tokens:
            return [text]
        result = []
        for start in range(0, len(token_ids), self.window_stride):
            segment = token_ids[start : start + self.window_tokens]
            if segment:
                result.append(self.tokenizer.decode(segment, skip_special_tokens=True))
            if start + self.window_tokens >= len(token_ids):
                break
        return result or [text]

    def _logits(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in np.asarray(scores).reshape(-1)]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedDoc],
        top_k: int = 10,
    ) -> list[RetrievedDoc]:
        if not candidates:
            return []

        # D(q,d): whole-candidate CE relevance (subject to the CE model's max length).
        whole_logits = self._logits([(query, candidate.text) for candidate in candidates])
        whole_scores = [_sigmoid(value) for value in whole_logits]

        # E(q,d): strongest CE match among overlapping local windows.
        window_pairs: list[tuple[str, str]] = []
        window_owners: list[int] = []
        for index, candidate in enumerate(candidates):
            for window in self._windows(candidate.text):
                window_pairs.append((query, window))
                window_owners.append(index)
        window_logits = self._logits(window_pairs)
        local_maxima = [0.0 for _ in candidates]
        for owner, logit in zip(window_owners, window_logits):
            local_maxima[owner] = max(local_maxima[owner], _sigmoid(logit))

        reranked = []
        for doc, whole_score, evidence_score in zip(candidates, whole_scores, local_maxima):
            combined = harmonic_mean_score(whole_score, evidence_score, epsilon=1e-8)
            reranked.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=combined,
                    source="ns-rerank",
                    metadata=doc.metadata,
                    score_components={
                        "whole_document_ce": whole_score,
                        "local_evidence_ce": evidence_score,
                        "ns_rerank": combined,
                    },
                )
            )
        reranked.sort(key=lambda doc: -doc.score)
        return reranked[:top_k]
