"""Answer generation with inline citations from retrieved context."""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from models import RetrievedDoc


SYSTEM_PROMPT = """You are a careful technical assistant. Answer using ONLY the information in the provided context.

Rules:
1. Cite evidence with inline brackets [1], [2], ... referring to the numbered docs below.
2. If the context does not contain the answer, say "I don't know based on the provided documents" — do not hallucinate.
3. Do not include information not present in the context, even if you know it from training data.
4. Be concise. Prefer 1-3 paragraphs over long lists.
"""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[str]  # doc_ids referenced


def format_context(docs: list[RetrievedDoc]) -> tuple[str, list[str]]:
    lines = []
    ids = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"[{i}] (doc_id={d.doc_id})\n{d.text}\n")
        ids.append(d.doc_id)
    return "\n".join(lines), ids


def generate_answer(
    query: str,
    docs: list[RetrievedDoc],
    model: str = "gpt-4o-mini",
    client: OpenAI | None = None,
) -> RAGAnswer:
    client = client or OpenAI()
    context, ids = format_context(docs)

    user = f"Question: {query}\n\nContext:\n{context}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return RAGAnswer(
        answer=response.choices[0].message.content,
        sources=ids,
    )
