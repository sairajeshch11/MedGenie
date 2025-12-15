from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from helpers.llm_client import ask_llm
import config as config
import prompts as prompts

from urgency import detect_urgency
from retrieval import retrieve_with_meta

_MED_TRIGGERS = [
    "side effects", "dose", "dosage", "interactions", "contraindications",
    "warnings", "how to take", "missed dose", "overdose", "medication", "drug",
    "ibuprofen", "acetaminophen", "paracetamol", "aspirin", "amoxicillin",
    "metformin", "atorvastatin", "lisinopril"
]

def is_medication_question(q: str) -> bool:
    t = q.lower()
    return any(k in t for k in _MED_TRIGGERS) or bool(re.search(r"\bmg\b|\bml\b|\bdose\b", t))


def answer_medication_question(query: str, top_k: int = 5) -> Dict[str, Any]:
    # Pull more context for medication questions
    pack = retrieve_with_meta(query, top_k=top_k, cand_k=20)
    ctx = pack["chunks"]
    diag = pack["diag"]

    sources: List[Tuple[str, str, str]] = []
    context_blocks: List[str] = []
    for i, c in enumerate(ctx, start=1):
        src = c.get("source", "")
        doc_id = c.get("title", "")
        chunk_id = c.get("id", "")
        sources.append((src, doc_id, chunk_id))
        context_blocks.append(f"[S{i}] ({src}) {c.get('text','')}")

    context = "\n\n".join(context_blocks)

    urgency, urgency_reason = detect_urgency(query, context)

    prompt = prompts.RAG_ANSWER_WITH_CITATIONS.format(question=query, context=context)

    answer = ask_llm(
        prompt,
        model=config.MODEL_DEFAULT,
        system=prompts.SYSTEM_DEFAULT,
        temperature=0.2,
    )

    # Confidence: keep it conservative
    dense = float(diag.get("dense_strength", 0.0))
    bm25  = float(diag.get("bm25_strength", 0.0))
    agr   = float(diag.get("agreement", 0.0))

    dense_n = 1.0 if dense >= 0.7 else max(0.0, dense / 0.7)
    bm25_n  = 1.0 if bm25 >= 5.0 else max(0.0, bm25 / 5.0)
    cite_n  = min(1.0, len(ctx) / 5.0)

    conf = 0.40 * dense_n + 0.25 * bm25_n + 0.20 * agr + 0.15 * cite_n
    conf = max(0.0, min(1.0, conf))
    conf_reason = f"cite={len(ctx)}, dense_strength={dense_n:.2f}, bm25_strength={bm25_n:.2f}, agreement={agr:.2f}"

    return {
        "answer": answer,
        "sources": sources,
        "confidence": conf,
        "confidence_reason": conf_reason,
        "urgency": urgency,
        "urgency_reason": urgency_reason,
    }
