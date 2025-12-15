import re

from typing import Any, Dict, List, Tuple

from llm_project.helpers.llm_client import ask_llm
import llm_project.config as config
import llm_project.prompts as prompts

from llm_project.urgency import detect_urgency
from llm_project.retrieval import retrieve_with_meta
# --- lightweight retrieval gating (prevents off-topic symptom chunks) ---
_STOP = {
    "what","to","do","for","a","an","the","and","or","of","in","on","with","is","are","was","were",
    "i","you","we","they","he","she","it","my","your","their","our","me","us","them",
    "should","can","could","would","will","may","might","just","please","help"
}

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", (s or "").lower())

def _core_query_terms(q: str) -> set[str]:
    # keep only meaningful terms
    return {w for w in _tok(q) if len(w) >= 4 and w not in _STOP}

def _count_term(text: str, term: str) -> int:
    # whole-word count
    return len(re.findall(rf"\b{re.escape(term)}\b", (text or "").lower()))
# --- end retrieval gating ---


def _confidence_from_diag(diag: Dict[str, Any], cite_count: int) -> Tuple[float, str]:
    dense = float(diag.get("dense_strength", 0.0))
    bm25  = float(diag.get("bm25_strength", 0.0))
    agr   = float(diag.get("agreement", 0.0))

    # Normalize-ish heuristics (keep simple and stable)
    dense_n = 1.0 if dense >= 0.7 else max(0.0, dense / 0.7)
    bm25_n  = 1.0 if bm25 >= 5.0 else max(0.0, bm25 / 5.0)
    cite_n  = min(1.0, cite_count / 3.0)

    conf = 0.45 * dense_n + 0.25 * bm25_n + 0.20 * agr + 0.10 * cite_n
    conf = max(0.0, min(1.0, conf))

    reason = f"cite={cite_count}, dense_strength={dense_n:.2f}, bm25_strength={bm25_n:.2f}, agreement={agr:.2f}"
    return conf, reason


def answer_question(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "sources": [(source_name, doc_id, chunk_id), ...],
        "confidence": float,
        "confidence_reason": str,
        "urgency": "LOW"|"MEDIUM"|"HIGH",
        "urgency_reason": str
      }
    """
    pack = retrieve_with_meta(query, top_k=top_k, cand_k=12)
    ctx = pack["chunks"]
    # --- retrieval gating: keep chunks that mention core query terms ---
    terms = _core_query_terms(query)
    if terms and isinstance(ctx, list):
        scored = []
        for _c in ctx:
            _txt = (_c.get("text") or "")
            _hits = sum(_count_term(_txt, _t) for _t in terms)
            scored.append((_hits, _c))
        kept = [_c for (_h, _c) in scored if _h > 0]
        if kept:
            ctx = kept[:top_k]
        else:
            # nothing matched; fall back to original top_k
            ctx = ctx[:top_k]
    elif isinstance(ctx, list):
        ctx = ctx[:top_k]
    # --- end retrieval gating ---

    ### GATE ctx FOR GENERIC SYMPTOM QUERIES ###
    q_terms = _core_query_terms(query)
    if q_terms:
        gated = []
        for c in (ctx or []):
            t = (c.get("text") or "")
            title = (c.get("doc_title") or c.get("title") or "")
            title_l = title.lower()

            # keep if title mentions any core term
            if any(term in title_l for term in q_terms):
                gated.append(c)
                continue

            # otherwise require stronger evidence from text (>= 2 mentions of a core term)
            strong = False
            tl = t.lower()
            for term in q_terms:
                if _count_term(tl, term) >= 2:
                    strong = True
                    break
            if strong:
                gated.append(c)

        # only replace ctx if gating didn't kill everything
        if gated:
            ctx = gated
    ### END GATE ###
    diag = pack["diag"]

    # Build context + sources
    # IMPORTANT: return dict sources including chunk text so the UI can format fields RAG-only.
    sources: List[Dict[str, Any]] = []
    context_blocks: List[str] = []
    for i, c in enumerate(ctx, start=1):
        src_name = c.get('source', '')
        # Keep legacy names but normalize meaning:
        # - doc_id should be a stable document identifier if available
        # - chunk_id should be a stable chunk identifier
        doc_id = c.get('doc_id') or c.get('document_id') or c.get('title') or ''
        chunk_id = c.get('chunk_id') or c.get('id') or ''
        chunk_text = c.get('text', '') or ''
        url = c.get('url') or c.get('source_url') or c.get('link') or ''
        title = c.get('title') or c.get('doc_title') or ''

        sources.append({
            'source': src_name,
            'doc_id': doc_id,
            'chunk_id': chunk_id,
            'title': title,
            'url': url,
            'text': chunk_text,
        })
        context_blocks.append(f"[S{i}] ({src_name}) {chunk_text}")

    context = "\n\n".join(context_blocks)
    urgency, urgency_reason = detect_urgency(query)

    # Medication mode (optional): import lazily to avoid circular imports
    try:
        from llm_project.medications import is_medication_question, answer_medication_question
        if is_medication_question(query):
            med_res = answer_medication_question(query, top_k=top_k)
            # keep urgency/confidence from med pipeline
            return med_res
    except Exception:
        # If meds pipeline isn't ready, just fall back to normal RAG
        pass

    prompt = prompts.RAG_ANSWER_WITH_CITATIONS.format(question=query, context=context)

    answer = ask_llm(
        prompt,
        model=config.MODEL_DEFAULT,
        system=prompts.SYSTEM_DEFAULT,
        temperature=0.2,
    )

    conf, conf_reason = _confidence_from_diag(diag, cite_count=len(ctx))

    return {
        "answer": answer,
        "sources": sources,
        "confidence": conf,
        "confidence_reason": conf_reason,
        "urgency": urgency,
        "urgency_reason": urgency_reason,
    }
