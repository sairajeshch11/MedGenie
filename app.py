import os
import sys
import streamlit as st


import json
import hashlib
from typing import Any, Dict, List, Optional

# Ensure parent folder is importable (so we can import /content/llm_project/nearby_hospitals.py)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Ensure project root (/content/llm_project) is on PYTHONPATH so `import llm_project.*` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rag import answer_question
from nearby_hospitals import find_nearby_hospitals
import re



# === LLM SOURCE FORMATTER HELPERS (RAG-ONLY) START ===

def _stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _truncate(text: str, max_chars: int = 2200) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:max_chars]


def _safe_fallback_item() -> Dict[str, str]:
    return {
        "title": "Not available",
        "url": "Not available",
        "why": "No relevant info found in retrieved context.",
        "excerpt": "No relevant excerpt available.",
    }


def _coerce_item(item: Any) -> Dict[str, str]:
    """Ensure shape + string types, enforce safe defaults."""
    base = _safe_fallback_item()
    if not isinstance(item, dict):
        return base

    title = item.get("title")
    url = item.get("url")
    why = item.get("why")
    excerpt = item.get("excerpt")

    out = {
        "title": str(title).strip() if isinstance(title, str) and title.strip() else base["title"],
        "url": str(url).strip() if isinstance(url, str) and url.strip() else base["url"],
        "why": str(why).strip() if isinstance(why, str) and why.strip() else base["why"],
        "excerpt": str(excerpt).strip() if isinstance(excerpt, str) and excerpt.strip() else base["excerpt"],
    }

    # Hard cap excerpt length for UI
    if len(out["excerpt"]) > 380:
        out["excerpt"] = out["excerpt"][:377].rstrip() + "..."

    return out


def _extract_json(text: str) -> Optional[Any]:
    """
    Tries to parse strict JSON.
    Also handles the common case where the model wraps JSON in extra text.
    """
    if not text:
        return None

    text = text.strip()

    # Fast path: strict JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Best-effort: find first JSON array/object substring
    first_obj = text.find("{")
    first_arr = text.find("[")
    if first_obj == -1 and first_arr == -1:
        return None
    start = first_obj if first_arr == -1 else (first_arr if first_obj == -1 else min(first_obj, first_arr))

    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)

    if start is None or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def llm_format_sources_single_call(
    question: str,
    sources_payload: List[Dict[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    max_chunk_chars: int = 2200,
) -> List[Dict[str, str]]:
    """
    One LLM call to format all source cards.
    STRICTLY RAG-ONLY: model receives only question + chunk_text + optional known_title/known_url.
    Returns list aligned to sources_payload order.
    Safe fallback on any failure.
    """

    # No sources => nothing to do
    if not sources_payload:
        return []

    # Build a stable cache key per question + payload content
    payload_for_hash = {
        "q": question or "",
        "sources": [
            {
                "provider": s.get("provider", ""),
                "doc_id": s.get("doc_id", ""),
                "chunk_id": s.get("chunk_id", ""),
                "known_title": s.get("known_title", ""),
                "known_url": s.get("known_url", ""),
                "chunk_text_hash": _stable_hash(s.get("chunk_text", "") or ""),
            }
            for s in sources_payload
        ],
    }
    cache_key = _stable_hash(json.dumps(payload_for_hash, sort_keys=True))

    # Cache storage (works even without st.cache_data)
    if "formatted_sources_cache" not in st.session_state:
        st.session_state["formatted_sources_cache"] = {}
    if cache_key in st.session_state["formatted_sources_cache"]:
        return st.session_state["formatted_sources_cache"][cache_key]

    # If no API key, fail safe immediately
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None

    if not api_key:
        fallback = [_safe_fallback_item() for _ in sources_payload]
        st.session_state["formatted_sources_cache"][cache_key] = fallback
        return fallback

    # Prepare sanitized sources for the prompt (truncate chunk text)
    prompt_sources = []
    for s in sources_payload:
        prompt_sources.append(
            {
                "provider": s.get("provider", ""),
                "doc_id": s.get("doc_id", ""),
                "chunk_id": s.get("chunk_id", ""),
                "known_title": s.get("known_title", "") or "",
                "known_url": s.get("known_url", "") or "",
                "chunk_text": _truncate(s.get("chunk_text", "") or "", max_chars=max_chunk_chars),
            }
        )

    system_msg = (
        "You format UI fields for retrieved sources.\n"
        "Rules:\n"
        "1) You MAY generate a short, user-friendly title based on the user question (3–8 words).\n"
        "2) Why: explain why this source is shown using ONLY chunk_text relevance (no outside facts). If indirect, say it may be indirectly related.\n"
        "3) Do NOT invent URLs. If known_url is missing/empty, return url exactly as: \"Not available\".\n"
        "4) Excerpt must be 1–3 sentences derived ONLY from chunk_text (cleaned), ideally <= 350 chars.\n"
        "5) Return STRICT JSON only. No markdown. No commentary.\n"
        "Output schema: array with same length and order as input."
    )
    user_msg = {
        "question": question or "",
        "sources": prompt_sources,
        "output_schema": [
            {
                "title": "string",
                "url": "string",
                "why": "string",
                "excerpt": "string"
            }
        ]
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
            ],
            temperature=0.2,
        )

        raw = resp.choices[0].message.content if resp and resp.choices else ""
        parsed = _extract_json(raw)

        # Must be a list with same length; else fallback
        if not isinstance(parsed, list) or len(parsed) != len(sources_payload):
            out = [_safe_fallback_item() for _ in sources_payload]
            st.session_state["formatted_sources_cache"][cache_key] = out
            return out

        out = [_coerce_item(x) for x in parsed]
        st.session_state["formatted_sources_cache"][cache_key] = out
        return out

    except Exception:
        out = [_safe_fallback_item() for _ in sources_payload]
        st.session_state["formatted_sources_cache"][cache_key] = out
        return out

# === LLM SOURCE FORMATTER HELPERS (RAG-ONLY) END ===


st.set_page_config(page_title="MedGenie", page_icon="🩺", layout="wide")
st.title("MedGenie")
st.caption("Hybrid retrieval (BM25 + FAISS) + grounded answers with citations.")


# ---------------- Sidebar tools (Hospitals) ----------------
with st.sidebar:
    st.header("Tools")
    st.caption("Find nearby hospitals (city / ZIP / address)")

    st.markdown("**Location**")
    loc = st.text_input(
        "Enter a city/ZIP/address",
        value="57069",
        label_visibility="collapsed",
        help="Examples: 57069, Sioux Falls SD, Visakhapatnam, 530016",
    )
    st.caption("Examples: 57069, Sioux Falls SD, Visakhapatnam, 530016")

    st.markdown("**Search radius (miles)**")
    radius_miles = st.slider("Radius (miles)", 1, 50, 10, label_visibility="collapsed")

    st.markdown("**How many results**")
    k = st.slider("Results", 3, 10, 5, label_visibility="collapsed")

    col_a, col_b = st.columns(2)
    find_clicked = col_a.button("Find hospitals", use_container_width=True)
    clear_clicked = col_b.button("Clear", use_container_width=True)

    if "hospitals" not in st.session_state:
        st.session_state.hospitals = []
    if "hospital_error" not in st.session_state:
        st.session_state.hospital_error = ""

    if clear_clicked:
        st.session_state.hospitals = []
        st.session_state.hospital_error = ""

    if find_clicked:
        st.session_state.hospital_error = ""
        with st.spinner("Searching..."):
            try:
                radius_km = float(radius_miles) * 1.60934
                try:
                    hospitals = find_nearby_hospitals(loc, radius_km=radius_km, max_results=k)
                except TypeError:
                    hospitals = find_nearby_hospitals(loc, radius_km, k)

                st.session_state.hospitals = hospitals or []
                if not st.session_state.hospitals:
                    st.session_state.hospital_error = (
                        "No hospitals found. Try a different location or increase the radius."
                    )
            except Exception:
                st.session_state.hospitals = []
                st.session_state.hospital_error = (
                    "Hospital search failed. Try again or change the location format."
                )

    st.markdown("---")
    st.caption(
        "Hospital results use OpenStreetMap (Nominatim). "
        "Results may be incomplete. For emergencies, call local emergency services."
    )

    if st.session_state.hospital_error:
        st.warning(st.session_state.hospital_error)

    hospitals = st.session_state.hospitals
    st.write(f"**Found:** {len(hospitals)}")

    for h in hospitals:
        if isinstance(h, dict):
            name = (h.get("name") or "Hospital").strip()
            addr = (h.get("address") or h.get("display_name") or "").strip()

            q = (name + " " + addr).replace(" ", "+")
            gmaps_url = f"https://www.google.com/maps/search/?api=1&query={q}"

            st.markdown(f"**{name}**")

            if addr:
                short = addr if len(addr) <= 110 else addr[:110] + "…"
                st.caption(short)
                with st.expander("Full address"):
                    st.write(addr)

            try:
                st.link_button("View on Google Maps", gmaps_url)
            except Exception:
                st.markdown(f"[View on Google Maps]({gmaps_url})")

            st.markdown("---")
        else:
            st.write(f"- {h}")

# ---------------- Chat UI ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask a medical question (demo).")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = answer_question(query, top_k=5)

        st.markdown(res.get("answer", ""))

        st.write("")
        st.write(f"**Urgency:** {res.get('urgency','')}")
        st.write(f"**Reason:** {res.get('urgency_reason','')}")
        st.write(f"**Confidence:** {res.get('confidence',0):.3f}")

        st.write("")
        st.subheader("Sources")

        # -------- Sources (user-friendly) --------
        def _clean_excerpt(s: str, max_chars: int = 320) -> str:
            s = (s or '').replace('\n', ' ')
            s = ' '.join(s.split())
            return (s[:max_chars] + '…') if len(s) > max_chars else s

        sources = res.get('sources', [])
        if sources:
            with st.expander('Sources', expanded=False):
                # --- LLM formatted source fields (RAG-only) ---

                # --- chunk text resolver (best-effort) ---
                def _resolve_chunk_text(_s, _doc_id="", _chunk_id=""):
                    # 1) If text already present on the source
                    if isinstance(_s, dict):
                        _t = (_s.get("chunk_text") or _s.get("excerpt") or _s.get("text") or "").strip()
                        if _t:
                            return _t
                
                    # 2) Try known helper functions that may exist elsewhere in the app
                    for _fn_name in ("get_chunk_text", "fetch_chunk_text", "load_chunk_text", "get_chunk_by_id", "get_chunk"):
                        _fn = globals().get(_fn_name)
                        if callable(_fn):
                            try:
                                # Try chunk_id first
                                if _chunk_id:
                                    _t = _fn(_chunk_id)
                                    if isinstance(_t, str) and _t.strip():
                                        return _t.strip()
                                # Try doc_id fallback
                                if _doc_id:
                                    _t = _fn(_doc_id)
                                    if isinstance(_t, str) and _t.strip():
                                        return _t.strip()
                            except Exception:
                                pass
                
                    # 3) Try common in-memory stores (dicts) that apps use for chunks
                    for _store_name in ("chunk_store", "CHUNK_STORE", "chunks", "CHUNKS", "doc_store", "DOC_STORE", "chunk_text_store"):
                        _store = globals().get(_store_name)
                        if isinstance(_store, dict):
                            for _key in (_chunk_id, _doc_id):
                                if _key and _key in _store:
                                    _val = _store.get(_key)
                                    # If dict, try typical keys
                                    if isinstance(_val, dict):
                                        _t = (_val.get("text") or _val.get("chunk_text") or _val.get("excerpt") or "").strip()
                                        if _t:
                                            return _t
                                    elif isinstance(_val, str) and _val.strip():
                                        return _val.strip()
                
                    return ""
                # --- end chunk text resolver ---
                # Build payload once and format in a single LLM call (cached in llm_format_sources_single_call)
                try:
                    _question_for_sources = question if 'question' in globals() else st.session_state.get('last_question', '')
                except Exception:
                    _question_for_sources = st.session_state.get('last_question', '')
                
                sources_payload = []
                for _s in (sources or []):
                    _provider = 'Source'
                    _known_title = ''
                    _known_url = ''
                    _doc_id = ''
                    _chunk_id = ''
                    _chunk_text = ''
                
                    if isinstance(_s, dict):
                        _provider = _s.get('source') or _s.get('provider') or _provider
                        _known_title = _s.get('title') or _s.get('doc_title') or ''
                        _known_url = _s.get('url') or _s.get('link') or _s.get('source_url') or ''
                        _doc_id = _s.get('doc_id') or _s.get('document_id') or _s.get('id') or ''
                        _chunk_id = _s.get('chunk_id') or _s.get('chunk') or ''
                        _chunk_text = _resolve_chunk_text(_s, _doc_id, _chunk_id)
                    elif isinstance(_s, tuple) and len(_s) >= 3:
                        _provider, _doc_id, _chunk_id = _s[0], _s[1], _s[2]
                        _known_title = _doc_id or ''
                        _chunk_text = ''
                
                    sources_payload.append({
                        'provider': _provider,
                        'doc_id': _doc_id,
                        'chunk_id': _chunk_id,
                        'known_title': _known_title,
                        'known_url': _known_url,
                        'chunk_text': _chunk_text,
                                            'matched_terms': [t for t in (re.findall(r"[a-zA-Z]+", (_question_for_sources or '').lower())) if t and t in (_chunk_text or '').lower()],
                        'match_hits': sum((_chunk_text or '').lower().count(t) for t in set(re.findall(r"[a-zA-Z]+", (_question_for_sources or '').lower()))),
})
                
                formatted_sources = llm_format_sources_single_call(_question_for_sources, sources_payload) if sources_payload else []
                # --- end LLM formatted source fields ---

                for idx, item in enumerate(sources, start=1):
                    provider = 'Source'
                    title = 'Untitled source'
                    url = None
                    doc_id = ''
                    chunk_id = ''
                    excerpt = ''
                    why = 'This source was used to help answer your question.'

                    if isinstance(item, dict):
                        provider = item.get('source') or item.get('provider') or provider
                        title = item.get('title') or item.get('doc_title') or title
                        url = item.get('url') or item.get('link') or item.get('source_url')
                        doc_id = item.get('doc_id') or item.get('document_id') or item.get('id') or ''
                        chunk_id = item.get('chunk_id') or item.get('chunk') or ''
                        excerpt = item.get('excerpt') or item.get('text') or ''
                        why = item.get('why') or why
                    elif isinstance(item, tuple) and len(item) >= 3:
                        provider, doc_id, chunk_id = item[0], item[1], item[2]
                        title = doc_id or title

                    # --- apply LLM formatted fields ---
                    _i = (idx - 1)  # idx starts at 1
                    _fmt = formatted_sources[_i] if (_i < len(formatted_sources)) else _safe_fallback_item()
                    # Override display fields only (keep doc_id/chunk_id for Technical details)
                    title = _fmt.get('title', title)
                    url = _fmt.get('url', url)
                    why = _fmt.get('why', why)
                    excerpt = _fmt.get('excerpt', excerpt)
                    # --- RAG-only fallback for 'why' when excerpt exists but LLM left why empty/fallback ---
                    _base = _safe_fallback_item()
                    if (why == _base.get('why')) and (excerpt and excerpt != _base.get('excerpt')):
                        try:
                            _q = (_question_for_sources or "").lower()
                        except Exception:
                            _q = ""
                        _q_words = set(re.findall(r"[a-zA-Z]+", _q))
                        _e_words = set(re.findall(r"[a-zA-Z]+", (excerpt or "").lower()))
                        _overlap = [w for w in sorted(_q_words & _e_words) if len(w) > 3][:6]
                        if _overlap:
                            why = "Excerpt includes terms from your question: " + ", ".join(_overlap) + "."
                        else:
                            why = "Excerpt was retrieved from the knowledge base; relevance may be indirect."
                    # --- end fallback ---

                    # --- end apply LLM formatted fields ---

                    short_id = (doc_id.split('_')[-1][:10] if doc_id else '')
                    header = f"[S{idx}] {provider}" + (f" | {short_id}" if short_id else "")
                    with st.expander(header, expanded=False):
                        st.markdown('**Title:** ' + str(title))
                        st.markdown('**Link:** ' + (str(url) if url else 'Not available'))
                        st.markdown('**Why it’s shown:** ' + str(why))
                        if excerpt:
                            st.markdown('**Excerpt:**')
                            st.info(_clean_excerpt(str(excerpt)), icon=None)
                        with st.expander('Technical details', expanded=False):
                            if doc_id: st.code('Document ID: ' + str(doc_id))
                            if chunk_id: st.code('Chunk ID: ' + str(chunk_id))
    # store assistant message text only (clean chat history)
    st.session_state.messages.append({"role": "assistant", "content": res.get("answer", "")})
