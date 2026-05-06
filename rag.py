"""
Retrieval-Augmented Generation: Chroma retrieval + Gemini generation with citations.

SMAI Assignment 3 — T10.1 Passport Assistant (Tier 1). Pre-trained embeddings only; no training.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL_NAME,
    TOP_K,
)

# Load .env from project root (directory containing this file), not only CWD
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()  # CWD fallback for notebooks / alternate launch paths

_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load the shared SentenceTransformer model (same as ingest)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_chroma_client() -> chromadb.PersistentClient:
    """Lazy-load Chroma persistent client."""
    global _chroma_client
    if _chroma_client is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def reset_runtime_caches() -> None:
    """
    Clear lazy Chroma/embedder singletons after rebuilding the index (e.g. ingest).
    Call this after `ingest.main()` so the next query uses the new collection.
    """
    global _embedder, _chroma_client
    _embedder = None
    _chroma_client = None


def get_collection():
    """Return the document collection or None if ingest has not created it."""
    client = get_chroma_client()
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return None


def vector_store_document_count() -> int:
    """Return number of stored chunks, or 0 if collection missing/empty."""
    collection = get_collection()
    if collection is None:
        return 0
    try:
        return int(collection.count())
    except Exception:
        return 0


def _get_gemini_api_key() -> str | None:
    """Read API key from environment; strip whitespace / newlines from .env."""
    raw = os.getenv("GEMINI_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    return key if key else None


def _safe_page_int(meta: dict, key: str = "page") -> int:
    """Parse page number from Chroma metadata without crashing on bad types."""
    v = meta.get(key, 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def retrieve_context(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed query, query ChromaDB, return chunks with:
    text, source (PDF filename), page, chunk_index.
    """
    collection = get_collection()
    if collection is None:
        raise RuntimeError(
            "Vector database collection not found. Run `python ingest.py` after adding PDFs."
        )

    if vector_store_document_count() == 0:
        raise RuntimeError(
            "Vector database is empty. Run `python ingest.py` after adding official PDFs."
        )

    embedder = get_embedder()
    query_embedding = embedder.encode(query, convert_to_numpy=True).tolist()

    # Avoid requesting more neighbors than exist (portable across Chroma versions)
    collection_count = vector_store_document_count()
    n_results = min(int(top_k), max(1, collection_count))

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]

    rows: list[dict] = []
    for doc_text, meta in zip(documents[0], metadatas[0]):
        if doc_text is None:
            continue
        source = (meta or {}).get("source", "unknown.pdf")
        chunk_idx_val = (meta or {}).get("chunk_index", 0)
        page_int = _safe_page_int(meta or {}, "page")
        try:
            chunk_int = int(chunk_idx_val)
        except (TypeError, ValueError):
            chunk_int = 0

        rows.append(
            {
                "text": doc_text,
                "source": str(source),
                "page": page_int,
                "chunk_index": chunk_int,
            }
        )

    return rows


def format_context(contexts: list[dict]) -> str:
    """Format retrieved chunks for the LLM (internal labels — must not appear in the answer)."""
    if not contexts:
        return ""

    blocks: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        src = ctx.get("source", "unknown")
        page = ctx.get("page", "?")
        body = ctx.get("text", "").strip()
        # "Excerpt" avoids the model echoing "SOURCE 1" in user-facing text.
        blocks.append(f"--- Excerpt {i} (file: {src}, page {page}) ---\n{body}")
    return "\n\n".join(blocks)


def build_prompt(question: str, contexts: list[dict], language: str) -> str:
    """
    Strict RAG prompt. language: 'en' or 'hi' (Hindi answers when hindi=True in generate_answer).
    """
    context_block = format_context(contexts)

    if language == "hi":
        lang_line = (
            "LANGUAGE: दें: साफ़ हिंदी (देवनागरी). आवश्यक हो तो आधिकारिक शब्द अंग्रेज़ी में रखें।\n\n"
        )
    else:
        lang_line = "LANGUAGE: Respond in English.\n\n"

    return f"""You are the "Passport Seva RAG Assistant". Answer ONLY using the official PDF excerpts below.

STRICT RULES (anti-hallucination):
- Use ONLY information supported by the excerpts. Do NOT use general knowledge or the web.
- Do NOT guess fees, deadlines, eligibility, required documents, or legal/government rules unless the excerpts state them.
- If the excerpts do not contain enough to answer, say exactly: "I could not find this in the official documents I have."
- Remind the user to verify the latest details on the official Passport Seva portal.
- Do not give individualized legal advice beyond what the excerpts say.

Answer guidelines (polished, citizen-friendly):
- Write clearly for a general reader. Use short paragraphs and numbered steps when the excerpts describe a process.
- Summarize naturally. If multiple excerpts repeat the same idea, merge it once—do not sound like a dump of snippets.
- Do NOT mention internal labels such as "Excerpt 1", "Excerpt 2", "SOURCE 1", or any "Excerpt N" / "SOURCE N" text in your answer.
- Do NOT paste bracketed citation tags into the answer body.
- Include only sections that have real content from the excerpts:
  - If there are clear steps, use a **Steps** (or "Steps to apply:") section with numbered steps.
  - If the excerpts list required documents, include a **Required documents** section; otherwise **omit that entire section** (do not write "not found" or "not stated" for documents).
  - If helpful cautions appear in the excerpts, add **Important notes**; otherwise omit that section.
- Aim for a helpful, professional length: use the details present in the excerpts rather than one-line answers when the material supports more.
- **Do NOT** add a "Sources", "References", or "Citations" section in your reply. The application shows official PDF filenames and pages separately below your message.

CONTEXT (Official PDF excerpts):

{context_block}

---

USER QUESTION:
{question}

{lang_line}Produce the final answer following the guidelines above. If the question cannot be answered from the excerpts, use only the exact "could not find" sentence and the portal reminder.
"""


def format_merged_sources_markdown(citations: list[dict]) -> str:
    """
    Single formatted list grouped by PDF: one bullet per file, pages merged (sorted, unique).
    Used as the one canonical "Sources" block in the UI.
    """
    if not citations:
        return ""

    from collections import defaultdict

    by_file: dict[str, set[int]] = defaultdict(set)
    order: list[str] = []
    for c in citations:
        fn = str(c.get("source", "unknown.pdf"))
        if fn not in order:
            order.append(fn)
        p = _safe_page_int(c, "page")
        if p > 0:
            by_file[fn].add(p)

    lines: list[str] = []
    for fn in order:
        pages = sorted(by_file.get(fn, set()))
        if not pages:
            lines.append(f"- `{fn}`")
        elif len(pages) == 1:
            lines.append(f"- `{fn}` (page {pages[0]})")
        else:
            lines.append(f"- `{fn}` (pages {', '.join(str(p) for p in pages)})")
    return "\n".join(lines)


def dedupe_citations(contexts: list[dict]) -> list[dict]:
    """Deduplicate (PDF filename, page) pairs; preserve order."""
    seen: set[tuple[str, int]] = set()
    citations: list[dict] = []
    for ctx in contexts:
        page_n = _safe_page_int(ctx, "page")
        key = (str(ctx.get("source", "")), page_n)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"source": key[0], "page": page_n})
    return citations


def generate_answer(
    question: str,
    hindi: bool = False,
) -> tuple[str, list[dict], str | None]:
    """
    Retrieve contexts, call Gemini, return (answer, citations, optional_user_error).

    citations: deduplicated list of {{source: filename, page: int}}.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        msg = (
            "Missing GEMINI_API_KEY. Set it in `.env` and add your key."
        )
        return "", [], msg

    if get_collection() is None or vector_store_document_count() == 0:
        msg = (
            "Vector database not ready. Add PDFs to `data/pdfs` and run: `python ingest.py`"
        )
        return "", [], msg

    language = "hi" if hindi else "en"

    try:
        contexts = retrieve_context(question.strip(), TOP_K)
    except Exception as exc:
        return "", [], f"Unable to retrieve information: {exc}"

    if not contexts:
        reply = (
            "I could not find this in the official documents I have.\n\n"
            "Add relevant PDFs, run `python ingest.py`, and verify on the official Passport Seva portal."
        )
        return reply, [], None

    prompt = build_prompt(question, contexts, language)

    try:
        client = genai.Client(api_key=api_key)
        # Allow fuller answers when excerpts are rich; low temperature for grounding.
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=2048,
                temperature=0.25,
            ),
        )

        raw_text = (response.text or "").strip()

        if not raw_text:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                parts = getattr(candidates[0].content, "parts", []) or []
                raw_text = " ".join(
                    getattr(p, "text", "") or ""
                    for p in parts
                ).strip()

        if not raw_text:
            reply = (
                "I could not produce a grounded answer right now.\n\n"
                "Retry shortly or verify on the official Passport Seva portal."
            )
            return reply, dedupe_citations(contexts), None

        return raw_text, dedupe_citations(contexts), None

    except Exception as exc:
        return "", [], f"The answer could not be generated ({exc}). Check API key, quota, and network."

