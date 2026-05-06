"""
Streamlit chat UI for the Passport Seva RAG Assistant.
SMAI Assignment 3 - Government-Services RAG (T10.1 Passport Assistant, Tier 1).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# Resolve .env relative to this file (works with `streamlit run` from any CWD)
_APP_ROOT = Path(__file__).resolve().parent
load_dotenv(_APP_ROOT / ".env")

from rag import (
    format_merged_sources_markdown,
    generate_answer,
    reset_runtime_caches,
    vector_store_document_count,
)


def _inject_streamlit_cloud_secrets() -> None:
    """Map Streamlit Community Cloud / Secrets to os.environ for rag.generate_answer."""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val:
                os.environ.setdefault("GEMINI_API_KEY", val)
    except Exception:
        # No local secrets.toml, or secrets unavailable
        pass

PAGE_TITLE = "Passport Seva RAG Assistant"
PAGE_ICON = "🛂"

# Demo questions aligned with official PDF themes (place matching PDFs in data/pdfs).
EXAMPLE_QUESTIONS = [
    "How do I apply for a new passport?",
    "What documents are required for passport application?",
    "How do I book an appointment?",
    "How do I renew my passport?",
    "How do I check application status?",
]


def init_session_state() -> None:
    """Initialize persistent UI state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "hindi" not in st.session_state:
        st.session_state.hindi = False
    if "inject_question" not in st.session_state:
        st.session_state.inject_question = None


def render_sidebar() -> None:
    """Sidebar: project info, instructions, Hindi toggle, clear chat."""
    with st.sidebar:
        st.markdown("### Passport Seva RAG Assistant")
        st.caption(
            "SMAI A3 - **T10.1 Passport Assistant** (Tier 1). "
            "Grounded in **your local** official PDFs only."
        )

        st.markdown("---")
        st.markdown("#### How to use")
        st.markdown(
            """
1. Download **official** Passport Seva PDFs into `data/pdfs`
2. Run: `python ingest.py`
3. Chat below (optional **Hindi** toggle)
            """
        )

        st.session_state.hindi = st.toggle(
            "Hindi answers (हिंदी में जवाब)",
            value=st.session_state.hindi,
            help="When on, Gemini answers in Hindi using the same retrieved PDF context.",
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        with st.expander("Build index (demo / Streamlit Cloud)", expanded=False):
            st.caption(
                "Runs the same logic as `python ingest.py` in this process. "
                "**PDFs must already be in `data/pdfs/`** (for Cloud, commit small official PDFs you are allowed to ship). "
                "First run downloads the embedding model and can take several minutes on the free tier."
            )
            if st.button("Build index from PDFs", use_container_width=True, type="secondary"):
                try:
                    import ingest as ingest_module

                    with st.spinner("Indexing PDFs (CPU) - please wait…"):
                        ingest_module.main()
                    reset_runtime_caches()
                    n = vector_store_document_count()
                    if n == 0:
                        st.warning(
                            "No chunks were stored. Add non-empty `.pdf` files under `data/pdfs/` "
                            "and try again (see terminal logs if running locally)."
                        )
                    else:
                        st.success(f"Indexed **{n}** chunk(s). You can chat below.")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")

        doc_count = vector_store_document_count()
        if doc_count == 0:
            st.warning(
                "No indexed documents yet. **Place PDFs in `data/pdfs/` first**, then run `python ingest.py` "
                "locally or click **Build index from PDFs** above."
            )
        else:
            st.success(
                f"Vector store ready (stored chunks ≈ **{doc_count}**). "
                "Citations use PDF filename + page."
            )


def render_example_chips() -> None:
    """Show quick example questions as buttons."""
    st.markdown("**Example questions**")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}"):
                st.session_state.inject_question = question
                st.rerun()


def display_message_history() -> None:
    """Redraw stored turns."""
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

            if role == "assistant":
                cites = msg.get("citations") or []
                err = msg.get("error")

                if err:
                    st.error(err)

                st.markdown("**Sources**")
                if cites:
                    st.markdown(format_merged_sources_markdown(cites))
                else:
                    st.warning(
                        "No PDF/page citations for this reply. "
                        "Confirm ingestion completed and the question matches your PDF set."
                    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
    _inject_streamlit_cloud_secrets()
    init_session_state()

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption(
        "Answers use **retrieved excerpts** from PDFs in `data/pdfs` via **RAG** (no model training). "
        "Verify urgent details on the **official Passport Seva** portal."
    )

    render_sidebar()
    render_example_chips()

    display_message_history()

    prompt_from_user = st.chat_input("Ask anything about Passport Seva (India)...")
    if st.session_state.inject_question:
        prompt_from_user = st.session_state.inject_question
        st.session_state.inject_question = None

    if not prompt_from_user:
        return

    st.session_state.messages.append({"role": "user", "content": prompt_from_user})

    with st.chat_message("user"):
        st.markdown(prompt_from_user)

    with st.chat_message("assistant"):
        with st.spinner("Searching official Passport Seva documents..."):
            answer, citations, err = generate_answer(
                prompt_from_user,
                hindi=bool(st.session_state.hindi),
            )

        st.markdown(answer if answer else "_No grounded answer produced._")

        if err:
            st.error(err)

        st.markdown("**Sources**")
        if citations:
            st.markdown(format_merged_sources_markdown(citations))
        else:
            st.warning(
                "No PDF/page citations returned for this turn. "
                "Add/ingest relevant PDFs or check your API key and vector store."
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer or "_No grounded answer produced._",
            "citations": citations,
            "error": err,
        }
    )


if __name__ == "__main__":
    main()
