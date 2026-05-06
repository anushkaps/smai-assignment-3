"""Central configuration for the Passport Seva RAG Assistant."""

import os
from pathlib import Path

# Paths anchored to project root (directory containing config.py), not process CWD.
_BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = str(_BASE_DIR / "data" / "pdfs")
CHROMA_DIR = str(_BASE_DIR / "chroma_db")
COLLECTION_NAME = "passport_docs"

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Google periodically retires model IDs; see https://ai.google.dev/gemini-api/docs/models
# Override if needed: export GEMINI_MODEL_NAME=gemini-2.5-flash (or check ListModels in AI Studio).
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Retrieval / chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
