# Passport Seva RAG Assistant

**SMAI Assignment 3 — Government-Services RAG Chatbot**

| Field | Value |
|--------|--------|
| **Variant** | **T10.1 Passport Assistant** |
| **Tier** | **Tier 1** |
| **Course context** | Statistical Methods / AI — retrieval-augmented demo (no model training) |

---

## Problem statement

Indian **Passport Seva** procedures (application, documents, appointments, Tatkaal, status, police verification) are spread across **official PDFs** and the portal. Users need **accurate, citeable** guidance. This project builds a **Streamlit RAG chatbot** that answers questions using **only** text retrieved from **locally stored official PDFs**, with **filename + page** citations—reducing unsupported “general knowledge” answers in a college demo setting.

---

## Solution overview

- **No training:** Uses **pre-trained** embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and a **free-tier-friendly LLM API** (Gemini).
- **RAG:** PDFs → **pypdf** text → **clean + chunk** → **embed** → **ChromaDB** → **retrieve top-k** → **strict Gemini prompt** → **answer + sources**.
- **UI:** Streamlit chat (`st.chat_input` / `st.chat_message`), **Hindi toggle**, example questions, sidebar instructions.

---

## Architecture

```
User Question → Query Embedding → ChromaDB Retrieval → Retrieved Official PDF Chunks → Gemini Prompt → Answer with Citations → Streamlit Chat UI
```

**Important:** This is **not** fine-tuning or training a model from scratch. **Embeddings are pre-trained**; **Gemini** is used only for **generation conditioned on retrieved context**.

---

## Tech stack

| Component | Choice |
|-----------|--------|
| Language | Python 3 |
| UI | [Streamlit](https://streamlit.io/) |
| PDF extraction | [pypdf](https://pypdf.readthedocs.io/) |
| Embeddings (CPU, pre-trained) | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | [ChromaDB](https://www.trychroma.com/) (`PersistentClient`) |
| LLM API | [Google Gemini API](https://ai.google.dev/) (`google-genai`, default `gemini-2.5-flash`; override with `GEMINI_MODEL_NAME`) |
| Config | `python-dotenv` |

---

## Folder structure

```
passport-rag-chatbot/
├── app.py
├── ingest.py
├── rag.py
├── config.py
├── requirements.txt         # App, ingestion, RAG + notebook deps (used by Streamlit Cloud)
├── requirements-dev.txt    # Alias: same as requirements.txt (optional installs)
├── .env
├── .gitignore               # Excludes .env, venv, chroma_db data, local secrets
├── .streamlit/
│   └── config.toml          # Streamlit server defaults (safe to commit)
├── README.md
├── notebooks/
│   └── evaluation.ipynb      # RAG evaluation (not training)
├── report/
│   └── technical-report.md  # Convert to PDF for submission
├── pitch/
│   └── one-slide-pitch.md   # Copy to slide / export PNG/PDF
├── data/
│   └── pdfs/
│       └── .gitkeep
└── chroma_db/               # Created by ingest (local persistence)
```

## Deliverables checklist (SMAI A3 Tier 1)

| Deliverable | Status |
|-------------|--------|
| Public GitHub repo structure | **`app.py`**, **`ingest.py`**, **`rag.py`**, **`config.py`**, **`requirements.txt`**, **`.env`**, **`README.md`**, **`notebooks/evaluation.ipynb`**, **`report/technical-report.md`**, **`pitch/one-slide-pitch.md`**, **`data/pdfs/`**, **`.gitignore`** |
| No training notebook | Use **`notebooks/evaluation.ipynb`** (retrieval + qualitative evaluation only) |
| Citations in answers | **PDF filename + page** from retrieval; listed under **Sources** in the UI |
| Anti-hallucination prompt | **`rag.py`** `build_prompt` — context-only, exact fallback sentence, portal verification |
| Deployment readiness | **`requirements.txt`**; Streamlit **Secrets** wiring in **`app.py`**; paths in **`config.py`** anchored to project root; see below |

---

## Official PDF dataset (manual download)

Download **official** Passport Seva / Ministry PDFs from the **Passport Seva** website and place them in `data/pdfs/`. **This repository does not redistribute government PDFs.**

Use **clear filenames** (rename after download if needed), for example:

- `passport-application-instruction-booklet.pdf`
- `passport-registration-appointment-advisory.pdf`
- `passport-online-appointment-booking-process.pdf`
- `passport-steps-to-apply.pdf`
- `passport-online-application-form-guide.pdf`
- `passport-application-at-psk-steps.pdf`
- `passport-application-status-check.pdf`
- `passport-tatkaal-undertaking.pdf`
- `passport-police-verification-reinitiate-request.pdf`
- `passport-manual-rti-disclosure.pdf`

---

## Streamlit Community Cloud deployment

### 1. Entry point

- **Main file:** `app.py` (this is the only Streamlit entry point).
- In [share.streamlit.io](https://share.streamlit.io), connect your GitHub repo and set **App path** / main file to **`app.py`**.

### 2. Dependencies

- Cloud installs from **`requirements.txt`** in the repo root (Streamlit, **PyTorch + torchvision**, sentence-transformers, Chroma, PDF parsing, Gemini SDK, plus **pandas**, **matplotlib**, **notebook** for evaluation).
- Local setup: **`pip install -r requirements.txt`**. **`requirements-dev.txt`** is the same bundle via `-r requirements.txt` if you prefer that habit.

### 3. Add `GEMINI_API_KEY` as a Streamlit secret (required on Cloud)

Local development can use a `.env` file. **Streamlit Community Cloud does not use `.env` from the repo** (and you must not commit secrets).

1. Open your app on **Streamlit Community Cloud** → **Manage app** (or app settings) → **Secrets**.
2. Paste a TOML block like this (replace with your real key from [Google AI Studio](https://aistudio.google.com/)):

   ```toml
   GEMINI_API_KEY = "paste_your_key_here"
   ```

3. Save. On each run, `app.py` copies this value into the process environment so **`rag.generate_answer`** can call Gemini (`_inject_streamlit_cloud_secrets`).

**Do not** commit API keys, `.env`, or `.streamlit/secrets.toml` (see `.gitignore`).

### 4. PDFs **must** be in `data/pdfs/` before indexing

Ingestion **only** reads PDFs from **`data/pdfs/`** (path is fixed in `config.py` relative to the project root).

1. Download **official** Passport Seva PDFs from the [Passport Seva portal](https://www.passportindia.gov.in/).
2. Place every file you want indexed under **`data/pdfs/`** (e.g. `data/pdfs/passport-steps-to-apply.pdf`).
3. **Then** build the vector index by **either**:
   - **Local:** `python ingest.py`, **or**
   - **In the deployed app:** open the sidebar → **Build index (demo / Streamlit Cloud)** → **Build index from PDFs** (runs the same pipeline in-process; slow on first run due to model download).

Without PDFs in that folder, ingestion prints an error and **no chunks** are stored, so the chatbot will not retrieve answers.

### 5. Vector index on Cloud (no automatic `ingest.py` on deploy)

Streamlit Community Cloud **does not** run `python ingest.py` automatically when your app boots. Practical options:

| Approach | When to use |
|----------|-------------|
| **A. Prebuilt `chroma_db/` in git** | Small index; you are allowed to commit generated files. Remove `chroma_db/*` from `.gitignore` for those files only, or force-add the folder. Fastest cold start; no waiting for ingest on Cloud. |
| **B. In-app “Build index”** | PDFs are in `data/pdfs/` in the repo (or you add them later and redeploy). Use the sidebar **Build index from PDFs** once after deploy; wait for CPU indexing to finish. Good for demos; first run is slow. |
| **C. Local-only index** | Run `ingest.py` on your machine, demo with `streamlit run app.py`, or record a **video** for submission if hosting without a stable index is awkward. |

The sidebar **Build index** button is intentionally simple: it calls `ingest.main()` and clears RAG caches—**no extra services**.

---

## Setup

### 1. Clone and enter the project

```bash
cd passport-rag-chatbot
```

### 2. Virtual environment (recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Use a `.env` file in the project root (do not commit secrets).

Edit `.env` and set:

```env
GEMINI_API_KEY=your_key_from_google_ai_studio
```

---

## Run ingestion

**Prerequisite:** Official PDF files must already be copied into **`data/pdfs/`** (see [official PDF dataset](#official-pdf-dataset-manual-download)).

Then:

```bash
python ingest.py
```

Expected logs: PDF count, file-by-file progress, **chunks per PDF**, **total chunks stored** under **`chroma_db/`**.

---

## Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal. Use the sidebar (**Hindi**, **Clear chat**) and **example questions**.

---

## Example questions

- How do I apply for a new passport?
- What documents are required?
- How do I book an appointment?
- How do I renew or reissue my passport?
- What is Tatkaal passport?
- How do I check passport application status?

---

## Evaluation approach

1. **Manual QA** after ingestion: ask each sample question in the Streamlit app (and in `notebooks/evaluation.ipynb`).
2. **Retrieval check:** For each question, inspect **top retrieved chunks** (source file + page). Relevance = excerpts should match the question’s topic.
3. **Answer checklist:** Answer must **stick to excerpts**; if missing, the model should output **“I could not find this in the official documents I have.”**
4. **Citation checklist:** UI shows **PDF filename** + **page** for deduplicated hits; align with where the information appears in the PDF when grading.

See **`notebooks/evaluation.ipynb`** for a **result table** template (question, expected topic, correctness, citation, notes).

---

## Ablation / retrieval analysis plan

| Experiment | What to vary | What to observe |
|------------|----------------|------------------|
| **A** | `TOP_K`: 3 vs 5 (`config.py`) | Precision vs recall of snippets; answer completeness |
| **B** | `CHUNK_SIZE`: 500 vs 900 | Fragmented vs coarser chunks; retrieval quality |
| **C** | Answer **with** vs **without** citing sources in the prompt | Grounding vs hallucination risk (demo only) |
| **D** | **English** vs **Hindi** toggle | Same PDFs; language quality + faithfulness |

Document outcomes in the **Technical Report** and optionally add short notes in the notebook.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torchvision'`** when running `streamlit run app.py`: install/reinstall deps so **`torch`** and **`torchvision`** are present (`requirements.txt` includes them). Activate your venv, then run `pip install -r requirements.txt` again. On Linux or for smaller CPU-only wheels you can use [PyTorch’s CPU index](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt` minus torch lines, or after: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

---

## Limitations

- **Depends on which PDFs you download** — coverage and freshness are **your** responsibility.
- **Outdated PDFs** → answers may not match the **live** portal.
- **Scanned** or image-only PDFs → `pypdf` may extract little text (no OCR in Tier 1).
- **Not legal advice** — users must **verify** on the **official Passport Seva** portal.
- **API limits** — Gemini/network errors are possible; handled with user-visible messages.

---

## Future improvements

- OCR for scanned PDFs  
- Semantic or heading-based chunking  
- Hybrid search (sparse + dense)  
- Automated eval with a small labeled Q/A sheet over your PDF set  

---

## References

1. Passport Seva — official portal (India): [https://www.passportindia.gov.in/](https://www.passportindia.gov.in/)  
2. Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)  
3. ChromaDB: [https://docs.trychroma.com/](https://docs.trychroma.com/)  
4. Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)  
5. Google Gemini API: [https://ai.google.dev/](https://ai.google.dev/)  
6. pypdf: [https://pypdf.readthedocs.io/](https://pypdf.readthedocs.io/)  

---

## LLM usage acknowledgement

Generative assistants (**ChatGPT**, **Cursor**, **Gemini**, etc.) may have been used for **code scaffolding**, **debugging**, and **drafting** README/report text. **Evaluation**, **PDF selection**, **ingestion runs**, **manual testing**, and **final academic analysis** are the responsibility of the project authors. Course plagiarism and disclosure rules apply—cite tools per your instructor’s policy.

---

## Assignment deliverables checklist

| Deliverable | Location |
|-------------|----------|
| Public GitHub repo | Push this project |
| README + requirements + app | This repo |
| Notebook | `notebooks/evaluation.ipynb` |
| Technical report (export to PDF) | `report/technical-report.md` |
| One-slide pitch (PNG/PDF) | `pitch/one-slide-pitch.md` → slide deck |
| Live demo | Streamlit Community Cloud / Hugging Face Spaces, or **recorded demo** |
| LLM acknowledgement | This README + report |

---

## License / disclaimer

Educational demo only. **Not** affiliated with the Government of India or Passport Seva.
