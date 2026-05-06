# Technical Report: Passport Seva RAG Assistant

**Course/Assignment:** SMAI Assignment 3 - Government-Services RAG Chatbot  
**Variant/Tier:** T10.1 Passport Assistant - Tier 1  
**Authors:** *[Your Name(s)]*  
**Date:** *[Submission Date]*  

---

## 1. Introduction

Passport Seva information (new passport application, re-issue/renewal, appointments, Tatkaal, police verification, and application status tracking) is often scattered across multiple official documents and portal pages. For a citizen, it can be time-consuming to find the correct step-by-step instructions and supporting details, and generic chatbots may produce **uncited** or **hallucinated** answers.

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot called **Passport Seva RAG Assistant**. The assistant answers user questions using **only** the text retrieved from a local set of **official Passport Seva PDFs**, and shows **citations** (PDF filename + page number) for transparency.

**Key point:** this is a **no-training** system. We do **not** fine-tune any model. We only use:
- a **pre-trained embedding model** to index and retrieve relevant PDF chunks, and
- a hosted LLM (Gemini) to **summarize** the retrieved context under strict anti-hallucination instructions.

---

## 2. Data

### 2.1 Document source and collection

The dataset is a curated set of **official Passport Seva PDFs** downloaded manually from the official Passport Seva portal and related Government of India links:

- Official portal: `https://www.passportindia.gov.in/`

The PDFs are placed in the repository at:

`data/pdfs/`

This repository **does not** ship or redistribute the PDFs by default; instead, the user downloads them manually and places them in this folder.

Example PDF filenames used in documentation (rename your downloads to match for clarity):
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

### 2.2 Extraction and storage

- **Parsing:** `pypdf` extracts text **page-by-page**.
- **Cleaning:** repeated spaces and repeated blank lines are removed; text is stripped.
- **Index storage:** cleaned text chunks are embedded and stored in **ChromaDB** at `chroma_db/`.

### 2.3 Model weights (no training)

This project uses **pre-trained weights** only:

- **Embedding model weights:** `sentence-transformers/all-MiniLM-L6-v2` are downloaded automatically (on first run) from the **Hugging Face Model Hub** by the `sentence-transformers` library and cached locally. These weights are used only for **inference** to compute embeddings for PDF chunks and user queries. No gradients are computed and no parameters are updated.
- **LLM weights (Gemini):** the Gemini model (configured via `GEMINI_MODEL_NAME`, default `gemini-2.5-flash`) runs on Google’s infrastructure via the **Gemini API**. The model weights are not downloaded by this project; we only send prompts and receive generated text.

### 2.4 Data limitations

- If downloaded PDFs are **outdated**, answers may miss the latest policy updates. Users must verify on the official portal.

---

## 3. Method

### 3.1 Overall RAG architecture

**Pipeline:**  
User Question → Query Embedding → ChromaDB Retrieval → Retrieved Official PDF Chunks → Gemini Prompt → Answer with Citations → Streamlit Chat UI

### 3.2 PDF ingestion (offline indexing) - `ingest.py`

1. **Load PDFs** from `data/pdfs/`.
2. **Extract text per page** with `pypdf`.
3. **Clean text** to reduce noise (spacing/blank lines).
4. **Chunking:** each page is split into overlapping character chunks:
   - `CHUNK_SIZE = 900`
   - `CHUNK_OVERLAP = 150`
5. **Embeddings:** each chunk is embedded using SentenceTransformers:
   - `sentence-transformers/all-MiniLM-L6-v2` (CPU embeddings)
6. **Vector store:** store in ChromaDB (`PersistentClient`) with metadata:
   - `source` = PDF filename
   - `page` = 1-indexed page number
   - `chunk_index` = chunk number on that page

On repeated runs, the Chroma collection is deleted and recreated so the index exactly matches the current PDFs.

### 3.3 Retrieval (online at question time) - `rag.py`

For a user question \(q\):
1. Compute embedding \(E(q)\) using the same embedding model.
2. Query ChromaDB to fetch top \(k\) chunks by cosine similarity:

$
\text{score}(q, d_i) = \cos(E(q), E(d_i))
$

where \(d_i\) is a stored chunk.

Configuration:
- `TOP_K = 5`

### 3.4 Prompting and answer generation (Gemini)

The retrieved chunks are inserted into a strict prompt with these constraints:
- Use **only** the retrieved PDF excerpts.
- Do **not** guess fees, deadlines, eligibility, required documents, or rules.
- If the answer is not in the excerpts, respond exactly:  
  **“I could not find this in the official documents I have.”**
- Do not expose internal retrieval labels (e.g., “SOURCE 1”, “Excerpt 2”).
- Mention that the user should verify the latest information on the official Passport Seva portal.

Gemini is used only for **generation** from context (no fine-tuning). The model id is configured by `GEMINI_MODEL_NAME` (default: `gemini-2.5-flash`).

### 3.5 Citations

Citations shown to the user are derived from the retriever metadata:
- PDF filename + page number
- deduplicated and grouped (pages merged per PDF)

This makes sources **traceable** and avoids repeating the same PDF/page multiple times.

### 3.6 Streamlit UI

The Streamlit app:
- uses `st.chat_input` and `st.chat_message`
- stores conversation in `st.session_state`
- displays one canonical **Sources** block per assistant reply
- includes a **Hindi toggle** (answer language instruction changes; retrieval stays the same)

---

## 4. Results

This section should allow evaluation **from the report alone**.

### 4.1 Sample evaluation questions

Use the following questions (also present in the evaluation notebook):
- How do I apply for a new passport?
- How do I book a passport appointment?
- How do I check my application status?
- What is Tatkaal passport?
- What happens at Passport Seva Kendra?
- How do I fill the online passport application form?

### 4.2 Qualitative evaluation table (fill with your runs)

| question | expected topic | answer correctness (Yes/No) | citation shown (Yes/No) | notes |
|----------|----------------|--------------------------|----------------------|------|
| How do I apply for a new passport? | Application flow | Yes | Yes | Not detailed enough as the referece pdf only had images and didnot have enough test data. |
| How do I book a passport appointment? | Appointment booking | Yes | Yes | Gave detailed steps with details and site links. |
| How do I check my application status? | Status tracking | Yes | Yes | Gave all the possible ways to do it and explained how to do it. |
| What is Tatkaal passport? | Tatkaal | Yes | Yes | Gave a clear and informative explination of Tatkaal and how it proceeds and valid references. |
| What happens at Passport Seva Kendra? | PSK/POPSK visit steps | Yes | Yes | Detailed information of what happens, how to book an apoinment and what are important steps to do before visiting. |
| How do I fill the online passport application form? | Online form guide | Yes | Yes | This is not detailed enough for the understanding and mentioned it could not find detailed information and gave an overview. |

### 4.3 Screenshots / evidence (recommended)

Add 3–5 screenshots into the final PDF export:
- A supported question with a detailed answer
- “Sources” block showing multiple PDFs/pages
- A question outside the PDF corpus showing the refusal sentence

**Screenshot placeholders (replace with your own):**
- Figure 1: New passport application steps + citations
- Figure 2: Appointment booking answer + citations
- Figure 3: Status check answer + citations
- Figure 4: Unsupported question → refusal behavior

---

## 5. Ablation Study

This ablation is designed for Tier 1: simple, interpretable changes.

### 5.1 Experiments

1. **Top-k retrieval:** `TOP_K = 3` vs `TOP_K = 5`
   - Expectation: \(k=3\) may improve precision but reduce recall; \(k=5\) may improve completeness but introduce noise.

2. **Chunk size:** `CHUNK_SIZE = 500` vs `CHUNK_SIZE = 900`
   - Expectation: smaller chunks can improve retrieval precision but may fragment instructions; larger chunks provide more context per hit.

3. **Citations instruction:** prompt with strict citation discipline vs a relaxed prompt
   - Expectation: strict source discipline reduces hallucinations and improves groundedness.

4. **English vs Hindi mode**
   - Expectation: same retrieval, but language quality differs; ensure factual content remains grounded.

### 5.2 Suggested reporting format

| Ablation | Setting 1 | Setting 2 | Observation | Better setting |
|----------|-----------|-----------|-------------|----------------|
| Top-k retrieval | TOP_K = 3 | TOP_K = 5 | TOP_K = 3 returned fewer and more focused chunks, but some answers missed supporting details. TOP_K = 5 gave more complete answers for procedural questions such as application flow and appointment booking, but occasionally included less relevant chunks. | TOP_K = 5 |
| Chunk size | CHUNK_SIZE = 500 | CHUNK_SIZE = 900 | Smaller chunks were easier to match to narrow questions, but sometimes split multi-step instructions across different chunks. Larger chunks preserved more surrounding context and helped the model answer process-based questions more completely. | CHUNK_SIZE = 900 |
| Language mode | English | Hindi | English answers were more direct and closely matched the source PDFs. Hindi mode improved accessibility, but some phrasing was less exact and needed checking against the retrieved English sources. | English for accuracy, Hindi for accessibility |

#### 5.2.1 Top-k Retrieval: TOP_K = 3 vs TOP_K = 5

When TOP_K was set to 3, the retrieved context was usually more focused. This reduced irrelevant information, especially for narrow questions such as checking application status. However, for broader procedural questions, such as applying for a new passport or booking an appointment, three chunks were sometimes not enough to cover all steps.

When TOP_K was set to 5, the answers were generally more complete because the model received more supporting context. The trade-off was that some retrieved chunks were less directly relevant. Overall, TOP_K = 5 worked better for this project because Passport Seva queries are often procedural and require multiple steps.

#### 5.2.2 Chunk Size: 500 vs 900

With CHUNK_SIZE = 500, retrieval was more precise because each chunk contained a smaller amount of text. However, some passport procedures were split across multiple chunks, so the generated answer sometimes missed surrounding instructions.

With CHUNK_SIZE = 900, each retrieved chunk contained more context. This helped the model answer multi-step questions more completely. The drawback was that larger chunks sometimes included extra text not directly related to the question. For this project, CHUNK_SIZE = 900 was more useful because official government instructions are often written as step-by-step procedures.

#### 5.2.3 English vs Hindi Mode

In English mode, the answers were usually more faithful to the retrieved PDFs because the source documents were primarily in English. The response wording was also closer to the original official instructions.

In Hindi mode, the chatbot became more accessible for users who prefer Hindi. However, because the retrieved source text was still from English PDFs, the translated answer needed extra checking for factual consistency. Hindi mode is useful as a user-facing feature, but English mode is more reliable for evaluation.

### 5.5 Summary of Ablation Findings

The best configuration for the final demo was TOP_K = 5, CHUNK_SIZE = 900, and a strict citation-based prompt. This combination gave the most complete and grounded answers for procedural Passport Seva questions. Hindi mode was useful for accessibility, but English mode was more suitable for evaluating factual faithfulness against the official PDFs.

---

## 6. Limitations

- **PDF dependency:** the assistant can only answer what is present in the downloaded PDFs.
- **Freshness risk:** if PDFs are outdated, answers may miss recent changes; users should verify on the official portal.
- **Scanned PDFs:** without OCR, scanned documents may extract poorly, reducing retrieval quality(Out of scope here).
- **API dependency:** Gemini requires network availability and quota; failures degrade generation.
- **Not official advice:** the assistant is an educational system and does not provide legal/government advice beyond cited text.

---

## 7. References

1. Passport Seva official portal: `https://www.passportindia.gov.in/`  
2. Streamlit documentation: `https://docs.streamlit.io/`  
3. ChromaDB documentation: `https://docs.trychroma.com/`  
4. Sentence-Transformers documentation: `https://www.sbert.net/`  
5. Google Gemini API documentation: `https://ai.google.dev/`  
6. pypdf documentation: `https://pypdf.readthedocs.io/`  

