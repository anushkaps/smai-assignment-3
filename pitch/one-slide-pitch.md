# One-slide pitch - Passport Seva RAG Assistant

**Use this content on a single LinkedIn-ready slide (16:9).** Export as PNG or PDF from PowerPoint / Google Slides / Canva.

---

## Title

**Passport Seva RAG Assistant**  
*Government-Services RAG Chatbot - Variant T10.1 (Tier 1)*

---

## Problem

Finding **reliable, step-by-step** Passport Seva information is hard: details are scattered across **official PDFs** and the portal, and generic chatbots may **guess** fees, documents, or rules.

---

## Solution

A **Streamlit RAG app** that answers from **your downloaded official PDFs** only: **retrieve** relevant pages → **generate** a grounded answer → show **citations** (PDF name + page). **No model training** - pre-trained embeddings + Gemini for text generation.

---

## How it works

1. **Ingest** PDFs → clean text → **chunk** → **embed** (`all-MiniLM-L6-v2`) → store in **ChromaDB**  
2. **Ask** a question → **embed query** → **retrieve top-k** chunks  
3. **Prompt Gemini** with strict “**context only**” rules → **answer + sources** in the UI  

---

## Tech stack

**Python · Streamlit · pypdf · sentence-transformers · ChromaDB · Gemini API · python-dotenv**

---

## Demo value

Shows **traceable** answers (which **file**, which **page**) for a **college RAG assignment** and encourages users to **verify** on the **official Passport Seva** portal.

---

## Impact

**Transparency + grounded responses** over blind generative answers - good baseline for **public-sector information** chatbots, with clear **limitations** (PDF freshness, scanned docs, not legal advice).

---

## Footer (optional one line)

*SMAI Assignment 3 · Educational project · Not an official government service*
