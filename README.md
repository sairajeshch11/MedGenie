!cd /content/llm_project && cat > README.md <<'EOF'
# MedGenie 🧬

## Overview

**MedGenie** is an intelligent medical information assistant designed to answer
health-related questions using **trusted medical knowledge sources** and
advanced **Retrieval-Augmented Generation (RAG)** techniques.

The application retrieves relevant medical context from curated datasets and
generates **grounded, explainable answers** instead of speculative responses.
Each answer is supported by sources, excerpts, and explanations indicating
*why* a particular source was used.

MedGenie is built as an end-to-end application and delivered through a clean,
interactive web interface powered by Streamlit.

---

## What MedGenie Does

- Answers medical and health-related questions in natural language
- Retrieves relevant information from authoritative medical sources
- Generates grounded answers using retrieved evidence only
- Displays supporting sources with explanations and excerpts
- Flags medical urgency when applicable
- Helps users locate nearby hospitals based on location input
- Provides medication-specific Q&A using a dedicated medication pipeline (when the question is medication-related)

---

## Trusted Medical Data Sources

MedGenie retrieves information from **authoritative, public medical datasets**,
including:

- **World Health Organization (WHO)** publications
- **MedlinePlus** (U.S. National Library of Medicine)
- **Public health and clinical guideline sources** commonly referenced by CDC-style datasets
- Curated medical text corpora indexed for semantic and keyword search

No external or unverified web content is used during answer generation.

---

## Key Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Hybrid retrieval using **FAISS (dense embeddings)** and **BM25 (keyword search)**
- Prevents hallucinations by grounding answers strictly in retrieved content

### 📚 Source Transparency
- Each answer includes:
  - Source title
  - Supporting excerpt
  - Explanation of *why the source is shown*
- Clearly indicates when information is indirect or limited

### ⚠️ Medical Urgency Detection
- Automatically classifies urgency as:
  - **LOW**
  - **MEDIUM**
  - **HIGH**
- Provides reasoning for the urgency level

### 🏥 Nearby Hospital Finder
- Allows users to search for nearby hospitals
- Accepts city, ZIP code, or address input
- Integrated directly into the app interface

### 🧠 Explainable AI Design
- No black-box answers
- Every response is traceable to retrieved medical context

---

## Technology Stack

- **Python**
- **OpenAI API** (LLM reasoning)
- **FAISS** (dense vector similarity search)
- **BM25** (keyword-based retrieval)
- **Sentence Transformers**
- **Streamlit** (web interface)

---

## Project Structure

llm_project/
│
├── app.py # Main application (UI + orchestration)
├── rag.py # RAG pipeline logic
├── retrieval.py # FAISS + BM25 retrieval
├── medications.py # Medication-specific logic
├── urgency.py # Urgency detection
├── nearby_hospitals.py # Hospital search tool
├── prompts.py # Prompt templates
├── helpers/ # LLM helpers
├── indexes/ # Prebuilt FAISS & BM25 indexes (required)
├── requirements.txt
└── README.md

---

## Deployment

MedGenie is deployed on **Streamlit Cloud**.

### Required Secrets
- `OPENAI_API_KEY`

Indexes are committed to the repository to ensure reliable runtime behavior
in cloud deployment environments.

---

## Academic Context

This project demonstrates:
- Practical application of **Information Retrieval (IR)**
- Real-world use of **RAG architectures**
- Responsible AI principles (grounding, transparency, explainability)
- End-to-end system design and deployment

---

## Disclaimer

MedGenie is an **educational and informational system**.
It does **not** replace professional medical advice, diagnosis, or treatment.

Always consult qualified healthcare professionals for medical concerns.

---

## Author

**Sai Rajesh Chittavarjula**  
Master’s Student — Computer Science  
Information Retrieval & AI Systems
EOF


