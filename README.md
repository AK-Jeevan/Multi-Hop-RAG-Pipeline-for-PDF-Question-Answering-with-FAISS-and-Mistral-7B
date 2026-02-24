# 📚 Multi-Hop RAG Pipeline for PDF Question Answering

A complete **multi-hop Retrieval-Augmented Generation (RAG)** system that answers complex questions from PDF documents.  
It performs iterative retrieval, reranking, query refinement, and grounded answer generation using transformer models.

---

## 🚀 Overview

This project demonstrates how to:

1. Extract text from a PDF
2. Chunk the text into overlapping passages
3. Encode passages using DPR (Dense Passage Retrieval)
4. Store embeddings in a FAISS index
5. Retrieve and rerank relevant passages
6. Refine the query for multi-hop retrieval
7. Generate a grounded answer using Mistral-7B

The system is especially suited for:
- 📜 Legal document QA
- 📖 Research papers
- 🏛️ Policy documents
- 📄 Long-form reports

---

## 🧠 Architecture

```
User Query
    ↓
Dense Retrieval (DPR Question Encoder)
    ↓
FAISS Vector Search
    ↓
Cross-Encoder Reranking
    ↓
Multi-Hop Query Reformulation
    ↓
Context Aggregation
    ↓
Mistral-7B Answer Generation
```

---

## 🛠️ Tech Stack

- **PDF Parsing:** `pypdf`
- **Embeddings:** `sentence-transformers`
  - facebook-dpr-question_encoder-single-nq-base
  - facebook-dpr-ctx_encoder-single-nq-base
- **Vector Search:** `FAISS`
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generation Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Frameworks:** `transformers`, `torch`, `numpy`

---

## 📂 Workflow Explained

### 1️⃣ Load PDF
Extracts raw text from a PDF file.

### 2️⃣ Chunk Text
Splits document into overlapping chunks for better semantic retrieval.

### 3️⃣ Embed & Index
Encodes chunks into dense vectors and stores them in FAISS for fast similarity search.

### 4️⃣ Retrieve
Finds top-k relevant chunks for a query.

### 5️⃣ Rerank
Uses a cross-encoder to score and reorder retrieved chunks.

### 6️⃣ Multi-Hop Retrieval
Refines the query using accumulated context and performs another retrieval pass.

### 7️⃣ Final Answer Generation
Builds a grounded prompt and generates a final answer using Mistral-7B.

---

## 📦 Installation

```bash
pip install pypdf sentence-transformers faiss-cpu transformers torch numpy
```

> ⚠️ For GPU acceleration, install `faiss-gpu` and ensure CUDA is configured.

---

## ▶️ Usage

1. Place your PDF in the project directory (e.g., `constitution.pdf`).
2. Run the script:

```bash
python RAG_Multi_Hop.py
```

3. Modify the query at the bottom of the file:

```python
response = multi_hop_rag(
    "Which case expanded Article 21 and what rights were added?"
)
```

---

## 🔄 Multi-Hop Retrieval Explained

Instead of retrieving once, the system:

- Retrieves initial relevant passages
- Uses them to refine the search query
- Retrieves again to fill missing information
- Combines all evidence before generating the final answer

This improves performance on:
- Cross-reference questions
- Case-law relationships
- Questions requiring implicit reasoning

---

## 🎯 Example Use Case

**Query:**
```
Which case expanded Article 21 and what rights were added?
```

**System Behavior:**
- Retrieves passages mentioning Article 21
- Identifies key legal cases
- Refines query to target specific judgments
- Produces a grounded answer using cited content

---

## ⚙️ Configuration Options

You can modify:

- `chunk_size` and `overlap`
- `top_k` retrieval count
- `top_n` reranked passages
- `max_hops`
- `temperature`
- `max_tokens`

---

## 📈 Why Multi-Hop RAG?

Single-pass retrieval may miss:
- Distributed facts
- Implicit references
- Context split across sections

Multi-hop retrieval enables deeper reasoning and better grounding.

---

## 🧩 Future Improvements

- Add citation extraction
- Add streaming responses
- Support multiple PDFs
- Add hybrid (BM25 + dense) retrieval
- Implement evaluation metrics (EM/F1)

---

## 📜 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome!  
If you find a bug or want to improve retrieval quality, feel free to open an issue.

---

## ⭐ If You Found This Useful

Give it a star and build something awesome with it!
