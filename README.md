# Nutrition PDF Q&A with LangGraph and FAISS

This project demonstrates a pipeline for question answering over a dataset of 7 nutrition-related PDFs. It leverages modern LLM tooling, including **LlamaParse**, **FAISS**, and **LangGraph**, to create a multi-step, self-evaluating retrieval-augmented generation (RAG) system.

---

## 🚀 Overview

### Key Components:
- **Dataset**: 7 PDF documents containing nutrition-related content.
- **Parsing**: [LlamaParse](https://llamahub.ai/tools/llama-parse) used to extract and clean text from PDFs.
- **Embeddings & Vector Store**: Embeddings created and stored using [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.
- **LangGraph Workflow**: A custom LangGraph flow carries out query handling, RAG-based answering, and self-evaluation using an LLM as a judge.

---

## 🧠 LangGraph Flow Logic

1. **Query Expansion**: The user’s initial query is semantically expanded to improve retrieval quality.
2. **RAG Response Generation**: Expanded query is used to retrieve relevant chunks from the vector store and generate a draft answer.
3. **Groundedness Check**: An LLM evaluates if the generated response is well-grounded in retrieved context (above a confidence threshold).
4. **Precision Check**: If groundedness passes, the response is evaluated for precision and clarity.
5. **Final Output**: If all evaluations pass, the final answer is presented to the user. Otherwise, the process is adjusted or halted.

Each decision point (groundedness, precision, etc.) is reviewed using an **LLM as a judge**, enabling reflective and adaptive reasoning.

---

## 🛠️ Technologies Used

- **Python**
- **LangGraph**
- **LangChain**
- **LlamaParse**
- **FAISS**
- **GroQLLM**
