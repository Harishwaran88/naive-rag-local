Naive/Traditional RAG Pipeline (Local, Framework-Free)

This repository contains a naive (traditional) Retrieval-Augmented Generation (RAG) implementation built from scratch, without using any RAG frameworks such as LangChain or LlamaIndex.


Purpose of This Project
 - This project is intended to:
	 > Build a strong conceptual understanding of RAG
	 > Avoid heavy abstractions from frameworks
	 > Serve as a learning and experimentation baseline
	 > Provide a clean starting point for advanced RAG pipelines


The pipeline uses:
 - Local LLMs via Ollama
 - Sentence-Transformer models for embeddings
 - Plain text files as input documents
 - Vector similarity search for context retrieval

The primary goal of this project is to understand and demonstrate the core building blocks of RAG systems in a transparent and modular way


Architecture Overview

    Documents → Chunks → Embeddings → Vector Store
    
	↓
    
	User Question → Embedding → Similarity Search → Context → LLM Prompt → Answer


Features
 - Framework-free RAG implementation
 - Fully local execution (no cloud APIs)
 - SentenceTransformer-based semantic embeddings
 - Vector similarity search using ChromaDB
 - Plain text (.txt) document input
 - Modular and readable Python code
 - Easy foundation for advanced RAG extensions


Tech Stack
| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Embeddings           | SentenceTransformers      |
| Vector Store         | ChromaDB                  |
| LLM Runtime          | Ollama                    |
| LLM Model            | `gemma:2b` (configurable) |
| Input Format         | Text files (`.txt`)       |


Project Workflow
1. Document Loading
    Text data is loaded from a local .txt file.

2. Text Chunking
    The document is split into overlapping chunks to:
    Improve retrieval relevance
    Prevent token overflow

3. Embedding Generation
    Each chunk is converted into a dense vector using a SentenceTransformer model.

4. Vector Storage
    Chunk embeddings and metadata are stored in a vector database for similarity search.

5. Query Processing
    The user’s question is embedded using the same embedding model.

6. Similarity Search
    The most relevant document chunks are retrieved from the vector store.

7. Prompt Construction & Answer Generation
    Retrieved chunks are injected into a prompt and sent to a local LLM running via Ollama to generate the final answer.


Prerequisites
 - Python 3.9+
 - Ollama installed and running locally


Install Dependencies
    pip install -r requirements.txt


Add Your Document
    Place a .txt file in the project directory and update the file path in the code accordingly.


Run the Application
    python main.py


Example Query
What is Sub Procedures?

Pipeline Behavior
 - Relevant chunks are retrieved from the document
 - Context is passed to the LLM
 - The answer is generated based only on retrieved content


Current Limitations
This is a naive RAG implementation, intentionally kept simple:
 - Dense retrieval only (no BM25 / hybrid search)
 - No reranking
 - No token-budget enforcement
 - Single-document input
 - No evaluation metrics

License
 - This project is provided for educational and experimental purposes.
 - You are free to use, modify, and extend it.