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
 - Plain text/pdf files as input documents
 - Vector similarity search for context retrieval

The primary goal of this project is to understand and demonstrate the core building blocks of RAG systems in a transparent and modular way


Architecture Overview

    Documents → Chunks → Embeddings → Vector Store
    
	↓
    
	User Question → Embedding → Similarity Search → Context → LLM Prompt → Answer

License
 - This project is provided for educational and experimental purposes.
 - You are free to use, modify, and extend it.