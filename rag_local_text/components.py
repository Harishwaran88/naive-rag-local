from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os, requests
from pathlib import Path


# ----------------------------------------------------------------
    # Document Loader: Reading text files
# ----------------------------------------------------------------

# Load text from a file path - local file system
def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------------------------------------------
    # Splitting text into smaller blocks
# ----------------------------------------------------------------

# This module provides utility functions for text processing.
def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


# ----------------------------------------------------------------
    # Embedding Model: Converting text to numbers using libraries like sentence-transformers or transformers
# ----------------------------------------------------------------
class Embedding_Manager():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.load_embedding_model()

    def load_embedding_model(self):
        try:
            print(f"Loading model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model '{self.model_name}' loaded successfully.")
            print("Embedding dimension:", self.model.get_sentence_embedding_dimension())
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            raise
    
    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded.")
        
        # print(f"Generating embeddings for {len(texts)} texts.")
        embedding = self.model.encode(texts)
        # print(f"Generated embeddings with shape: {embedding.shape}")
        return embedding


# ----------------------------------------------------------------
    # Vector Store: Managing the storage and retrieval using ChromaDB.
# ----------------------------------------------------------------
class VectorStore_Manager:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "/vector"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_chroma_db()
    
    def initialize_chroma_db(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings collection"}
                )
            
            print(f"VectorStore_DB initialized at '{self.persist_directory}' with collection '{self.collection_name}'.")
            print(f"existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents_vector_db(self, texts: List[str], embeddings: np.ndarray):
        if not self.collection:
            raise ValueError("VectorStore_DB collection not initialized.")

        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings.")

        print(f"Adding {len(texts)} documents to the vector store.")

        ids_list = []
        metadatas_list = []
        documents_text_list = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(texts, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids_list.append(doc_id)

            metadatas_list.append({
                "doc_index": i,
                "content_length": len(doc)
            })

            documents_text_list.append(doc)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids_list,
                embeddings=embeddings_list,
                metadatas=metadatas_list,
                documents=documents_text_list
            )
            print(f"Successfully added {len(documents_text_list)} documents.")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to VectorStore_DB: {e}")
            raise
    
    def reset_collection(self):
        print(f"Deleted the created collection {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
# ----------------------------------------------------------------
    # Retriever: Performing similarity search (often using Cosine Similarity via NumPy).
# ----------------------------------------------------------------
class RAG_Retriever_Manager:
    def __init__(self, vectorestore_manager: any, embedding_manager: any):
        self.vectore_store = vectorestore_manager
        self.embedding_manager = embedding_manager
    
    def retrieve_context(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[Dict[str, Any]]:
        # print(f"Retrieving context from documents for query: '{query}")
        # print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        try:
            results = self.vectore_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                # print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs
        except Exception as e:
            print(f"Error during retrievel: {e}")
            return []

# ----------------------------------------------------------------
    # RAG Pipelines
# ----------------------------------------------------------------
class RAG_Pipeline_Manager:
    def __init__(self, reg_retriever, llm, top_k=3, score_threshold=0.0):
        self.retriever = reg_retriever
        self.llm_func = llm
        self.top_k = top_k
        self.score_threshold = score_threshold

    def query_llm(self, user_question):
        retrieved_docs = self.retriever.retrieve_context(user_question, top_k=self.top_k, score_threshold=self.score_threshold)
        # print(f"retrieved_docs: {retrieved_docs}")
        context_docs = [doc['content'] for doc in retrieved_docs if 'content' in doc]
        if not context_docs:
            return "No relevant context found to answer the question."
        
        return self.llm_func(user_question, context_docs)
    

# ----------------------------------------------------------------
    # Making a call to local LLM.
# ----------------------------------------------------------------
def do_llm_call(user_question, context_docs):
    context = "\n\n".join(context_docs)

    prompt = f"""
            <SYSTEM_INSTRUCTIONS>
            - You are a professional and helpful Assistant.
            - GREETING RULE: If the user greets you (e.g., "Hi", "Hello"), respond with a warm welcome and ask how you can assist.
            - ANCHORING RULE: Answer the question using ONLY the provided Context below. 
            - CONSTRAINTS: If the answer is not contained within the Context, strictly state: "I am sorry, but the provided documents do not contain information to answer that question."
            - BREVITY: Keep your answers concise and factual.
            </SYSTEM_INSTRUCTIONS>

            <CONTEXT>
            {context}
            </CONTEXT>

            <USER_QUESTION>
            {user_question}
            </USER_QUESTION>

            ANSWER:
            """.strip()

    # gemma3:1b , "gemma:2b"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b", 
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 1024
            }
        },
        timeout=60
    )

    return response.json()["response"]