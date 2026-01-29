from components import load_text, chunk_text_approach1, Embedding_Manager, VectorStore_Manager, RAG_Retriever_Manager, RAG_Pipeline_Manager, do_llm_call, get_currentWD
from pathlib import Path


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

currentWD = get_currentWD()
folder_path = fr"{currentWD}\Sample_Data"

CHROMA_PERSIST_DIR = fr"{currentWD}\vector"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

TOP_K = 3
FINAL_CONTEXT_K = 3
SCORE_THRESHOLD = 0.2


# EMBEDDING MODEL SETUP - Initialize the embedding manager with the configured embedding model
EmbeddingManager = Embedding_Manager(model_name=EMBEDDING_MODEL_NAME)
print("Loaded embedding model")


# VECTOR DATABASE SETUP - Initialize Chroma vector store manager
vectorStore = VectorStore_Manager(collection_name=COLLECTION_NAME, persist_directory=CHROMA_PERSIST_DIR)
print("Initialized vector storage")


# Clear existing collection data
vectorStore.reset_collection()


# RAG RETRIEVER SETUP - Initialize the RAG retriever manager
rag_retreiver_manager = RAG_Retriever_Manager(vectorestore_manager=vectorStore, embedding_manager=EmbeddingManager)
print("Initialized the RAG retriever")


# DATA LOADING - Load raw text data from the given file path
doc_contents = []
for txtFile in Path(folder_path).iterdir():
    if txtFile.suffix == ".txt":
        text_data = load_text(txtFile)

        # TEXT CHUNKING - fixed-size overlapping word chunks
        list_of_chunks = chunk_text_approach1(text=text_data, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


        # TEXT → EMBEDDING CONVERSION - Generate vector embeddings for each text chunk
        convert_to_embedding = EmbeddingManager.generate_embedding(texts=list_of_chunks)
        print("Generated vector embeddings")

        doc_contents.append({
            "File_Name": txtFile.name,
            "Chunk": list_of_chunks,
            "Chunk-Embedding": convert_to_embedding
        })


# Store text chunks along with their embeddings into the vector database
for doc_content in doc_contents:
    vectorStore.add_documents_vector_db(texts=doc_content["Chunk"], embeddings=doc_content["Chunk-Embedding"])
    print("Adding data to vectore database")



# RAG PIPELINE SETUP - Initialize the full RAG pipeline
RAG_Pipeline = RAG_Pipeline_Manager(reg_retriever=rag_retreiver_manager, llm=do_llm_call, top_k=TOP_K, score_threshold=SCORE_THRESHOLD)
print("Initialized the RAG pipeline")


user_query = ""
while True:
    user_query = input("Enter your question? To end chat type 'exit': ").strip()
    
    if not user_query:
        print("Please enter a valid question.")
        continue
        
    if user_query.lower() == "exit":
        print("Ending chat...")
        break
    
    Response = RAG_Pipeline.query_llm(user_query)

    print("\nAssistant:\n", Response)