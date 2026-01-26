from components import *


def get_currentWD():
    current_path = Path.cwd()
    return current_path

currentWD = get_currentWD()
f_path = fr"{currentWD}\Sample_Data\VBA Tutorial.txt"
# vector_path = fr"{currentWD}\vector"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = fr"{currentWD}\vector"
COLLECTION_NAME = "documents"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

TOP_K = 3
FINAL_CONTEXT_K = 3
SCORE_THRESHOLD = 0.2

# -------------------------
# DATA LOADING - Load raw text data from the given file path
# -------------------------
text_data = load_text(f_path)
print("Loaded text data")

# -------------------------
# TEXT CHUNKING
    # Split the loaded text into smaller overlapping chunks
    # chunk_size=400 characters, overlap=80 characters between chunks
# -------------------------
list_of_chunks = chunk_text(text=text_data, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
print(f"Total Chunks Created: {len(list_of_chunks)}")


# -------------------------
# EMBEDDING MODEL SETUP
# Initialize the embedding manager with the configured embedding model
# -------------------------
EmbeddingManager = Embedding_Manager(model_name=EMBEDDING_MODEL_NAME)
print("Loaded embedding model")

# -------------------------
# VECTOR DATABASE SETUP
# Initialize Chroma vector store manager
# collection_name: logical name of the vector collection
# persist_directory: directory where vectors will be saved
# -------------------------
vectorStore = VectorStore_Manager(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIR
)
print("Initialized vector storage")

# -------------------------
# TEXT → EMBEDDING CONVERSION
# Generate vector embeddings for each text chunk
# -------------------------
convert_to_embedding = EmbeddingManager.generate_embedding(
    texts=list_of_chunks
)
print("Generated vector embeddings")

# -------------------------
# STORE EMBEDDINGS IN VECTOR DB
# Store text chunks along with their embeddings into the vector database
# -------------------------
vectorStore.add_documents_vector_db(
    texts=list_of_chunks,
    embeddings=convert_to_embedding
)
print("Adding data to vectore database")

# -------------------------
# RAG RETRIEVER SETUP
# Initialize the RAG retriever manager
# This connects the vector store with the embedding manager
# -------------------------
rag_retreiver_manager = RAG_Retriever_Manager(
    vectorestore_manager=vectorStore,
    embedding_manager=EmbeddingManager
)
print("Initialized the RAG retriever")


# -------------------------
# RAG PIPELINE SETUP
# Initialize the full RAG pipeline
# reg_retriever: retriever manager
# llm: function to call the LLM
# top_k: number of retrieved chunks passed to LLM
# -------------------------
RAG_Pipeline = RAG_Pipeline_Manager(
    reg_retriever=rag_retreiver_manager,
    llm=do_llm_call,
    top_k=TOP_K,
    score_threshold=SCORE_THRESHOLD
)
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