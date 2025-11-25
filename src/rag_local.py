import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from constants import CHROMA_PATH, DATA_PATH


def load_documents(filename):
    """Loads documents from the specified data path."""
    
    pdf_path = os.path.join(DATA_PATH, filename)
    loader = PyPDFLoader(pdf_path)
    
    # loader = UnstructuredPDFLoader(pdf_path) # Alternative
    documents = loader.load()
    
    print(f"Loaded {len(documents)} page(s) from {pdf_path}")
    
    return documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    
    return all_splits

def get_embedding_function(model_name="nomic-embed-text"):
    """Initializes the Ollama embedding function."""
    
    # Ensure Ollama server is running (ollama serve)
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    
    return embeddings

def get_embedding_function_hf(model_name="all-MiniLM-L6-v2"):
    """Initializes HuggingFace embeddings (runs locally)."""
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Initialized HuggingFace embeddings with model: {model_name}")
    
    return embeddings

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    
    return vectorstore

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    """Indexes document chunks into the Chroma vector store."""
    
    print(f"Indexing {len(chunks)} chunks...")
    # Use from_documents for initial creation.
    # This will overwrite existing data if the directory exists but isn't a valid Chroma DB.
    # For incremental updates, initialize Chroma first and use vectorstore.add_documents().
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist() # Ensure data is saved
    print(f"Indexing complete. Data saved to: {persist_directory}")
    
    return vectorstore

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """Creates the RAG chain."""
    # Initialize the LLM
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0, # Lower temperature for more factual RAG answers
        num_ctx=context_window # IMPORTANT: Set context window size
    )
    
    print(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or "mmr"
        search_kwargs={'k': 3} # Retrieve top 3 relevant chunks
    )
    print("Retriever initialized.")

    # Define the prompt template
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    print("RAG chain created.")
    
    return rag_chain

def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    
    print("\nQuerying RAG chain...")
    print(f"Question: {question}")
    
    response = chain.invoke(question)
    
    print("\nResponse:")
    print(response)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Documents
    docs = load_documents('nke-10k-2023.pdf')

    # 2. Split Documents
    chunks = split_documents(docs)

    # 3. Get Embedding Function
    embedding_function = get_embedding_function_hf() # Using Ollama nomic-embed-text

    # 4. Index Documents (Only needs to be done once per document set)
    # Check if DB exists, if not, index. For simplicity, we might re-index here.
    # A more robust approach would check if indexing is needed.
    print("Attempting to index documents...")
    vector_store = index_documents(chunks, embedding_function)
    # To load existing DB instead:
    # vector_store = get_vector_store(embedding_function)

    # 5. Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="gemma3:1b") # Use the chosen model

    # 6. Query
    query_question = "When was Nike incorporated?" # Replace with a specific question
    query_rag(rag_chain, query_question)