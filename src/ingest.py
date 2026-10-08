import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

def load_documents(target_path: str = "data"):
    """
    Load PDF documents from a single file or a directory containing PDFs.
    Normalizes metadata with clean file basenames and 1-based page numbers.
    """
    documents = []
    pdf_files = []

    if os.path.isfile(target_path):
        if target_path.lower().endswith(".pdf"):
            pdf_files = [target_path]
        else:
            raise ValueError(f"Target file '{target_path}' is not a PDF.")
    elif os.path.isdir(target_path):
        pdf_files = glob.glob(os.path.join(target_path, "**", "*.pdf"), recursive=True)
    else:
        raise FileNotFoundError(f"Path '{target_path}' does not exist.")

    if not pdf_files:
        print(f"Warning: No PDF files found in '{target_path}'.")
        return []

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    for file_path in pdf_files:
        try:
            print(f"  -> Loading '{os.path.basename(file_path)}'...")
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(file_path)
                # Store 1-based page number for intuitive user citations
                raw_page = doc.metadata.get("page", 0)
                doc.metadata["page_number"] = int(raw_page) + 1
            documents.extend(loaded_docs)
            print(f"     Loaded {len(loaded_docs)} pages.")
        except Exception as e:
            print(f"     Error loading {file_path}: {e}")

    return documents

def build_vector_store(data_source: str = "data", db_output_dir: str = "chroma_db", chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Chunks loaded documents, embeds them using FastEmbed, and builds/updates Chroma DB.
    """
    documents = load_documents(data_source)
    if not documents:
        print("No documents loaded. Aborting vector store generation.")
        return None

    # 1. Text Chunking with Overlap
    print(f"\nSplitting documents into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks across all documents.")

    # 2. Generating Embeddings using FastEmbed
    print("\nLoading ONNX fast embedding model ('BAAI/bge-small-en-v1.5')...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 3. Store in Chroma Vector Store
    print(f"Building/updating Chroma vector database at '{db_output_dir}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_output_dir
    )
    print(f"Vector store successfully saved to '{db_output_dir}'!\n")
    return vectorstore

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_vector_store(target)