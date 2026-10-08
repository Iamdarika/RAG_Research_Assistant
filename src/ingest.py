import os
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False


def extract_pdf_documents(file_path: str) -> List[Document]:
    """
    Extracts text from PDF with academic layout awareness (columns, tables, headers).
    Uses pymupdf4llm when available, with graceful fallback to PyPDFLoader.
    """
    documents = []
    base_name = os.path.basename(file_path)

    if PYMUPDF4LLM_AVAILABLE:
        try:
            page_chunks = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            for page in page_chunks:
                text = page.get("text", "").strip()
                if not text:
                    continue
                meta = page.get("metadata", {})
                page_num = meta.get("page_number", 1)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": base_name,
                        "file_path": file_path,
                        "page_number": int(page_num),
                        "page_count": meta.get("page_count", len(page_chunks)),
                        "title": meta.get("title", base_name),
                    }
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"     pymupdf4llm parser encountered an issue ({e}). Falling back to PyPDFLoader...")

    # Fallback to PyPDFLoader
    loader = PyPDFLoader(file_path)
    loaded_docs = loader.load()
    for doc in loaded_docs:
        doc.metadata["source"] = base_name
        doc.metadata["file_path"] = file_path
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["page_number"] = int(raw_page) + 1
        documents.append(doc)
    return documents


def load_documents(target_path: str = "data") -> List[Document]:
    """
    Scans a folder or individual file for PDFs and extracts layout-aware documents.
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
            print(f"  -> Extracting '{os.path.basename(file_path)}'...")
            docs = extract_pdf_documents(file_path)
            documents.extend(docs)
            print(f"     Extracted {len(docs)} pages.")
        except Exception as e:
            print(f"     Error processing {file_path}: {e}")

    return documents


def build_vector_store(
    data_source: str = "data",
    db_output_dir: str = "chroma_db",
    chunk_size: int = 800,
    chunk_overlap: int = 120
):
    """
    Processes PDFs, generates overlapping chunks, embeds with FastEmbed,
    and updates both ChromaDB and the BM25 chunks catalog.
    """
    os.makedirs(db_output_dir, exist_ok=True)
    documents = load_documents(data_source)
    if not documents:
        print("No documents loaded. Aborting vector store generation.")
        return None

    # 1. Text Chunking
    print(f"\nChunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks across all documents.")

    # 2. Save Chunks Catalog for BM25 Keyword Search
    chunks_catalog_path = os.path.join(db_output_dir, "processed_chunks.json")
    print(f"Saving BM25 chunks catalog to '{chunks_catalog_path}'...")
    catalog_data = [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(chunks_catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, ensure_ascii=False, indent=2)

    # 3. Dense Embeddings via FastEmbed
    print("\nLoading ONNX embedding model ('BAAI/bge-small-en-v1.5')...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 4. Store in Chroma Vector Store
    print(f"Building/updating Chroma vector database at '{db_output_dir}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_output_dir
    )
    print(f"Vector store and BM25 index successfully saved to '{db_output_dir}'!\n")
    return vectorstore


def get_library_summary(db_dir: str = "chroma_db") -> List[Dict[str, Any]]:
    """
    Returns statistics on indexed papers (titles, pages, chunk counts).
    """
    catalog_path = os.path.join(db_dir, "processed_chunks.json")
    if not os.path.exists(catalog_path):
        return []

    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        papers: Dict[str, Dict[str, Any]] = {}
        for item in data:
            meta = item.get("metadata", {})
            source = meta.get("source", "Unknown")
            page = meta.get("page_number", 1)
            
            if source not in papers:
                papers[source] = {
                    "source": source,
                    "max_page": page,
                    "chunks": 0
                }
            papers[source]["chunks"] += 1
            if page > papers[source]["max_page"]:
                papers[source]["max_page"] = page

        return list(papers.values())
    except Exception as e:
        print(f"Error loading library summary: {e}")
        return []


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    build_vector_store(target)