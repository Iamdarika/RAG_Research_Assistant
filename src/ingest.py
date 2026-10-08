import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store(pdf_path: str, db_output_dir: str = "chroma_db"):
    # 1. Load PDF Document
    print(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 2. Text Chunking with Overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Generating Embeddings using FastEmbed
    print("Loading ONNX fast embedding model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 4. Store in Chroma Vector Store
    print("Building Chroma vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_output_dir
    )
    print(f"Vector store successfully saved to '{db_output_dir}'!")

if __name__ == "__main__":
    pdf_file_path = "data/sample_paper.pdf"
    if os.path.exists(pdf_file_path):
        build_vector_store(pdf_file_path)
    else:
        print(f"Please place a PDF at '{pdf_file_path}' to run the script!")