import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-3.5-flash-lite", "gemini-3.8-flash"]

def get_rag_components(db_dir: str = "chroma_db", model_name: str = DEFAULT_MODEL, temperature: float = 0.3):
    """
    Initializes embeddings, Chroma vector store, retriever, and Gemini model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a valid GEMINI_API_KEY in your .env file!")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )
    return retriever, llm

def query_rag(question: str, db_dir: str = "chroma_db", model_name: str = DEFAULT_MODEL, return_data: bool = False):
    """
    Queries the RAG pipeline, retrieving context and citing document sources and page numbers.
    Includes automatic model fallback if a model hits rate limits or high demand.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a valid GEMINI_API_KEY in your .env file!")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)

    # 2. Extract citations metadata
    citations = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            citations.append({
                "source": source,
                "page": page,
                "snippet": snippet
            })

    # 3. Format context with document metadata for the LLM
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page_number', doc.metadata.get('page', 0) + 1)}]:\n{doc.page_content}"
        for doc in docs
    )

    # 4. Prompt with in-text citation instruction
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert research assistant. Answer the user's question accurately based ONLY on the provided context.\n"
            "Whenever you state a fact or detail, cite the source and page number in brackets (e.g. [paper.pdf, Page 3]).\n"
            "If the context does not contain enough information to answer the question, state that clearly."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    print(f"\nQuerying RAG System for: '{question}'...\n")

    # Candidate models to try in sequence
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    response = None
    last_error = None

    for candidate_model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=candidate_model,
                temperature=0.3,
                google_api_key=api_key
            )
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": context_text, "question": question})
            break
        except Exception as e:
            err_msg = str(e)
            last_error = e
            if any(term in err_msg for term in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                print(f"Notice: Model '{candidate_model}' reached limit or was busy. Trying fallback model...")
                continue
            else:
                raise e

    if response is None:
        raise RuntimeError(f"All candidate models failed. Last error: {last_error}")

    print("=== FINAL AI RESPONSE ===")
    print(response.strip())
    print("\n--- SOURCES & CITATIONS ---")
    if citations:
        for i, cite in enumerate(citations, 1):
            print(f"[{i}] {cite['source']} (Page {cite['page']}): \"{cite['snippet']}\"")
    else:
        print("No source documents retrieved.")

    if return_data:
        return {
            "answer": response.strip(),
            "citations": citations,
            "raw_docs": docs
        }
    return response.strip()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What is a Markov Decision Process and what are the two interacting entities?"
    query_rag(user_query)