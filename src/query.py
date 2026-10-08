import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever import HybridRerankRetriever

# Load environment variables
load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-3.5-flash-lite", "gemini-3.8-flash"]


def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Instantiates the Gemini LLM with API key validation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a valid GEMINI_API_KEY in your .env file!")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )


def rephrase_query(
    question: str,
    chat_history: List[Dict[str, str]],
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Rewrites a follow-up user question into a standalone academic search query
    based on the preceding conversation context.
    """
    if not chat_history:
        return question

    # Format last 4 turns for context
    recent_history = chat_history[-4:]
    history_lines = []
    for turn in recent_history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "").strip()
        # Truncate assistant responses so prompt stays lean
        if len(content) > 300:
            content = content[:297] + "..."
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an academic query optimizer. Given the chat history and the user's latest follow-up question, "
            "reformulate the question into a concise, standalone search query that contains all necessary technical context. "
            "Do NOT answer the question. Return ONLY the reformulated query text without quotes or preamble."
        )),
        ("human", "Conversation History:\n{history}\n\nLatest User Question: {question}\n\nStandalone Search Query:")
    ])

    for candidate in [model_name] + [m for m in FALLBACK_MODELS if m != model_name]:
        try:
            llm = get_llm(candidate, temperature=0.0)
            chain = prompt | llm | StrOutputParser()
            standalone = chain.invoke({"history": history_text, "question": question}).strip()
            if standalone:
                return standalone.strip('"\'')
        except Exception:
            continue
    return question


def query_rag(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    db_dir: str = "chroma_db",
    model_name: str = DEFAULT_MODEL,
    use_reranker: bool = True,
    final_k: int = 4,
    return_data: bool = False
) -> Any:
    """
    Executes the full Academic RAG pipeline:
    1. Conversational Query Rephrasing (if history present).
    2. Hybrid Search (Dense Chroma + Sparse BM25) + FlashRank Cross-Encoder Reranking.
    3. Grounded Synthesis via Gemini with in-text bracket citations.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide a valid GEMINI_API_KEY in your .env file!")

    # 1. Conversational Query Rephrasing
    search_query = question
    if chat_history:
        search_query = rephrase_query(question, chat_history, model_name=model_name)
        if search_query != question:
            print(f"[Query Rewriter]: '{question}' -> '{search_query}'")

    # 2. Hybrid Retrieval & Reranking
    retriever = HybridRerankRetriever(
        db_dir=db_dir,
        final_k=final_k,
        use_reranker=use_reranker
    )
    retrieved_results = retriever.retrieve(search_query, final_k=final_k)

    # 3. Format Context & Extract Citations
    context_blocks = []
    citations = []
    seen = set()

    for item in retrieved_results:
        source = item["source"]
        page = item["page"]
        score = item["score"]
        content = item["content"]
        key = (source, page)

        context_blocks.append(f"[Source: {source}, Page {page} | Relevance Score: {score:.3f}]:\n{content}")

        if key not in seen:
            seen.add(key)
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            citations.append({
                "source": source,
                "page": page,
                "score": score,
                "snippet": snippet
            })

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found in the documents."

    # 4. Prompt Setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a rigorous, expert research assistant analyzing academic papers.\n"
            "Answer the user's question thoroughly and accurately based ONLY on the provided context.\n"
            "Guidelines:\n"
            "- Whenever you state a technical fact, theorem, methodology, or finding, cite the source and page in brackets: [SourceFile.pdf, Page X].\n"
            "- If the context discusses equations, algorithms, or definitions, explain them clearly.\n"
            "- If the provided context does not contain enough evidence to answer the question, state that clearly without guessing."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # 5. Model Execution with Resilient Fallback
    response = None
    last_error = None
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    for candidate_model in candidates:
        try:
            llm = get_llm(candidate_model, temperature=0.25)
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

    answer = response.strip()

    if return_data:
        return {
            "answer": answer,
            "citations": citations,
            "search_query": search_query,
            "retrieved_results": retrieved_results
        }

    # CLI Output Display
    print("\n=== FINAL AI RESPONSE ===")
    print(answer)
    print("\n--- SOURCES & CITATIONS ---")
    if citations:
        for i, cite in enumerate(citations, 1):
            print(f"[{i}] {cite['source']} (Page {cite['page']}, Score: {cite['score']:.4f}):\n    \"{cite['snippet']}\"")
    else:
        print("No source documents retrieved.")

    return answer


def interactive_cli():
    """Runs a multi-turn interactive chat session in the terminal."""
    print("=" * 65)
    print("📚 Academic RAG Assistant — Interactive Terminal Mode")
    print("Type your questions below. Commands: 'exit' to quit, 'clear' to reset chat.")
    print("=" * 65)

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                history.clear()
                print("Conversation history cleared.")
                continue

            result = query_rag(user_input, chat_history=history, return_data=True)
            print(f"\n[AI Assistant]:\n{result['answer']}")

            print("\n[Citations]:")
            for i, c in enumerate(result["citations"], 1):
                print(f"  [{i}] {c['source']} (Page {c['page']}, Score: {c['score']:.2f})")

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result["answer"]})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_rag(" ".join(sys.argv[1:]))
    else:
        interactive_cli()