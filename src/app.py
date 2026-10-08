import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Add project root and src directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingest import build_vector_store, get_library_summary
from query import query_rag, DEFAULT_MODEL, FALLBACK_MODELS

# Streamlit Page Config
st.set_page_config(
    page_title="Academic RAG Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .source-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .badge-score {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .rephrased-badge {
        background-color: #f1f5f9;
        color: #475569;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to your Academic RAG Assistant!**\n\nI use **Layout-Aware PDF Extraction**, **Hybrid Search (BM25 + Chroma)**, and **FlashRank Cross-Encoder Reranking** to provide grounded, citation-backed answers. Ask any technical question or upload new papers to get started."
        }
    ]

# Sidebar Controls
with st.sidebar:
    st.title("🔬 Research Controls")

    # API Key check
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ `GEMINI_API_KEY` not found in `.env`!")
    else:
        st.success("✅ Gemini API connected")

    st.divider()

    # Retrieval Engine Settings
    st.subheader("⚡ Retrieval & Reranker")
    use_reranker = st.toggle("FlashRank Reranking", value=True, help="Cross-encoder scoring for maximum precision")
    num_chunks = st.slider("Top Citations (K)", min_value=2, max_value=8, value=4, step=1)
    
    st.subheader("🤖 LLM Settings")
    model_choice = st.selectbox(
        "Active Model",
        [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL],
        index=0
    )

    st.divider()

    # Document Upload & Ingestion
    st.subheader("📄 Ingest Research Papers")
    uploaded_files = st.file_uploader(
        "Upload PDFs (Academic layout aware)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Index Papers", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF file.")
        else:
            os.makedirs("data", exist_ok=True)
            saved_files = []
            for uploaded_file in uploaded_files:
                save_path = os.path.join("data", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(uploaded_file.name)

            with st.spinner(f"Extracting layouts & embedding {len(saved_files)} file(s)..."):
                try:
                    build_vector_store("data")
                    st.success(f"Successfully indexed {len(saved_files)} papers!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    # Indexed Library Overview
    st.divider()
    st.subheader("📚 Indexed Papers")
    library = get_library_summary()
    if library:
        for doc in library:
            st.markdown(f"📄 **`{doc['source']}`**  \n└ {doc['max_page']} pages • {doc['chunks']} chunks")
    else:
        st.caption("No papers cataloged yet.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat history reset. How can I help you with your research?"}
            ]
            st.rerun()
    with col2:
        # Export chat as Markdown
        if len(st.session_state.messages) > 1:
            chat_md = "# Research Assistant Chat Report\n\n"
            for m in st.session_state.messages:
                role = "User" if m["role"] == "user" else "Assistant"
                chat_md += f"### {role}\n{m['content']}\n\n"
            st.download_button(
                label="📥 Export Report",
                data=chat_md,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True
            )

# Main UI Header
st.title("🔬 Academic RAG Research Assistant")
st.caption("State-of-the-art grounded QA: PyMuPDF4LLM Markdown parsing • BM25 + Vector Hybrid Retrieval • FlashRank Cross-Encoder • Contextual Memory")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show rephrased search query if available
        if message.get("search_query") and message.get("search_query") != message.get("original_query"):
            st.markdown(f"<span class='rephrased-badge'>🔍 Rephrased Query: {message['search_query']}</span>", unsafe_allow_html=True)

        st.markdown(message["content"])

        # Show citations
        if "citations" in message and message["citations"]:
            with st.expander(f"📌 {len(message['citations'])} Verified Citations & Evidence", expanded=False):
                for i, citation in enumerate(message["citations"], 1):
                    score_pct = f"{citation['score'] * 100:.1f}%" if citation['score'] <= 1.0 else f"{citation['score']:.2f}"
                    st.markdown(
                        f"**[{i}] `{citation['source']}` — Page {citation['page']}** &nbsp; "
                        f"<span class='badge-score'>Relevance: {score_pct}</span>\n\n"
                        f"> \"{citation['snippet']}\"",
                        unsafe_allow_html=True
                    )

# Chat Input & Processing
if prompt := st.chat_input("Ask a research question or follow-up..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving hybrid evidence & reranking passages..."):
            try:
                # Pass chat history excluding the latest user message
                history_for_rephrasing = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                result = query_rag(
                    question=prompt,
                    chat_history=history_for_rephrasing,
                    model_name=model_choice,
                    use_reranker=use_reranker,
                    final_k=num_chunks,
                    return_data=True
                )

                answer = result["answer"]
                citations = result.get("citations", [])
                search_query = result.get("search_query", prompt)

                # Show rephrased search query if different
                if search_query != prompt:
                    st.markdown(f"<span class='rephrased-badge'>🔍 Contextual Search: {search_query}</span>", unsafe_allow_html=True)

                st.markdown(answer)

                if citations:
                    with st.expander(f"📌 {len(citations)} Verified Citations & Evidence", expanded=False):
                        for i, citation in enumerate(citations, 1):
                            score_pct = f"{citation['score'] * 100:.1f}%" if citation['score'] <= 1.0 else f"{citation['score']:.2f}"
                            st.markdown(
                                f"**[{i}] `{citation['source']}` — Page {citation['page']}** &nbsp; "
                                f"<span class='badge-score'>Relevance: {score_pct}</span>\n\n"
                                f"> \"{citation['snippet']}\"",
                                unsafe_allow_html=True
                            )

                # Store in session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "search_query": search_query,
                    "original_query": prompt
                })

            except Exception as e:
                st.error(f"Error querying assistant: {e}")
