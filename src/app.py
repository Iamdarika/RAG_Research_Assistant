import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Add project root and src directory to sys.path so imports work cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingest import build_vector_store
from query import query_rag

# Streamlit Page Config
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .source-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .badge-source {
        background-color: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to your RAG Research Assistant!**\n\nI can answer technical questions with grounded citations from your research papers. Upload new PDFs using the sidebar, or ask a question directly about already indexed documents."
        }
    ]

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")

    # API Key check
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ `GEMINI_API_KEY` is not found in `.env`!")
    else:
        st.success("✅ Gemini API connected")

    st.divider()

    st.subheader("📄 Upload & Ingest Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Ingest Papers", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            os.makedirs("data", exist_ok=True)
            saved_count = 0
            for uploaded_file in uploaded_files:
                save_path = os.path.join("data", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            
            with st.spinner(f"Ingesting and embedding {saved_count} document(s)..."):
                try:
                    build_vector_store("data")
                    st.success(f"Successfully processed {saved_count} paper(s)!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    st.divider()

    st.subheader("🤖 LLM Settings")
    model_choice = st.selectbox(
        "Model",
        ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.rerun()

# Main Header
st.title("📚 RAG Research Assistant")
st.caption("Grounded, cited AI answers powered by LangChain, FastEmbed, Chroma, and Google Gemini.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📌 Source Citations & References", expanded=False):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(
                        f"**[{i}] `{citation['source']}` — Page {citation['page']}**\n\n"
                        f"> \"{citation['snippet']}\""
                    )

# Chat Input & Response Processing
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing papers and retrieving relevant sections..."):
            try:
                result = query_rag(
                    question=prompt,
                    model_name=model_choice,
                    return_data=True
                )
                answer = result["answer"]
                citations = result.get("citations", [])

                st.markdown(answer)
                if citations:
                    with st.expander("📌 Source Citations & References", expanded=False):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(
                                f"**[{i}] `{citation['source']}` — Page {citation['page']}**\n\n"
                                f"> \"{citation['snippet']}\""
                            )

                # Persist in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
            except Exception as e:
                st.error(f"Error querying assistant: {e}")
