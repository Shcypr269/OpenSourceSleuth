"""
SourceSleuth Web UI - Streamlit Frontend

A web interface for searching orphaned quotes across your academic PDFs.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from src.vector_store import VectorStore
from src.pdf_processor import process_pdf_directory, extract_text_from_pdf

# --- Page Configuration ---
st.set_page_config(
    page_title="SourceSleuth | Citation Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main { 
        background-color: #f5f7f9; 
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #01579b; 
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover { 
        background-color: #0277bd; 
    }
    .citation-card { 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 5px solid #01579b; 
        background: white; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        margin-bottom: 1rem; 
    }
    .citation-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #01579b;
    }
    .stat-label {
        color: #666;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.environ.get("SOURCESLEUTH_PDF_DIR", str(PROJECT_ROOT / "student_pdfs")))
DATA_DIR = Path(os.environ.get("SOURCESLEUTH_DATA_DIR", str(PROJECT_ROOT / "data")))


# --- Helper Functions ---
@st.cache_resource
def get_vector_store():
    """Load or initialize the vector store."""
    store = VectorStore(data_dir=DATA_DIR)
    store.load()
    return store


def ingest_pdfs_ui(pdf_directory):
    """Ingest PDFs and return status."""
    try:
        chunks = process_pdf_directory(pdf_directory)
        if not chunks:
            return False, "No text could be extracted from PDFs."
        
        store = get_vector_store()
        added = store.add_chunks(chunks)
        store.save()
        
        files_set = {c.filename for c in chunks}
        return True, f"✅ Ingested {len(files_set)} PDF(s), {added} chunks"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def format_confidence(score):
    """Format confidence score with badge."""
    if score >= 0.75:
        return f"🟢 High ({score:.2f})"
    elif score >= 0.50:
        return f"🟡 Medium ({score:.2f})"
    else:
        return f"🔴 Low ({score:.2f})"


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Configure your search parameters")
    
    st.divider()
    
    # Search settings
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    min_score = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    st.divider()
    
    # File upload section
    st.markdown("### 📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to add to the search index"
    )
    
    if uploaded_files:
        if st.button("📥 Process Uploaded PDFs"):
            with st.spinner("Processing PDFs..."):
                # Create temp directory for uploaded files
                temp_dir = tempfile.mkdtemp()
                saved_paths = []
                
                for uploaded_file in uploaded_files:
                    save_path = Path(temp_dir) / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(save_path)
                
                # Process uploaded PDFs
                chunks = []
                for pdf_path in saved_paths:
                    try:
                        doc = extract_text_from_pdf(pdf_path)
                        from src.pdf_processor import chunk_text
                        doc_chunks = chunk_text(doc)
                        chunks.extend(doc_chunks)
                    except Exception as e:
                        st.error(f"Failed to process {pdf_path.name}: {e}")
                
                if chunks:
                    store = get_vector_store()
                    added = store.add_chunks(chunks)
                    store.save()
                    st.success(f"✅ Added {added} chunks from {len(uploaded_files)} PDF(s)")
                else:
                    st.warning("No text could be extracted from uploaded PDFs.")
    
    st.divider()
    
    # Maintenance section
    st.markdown("### 🛠️ Maintenance")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Index", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Index", use_container_width=True):
            store = get_vector_store()
            store.clear()
            # Remove persisted files
            index_path = DATA_DIR / "sourcesleuth.index"
            meta_path = DATA_DIR / "sourcesleuth_metadata.json"
            if index_path.exists():
                index_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
            st.cache_resource.clear()
            st.success("Index cleared!")
            st.rerun()
    
    st.divider()
    
    # Info section
    st.markdown("### ℹ️ About")
    st.markdown("""
    **SourceSleuth** helps you recover citations for orphaned quotes using local semantic search.
    
    - 🔒 All data stays on your machine
    - 🚀 No API keys required
    - 📚 Powered by FAISS + SentenceTransformers
    """)
    
    st.markdown("---")
    st.caption(f"Version 1.0.0 | Apache 2.0 License")


# --- Main UI ---
st.title("🔍 SourceSleuth")
st.caption("Recover citations for orphaned quotes using local semantic search — powered by MCP")

# Load vector store and get stats
store = get_vector_store()
stats = store.get_stats()

# Display stats in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['total_chunks']}</div>
        <div class="stat-label">Total Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['num_files']}</div>
        <div class="stat-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{stats['embedding_dim']}</div>
        <div class="stat-label">Dimensions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">CPU</div>
        <div class="stat-label">Mode</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Input Area
st.subheader("📝 Search for Orphaned Quote")
query = st.text_area(
    "Paste the quote or paraphrase you want to find the source for:",
    placeholder="e.g., 'The attention mechanism allows models to focus on specific parts of the input sequence...',",
    height=100
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    search_clicked = st.button("🕵️ Find Sources")

# --- Results Section ---
if search_clicked and query:
    st.divider()
    st.subheader("📍 Search Results")
    
    with st.spinner("Searching your documents..."):
        results = store.search(query=query, top_k=top_k)
    
    if not results:
        st.warning("No matching sources found. Try uploading more PDFs or adjusting your search query.")
    else:
        # Filter by minimum score
        filtered_results = [r for r in results if r["score"] >= min_score]
        
        if not filtered_results:
            st.warning(f"No results above the minimum score threshold ({min_score}). Try lowering the threshold.")
        else:
            st.success(f"Found {len(filtered_results)} potential match(es)!")
            
            # Display results
            for i, result in enumerate(filtered_results, start=1):
                with st.container():
                    st.markdown(f"""
                    <div class="citation-card">
                        <h4>📄 {result['filename']}</h4>
                        <p><i>"{result['text'][:300]}{'...' if len(result['text']) > 300 else ''}"</i></p>
                        <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.85em;">
                            <span><b>📃 Page:</b> {result['page']}</span>
                            <span><b>🎯 Confidence:</b> {format_confidence(result['score'])}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show full context in expander
                    with st.expander(f"View full context for result {i}"):
                        st.text(result['text'])
                        st.caption(f"Character range: {result.get('start_char', 'N/A')} - {result.get('end_char', 'N/A')}")

elif search_clicked and not query:
    st.warning("⚠️ Please enter a quote to search.")

else:
    # Show example queries
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    - *"Attention is all you need for sequence transduction"*
    - *"Wave interference produces a pattern of bright and dark fringes"*
    - *"The photoelectric effect demonstrates the particle nature of light"*
    
    Paste your orphaned quote above and click **Find Sources** to begin!
    """)

# --- Document List Section ---
if stats['num_files'] > 0:
    st.divider()
    st.subheader("📚 Indexed Documents")
    
    # Create DataFrame for display
    df = pd.DataFrame({
        "Filename": stats['ingested_files'],
        "Status": ["✅ Indexed"] * len(stats['ingested_files'])
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- Footer ---
st.divider()
st.markdown("""
<center>
    <p style="color: #666;">
        Powered by <b>Model Context Protocol (MCP)</b> | 
        <a href="https://github.com/Ishwarpatra/OpenSourceSleuth">GitHub</a>
    </p>
</center>
""", unsafe_allow_html=True)
