import os
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from src.vector_store import VectorStore
from src.pdf_processor import process_pdf_directory, extract_text_from_pdf
from src.ocr_processor import process_ocr_pdf_directory, extract_text_from_image_file, check_easyocr_availability, check_tesseract_availability

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


def ingest_pdfs_ui(pdf_directory, use_ocr=False):
    """Ingest PDFs and return status."""
    try:
        if use_ocr:
            chunks = process_ocr_pdf_directory(pdf_directory, force_ocr=use_ocr)
        else:
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
    st.markdown("### 📁 Upload Files")
    
    # Check OCR availability
    ocr_available = check_easyocr_availability() or check_tesseract_availability()
    
    if not ocr_available:
        st.warning("⚠️ OCR not available. Install `easyocr` or `pytesseract` for image support.")
    
    uploaded_pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to add to the search index"
    )
    
    uploaded_images = st.file_uploader(
        "Upload Images (for OCR)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        help="Upload image files to extract text using OCR",
        disabled=not ocr_available
    )

    # OCR option for PDFs
    use_ocr = False
    if uploaded_pdfs and ocr_available:
        use_ocr = st.checkbox(
            "Use OCR for PDFs",
            value=False,
            help="Enable OCR for scanned/image-only PDFs. Slower but can extract text from images."
        )

    if uploaded_pdfs or uploaded_images:
        if st.button("📥 Process Uploaded Files"):
            with st.spinner("Processing files..."):
                # Create temp directory for uploaded files
                temp_dir = tempfile.mkdtemp()
                chunks = []

                # Process PDFs
                for uploaded_file in uploaded_pdfs:
                    save_path = Path(temp_dir) / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        if use_ocr:
                            from src.ocr_processor import extract_text_from_image_pdf, chunk_text as ocr_chunk, OCRDocument, PageSpan
                            doc = extract_text_from_image_pdf(save_path)
                            doc_chunks = ocr_chunk(doc)
                            chunks.extend(doc_chunks)
                            st.success(f"✅ OCR processed: {uploaded_file.name}")
                        else:
                            doc = extract_text_from_pdf(save_path)
                            from src.pdf_processor import chunk_text
                            doc_chunks = chunk_text(doc)
                            chunks.extend(doc_chunks)
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")

                # Process Images with OCR
                if ocr_available:
                    for uploaded_img in uploaded_images:
                        save_path = Path(temp_dir) / uploaded_img.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_img.getbuffer())

                        try:
                            text, confidence = extract_text_from_image_file(save_path)
                            if text.strip():
                                from src.ocr_processor import OCRDocument, PageSpan, chunk_text as ocr_chunk
                                document = OCRDocument(
                                    filename=uploaded_img.name,
                                    full_text=text,
                                    page_spans=[PageSpan(page=1, start_char=0,
                                                        end_char=len(text), confidence=confidence)],
                                    ocr_engine="easyocr" if check_easyocr_availability() else "pytesseract",
                                )
                                img_chunks = ocr_chunk(document)
                                chunks.extend(img_chunks)
                                st.success(f"✅ OCR extracted from {uploaded_img.name} (confidence: {confidence:.2f})")
                            else:
                                st.warning(f"⚠️ No text found in {uploaded_img.name}")
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_img.name}: {e}")

                if chunks:
                    store = get_vector_store()
                    added = store.add_chunks(chunks)
                    store.save()
                    st.success(f"✅ Added {added} chunks from {len(uploaded_pdfs) + len(uploaded_images)} file(s)")
                else:
                    st.warning("No text could be extracted from uploaded files.")
    
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

# OCR Document Scanner Section
st.subheader("📷 OCR Document Scanner")
st.markdown("Scan text from images or scanned PDFs and search for citations")

ocr_tab1, ocr_tab2 = st.tabs(["📄 Scan Single Document", "📚 Batch OCR Processing"])

with ocr_tab1:
    col1, col2 = st.columns(2)
    with col1:
        single_file = st.file_uploader(
            "Upload image or scanned PDF",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"],
            key="single_ocr",
            disabled=not ocr_available
        )
    
    with col2:
        if single_file:
            st.info(f"📄 **{single_file.name}**")
            
            if st.button("🔍 Scan & Search", key="scan_single"):
                with st.spinner("Performing OCR..."):
                    temp_dir = tempfile.mkdtemp()
                    save_path = Path(temp_dir) / single_file.name
                    
                    with open(save_path, "wb") as f:
                        f.write(single_file.getbuffer())
                    
                    try:
                        if single_file.name.lower().endswith('.pdf'):
                            from src.ocr_processor import extract_text_from_image_pdf, chunk_text as ocr_chunk
                            doc = extract_text_from_image_pdf(save_path)
                            text = doc.full_text
                            chunks = ocr_chunk(doc)
                        else:
                            text, confidence = extract_text_from_image_file(save_path)
                            from src.ocr_processor import OCRDocument, PageSpan, chunk_text as ocr_chunk
                            doc = OCRDocument(
                                filename=single_file.name,
                                full_text=text,
                                page_spans=[PageSpan(page=1, start_char=0, end_char=len(text), confidence=confidence)],
                            )
                            chunks = ocr_chunk(doc)
                        
                        # Add to store
                        store = get_vector_store()
                        added = store.add_chunks(chunks)
                        store.save()
                        
                        st.success(f"✅ Extracted text and added {added} chunks")
                        
                        # Show extracted text
                        with st.expander("📝 View Extracted Text"):
                            st.text(text)
                        
                        # Auto-search the extracted text
                        if text.strip():
                            st.markdown("🔎 **Searching for related content...**")
                            results = store.search(query=text[:500], top_k=3)
                            if results:
                                st.markdown("📍 **Related content found:**")
                                for r in results:
                                    if r['filename'] != single_file.name:
                                        st.markdown(f"- **{r['filename']}** (p.{r['page']}, score: {r['score']:.2f})")
                    except Exception as e:
                        st.error(f"❌ OCR failed: {e}")

with ocr_tab2:
    st.markdown("Process multiple images or scanned PDFs at once")
    
    batch_files = st.file_uploader(
        "Upload multiple files",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "pdf"],
        accept_multiple_files=True,
        key="batch_ocr",
        disabled=not ocr_available
    )
    
    if batch_files:
        if st.button("🚀 Process All with OCR"):
            with st.spinner("Processing files with OCR..."):
                temp_dir = tempfile.mkdtemp()
                all_chunks = []
                
                progress_bar = st.progress(0)
                
                for i, file in enumerate(batch_files):
                    save_path = Path(temp_dir) / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        if file.name.lower().endswith('.pdf'):
                            from src.ocr_processor import extract_text_from_image_pdf, chunk_text as ocr_chunk
                            doc = extract_text_from_image_pdf(save_path)
                            chunks = ocr_chunk(doc)
                        else:
                            text, confidence = extract_text_from_image_file(save_path)
                            from src.ocr_processor import OCRDocument, PageSpan, chunk_text as ocr_chunk
                            doc = OCRDocument(
                                filename=file.name,
                                full_text=text,
                                page_spans=[PageSpan(page=1, start_char=0, end_char=len(text), confidence=confidence)],
                            )
                            chunks = ocr_chunk(doc)
                        
                        all_chunks.extend(chunks)
                    except Exception as e:
                        st.error(f"Failed {file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(batch_files))
                
                if all_chunks:
                    store = get_vector_store()
                    added = store.add_chunks(all_chunks)
                    store.save()
                    st.success(f"✅ Processed {len(batch_files)} files, added {added} chunks")
                else:
                    st.warning("No text extracted from files")

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
