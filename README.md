# 🔍 SourceSleuth

> **Recover citations for orphaned quotes using local semantic search — powered by MCP.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)

---

## 🎯 The Problem

Every student has been there: you're polishing your research paper and find a brilliant quote — but you've lost the citation. Which paper was it from? Which page?

**SourceSleuth** solves this by running a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that semantically searches your academic PDFs. Connect it to your AI assistant (Claude Desktop, Cursor, Windsurf) or use the web UI to ask: *"Where did I get this quote?"*

Everything runs **locally on your machine** — no data leaves your laptop, no API keys needed.

---

## ✨ Features

| Capability | Type | Description |
|---|---|---|
| `find_orphaned_quote` | 🔧 Tool | Semantic search across all your PDFs for a quote or paraphrase |
| `ingest_pdfs` | 🔧 Tool | Batch-ingest a folder of PDFs into the local vector store |
| `ingest_arxiv` | 🔧 Tool | Preprocess & ingest arXiv paper abstracts for citation recovery |
| `get_store_stats` | 🔧 Tool | View statistics about indexed documents |
| `sourcesleuth://pdfs/{filename}` | 📄 Resource | Read the full text of any indexed PDF |
| `cite_recovered_source` | 💬 Prompt | Format recovered sources into proper APA/MLA/Chicago citations |
| **Web UI** | 🌐 Interface | Interactive browser-based search interface |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  MCP Host (Claude Desktop / Cursor / Windsurf)               │
│  ┌────────────────┐                                          │
│  │  MCP Client    │  ← stdio transport →  SourceSleuth MCP  │
│  └────────────────┘                        Server            │
└──────────────────────────────────────────────────────────────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              │                 │                 │
                        PDF Processor    Vector Store      SentenceTransformer
                        (PyMuPDF)        (FAISS)           (all-MiniLM-L6-v2)
                              │                 │
                       student_pdfs/        data/
                       (your papers)     (persisted index)
```

### Components

| Module | Responsibility |
|---|---|
| `src/mcp_server.py` | FastMCP server — exposes tools, resources, and prompts |
| `src/pdf_processor.py` | PDF text extraction (PyMuPDF) and chunking |
| `src/vector_store.py` | FAISS index management, embedding, persistence |
| `src/dataset_preprocessor.py` | arXiv metadata preprocessing, LaTeX cleaning, filtering |
| `src/ingest.py` | CLI tool for standalone ingestion |
| `app.py` | Streamlit web UI |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- An MCP-compatible host (e.g., [Claude Desktop](https://claude.ai/desktop)) — optional

### 1. Clone & Install

```bash
git clone https://github.com/Ishwarpatra/OpenSourceSleuth.git
cd sourcesleuth

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -e ".[dev,ui]"
```

### 2. Add Your PDFs

Drop your academic PDF files into the `student_pdfs/` directory:

```bash
cp ~/Downloads/research_paper.pdf student_pdfs/
```

### 3. Configure Your MCP Host (Optional)

#### Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/sourcesleuth"
    }
  }
}
```

#### Cursor / Windsurf

Add to your MCP settings:

```json
{
  "sourcesleuth": {
    "command": "python",
    "args": ["-m", "src.mcp_server"],
    "cwd": "/path/to/sourcesleuth"
  }
}
```

---

## 🖥️ Web UI

SourceSleuth includes a modern web interface for easy searching:

### Launch the Web UI

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

### Features

- 🔍 **Search Interface** — Paste orphaned quotes and find sources
- 📊 **Statistics Dashboard** — View indexed documents and chunks
- 📁 **PDF Upload** — Upload PDFs directly through the browser
- ⚙️ **Search Settings** — Adjust result count and similarity threshold
- 🗑️ **Index Management** — Refresh or clear the vector store

---

## 🖥️ CLI Usage

SourceSleuth includes a standalone CLI tool for ingestion without requiring an MCP host.

### Commands

```bash
# Ingest PDFs from a directory
sourcesleuth-ingest pdfs --directory /path/to/pdfs

# Ingest arXiv papers (by category)
sourcesleuth-ingest arxiv --category cs. --max-records 5000

# View vector store statistics
sourcesleuth-ingest stats

# Clear the vector store
sourcesleuth-ingest clear
```

### Using Python Directly

```bash
# Ingest PDFs
python -m src.ingest pdfs

# Ingest arXiv papers
python -m src.ingest arxiv --category cs.AI

# View stats
python -m src.ingest stats
```

---

## 📖 AI/ML Documentation

*Per hackathon reproducibility requirements, all model and data choices are documented here.*

### Dataset & Preprocessing

#### PDF Pipeline

| Parameter | Value | Rationale |
|---|---|---|
| **Input data** | Student's local PDF files | Privacy-first: no data leaves the machine |
| **Text extraction** | PyMuPDF (`fitz`) | Fast, accurate, handles complex layouts |
| **Chunk size** | 500 tokens (~2,000 chars) | Balances granularity (finding specific quotes) with context (retaining surrounding text) |
| **Chunk overlap** | 50 tokens (~200 chars) | Ensures sentences split at chunk boundaries remain recoverable |
| **Token estimation** | ~4 chars/token | Approximation for English academic text |

#### arXiv Metadata Pipeline

| Parameter | Value | Rationale |
|---|---|---|
| **Source** | [Kaggle arXiv Dataset](https://www.kaggle.com/Cornell-University/arxiv) | ~2.97M papers, comprehensive academic coverage |
| **Raw size** | ~5 GB (JSON-Lines) | One JSON object per line, memory-efficient streaming |
| **Processing** | Stream-read, line-by-line | Never loads full file into memory |
| **Text cleaning** | Strip LaTeX (`\textbf`, `$...$`, accents) | Produces clean text suitable for embedding |
| **Filtering** | By arXiv category prefix (e.g. `cs.`) and/or date | Creates focused, manageable subsets |
| **Output** | Cleaned JSON-Lines file | Each record: id, title, authors, abstract, categories, doi |

### Model Architecture

| Parameter | Value |
|---|---|
| **Model** | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Type** | Sentence-Transformer (bi-encoder) |
| **Embedding dimension** | 384 |
| **Model size** | ~80 MB |
| **Training data** | 1B+ sentence pairs (NLI, paraphrase, QA) |
| **Hardware requirement** | CPU only (no GPU needed) |

**Why this model?**

1. **CPU-efficient**: Runs on any student laptop without a GPU.
2. **High quality**: Strong zero-shot performance on semantic similarity tasks.
3. **Small footprint**: ~80 MB download, fast inference.
4. **Well-maintained**: Part of the widely-used Sentence-Transformers library.

### Vector Search

| Parameter | Value |
|---|---|
| **Index type** | FAISS `IndexFlatIP` |
| **Similarity metric** | Cosine similarity (via L2-normalized inner product) |
| **Search complexity** | O(n) exact search |
| **Persistence** | Binary FAISS index + JSON metadata |

**Why FAISS Flat Index?**

For the expected corpus size (< 100k chunks from a student's PDF library), exact search is both fast enough and guarantees the best possible results. Approximate indices (IVF, HNSW) add complexity without meaningful benefit at this scale.

---

## ⚙️ Configuration

SourceSleuth uses environment variables for configuration:

| Variable | Default | Description |
|---|---|---|
| `SOURCESLEUTH_PDF_DIR` | `./student_pdfs` | Directory containing PDF files |
| `SOURCESLEUTH_DATA_DIR` | `./data` | Directory for persisted vector store |

Example:

```bash
export SOURCESLEUTH_PDF_DIR="/home/student/papers"
export SOURCESLEUTH_DATA_DIR="/home/student/.sourcesleuth/data"
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run integration tests with your PDFs
pytest tests/test_integration_pdfs.py -v

# Run a specific test module
pytest tests/test_pdf_processor.py -v
```

---

## 📂 Project Structure

```
sourcesleuth/
├── src/
│   ├── __init__.py              # Package init
│   ├── mcp_server.py            # MCP server (tools, resources, prompts)
│   ├── pdf_processor.py         # PDF extraction & chunking
│   ├── vector_store.py          # FAISS vector store
│   ├── dataset_preprocessor.py  # arXiv metadata preprocessing
│   └── ingest.py                # CLI ingestion tool
├── student_pdfs/                # Your PDF files go here
├── data/                        # Persisted vector store + arXiv dataset
├── tests/
│   ├── test_pdf_processor.py    # PDF processor tests
│   ├── test_vector_store.py     # Vector store tests
│   ├── test_mcp_server.py       # MCP tool tests
│   ├── test_dataset_preprocessor.py  # Preprocessor tests
│   └── test_integration_pdfs.py # Integration tests with real PDFs
├── app.py                       # Streamlit web UI
├── pyproject.toml               # Project config & dependencies
├── requirements.txt             # Pip requirements
├── README.md                    # This file
├── CONTRIBUTING.md              # Contributor guide
├── ROADMAP.md                   # Development roadmap
└── LICENSE                      # Apache 2.0 License
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Ideas

- 🐛 **Bug fixes**: Find and fix edge cases in PDF parsing
- 📄 **Format support**: Add EPUB, DOCX, or Markdown ingestion
- 🧠 **Model options**: Support alternative embedding models
- 🎨 **Output formatting**: Improve citation formatting for more styles
- 📊 **Analytics**: Add a tool to compare two quotes for similarity
- 🧪 **Testing**: Increase test coverage

---

## 📜 License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) — The open standard for AI tool integration
- [Sentence-Transformers](https://sbert.net) — State-of-the-art sentence embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io) — Fast PDF text extraction
