"""All settings for the project live here, so you only have one place to change things."""

from pathlib import Path

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "data" / "index"

# The reports we search. `url` is None when the site blocks scripted downloads
# and the PDF has to be saved by hand (see README).
REPORTS = [
    {
        "company": "NVIDIA",
        "title": "NVIDIA Form 10-K, fiscal year 2025",
        "file": "nvidia_fy2025_10k.pdf",
        "url": "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q4/177440d5-3b32-4185-8cc8-95500a9dc783.pdf",
    },
    {
        "company": "Qualcomm",
        "title": "Qualcomm Form 10-K, fiscal year 2025",
        "file": "qualcomm_fy2025_10k.pdf",
        "url": "https://s204.q4cdn.com/645488518/files/doc_financials/2025/q4/QCOM-09-28-25-FY2025-10-K-Final.pdf",
    },
    {
        "company": "Arm",
        "title": "Arm Holdings Form 20-F, fiscal year 2025",
        "file": "arm_fy2025_20f.pdf",
        "url": None,
    },
]

# Chunking: each chunk stays inside one page so every citation points to an exact page.
CHUNK_WORDS = 220
OVERLAP_LINES = 3

# Models. The embedding and re-ranking models run locally and are free.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemini-flash-latest"  # alias that always points to Google's current Flash model

# Retrieval
CANDIDATES = 30  # how many results each search method returns before fusion
TOP_K = 6        # how many chunks we finally give to the LLM
