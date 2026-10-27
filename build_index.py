"""Read every downloaded PDF, cut it into chunks, embed the chunks, save to data/index/.

Run once after downloading (takes a few minutes on a laptop CPU):
    python build_index.py
"""

import json

import numpy as np

from config import INDEX_DIR, PDF_DIR, REPORTS
from rag.chunker import chunk_pages, searchable_text
from rag.pdf_reader import read_pdf
from rag.search import load_embedder


def build_chunks(rebuild_lines: bool = True) -> list[dict]:
    chunks = []
    for report in REPORTS:
        path = PDF_DIR / report["file"]
        if not path.exists():
            print(f"Skipping {report['company']}: {path.name} not found")
            continue
        pages = read_pdf(path, rebuild_lines=rebuild_lines)
        report_chunks = chunk_pages(pages, report["company"])
        print(f"{report['company']}: {len(pages)} pages -> {len(report_chunks)} chunks")
        chunks += report_chunks
    return chunks


def main():
    chunks = build_chunks()
    embed_chunks, _ = load_embedder()
    embeddings = embed_chunks([searchable_text(c) for c in chunks])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    print(f"Saved {len(chunks)} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
