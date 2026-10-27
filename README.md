# 📊 Chip Report Analyst

Ask plain-English questions about the fiscal 2025 annual reports of **Arm, NVIDIA and Qualcomm**,
and get answers that cite the exact page they came from.

> *"Compare the research and development spending of NVIDIA and Qualcomm."*
>
> NVIDIA spent $12,914 million on R&D in fiscal 2025 [1], while Qualcomm spent $9,042 million [4]...

It is a **RAG** (Retrieval-Augmented Generation) system: instead of trusting what an LLM
remembers, it first *finds* the relevant pages in the reports and then makes the LLM answer
**only** from those pages.

## What makes it more than a "chat with a PDF" demo

| Problem | What this project does |
|---|---|
| Plain PDF text extraction breaks tables into one cell per line, so numbers lose their row labels | Rebuilds table rows from word positions, and strips page headers/footers (`rag/pdf_reader.py`) |
| Keyword search misses synonyms; vector search misses exact numbers | **Hybrid search**: BM25 + embeddings, merged with Reciprocal Rank Fusion, then a cross-encoder **re-ranker** (`rag/search.py`) |
| "Compare A and B" questions get flooded by results from one company | Detects the companies named in the question and searches each one separately |
| LLMs sometimes invent or mix up figures | Every number in the answer is checked against the pages it cites (`rag/answer.py`) |
| No way to know if a change helped | A 42-question test set with known answer pages, and an evaluation script (`eval/`) |

## Results

_Evaluation results coming soon (see `eval/`)._

## How it works

```
 PDFs ──► pdf_reader.py ──► chunker.py ──► build_index.py ──► data/index/
          rebuild table     ~220-word         embed every       chunks.json
          rows, drop        chunks, each      chunk             embeddings.npy
          headers/footers   tagged with
                            company, page,
                            section
                                                     │
 Question ──► search.py ─────────────────────────────┘
              BM25 + vector search ─► RRF fusion ─► re-rank ─► top 6 chunks
                                                                  │
              answer.py ◄─────────────────────────────────────────┘
              prompt with numbered sources ─► Gemini ─► answer with [1] [2] citations
              ─► check every number against the cited chunks
```

| File | What it does |
|---|---|
| `config.py` | Every setting in one place (reports, models, chunk size) |
| `download_reports.py` | Downloads the PDFs |
| `rag/pdf_reader.py` | PDF → clean lines, rebuilding table rows |
| `rag/chunker.py` | Lines → chunks tagged with company, page and section |
| `build_index.py` | Embeds all chunks and saves the index |
| `rag/search.py` | Hybrid search, fusion, re-ranking, company detection |
| `rag/llm.py` | The only file that calls the LLM (Gemini) |
| `rag/answer.py` | Prompt, citations, number checking |
| `app.py` | Streamlit web app |
| `eval/` | Test questions and the evaluation script |
| `tests/` | Unit tests (no internet or API key needed) |

## Run it yourself

**1. Install** (Python 3.11+)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**2. Get the reports**

```bash
python download_reports.py
```

This downloads the NVIDIA and Qualcomm 10-Ks. Arm's investor site blocks scripts, so get Arm's
fiscal 2025 **Form 20-F** by hand: open the
[SEC copy](https://www.sec.gov/Archives/edgar/data/1973239/000197323925000016/arm-20250331.htm),
press **Ctrl+P → Save as PDF**, and save it as `data/pdfs/arm_fy2025_20f.pdf`.
(The browser adds a header and footer to every page; `pdf_reader.py` removes them.)
The app also works with just the two 10-Ks.

**3. Build the search index** (a few minutes; downloads two small local models the first time)

```bash
python build_index.py
```

**4. Add a Gemini API key** (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

Copy `.env.example` to `.env` and paste your key. Without a key the app still shows the matching pages.

**5. Run**

```bash
streamlit run app.py
```

**Tests and evaluation**

```bash
pytest                                   # unit tests
python -m eval.run_eval                  # compare search modes
python -m eval.run_eval --compare-extraction
python -m eval.run_eval --llm            # answer accuracy (needs the API key)
```

## Things to try next

- Answer from the financial-statement tables with SQL instead of text search
- Show the PDF page image next to each source
- Add more years to compare trends

## Data

The reports are public filings published by each company. They are not included in this repository;
`download_reports.py` fetches them from the companies' own investor-relations sites.
This project is not affiliated with Arm, NVIDIA or Qualcomm.
