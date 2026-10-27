"""The web app.  Run:  streamlit run app.py"""

import streamlit as st

from config import INDEX_DIR
from rag.answer import answer_question
from rag.llm import has_api_key
from rag.search import load_searcher

EXAMPLES = [
    "Compare the research and development spending of Arm, NVIDIA and Qualcomm.",
    "What was NVIDIA's Data Center revenue in fiscal year 2025?",
    "What is the lawsuit between Arm and Qualcomm about?",
    "Which foundries do NVIDIA and Qualcomm depend on?",
    "What risk does Qualcomm face from customers building their own modems?",
]

st.set_page_config(page_title="Chip Report Analyst", page_icon="📊", layout="wide")
st.title("📊 Chip Report Analyst")
st.caption("Ask questions about the fiscal 2025 annual reports of Arm, NVIDIA and Qualcomm. "
           "Every answer cites the page it came from.")

if not (INDEX_DIR / "chunks.json").exists():
    st.error("No index yet. Run `python download_reports.py` and then `python build_index.py`.")
    st.stop()


@st.cache_resource  # load the models once, not on every click
def get_searcher():
    return load_searcher()


searcher = get_searcher()

with st.sidebar:
    st.subheader("Reports loaded")
    st.write(", ".join(searcher.companies))
    st.subheader("Try a question")
    for example in EXAMPLES:
        st.button(example, on_click=lambda q=example: st.session_state.update(question=q))

question = st.text_input("Your question", key="question")
if not question:
    st.stop()

if has_api_key():
    with st.spinner("Reading the reports..."):
        result = answer_question(question, searcher)
    st.markdown(result["answer"].replace("$", r"\$"))  # "$" would otherwise start a math formula

    numbers = result["numbers"]
    if numbers["unverified"]:
        listed = ", ".join(f"{n:,g}" for n in numbers["unverified"])
        st.warning(f"Double-check these numbers, they are not on the cited pages: {listed}")
    elif numbers["verified"]:
        st.success(f"All {len(numbers['verified'])} numbers in the answer appear on the cited pages.")
    sources, cited = result["sources"], result["cited"]
else:
    st.info("No GEMINI_API_KEY found, so here are the matching pages only. See the README to add a key.")
    sources, cited = searcher.search(question), []

st.subheader("Sources")
for n, chunk in enumerate(sources, start=1):
    label = f"[{n}] {chunk['company']} · page {chunk['page']} · {chunk['section']}"
    with st.expander(label + ("  ✅ cited" if n in cited else "")):
        st.text(chunk["text"])
