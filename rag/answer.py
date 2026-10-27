"""Step 4: write the answer, with citations, and check its numbers.

1. Give the LLM the retrieved chunks, numbered [1], [2], ...
2. Tell it to answer ONLY from those chunks and cite them, or say it couldn't find it.
3. Check every number in the answer against the chunks it cited. LLMs sometimes
   invent or mix up figures; this catches that without needing another LLM call.
"""

import re

from rag.llm import ask_gemini

NOT_FOUND = "I couldn't find this in the reports."

PROMPT = """You answer questions about company annual reports (Form 10-K / 20-F).

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- After every fact, cite its source number like [2]. Copy figures exactly as written.
- Say which fiscal year a number belongs to; companies' fiscal years end on different dates.
- If the sources do not contain the answer, reply exactly: {not_found}

Sources:
{sources}

Question: {question}
Answer:"""

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")    # [1] or [1, 3]
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")       # 130,497 or 75.0 or 2025


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] {c['company']}, page {c['page']}, {c['section']}\n{c['text']}"
        for n, c in enumerate(chunks, start=1)
    )


def cited_numbers(answer: str) -> set[int]:
    """'Revenue grew [1] and margin rose [2, 3]' -> {1, 2, 3}"""
    found = set()
    for group in CITATION.findall(answer):
        found.update(int(n) for n in group.split(","))
    return found


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUMBER.findall(text)]


def check_numbers(answer: str, chunks: list[dict]) -> dict:
    """Split the answer's numbers into ones found in the cited chunks and ones that weren't.

    We compare values, not strings, so '75%' in the answer matches '75.0 %' in the report.
    A number in the 'unverified' list is not necessarily wrong (the LLM may have
    converted $130,497 million into $130.5 billion), but it is worth a second look.
    """
    cited = [chunks[n - 1] for n in cited_numbers(answer) if 1 <= n <= len(chunks)]
    evidence = cited or chunks  # no citations? check against everything retrieved
    known = set(numbers_in(" ".join(c["text"] for c in evidence)))

    answer_without_citations = CITATION.sub("", answer)
    verified, unverified = [], []
    for value in numbers_in(answer_without_citations):
        (verified if value in known else unverified).append(value)
    return {"verified": verified, "unverified": unverified}


def answer_question(question: str, searcher, llm=ask_gemini) -> dict:
    chunks = searcher.search(question)
    prompt = PROMPT.format(not_found=NOT_FOUND, sources=format_sources(chunks), question=question)
    answer = llm(prompt).strip()
    return {
        "answer": answer,
        "sources": chunks,
        "cited": sorted(cited_numbers(answer)),
        "numbers": check_numbers(answer, chunks),
    }
