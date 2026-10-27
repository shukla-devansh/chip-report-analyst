"""Measure how well the system works, so the README can show real numbers.

    python -m eval.run_eval                      # retrieval: compare the 4 search modes
    python -m eval.run_eval --compare-extraction # also: plain text vs rebuilt table rows
    python -m eval.run_eval --llm                # also: answer accuracy (needs GEMINI_API_KEY)

Retrieval hit rate: for each question we know which page(s) hold the answer.
A question is a "hit" if, for every company it asks about, at least one of the
returned chunks comes from a correct page.
"""

import argparse
import json
from pathlib import Path

from config import TOP_K
from rag.answer import NOT_FOUND, answer_question
from rag.search import MODES, HybridSearcher, load_embedder, load_reranker, load_searcher

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))


def is_hit(results: list[dict], expected: dict) -> bool:
    return all(
        any(r["company"] == company and r["page"] in pages for r in results)
        for company, pages in expected.items()
    )


def answer_is_correct(answer: str, answer_contains: list[list[str]]) -> bool:
    """Every group must match at least once. No groups means the right answer is 'not found'."""
    if not answer_contains:
        return NOT_FOUND.lower().rstrip(".") in answer.lower()
    return all(any(option.lower() in answer.lower() for option in group) for group in answer_contains)


def retrieval_hit_rate(searcher, mode: str) -> float:
    # Skip questions about companies whose PDF isn't in the index yet.
    answerable = [q for q in QUESTIONS
                  if q["expected"] and all(c in searcher.companies for c in q["expected"])]
    hits = sum(is_hit(searcher.search(q["question"], k=TOP_K, mode=mode), q["expected"]) for q in answerable)
    return hits / len(answerable)


def compare_modes(searcher):
    print(f"\nRetrieval hit@{TOP_K} by search mode")
    for mode in MODES:
        print(f"  {mode:<14} {retrieval_hit_rate(searcher, mode):.0%}")


def compare_extraction():
    """Build a second, temporary index from plain get_text() and compare."""
    from build_index import build_chunks
    from rag.chunker import searchable_text

    embed_chunks, embed_query = load_embedder()
    rerank = load_reranker()
    print(f"\nRetrieval hit@{TOP_K} (hybrid+rerank) by PDF extraction method")
    for label, rebuild in [("plain get_text()", False), ("rebuilt table rows", True)]:
        chunks = build_chunks(rebuild_lines=rebuild)
        searcher = HybridSearcher(chunks, embed_chunks([searchable_text(c) for c in chunks]), embed_query, rerank)
        print(f"  {label:<20} {retrieval_hit_rate(searcher, 'hybrid+rerank'):.0%}")


def evaluate_answers(searcher):
    correct, checked_numbers, unverified_numbers = 0, 0, 0
    print("\nAnswer accuracy (LLM)")
    for q in QUESTIONS:
        result = answer_question(q["question"], searcher)
        ok = answer_is_correct(result["answer"], q["answer_contains"])
        correct += ok
        checked_numbers += len(result["numbers"]["verified"]) + len(result["numbers"]["unverified"])
        unverified_numbers += len(result["numbers"]["unverified"])
        print(f"  {'PASS' if ok else 'FAIL'}  {q['question']}")
        if not ok:
            print(f"        got: {result['answer'][:200]!r}")
    print(f"\n  Correct answers: {correct}/{len(QUESTIONS)} ({correct / len(QUESTIONS):.0%})")
    if checked_numbers:
        print(f"  Numbers found in cited pages: {checked_numbers - unverified_numbers}/{checked_numbers}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-extraction", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()

    searcher = load_searcher()
    print(f"{len(QUESTIONS)} questions, companies in index: {', '.join(searcher.companies)}")
    compare_modes(searcher)
    if args.compare_extraction:
        compare_extraction()
    if args.llm:
        evaluate_answers(searcher)


if __name__ == "__main__":
    main()
