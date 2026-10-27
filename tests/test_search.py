import numpy as np

from rag.search import HybridSearcher, companies_in, reciprocal_rank_fusion, tokenize


def test_tokenize_keeps_numbers_whole():
    assert tokenize("Revenue $ 130,497 and 75.0%") == ["revenue", "130,497", "and", "75.0"]


def test_companies_in():
    available = ["Arm", "NVIDIA", "Qualcomm"]
    assert companies_in("Compare Arm and NVIDIA", available) == ["NVIDIA", "Arm"]
    assert companies_in("What is QCOM's revenue?", available) == ["Qualcomm"]
    assert companies_in("Who makes the chips?", available) == []
    assert companies_in("Tell me about Arm", ["NVIDIA"]) == []  # not in the index


def test_rrf_rewards_items_ranked_high_in_both_lists():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1])
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2  # 2nd in both beats 1st in one


def make_searcher():
    chunks = [
        {"company": "NVIDIA", "page": 1, "section": "Item 7", "text": "Revenue was 130,497 million"},
        {"company": "NVIDIA", "page": 2, "section": "Item 1A", "text": "Export controls limit sales to China"},
        {"company": "Qualcomm", "page": 1, "section": "Item 7", "text": "Revenues were 44,284 million"},
        {"company": "Qualcomm", "page": 2, "section": "Item 1A", "text": "Apple may build its own modems"},
    ]
    # Fake embeddings: each chunk gets its own direction, and every query looks like chunk 1.
    embeddings = np.eye(4)
    return HybridSearcher(chunks, embeddings, embed_query=lambda q: np.array([0, 1.0, 0, 0]))


def test_bm25_finds_exact_numbers():
    results = make_searcher().search("130,497", k=1, mode="bm25")
    assert results[0]["page"] == 1 and results[0]["company"] == "NVIDIA"


def test_search_only_looks_at_the_named_company():
    results = make_searcher().search("Qualcomm revenue", k=2, mode="hybrid")
    assert {r["company"] for r in results} == {"Qualcomm"}


def test_comparison_questions_get_results_from_every_company():
    results = make_searcher().search("Compare NVIDIA and Qualcomm revenue", k=2, mode="hybrid")
    assert {r["company"] for r in results} == {"NVIDIA", "Qualcomm"}
