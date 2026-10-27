from rag.answer import answer_question, check_numbers, cited_numbers

CHUNKS = [
    {"company": "NVIDIA", "page": 38, "section": "Item 7", "text": "Revenue $ 130,497 $ 60,922\nGross margin 75.0 %"},
    {"company": "Qualcomm", "page": 59, "section": "Item 8", "text": "Total revenues 44,284 38,962"},
]


class FakeSearcher:
    def search(self, question):
        return CHUNKS


def test_cited_numbers():
    assert cited_numbers("Revenue grew [1] and margin rose [1, 2].") == {1, 2}
    assert cited_numbers("No citations here.") == set()


def test_numbers_are_checked_against_cited_sources():
    result = check_numbers("Revenue was $130,497 million [1] with a 75% margin [1].", CHUNKS)
    assert result == {"verified": [130497.0, 75.0], "unverified": []}


def test_made_up_numbers_are_flagged():
    result = check_numbers("Revenue was $999 million [1].", CHUNKS)
    assert result["unverified"] == [999.0]


def test_a_number_from_an_uncited_source_is_flagged():
    # 44,284 is in source [2], but the answer only cites [1].
    assert check_numbers("Qualcomm made 44,284 [1].", CHUNKS)["unverified"] == [44284.0]


def test_answer_question_uses_retrieved_sources():
    seen_prompts = []

    def fake_llm(prompt):
        seen_prompts.append(prompt)
        return "NVIDIA revenue was $130,497 million [1]."

    result = answer_question("What was NVIDIA's revenue?", FakeSearcher(), llm=fake_llm)

    assert "[1] NVIDIA, page 38" in seen_prompts[0]
    assert result["cited"] == [1]
    assert result["numbers"]["unverified"] == []
