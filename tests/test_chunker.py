from rag.chunker import chunk_pages, find_section, searchable_text, split_lines


def test_find_section():
    assert find_section("Item 1A. Risk Factors") == "Item 1A. Risk Factors"
    assert find_section("CONSOLIDATED BALANCE SHEETS") == "Consolidated Balance Sheets"
    assert find_section("Refer to the Consolidated Balance Sheets for details.") is None
    assert find_section("Revenue grew 114%") is None


def test_split_lines_overlaps_neighbouring_pieces():
    lines = [f"line {i} has five words" for i in range(10)]  # 5 words each
    pieces = split_lines(lines, max_words=20, overlap=1)
    assert pieces[0] == lines[:4]
    assert pieces[1][0] == lines[3]  # last line of piece 1 repeats as first of piece 2
    assert pieces[-1][-1] == lines[-1]  # nothing is lost


def test_chunks_keep_page_and_section():
    pages = [
        {"page": 1, "lines": ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. MD&A", "Item 8. Financials"]},
        {"page": 2, "lines": ["Item 1A. Risk Factors", "Export rules may hurt sales."]},
        {"page": 3, "lines": ["More risks continue on this page."]},
    ]
    chunks = chunk_pages(pages, "NVIDIA")

    assert chunks[0]["section"] == "Front matter"  # page 1 is a table of contents
    assert [c["page"] for c in chunks] == [1, 2, 3]
    assert chunks[2]["section"] == "Item 1A. Risk Factors"  # carried over from page 2
    assert searchable_text(chunks[2]).startswith("NVIDIA | Item 1A. Risk Factors")
