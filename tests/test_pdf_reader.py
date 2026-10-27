import pymupdf

from rag.pdf_reader import group_words_into_lines, read_pdf, remove_repeated_lines


def word(x, y, text):
    """A fake PyMuPDF word: (x0, y0, x1, y1, text)."""
    return (x, y - 10, x + 30, y, text)


def test_words_at_same_height_become_one_line():
    words = [word(200, 100, "130,497"), word(10, 100, "Revenue"), word(150, 101, "$"), word(10, 120, "Next")]
    assert group_words_into_lines(words) == ["Revenue $ 130,497", "Next"]


def test_read_pdf_rebuilds_table_rows(tmp_path):
    # Make a tiny PDF where a table row is written as separate pieces of text.
    path = tmp_path / "table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Revenue")
    page.insert_text((200, 100), "$")
    page.insert_text((220, 100), "130,497")
    doc.save(path)

    pages = read_pdf(path)
    assert pages == [{"page": 1, "lines": ["Revenue $ 130,497"]}]


def test_headers_and_footers_are_removed():
    content = ["Revenue grew", "Risks increased", "Cash fell"]
    pages = [{"page": n, "lines": ["Table of Contents", content[n - 1], f"report.htm {n}/3"]}
             for n in (1, 2, 3)]
    cleaned = remove_repeated_lines(pages)
    assert [p["lines"] for p in cleaned] == [[c] for c in content]
