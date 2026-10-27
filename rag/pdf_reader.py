"""Step 1: turn a PDF into clean lines of text, page by page.

Why not just use page.get_text()? In annual reports, plain text extraction breaks
tables apart and puts every cell on its own line:

    Revenue
    $
    130,497
    $
    60,922

The row is gone, so a search for "revenue 130,497" finds nothing useful. Instead we
take every word with its position on the page, then group words that sit at the
same height back into one line:

    Revenue $ 130,497 $ 60,922 Up 114%
"""

import re
from collections import Counter

import pymupdf

SAME_LINE_TOLERANCE = 3  # words whose bottoms are within 3 points count as one line


def group_words_into_lines(words: list[tuple]) -> list[str]:
    """Group PyMuPDF words (x0, y0, x1, y1, text, ...) into lines, top to bottom."""
    words = sorted(words, key=lambda w: (round(w[3]), w[0]))

    rows = []  # each row: [bottom_y, [words]]
    for word in words:
        bottom = word[3]
        if rows and abs(rows[-1][0] - bottom) <= SAME_LINE_TOLERANCE:
            rows[-1][1].append(word)
        else:
            rows.append([bottom, [word]])

    lines = []
    for _, row_words in rows:
        row_words.sort(key=lambda w: w[0])  # left to right
        lines.append(" ".join(w[4] for w in row_words))
    return lines


def remove_repeated_lines(pages: list[dict], min_share: float = 0.5) -> list[dict]:
    """Drop headers and footers: lines that appear on more than half of the pages.

    Digits are ignored when comparing, so 'arm-20250331.htm 51/351' on page 51 and
    'arm-20250331.htm 52/351' on page 52 count as the same footer.
    """
    if len(pages) < 3:  # too few pages to tell a footer from real content
        return pages

    def pattern(line):
        return re.sub(r"\d", "#", line)

    counts = Counter(p for page in pages for p in {pattern(line) for line in page["lines"]})
    repeated = {p for p, n in counts.items() if n > min_share * len(pages)}
    return [{"page": page["page"], "lines": [l for l in page["lines"] if pattern(l) not in repeated]}
            for page in pages]


def read_pdf(path, rebuild_lines: bool = True) -> list[dict]:
    """Return [{"page": 1, "lines": [...]}, ...] with page numbers as shown in a PDF viewer.

    `rebuild_lines=False` uses plain get_text() instead. It only exists so the
    evaluation can measure how much the line rebuilding helps.
    """
    pages = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            if rebuild_lines:
                lines = group_words_into_lines(page.get_text("words"))
            else:
                lines = page.get_text().splitlines()
            lines = [line.strip() for line in lines if line.strip()]
            pages.append({"page": index + 1, "lines": lines})
    return remove_repeated_lines(pages)
