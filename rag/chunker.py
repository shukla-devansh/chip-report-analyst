"""Step 2: cut pages into small chunks that the search can find.

Rules:
- A chunk never crosses a page, so every answer can cite one exact page.
- A chunk is about CHUNK_WORDS words. Neighbouring chunks share a few lines
  (overlap) so a sentence cut at the boundary still appears whole in one of them.
- Each chunk remembers which report section ("Item 1A. Risk Factors") it came from.
"""

import re

from config import CHUNK_WORDS, OVERLAP_LINES

# 10-K and 20-F filings are split into "Items", e.g. "Item 7. Management's Discussion..."
ITEM_HEADING = re.compile(r"^Item\s+\d+[A-Z]?\.\s*\S")
# The financial statements come with their own headings, e.g. "CONSOLIDATED BALANCE SHEETS".
STATEMENT_HEADING = re.compile(
    r"(consolidated (balance sheets?|statements? of [\w’' ]+)|notes to (the )?consolidated financial statements)",
    re.IGNORECASE,
)
# After the main report come signatures and legal exhibits (stock plans, subsidiary lists...).
EXHIBIT_HEADING = re.compile(r"(signatures|exhibit \d+(\.\d+)?)", re.IGNORECASE)


def find_section(line: str) -> str | None:
    """Return a clean section name if the line is a heading, else None.

    'Item 1A. Risk Factors'        -> 'Item 1A. Risk Factors'
    'CONSOLIDATED BALANCE SHEETS'  -> 'Consolidated Balance Sheets'
    'Refer to the Consolidated...' -> None (a sentence, not a heading)
    """
    line = line.replace("\xa0", " ").strip()
    line = re.sub(r"\s+\d+$", "", line)  # drop a page number: "Item 19. Exhibits 199"
    if ITEM_HEADING.match(line) and not line.endswith("."):  # a heading, not a sentence
        return line[:80]
    if STATEMENT_HEADING.fullmatch(line) or EXHIBIT_HEADING.fullmatch(line):
        return line.title()
    return None


def split_lines(lines: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Group lines into pieces of about max_words words, repeating `overlap` lines between pieces."""
    pieces, current, words = [], [], 0
    for line in lines:
        current.append(line)
        words += len(line.split())
        if words >= max_words:
            pieces.append(current)
            current = current[-overlap:] if overlap else []
            words = sum(len(l.split()) for l in current)
    # Keep the leftover lines, unless they are only the overlap we already used.
    if current and (not pieces or current != pieces[-1][-len(current):]):
        pieces.append(current)
    return pieces


def chunk_pages(pages: list[dict], company: str) -> list[dict]:
    """Turn the output of read_pdf() into chunks with company, page and section."""
    chunks = []
    section = "Front matter"
    for page in pages:
        headings = [h for h in map(find_section, page["lines"]) if h]
        # The table of contents lists every Item; it should not change the section.
        is_table_of_contents = len(headings) >= 4
        if headings and not is_table_of_contents:
            section = headings[0]

        for piece in split_lines(page["lines"], CHUNK_WORDS, OVERLAP_LINES):
            chunks.append({
                "id": f"{company}-p{page['page']}-{len(chunks)}",
                "company": company,
                "page": page["page"],
                "section": section,
                "text": "\n".join(piece),
            })
        if headings and not is_table_of_contents:
            section = headings[-1]  # the next page continues the last section started here
    return chunks


def searchable_text(chunk: dict) -> str:
    """Text used for searching. Adding the company and section to the chunk means a
    question like "NVIDIA risk factors" matches even if the chunk itself never says "NVIDIA"."""
    return f"{chunk['company']} | {chunk['section']}\n{chunk['text']}"
