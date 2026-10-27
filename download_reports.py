"""Download the annual reports listed in config.py into data/pdfs/.

Run:  python download_reports.py
"""

import urllib.request

from config import PDF_DIR, REPORTS

# Some investor sites reject requests that don't look like a browser.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for report in REPORTS:
        path = PDF_DIR / report["file"]
        if path.exists():
            print(f"[ok]      {report['file']} already downloaded")
        elif report["url"] is None:
            print(f"[manual]  {report['file']} - save it from your browser (see README, step 2)")
        else:
            print(f"[download] {report['title']} ...")
            request = urllib.request.Request(report["url"], headers=HEADERS)
            with urllib.request.urlopen(request) as response:
                path.write_bytes(response.read())


if __name__ == "__main__":
    main()
