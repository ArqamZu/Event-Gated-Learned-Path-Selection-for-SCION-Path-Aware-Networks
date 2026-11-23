"""Converts report/Phase3_Report.docx to PDF (docx -> HTML with mammoth, then headless Chrome print).

Usage: python make_pdf.py <input.docx> <output.pdf>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import mammoth

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: Calibri, Carlito, 'Liberation Sans', Arial, sans-serif; font-size: 10.3pt; line-height: 1.38;
       color: #111; background: #fff; }
h1 { font-size: 15pt; color: #1F3A5F; margin: 18pt 0 6pt; border-bottom: 1px solid #c9d3e0; padding-bottom: 2pt;
     break-after: avoid; }
h2 { font-size: 12.3pt; color: #1F3A5F; margin: 12pt 0 4pt; break-after: avoid; }
h3 { font-size: 11pt; color: #1F3A5F; break-after: avoid; }
p { margin: 0 0 5pt; text-align: justify; }
body > p:nth-of-type(1) { text-align: center; font-size: 17pt; margin-top: 4pt; }
body > p:nth-of-type(2) { text-align: center; font-size: 12pt; font-style: italic; }
body > p:nth-of-type(3) { text-align: center; margin-bottom: 14pt; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2.5pt; text-align: justify; }
p:has(img) { text-align: center; margin: 8pt 0 2pt; break-inside: avoid; break-after: avoid; }
img { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; margin: 2pt 0 10pt; font-size: 8pt; line-height: 1.25; break-inside: auto; }
tr { break-inside: avoid; }
table tr:first-child { break-after: avoid; }
p:has(+ table) { break-after: avoid; }
td { border: 0.6pt solid #9fb0c6; padding: 2.5pt 4pt; vertical-align: top; }
td p { margin: 0; text-align: left; }
table tr:first-child td { background: #DCE6F2; font-weight: bold; }
table tr:not(:first-child):has(strong) td { background: #EEF4FB; }
"""


def main(src, dst):
    with open(src, "rb") as fh:
        html = mammoth.convert_to_html(fh).value
    # mammoth drops picture sizes: restore each figure's width from the docx (text width = 16.6 cm)
    from docx import Document
    widths = [s.width.cm for s in Document(src).inline_shapes]
    parts = html.split("<img ")
    html = parts[0] + "".join(f'<img style="width:{min(100, w / 16.6 * 100):.0f}%" ' + p
                              for w, p in zip(widths, parts[1:]))
    page = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        h = Path(tmp) / "report.html"
        h.write_text(page, encoding="utf-8")
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}/profile", f"--print-to-pdf={dst}", h.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
