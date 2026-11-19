"""Build the Word and PDF versions of EXPERIMENTS.md, with the figures in place.

Needs pandoc, python-docx and LibreOffice (soffice) on the PATH.

    python docs/make_figures.py
    python docs/build_document.py

    python docs/make_figures.py --lang fr
    python docs/build_document.py --lang fr      # French version, from EXPERIMENTS_fr.md
"""

import os
import re
import subprocess
import sys
import tempfile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = {
    "en": dict(source="EXPERIMENTS.md", name="Design and experiment log", contents="Contents",
               title="Deepfake video detection with motion-compensated optical flow",
               subtitle="Design and experiment log", lang="en-GB"),
    "fr": dict(source="EXPERIMENTS_fr.md", name="Journal de conception et d'expériences",
               contents="Sommaire",
               title="Détection de vidéos deepfake par flot optique compensé du mouvement de la tête",
               subtitle="Conception et journal d'expériences", lang="fr-FR"),
}
L = LANGS["fr" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "fr" else "en"]
NAME, TITLE = L["name"], L["title"]
TOTAL = Cm(17)


def markdown_for_word(path):
    s = open(path, encoding="utf-8").read()
    a = s.index(L["contents"] + "\n")
    s = s[:a] + s[s.index("---", a) + 3:]                       # Word gets its own contents list
    s = re.sub(r"!\[([^\]]*)\]\((docs/figures/[^)]+)\)", r"![\1](\2){width=16cm}", s)
    s = re.sub(r"\n---\n", "\n", s)                              # sections start on a new page
    rest = s.split("\n", 1)[1]
    heads = [line[3:] for line in rest.splitlines() if line.startswith("## ")]
    i = rest.index("\n## 1.")
    rest = rest[:i] + "\n\n**" + L["contents"] + "**\n\n" + "\n".join(f"| {h}" for h in heads) + "\n\n" + rest[i:]
    out = []
    for line in rest.split("\n"):                                # pandoc needs a blank line before a list
        if (re.match(r"^(- |\d+\. )", line) and out and out[-1].strip()
                and not re.match(r"^(- |\d+\. |  |\|)", out[-1])):
            out.append("")
        out.append(line)
    meta = '---\ntitle: "%s"\nsubtitle: "%s"\n---\n' % (TITLE, L["subtitle"])
    return meta + "\n".join(out)


def style(src, dst):
    d = Document(src)
    styles = {x.name: x for x in d.styles}
    for sec in d.sections:
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2)

    def font(name, family="Calibri", size=None, color=None, bold=None):
        if name not in styles:
            return
        st = styles[name]
        st.font.name = family
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(k), family)
        for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rf.attrib.pop(qn(k), None)
        if size:
            st.font.size = Pt(size)
        if color:
            st.font.color.rgb = RGBColor.from_string(color)
        if bold is not None:
            st.font.bold = bold

    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        font(name, size=10.5)
    lang = OxmlElement("w:lang")                                 # spelling and hyphenation language
    lang.set(qn("w:val"), L["lang"])
    styles["Normal"].element.get_or_add_rPr().append(lang)
    font("Title", size=24, color="1F3864", bold=True)
    font("Subtitle", size=14, color="2F6DB5")
    font("Heading 1", size=16, color="1F3864", bold=True)
    font("Heading 2", size=13, color="2F6DB5", bold=True)
    font("Heading 3", size=11.5, color="2F6DB5", bold=True)
    font("Source Code", "Consolas", 8.5)
    font("Verbatim Char", "Consolas", 8.5)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        pf = styles[name].paragraph_format
        pf.keep_with_next = True
        pf.space_before, pf.space_after = Pt(14 if name == "Heading 1" else 10), Pt(6)
    first = True
    for p in d.paragraphs:
        if p.style.name == "Heading 1":
            p.paragraph_format.page_break_before = not first
            first = False
        if p.style.name in ("Image Caption", "Caption"):
            for r in p.runs:
                r.font.size, r.font.italic = Pt(9), True
    for t in d.tables:
        table_style(t)
    d.save(dst)


def table_style(t):
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:color", "B0B7C3")):
            e.set(qn(k), v)
        borders.append(e)
    tblPr.append(borders)
    # column widths: proportional to the text, but the longest word of a column fits on one line
    ncol = len(t.columns)
    weights, mins = [], []
    for j in range(ncol):
        texts = [r.cells[j].text for r in t.rows]
        lens = [len(x) for x in texts]
        word = max((len(w) for x in texts for w in x.split()), default=4)
        weights.append(max(4.0, min(0.5 * max(lens) + 0.5 * sum(lens) / len(lens), 90)))
        mins.append(Cm(min(word * 0.2 + 0.4, 6.0)))
    fixed = [False] * ncol
    while True:
        free = TOTAL - sum(m for m, f in zip(mins, fixed) if f)
        wsum = sum(w for w, f in zip(weights, fixed) if not f)
        widths = [m if f else int(free * w / wsum) for w, m, f in zip(weights, mins, fixed)]
        bad = [j for j in range(ncol) if not fixed[j] and widths[j] < mins[j]]
        if not bad:
            break
        for j in bad:
            fixed[j] = True
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    tw = tblPr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW")
        tblPr.append(tw)
    tw.set(qn("w:type"), "dxa")
    tw.set(qn("w:w"), str(int(TOTAL / 635)))
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w / 635)))
    for i, row in enumerate(t.rows):
        for cell, w in zip(row.cells, widths):
            cell.width = Emu(w)
            if i == 0:
                shd = OxmlElement("w:shd")
                for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", "E8EEF7")):
                    shd.set(qn(k), v)
                cell._tc.get_or_add_tcPr().append(shd)
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(1)
                for r in para.runs:
                    r.font.size = Pt(8.5)
                    if i == 0:
                        r.font.bold = True


def main():
    out_dir = os.path.join(ROOT, "docs")
    with tempfile.TemporaryDirectory() as tmp:
        md = os.path.join(tmp, "doc.md")
        raw = os.path.join(tmp, "raw.docx")
        with open(md, "w", encoding="utf-8") as f:
            f.write(markdown_for_word(os.path.join(ROOT, L["source"])))
        subprocess.run(["pandoc", md, "-f", "markdown+pipe_tables+line_blocks", "-t", "docx",
                        "--shift-heading-level-by=-1", "-o", raw], cwd=ROOT, check=True)
        docx_path = os.path.join(out_dir, NAME + ".docx")
        style(raw, docx_path)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, docx_path],
                   check=True, stdout=subprocess.DEVNULL)
    print("wrote", docx_path, "and the PDF next to it")


if __name__ == "__main__":
    sys.exit(main())
