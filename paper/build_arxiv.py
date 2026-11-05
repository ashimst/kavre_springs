"""
Assemble the arXiv submission: paper/arxiv/ and paper/arxiv_submission.zip, containing manuscript.tex,
manuscript.bbl (written by make_bbl.py; arXiv does not run BibTeX), supplement_text.tex, si_tables.tex and every
figure referenced by the LaTeX sources, flattened into figures/.

Usage (from the repository root):  python -m src.report && python paper/build_arxiv.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUT = PAPER / "arxiv"
SOURCES = ["manuscript.tex", "supplement_text.tex", "si_tables.tex"]


def main():
    subprocess.run([sys.executable, str(PAPER / "make_bbl.py")], check=True)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    tex_all = ""
    for f in SOURCES + ["manuscript.bbl"]:
        shutil.copy(PAPER / f, OUT / f)
        tex_all += (PAPER / f).read_text(encoding="utf-8")
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_all)))
    missing = []
    for f in figs:
        # freshly generated figures take precedence; paper/figures is refreshed so local builds match arXiv
        src = next((p for p in [ROOT / "results/figures" / f, PAPER / "figures" / f] if p.exists()), None)
        if src is None:
            missing.append(f)
            continue
        if src.parent != PAPER / "figures":
            shutil.copy(src, PAPER / "figures" / Path(f).name)
        shutil.copy(src, OUT / "figures" / Path(f).name)
    if missing:
        raise SystemExit(f"missing figures: {missing}")
    # arXiv: no line numbers, no local graphicspath outside the package
    m = (OUT / "manuscript.tex").read_text(encoding="utf-8")
    m = m.replace("\\linenumbers\n", "").replace("{figures/}{../results/figures/}{./}", "{figures/}{./}")
    (OUT / "manuscript.tex").write_text(m, encoding="utf-8")
    zpath = PAPER / "arxiv_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    size = zpath.stat().st_size / 1e6
    print(f"wrote {zpath.relative_to(ROOT)} ({len(figs)} figures, {size:.1f} MB)")


if __name__ == "__main__":
    main()
