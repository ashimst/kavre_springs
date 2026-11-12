"""
Make paper/ a self-contained, arXiv-ready source folder.

1. Copy the figures cited in the manuscript from results/figures/ to paper/figures/.
2. Generate paper/manuscript.bbl with the manuscript's elsarticle-harv BibTeX style.
   arXiv does not run BibTeX, so the .bbl must be supplied.

Upload the contents of paper/ to arXiv or Overleaf as they are (Overleaf may also run BibTeX itself).
Usage (from the repository root, after `python -m src.report`):  python -m src.build_paper
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = ROOT / "results" / "figures"
def refresh_figures(tex: str) -> int:
    """Copy every figure the manuscript includes from results/figures to paper/figures."""
    (PAPER / "figures").mkdir(exist_ok=True)
    names = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    for name in names:
        src = FIGURES / Path(name).name
        if not src.exists():
            raise SystemExit(f"figure {name} not found in {FIGURES}; run `python -m src.figures` first")
        shutil.copy(src, PAPER / "figures" / src.name)
    return len(names)


def main():
    tex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    print(f"copied {refresh_figures(tex)} figures to paper/figures")
    cited = []
    for m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            if k.strip() not in cited:
                cited.append(k.strip())
    bib_text = (PAPER / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))
    missing = [k for k in cited if k not in bib_keys]
    if missing:
        raise SystemExit(f"citations missing from references.bib: {missing}")
    bibtex = shutil.which("bibtex")
    if not bibtex:
        raise SystemExit("bibtex executable not found; install a TeX distribution and retry")
    with tempfile.TemporaryDirectory(prefix="kavre-bibtex-") as temp_dir:
        temp = Path(temp_dir)
        shutil.copyfile(PAPER / "references.bib", temp / "references.bib")
        aux = temp / "manuscript.aux"
        aux.write_text(
            "\\relax\n"
            f"\\citation{{{','.join(cited)}}}\n"
            "\\bibstyle{elsarticle-harv}\n"
            "\\bibdata{references}\n",
            encoding="utf-8",
        )
        subprocess.run([bibtex, "manuscript"], cwd=temp, check=True)
        generated_bbl = temp / "manuscript.bbl"
        if not generated_bbl.exists():
            raise SystemExit("bibtex completed without generating manuscript.bbl")
        shutil.copyfile(generated_bbl, PAPER / "manuscript.bbl")
    print(f"wrote manuscript.bbl with {len(cited)} citations using elsarticle-harv")


if __name__ == "__main__":
    main()
