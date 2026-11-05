"""
Write manuscript.bbl (natbib author-year, plainnat-like) directly from references.bib and the citations used in
manuscript.tex, so that the arXiv package compiles without running BibTeX (arXiv requires the .bbl).

Usage (from the repository root):  python paper/make_bbl.py
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def parse_bib(text: str) -> dict:
    entries, i = {}, 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").search(text, i)
        if not m:
            break
        depth, j = 1, m.end()
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body, fields, k = text[m.end(): j - 1], {}, 0
        while True:
            f = re.compile(r"\s*(\w+)\s*=\s*").match(body, k)
            if not f:
                break
            k = f.end()
            if body[k] == "{":
                d, start = 1, k + 1
                k += 1
                while d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[start: k - 1]
            elif body[k] == '"':
                end = body.index('"', k + 1)
                val, k = body[k + 1: end], end + 1
            else:
                end = re.compile(r"[,\n]").search(body, k).start()
                val, k = body[k:end].strip(), end
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", val).strip()
            c = re.compile(r"\s*,").match(body, k)
            if c:
                k = c.end()
        entries[m.group(2)] = {"type": m.group(1).lower(), **fields}
        i = j
    return entries


def split_authors(a: str):
    out = []
    for name in re.split(r"\s+and\s+", a):
        name = name.strip()
        if name.startswith("{") and name.endswith("}"):
            out.append((name, name))  # corporate author
        elif name.lower() == "others":
            out.append(("others", "others"))
        elif "," in name:
            last, first = [x.strip() for x in name.split(",", 1)]
            out.append((last, f"{first} {last}"))
        else:
            parts = name.split()
            out.append((parts[-1], name))
    return out


def label(auth, year):
    last = [a[0] for a in auth if a[0] != "others"]
    many = len(auth) > 2 or any(a[0] == "others" for a in auth)
    short = last[0] + (" et~al." if many else (f" and {last[1]}" if len(last) == 2 else ""))
    long_ = ", ".join(last[:-1]) + f", and {last[-1]}" if len(last) > 2 else short
    return short, long_


def fmt_authors(auth):
    names = [a[1] for a in auth if a[0] != "others"]
    s = names[0] if len(names) == 1 else (" and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + ", and " + names[-1])
    return s + (" et~al." if any(a[0] == "others" for a in auth) else "")


def entry_text(e):
    t, parts = e["type"], []
    title = e.get("title", "")
    year = e.get("year", "")
    if t == "article":
        venue = f"\\emph{{{e.get('journal', '')}}}"
        vol = e.get("volume", "")
        num = f"\\penalty0 ({e['number']})" if e.get("number") else ""
        pages = f":\\penalty0 {e['pages']}" if e.get("pages") else ""
        parts = [f"{title}.", f"{venue}, {vol}{num}{pages}, {year}."]
    elif t in ("book",):
        series = f"volume {e['volume']} of \\emph{{{e['series']}}}. " if e.get("series") else ""
        parts = [f"\\emph{{{title}}}.", f"{series}{e.get('publisher', '')}, {e.get('address', '') + ', ' if e.get('address') else ''}{year}."]
    elif t in ("inproceedings", "incollection"):
        parts = [f"{title}.", f"In \\emph{{{e.get('booktitle', '')}}}, {('pages ' + e['pages'] + ', ') if e.get('pages') else ''}{year}."]
    elif t == "techreport":
        parts = [f"{title}.", f"Technical report, {e.get('institution', '')}, {year}."]
    else:  # misc
        hp = e.get("howpublished", "")
        parts = [f"{title}.", ", ".join(x for x in [hp, e.get("note", ""), year] if x) + "."]
    if e.get("doi"):
        parts.append(f"\\doi{{{e['doi']}}}.")
    elif e.get("url"):
        parts.append(f"\\url{{{e['url']}}}.")
    return "\n\\newblock ".join(parts)


def main():
    tex = "\n".join((PAPER / f).read_text(encoding="utf-8") for f in ["manuscript.tex", "supplement_text.tex"]
                    if (PAPER / f).exists())
    cited = []
    for m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            if k.strip() not in cited:
                cited.append(k.strip())
    bib = parse_bib((PAPER / "references.bib").read_text(encoding="utf-8"))
    missing = [k for k in cited if k not in bib]
    if missing:
        raise SystemExit(f"citations missing from references.bib: {missing}")
    items = []
    for k in cited:
        e = bib[k]
        auth = split_authors(e.get("author", "{Anonymous}"))
        items.append((auth[0][0].strip("{}").lower(), e.get("year", ""), k, e, auth))
    items.sort(key=lambda r: (r[0], r[1], r[2]))
    # author-year disambiguation (a, b, ...)
    seen = {}
    for it in items:
        seen.setdefault((label(it[4], it[1])[0], it[1]), []).append(it[2])
    lines = [f"\\begin{{thebibliography}}{{{len(items)}}}",
             "\\providecommand{\\natexlab}[1]{#1}",
             "\\providecommand{\\url}[1]{\\texttt{#1}}",
             "\\expandafter\\ifx\\csname urlstyle\\endcsname\\relax",
             "  \\providecommand{\\doi}[1]{doi: #1}\\else",
             "  \\providecommand{\\doi}{doi: \\begingroup \\urlstyle{rm}\\Url}\\fi", ""]
    for _, year, key, e, auth in items:
        short, long_ = label(auth, year)
        same = seen[(short, year)]
        suffix = f"\\natexlab{{{'abcdefgh'[same.index(key)]}}}" if len(same) > 1 else ""
        lines += [f"\\bibitem[{short}({year}{suffix}){long_}]{{{key}}}", f"{fmt_authors(auth)}.",
                  f"\\newblock {entry_text(e)}", ""]
    lines.append("\\end{thebibliography}")
    (PAPER / "manuscript.bbl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote manuscript.bbl with {len(items)} entries")


if __name__ == "__main__":
    main()
