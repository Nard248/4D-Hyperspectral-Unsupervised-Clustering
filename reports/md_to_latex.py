"""Minimal Markdown -> LaTeX converter for the SpectraForge paper. One Markdown source -> identical
.docx (md_to_docx) and .tex (this). Handles: ATX headings, paragraphs, **bold**, `code`, bullet/
numbered lists, pipe tables (booktabs), images (figure+includegraphics), blockquotes, rules.

Usage:  python reports/md_to_latex.py <input.md> <output.tex>
"""
from __future__ import annotations

import re
import sys

_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")
_IMG = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\usepackage{float}
\definecolor{codecol}{RGB}{176,48,96}
\newcommand{\code}[1]{\texttt{\textcolor{codecol}{#1}}}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\begin{document}
"""


def esc(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def inline(text):
    out = []
    for tok in _INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            out.append(r"\textbf{" + esc(tok[2:-2]) + "}")
        elif tok.startswith("`") and tok.endswith("`"):
            out.append(r"\code{" + esc(tok[1:-1]) + "}")
        else:
            out.append(esc(tok))
    return "".join(out)


def is_row(l):
    return l.strip().startswith("|") and l.strip().endswith("|")


def cells(l):
    return [c.strip() for c in l.strip().strip("|").split("|")]


def convert(md, tex):
    lines = open(md, encoding="utf-8").read().splitlines()
    out = [PREAMBLE]
    i, in_list = 0, None
    HEAD = {1: "title", 2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph"}

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{" + in_list + "}")
            in_list = None

    while i < len(lines):
        line = lines[i]; s = line.strip()
        if not s:
            close_list(); i += 1; continue
        m = _IMG.match(s)
        if m:
            close_list()
            out.append(r"\begin{figure}[H]\centering")
            out.append(r"\includegraphics[width=\linewidth]{" + m.group(2) + "}")
            if m.group(1):
                out.append(r"\caption{" + esc(m.group(1)) + "}")
            out.append(r"\end{figure}")
            i += 1; continue
        if re.fullmatch(r"-{3,}|\*{3,}", s):
            close_list(); out.append(r"\noindent\hrulefill"); i += 1; continue
        h = re.match(r"(#{1,6})\s+(.*)", s)
        if h:
            close_list(); lvl = len(h.group(1)); txt = inline(h.group(2))
            if lvl == 1:
                out.append(r"\begin{center}{\LARGE\bfseries " + txt + r"}\end{center}\vspace{1em}")
            else:
                out.append("\\%s*{%s}" % (HEAD.get(lvl, "paragraph"), txt))
            i += 1; continue
        if is_row(line) and i + 1 < len(lines) and re.search(r"-{2,}", lines[i + 1]):
            close_list()
            hdr = cells(line); n = len(hdr); body = []
            i += 2
            while i < len(lines) and is_row(lines[i]):
                body.append(cells(lines[i])); i += 1
            out.append(r"\begin{center}\small\begin{tabular}{" + "l" * n + "}\\toprule")
            out.append(" & ".join(inline(c) for c in hdr) + r" \\\midrule")
            for row in body:
                row = (row + [""] * n)[:n]
                out.append(" & ".join(inline(c) for c in row) + r" \\")
            out.append(r"\bottomrule\end{tabular}\end{center}")
            continue
        if s.startswith(">"):
            close_list(); out.append(r"\begin{quote}" + inline(s.lstrip("> ")) + r"\end{quote}"); i += 1; continue
        if re.match(r"[-*]\s+", s):
            if in_list != "itemize":
                close_list(); out.append(r"\begin{itemize}"); in_list = "itemize"
            out.append(r"\item " + inline(re.sub(r"^[-*]\s+", "", s))); i += 1; continue
        if re.match(r"\d+\.\s+", s):
            if in_list != "enumerate":
                close_list(); out.append(r"\begin{enumerate}"); in_list = "enumerate"
            out.append(r"\item " + inline(re.sub(r"^\d+\.\s+", "", s))); i += 1; continue
        close_list(); out.append(inline(s)); i += 1
    close_list()
    out.append(r"\end{document}")
    open(tex, "w", encoding="utf-8").write("\n".join(out))
    print("wrote", tex)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
