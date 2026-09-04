"""Render the three method equations with LaTeX (pdflatex + pdftoppm) as crisp PNGs.
Run from the repo root."""
import subprocess, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "figures"; TMP = ROOT / "data" / "eq_tmp"
TMP.mkdir(parents=True, exist_ok=True)
EQS = {
    "eq1": r"z \coloneqq \frac{1}{G}\sum_{g=1}^{G} E_g(X_g), \qquad \hat{X}_g \coloneqq D_g(z), \qquad \mathcal{L} \coloneqq \sum_{g=1}^{G}\lVert X_g - D_g(z)\rVert_2^2",
    "eq2": r"I_{g,c} \coloneqq \sum_{j=1}^{m}\sum_{\delta\in\mathcal{D}} w_j\,\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T}\Big|\big[D_g(z+\delta e_j)-D_g(z)\big]_{n,t,c}\Big|",
    "eq3": r"c^\star = \arg\max_{(g,c)\notin S}\Big[\tilde{I}_{g,c} - \lambda \max_{s\in S}\,\mathrm{sim}\big((g,c),s\big)\Big], \qquad \tilde{I}_{g,c} \coloneqq \frac{I_{g,c}}{\max_{c'} I_{g,c'}}",
}
TEX = r"""\documentclass[preview,border=3pt,12pt,varwidth=60cm]{standalone}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{lmodern}
\begin{document}
\Large $\displaystyle %s$
\end{document}"""
for name, eq in EQS.items():
    tex = TMP / f"{name}.tex"; tex.write_text(TEX % eq)
    subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory", str(TMP), str(tex)],
                   check=True, capture_output=True)
    subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile", str(TMP / f"{name}.pdf"), str(OUT / name)], check=True)
    print("wrote", OUT / f"{name}.png")
