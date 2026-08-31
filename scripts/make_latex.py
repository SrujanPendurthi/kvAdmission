"""Generate the LaTeX submission, with the figure drawn from real results.

The figure is emitted as TikZ rather than an included image so the paper is a
single self-contained source tree: no SVG-to-PDF conversion step, nothing to
regenerate by hand when the numbers change.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)

# ---- figure data, straight from the distance-curve run ----
d = json.load(open(ROOT / "results/gpu-distance-curve.json"))
W, N, STREAM = 512, 16, 8192
def bins(pol):
    b = defaultdict(list)
    for r in d:
        if r["policy"] == pol:
            b[min(int(r["distance_back"] // W), N - 1)].append(r["hit"])
    return [(sum(v) / len(v) if v else None) for v in (b.get(i, []) for i in range(N))]

def tikz_row(pol, colour, y):
    out = []
    for i, v in enumerate(bins(pol)):
        x = i * (10.0 / N)
        w = 10.0 / N - 0.04
        out.append(f"    \\fill[black!8] ({x:.3f},{y:.2f}) rectangle ++({w:.3f},1.0);")
        if v:
            out.append(f"    \\fill[{colour}] ({x:.3f},{y:.2f}) rectangle ++({w:.3f},{v:.3f});")
    return "\n".join(out)

hx = (2048 / STREAM) * 10.0
figure = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}[x=1cm,y=1cm]
% recent
{tikz_row("recent", "tealaccent", 2.6)}
  \\node[anchor=east,font=\\scriptsize\\ttfamily] at (-0.15,3.1) {{recent}};
% random
{tikz_row("random", "tealaccent", 1.3)}
  \\node[anchor=east,font=\\scriptsize\\ttfamily] at (-0.15,1.8) {{random}};
% h2o
{tikz_row("h2o", "rustaccent", 0.0)}
  \\node[anchor=east,font=\\scriptsize\\ttfamily] at (-0.15,0.5) {{h2o}};
% horizon
  \\draw[rustaccent,dashed,thick] ({hx:.3f},-0.15) -- ({hx:.3f},3.75);
  \\node[anchor=south west,font=\\scriptsize,rustaccent] at ({hx:.3f},3.75)
    {{horizon $d=B=2048$}};
% axis
  \\draw[black!40] (0,-0.15) -- (10,-0.15);
  \\foreach \\f/\\l in {{0/0, 0.25/2048, 0.5/4096, 0.75/6144, 1.0/8192}}
    \\node[anchor=north,font=\\scriptsize] at ({{\\f*10}},-0.2) {{\\l}};
  \\node[anchor=north,font=\\scriptsize] at (5,-0.62)
    {{distance back from query (tokens)}};
\\end{{tikzpicture}}
\\caption{{\\textbf{{The retention horizon.}} Recall in 512-token distance bins at
$B{{=}}2048$ (Qwen3-1.7B, 4 seeds $\\times$ 16 needles); bar height is recall,
grey is the empty cell. \\texttt{{recent}} holds $1.00$ out to $d{{=}}B$ and is
\\emph{{exactly}} zero beyond it; \\texttt{{h2o}} is the mirror image anchored to
the start of the stream. Neither reaches the ${{\\approx}}4000$ tokens between
them.}}
\\label{{fig:horizon}}
\\end{{figure}}"""

(OUT / "figure.tex").write_text(figure)
print("wrote paper/figure.tex")
for pol in ("recent", "random", "h2o"):
    print(f"  {pol:8s}", " ".join("--" if v is None else f"{v:.2f}" for v in bins(pol)))
