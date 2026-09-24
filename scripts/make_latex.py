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
grey is the empty cell. \\texttt{{recent}} holds $1.00$ out to $d{{=}}B$ and
scores nothing beyond it; \\texttt{{h2o}} is the mirror image anchored to
the start of the stream. Neither reaches the ${{\\approx}}4000$ tokens between
them.}}
\\label{{fig:horizon}}
\\end{{figure}}"""

(OUT / "figure.tex").write_text(figure)
print("wrote paper/figure.tex")
for pol in ("recent", "random", "h2o"):
    print(f"  {pol:8s}", " ".join("--" if v is None else f"{v:.2f}" for v in bins(pol)))


# ---- tables and inline numbers: every k/n in the paper comes from here ----
import math, statistics
from collections import Counter
from scipy.stats import fisher_exact

from kvadmission.stats import mean_ci, wilson


def load(name):
    return json.load(open(ROOT / f"results/{name}.json"))


def band(r):
    """Thirds of the stream by distance back from the query. The one band
    definition every table uses."""
    f = r["distance_back"] / r["stream_len"]
    return "near" if f < 1 / 3 else ("mid" if f < 2 / 3 else "far")


def tally(rows, **match):
    c = Counter()
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            for b in (band(r), "all"):
                c[b, "n"] += 1
                c[b, "k"] += bool(r["hit"])
    return c


def kn(c, b="all"):
    return c[b, "k"], c[b, "n"]


def frac(k, n, bold=False):
    body = f"\\mathbf{{{k}/{n}}}" if bold else f"{k}/{n}"
    return f"${body}$"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{{\\scriptsize[{lo:.2f}, {hi:.2f}]}}"


macros = {}
def macro(name, value):
    macros[name] = value


# oracle vs heuristics on the same needle streams, both scales
phase1, oracle17 = load("phase1"), load("oracle")
sweep8, oracle8 = load("scale8b-sweep"), load("scale8b-oracle")
rows = [("\\texttt{recent}", "1.7B", tally(phase1, policy="recent", budget=2048), False),
        ("\\texttt{h2o}", "1.7B", tally(phase1, policy="h2o", budget=2048), False),
        ("\\textbf{oracle}", "1.7B", tally(oracle17, budget=2048), True),
        None,
        ("\\texttt{recent}", "8B", tally(sweep8, policy="recent", budget=2048), False),
        ("\\textbf{oracle}", "8B", tally(oracle8, budget=2048), True)]
body = []
for row in rows:
    if row is None:
        body.append("\\midrule")
        continue
    name, model, c, bold = row
    cells = [frac(*kn(c, b), bold=bold) for b in ("near", "mid", "far")]
    k, n = kn(c)
    body.append(f"{name} & {model} & " + " & ".join(cells)
                + f" & {frac(k, n, bold)} & {ci(k, n)} \\\\")
tab_oracle = ("\\begin{tabular}{llccccl}\n\\toprule\n"
              "policy, $B{=}2048$ & model & near & middle & far & total & 95\\% CI \\\\\n"
              "\\midrule\n" + "\n".join(body)
              + "\n\\bottomrule\n\\end{tabular}")
for tag, c in (("OracleSmall", tally(oracle17, budget=2048)),
               ("OracleLarge", tally(oracle8, budget=2048)),
               ("RecentSmall", tally(phase1, policy="recent", budget=2048)),
               ("RecentLarge", tally(sweep8, policy="recent", budget=2048))):
    macro(tag, "{}/{}".format(*kn(c)))

# coverage: degraded oracles, end to end and through the screen, side by side
po, spo = load("partial-oracle"), load("screen-partial-oracle")
screen_of = {(r["coverage"], r["fp_rate"]): r for r in spo}
body = []
for cov in (1.0, 0.75, 0.5, 0.25):
    c = tally(po, coverage=cov, fp_rate=0.0)
    sc = screen_of[cov, 0.0]
    k, n = kn(c)
    passes = sc["spans_whole"] == sc["n_spans"]
    body.append(f"{cov:.2f} & {frac(sc['spans_whole'], sc['n_spans'], passes)} & "
                f"{frac(k, n)} {ci(k, n)} & {frac(*kn(c, 'mid'), bold=cov >= 0.75)} \\\\")
tab_coverage = ("\\begin{tabular}{cccc}\n\\toprule\n"
                "coverage & screen: spans whole & end-to-end recall [95\\% CI] & middle band \\\\\n"
                "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")
macro("CovMidFull", "{}/{}".format(*kn(tally(po, coverage=1.0, fp_rate=0.0), "mid")))
macro("CovMidThreeQ", "{}/{}".format(*kn(tally(po, coverage=0.75, fp_rate=0.0), "mid")))

# candidate screen, answer-phrase spans on QA
screen = load("screen3")
FRAMING = {"surprisal": "information content", "norm_attention": "attention received",
           "key_norm": "activation magnitude", "rarity": "lexical statistics",
           "answer_shape": "answer form", "lookahead": "128 tokens of future",
           "random": "---"}
ORDER = ["surprisal", "norm_attention", "key_norm", "rarity", "answer_shape",
         "lookahead", "random"]
by = {(r["task"], r["signal"]): r for r in screen}
body = ["learned admission $\\times3$ & learned features & --- & --- & $0/12$ mid & \\\\",
        "query-aware retention & query similarity & --- & --- & $0/12$ mid & \\\\"]
best = max(by[t, s]["mean_coverage"] for t in ("needle",) for s in ORDER if s != "random")
for s in ORDER:
    ne, qa = by["needle", s], by["qa", s]
    name = "\\emph{random control}" if s == "random" else f"\\texttt{{{s.replace('_', '{\\_}')}}}"
    covs = [f"{r['mean_coverage']:.3f}" for r in (ne, qa)]
    if ne["mean_coverage"] == best:
        covs[0] = f"\\textbf{{{covs[0]}}}"
    body.append(f"{name} & {FRAMING[s]} & {covs[0]} & {covs[1]} & "
                f"${ne['spans_whole']}/{ne['n_spans']}$ & ${qa['spans_whole']}/{qa['n_spans']}$ \\\\")
tab_screen = ("\\begin{tabular}{llcccc}\n\\toprule\n"
              " & & \\multicolumn{2}{c}{coverage} & \\multicolumn{2}{c}{spans whole} \\\\\n"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
              "signal & framing & needle & QA & needle & QA \\\\\n\\midrule\n"
              + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")
n_sig = [s for s in ORDER if s != "random"]
macro("ScreenSpansNeedle", str(by["needle", "surprisal"]["n_spans"]))
macro("ScreenSpansQA", str(by["qa", "surprisal"]["n_spans"]))
macro("RandomWholeQA", "{spans_whole}/{n_spans}".format(**by["qa", "random"]))
macro("AnswerShapeWholeQA", "{spans_whole}/{n_spans}".format(**by["qa", "answer_shape"]))
macro("AnswerShapeWholeNeedle", "{spans_whole}/{n_spans}".format(**by["needle", "answer_shape"]))
macro("ScreenBestWhole", "{}/{}".format(
    *max(((by[t, s]["spans_whole"], by[t, s]["n_spans"]) for t in ("needle", "qa") for s in n_sig))))
for s in ("surprisal", "lookahead", "answer_shape", "norm_attention"):
    tag = s.replace("_", "").capitalize()
    macro(f"{tag}Needle", f"{by['needle', s]['mean_coverage']:.3f}")
    macro(f"{tag}QA", f"{by['qa', s]['mean_coverage']:.3f}")
macro("AnswerShapeRatio", f"{by['needle', 'answer_shape']['mean_coverage'] / 0.25:.1f}")

# fragmentation: fixed evidence, varying number of runs
frag = {m: load(f"fragcount-{m}") for m in ("smollm2", "qwen")}
body = []
for runs in (8, 4, 2, 1):
    s_ = tally(frag["smollm2"], runs=runs)
    q_ = tally(frag["qwen"], runs=runs)
    k, n = kn(s_)
    b = runs in (8, 4)
    body.append(f"{'\\textbf{%d}' % runs if b else runs} & {1600 // runs} & "
                f"{'\\textbf{%.2f}' % (k / n) if b else '%.2f' % (k / n)} {ci(k, n)} & "
                f"{kn(q_)[0] / kn(q_)[1]:.2f} \\\\")
tab_frag = ("\\begin{tabular}{cccc}\n\\toprule\n"
            "runs & window & SmolLM2-1.7B [95\\% CI] & Qwen3-1.7B \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")
k8, n8 = kn(tally(frag["smollm2"], runs=8))
k4, n4 = kn(tally(frag["smollm2"], runs=4))
macro("FragP", f"{fisher_exact([[k8, n8 - k8], [k4, n4 - k4]])[1]:.4f}")

# horizon at p = 0.5
hz = load("horizon-p05")
inside = [r for r in hz if r["distance_back"] < 4096]
beyond = [r for r in hz if r["distance_back"] >= 4096]
macro("HorizonInside", f"{sum(r['hit'] for r in inside)}/{len(inside)}")
macro("HorizonBeyond", f"{sum(r['hit'] for r in beyond)}/{len(beyond)}")
macro("HorizonBeyondUpper", f"{wilson(sum(r['hit'] for r in beyond), len(beyond))[1]:.2f}")
d1 = [r for r in load("gpu-distance-curve") if r["policy"] == "recent"]
far1 = [r for r in d1 if r["distance_back"] >= 2048]
macro("RecentBeyond", f"{sum(r['hit'] for r in far1)}/{len(far1)}")
macro("RecentBeyondUpper", f"{wilson(sum(r['hit'] for r in far1), len(far1))[1]:.2f}")

# crowding transfer on real QA
bench = load("gpu-benchmark")
o512 = tally(bench, policy="oracle", budget=512)
full = tally(bench, policy="full")
ceiling = kn(full)[0] / kn(full)[1]
macro("CrowdPred", f"{512 / 1332:.2f}")
macro("CrowdPredScaled", f"{512 / 1332 * ceiling:.2f}")
macro("CrowdObs", "{}/{}".format(*kn(o512)))
lo, hi = wilson(*kn(o512))
macro("CrowdObsCI", f"[{lo:.2f}, {hi:.2f}]")
macro("QACeiling", f"{ceiling:.2f}")



# ---- crowding figure: the law, predicted against observed ----
def flagged_est(fp, length=8192):
    return int(fp * length + 96)   # the estimate run_partial_oracle.py records

crowd = load("partial-oracle-crowding")
pts = []
for fp in sorted({r["fp_rate"] for r in po if r["coverage"] == 1.0}
                 | {r["fp_rate"] for r in crowd}):
    src = crowd if any(r["fp_rate"] == fp for r in crowd) else po
    k, n = kn(tally(src, coverage=1.0, fp_rate=fp))
    pts.append((flagged_est(fp) / 2048, k, n))
XS, YS = 3.6, 3.2          # cm per unit of |flagged|/B, and per unit of recall
XMAX = 2.8
curve = " -- ".join(f"({x * XS:.3f},{min(1, 1 / x) * YS:.3f})"
                    for x in [i / 20 for i in range(1, int(XMAX * 20) + 1)])
marks = []
for x, k, n in pts:
    lo, hi = wilson(k, n)
    marks.append(f"  \\draw[tealaccent,thick] ({x * XS:.3f},{lo * YS:.3f}) -- ({x * XS:.3f},{hi * YS:.3f});\n"
                 f"  \\fill[tealaccent] ({x * XS:.3f},{k / n * YS:.3f}) circle (1.6pt);")
qx = 1332 / 512
qk, qn = kn(o512)
qlo, qhi = wilson(qk, qn)
fig2 = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}[x=1cm,y=1cm]
  \\draw[black!40] (0,0) -- ({XMAX * XS:.2f},0);
  \\draw[black!40] (0,0) -- (0,{YS * 1.05:.2f});
  \\foreach \\y in {{0,0.25,0.5,0.75,1}}
    \\node[anchor=east,font=\\scriptsize] at (0,{{\\y*{YS}}}) {{\\y}};
  \\foreach \\x in {{0,0.5,1,1.5,2,2.5}}
    \\node[anchor=north,font=\\scriptsize] at ({{\\x*{XS}}},0) {{\\x}};
  \\draw[black!25,dashed] ({XS:.2f},0) -- ({XS:.2f},{YS * 1.05:.2f});
  \\draw[black!70,thick] (0.05,{YS:.3f}) -- {curve};
  \\node[anchor=west,font=\\scriptsize] at ({2.05 * XS:.2f},{0.92 * YS:.2f}) {{$\\min(1, B/|\\mathrm{{flagged}}|)$}};
{chr(10).join(marks)}
  \\draw[rustaccent,thick] ({qx * XS:.3f},{qlo * YS:.3f}) -- ({qx * XS:.3f},{qhi * YS:.3f});
  \\fill[rustaccent] ({qx * XS - 0.07:.3f},{qk / qn * YS - 0.07:.3f}) rectangle ++(0.14,0.14);
  \\node[anchor=north,font=\\scriptsize] at ({XMAX * XS / 2:.2f},-0.4) {{flagged set / budget, $|\\mathrm{{flagged}}|/B$}};
  \\node[anchor=south,rotate=90,font=\\scriptsize] at (-0.7,{YS / 2:.2f}) {{recall}};
\\end{{tikzpicture}}
\\caption{{\\textbf{{Precision is free until the budget binds.}} Recall of the
coverage-1.00 oracle as false positives grow the flagged set (circles: synthetic
needles, $B{{=}}2048$, 24 queries each, 95\\% Wilson intervals). The curve is
Eq.~\\ref{{eq:crowding}}, not fitted. The square is real SQuAD QA at $B{{=}}512$,
where the flagged set is 1332 paragraph tokens; its uncompressed ceiling is
${{\\QACeiling}}$, so the law predicts ${{\\CrowdPredScaled}}$ to
${{\\CrowdPred}}$ and ${{\\CrowdObs}}$ is observed.}}
\\label{{fig:crowding}}
\\end{{figure}}"""
(OUT / "figure-crowding.tex").write_text(fig2)
print("wrote paper/figure-crowding.tex")
for x, k, n in pts:
    print(f"  flagged/B={x:.2f}  {k}/{n}  predicted {min(1, 1 / x):.2f}")


# ---- screen figure: every candidate against chance and against the bar ----
XW, ROW = 9.0, 0.42        # cm for coverage 0..1, cm per signal row
lines = []
for i, s in enumerate(ORDER):
    y = -i * ROW
    name = s.replace("_", "\\_")
    lines.append(f"  \\node[anchor=east,font=\\scriptsize\\ttfamily] at (-0.1,{y:.2f}) {{{name}}};")
    for task, colour, dy in (("needle", "tealaccent", 0.07), ("qa", "rustaccent", -0.07)):
        m, lo, hi = mean_ci(by[task, s]["per_span"])
        lines.append(f"  \\draw[{colour}] ({lo * XW:.3f},{y + dy:.2f}) -- ({hi * XW:.3f},{y + dy:.2f});")
        lines.append(f"  \\fill[{colour}] ({m * XW:.3f},{y + dy:.2f}) circle (1.5pt);")
bottom = -(len(ORDER) - 1) * ROW - 0.3
fig3 = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}[x=1cm,y=1cm]
  \\fill[black!6] ({0.95 * XW:.2f},{bottom:.2f}) rectangle ({XW:.2f},0.3);
  \\draw[black!40,dashed] ({0.25 * XW:.2f},{bottom:.2f}) -- ({0.25 * XW:.2f},0.3)
    node[anchor=south,font=\\scriptsize] {{chance}};
  \\node[anchor=south,font=\\scriptsize] at ({0.975 * XW:.2f},0.3) {{bar}};
{chr(10).join(lines)}
  \\draw[black!40] (0,{bottom:.2f}) -- ({XW:.2f},{bottom:.2f});
  \\foreach \\x in {{0,0.25,0.5,0.75,1}}
    \\node[anchor=north,font=\\scriptsize] at ({{\\x*{XW}}},{bottom:.2f}) {{\\x}};
  \\node[anchor=north,font=\\scriptsize] at ({XW / 2:.2f},{bottom - 0.35:.2f})
    {{coverage of answer tokens by the signal's top $B$}};
\\end{{tikzpicture}}
\\caption{{\\textbf{{No candidate approaches the bar.}} Mean coverage over answer
spans at $B{{=}}2048$ with 95\\% intervals, on synthetic needles
(\\textcolor{{tealaccent}}{{upper}}) and real QA answer phrases
(\\textcolor{{rustaccent}}{{lower}}); {by["needle", "surprisal"]["n_spans"]} and
{by["qa", "surprisal"]["n_spans"]} spans. Chance is $B/L = 0.25$; the shaded band is where the specification
requires a signal to be.}}
\\label{{fig:screen}}
\\end{{figure}}"""
(OUT / "figure-screen.tex").write_text(fig3)
print("wrote paper/figure-screen.tex")


# ---- the pooled dead zone: every compressed middle-band needle trial ----
DEAD = ["phase1", "gpu-smollm2-sweep", "scale8b-sweep",            # four heuristics
        "phase2-eval", "phase2b-eval", "phase-2c-b1-eval"]         # three learned
mid = [r for f in DEAD for r in load(f)
       if r["policy"] not in ("full", "oracle") and r["budget"] < r["stream_len"]
       and band(r) == "mid"]
k = sum(bool(r["hit"]) for r in mid)
macro("DeadK", str(k))
macro("DeadN", str(len(mid)))
macro("DeadUpper", f"{100 * wilson(k, len(mid))[1]:.1f}\\%")
qa_mid = [r for r in bench if r["policy"] in ("recent", "h2o", "snapkv", "random")
          and r["budget"] < r["stream_len"] and band(r) == "mid"]
qk = sum(bool(r["hit"]) for r in qa_mid)
macro("DeadQA", f"{qk}/{len(qa_mid)}")
macro("DeadQAUpper", f"{100 * wilson(qk, len(qa_mid))[1]:.1f}\\%")
macro("DeadQAFull", "{}/{}".format(*kn(tally(bench, policy="full"), "mid")))

# ---- quantised storage: the horizon law says only tokens-kept matters ----
# results/graded.json. Equal bits per layer, so int2 holds ~5.3x the tokens.
graded = load("graded")
for cfg, name in (("uniform0", "Fp"), ("uniform2", "IntFour"),
                  ("uniform3", "IntTwo"), ("kl", "Kl")):
    for b, bname in ((512, "Small"), (1024, "Large")):
        c = tally(graded, config=cfg, budget=b)
        if not c["all", "n"]:
            continue
        macro(f"Graded{name}{bname}", frac(*kn(c)))
        macro(f"Graded{name}{bname}CI", ci(*kn(c)))
        macro(f"Graded{name}{bname}Mid", frac(*kn(c, "mid")))
        res = [r["resident"] for r in graded
               if r["config"] == cfg and r["budget"] == b]
        macro(f"Graded{name}{bname}Tokens", f"{res[0]}")

# ---- the supervised probe: the first realisable signal to pass the screen ----
# results/probe-scorer.json (screen), probe-e2e*.json (end to end),
# probe-gen-*.json (does it survive a change of answer style or of model?)
probe_screen = {(r["train"], r["test"]): r for r in load("probe-scorer")}
for (tr, te), r in probe_screen.items():
    name = f"Probe{tr.capitalize()}{te.capitalize()}"
    macro(name, f"{r['spans_whole']}/{r['n_spans']}")
    macro(name + "CI", ci(r["spans_whole"], r["n_spans"]))
    macro(name + "Rand", f"{r['random_whole']}/{r['n_spans']}")

for tag, rows_ in (("", load("probe-e2e")), ("QA", load("probe-e2e-qa")),
                  ("Style", load("probe-gen-style")),
                  ("Smol", load("probe-gen-model"))):
    for pol, pname in (("probe", "Probe"), ("recent", "Recent"),
                       ("probe-int2", "ProbeIntTwo")):
        for b, bname in ((1024, "Small"), (2048, "Large")):
            c = tally(rows_, policy=pol, budget=b)
            if not c["all", "n"]:
                continue
            macro(f"{pname}E{tag}{bname}", frac(*kn(c)))
            macro(f"{pname}E{tag}{bname}CI", ci(*kn(c)))
            macro(f"{pname}E{tag}{bname}Mid", frac(*kn(c, "mid")))

# the probe table: screen verdict beside what happened end to end
probe_rows = [
    ("needle, in-task", "ProbeNeedleNeedle", "ProbeESmall", "ProbeELarge", True),
    ("needle, phrase answers", None, "ProbeEStyleSmall", None, False),
    ("needle, SmolLM2-1.7B", None, "ProbeESmolSmall", None, False),
    ("QA, in-task", "ProbeQaQa", "ProbeEQASmall", "ProbeEQALarge", False),
    ("needle, trained on QA", "ProbeQaNeedle", None, None, False),
]
tab = ["\\begin{tabular}{lcccc}", "\\toprule",
       "condition & screen (spans whole) & \\multicolumn{2}{c}{end to end} "
       "& \\texttt{recent} \\\\",
       "& & $B{=}1024$ & $B{=}2048$ & $B{=}1024$ \\\\", "\\midrule"]
for label, screen, e_small, e_large, bold in probe_rows:
    cells = [label,
             f"\\{screen}" if screen else "---",
             (f"\\textbf{{\\{e_small}}}" if bold and e_small else
              f"\\{e_small}" if e_small else "---"),
             f"\\{e_large}" if e_large else "---"]
    rec = {"ProbeESmall": "RecentESmall", "ProbeEStyleSmall": "RecentEStyleSmall",
           "ProbeESmolSmall": "RecentESmolSmall", "ProbeEQASmall": "RecentEQASmall"}
    cells.append(f"\\{rec[e_small]}" if e_small in rec else "---")
    tab.append(" & ".join(cells) + " \\\\")
tab += ["\\bottomrule", "\\end{tabular}"]
tab_probe = "\n".join(tab)

# ---- write tables.tex: every macro and table above ----
out = ["% Generated by scripts/make_latex.py from results/*.json. Do not edit."]
out += [f"\\newcommand{{\\{k_}}}{{{v}}}" for k_, v in macros.items()]
for name, t in (("TabOracle", tab_oracle), ("TabCoverage", tab_coverage),
                ("TabScreen", tab_screen), ("TabFrag", tab_frag),
                ("TabProbe", tab_probe)):
    out.append(f"\\newcommand{{\\{name}}}{{%\n{t}}}")
(OUT / "tables.tex").write_text("\n".join(out) + "\n")
print("wrote paper/tables.tex")
for k_, v in macros.items():
    print(f"  {k_:20s} {v}")
