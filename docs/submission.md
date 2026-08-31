# The Mid-Context Dead Zone in KV Cache Compression

**Submission draft — 2026-08-31.** Findings logs for every run, including the
errors, are in `docs/phase*.md`. Corrections to earlier drafts of this paper
are collected in Appendix A rather than left inline.

---

## Abstract

KV cache compression is evaluated on aggregate accuracy at a given budget,
which hides *where* in the context a method succeeds. Measuring recall against
distance from the query, we find that the token-scoring rules published
methods are built on partition the context by position rather than utility:
recency scoring recovers only the near field, accumulated-attention scoring
only the far field, and **no rule we tested recovers the middle at any
budget** — 0 successes in 340 middle-band trials across three models spanning
a 4.7× size range (95% upper bound 0.9%), against an uncompressed cache that
answers all of them.

We give the mechanism. Admission at rate `p` followed by FIFO eviction to
budget `B` leaves a contiguous window over the most recent `B/p` positions, so
retention is a **step function with a hard horizon**: beyond it, survival is
exactly zero rather than merely unlikely. Recency is therefore emergent rather
than chosen — any rule that does not actively protect distant tokens decays
onto it.

An **oracle retention** control — identical budget, chunking and eviction, but
told which tokens carry the answers — scores 40/40 at 1.7B and 24/24 at 8B,
matching the uncompressed cache on a quarter of the context. **The dead zone
is not a capacity limit. It is an observability limit**, and we give the
specification a realisable signal would have to meet: essentially total
coverage of answer tokens, with a flagged set inside the budget. We then use
that specification as a *screening test* and reject **eight candidate
signals** — learned admission under three reward formulations, query-aware
retention, and five write-time scoring functions including information
content and a de-biased variant of accumulated attention. One was rejected
for $0 with its end-to-end failure correctly predicted in advance from five
pre-registered predictions.

Finally we find a second constraint no prior work in this line varies: at
fixed evidence, distance and budget, **splitting the retained set into more
pieces destroys recall** on one of two models — 0.54 at eight fragments
against 0.92 at four (p=0.008). Every published rule here selects tokens
individually, which is the maximally fragmented case.

## 1. Introduction

A long-context model writes every token into its KV cache; under a budget most
must go. The literature decides what to discard by ranking tokens with
attention-derived rules — accumulated attention (H2O), observation-window
voting (SnapKV), recency plus attention sinks (StreamingLLM), layer-wise
budget allocation (PyramidKV, Ada-KV) — and reports aggregate accuracy at a
compression ratio.

Those methods differ in engineering, but the component deciding *what to keep*
is in each case a token-scoring rule, and that is what we isolate and test.
The aggregate metric cannot distinguish a rule that retains information
uniformly from one that retains a biased subset well. We ask a narrower
question: **given a fact planted at distance `d` before the query, does the
compressed cache still support answering it?**

**Contributions.**

1. A **distance-resolved measurement** of KV compression, showing that
   published scoring rules partition context by position and that a
   mid-context dead zone survives every rule, budget, model and task we tried
   (§3).
2. A **mechanism** — a hard retention horizon at `d = B/p` — that predicts
   `recent` exactly and explains why recency is emergent (§4).
3. An **oracle control** establishing that the dead zone is observability, not
   capacity, at two model scales (§5).
4. A **specification** for what a realisable signal must achieve, and its use
   as a cheap **screening test**; eight candidate signals rejected, one with
   its end-to-end failure predicted in advance from a $0 measurement (§6, §7).
5. **Fragmentation** as an independent constraint on retained-set structure
   that no prior work in this line varies (§8).

## 2. Setup and scope

- **Models.** Qwen3-1.7B (primary), SmolLM2-1.7B-Instruct and Qwen3-8B, bf16,
  frozen throughout. No model weights are trained anywhere in this work.
- **Streams.** *Synthetic:* 8192 tokens of prose with eight facts of the form
  `The secret code for <topic> is <6 digits>` at evenly spaced, per-seed
  jittered anchors, each queried from the end. *Real:* 8192 tokens of
  distractor SQuAD paragraphs with eight gold paragraphs planted at the same
  anchors, queried by their human-written questions.
- **Compression.** Chunked prefill at 128 tokens; after each chunk the cache is
  compressed to budget `B` per layer, first four tokens pinned as sinks.
  Budgets 256/512/1024/2048.
- **Metrics.** Exact substring match on the synthetic task; containment plus
  token F1 on real QA, against a measured closed-book floor. Zero-hit cells
  report a rule-of-three upper bound rather than a standard error.
- **Banding.** Near/middle/far terciles of distance-back from the query.

**What the baselines are, precisely.** `recent`, `h2o` and `snapkv` implement
the *token-scoring rule* of StreamingLLM, H2O and SnapKV — recency plus sinks,
accumulated attention mass, observation-window voting — at token granularity
under a uniform per-layer budget. They are **not** reproductions of those
systems: no per-head budgets, no layer-wise allocation, none of the kernel and
memory engineering those papers contribute. Absolute numbers are not
comparable to published ones and nothing here is a benchmark of any published
system. What is under test is the scoring rule, because that is the component
the dead zone is a property of.

**Harness correctness.** At unbounded budget, logits are bit-identical to a
stock `DynamicCache` under the same chunking, verified on MPS and CUDA.
Positions are always passed explicitly, since compression makes cache length
diverge from stream position — a silent corruption otherwise. 129 tests.

## 3. The dead zone

Per-rule recall at B=2048, a quarter of the stream, on Qwen3-1.7B:

| rule | near | middle | far |
|---|---|---|---|
| `recent` | **10/14** | 0/12 | **0/14** |
| `random` | 7/14 | 0/12 | 0/14 |
| `snapkv` | 4/14 | 0/12 | 0/14 |
| `h2o` | **0/14** | 0/12 | **8/14** |
| `full` | 1.00 | 1.00 | 1.00 |

`recent` and `h2o` are exact complements: each scores strongly in one band and
**exactly zero** in the other, and neither reaches the middle. This follows
from the rules themselves. Accumulated attention sums over query rows, and the
earliest tokens are visible to the most of them, so it leans toward the start
by construction, as recency leans toward the end. **Both are functions of
position; neither is a function of utility.**

**Across models.** The same sweep on SmolLM2-1.7B gives 0/56 in the middle
band against an uncompressed 7/7. On Qwen3-8B — same family as the primary
model, so scale is the only variable — 0/56 against a ceiling that is perfect
in all three bands, with the complementarity intact (`recent` 6/9 near and 0/8
far; `h2o` 0/9 near and 4/8 far). Pooled: **0 of 340 middle-band trials**,
95% upper bound 0.9%.

**On real QA.** Replacing the synthetic task with planted-evidence SQuAD —
human-written questions, natural-language answers, containment plus F1 scoring
against a measured **closed-book floor of 0/24** — gives 1/64 in the middle
band (95% upper 7.2%) against an uncompressed 6/8. The single hit is `h2o` at
d=4729, gold `rivers`, generated `certain rivers`; the closed-book control
confirms it is retrieval rather than a guess. Distractors are drawn only from
articles no gold paragraph belongs to, since SQuAD groups paragraphs by topic
and a same-article distractor could restate the answer.

Two caveats specific to that task: its far band has a difficulty ceiling
(`full` scores 4/7), so far-band zeros there are weak evidence; and it
discriminates only on Qwen3-1.7B — on SmolLM2 the uncompressed ceiling is
11/24 against a 3/24 floor, an eight-trial window in which nothing separates.

**Why this does not rest on baseline fidelity.** Three parts of the argument
are independent of whether our rules match the published systems. The ceiling
is *measured*, not assumed: `full` uses the same harness and answers
everything. The oracle (§5) is measured at the *same budget* through the
*same* eviction path and answers everything, so a 2048-token cache built by
this harness can reach the middle. And the mechanism (§4) is geometric,
holding exactly for `recent`, whose rule has no implementation freedom worth
disputing. A faithful reimplementation could move the near- and far-band
numbers; it could not move the middle without replacing the scoring rule.

## 4. Mechanism: a hard horizon

Sixteen needles per stream, four seeds, B=2048, chunk=128, binned at
256-token resolution:

| rule | d < 2048 | d ≥ 2048 |
|---|---|---|
| `recent` | **16/16** | **0/48** |
| `random` | 13/16 | **0/48** |
| `h2o` | 0/16 | 10/48 — all at d > 6600 |

Bin by bin, `recent` is 1.00 below 2048 and 0.00 above. Under admission at rate
`p` followed by FIFO eviction to budget `B`, occupancy saturates and the cache
becomes a contiguous window over the most recent `B/p` raw positions:

> **P(token resident at distance `d`) = `p` if `d < B/p`, else 0**

`recent` (`p`=1) is predicted exactly. Tested at `p`=0.5, where the horizon
should move to 4096: **2/32 inside, 0/32 beyond**. Retention tracks `B/p`.

The claim this sharpens is not that distant tokens are unlikely to survive but
that **beyond the horizon they survive with probability exactly zero**, so any
rule that does not explicitly protect a token past `B/p` loses it with
certainty. The middle band lies entirely beyond the horizon at every budget
tested. Compression is not a ranking problem but a rescue problem.

`h2o` has a horizon at the opposite end — its hits are a ~1300-token window
over the *start* of the stream — because accumulated attention concentrates
there. Two rules, two windows, neither covering the middle.

## 5. Capacity exists: the oracle

**Oracle retention** uses the identical budget, chunking, sink prefix, FIFO
eviction and evaluation, changing exactly one thing: the policy is told which
token spans carry the answers. It runs through the same `apply_eviction` path
as every baseline.

| | near | middle | far |
|---|---|---|---|
| best rule (B=2048) | 0.71 | **0.00** | 0.57 |
| best learned policy | 0.71 | **0.00** | 0.00 |
| **oracle, B=2048** | **1.00** | **1.00** | **1.00** |
| `full` | 1.00 | 1.00 | 1.00 |

**40/40 at 1.7B and 24/24 at 8B**, matching the uncompressed cache on a
quarter of the context. Nothing about the budget, the chunking or FIFO
eviction prevents mid-context recall. **The dead zone is not intrinsic to
fixed-budget streaming compression.**

What this makes of every failure in §7: those policies were asked to predict,
from write-time features alone, something the write-time input does not
determine. A token's utility here is fixed by a query that has not arrived.
Recency is the optimal policy under that information constraint — when nothing
distinguishes which distant token will matter, keeping the nearest maximises
expected utility. The horizon of §4 is not a wall; it is the default any
*uninformed* rule collapses onto.

## 6. What a realisable signal must achieve

Degrading the oracle along two axes measures the specification. `coverage` is
a proxy's recall over answer tokens; `fp_rate` its false-positive rate.

| coverage | overall | middle band |
|---|---|---|
| 1.00 | 24/24 | **7/7** |
| 0.75 | 12/24 | **1/7** |
| 0.50 | 8/24 | 0/7 |
| 0.25 | 6/24 | 0/7 |

**Coverage is unforgiving.** Missing a quarter of the answer tokens collapses
the middle band from 7/7 to 1/7. An answer spans few tokens and retaining most
of a six-digit code is worth nothing.

**Precision is free until the budget binds**, and then follows a second law:

> **recall = min( 1, B / |flagged| )**

Accurate to 0.05 across the range, and it **transfers to a task it was not
fitted on**: on real QA the eight gold paragraphs occupy 1332 tokens, and at
B=512 the predicted survival fraction of 0.38 matches the observed 0.38
exactly.

**The specification is two-sided:** a signal must identify *essentially every*
token of an answer span and keep its *total* flagged set within budget. It may
be entirely indiscriminate about which extra tokens it keeps, so long as it
does not keep too many.

**This is a screening test, not just a description.** Any candidate signal can
be rejected before an end-to-end run by measuring what fraction of answer
spans falls inside its top `B` and comparing against the curve above. §7 uses
it that way.

## 7. Four candidate signals, screened and rejected

**7.1 Learned admission, three reward formulations.** A 297K-parameter head
over `[hidden state, ‖k‖, relative position, occupancy, λ]`, trained by
average-reward policy gradient against per-position KL with a Lagrangian
occupancy constraint. Plain chunk-mean KL converged to behaviour
indistinguishable from `recent` — long-dependency positions are ~1.5% of a
stream and carry ~15% of KL mass, so 85% of the gradient rewards local
prediction, which recency already maximises. Divergence-weighting the reward
fixed the mass problem and destabilised: it matched the first run's KL by step
150, then collapsed to admitting nothing. A mixed objective reproduced
recency. Middle-band recall: 0/12 in all three.

**7.2 Query-aware admission.** Tokens scored by maximum cosine similarity to a
pooled embedding of the *pending queries* — already a semi-oracle, since a
streaming policy would not hold them. **0/12 in the middle band at both
budgets**, and it forfeits the near band recency gets for free, inverting to a
far-band bias. It sits between the heuristics and the oracle in information
and lands on the heuristic side: **what the oracle holds is not "what is being
asked" but "which tokens answer it,"** and embedding similarity does not
recover the second from the first.

**7.3 Surprisal.** Every rule above is a function of position. Surprisal is
not: score each token by `−log p(token | prefix)`, keeping what the context
cannot reconstruct. It is genuinely write-time (the logits already exist
during prefill), and top-`B` satisfies half the §6 specification by
construction.

The screen rejected it for **$0**:

| task | coverage @ B=2048 | chance | answer-token percentile |
|---|---|---|---|
| needle | **0.107** | 0.250 | 0.410 |
| real QA | **0.248** | 0.250 | 0.484 |

On needles it selects *against* the answer; on real QA it is exactly
indifferent. The needle case has an arithmetic explanation: a uniform digit is
`−log(1/10)` = **2.30 nats**, the information-theoretic *ceiling* for a digit,
while unexpected English words run 6–9 nats out of a 150k vocabulary. Measured
corpus median is 2.79. Surprisal would keep the topic word (`' jun'`, 9.02) and
discard the digits — exactly backwards.

Five predictions were **pre-registered and committed before the GPU was
rented**. All five held: needle middle 0/7, needle near 0/9 against `recent`'s
10/14, needle overall 0/24, QA overall 2/24 with middle 0/8, and beating
`recent` on neither. Surprisal is the first policy tested here that is worse
than every heuristic in every band — with no recency term it has no positional
fallback, which cleanly separates *selection* from *geometry*.

**7.4 A wider screen.** Five more candidates, screened together against a
random control. Coverage at B=2048; chance is 0.250.

| signal | needle | spans whole | real QA | spans whole |
|---|---|---|---|---|
| `answer_shape` — digits, capitalised words | **0.541** | 0/16 | 0.271 | 0/16 |
| `key_norm` — ‖k‖ | 0.384 | 0/16 | 0.244 | 0/16 |
| `norm_attention` — attention per query row | 0.230 | 1/16 | 0.290 | 2/16 |
| `rarity` — inverse in-document frequency | 0.102 | 0/16 | 0.265 | 0/16 |
| `surprisal` (§7.3) | 0.107 | 0/16 | 0.240 | 0/16 |
| *random control* | 0.248 | 0/16 | 0.249 | 0/16 |

**None passes.** The best whole-span retention across twelve signal-task
pairs is 2/16, against the ~1.00 the specification requires.

Two results here are worth more than the table. First, **normalised attention
fails**, and it was the most promising repair available: §3 diagnoses H2O's
far-band bias as an artifact of *summing* attention over a variable number of
query rows, so dividing by that count is the obvious fix. It lands *below
chance* on needles (0.230) and barely above on real QA (0.290). Removing the
positional bias does not reveal a utility signal underneath — it removes the
bias that was giving H2O its only wins.

Second, **`answer_shape` is the strongest signal found anywhere in this work
and still fails.** At 0.541 on needles it is 2.2× chance, and it should be:
the rule flags digits, and the answers *are* digits. It is a prior written
with knowledge of the answer format. Yet it retains **0/16 spans whole**,
because keeping about half the digits of a six-digit code is worth exactly
nothing under exact match.

That is the cleanest statement of §6's coverage requirement we have.
**Knowing the *form* of the answer is not enough; the specification demands
the specific tokens.** A signal can be twice as good as chance at finding
answer-shaped tokens and still recover nothing.

On real QA every signal collapses to within 0.04 of the random control. None
of the six discriminates at all.

**The pattern across all eight.** Learned write-time features, query
similarity, information content, attention received, key norm, in-document
rarity and answer-form priors all fail. The framings are independent —
information-theoretic, attention-based, lexical, learned — and they fail in
the same place, which is the case for reading this as a property of the
problem rather than of any one attempt: **a token's future utility is not a
function of anything observable when it is written.**

The closest miss is instructive about *why*. `answer_shape` finds
answer-shaped tokens at over twice chance and still recovers nothing, because
§6 does not ask for tokens that look like answers — it asks for **every token
of the specific span that answers the query that has not yet arrived.**

## 8. Retained-set structure is a second constraint

The oracle's 40/40 does not replicate on SmolLM2-1.7B, which recovers 12/24.
Isolating the cause: holding the stream, needles, distances, flagged tokens
(1600) and recency fill constant, and varying only how many contiguous runs
the retained evidence occupies —

| runs | window | SmolLM2 | 95% CI | Qwen3 |
|---|---|---|---|---|
| **8** | 200 | **0.54** | [0.35, 0.72] | 0.96 |
| **4** | 400 | **0.92** | [0.74, 0.98] | 1.00 |
| 2 | 800 | 0.83 | [0.64, 0.93] | 1.00 |
| 1 | 1600 | 0.79 | [0.60, 0.91] | 1.00 |

**Fisher exact, 8 versus 4: p = 0.0078.** The effect is confined to the dead
zone — SmolLM2's near band is 9/9, 11/11, 8/8, 13/13, perfect at every
setting, because near-band needles sit inside the recency block with no gap to
bridge. Middle goes 3/7 → 7/8, far 1/8 → 4/5.

Window *width* does not explain it: a separate sweep varied width at fixed
count over a 13× range and moved 0.50 → 0.58, and the trend here is not
monotone in width — one run has the widest windows and scores worst of the
three low-count conditions.

**The consequence for method design is concrete.** H2O, SnapKV and their
relatives score and select tokens *individually*, producing maximally
fragmented caches. On a model with SmolLM2's sensitivity, retaining exactly
the right tokens in too many pieces buys far less than the same budget spent
on fewer, larger blocks. We do not identify the mechanism, and the 8-versus-4
threshold comes from one budget and one stream length.

## 9. Limitations

- **Two task families, both short-answer.** SQuAD QA fixes the corpus, query
  and metric, but extractive short-answer QA is still far from multi-hop
  reasoning, summarisation or agent traces, and neither task is a published
  benchmark run end to end.
- **Three models, 1.7B–8B, one architecture family plus one outsider.**
  Nothing above 8B. The real-QA benchmark is effectively single-model.
- **Scoring rules, not systems.** §2. A faithful reimplementation is the way
  to falsify §3's prediction that the full systems inherit the dead zone.
- **The screening test is supported by one successful rejection.** A signal
  that *passes* it and then succeeds would test the other direction; we have
  none, and a screen that only ever says no is consistent with being right and
  with being uselessly conservative.
- **The oracle is an upper bound, not a method.** It requires knowing the
  answers in advance.
- **One architectural family.** FIFO eviction, fixed per-layer budget,
  decisions shared across layers — the last forced by a single global
  attention mask. Per-head and per-layer policies untested.
- **The plateau of §4 has no clean quantitative model.** The horizon is
  verified at three admit rates; the plateau goes as `p^L` for an `L`-token
  span, directionally right with an inconsistent effective `L`.

## 10. Related work

Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452)
via differentiable gating and supervised distillation. RL for cache decisions
with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc,
episodic, uniform-budget eviction. Online Lagrangian dual ascent for KV
allocation exists (TriRoute, arXiv 2607.06601). Learned eviction for hybrid
linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method.** We are not aware of
prior work reporting recall by distance-band, and the near/far complementarity
and mid-context dead zone are not documented elsewhere.

**What we do and do not claim about prior systems.** We did not run H2O,
SnapKV or StreamingLLM, and we do not report that they fail on their own
benchmarks. We report that the scoring rules they introduced are functions of
position rather than utility, and that under a fixed per-layer budget each
leaves the middle unrecoverable while an oracle at the same budget recovers
all of it. Whether the full systems inherit this is our prediction, not our
measurement.

## Appendix A — Corrections to earlier drafts

Recorded because each was a natural error to make and the reasoning that
produced it is worth seeing.

1. **"The dead zone is intrinsic."** Argued from three training failures. The
   oracle refutes it. Three failures under related objectives are weak
   evidence about a hypothesis space; one control that changes a single
   variable is worth more than all three.
2. **The geometric survival law.** An earlier draft modelled retention as
   `(B/(B+chunk))^(d/chunk)` — the law for *random* eviction. FIFO removes the
   oldest deterministically, so the rounds are not independent. It predicted
   0.38 survival at d=2048 against an observed 0/48. Its apparent fit came
   from tercile bands straddling the horizon: coarse banding hid a step.
3. **"Halving the admit rate halves the plateau."** It does not. The residency
   law governs *token* survival; recall needs a whole multi-token span, so the
   plateau goes as `p^L`. Predicted 0.50 at `p`=0.5, observed 0.062.
4. **"Contiguity, untested."** Named as the leading explanation for §8's
   model-dependence without qualification. Split into fragment *length*
   (tested, false) and fragment *count* (tested, true).
5. **Baseline scope.** Earlier drafts said "no method recovers the middle,"
   which overstates what was run. §2 and §10 now scope the claim to scoring
   rules.

## Reproducibility

Code, seeds, raw per-trial results and all training histories: this
repository. `docs/phase1-findings.md` through `docs/phase10-prediction.md`
record each run including the errors made along the way. 129 tests; harness
verified bit-identical against a stock cache on two backends. Total compute
cost of every experiment reported here: under $8 of rented GPU time.
