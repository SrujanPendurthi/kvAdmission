# The Mid-Context Dead Zone in KV Cache Compression

**Draft — 2026-08-29. Workshop-length. Not submission-ready: see §8.**

*Revised after the oracle retention control (§7) refuted this draft's earlier claim that the dead zone was intrinsic.*

---

## Abstract

KV cache compression is evaluated on aggregate accuracy at a given budget, which hides *where* in the context a method succeeds. We measure recall as a function of distance from the query and find that published heuristics partition the context by positional bias rather than utility: recency methods recover only the near field, attention-accumulation methods only the far field, and **no method at any budget recovers the middle**. Across **228 middle-band trials** spanning four heuristics, three learned policies, four budgets and five seeds, we observe **zero successes** (95% upper bound 1.3%), while an uncompressed cache answers 100% of the same queries.

We give a mechanism. Under repeated compression the cache is capped at `B + chunk`, so a token at distance `d` must survive `d/chunk` independent eviction rounds and survives with probability `(B/(B+chunk))^(d/chunk)`. This predicts the observed cutoff for random admission quantitatively, and implies that **recency is emergent rather than chosen**: any policy that does not actively protect distant tokens converges to it for free.

We then attempt to train past the dead zone with a learned admission policy under three reward formulations. All three fail: two converge to behaviour indistinguishable from a recency window despite measurably improving on their own objective, and one destabilises.

Finally we run an **oracle retention** control — the same budget, chunking, eviction geometry and evaluation, but the policy is told which tokens carry the answers. The oracle scores **40/40, perfect recall in every band**. The dead zone is therefore **not** a capacity or geometry limit: a 2048-token cache holding the right 2048 tokens answers every query the uncompressed cache does. The failure is one of **observability** — no signal available at write time identifies which tokens will matter.

## 1. Motivation

A long-context model writes every token into its KV cache; under a budget most must go. The literature decides what to discard by ranking tokens with attention-derived heuristics — accumulated attention (H2O), observation-window voting (SnapKV), recency plus attention sinks (StreamingLLM), layer-wise budget allocation (PyramidKV, Ada-KV) — and reports aggregate accuracy at a compression ratio.

That metric cannot distinguish a method that retains information uniformly from one that retains a biased subset well. We ask a narrower question: **given a fact planted at distance `d` before the query, does the compressed cache still support answering it?**

## 2. Setup

- **Model.** Qwen3-1.7B, bf16, frozen throughout. No model weights are trained anywhere in this work.
- **Streams.** 8192 tokens of prose with eight facts of the form `The secret code for <topic> is <6 digits>` at evenly spaced anchors with per-seed jitter, each queried from the end of the stream.
- **Compression.** Chunked prefill at 128 tokens; after each chunk the cache is compressed to budget `B` per layer, first four tokens pinned as attention sinks. Budgets 256/512/1024/2048.
- **Policies.** `full` (ceiling), `recent`, `h2o`, `snapkv`, `random` at matched budget.
- **Metric.** Exact substring match of the code in 24 greedily decoded tokens, grouped into near/middle/far terciles of distance-back. Zero-hit cells report a rule-of-three upper bound rather than a standard error, which is zero and misleadingly tight.

Harness correctness rests on an identity test: at unbounded budget, logits are **bit-identical** to a stock `DynamicCache` under the same chunking, verified independently on MPS and CUDA. Positions are always passed explicitly, since compression makes cache length diverge from stream position — a silent corruption otherwise.

## 3. Result: the dead zone

| region | hits / trials | rate | 95% upper |
|---|---|---|---|
| near | 31 / 224 | 0.138 | — |
| **middle** | **0 / 192** | **0.000** | **0.016** |
| far | 9 / 224 | 0.040 | — |

`full` scores 1.00 in all three bands, so every zero is attributable to compression rather than task difficulty.

Per-policy at B=2048 — a quarter of the stream:

| policy | near | middle | far |
|---|---|---|---|
| `recent` | **10/14** | 0/12 | **0/14** |
| `random` | 7/14 | 0/12 | 0/14 |
| `snapkv` | 4/14 | 0/12 | 0/14 |
| `h2o` | **0/14** | 0/12 | **8/14** |

`recent` and `h2o` are exact complements: each scores strongly in one band and **exactly zero** in the other, and neither reaches the middle. H2O's accumulated-attention sum favours tokens visible to the most query rows — the oldest — so it inverts recency rather than improving on it. Neither bias tracks whether a token is useful.

## 4. Mechanism: recency is emergent

`random` admission scored identically to `recent` at every budget. This is not noise; it follows from repeated compression.

Because compression runs every chunk, the cache never exceeds `B + chunk`, so each round discards only `chunk` of `B + chunk` tokens — retention per round is `B/(B+chunk)`, not `B/L`. A token at distance `d` faces `d/chunk` rounds:

> **P(survive) = ( B / (B + chunk) )^(d / chunk)**

At B=2048, chunk=128 this is `0.941^(d/128)`. Against observed `random` recall:

| band | d range | predicted | observed |
|---|---|---|---|
| near | 303–2615 | 0.527 | 0.50 |
| middle | 2778–5290 | 0.157 | 0.00 |
| far | 5719–7873 | 0.043 | 0.00 |

The near band is predicted almost exactly; the model over-predicts the middle, and a multi-token span model errs the other way, so the truth lies between and depends on tokenisation.

The qualitative claim is robust: **survival decays geometrically with distance, so any policy that does not actively protect distant tokens degenerates into recency.** Compression is less a ranking problem than a rescue problem — the policy must counteract an exponential already in progress.

## 5. Independent corroboration

The dead zone is also visible distributionally, on a different corpus and a different measurement. On Python source we identify *long-dependency positions* — those whose next token repeats an identifier last seen ≥2048 tokens earlier — and compare per-position `KL(p_full ‖ p_compressed)` there against elsewhere:

| B | KL ratio | KL at long-dep | KL baseline |
|---|---|---|---|
| 512 | 3.47 | 1.084 | 0.283 |
| 1024 | 6.34 | 1.050 | 0.157 |
| 2048 | **12.48** | 1.031 | 0.082 |

Raising the budget collapses baseline KL (0.283 → 0.082) and leaves long-dependency KL essentially unchanged (1.084 → 1.031). **More cache fixes local prediction and does nothing for distant dependencies**, because under FIFO anything past `B` is gone regardless of how large `B` is.

Two independent measurements — behavioural recall on synthetic needles, distributional divergence on real code — agree: budget is not the binding constraint, retention policy is.

## 6. Three attempts to train past it

We trained a 297K-parameter admission head (frozen base model, one admit/skip decision per token shared across layers, FIFO eviction, Lagrangian price on occupancy, average-reward policy gradient) under three rewards derived from `−KL(p_full ‖ p_compressed)`.

| reward | near | middle | far | outcome |
|---|---|---|---|---|
| plain chunk mean | 0.71 | **0.00** | 0.00 | indistinguishable from `recent` |
| divergence-weighted (`w=KL`) | 0.00 | **0.00** | 0.00 | collapsed to admitting nothing |
| mixed `0.7·weighted + 0.3·mean` | 0.71 | **0.00** | 0.00 | indistinguishable from `recent` |

**Run 1 — averaging drowns the signal.** Training succeeded on its own terms (KL 0.23 → 0.12) but occupancy *rose* from 1634 to 1766 as KL fell: the policy learned to admit more, which under FIFO is recency. The cause is arithmetic — long-dependency positions are ~1.5% of a stream, so even at 12.5× KL contrast they carry only **14.9%** of total KL mass, handing 85% of the gradient to local prediction.

**Run 2 — weighting destabilises.** Weighting positions by their own divergence reached run 1's quality in half the steps, then collapsed to admitting nothing (final occupancy 4, the pinned sinks). Average reward degraded alongside the metric, ruling out reward hacking; this was optimization failure under a high-variance ratio-of-moments objective.

**Run 3 — fixing both changes nothing.** With a mixed reward, an entropy bonus, and best-checkpoint retention, training was stable and the retained checkpoint was **better than run 1 on the training objective** (`kl_raw` 0.103 vs 0.121, occupancy 1775 vs 1773). Its behaviour was nevertheless identical to a recency window.

**Pooled across all three learned policies: 0 / 36 middle-band hits.** Combined with the heuristics, **0 / 228 (95% upper bound 1.3%)**.

Run 3 is the informative one. It was designed to test the run-1 diagnosis — that the reward drowned long-range signal — and the diagnosis was correct arithmetically yet did not bind. A policy can measurably improve at matching the full-cache distribution while remaining behaviourally indistinguishable from keeping the most recent `B` tokens.

§7 shows why none of these could have worked.

## 7. The dead zone is observability, not geometry

Three reward formulations all landing on recency admits two explanations. Either the dead zone is **intrinsic** — a fixed-budget streaming cache simply cannot carry mid-context information — or it is a matter of **observability**, where the information is retainable but nothing at write time identifies what to retain.

We ran the control that separates them. **Oracle retention** uses the identical budget, chunking, sink prefix, FIFO eviction and evaluation harness, changing exactly one thing: the policy is told which token spans carry the answers, scores them astronomically, and ranks everything else by recency. It is implemented through the same `apply_eviction` path the baselines use.

| policy | near | middle | far | trials |
|---|---|---|---|---|
| best heuristic (B=2048) | 0.71 | **0.00** | 0.57 | — |
| best learned policy | 0.71 | **0.00** | 0.00 | — |
| **oracle retention** | **1.00** | **1.00** | **1.00** | **40/40** |
| `full` (no compression) | 1.00 | 1.00 | 1.00 | — |

**The oracle matches the uncompressed cache exactly, at a quarter of the context, on every seed.**

This settles it. Nothing about the budget, the chunking, FIFO eviction, or geometric survival prevents mid-context recall. A 2048-token cache holding the *right* 2048 tokens answers every query. **The dead zone is not intrinsic to fixed-budget streaming compression.**

> An earlier version of this draft argued the opposite — that the dead zone was probably intrinsic, on the strength of three training failures. That inference was wrong, and the oracle is what refutes it. Three failures under related objectives are weak evidence about a hypothesis space; one control is decisive. We record the error because the reasoning that produced it is a natural one to repeat.

**What this makes of §6.** The three training failures were not optimization failures. They were policies asked to predict, from write-time features alone, something the write-time input does not determine. A token's utility here is fixed by a query that has not arrived and is not inferable from the token, its hidden state, its key norm, or its local attention. The learned policies converged to recency because **recency is the optimal policy under that information constraint** — when nothing distinguishes which distant token will matter, keeping the nearest ones maximises expected utility.

This reframes the geometric-survival mechanism of §4. It is not a wall; it is the default that any *uninformed* policy decays toward. The exponential is escapable, but only with information that write-time features do not carry.

### 7.1 How good must a proxy be?

The oracle establishes that the capacity exists. Degrading it along two axes measures what a realisable signal would need to achieve. `coverage` is the proxy's recall over answer tokens; `fp_rate` is its false-positive rate over everything else.

| coverage | fp | overall | middle-band |
|---|---|---|---|
| 1.00 | 0.00 | 24/24 | 7/7 |
| 0.75 | 0.00 | 12/24 | **1/7** |
| 0.50 | 0.00 | 8/24 | 0/7 |
| 0.25 | 0.00 | 6/24 | 0/7 |

**Coverage is unforgiving.** Missing a quarter of the answer tokens collapses middle-band recall from 7/7 to 1/7. An answer spans only a few tokens, and retaining most of a six-digit code is worth nothing.

**Precision is free — until the budget binds.** The flagged set is approximately `fp_rate · L`, so with `L`=8192 and `B`=2048 the crowding threshold is `fp_rate ≈ 0.25`. Below it every flagged token fits and false positives cost nothing; above it they evict true ones. Recall then follows a second simple law:

> **recall = min( 1, B / |flagged| )**

| fp | flagged | predicted | observed |
|---|---|---|---|
| 0.05 | 506 | 1.00 | 1.00 |
| 0.20 | 1734 | 1.00 | 1.00 |
| 0.30 | 2554 | 0.80 | 0.75 |
| 0.40 | 3373 | 0.61 | 0.62 |
| 0.50 | 4192 | 0.49 | 0.50 |

Accurate to within 0.05 across the range. Once the flagged set overflows, survivors are a proportional subset and a given answer token survives with probability `B/|flagged|`.

**The specification this yields is two-sided:** a proxy must identify *essentially every token* of an answer span, and must keep its *total* flagged set within the budget. It may be entirely indiscriminate about which extra tokens it keeps, so long as it does not keep too many. A policy flagging 30% of a stream is worse than useless even with perfect coverage.

This also explains the trained policies more precisely than §6 does on its own. A soft admission head produces graded scores across a span, and partial retention of a span scores zero here — so gradient signal that improves average token-level accuracy need not improve recall at all.

**The consequence for method design is sharp.** Effort spent on reward shaping, credit assignment, or optimizer stability at the admission decision is misdirected — those were the three things we tried. What is needed is a policy with access to signal that write-time features lack:

- **Query-aware admission**, where retention decisions can consult the query, moving the problem from prediction to retrieval
- **Deferred or revisable decisions** — a second pass, or re-admission from a cheaper store, so utility can be assessed after it is revealed
- **Learned proxies for future queryability** trained across many query distributions, if such a signal exists at all — which the oracle shows is the open empirical question, not whether the capacity exists

## 8. Limitations

Substantial. We do not consider this submission-ready without at least the first two.

- **One model.** Qwen3-1.7B only. The dead zone may be a small-model artifact.
- **One task family.** Synthetic verbatim needle retrieval is the easiest case for a cache and least like real long-context work; exact substring match scores a paraphrase as a miss. Untested on RULER, LongBench-v2, or agent traces.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads; absolute numbers are not comparable to published ones.
- **Coarse banding.** "Middle" spans d≈2731–5461; a finer sweep would locate the boundary rather than assume a tercile.
- **Three training runs, no hyperparameter search.** The claim is not that no policy can succeed — the oracle proves one can — but that these three did not, for a reason §7 identifies.
- **The oracle is an upper bound, not a method.** It requires knowing the answers in advance. It establishes that the capacity exists; it says nothing about whether an implementable policy can find that signal.
- **One architectural family.** All results assume FIFO eviction with a fixed per-layer budget and a decision shared across layers. Per-head or per-layer policies were not tested; the layer-sharing constraint was forced by the single global attention mask.

## 9. What comes next

The oracle experiment that previously headed this list has been run; §7 reports it. What remains:

1. **Replicate on a second model and a real benchmark.** Highest value, cheap, and the paper does not survive a failure to replicate. Still the first thing to do.
2. **Query-aware admission.** The oracle shows the capacity exists at B=2048. The natural next question is how much of that gap a policy closes when it may consult the query at write time — which converts an unanswerable prediction problem into a retrieval one.
3. **Partial-oracle ablation.** Degrade the oracle — noisy spans, partial coverage, delayed knowledge — to measure how much signal is actually required. This bounds how good a learned proxy would need to be, and is cheap.
4. **Finer distance resolution**, to locate the boundary rather than assume a tercile.

## 10. Related work

The occupied cells around this work are worth stating precisely. Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452) via differentiable gating and supervised distillation. RL for cache decisions with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc eviction, episodic, uniform budgets — and its ablation already establishes RL over supervised in that setting. Online Lagrangian dual ascent for KV allocation exists (TriRoute, arXiv 2607.06601). Learned eviction for hybrid linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method.** We are not aware of prior work reporting recall by distance-band, and the near/far complementarity and mid-context dead zone are not documented elsewhere.

## Reproducibility

Code, seeds, raw per-trial results and all three training histories: this repository. `docs/phase1-findings.md`, `docs/phase2-findings.md`, `docs/phase2b-findings.md` record each run including the errors made along the way. 86 tests, harness verified bit-identical against a stock cache on two backends.
