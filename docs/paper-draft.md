# The Mid-Context Dead Zone in KV Cache Compression

**Draft — 2026-08-29. Workshop-length. Not submission-ready: see §8.**

---

## Abstract

KV cache compression is evaluated on aggregate accuracy at a given budget, which hides *where* in the context a method succeeds. We measure recall as a function of distance from the query and find that published heuristics partition the context by positional bias rather than utility: recency methods recover only the near field, attention-accumulation methods only the far field, and **no method at any budget recovers the middle**. Across **228 middle-band trials** spanning four heuristics, three learned policies, four budgets and five seeds, we observe **zero successes** (95% upper bound 1.3%), while an uncompressed cache answers 100% of the same queries.

We give a mechanism. Under repeated compression the cache is capped at `B + chunk`, so a token at distance `d` must survive `d/chunk` independent eviction rounds and survives with probability `(B/(B+chunk))^(d/chunk)`. This predicts the observed cutoff for random admission quantitatively, and implies that **recency is emergent rather than chosen**: any policy that does not actively protect distant tokens converges to it for free.

We then attempt to train past the dead zone with a learned admission policy under three reward formulations. All three fail, and their failure is the paper's second contribution: two converge to behaviour indistinguishable from a recency window despite measurably improving on their own objective, and one destabilises. We argue the dead zone may be **intrinsic to fixed-budget streaming compression** rather than an artifact of objective design.

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

## 7. Interpretation

Three reward formulations; every one that trains stably lands on recency. We take this as evidence, not proof, that the dead zone is **intrinsic to fixed-budget streaming compression** rather than a fixable objective problem.

The mechanism of §4 suggests why. A token at d=4000 with B=2048, chunk=128 must survive ~31 consecutive eviction rounds. Protecting it requires a decision made at write time — before any query exists — that outweighs 31 rounds of competition from tokens with immediate predictive value. If no admission-time signal carries that information, no reward over admission-time decisions can recover it, and the failure is one of observability rather than optimization.

This is a falsifiable claim, and §9 names the experiment that would refute it.

## 8. Limitations

Substantial. We do not consider this submission-ready without at least the first two.

- **One model.** Qwen3-1.7B only. The dead zone may be a small-model artifact.
- **One task family.** Synthetic verbatim needle retrieval is the easiest case for a cache and least like real long-context work; exact substring match scores a paraphrase as a miss. Untested on RULER, LongBench-v2, or agent traces.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads; absolute numbers are not comparable to published ones.
- **Coarse banding.** "Middle" spans d≈2731–5461; a finer sweep would locate the boundary rather than assume a tercile.
- **Three training runs, no hyperparameter search.** The intrinsic-limitation claim in §7 is an interpretation of three failures, not a proof. A better-designed policy may yet succeed.
- **One architectural family.** All results assume FIFO eviction with a fixed per-layer budget and a decision shared across layers. Per-head or per-layer policies were not tested; the layer-sharing constraint was forced by the single global attention mask.

## 9. What would settle it

1. **Replicate on a second model and a real benchmark.** Highest value, cheap, and the paper does not survive a failure to replicate.
2. **An oracle admission experiment.** Give the policy the query at write time. If oracle admission recovers the middle band, the dead zone is an observability problem and §7 is right. If even an oracle fails, the limitation is the budget geometry itself. **This single experiment distinguishes the two hypotheses and should be run before anything else.**
3. **Finer distance resolution** to characterise the boundary.

## 10. Related work

The occupied cells around this work are worth stating precisely. Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452) via differentiable gating and supervised distillation. RL for cache decisions with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc eviction, episodic, uniform budgets — and its ablation already establishes RL over supervised in that setting. Online Lagrangian dual ascent for KV allocation exists (TriRoute, arXiv 2607.06601). Learned eviction for hybrid linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method.** We are not aware of prior work reporting recall by distance-band, and the near/far complementarity and mid-context dead zone are not documented elsewhere.

## Reproducibility

Code, seeds, raw per-trial results and all three training histories: this repository. `docs/phase1-findings.md`, `docs/phase2-findings.md`, `docs/phase2b-findings.md` record each run including the errors made along the way. 86 tests, harness verified bit-identical against a stock cache on two backends.
