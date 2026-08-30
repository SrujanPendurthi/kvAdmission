# The Mid-Context Dead Zone in KV Cache Compression

**Draft — 2026-08-30. Workshop-length. Not submission-ready: see §8.**

*Revised twice. The oracle retention control (§7) refuted this draft's earlier claim that the dead zone was intrinsic. A fine-resolution distance sweep (§4) then refuted its stated mechanism, and a second-model replication (§3, §7) narrowed the oracle claim.*

---

## Abstract

KV cache compression is evaluated on aggregate accuracy at a given budget, which hides *where* in the context a method succeeds. We measure recall as a function of distance from the query and find that published heuristics partition the context by positional bias rather than utility: recency methods recover only the near field, attention-accumulation methods only the far field, and **no method at any budget recovers the middle**. Across **284 middle-band trials** spanning two models, five heuristics, three learned policies, four budgets and five seeds, we observe **zero successes** (95% upper bound 1.1%), while an uncompressed cache answers 100% of the same queries.

We give a mechanism. Under admission at rate `p` followed by FIFO eviction to budget `B`, the cache is a contiguous window over the most recent `B/p` positions, so recall is a **step function with a hard horizon at `d = B/p`** — beyond it, retention is exactly zero rather than merely unlikely. A 256-token-resolution sweep confirms the step for both recency and rate-matched random admission. This implies **recency is emergent rather than chosen**: any policy that does not actively protect distant tokens converges to it for free.

We then attempt to train past the dead zone with a learned admission policy under three reward formulations. All three fail: two converge to behaviour indistinguishable from a recency window despite measurably improving on their own objective, and one destabilises. A query-aware policy that may consult the pending queries at write time also scores zero in the middle band.

Finally we run an **oracle retention** control — the same budget, chunking, eviction geometry and evaluation, but the policy is told which tokens carry the answers. On Qwen3-1.7B the oracle scores **40/40, perfect recall in every band**, matching the uncompressed cache at a quarter of the context. The dead zone is therefore **not** a capacity or geometry limit on that model; the failure is one of **observability**. On SmolLM2-1.7B the same oracle recovers only 12/24, so how fully the capacity can be exploited is **model-dependent** — a result we did not anticipate and cannot yet explain.

## 1. Motivation

A long-context model writes every token into its KV cache; under a budget most must go. The literature decides what to discard by ranking tokens with attention-derived heuristics — accumulated attention (H2O), observation-window voting (SnapKV), recency plus attention sinks (StreamingLLM), layer-wise budget allocation (PyramidKV, Ada-KV) — and reports aggregate accuracy at a compression ratio.

That metric cannot distinguish a method that retains information uniformly from one that retains a biased subset well. We ask a narrower question: **given a fact planted at distance `d` before the query, does the compressed cache still support answering it?**

## 2. Setup

- **Models.** Qwen3-1.7B (primary) and SmolLM2-1.7B-Instruct (replication), bf16, frozen throughout. No model weights are trained anywhere in this work.
- **Streams.** 8192 tokens of prose with eight facts of the form `The secret code for <topic> is <6 digits>` at evenly spaced anchors with per-seed jitter, each queried from the end of the stream.
- **Compression.** Chunked prefill at 128 tokens; after each chunk the cache is compressed to budget `B` per layer, first four tokens pinned as attention sinks. Budgets 256/512/1024/2048.
- **Policies.** `full` (ceiling), `recent`, `h2o`, `snapkv`, `random` at matched budget, plus a query-aware policy (§7.3) and an oracle (§7).
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

### 3.1 Replication on a second model

The same sweep on SmolLM2-1.7B-Instruct, 3 seeds × 8 needles × 2 budgets:

| policy | B | near | middle | far |
|---|---|---|---|---|
| `full` | 8192 | 9/9 | **7/7** | 7/8 |
| `recent` | 2048 | 6/9 | **0/7** | 0/8 |
| `random` | 2048 | 4/9 | **0/7** | 0/8 |
| `snapkv` | 2048 | 3/9 | **0/7** | 0/8 |
| `h2o` | 2048 | 0/9 | **0/7** | 0/8 |
| all four | 512 | 3/36 | **0/28** | 0/32 |

**0/56 middle-band hits across every compressed cell** (95% upper 5.4%) against an uncompressed ceiling of 7/7. The dead zone is not a Qwen artifact.

One difference: `h2o` scores 0/8 in the far band here, against 8/14 on Qwen. SmolLM2's attention mass is distributed differently enough that accumulated-attention scoring does not reliably hold the opening of the stream. It rescues the middle on neither model.

## 4. Mechanism: a hard horizon, not a slope

`random` admission scored identically to `recent` at every budget. This is not noise; it follows from the geometry of repeated compression, and a fine-resolution sweep pins down the form.

We planted 16 needles per stream over 4 seeds at B=2048, chunk=128, and binned recall at 256-token resolution rather than in terciles:

| policy | d < 2048 | d ≥ 2048 |
|---|---|---|
| `recent` | **16/16** | **0/48** |
| `random` | 13/16 | **0/48** |
| `h2o` | 0/16 | 10/48 — all at d > 6600 |

Bin by bin, `recent` scores 1.00 in every bin below 2048 and 0.00 in every bin above. **This is a step function at `d = B`.**

Under admission at rate `p` followed by FIFO eviction to budget `B`, occupancy saturates at `B` and the cache becomes a contiguous window over the most recent `B/p` raw positions, thinned to density `p`:

> **P(token at distance `d` is resident) = `p` if `d < B/p`, else 0**

| policy | p | predicted horizon | predicted plateau | observed |
|---|---|---|---|---|
| `recent` | 1.000 | 2048 | 1.00 | horizon in (2048, 2304]; plateau 16/16 |
| `random` | 0.941 | 2176 | 0.94 single token, 0.69 for a 6-token span | horizon in (2048, 2304]; plateau 13/16 = 0.81 |

`recent` is predicted exactly. `random`'s plateau falls between the single-token and whole-span predictions, which is where a partially-redundant span should land. The 256-token bins cannot separate a horizon of 2048 from one of 2176. The law makes a further prediction we have not tested: **halving the admit rate should double the horizon and halve the plateau.**

> **Correction.** An earlier draft modelled survival as `(B/(B+chunk))^(d/chunk)` — a token surviving `d/chunk` independent rounds. That is the law for *random* eviction. FIFO does not thin the cache uniformly; it removes the oldest deterministically, so the rounds are not independent. The geometric law predicts `random` survival of 0.38 at `d`=2048 and 0.16 in the middle band, against an observed 0/48 for every `d` ≥ 2048. Its apparent fit came from tercile bands that straddled the horizon, and band-level averaging hid a step. We record the error because coarse banding concealing a sharp transition is a general hazard in this measurement.

The correction strengthens the qualitative claim. It is not that distant tokens are unlikely to survive — **beyond the horizon they survive with probability exactly zero**, so any policy that does not explicitly protect a token past `B/p` loses it with certainty. The middle band lies entirely beyond the horizon at every budget tested; that is the dead zone in one line. Compression is not a ranking problem but a rescue problem.

`h2o` has a horizon at the opposite end: its 10 hits all fall at `d > 6600`, a window roughly 1300 tokens wide anchored to the *start* of the stream, because accumulated attention concentrates on the earliest tokens. Two policies, two windows, neither covering the middle.

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

**On Qwen3-1.7B the oracle matches the uncompressed cache exactly, at a quarter of the context, on every seed.**

Nothing about the budget, the chunking, or FIFO eviction prevents mid-context recall. A 2048-token cache holding the *right* 2048 tokens answers every query. **The dead zone is not intrinsic to fixed-budget streaming compression.**

### 7.1 The ceiling is model-dependent

The same oracle on SmolLM2-1.7B-Instruct does not replicate that result:

| | near | middle | far | total |
|---|---|---|---|---|
| `full` (uncompressed) | 9/9 | 7/7 | 7/8 | 23/24 |
| **oracle B=2048** | **9/9** | **2/7** | **1/8** | **12/24** |
| `recent` B=2048 | 6/9 | 0/7 | 0/8 | 6/24 |

The oracle doubles `recent`'s total and produces the first non-zero middle-band cell any policy has reached on this model — but it recovers half the trials, not all of them, against an uncompressed ceiling of 23/24.

The pattern is specific: the oracle is perfect exactly where retention coincides with contiguous recency (near, 9/9) and poor wherever it depends on the flag (middle 2/7, far 1/8). Our leading hypothesis is **contiguity**: the oracle's cache is spliced — answer spans from position ~2000 adjacent to recency fill from ~7000, with a gap in position indices — and Qwen3 reads a spliced cache where SmolLM2 does not. This is fitted to three seeds and is untested; the direct control is an oracle retaining a *contiguous block* of equal size at equal distance, which we have not run.

**The honest statement is narrower than this section's opening.** What replicates across both models is that the budget is not the binding constraint: the oracle beats every heuristic at the same budget on both. What does not replicate is that the full uncompressed ceiling is reachable. Whether a model can exploit a non-contiguous retained set is a variable no prior work in this line varies, and on this evidence it matters.

> An earlier version of this draft argued the opposite — that the dead zone was probably intrinsic, on the strength of three training failures. That inference was wrong, and the oracle refutes it on both models: at the same budget the oracle beats every heuristic, and on Qwen it reaches the uncompressed ceiling. Three failures under related objectives are weak evidence about a hypothesis space, and a control that changes exactly one variable is worth more than all three. We record the error because the reasoning that produced it is a natural one to repeat — and note that §7.1 is the same lesson applied to the control itself, which we had also treated as settled after one model.

**What this makes of §6.** The three training failures were not optimization failures. They were policies asked to predict, from write-time features alone, something the write-time input does not determine. A token's utility here is fixed by a query that has not arrived and is not inferable from the token, its hidden state, its key norm, or its local attention. The learned policies converged to recency because **recency is the optimal policy under that information constraint** — when nothing distinguishes which distant token will matter, keeping the nearest ones maximises expected utility.

This reframes the horizon of §4. It is not a wall; it is the default that any *uninformed* policy collapses onto. The horizon is escapable — the oracle steps past it — but only with information that write-time features do not carry.

### 7.2 How good must a proxy be?

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

### 7.3 Query-aware admission: knowing the question is not enough

§7.2's specification asks for a signal that identifies answer tokens at write time. The most obvious candidate is the query itself. We tested a **query-aware** policy that scores each token by maximum cosine similarity between its representation and a pooled embedding of the pending queries. This is already a semi-oracle — a streaming policy would not hold the queries at write time — so it is an upper bound on what write-time query access buys.

| policy | B | near | middle | far |
|---|---|---|---|---|
| query-aware | 2048 | 0/14 | **0/12** | 5/14 |
| query-aware | 512 | 0/14 | **0/12** | 0/14 |
| `recent` (reference) | 2048 | 10/14 | 0/12 | 0/14 |
| oracle (reference) | 2048 | 1.00 | **1.00** | 1.00 |

**Zero in the middle band at both budgets**, and it forfeits the near band that recency gets for free, inverting into a far-band bias like `h2o`.

This failure is informative because of where it sits. Query-aware admission has strictly more information than the heuristics and strictly less than the oracle, and it lands on the heuristic side of the gap. **The information the oracle holds is not "what is being asked" but "which tokens answer it,"** and embedding similarity does not recover the second from the first.

An earlier version of this section recommended query-aware admission first, on the reasoning that it "converts an unanswerable prediction problem into a retrieval one." That framing survives; the cheap instantiation of it does not. A retrieval index over evicted tokens is a materially different mechanism from cosine reranking at write time, and remains open — but write-time query access is not, on its own, the missing ingredient.

## 8. Limitations

Substantial. We do not consider this submission-ready without the first item.

- **One task family.** Synthetic verbatim needle retrieval is the easiest case for a cache and least like real long-context work; exact substring match scores a paraphrase as a miss. Untested on RULER, LongBench-v2, or agent traces. **This is now the only limitation with nothing measured against it.**
- **Two models, one size class.** Qwen3-1.7B and SmolLM2-1.7B-Instruct. The dead zone replicates across both, but both are ~1.7B; it may still be a small-model artifact, and the oracle result already differs between them.
- **The oracle ceiling does not replicate.** §7.1. We report a hypothesis (contiguity) we have not tested.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads; absolute numbers are not comparable to published ones.
- **Three training runs, no hyperparameter search.** The claim is not that no policy can succeed — the oracle shows one can on at least one model — but that these three did not, for a reason §7 identifies.
- **The oracle is an upper bound, not a method.** It requires knowing the answers in advance.
- **One query-aware instantiation.** Pooled last-hidden-state cosine. A negative result about one similarity function is weak evidence about query-aware admission generally.
- **One architectural family.** FIFO eviction, fixed per-layer budget, decision shared across layers — forced by the single global attention mask. Per-head and per-layer policies untested.
- **The horizon law is fitted on two admit rates.** `p`=1.0 and `p`=0.941 are close together; the prediction that halving `p` doubles the horizon is untested.

## 9. What comes next

Three of the four items previously listed here have been run: the second-model replication (§3.1), the finer distance sweep (§4, which corrected the mechanism), and query-aware admission (§7.3, which failed). What remains, in order of value:

1. **A real benchmark.** RULER or LongBench-v2. The only limitation with no measurement against it, and the paper does not survive a failure to reproduce the effect on a non-synthetic task.
2. **The contiguity control.** An oracle retaining a contiguous block versus scattered spans, same size, same distance. Cheap, and it decides whether §7.1's model-dependence is about retained-set *shape* — which would be a finding in its own right.
3. **Vary the admit rate.** One run at `p`=0.5 tests the horizon law's only untested prediction.
4. **Retrieval over evicted tokens**, rather than write-time query scoring. §7.3 rules out the cheap version of query-awareness, not the direction.

## 10. Related work

The occupied cells around this work are worth stating precisely. Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452) via differentiable gating and supervised distillation. RL for cache decisions with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc eviction, episodic, uniform budgets — and its ablation already establishes RL over supervised in that setting. Online Lagrangian dual ascent for KV allocation exists (TriRoute, arXiv 2607.06601). Learned eviction for hybrid linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method.** We are not aware of prior work reporting recall by distance-band, and the near/far complementarity and mid-context dead zone are not documented elsewhere.

## Reproducibility

Code, seeds, raw per-trial results and all three training histories: this repository. `docs/phase1-findings.md`, `docs/phase2-findings.md`, `docs/phase2b-findings.md` and `docs/phase3-batch-findings.md` record each run including the errors made along the way. 100 tests, harness verified bit-identical against a stock cache on two backends.
