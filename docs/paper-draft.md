# The Mid-Context Dead Zone in KV Cache Compression

**Draft — 2026-08-28. Not submission-ready. See §8 for what is missing.**

---

## Abstract

KV cache compression methods are evaluated on aggregate accuracy at a given budget, which hides *where* in the context they succeed. We measure recall as a function of distance from the query and find that published heuristics partition the context by positional bias rather than by utility: recency-based methods recover only the near field, attention-accumulation methods only the far field, and **no method at any budget recovers the middle**. Across 192 middle-band trials spanning four policies, four budgets, and five seeds, we observe zero successes (95% upper bound 1.6%), while an uncompressed cache answers 100% of the same queries.

We give a mechanism. Under repeated compression the cache is capped at `B + chunk`, so a token at distance `d` must survive `d/chunk` independent eviction rounds and its survival decays geometrically as `(B/(B+chunk))^(d/chunk)`. This predicts the observed cutoff for random admission quantitatively. The consequence is that **recency is not a policy choice but an emergent property**: any policy that does not actively protect distant tokens converges to it for free, which explains why random admission and a recency window are empirically indistinguishable.

We further report that training a policy to fix this is harder than the framing suggests, and document two specific reward-specification failures.

## 1. Motivation

A long-context model writes every token into its KV cache. Under a memory budget most of it must go, and the literature decides what to discard by ranking tokens with attention-derived heuristics: accumulated attention (H2O), observation-window voting (SnapKV), recency plus attention sinks (StreamingLLM), and layer-wise budget allocation (PyramidKV, Ada-KV).

These are evaluated by aggregate accuracy at a compression ratio. That metric cannot distinguish a method that retains information uniformly from one that retains a biased subset well. We ask a narrower question: **given a fact planted at distance `d` before the query, does the compressed cache still support answering it?**

## 2. Setup

- **Model.** Qwen3-1.7B, bf16, frozen. All decisions are made by external policies; no model weights are trained.
- **Streams.** 8192 tokens of prose with eight facts of the form `The secret code for <topic> is <6 digits>` planted at evenly spaced anchors with per-seed jitter. Each fact is queried from the end of the stream.
- **Compression.** Chunked prefill at 128 tokens; after each chunk the cache is compressed to budget `B` per layer, with the first four tokens pinned as attention sinks. Budgets 256, 512, 1024, 2048 against an 8192-token stream.
- **Policies.** `full` (no compression, ceiling), `recent` (StreamingLLM-style window), `h2o` (accumulated attention), `snapkv` (attention voted over a 32-token observation window), `random` at matched budget.
- **Metric.** Exact substring match of the planted code in 24 greedily decoded tokens. Results are grouped into near, middle, and far terciles of distance-back.

Correctness of the compression harness is established by an identity test: with an unbounded budget, logits are bit-identical to a stock `DynamicCache` under the same chunking, on both MPS and CUDA. Positions are passed explicitly throughout, since compression makes cache length diverge from stream position.

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

`recent` and `h2o` are exact complements. Each scores strongly in one band and **exactly zero** in the other, and neither reaches the middle. H2O's accumulated-attention sum favours tokens visible to the most query rows — that is, the oldest — which is why it inverts recency rather than improving on it. Neither bias has anything to do with whether a token is useful.

## 4. Mechanism: recency is emergent

`random` admission scored identically to `recent` at every budget, which initially looked like an implementation error. It is a property of repeated compression.

Because compression runs every chunk, the cache never exceeds `B + chunk`, so each round discards only `chunk` of `B + chunk` tokens — retention per round is `B/(B+chunk)`, not `B/L`. A token at distance `d` faces `d/chunk` rounds:

> **P(survive) = ( B / (B + chunk) )^(d / chunk)**

At B=2048, chunk=128 this is `0.941^(d/128)`. Against observed `random` recall:

| band | d range | predicted | observed |
|---|---|---|---|
| near | 303–2615 | 0.527 | 0.50 |
| middle | 2778–5290 | 0.157 | 0.00 |
| far | 5719–7873 | 0.043 | 0.00 |

The near band is predicted almost exactly. The model over-predicts the middle, and treating the fact as a multi-token span whose pieces must all survive errs the other way — the truth is between, and depends on code tokenisation and whether partial survival suffices.

The qualitative claim is robust: **survival decays geometrically with distance, so any policy that does not actively protect distant tokens degenerates into recency.** A compression policy is less a ranking problem than a rescue problem — it must counteract an exponential that is already running.

## 5. Independent corroboration

The dead zone is also visible distributionally, on a different corpus and a different measurement. On streams of Python source we identify *long-dependency positions* — those whose next token repeats an identifier last seen ≥2048 tokens earlier — and compare per-position `KL(p_full ‖ p_compressed)` there against everywhere else:

| B | mean KL ratio | KL at long-dep | KL baseline |
|---|---|---|---|
| 512 | 3.47 | 1.084 | 0.283 |
| 1024 | 6.34 | 1.050 | 0.157 |
| 2048 | **12.48** | 1.031 | 0.082 |

Increasing the budget collapses baseline KL (0.283 → 0.082) while leaving long-dependency KL essentially unchanged (1.084 → 1.031). **More cache fixes local prediction and does nothing for distant dependencies**, because under FIFO anything past `B` is gone regardless of how large `B` is.

Two independent measurements — behavioural recall on synthetic needles, distributional divergence on real code — reach the same conclusion: budget is not the binding constraint, retention policy is.

## 6. Negative results on learned policies

We attempted to train a small admission policy (297K parameters, frozen base model) against a dense reward `−KL(p_full ‖ p_compressed)`, with a Lagrangian price on cache occupancy and average-reward policy gradient. Two runs, both failures, both instructive:

**Run 1 — the reward drowns its own signal.** The trained policy scored 0.71 / 0.00 / 0.00, indistinguishable from `recent`. Training succeeded on its own terms (KL 0.23 → 0.12), but occupancy rose from 1634 to 1766 as KL fell: the policy learned to *admit more*, which under FIFO is recency. The cause is arithmetic. Long-dependency positions are ~1.5% of a stream, so even at 12.5× KL contrast they carry only **14.9%** of total KL mass. A reward that averages over positions hands 85% of its gradient to local prediction, which recency already maximises.

**Run 2 — weighting the reward destabilised training.** Weighting each position by its own divergence (`w_t = KL_t`) reached run 1's quality in half the steps (KL 0.119, occupancy 1775 at step 150) and then collapsed to admitting nothing at all — final occupancy 4, the pinned sinks, and 0.00 in every band. Average reward degraded alongside the metric, ruling out reward hacking; this was optimization failure under a high-variance ratio-of-moments objective with no trust region or entropy regularisation.

We report these because the failure modes are specific and reusable: an averaged dense reward is systematically biased toward recency in this setting, and a divergence-weighted reward is reducible by flattening rather than lowering the divergence.

## 7. Related work

The gap this occupies is narrow and worth stating precisely. Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452) using differentiable gating and supervised distillation. RL for cache decisions with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc eviction, episodic, with uniform budgets — and its ablation already establishes RL over supervised for that setting. Online Lagrangian dual ascent for KV allocation exists (TriRoute). Learned eviction for hybrid linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method:** to our knowledge no prior work reports recall by distance-band, and the near/far complementarity and the mid-context dead zone are not documented elsewhere.

## 8. Limitations

These are substantial and we do not think the paper is submission-ready without addressing at least the first two.

- **One model.** Qwen3-1.7B only. The dead zone may be a small-model artifact.
- **One task family.** Synthetic verbatim needle retrieval is the easiest possible case for a cache and the least like real long-context work. Exact substring match scores a paraphrase as a miss. Whether the dead zone appears on RULER, LongBench-v2, or agent traces is untested.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads, not the reference implementations; absolute numbers are not comparable to published ones.
- **Coarse banding.** "Middle" spans d≈2731–5461. A finer sweep would locate the boundary rather than assume a tercile.
- **The learned-policy results are two failures, not a method.** They currently support no positive claim.

## 9. What would make this a paper

1. **Replicate the dead zone on a second model and a real benchmark.** This is the single highest-value experiment and it is cheap. If it does not replicate, there is no paper.
2. **Finer distance resolution** to characterise the boundary rather than assert a tercile.
3. Either a learned policy that achieves nonzero middle-band recall, **or** an argument that the dead zone is intrinsic to fixed-budget streaming compression. Both are publishable; the second may be more interesting and is currently better supported by our own evidence.
