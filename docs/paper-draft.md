# The Mid-Context Dead Zone in KV Cache Compression

**Draft — 2026-08-30. Workshop-length. Not submission-ready: see §8.**

*Revised three times. The oracle retention control (§7) refuted this draft's earlier claim that the dead zone was intrinsic. A fine-resolution distance sweep (§4) then refuted its stated mechanism, a second-model replication (§3, §7) narrowed the oracle claim, and a real-QA benchmark (§3.2) replaced the synthetic task this draft previously rested on entirely.*

---

## Abstract

KV cache compression is evaluated on aggregate accuracy at a given budget, which hides *where* in the context a method succeeds. We measure recall as a function of distance from the query and find that published heuristics partition the context by positional bias rather than utility: recency methods recover only the near field, attention-accumulation methods only the far field, and **no method at any budget recovers the middle**. Across **284 middle-band trials** on synthetic needles — two models, five heuristics, three learned policies, four budgets, five seeds — we observe **zero successes** (95% upper bound 1.1%) against an uncompressed cache that answers 100% of the same queries. On real planted-evidence QA the figure is **1/64** (95% upper 7.2%) against an uncompressed 6/8.

We give a mechanism. Under admission at rate `p` followed by FIFO eviction to budget `B`, the cache is a contiguous window over the most recent `B/p` positions, so recall is a **step function with a hard horizon at `d = B/p`** — beyond it, retention is exactly zero rather than merely unlikely. A 256-token-resolution sweep confirms the step for both recency and rate-matched random admission. This implies **recency is emergent rather than chosen**: any policy that does not actively protect distant tokens converges to it for free.

We then attempt to train past the dead zone with a learned admission policy under three reward formulations. All three fail: two converge to behaviour indistinguishable from a recency window despite measurably improving on their own objective, and one destabilises. A query-aware policy that may consult the pending queries at write time also scores zero in the middle band.

We run an **oracle retention** control — the same budget, chunking, eviction geometry and evaluation, but the policy is told which tokens carry the answers. On Qwen3-1.7B the oracle scores **40/40, perfect recall in every band**, matching the uncompressed cache at a quarter of the context. The dead zone is therefore **not** a capacity or geometry limit on that model; the failure is one of **observability**. On SmolLM2-1.7B the same oracle recovers only 12/24. Isolating the cause shows it is **fragmentation**: holding evidence, distance and token budget fixed and varying only how many contiguous runs the retained set occupies moves SmolLM2 from 0.54 at eight fragments to 0.92 at four (p=0.008), while Qwen3 sits at ceiling throughout. The capacity is there on both models, but on one of them it is only reachable if the retained set stays coarse — a constraint no token-level heuristic respects.

Finally we replace the synthetic task entirely with **planted-evidence QA over real SQuAD passages**, scored by containment and token F1 against a measured closed-book floor of 0/24. On Qwen3 every finding survives: middle-band recall across four heuristics and two budgets is **1/64** against an uncompressed 6/8; `recent` and `h2o` remain near-exact complements (near 7/9 vs 1/9, far 0/7 vs 3/7); and the oracle scores **18/24 against the uncompressed 19/24** at a quarter of the context, where the best heuristic scores 7/24. This benchmark is single-model: extending it to SmolLM2 leaves an eight-trial window between a 3/24 floor and an 11/24 ceiling, too narrow to separate any policy from any other.

## 1. Motivation

A long-context model writes every token into its KV cache; under a budget most must go. The literature decides what to discard by ranking tokens with attention-derived heuristics — accumulated attention (H2O), observation-window voting (SnapKV), recency plus attention sinks (StreamingLLM), layer-wise budget allocation (PyramidKV, Ada-KV) — and reports aggregate accuracy at a compression ratio.

That metric cannot distinguish a method that retains information uniformly from one that retains a biased subset well. We ask a narrower question: **given a fact planted at distance `d` before the query, does the compressed cache still support answering it?**

## 2. Setup

- **Models.** Qwen3-1.7B (primary) and SmolLM2-1.7B-Instruct (replication), bf16, frozen throughout. No model weights are trained anywhere in this work.
- **Streams.** Two task families. *Synthetic:* 8192 tokens of prose with eight facts of the form `The secret code for <topic> is <6 digits>` at evenly spaced anchors with per-seed jitter, each queried from the end. *Real (§3.2):* 8192 tokens of distractor SQuAD paragraphs with eight gold paragraphs planted at the same anchors, queried by their human-written questions.
- **Metrics.** Exact substring match on the synthetic task; containment plus token F1 on real QA, against a measured closed-book floor. Zero-hit cells report a rule-of-three upper bound rather than a standard error, which is zero and misleadingly tight.
- **Compression.** Chunked prefill at 128 tokens; after each chunk the cache is compressed to budget `B` per layer, first four tokens pinned as attention sinks. Budgets 256/512/1024/2048.
- **Policies.** `full` (ceiling), `recent`, `h2o`, `snapkv`, `random` at matched budget, plus a query-aware policy (§7.3) and an oracle (§7).
- **Banding.** Near/middle/far terciles of distance-back from the query.

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

### 3.2 Replication on real QA

Synthetic verbatim recall was this work's weakest point: the easiest possible case for a cache, scored by a metric that counts a paraphrase as a miss. We replaced it while keeping the distance axis everything depends on.

Streams are distractor SQuAD paragraphs; the evidence is a real SQuAD paragraph planted at a controlled anchor; the query is the human-written question; the answer is a natural-language span scored by containment, with token F1 reported alongside. This is RULER-style planted QA rather than LongBench-v2 because LongBench-v2 ships no evidence spans, and recall *against distance* is the whole measurement.

Two controls make the numbers readable. **Closed-book floor: 0/24** — given the few-shot prompt and question but no haystack at all, the model answers none of them, so containment scoring is not measuring guesswork. And distractors are drawn only from articles no gold paragraph belongs to, since SQuAD groups paragraphs by topic and a same-article distractor could restate the answer.

Qwen3-1.7B, 8192 tokens, 8 questions per stream, 3 seeds. Entries are `hits/trials (mean F1)`:

| policy | B | near | middle | far | total |
|---|---|---|---|---|---|
| `full` | 8192 | 9/9 (0.99) | **6/8 (0.71)** | 4/7 (0.67) | 19/24 |
| **oracle** | 2048 | 9/9 (0.99) | **5/8 (0.66)** | 4/7 (0.67) | **18/24** |
| oracle | 512 | 8/9 (0.88) | 1/8 (0.17) | 0/7 (0.04) | 9/24 |
| `recent` | 2048 | **7/9 (0.78)** | 0/8 (0.06) | **0/7 (0.18)** | 7/24 |
| `h2o` | 2048 | **1/9 (0.20)** | 1/8 (0.20) | **3/7 (0.58)** | 5/24 |
| `snapkv` | 2048 | 5/9 (0.55) | 0/8 (0.06) | 0/7 (0.15) | 5/24 |
| `random` | 2048 | 5/9 (0.56) | 0/8 (0.06) | 0/7 (0.18) | 5/24 |

Three findings carry over intact.

**The dead zone.** Middle band across all four heuristics and both budgets: **1/64**, exact binomial 95% upper **7.2%**, against an uncompressed 6/8. It is not the clean zero the needle task gave, and we state it as near-zero rather than zero. The single hit is `h2o` at d=4729, gold `rivers`, generated `certain rivers` — and the closed-book control confirms it is a retrieval, not a guess.

**The complementarity.** At B=2048, `recent` scores 7/9 near and 0/7 far; `h2o` scores 1/9 near and 3/7 far. The two families remain near-exact complements across a change of task, corpus, query form and metric.

**The oracle.** 18/24 against the uncompressed 19/24, matching band for band, at a quarter of the context — while the best heuristic at the same budget manages 7/24. The gap between the oracle and the heuristics is far larger than the gap between the oracle and no compression at all.

One caveat is specific to this task: **the far band has a difficulty ceiling**, since `full` itself scores only 4/7 there. Far-band zeros are therefore much less informative here than on needles, where the ceiling was 1.00. The middle band carries the claim: ceiling 6/8, every heuristic at zero.

**This benchmark ran on Qwen3 only, and the attempt to extend it failed.** On SmolLM2-1.7B the same benchmark has a closed-book floor of 3/24 and an uncompressed ceiling of **11/24** — an eight-trial window at n=24. Inside it nothing separates: the oracle scores 9/24 against a best heuristic of 7/24, where on Qwen the same comparison is 18/24 against 7/24. The run therefore cannot distinguish *the oracle does not help SmolLM2* from *SmolLM2 cannot do this task*, and we report it as null information rather than as either a replication or a refutation. The dead zone does point the same way there — heuristic middle band 4/64 against a 4/8 ceiling, both recency policies at 0/8 — but with a 95% upper bound of 13.7%. Details in `docs/phase5-smollm2-qa-findings.md`.

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

The pattern is specific: the oracle is perfect exactly where retention coincides with contiguous recency (near, 9/9) and poor wherever it depends on the flag (middle 2/7, far 1/8). The hypothesis this suggests is **contiguity**: the oracle's cache is spliced — answer spans from position ~2000 adjacent to recency fill from ~7000, with a gap in the position indices — and Qwen3 reads a spliced cache where SmolLM2 does not.

That splits into two claims, and we have now tested one of them.

**Fragment length: tested and false.** Dilating each answer span by a radius `r`, so the same needles sit inside progressively longer contiguous windows at the same budget:

| r | flagged | Qwen3 | SmolLM2 | SmolLM2 near |
|---|---|---|---|---|
| 0 | 125 | 24/24 | 12/24 | **9/9** |
| 16 | 381 | 24/24 | 12/24 | **9/9** |
| 48 | 893 | 24/24 | 14/24 | **9/9** |
| 96 | 1661 | 23/24 | 14/24 | **9/9** |
| 192 | 3197 *(over budget)* | 15/24 | 14/24 | 8/9 |

Under budget SmolLM2 moves from 12/24 to 14/24 — about 0.8 standard errors at n=24. Widening a bare 16-token answer island into a ~200-token block of coherent prose does essentially nothing. If missing sentence-context were the reason, this curve would climb steeply.

**Fragment count: tested, and it is the variable.** Holding the stream, the needles, their distances, the flagged token total (1600) and the recency fill all constant, and varying only how many contiguous runs the retained evidence occupies:

| k | runs | window | SmolLM2 | 95% CI | Qwen3 |
|---|---|---|---|---|---|
| **8** | 8 | 200 | **13/24 = 0.54** | [0.35, 0.72] | 23/24 |
| **4** | 4 | 400 | **22/24 = 0.92** | [0.74, 0.98] | 24/24 |
| 2 | 2 | 800 | 20/24 = 0.83 | [0.64, 0.93] | 24/24 |
| 1 | 1 | 1600 | 19/24 = 0.79 | [0.60, 0.91] | 24/24 |

**Fisher exact, k=8 versus k=4: p = 0.0078.** Recall steps up between 8 fragments and 4 and is flat within noise thereafter. Qwen3 is at ceiling throughout and shows nothing, which is what a control with no headroom should do.

**The effect is confined to the dead zone.** SmolLM2's near band is 9/9, 11/11, 8/8, 13/13 — perfect at every k, because near-band needles sit inside the recency block with no gap between them and the query. Middle goes 3/7 → 7/8 and far 1/8 → 4/5.

`k` and window width co-vary by construction, but width does not explain this. The radius sweep above varied width at fixed count over a 13× range and moved 0.50 → 0.58; and the trend here is not monotone in width, since k=1 has the widest windows and scores worst of the three low-k conditions.

**So the shortfall is neither missing evidence nor missing local context: it is that the retained set is broken into too many pieces, and SmolLM2 cannot bridge that many jumps in position index while Qwen3 can.** What this does not identify is the mechanism, or whether the threshold generalises past this budget and stream length. Details in `docs/phase6-contiguity-findings.md` and `docs/phase7-fragment-count-findings.md`.

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

**The law transfers to a task it was not fitted on.** On the real-QA benchmark of §3.2 the eight gold paragraphs occupy a mean of 1332 tokens — real prose, not 48 synthetic digit tokens:

| B | flagged | predicted | observed |
|---|---|---|---|
| 2048 | 1332 | 1.00 | 0.75 (against a 0.79 ceiling) |
| 512 | 1332 | **0.38** | **0.38** |

At B=2048 the flagged set fits and the oracle tracks the uncompressed ceiling; at B=512 it overflows and the predicted survival fraction matches the observed rate exactly. This is why the oracle collapses at the smaller budget in §3.2's table — not a failure of the signal, but the budget refusing to hold what the signal correctly identified.

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

Substantial, though the limitation that previously headed this list — no non-synthetic task — has been addressed in §3.2.

- **Two task families, both short-answer.** Synthetic needles and planted-evidence SQuAD QA. The second fixes the corpus, query and metric, but SQuAD is extractive short-answer QA — still far easier than multi-hop reasoning, summarisation, or agent traces, and neither task is a published benchmark run end to end.
- **The real-QA far band has a difficulty ceiling.** `full` scores only 4/7 there, so far-band zeros in §3.2 are much weaker evidence than the middle band's.
- **Small n on real QA.** 24 trials per cell, 8 per band; the middle-band 95% upper bound is 7.2%, not 1%.
- **Two models, one size class.** Qwen3-1.7B and SmolLM2-1.7B-Instruct. The dead zone replicates across both on synthetic needles, but both are ~1.7B; it may still be a small-model artifact, and the oracle result already differs between them. **The real-QA benchmark is effectively single-model**: it discriminates on Qwen3 and does not on SmolLM2, whose uncompressed ceiling on the task is 11/24 against a 3/24 floor.
- **The oracle ceiling does not replicate on SmolLM2 at high fragmentation**, and §7.1 identifies why (too many position-index gaps) but not the mechanism. The threshold between 8 and 4 fragments is measured at one budget, one stream length and one needle count.
- **Simplified baselines.** H2O and SnapKV are reimplementations scored at token granularity across all heads; absolute numbers are not comparable to published ones.
- **Three training runs, no hyperparameter search.** The claim is not that no policy can succeed — the oracle shows one can on at least one model — but that these three did not, for a reason §7 identifies.
- **The oracle is an upper bound, not a method.** It requires knowing the answers in advance.
- **One query-aware instantiation.** Pooled last-hidden-state cosine. A negative result about one similarity function is weak evidence about query-aware admission generally.
- **One architectural family.** FIFO eviction, fixed per-layer budget, decision shared across layers — forced by the single global attention mask. Per-head and per-layer policies untested.
- **The horizon law is fitted on two admit rates.** `p`=1.0 and `p`=0.941 are close together; the prediction that halving `p` doubles the horizon is untested.

## 9. What comes next

All four items previously listed here have been run: the second-model replication (§3.1), the finer distance sweep (§4, which corrected the mechanism), query-aware admission (§7.3, which failed), and the real-QA benchmark (§3.2, where everything replicated). What remains, in order of value:

1. **Block-structured retention as a method.** §7.1 shows a policy that keeps the right tokens in too many pieces is worth much less than the same budget spent on fewer, larger blocks — and every published heuristic here selects tokens individually. A baseline that snaps selections to contiguous blocks is a cheap, concrete change to test against H2O and SnapKV.
2. **Real QA on a stronger model.** Qwen2.5-3B or Llama-3.1-8B. §3.2's extension to SmolLM2 failed for lack of headroom, not for lack of seeds; the fix is a model that can do the task uncompressed, not more trials.
3. **Vary the admit rate.** One run at `p`=0.5 tests the horizon law's only untested prediction.
4. **Retrieval over evicted tokens**, rather than write-time query scoring. §7.3 rules out the cheap version of query-awareness, not the direction.

## 10. Related work

The occupied cells around this work are worth stating precisely. Write-time admission with a frozen backbone exists (WG-KV, arXiv 2512.17452) via differentiable gating and supervised distillation. RL for cache decisions with a frozen LLM exists (Learning to Evict, ICML 2026) for post-hoc eviction, episodic, uniform budgets — and its ablation already establishes RL over supervised in that setting. Online Lagrangian dual ascent for KV allocation exists (TriRoute, arXiv 2607.06601). Learned eviction for hybrid linear/full attention exists (arXiv 2510.20787).

**What is new here is the measurement, not the method.** We are not aware of prior work reporting recall by distance-band, and the near/far complementarity and mid-context dead zone are not documented elsewhere.

## Reproducibility

Code, seeds, raw per-trial results and all three training histories: this repository. `docs/phase1-findings.md`, `docs/phase2-findings.md`, `docs/phase2b-findings.md`, `docs/phase3-batch-findings.md` `docs/phase4-benchmark-findings.md` `docs/phase5-smollm2-qa-findings.md`, `docs/phase6-contiguity-findings.md` and `docs/phase7-fragment-count-findings.md` record each run including the errors made along the way. 123 tests, harness verified bit-identical against a stock cache on two backends.
