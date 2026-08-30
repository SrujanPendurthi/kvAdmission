# Phase 3 — Batch of Four Controls: Replication, Horizon, Query-Awareness

**Date:** 2026-08-30
**Runs:** `results/gpu-smollm2-sweep.json`, `gpu-smollm2-oracle.json`, `gpu-distance-curve.json`, `gpu-queryaware.json`, `gpu-queryaware-512.json`
**Hardware:** RTX 3090 via vast.ai, 5.1 h useful of 17.2 h billed, $3.08 ($2.15 of it idle — see Cost)
**Verdict:** the dead zone replicates; the §4 mechanism was wrong and is corrected; the oracle result is weaker on a second model than on the first.

---

## 1. Second model: the dead zone replicates

SmolLM2-1.7B-Instruct, same harness, 3 seeds × 8 needles × 2 budgets × 4 compressed policies.

| policy | B | near | middle | far |
|---|---|---|---|---|
| `full` | 8192 | 9/9 | **7/7** | 7/8 |
| `recent` | 2048 | 6/9 | **0/7** | 0/8 |
| `random` | 2048 | 4/9 | **0/7** | 0/8 |
| `snapkv` | 2048 | 3/9 | **0/7** | 0/8 |
| `h2o` | 2048 | 0/9 | **0/7** | 0/8 |
| all four | 512 | 3/36 | **0/28** | 0/32 |

**Middle band: 0/56 across every compressed cell** (95% upper bound 5.4%), against an uncompressed ceiling of 7/7. The dead zone is not a Qwen artifact.

One difference from Qwen3-1.7B: `h2o` scores 0/8 in the far band here, where on Qwen it scored 8/14. SmolLM2's attention mass is distributed differently enough that accumulated-attention scoring does not reliably retain the opening of the stream. It does not rescue the middle on either model.

## 2. Distance curve: the mechanism in §4 was wrong

192 trials, 16 needles per stream, 4 seeds, B=2048, chunk=128, 256-token resolution.

| policy | d < 2048 | d ≥ 2048 |
|---|---|---|
| `recent` | **16/16** | **0/48** |
| `random` | 13/16 | **0/48** |
| `h2o` | 0/16 | 10/48 (all at d > 6600) |

Bin by bin, `recent` is 1.00 in every bin below 2048 and 0.00 in every bin above it. **This is a step function at `d = B`, not the geometric decay §4 claimed.**

### What was wrong

§4 modelled a token at distance `d` as surviving `d/chunk` independent rounds, each retaining `B/(B+chunk)`, giving `P = (B/(B+chunk))^(d/chunk)`. That is the survival law for **random eviction**. The harness uses **FIFO** eviction, which does not thin the cache uniformly — it removes the oldest tokens deterministically. The rounds are not independent, and the model does not apply.

The geometric law predicts `random` survival of 0.38 at d=2048 and 0.16 in the middle band. Observed: **0/48 for every d ≥ 2048.** It over-predicts everywhere past the horizon. Its apparent agreement in §4's table came from coarse tercile bands that straddled the horizon; band-level averaging hid a step.

### The correct law

Under admission at rate `p` followed by FIFO eviction to budget `B`, occupancy saturates at `B` and the cache is a contiguous window over the most recent `B/p` raw positions, thinned to density `p`:

> **P(token at distance d is resident) = p if d < B/p, else 0**

A **hard horizon at `d = B/p`, with plateau height `p`.**

| policy | p | predicted horizon | predicted plateau | observed |
|---|---|---|---|---|
| `recent` | 1.000 | 2048 | 1.00 | horizon in (2048, 2304]; plateau 16/16 |
| `random` | 0.941 | 2176 | 0.94 (0.69 for a 6-token span) | horizon in (2048, 2304]; plateau 13/16 = 0.81 |

`recent` is predicted exactly. `random`'s plateau of 0.81 falls between the single-token prediction (0.94) and the all-six-tokens-survive prediction (0.94⁶ = 0.69), which is where a partially-redundant span should land. The 256-token bin width cannot separate a horizon of 2048 from one of 2176.

**This is a falsifiable prediction the old law did not make:** halving the admit rate should double the horizon and halve the plateau. Untested; one cheap run.

### Why this matters for the paper's argument

The qualitative conclusion is unchanged and arguably strengthened. The old claim was that distant tokens decay away exponentially, so a policy must fight an exponential. The corrected claim is sharper: **there is a hard horizon, and beyond it recall is exactly zero, not merely unlikely.** Any policy that does not explicitly protect a token past `B/p` loses it with probability one. The middle band sits entirely beyond the horizon at every budget tested, which is the dead zone stated in one line.

`h2o` has a horizon at the opposite end: 10/48 hits, all at `d > 6600`, i.e. a window roughly 1300 tokens wide over the *start* of the stream. Accumulated attention concentrates on the earliest tokens, so H2O's window is anchored to the beginning rather than the end. Two policies, two windows, neither covering the middle.

## 3. Second-model oracle: the ceiling does not fully replicate

Same oracle policy (told which spans carry the answers), SmolLM2, B=2048, 3 seeds.

| | near | middle | far | total |
|---|---|---|---|---|
| `full` (uncompressed) | 9/9 | 7/7 | 7/8 | 23/24 |
| **oracle B=2048** | **9/9** | **2/7** | **1/8** | **12/24** |
| `recent` B=2048 | 6/9 | 0/7 | 0/8 | 6/24 |
| oracle on Qwen3-1.7B | 1.00 | 1.00 | 1.00 | **40/40** |

**This is the most important negative result in the batch.** On Qwen the oracle matched the uncompressed cache exactly. On SmolLM2 it recovers only half the trials, and mid-band recall is 2/7 against a ceiling of 7/7.

It is not nothing: 2/7 is the first non-zero middle-band cell any policy has produced on SmolLM2, and the oracle doubles `recent`'s total. But the clean claim "a 2048-token cache holding the right 2048 tokens answers every query" holds on one model and not the other.

**Leading hypothesis, untested.** The oracle's cache is discontinuous — answer spans from position ~2000 sit adjacent to recency fill from position ~7000, with a gap in the position indices. Qwen3 appears able to read a spliced cache; SmolLM2 appears not to, and degrades in exactly the pattern that predicts: perfect where retention coincides with contiguous recency (near, 9/9), poor where it depends on the flag (middle 2/7, far 1/8). This is a hypothesis fitted to three seeds of data and should be tested directly — e.g. by comparing an oracle that keeps spans against one that keeps a contiguous block of equal size at the same distance.

The honest revision: the capacity claim is **model-dependent**. What the oracle establishes robustly is that the budget is not the binding constraint; what it does not establish is that every model can exploit a non-contiguous retained set.

## 4. Query-aware admission: knowing the question is not enough

Retention scored by maximum cosine similarity between each token's key-side representation and a pooled embedding of the pending queries — a semi-oracle, since a streaming policy would not have the queries at write time. Qwen3-1.7B, 5 seeds.

| B | near | middle | far |
|---|---|---|---|
| 2048 | 0/14 | **0/12** | 5/14 |
| 512 | 0/14 | **0/12** | 0/14 |
| `recent` 2048 (reference) | 10/14 | 0/12 | 0/14 |

**Zero in the middle band at both budgets**, and it destroys the near band that recency gets for free. Query-aware admission inverts to a far-band bias, behaving like `h2o` rather than like the oracle.

This is a genuinely informative failure. It sits between the heuristics (no query access) and the oracle (exact answer spans) and lands on the heuristic side. **The information the oracle has is not "what is being asked" — it is "which tokens answer it."** Query text plus embedding similarity does not recover the second from the first, at least not with pooled last-hidden-state cosine.

§7's first recommendation was query-aware admission, on the reasoning that it "converts an unanswerable prediction problem into a retrieval one." It converts it into a *retrieval* problem, and this retrieval implementation fails. That does not condemn the direction — a proper retrieval index over evicted tokens is a different mechanism from cosine reranking at write time — but the cheap version does not work, and §7 should stop implying it would.

## 5. What this batch closes and opens

Closes three of §9's four items:

- **Second model** — done, dead zone replicates (§1)
- **Finer distance resolution** — done, and it corrected the mechanism (§2)
- **Query-aware admission** — done, and it failed (§4)

Opens two new ones:

- **The oracle ceiling is model-dependent** (§3). Was previously believed settled.
- **Contiguity may matter as much as content.** If the SmolLM2 hypothesis holds, a retained set's *shape* is a variable no prior work in this line varies.

Still outstanding: a real benchmark (RULER / LongBench-v2). It is now the only §8 limitation with nothing measured against it.

## Cost

$3.08 for 17.2 h billed against 5.1 h of useful compute. The batch finished at 02:06 and the box was destroyed at 14:16 — roughly 12 h of pure idle billing.

The self-stop script calls `poweroff` inside the container. **That does not stop vast.ai billing** — the container halts, the instance stays rented. The same thing happened on the earlier $2.63 overrun and the fix was known: only `vastai destroy instance <id>` from outside stops the meter, and the API key deliberately does not live on the rented box.

The correct mechanism is a local watchdog that polls and destroys, which was available and not used. Anything queued unattended in future needs one; the in-container `poweroff` is a convenience, not a kill switch, and should be labelled as such in `run_batch_selfstop.sh`.
