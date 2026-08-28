# Phase 2 Signal Check — Does Dense KL Carry Long-Range Signal?

**Date:** 2026-08-27
**Run:** `results/signal-check.log` — Qwen3-1.7B, 8192-token code streams, 3 budgets × 3 seeds
**Plan:** `docs/superpowers/plans/2026-08-27-kv-admission-phase2-policy.md` Task 2
**Verdict: PASS.** Proceed to Task 3.

---

## The question

Spec §4.3's reward is `−KL(p_full ‖ p_compressed)` on next-token distributions. That reward can only teach mid-context retention if removing distant context measurably changes what the model predicts.

If it does not — if prediction is locally determined everywhere — then KL is flat with respect to distant tokens, the policy gets no gradient telling it to retain anything far back, and it converges to recency. Which is precisely the failure phase 1 documented. The trainer would then be an expensive way to reproduce `recent`.

## Method

Training streams are concatenated local Python source (stdlib + site-packages), never the Pride and Prejudice eval haystack and never needle-formatted text.

A **long-dependency position** is one whose next token repeats an identifier (≥4 chars, `str.isidentifier()`) last seen at least 2048 tokens earlier. Code makes these mechanically detectable — a class defined at token 200 and referenced at token 6000 is a real dependency, not an assumed one.

For each budget, the stream is prefilled twice in lockstep: a reference cache and a FIFO-compressed cache. Per-position `KL(p_full ‖ p_compressed)` is then averaged separately over long-dependency positions and over all other positions past the warmup point.

## Results

| B | seed | KL at long-dep | KL baseline | ratio | n |
|---|---|---|---|---|---|
| 512 | 0 | 0.688 | 0.232 | 2.96 | 68 |
| 512 | 1 | 2.096 | 0.389 | 5.39 | 174 |
| 512 | 2 | 0.470 | 0.228 | 2.06 | 121 |
| 1024 | 0 | 0.680 | 0.135 | 5.04 | 68 |
| 1024 | 1 | 2.008 | 0.184 | 10.93 | 174 |
| 1024 | 2 | 0.462 | 0.152 | 3.05 | 121 |
| 2048 | 0 | 0.674 | 0.040 | 16.69 | 68 |
| 2048 | 1 | 2.011 | 0.137 | 14.70 | 174 |
| 2048 | 2 | 0.409 | 0.068 | 6.04 | 121 |

Aggregated:

| B | mean ratio | range | mean KL long | mean KL base |
|---|---|---|---|---|
| 512 | 3.47 | 2.06–5.39 | 1.084 | 0.283 |
| 1024 | 6.34 | 3.05–10.93 | 1.050 | 0.157 |
| 2048 | **12.48** | 6.04–16.69 | 1.031 | 0.082 |

**Gate: ratio ≥1.5 required — minimum observed 2.06. PASS.**
**Gate: n_long ≥10 required — minimum observed 68. PASS.**

## The interesting part

I expected the ratio to *compress* at looser budgets: less eviction, less divergence everywhere, contrast washing out. **The opposite happened — the ratio rises steeply with budget, from 3.5 to 12.5.**

The two columns explain why:

- **`kl_base` collapses** as budget grows: 0.283 → 0.157 → 0.082. More cache fixes local prediction, as expected.
- **`kl_long` barely moves**: 1.084 → 1.050 → 1.031. More cache does **not** fix long-dependency positions.

Under FIFO admission, anything further back than `B` is gone regardless of how large `B` is. At 8192 tokens with B=2048, three quarters of the stream is unreachable, and quadrupling the budget from 512 does essentially nothing for it.

**This is phase 1's dead zone, measured a completely different way.** Phase 1 found it behaviorally — zero needle recall in the middle band at every budget. This finds it distributionally — KL at long-dependency positions is flat in budget. Two independent measurements, one conclusion: budget is not the binding constraint, retention policy is.

## What this means for phase 2

Better than "the reward works." **The reward's signal-to-noise improves at exactly the budgets we care about.** At B=2048 the contrast is 12.5×, so the gradient is dominated by long-dependency positions — the reward naturally concentrates on the failure the policy exists to fix, without any reweighting.

That removes the option I was holding in reserve (reweighting the reward toward high-KL positions). It is unnecessary; the budget does it for free.

## Caveats

- **Seed 1 is an outlier in magnitude** (kl_long ≈ 2.0 versus ≈0.5 for seeds 0 and 2) and has the most long-dependency positions (174). Different files, different identifier density. The *ratio* holds across all three, but absolute KL varies by ~4× between streams, so training batches should mix seeds rather than train on one stream.
- **The identifier heuristic is crude.** `str.isidentifier()` on detokenized pieces catches Python names but also keywords and common English words appearing in comments and docstrings. It is a proxy for "long-range dependency", not a parser. It is good enough to answer the gate question; it is not a ground-truth label to train against.
- **Correlation, not causation.** High KL at these positions shows the compressed model diverges there. It does not prove that admitting the specific referenced token would fix it. Task 7's evaluation is what tests that.
- **Code only.** Whether the same contrast exists in prose, dialogue, or agent traces is untested. Generalization is a phase 4 question.

## Decision

**Proceed to Task 3.** Spec §4.3's reward stands unchanged. Tasks 3–7 can now be written as originally specified: `AdmissionPolicy` head, Lagrangian λ controller, average-reward policy gradient, training loop, evaluation against the phase 1 baselines.
