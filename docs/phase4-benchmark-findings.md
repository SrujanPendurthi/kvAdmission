# Phase 4 — The Dead Zone on Real QA

**Date:** 2026-08-30
**Runs:** `results/gpu-benchmark.json` (264 rows), `results/closed-book.json` (24 rows)
**Hardware:** RTX 3090 via vast.ai, ~25 min, **$0.14**
**Verdict:** the dead zone, the near/far complementarity, the oracle result and the crowding law all replicate on a non-synthetic task.

---

## What changed

Every earlier result measured recall of a planted 6-digit code in Pride and Prejudice, scored by exact substring match. That was §8's load-bearing limitation: the easiest possible case for a cache, least like real long-context work, and a metric that scores a paraphrase as a miss.

This run replaces the task while keeping the one axis everything depends on:

| | needle task | this benchmark |
|---|---|---|
| haystack | 19th-century fiction | distractor SQuAD paragraphs |
| evidence | `The secret code for X is 481923` | a real SQuAD paragraph |
| query | templated | human-written question |
| answer | 6 digits | natural-language span |
| metric | exact substring | containment + token F1 |
| evidence position | known | known (planted at a controlled anchor) |

The last row is why this is RULER-style planted QA rather than LongBench-v2. LongBench-v2 is genuinely non-synthetic but ships no evidence spans, and recall *against distance* is the entire measurement here.

## Controls

**Closed-book floor: 0/24, mean F1 0.046.** With no haystack at all — few-shot prompt and question only — the model answers none of the 24 questions. Containment scoring is generous to short common-word golds, so this control decides whether a "hit" is a retrieval or a guess. It is a retrieval.

**Distractor isolation.** SQuAD groups paragraphs by article, so a same-article distractor can restate the answer. Distractors are drawn only from articles no gold paragraph belongs to; without this, a hit would prove nothing about whether the planted evidence survived compression.

## Result

Qwen3-1.7B, 8192-token streams, 8 questions each, 3 seeds. Cell entries are `hits/trials (mean F1)`.

| policy | B | near | middle | far | total |
|---|---|---|---|---|---|
| `full` | 8192 | 9/9 (0.99) | **6/8 (0.71)** | 4/7 (0.67) | 19/24 |
| **oracle** | 2048 | 9/9 (0.99) | **5/8 (0.66)** | 4/7 (0.67) | **18/24** |
| oracle | 512 | 8/9 (0.88) | 1/8 (0.17) | 0/7 (0.04) | 9/24 |
| `recent` | 2048 | **7/9 (0.78)** | 0/8 (0.06) | **0/7 (0.18)** | 7/24 |
| `recent` | 512 | 2/9 (0.28) | 0/8 (0.06) | 0/7 (0.12) | 2/24 |
| `h2o` | 2048 | **1/9 (0.20)** | 1/8 (0.20) | **3/7 (0.58)** | 5/24 |
| `h2o` | 512 | 1/9 (0.16) | 0/8 (0.04) | 1/7 (0.20) | 2/24 |
| `snapkv` | 2048 | 5/9 (0.55) | 0/8 (0.06) | 0/7 (0.15) | 5/24 |
| `snapkv` | 512 | 2/9 (0.27) | 0/8 (0.00) | 0/7 (0.12) | 2/24 |
| `random` | 2048 | 5/9 (0.56) | 0/8 (0.06) | 0/7 (0.18) | 5/24 |
| `random` | 512 | 4/9 (0.50) | 0/8 (0.00) | 0/7 (0.12) | 4/24 |

### 1. The dead zone replicates

**Middle band across all four heuristics and both budgets: 1/64** (exact binomial 95% upper **7.2%**), against an uncompressed 6/8 and a closed-book floor of 0/24.

It is not the clean zero the needle task produced, and the difference should be stated rather than smoothed over. The one hit is `h2o` at B=2048, seed 0, d=4729 — gold `rivers`, generated `certain rivers`. The closed-book control says this is a real retrieval: the model does not produce `rivers` from the question alone. So the honest claim on real QA is **near-zero, not zero**.

### 2. The near/far complementarity replicates

At B=2048 the two families remain near-exact complements:

- `recent`: near **7/9**, far **0/7**
- `h2o`: near **1/9**, far **3/7**

This was the paper's headline observation on synthetic needles. It survives the change of task, corpus, query form and metric.

### 3. The oracle nearly matches the uncompressed cache

At B=2048 — a quarter of the context — the oracle scores **18/24 against the uncompressed 19/24**, and matches band for band: 9/9 near (identical), 5/8 middle (against 6/8), 4/7 far (identical). Mean F1 0.786 against 0.803.

The best heuristic at the same budget scores 7/24. **The gap between the oracle and the heuristics is far larger than the gap between the oracle and no compression at all** — which is the observability claim, restated on a real task.

Note this is the *Qwen* result replicating. Whether the SmolLM2 shortfall (`docs/phase3-batch-findings.md` §3) also appears on real QA is untested.

### 4. The crowding law transfers

The oracle collapses at B=512, and the law fitted on synthetic needles predicts it. The eight gold paragraphs occupy a mean of **1332 tokens**:

> **recall = min( 1, B / |flagged| )**

| B | flagged | predicted | observed |
|---|---|---|---|
| 2048 | 1332 | 1.00 | 0.75 (ceiling 0.79) |
| 512 | 1332 | **0.38** | **0.38** |

At B=2048 the flagged set fits and the oracle tracks the uncompressed ceiling. At B=512 it overflows, and the predicted survival fraction of 0.38 matches the observed rate exactly. This is the law validated on a task it was not fitted on, where the flagged set is 1332 real paragraph tokens rather than 48 synthetic digit tokens.

## What this does not show

- **The far band has a task-difficulty ceiling.** `full` itself scores only 4/7 there, so far-band zeros are much less informative here than on needles, where the ceiling was 1.00. The middle band is the load-bearing cell: the ceiling is 6/8 and every heuristic is at zero.
- **One model.** Qwen3-1.7B. The SmolLM2 replication was run on needles, not on this.
- **Small n.** 24 trials per cell, 8 per band. The 95% upper bound on the middle band is 7.2%, not 1%.
- **SQuAD is short-answer extractive QA.** Harder than a digit string, still far easier than multi-hop reasoning or agent traces.
- **Containment is generous.** F1 is reported alongside every cell for this reason, and the closed-book control bounds the guessing rate at zero for *these* questions, not in general.

## Process notes

Two bugs were caught locally before spending anything, and one was not.

**Caught.** A first ceiling run scored 1/4. The generations were substantively correct — "The Mongol armies would divert certain rivers…" against gold `rivers` — but token F1 punishes a sentence answered against a one-word gold, scoring 0.125. The fix was an in-context example (probed against three prompt forms: 4/6 and F1 0.61, versus 3/6 and F1 0.21 for a bare instruction) plus containment as the primary criterion, which is what the needle task already used. Also `_normalize` dropped unicode dashes, scoring an otherwise exact `100–5,000 hp` match as zero.

**Not caught.** `make_oracle_decider` returns `(decide, need_attn)`; the sweep script assigned the whole tuple and died with `'tuple' object is not callable` on the first oracle cell of a paid run, losing seeds 1–2. A one-line test now asserts the return contract. Worse, the run script echoed `ALL_DONE` unconditionally, so the watchdog would have destroyed the box as if the work had finished; the signal is now gated on exit code.

**Cost: $0.14**, against $3.08 for the previous batch. The difference is entirely the local watchdog that polls for completion and calls `vastai destroy` — the in-container `poweroff` never stopped billing.
