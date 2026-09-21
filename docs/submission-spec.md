# What a KV Cache Retention Signal Must Achieve

### A specification, and a test that rejects candidates for nothing

**Alternative framing of the same evidence.** `docs/submission.md` presents
this work as a measurement paper about the mid-context dead zone. This version
leads with the specification and the screening test, which is what survived a
literature search (§9) most intact. Same experiments, same numbers, different
claim.

---

## Abstract

KV cache compression methods proliferate, each proposing a rule for scoring
which tokens to keep, each evaluated on aggregate accuracy at a compression
ratio. What none of them states is the **target**: what would a retention
signal actually have to achieve to work?

We answer that with an oracle. Given a fixed budget, chunked prefill and FIFO
eviction, a policy *told which tokens carry the answers* recovers **40/40 at
1.7B and 24/24 at 8B**, matching an uncompressed cache on a quarter of the
context, while the best heuristic on the same streams reaches 10/40 and 6/24, and no
heuristic reaches the middle of the context at all. The capacity is therefore
already there. What is missing is a signal.

Degrading that oracle along two axes yields the specification. **Coverage is
unforgiving**: retaining 75% of answer tokens collapses mid-context recall
from 7/7 to 1/7, because most of a six-digit code is worth nothing.
**Precision is free until the budget binds**, after which recall follows
`min(1, B/|flagged|)` — a law that transfers unchanged to a task it was not
fitted on, predicting 0.38 and observing 0.38.

The specification is then usable as a **screening test**. Any candidate signal
can be rejected by measuring what fraction of answer spans falls inside its
top `B`, with no policy implemented and no GPU rented. We validate this once —
rejecting surprisal for $0, pre-registering five predictions, and confirming
all five — and then apply it to **ten candidate signals across eight
independent framings**: learned write-time features, query similarity,
information content, attention received, key norm, lexical rarity, and
answer-form priors. None passes. The closest, a prior that flags digits on a
task whose answers *are* digits, reaches 2.2× chance and still retains 0 of 16
spans whole.

We give the mechanism for why position-based rules cannot clear the bar
(§6), and isolate a second structural constraint the specification does not
capture: at matched evidence, distance and budget, **fragmenting the retained
set destroys recall** (0.54 at eight pieces against 0.92 at four, p=0.008).

## 1. The missing target

A long-context model writes every token into its KV cache; under a budget most
must go. The literature decides what to discard by ranking tokens —
accumulated attention (H2O), observation-window voting (SnapKV), recency plus
sinks (StreamingLLM), layer-wise allocation (PyramidKV, Ada-KV) — and reports
aggregate accuracy at a compression ratio.

That evaluation answers "is this rule better than that one." It does not
answer "how good would a rule have to be," and without that number a
proposed signal cannot be assessed except by building it and running it. Every
candidate costs an implementation and a GPU run, and a negative result teaches
nothing transferable.

This paper supplies the missing target and makes it cheap to check against.

**Contributions.**

1. An **oracle control** showing the capacity for mid-context retention
   already exists at two model scales — the failure is observability, not
   budget (§3).
2. A **two-sided specification** for a realisable signal: near-total coverage
   of answer spans, with a flagged set inside the budget. Both halves are
   quantitative laws, one of which transfers to a task it was not fitted on
   (§4).
3. A **screening test** derived from that specification, validated with five
   pre-registered predictions on a candidate rejected for $0 (§5).
4. **Ten candidate signals rejected** across eight framings, with
   the near-miss explaining precisely what the specification demands (§5).
5. A **mechanism** — a hard retention horizon at `d = B/p` — explaining why
   position-based rules cannot clear the bar however they are tuned (§6).
6. A **controlled isolation of fragmentation** as a second constraint the
   specification does not capture (§7).

## 2. Setup

- **Models.** Qwen3-1.7B, SmolLM2-1.7B-Instruct, Qwen3-8B; bf16, frozen. No
  model weights are trained anywhere in this work.
- **Streams.** *Synthetic:* 8192 tokens of prose with eight facts
  `The secret code for <topic> is <6 digits>` at jittered even anchors.
  *Real:* 8192 tokens of distractor SQuAD paragraphs with eight gold
  paragraphs planted at the same anchors, queried by their human-written
  questions, scored by containment and token F1 against a measured
  **closed-book floor of 0/24**.
- **Compression.** Chunked prefill at 128 tokens, compressed to budget `B` per
  layer after each chunk, first four tokens pinned as sinks.
- **Baselines are scoring rules, not systems.** `recent`, `h2o` and `snapkv`
  implement the *token-scoring rule* of StreamingLLM, H2O and SnapKV at token
  granularity under a uniform per-layer budget — no per-head budgets, no layer
  allocation, none of the systems engineering those papers contribute.
  Absolute numbers are not comparable to published ones and nothing here
  benchmarks a published system.
- **Harness.** At unbounded budget, logits are bit-identical to a stock
  `DynamicCache` under the same chunking, verified on MPS and CUDA. 129 tests.

## 3. The capacity already exists

**Oracle retention** uses the identical budget, chunking, sink prefix, FIFO
eviction and evaluation as every baseline, through the same `apply_eviction`
path. One thing changes: it is told which token spans carry the answers.

| policy, B=2048 | model | near | middle | far | total |
|---|---|---|---|---|---|
| `recent` | 1.7B | 10/14 | **0/12** | 0/14 | 10/40 |
| `h2o` | 1.7B | 0/14 | **0/12** | 8/14 | 8/40 |
| **oracle** | 1.7B | **14/14** | **12/12** | **14/14** | **40/40** |
| `recent` | 8B | 6/9 | **0/7** | 0/8 | 6/24 |
| **oracle** | 8B | **9/9** | **7/7** | **8/8** | **24/24** |

**40/40 at Qwen3-1.7B and 24/24 at Qwen3-8B.** A 2048-token cache holding the
*right* 2048 tokens answers every query an uncompressed 8192-token cache does.

The gap this exposes is not between compressed and uncompressed. It is between
the oracle and every heuristic at the *same budget* — 40/40 against 10/40, 24/24 against 6/24. What
separates them is information, not capacity.

**Motivating context: nothing reaches the middle.** Across 340 middle-band
trials on synthetic needles — three models spanning a 4.7× size range, five
scoring rules, three learned policies, four budgets, five seeds — we observe
**zero successes** (95% upper bound 0.9%). On real planted-evidence QA the
figure is 1/64 against an uncompressed 6/8. That compressed caches
under-retain the middle is consistent with prior reports (§9); what makes it
the right motivation here is that the oracle removes it entirely at the same
budget.

## 4. The specification

Degrading the oracle along two axes measures what a realisable signal must
achieve. `coverage` is a proxy's recall over answer tokens; `fp_rate` its
false-positive rate over everything else.

### 4.1 Coverage is unforgiving

| coverage | overall | middle band |
|---|---|---|
| 1.00 | 24/24 | **7/7** |
| 0.75 | 12/24 | **1/7** |
| 0.50 | 8/24 | 0/7 |
| 0.25 | 6/24 | 0/7 |

Missing a quarter of the answer tokens collapses the middle band from 7/7 to
1/7. An answer spans few tokens, and retaining most of a six-digit code is
worth nothing: the digits are mutually unpredictable, so every one must
survive. **A signal that is usually right about which tokens matter is not
good enough.**

### 4.2 Precision is free until the budget binds

Below the point where the flagged set exceeds `B`, false positives cost
nothing — they occupy budget that would otherwise hold recency fill. Above it,
recall follows a second law:

> **recall = min( 1, B / |flagged| )**

| fp_rate | flagged | predicted | observed |
|---|---|---|---|
| 0.05 | 506 | 1.00 | 1.00 |
| 0.20 | 1734 | 1.00 | 1.00 |
| 0.30 | 2554 | 0.80 | 0.75 |
| 0.40 | 3373 | 0.61 | 0.62 |
| 0.50 | 4192 | 0.49 | 0.50 |

Accurate to 0.05 across the range. **It transfers to a task it was not fitted
on**: on real SQuAD QA the eight gold paragraphs occupy 1332 tokens of natural
prose rather than 48 synthetic digit tokens, and at B=512 the law predicts
0.38 against an observed **0.38**.

### 4.3 The specification, stated

> A retention signal must identify **essentially every token** of an answer
> span, and keep its **total flagged set within the budget**. It may be
> entirely indiscriminate about which *extra* tokens it keeps, so long as it
> does not keep too many.

A signal flagging 30% of a stream is worse than useless even with perfect
coverage. A signal with 75% coverage and a perfectly sized flagged set is
worth almost nothing. Both halves bind.

## 5. The specification as a test

The coverage half is measurable **without implementing a policy at all**:
compute the signal over a stream, take its top `B`, and ask what fraction of
answer spans falls inside. Against §4.1's curve, anything short of ~1.00
spans-retained-whole predicts end-to-end failure.

### 5.1 Validating the test

We ran it once as a genuine prediction. Surprisal — score each token by
`−log p(token | prefix)`, keeping what the context cannot reconstruct — is a
principled, free, non-positional candidate. The screen rejected it locally for
**$0**: coverage 0.107 on needles against a 0.250 chance rate, 0.248 on real
QA, with 0/16 spans retained whole on needles and 2/16 on QA.

**Five predictions were then committed to version control before a GPU was
rented.** All five held: needle middle band 0/7, needle near band 0/9 against
`recent`'s 10/14, needle overall 0/24, QA overall 2/24 with middle 0/8, and
beating `recent` on neither task. Every number landed at or below its
predicted bound — the conservative direction for a screen.

### 5.2 Ten candidates, none passing

| signal | framing | needle cov. | QA cov. | spans whole |
|---|---|---|---|---|
| learned admission ×3 rewards | learned features | — | — | 0/12 mid |
| query-aware retention | query similarity | — | — | 0/12 mid |
| `surprisal` | information content | 0.107 | 0.240 | 0/16 |
| `norm_attention` | attention received | 0.230 | 0.290 | ≤2/16 |
| `key_norm` | activation magnitude | 0.384 | 0.244 | 0/16 |
| `rarity` | lexical statistics | 0.102 | 0.265 | 0/16 |
| `answer_shape` | answer form | **0.541** | 0.271 | 0/16 |
| `lookahead` | 128 tokens of future | 0.163 | 0.228 | 0/16 |
| *random control* | — | 0.248 | 0.249 | 0/16 |

Chance is 0.250. QA coverage is over the whole gold paragraph, the span the screener records; over the answer phrase alone `surprisal` scores 0.248 with 2/16 whole. Best whole-span retention across twelve signal-task pairs is
2/16 against the ~1.00 required.

Two entries carry the argument.

**`norm_attention` fails, and it was the best available repair.** Accumulated
attention leans toward the start of the stream because the sum runs over a
variable number of query rows and early tokens are visible to more of them;
dividing by that count is the obvious fix. It lands *below chance* on needles
and barely above on QA. Removing the positional bias does not reveal a utility
signal underneath — it removes the bias that was producing H2O's only wins.

**`answer_shape` is the strongest signal found anywhere here and still
fails.** At 2.2× chance on needles it is doing exactly what it was built to
do: it flags digits, and the answers are digits. It is a prior written with
knowledge of the answer *format*. It retains **0/16 spans whole**, because
half a six-digit code is worth nothing.

That is the specification made concrete. **Knowing the form of the answer is
not enough. §4 demands every token of the specific span answering a query that
has not yet arrived.**

**`lookahead` fails, and it was the hardest test.** Every other candidate must
infer a token's future from its past. This one is given 128 tokens of that
future — the attention a token receives from the immediately following chunk,
implementable by delaying compression a single chunk. It scores **0.163 on
needles and 0.228 on real QA, both below the 0.250 chance rate**, with 0/16
spans retained whole on either.

The explanation is that a chunk's attention to the chunk before it is
dominated by local syntactic continuation: the next 128 tokens attend
backwards for grammatical coherence, not for semantic importance. Short-range
lookahead is therefore a proxy for *very* local recency, which is if anything
anti-correlated with being queried thousands of tokens later.

This is the sharpest form of the claim available here. **The information that
determines a token's retention value is not in its past, and not in its
immediate future either.** It is in a query that has not arrived.


## 6. Why position-based rules cannot clear the bar

Sixteen needles per stream, four seeds, B=2048, chunk=128, binned at 256-token
resolution:

| rule | d < 2048 | d ≥ 2048 |
|---|---|---|
| `recent` | **16/16** | **0/48** |
| `random` | 13/16 | **0/48** |
| `h2o` | 0/16 | 10/48 — all at d > 6600 |

Under admission at rate `p` followed by FIFO eviction to budget `B`, occupancy
saturates and the cache becomes a contiguous window over the most recent `B/p`
raw positions:

> **P(token resident at distance `d`) = `p` if `d < B/p`, else 0**

`recent` (`p`=1) is predicted exactly. Tested at `p`=0.5, where the horizon
should double: **2/32 inside 4096, 0/32 beyond**. The residency law governs *tokens*; recall needs every token of an `L`-token answer, so the in-horizon plateau is `p^L` rather than `p` — hence 2/32 and not 16/32. What the test confirms is the horizon.

![Recall against distance from the query, B=2048. `recent` holds 1.00 out to
the horizon at d=B and is exactly zero past it; `h2o` is the mirror image,
anchored to the start of the stream; neither reaches the ~4000 tokens between
them.](fig-horizon.svg)

**Figure 1.** Recall in 512-token distance bins, 4 seeds × 16 needles,
Qwen3-1.7B at B=2048. The dashed line marks the predicted horizon `d = B/p`.


Beyond the horizon, survival is **exactly zero**, not merely unlikely. Any rule
that does not actively protect a token past `B/p` loses it with certainty, so
recency is not a design choice but the attractor every uninformed rule decays
onto. `h2o` has the mirror-image horizon — its hits are a ~1300-token window
over the stream's *start*, because accumulated attention concentrates there.

**The two rule families are exact complements**: at B=2048, `recent` scores
10/14 near and **0/14** far; `h2o` scores **0/14** near and 8/14 far. Each owns
one end of the context, scores exactly zero at the other, and neither reaches
the middle. Both are functions of position; neither is a function of utility.
No amount of tuning within either family changes that, which is why §5's
search is over signals rather than over hyperparameters.

## 7. A constraint the specification misses

The specification is about *which* tokens are retained. It says nothing about
their arrangement, and arrangement turns out to matter.

The oracle's 40/40 does not replicate on SmolLM2-1.7B, which reaches 12/24.
Holding the stream, needles, distances, flagged token count (1600) and recency
fill all constant, and varying only how many contiguous runs the retained
evidence occupies:

| runs | window | SmolLM2 | 95% CI | Qwen3 |
|---|---|---|---|---|
| **8** | 200 | **0.54** | [0.35, 0.72] | 0.96 |
| **4** | 400 | **0.92** | [0.74, 0.98] | 1.00 |
| 2 | 800 | 0.83 | [0.64, 0.93] | 1.00 |
| 1 | 1600 | 0.79 | [0.60, 0.91] | 1.00 |

**Fisher exact, 8 versus 4: p = 0.0078.** The effect is confined to the dead
zone — SmolLM2's near band is perfect at every setting (9/9, 11/11, 8/8,
13/13), because near-band evidence sits inside the recency block with no gap
to bridge. Width does not explain it: a separate sweep varied window width at
fixed count over a 13× range and moved 0.50 → 0.58, and the trend here is not
monotone in width.

That fragmentation harms compressed caches is known (§9). What this adds is
the controlled attribution to fragment *count*, and the consequence for §4:
**a signal can satisfy the specification exactly and still fail**, if it
delivers the right tokens in too many pieces.

## 8. Limitations

- **The test has only ever said no.** One successful rejection validates it;
  a signal that *passes* and then succeeds would test the other direction, and
  we have none. A screen that rejects everything is consistent with being
  correct and with being uselessly conservative. We attempted to produce a
  positive case with the `lookahead` candidate and it failed the screen too,
  so this limitation stands open.
- **The oracle is an upper bound, not a method.** It requires knowing the
  answers in advance. This paper specifies a target and supplies a test; it
  does not supply a signal that meets the target.
- **Two task families, both short-answer.** Synthetic needles and extractive
  SQuAD QA, neither a published benchmark run end to end. The real-QA
  benchmark discriminates only on Qwen3-1.7B.
- **Three models, 1.7B–8B**, one architecture family plus one outsider.
- **Scoring rules, not systems** (§2).
- **One architectural family**: FIFO eviction, fixed per-layer budget,
  decisions shared across layers — the last forced by a single global
  attention mask.
- **The coverage curve is measured on the synthetic task only**, and the
  screen inherits that. Its transfer to QA rests on the crowding law's
  transfer, which was verified, and not on a second coverage sweep.

## 9. Related work

**Position bias without compression.** Liu et al. (2024, *Lost in the Middle*,
TACL 12:157–173) show a U-shaped accuracy curve over the position of relevant
information in *uncompressed* context. Our `full` baseline scores **1.00 in
every band on all three models**, so at these lengths the middle is answered
perfectly without a budget and the effect we motivate with is created by
compression rather than inherited from the model.

**Position-resolved evaluation of compression.** IndexMem (arXiv 2605.25475)
reports needle-in-a-haystack heatmaps over context length and needle depth and
observes position-dependent failure of SnapKV and PyramidKV. Reporting recall
by position is therefore not new. What §6 adds is the complementarity and the
horizon mechanism.

**Fragmentation.** AMS (arXiv 2605.23200) names *Region Wipe-out* — global
top-k "trigger[s] the severe eviction of contiguous reasoning blocks" — reports
that token-wise selection shows a "lost in the middle" retention pattern, and
proposes region-aware segment quotas as the remedy. §7's phenomenon and its
fix are theirs, in a decoding-time chain-of-thought setting; our contribution
there is the controlled attribution to fragment count.

**Non-uniform damage.** Chen et al. (*The Pitfalls of KV Cache Compression*,
arXiv 2510.00231) evaluate five methods on Llama3-8B and Qwen2.5-14B under
multi-instruction prompting, finding that some instructions degrade far faster
than others. Same theme on a different axis: their unit is an instruction,
ours is a fact at a distance.

**Methods.** Write-time admission with a frozen backbone (*KV Admission*, arXiv
2512.17452); RL for post-hoc eviction (Learning to Evict, ICML 2026); online
Lagrangian KV allocation (TriRoute, arXiv 2607.06601); learned eviction for
hybrid attention (arXiv 2510.20787).

**What is new here** is not that compression hurts, nor where. It is the
**target** — an oracle establishing that the capacity exists, and two laws
saying what a signal must achieve to reach it — and the **test** that turns
that target into a rejection criterion costing nothing to apply.

## Reproducibility

Code, seeds, raw per-trial results and every training history: this
repository. `docs/phase1-findings.md` through `docs/phase10-prediction.md`
record each run including the errors made along the way, and
`docs/submission.md` presents the same evidence under a measurement-first
framing. 129 tests; harness verified bit-identical against a stock cache on
two backends. Total rented GPU cost of every experiment reported here: under
$8.
