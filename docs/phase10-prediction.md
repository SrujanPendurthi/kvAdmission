# Phase 10 — Pre-registered Predictions for the Surprisal End-to-End Run

**Written before the run.** Committed before the GPU was rented, so the
screening method in `docs/phase9-surprisal-findings.md` is tested rather than
fitted.

## What is being tested

Two things at once:

1. **Does surprisal admission fail end-to-end?** Phase 9 says yes, from a
   coverage diagnostic alone.
2. **Is the coverage diagnostic a valid screening tool?** That is the more
   useful claim, and it is only supported if the predictions below hold.

## The screening argument

§7.2's partial-oracle ablation measured recall against coverage over answer
tokens: 1.00 coverage gave 7/7 in the middle band, 0.75 gave 1/7, and 0.50 and
below gave 0/7. Phase 9 measured surprisal's coverage:

| task | coverage @ B=2048 | spans fully retained |
|---|---|---|
| needle | 0.107 | 0/16 |
| real QA | 0.248 | 2/16 |

Both are far below the 0.75 point at which the middle band had already
collapsed to 1/7.

## Predictions

**P1 — Needle, B=2048: middle-band recall 0/7 or 0/8.** Coverage 0.107 is
below every point on the partial-oracle curve that produced a non-zero cell.

**P2 — Needle, B=2048: near-band recall also near zero, below `recent`'s
10/14.** Surprisal carries no recency term (`recency_weight=0`), so unlike
every heuristic baseline it has no positional fallback. This is the sharper
prediction: the failure should be uniform across bands rather than
concentrated away from the query.

**P3 — Needle overall: ≤ 2/24 hits.** A whole 6-token span must survive, and
0/16 spans were fully retained in the diagnostic.

**P4 — Real QA, B=2048: overall ≤ 4/24, and middle band 0/8.** Coverage 0.248
is at chance, and 2/16 answer phrases were fully retained.

**P5 — Surprisal will not beat `recent` on either task.**

## What would falsify the screening method

Any of: needle middle band non-zero; needle overall above 4/24; QA overall
above 8/24. If surprisal performs materially better than the coverage
diagnostic predicts, then coverage is not sufficient to screen a candidate
signal and phase 9's method note should be withdrawn.

## What would falsify it in the other direction

If surprisal scores *worse* than predicted — 0/24 everywhere on both tasks —
the screening method is conservative but sound, and the negative result stands
more strongly.

---

## Outcome — appended after the run

Box `49342091`, RTX 3090, ~12 min, **$0.15**. 129 tests passed on CUDA first.

**All five predictions hold. None of the falsification thresholds was crossed.**

| | prediction | observed | |
|---|---|---|---|
| P1 | needle middle = 0 | **0/7** | holds |
| P2 | needle near ≪ `recent`'s 10/14 | **0/9** | holds |
| P3 | needle overall ≤ 2/24 | **0/24** | holds |
| P4 | QA overall ≤ 4/24, middle = 0 | **2/24**, middle **0/8** | holds |
| P5 | beats `recent` on neither | 0/24 and 2/24 vs 7/24 | holds |

At B=2048:

| policy | task | near | middle | far | total |
|---|---|---|---|---|---|
| surprisal | needle | **0/9** | **0/7** | **0/8** | **0/24** |
| surprisal | real QA | 1/9 | **0/8** | 1/7 | 2/24 |
| `recent` | needle | 10/14 | 0/12 | 0/14 | 7/24 |

## What this establishes

**The coverage diagnostic is a valid screen.** Phase 9 rejected surprisal from a $0 local measurement and predicted the end-to-end numbers before the box existed. The predictions were conservative — the actual result is at or below every bound — which is the safe direction for a screening tool: it will not wave through a signal that then fails, though it may in principle reject one that would have worked.

That claim is worth more than this particular negative result. §7.2 is no longer only a description of how the oracle degrades; it is a **test any candidate write-time signal must pass before anyone spends GPU time on it.** Measure the fraction of answer spans landing inside the signal's top `B`, compare against the partial-oracle curve, and reject anything short of ~1.00 fully retained.

**Surprisal is the first policy tested here that is worse than every heuristic everywhere.** `recent` owns the near band, `h2o` the far band; surprisal scores zero in all three on needles. With `recency_weight=0` it has no positional fallback, and the result is the cleanest possible separation of *selection* from *geometry*: the budget, the chunking and the eviction path are identical to every baseline, and the only thing changed is which tokens are chosen.

## Remaining caveats

- Two seeds for the diagnostic, three for the run, one model.
- The screening claim is supported by one successful prediction. A signal that *passes* the screen and then succeeds end-to-end would test the other direction, and nothing here provides one.
- `recency_weight` remains untested. A surprisal/recency blend would likely recover the near band, but that is recency doing the work, not surprisal.
