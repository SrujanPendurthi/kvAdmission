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
