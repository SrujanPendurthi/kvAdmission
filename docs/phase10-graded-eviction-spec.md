# Phase 10: graded eviction and root admission

Status: design, not started. Budget: ~$50 of rented GPU (about 100 RTX 4090
hours at ~$0.45/h, see `scripts/rent.md`). The M4 is used for everything that
fits on it.

## Goal

Replace FIFO eviction with a *graded* one: a low-priority entry loses
precision step by step and leaves the cache only when its contribution to
the model's output is negligible. Then replace blind admission with signals
that try to keep the "roots" of the stream. Each piece is tested separately
against FIFO and the oracle, at equal memory in bytes.

## What the paper already tells us

- No write-time signal among the ten tested predicts which spans a future
  question needs (`paper/paper.tex`, the screen). Every admission idea here
  must pass the same screen (`kvadmission/screen.py`) before any GPU is spent
  on it.
- Recall is set by the retention horizon d = B/p. **Quantisation alone moves
  B**: int4 fits 4x the tokens of fp16 in the same bytes. Graded eviction can
  look good just because it stores more tokens per byte, whatever the
  priority rule is. So the headline baseline is *uniform* quantisation plus
  FIFO at equal bytes (KIVI-style), not fp16 FIFO.
- The KL probe (predicting the last 64 tokens) is a local signal, like the
  recency-type signals that failed. We expect it to rank middle-band evidence
  as unimportant. The graded ladder's value is that such evidence decays to a
  cheap level instead of vanishing.

## Eviction design

Each resident entry (token, layer) has a level:

```
fp16 -> int8 -> int4 -> int2 -> merged/low-rank -> evicted
```

The budget is in bytes per layer, not tokens.

**Priority.** An entry's priority is the KL divergence it causes if taken
one step down:

```
dKL(i) = KL( p(probe | cache) || p(probe | cache with i one step down) )
```

Here `probe` is the last 64 stream tokens, scored with teacher forcing.

**Gate** (per-entry step):

- If `dKL(i) < eps`, step entry i down even when under budget. This is free
  memory.
- If `dKL(evict i) < eps`, evict it.
- If still over budget, apply the cheapest remaining steps, lowest dKL first,
  even above eps, until the budget holds. Without this fallback, a
  per-entry threshold cannot promise a memory ceiling.

**Cost control.** An exact dKL costs one probe forward per candidate, which
is too many per token. Two ways to cut it:

1. Exact KL at chunk granularity: step a whole 128-token chunk and do one
   probe forward. At 8k tokens that is ~64 forwards per decision round.
2. Within a chunk, rank tokens with a first-order proxy: the attention mass
   the probe queries put on the entry, times the norm of its quantisation
   error on the value. Check the proxy against exact KL on a sample
   (rank correlation) before trusting it.

**Storage.** Stage 1 simulates quantisation: quantise and dequantise in
place, and count bytes by level. That measures recall honestly, but not
speed. Real packed storage waits until recall justifies it.

## Admission design (after eviction works)

Three separate signals, in this order, each screened first:

1. **Hub tokens.** A token is kept if later tokens in its own and the next
   chunk give it high attention. Cheap. The H2O-type signal already failed
   under FIFO. It is retested here because under graded eviction a false
   reject costs precision instead of the token.
2. **Learned scorer.** A small head on the KV at write time, trained to
   predict the oracle flags (`kvadmission/oracle.py:oracle_flags`) from
   the needle and SQuAD streams, and evaluated on held-out seeds and a
   held-out task. Extends `kvadmission/policy.py` / `train.py`.
3. **Summary tokens.** Each chunk is compressed into k gist tokens, which
   needs fine-tuning. Probably exceeds $50 at 1.7B. Do it only if (1) or (2)
   leaves budget, else make it the next phase.

## Stages

| # | Work | Where | GPU h |
|---|---|---|---|
| 0 | Mixed-level cache, byte accounting, quantise ladder, probe KL; tests | M4 | 0 |
| 1 | Quantise ladder + KL gate, 1.7B, needle + QA, 5 seeds, 3 byte budgets | M4 then 4090 | ~15 |
| 2 | Add merge + low-rank rungs; proxy vs exact KL check | 4090 | ~15 |
| 3 | Hub admission: screen, then end to end | M4 + 4090 | ~10 |
| 4 | Learned scorer: gather flags, train, screen, end to end | 4090 | ~30 |
| 5 | Best config on Qwen3-8B, RULER subset at 32k | 4090/A100 | ~25 |
| - | Summary tokens | deferred | - |

## Baselines (all at equal bytes)

- fp16 FIFO: today's baseline.
- Uniform int8 / int4 / int2 FIFO. This is the key control.
- Uniform int4 plus H2O-style score eviction.
- Oracle retention (the upper bound), at fp16 and at int4.

## Success criteria

- Graded eviction must beat uniform-quantisation FIFO at equal bytes on
  middle-band recall. The Wilson intervals must not overlap, or a Fisher
  test must give p < 0.05. Beating only fp16 FIFO does not count.
- An admission signal must pass the screen, meaning it keeps more answer
  spans whole than random with non-overlapping intervals, before its end-to-end
  run.

## Kill criteria

- If the KL gate adds nothing over uniform int4 FIFO after stage 1, stop the
  priority work and report the negative result. The finding would then be
  "bytes, not priority".
- Stop at $50 even if a stage is unfinished, and record where it stopped.

## Stage 1 result (M4, 2026-09-21)

`scripts/run_graded.py`, results in `results/graded.json`. Needle task, Qwen3-1.7B,
8k tokens, 3 seeds x 8 needles. Budgets are equal in bits (fp16-token equivalents).
Keys are quantised per channel. With per-token int2 keys, every needle decoded
as "that that that", 0/8 (seed 0, B=1024).

| config | B=512 hits | tokens kept | B=1024 hits | middle band | tokens kept |
|---|---|---|---|---|---|
| fp16 FIFO | 1/24 | 512 | 3/24 | 0/7 | 1024 |
| uniform int8 | 3/24 | 900 | 6/24 | 0/7 | 1796 |
| uniform int4 | 4/24 | 1540 | 9/24 | 0/7 | 3204 |
| **uniform int2** | **8/24** | 2692 | **14/24** | **7/7** | 5380 |
| age ladder | 5/24 | 1796 | 12/24 | 4/7 | 4484 |
| KL ladder | 6/24 | 1711 | 12/24 | 3/7 | 3587 |

Recall follows the number of tokens kept, not the priority rule. The KL ladder
spends bits keeping recent chunks at fp16/int4, because the probe (the
incoming text) is local and rates old chunks as cheap to evict. That is the
paper's write-time failure again. Uniform int2 hits every middle-band needle at
B=1024; the ladders do not. Intervals overlap at n=24 (e.g. uniform int2
14/24 [0.39, 0.76] vs KL 12/24 [0.31, 0.69]), but no cell shows KL ahead of
uniform int2.

**Kill criterion met: bytes, not priority.** The KL gate is dropped. What
the data does support is packing more tokens per byte, which extends the
retention horizon. Next test: rungs denser than int2 (merge, low-rank),
applied uniformly.
