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

## Stage 2 result: how far "more tokens per byte" goes (M4, 2026-09-22)

Same harness, adding rungs below int2: int1, and merging 2 or 4 neighbouring
entries into their mean (`uniform3m2` = int2 merged in pairs). Uniform rows were
rerun after two fixes: FIFO now evicts single tokens rather than whole chunks
(at B=256 fp16 kept 132 of 256 tokens before), and the sink prefix is no longer
quantised. 3 seeds x 8 needles per cell.

| config | tokens kept at B=512 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|
| fp16 | 512 | 0/24 | 1/24 | 3/24 |
| int8 | 910 | 1/24 | 3/24 | 6/24 |
| int4 | 1638 | 3/24 | 4/24 | 9/24 |
| **int2** | 2730 | **3/24** | **8/24** | **16/24** |
| int1 | 4096 | 0/24 | 0/24 | 0/24 |
| int4, merged x2 | 1638 entries / 3276 tokens | 0/24 | 0/24 | 1/24 |
| int2, merged x2 | 2730 entries / 5461 tokens | 2/24 | 0/24 | 0/24 |
| int2, merged x4 | 2144 entries / 8192 tokens | 0/24 | 0/24 | 0/24 |

**int2 is the floor.** Below it the cache holds more tokens and remembers
nothing: int1 keeps 4096 tokens at B=512 and scores 0/24, and merging the mean
of neighbouring entries destroys the needle even when the whole stream fits.
Averaging is the wrong operator for exact recall; a merged entry is also not
up-weighted in the softmax, which a real implementation would fix.

The gain from int2 is large and is entirely a horizon effect. At B=1024, int2
scores 16/24 [0.47, 0.82] with the middle band at 7/7, against int4's 9/24
[0.21, 0.57] at 0/7 and fp16's 3/24 at 0/7. Keeping 5461 tokens instead of 1024
moves the horizon past the middle band. This is KIVI's finding reached from the
horizon law, not a new method.

Where that leaves phase 10: the eviction question is answered, and the answer
is "quantise to int2, keep FIFO". Priority ordering (age or KL) and formats
denser than int2 both lose.

## Stage 3-4 result: a supervised probe passes the screen AND works (M4, 2026-09-24)

Stages 3 and 4 were partly already answered. Hub tokens are the screen's
`norm_attention` (4/64 needle spans whole) and `lookahead` (0/64); both fail, so
no end-to-end run was spent on them. Learned admission by RL is phases 2-2c:
recency, then collapse.

What was untried is the *supervised* version (`kvadmission/probe.py`): label
answer-span tokens 1, fit a logistic regression on the layer-14 hidden state,
which a causal forward pass already computes at write time.

Screen, B=2048, held-out seeds (`results/probe-scorer.json`):

| trained on | tested on | spans whole | random |
|---|---|---|---|
| needle | needle | **64/64** [0.94, 1.00] | 0/64 |
| needle | qa | 7/61 | 2/61 |
| qa | needle | 0/64 | 0/64 |
| qa | qa | 11/61 [0.10, 0.29] | 2/61 |

End to end on needle seeds 100-102, 24 needles (`results/probe-e2e.json`):

| budget | policy | hits | middle | far |
|---|---|---|---|---|
| 1024 | probe | **24/24** [0.86, 1.00] | 7/7 | 8/8 |
| 1024 | recency | 3/24 | 0/7 | 0/8 |
| 1024 | probe + int2 | 19/24 | 6/7 | 6/8 |
| 2048 | probe | **24/24** | 7/7 | 8/8 |
| 2048 | recency | 6/24 | 0/7 | 0/8 |
| 2048 | probe + int2 | 18/24 | 4/7 | 5/8 |

Three things follow.

1. **The screen has now passed a realisable signal, and the prediction held.**
   The paper's Limitations section says the test has only ever passed an oracle.
   That is now out of date for the needle task: a linear probe passes the screen
   at 64/64 and then reaches the oracle's ceiling end-to-end, 24/24 at both
   budgets, against recency's 3/24 and 6/24.
2. **It is task-specific, not a general signal.** Trained on QA it finds nothing
   in needles (0/64) and only 11/61 on QA itself, where the screen predicts
   end-to-end failure. The probe learns what *this task's* answers look like; it
   is a trained oracle for one distribution, not a solution to the write-time
   problem the paper poses. The honest claim is: when the task distribution is
   known in advance, the capacity the oracle demonstrates is reachable by a
   cheap learned signal.
3. **int2 and a good signal do not compose.** int2 costs 5-6 spans once the
   right tokens are kept. Quantisation buys reach, which only helps while reach
   is the binding constraint.
