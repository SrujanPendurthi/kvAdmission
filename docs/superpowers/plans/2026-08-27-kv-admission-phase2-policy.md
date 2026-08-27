# Phase 2: Learned Admission Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a small admission policy against a frozen Qwen3-1.7B that achieves nonzero mid-context recall at a budget where every phase 1 baseline scores exactly zero.

**Architecture:** Two caches run in lockstep over the same stream — one uncompressed reference, one under the policy's admission decisions. Per-position KL between their next-token distributions is the dense reward. A ~1M-parameter head reads the frozen model's hidden states and emits one admit/skip decision per token, shared across layers. A Lagrangian multiplier prices cache occupancy. Training is average-reward policy gradient with a value baseline.

**Tech Stack:** Python 3.13, PyTorch 2.8 (MPS), transformers 5.14.1, pytest. Inference-only on the base model; gradients touch the policy head alone.

**Spec:** `docs/superpowers/specs/2026-08-27-kv-admission-middle-context-design.md`
**Prior phase:** `docs/phase1-findings.md`

## Global Constraints

- **The base model is frozen.** Gradients flow only into the policy head. Every base-model call is under `torch.no_grad()`.
- **One admission decision per token, shared across all layers** — spec §4.2. Per-layer decisions desynchronise layer lengths and break the single global attention mask. `prefill` asserts this; do not work around the assertion.
- **Training corpus is local Python source** (stdlib + site-packages). It is never Pride & Prejudice, which is the eval haystack. Training on the eval corpus invalidates the result.
- **Never train on needle-formatted streams.** The eval plants facts as `The secret code for X is NNNNNN`. A policy that learns that surface form has learned nothing about utility. Long-range dependencies in training come from code identifiers, not planted facts.
- **`chunk=128` everywhere.** Survival is `(B/(B+chunk))^(d/chunk)`; changing chunk changes the experiment and breaks comparability with phase 1.
- Sinks = first 4 tokens, never evicted. FIFO eviction when over budget.
- Tests are plain `assert` run by pytest. No fixtures or mocks unless a task says so.

---

### Task 1: Dual prefill and per-position KL

**Files:**
- Create: `kvadmission/dual.py`
- Test: `tests/test_dual.py`

**Interfaces:**
- Consumes: `AdmissionCache`, `runner.prefill` conventions from phase 1
- Produces:
  - `dual_prefill(model, ids, device, budget, decide, chunk=128, sinks=4, need_attn=False) -> tuple[torch.Tensor, int]` — steps a reference cache (uncompressed) and a compressed cache over the same stream, returns `(kl, stream_len)` where `kl` is a 1-D float tensor of length `stream_len`, `kl[t]` being `KL(p_full(·|x_<=t) ‖ p_compressed(·|x_<=t))`.

Both caches must advance in lockstep chunk by chunk. Running one prefill to completion and then the other would need every position's logits held at once — 64 chunks × 128 positions × 152k vocab is far past 24 GB.

This primitive serves Task 2's diagnostic and Task 5's reward. Build it once.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dual.py`:

```python
import torch

from kvadmission.dual import dual_prefill
from kvadmission.runner import load_model

MODEL, TOK, DEVICE = load_model()


def _ids(n=512):
    text = "def alpha(x):\n    return x + 1\n\n" * 200
    return TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :n]


def _admit_all(layer_idx, cache, attentions, n_new):
    cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))


def test_kl_is_zero_when_budget_is_unbounded():
    """No compression means the two caches are identical, so KL must vanish."""
    ids = _ids()
    kl, n = dual_prefill(MODEL, ids, DEVICE, budget=None, decide=None, chunk=128)
    assert n == ids.shape[1]
    assert kl.shape == (ids.shape[1],)
    assert kl.max().item() < 1e-4, f"max KL {kl.max().item()}"


def test_kl_is_positive_when_compressing_hard():
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert kl.max().item() > 0.01, f"max KL {kl.max().item()}"


def test_kl_is_never_negative():
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert kl.min().item() >= -1e-5, f"min KL {kl.min().item()}"


def test_tighter_budget_gives_more_divergence():
    ids = _ids()
    loose, _ = dual_prefill(MODEL, ids, DEVICE, budget=256, decide=_admit_all, chunk=128)
    tight, _ = dual_prefill(MODEL, ids, DEVICE, budget=64, decide=_admit_all, chunk=128)
    assert tight.mean().item() > loose.mean().item()


def test_early_positions_have_near_zero_kl():
    """Before the budget is exhausted nothing has been evicted yet."""
    ids = _ids()
    kl, _ = dual_prefill(MODEL, ids, DEVICE, budget=256, decide=_admit_all, chunk=128)
    assert kl[:120].max().item() < 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/srujanpendurthi/Projects/Current/kvAdmission && python3 -m pytest tests/test_dual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.dual'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/dual.py`:

```python
"""Lockstep prefill of a reference and a compressed cache, yielding per-position KL.

Both caches advance one chunk at a time so only one chunk of logits is ever
resident. Holding every position's logits for both runs would be 64 x 128 x
152k floats per run - far past the machine's memory.

KL(p_full || p_compressed) is the dense reward signal: it is zero wherever
compression changed nothing, and grows where the compressed cache has lost
something the model wanted.
"""

import torch
from transformers import DynamicCache

from kvadmission.cache import AdmissionCache


@torch.no_grad()
def dual_prefill(model, ids, device, budget, decide, chunk=128, sinks=4,
                 need_attn=False):
    """Step a reference and a compressed cache together over one stream.

    Returns (kl, stream_len). `kl[t]` is the divergence at position t between
    the uncompressed model and the model reading the compressed cache.
    """
    total = ids.shape[1]
    n_layers = model.config.num_hidden_layers
    ref = DynamicCache()
    comp = AdmissionCache(budget=budget, sinks=sinks)
    kl_parts = []

    for start in range(0, total, chunk):
        piece = ids[:, start:start + chunk].to(device)
        n_new = piece.shape[1]
        pos = torch.arange(start, start + n_new, device=device).unsqueeze(0)

        ref_out = model(
            piece, past_key_values=ref, position_ids=pos,
            attention_mask=torch.ones(
                1, ref.get_seq_length() + n_new, device=device, dtype=torch.long
            ),
            use_cache=True,
        )
        comp_out = model(
            piece, past_key_values=comp, position_ids=pos,
            attention_mask=torch.ones(
                1, comp.get_seq_length() + n_new, device=device, dtype=torch.long
            ),
            output_attentions=need_attn,
            use_cache=True,
        )

        lp_ref = torch.log_softmax(ref_out.logits[0].float(), dim=-1)
        lp_com = torch.log_softmax(comp_out.logits[0].float(), dim=-1)
        # KL(ref || comp), summed over vocab, one scalar per position
        kl_parts.append(
            (lp_ref.exp() * (lp_ref - lp_com)).sum(-1).clamp_min(0.0).cpu()
        )

        if decide is not None:
            for li in range(n_layers):
                decide(li, comp, comp_out.attentions if need_attn else None, n_new)
            occ = {comp.layers[li].keys.shape[-2] for li in range(n_layers)}
            if len(occ) > 1:
                raise RuntimeError(f"layers desynchronised: {sorted(occ)}")

        del ref_out, comp_out, lp_ref, lp_com

    return torch.cat(kl_parts), total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dual.py -v`
Expected: PASS, 5 passed. Slow — two forward passes per chunk.

- [ ] **Step 5: Commit**

```bash
git add kvadmission/dual.py tests/test_dual.py
git commit -m "feat: lockstep dual prefill with per-position KL"
```

---

### Task 2: The signal diagnostic — THE GATE

**Files:**
- Create: `kvadmission/signal.py`
- Create: `docs/phase2-signal-check.md`
- Test: `tests/test_signal.py`

**Interfaces:**
- Consumes: `dual_prefill` from Task 1
- Produces:
  - `code_corpus(max_files: int = 400) -> list[str]` — paths to local Python sources over 20 KB, excluding tests.
  - `build_code_stream(tok, target_len: int, seed: int = 0) -> torch.Tensor` — concatenated source, tokenized to exactly `target_len`.
  - `long_dependency_positions(tok, ids, min_distance: int = 2048, min_len: int = 4) -> list[int]` — positions `t` where token `t+1` repeats an identifier last seen at least `min_distance` tokens earlier.
  - `signal_report(model, tok, device, budget: int, length: int, seed: int) -> dict` — `{"kl_long": float, "kl_baseline": float, "ratio": float, "n_long": int}`.

**This task decides whether phase 2 is viable at all.**

The reward is KL between full and compressed next-token distributions. On text where prediction depends only on local context, compression of distant tokens changes nothing, KL stays flat, and the policy has no gradient telling it to retain anything far away — it would converge to recency, which phase 1 already showed is the failure mode.

So: does removing distant context measurably move the distribution on this corpus? If KL at long-dependency positions is not clearly above baseline positions, **the dense reward cannot teach mid-context retention and phase 2 must be re-scoped before any trainer is written.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signal.py`:

```python
import torch
from transformers import AutoTokenizer

from kvadmission.signal import (
    build_code_stream, code_corpus, long_dependency_positions,
)

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_corpus_finds_local_python_sources():
    paths = code_corpus(max_files=50)
    assert len(paths) > 10
    assert all(p.endswith(".py") for p in paths)


def test_corpus_excludes_the_eval_haystack():
    """Training on the eval corpus would invalidate the whole phase."""
    paths = code_corpus(max_files=400)
    assert not any("haystack" in p or "1342" in p for p in paths)


def test_code_stream_has_exact_length():
    ids = build_code_stream(TOK, target_len=2048)
    assert ids.shape == (1, 2048)


def test_code_stream_is_reproducible_and_seed_varies():
    a = build_code_stream(TOK, 1024, seed=1)
    b = build_code_stream(TOK, 1024, seed=1)
    c = build_code_stream(TOK, 1024, seed=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_long_dependency_positions_found_in_real_code():
    ids = build_code_stream(TOK, 8192, seed=0)
    pos = long_dependency_positions(TOK, ids, min_distance=2048)
    assert len(pos) > 5, f"only {len(pos)} long-dependency positions"
    assert all(0 <= p < 8192 for p in pos)


def test_long_dependency_respects_min_distance():
    """A synthetic stream whose only repeat is close by must yield nothing."""
    text = "alpha_variable = 1\nalpha_variable = 2\n" * 3
    ids = TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert long_dependency_positions(TOK, ids, min_distance=2048) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_signal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.signal'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/signal.py`:

```python
"""Does distant context actually move the next-token distribution?

The dense KL reward can only teach mid-context retention if removing distant
tokens measurably changes what the model predicts. Code is the corpus of choice:
an identifier defined thousands of tokens earlier and used again is a genuine,
mechanically detectable long-range dependency.

Deliberately NOT Pride and Prejudice - that is the evaluation haystack, and
training on it would invalidate the result.
"""

import random
import statistics
import sysconfig
from pathlib import Path

import torch

from kvadmission.dual import dual_prefill


def code_corpus(max_files=400):
    """Local Python sources over 20 KB, tests excluded."""
    roots = [Path(sysconfig.get_paths()["stdlib"]),
             Path(sysconfig.get_paths()["purelib"])]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            s = str(p)
            if "test" in s or "site-packages/pip" in s:
                continue
            try:
                if p.stat().st_size > 20_000:
                    out.append(s)
            except OSError:
                continue
            if len(out) >= max_files:
                return sorted(out)
    return sorted(out)


def build_code_stream(tok, target_len, seed=0):
    """Concatenate shuffled source files and cut to exactly target_len tokens."""
    paths = code_corpus()
    rng = random.Random(seed)
    rng.shuffle(paths)

    ids = []
    for p in paths:
        try:
            text = Path(p).read_text(errors="ignore")
        except OSError:
            continue
        ids += tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_len:
            break
    if len(ids) < target_len:
        raise RuntimeError(f"corpus too small: {len(ids)} < {target_len}")
    return torch.tensor([ids[:target_len]])


def long_dependency_positions(tok, ids, min_distance=2048, min_len=4):
    """Positions predicting an identifier last seen at least min_distance back."""
    toks = tok.convert_ids_to_tokens(ids[0].tolist())
    last_seen, out = {}, []
    for i, raw in enumerate(toks):
        key = raw.replace("Ġ", "").strip()  # strip the GPT-2 space marker
        if len(key) >= min_len and key.isidentifier():
            prev = last_seen.get(key)
            if prev is not None and i - prev >= min_distance and i > 0:
                out.append(i - 1)
            last_seen[key] = i
    return out


def signal_report(model, tok, device, budget, length, seed):
    """Compare KL at long-dependency positions against everywhere else."""
    ids = build_code_stream(tok, length, seed=seed)
    long_pos = long_dependency_positions(tok, ids)

    def admit_all(layer_idx, cache, attentions, n_new):
        cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))

    kl, _ = dual_prefill(model, ids, device, budget=budget, decide=admit_all)

    # only positions past the point where eviction actually starts
    warm = budget
    long_pos = [p for p in long_pos if p >= warm]
    other = [p for p in range(warm, length) if p not in set(long_pos)]
    if not long_pos:
        return {"kl_long": 0.0, "kl_baseline": 0.0, "ratio": 0.0, "n_long": 0}

    kl_long = statistics.mean(kl[p].item() for p in long_pos)
    kl_base = statistics.mean(kl[p].item() for p in other)
    return {
        "kl_long": kl_long,
        "kl_baseline": kl_base,
        "ratio": kl_long / kl_base if kl_base > 0 else float("inf"),
        "n_long": len(long_pos),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_signal.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Run the diagnostic across seeds and budgets**

```bash
cd /Users/srujanpendurthi/Projects/Current/kvAdmission
python3 -u -c "
from kvadmission.runner import load_model
from kvadmission.signal import signal_report
m, t, d = load_model()
for budget in (512, 1024, 2048):
    for seed in range(3):
        r = signal_report(m, t, d, budget=budget, length=8192, seed=seed)
        print(f'B={budget} seed={seed} kl_long={r[\"kl_long\"]:.4f} '
              f'kl_base={r[\"kl_baseline\"]:.4f} ratio={r[\"ratio\"]:.2f} n={r[\"n_long\"]}')
" 2>&1 | tee results/signal-check.log
```

- [ ] **Step 6: Write the gate decision**

Create `docs/phase2-signal-check.md` recording, with numbers:

1. Mean KL at long-dependency positions vs baseline positions, per budget and seed.
2. The ratio, and whether it is consistently above 1.
3. How many long-dependency positions exist per 8192-token stream.

**Gate:**

- **ratio ≳ 1.5 consistently** — dense KL carries long-range signal. Proceed to Task 3.
- **ratio ≈ 1.0** — distant context does not move the distribution on this corpus. **STOP.** The dense reward cannot teach mid-context retention. Re-scope with your human partner: sparse task reward at query points, or a different corpus with stronger dependencies.
- **`n_long` under ~10 per stream** — too few positions to train on regardless of ratio. Raise `length`, lower `min_distance`, or pick a denser corpus, then re-run before deciding.

Record the outcome honestly. A failed gate here is worth far more than a trained policy that silently learned recency.

- [ ] **Step 7: Commit**

```bash
printf '\nresults/signal-check.log\n' >> .gitignore
git add kvadmission/signal.py tests/test_signal.py docs/phase2-signal-check.md .gitignore
git commit -m "feat: KL signal diagnostic, the phase 2 viability gate"
```

---

### Tasks 3–7: gated on Task 2

The remaining tasks build the policy head, the Lagrangian controller, the average-reward trainer, the training loop, and the evaluation against phase 1 baselines.

**They are deliberately not written out yet.** Their shape depends on Task 2's answer:

- If the ratio is strong, the reward stays as spec §4.3 and the policy trains on dense KL over code streams.
- If the ratio is marginal, the reward needs reweighting toward high-KL positions, which changes the trainer.
- If the ratio is flat, the dense reward is abandoned for sparse task reward, which changes the algorithm, the sample budget, and probably the feasibility on this hardware.

Writing three variants of a training loop before knowing which one applies is exactly the speculative work the earlier phases avoided. **Task 2 runs first; the rest of this plan gets written against its result.**

Sketch, so the direction is on record:

| # | Task | Deliverable |
|---|---|---|
| 3 | `AdmissionPolicy` head — hidden state, ‖K‖, recency, occupancy, λ; one decision per token shared across layers | forward/sample tested without a model |
| 4 | Lagrangian λ controller | λ drives occupancy to target on a synthetic stream |
| 5 | Average-reward policy gradient with value baseline | learns a known-optimal toy policy |
| 6 | Training loop with checkpointing | loss curve, λ trace, occupancy trace |
| 7 | Eval against phase 1 baselines on the needle benchmark | mid-band recall vs the 0/192 baseline |

---

## Self-Review

**Spec coverage.** §4.3's KL reward → Task 1 (the measurement) and Task 2 (whether it carries signal). §4.2's shared-decision constraint → global constraints, asserted in `dual_prefill`. §4.2's policy features → Task 3 sketch. §4.4's WG-KV head-to-head is phase 3 and out of scope here.

**Deliberate gap.** Tasks 3–7 are sketched rather than specified, because Task 2 can invalidate the reward design they would be built on. This is the one place a plan should not front-run its own gate.

**Type consistency.** `decide(layer_idx, cache, attentions, n_new)` matches phase 1 exactly, so phase 1 policies work unchanged in `dual_prefill`. `budget=None` means unbounded everywhere. `dual_prefill` returns `(kl, stream_len)`; `signal_report` returns a dict with `kl_long`, `kl_baseline`, `ratio`, `n_long`.

**Cost note.** `dual_prefill` runs two forward passes per chunk, so it is ~2× phase 1 per stream. The Task 2 diagnostic is 9 streams of 8192 — roughly 20–30 minutes. Cheap relative to what it can prevent.
