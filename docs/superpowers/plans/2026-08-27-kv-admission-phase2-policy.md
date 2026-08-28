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

### Task 3: The admission policy head

**Files:**
- Create: `kvadmission/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — deliberately testable without a model
- Produces:
  - `AdmissionPolicy(hidden_size: int, proj: int = 128, n_extra: int = 4)` — `nn.Module`.
  - `AdmissionPolicy.forward(h: Tensor, extra: Tensor) -> tuple[Tensor, Tensor]` — `h` is `(n_tokens, hidden_size)`, `extra` is `(n_tokens, n_extra)`. Returns `(admit_logits, values)`, both `(n_tokens,)`.
  - `AdmissionPolicy.act(h, extra, generator=None) -> tuple[Tensor, Tensor, Tensor]` — returns `(keep, log_prob, value)`. `keep` is a bool tensor `(n_tokens,)`, `log_prob` is the summed log-probability of the sampled decisions (a scalar), `value` is the mean state value (a scalar).

Feature layout for `extra`, in this exact column order: `[k_norm, rel_pos, occupancy_frac, lam]`.

- `k_norm` — mean ‖K‖ across full-attention layers for that token, standardised within the chunk
- `rel_pos` — position within the current chunk, scaled to [0, 1]
- `occupancy_frac` — `occupancy / budget` at decision time
- `lam` — the current Lagrangian multiplier

For Qwen3-1.7B (`hidden_size` 2048) this is 2048×128 + two 132→128→1 MLPs ≈ **300K parameters**.

**Deviation from spec §4.2, recorded deliberately.** The spec lists "the token's attention mass within its own prefill chunk" as a feature. This version omits it: capturing it forces `output_attentions=True`, which forces eager attention and materialises `(1, H, 128, kv)` tensors on every chunk of every rollout. Task 2 showed hidden states already carry strong long-range signal (ratio up to 12.5), so the feature is not needed to get started. Adding it is a cheap ablation once the policy trains at all.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_policy.py`:

```python
import torch

from kvadmission.policy import AdmissionPolicy


def _inputs(n=16, hidden=64):
    torch.manual_seed(0)
    return torch.randn(n, hidden), torch.randn(n, 4)


def test_forward_shapes():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    logits, values = p(h, extra)
    assert logits.shape == (16,)
    assert values.shape == (16,)


def test_act_returns_bool_mask_of_right_length():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    keep, log_prob, value = p.act(h, extra)
    assert keep.shape == (16,)
    assert keep.dtype == torch.bool
    assert log_prob.ndim == 0
    assert value.ndim == 0


def test_act_is_reproducible_under_a_generator():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    a, _, _ = p.act(h, extra, generator=torch.Generator().manual_seed(3))
    b, _, _ = p.act(h, extra, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_log_prob_matches_bernoulli():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    keep, log_prob, _ = p.act(h, extra, generator=torch.Generator().manual_seed(1))
    logits, _ = p(h, extra)
    expected = torch.distributions.Bernoulli(logits=logits).log_prob(
        keep.float()
    ).sum()
    assert torch.allclose(log_prob, expected, atol=1e-5)


def test_gradients_reach_every_parameter():
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    _, log_prob, value = p.act(h, extra)
    (log_prob + value).backward()
    missing = [n for n, q in p.named_parameters() if q.grad is None]
    assert missing == [], f"no gradient reached: {missing}"


def test_confident_logits_drive_the_decision():
    """A large positive bias must admit everything; large negative, nothing."""
    p = AdmissionPolicy(hidden_size=64)
    h, extra = _inputs()
    with torch.no_grad():
        p.head[-1].bias.fill_(20.0)
    keep, _, _ = p.act(h, extra)
    assert keep.all()
    with torch.no_grad():
        p.head[-1].bias.fill_(-20.0)
    keep, _, _ = p.act(h, extra)
    assert not keep.any()


def test_parameter_count_is_small():
    p = AdmissionPolicy(hidden_size=2048)
    total = sum(q.numel() for q in p.parameters())
    assert total < 1_500_000, f"{total} parameters is too large for a frozen-base head"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.policy'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/policy.py`:

```python
"""The learned admission head.

Reads what the frozen model already computed and emits one admit/skip decision
per token, shared across layers (spec 4.2 - per-layer decisions desynchronise
layer lengths and break the single global attention mask).

Small on purpose: the base model is frozen, so this is the only thing carrying
gradients, and it must be cheap enough to run once per chunk per rollout.
"""

import torch
from torch import nn


class AdmissionPolicy(nn.Module):
    """Bernoulli admission policy with a shared value head.

    extra features, in column order: [k_norm, rel_pos, occupancy_frac, lam]
    """

    def __init__(self, hidden_size, proj=128, n_extra=4):
        super().__init__()
        self.proj = nn.Linear(hidden_size, proj)
        self.head = nn.Sequential(
            nn.Linear(proj + n_extra, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(proj + n_extra, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, h, extra):
        z = torch.cat([torch.tanh(self.proj(h)), extra], dim=-1)
        return self.head(z).squeeze(-1), self.value(z).squeeze(-1)

    def act(self, h, extra, generator=None):
        """Sample admission decisions. Returns (keep, summed log_prob, mean value)."""
        logits, values = self(h, extra)
        probs = torch.sigmoid(logits)
        noise = torch.rand(probs.shape, generator=generator, device=probs.device)
        keep = noise < probs.detach()
        log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(
            keep.float()
        ).sum()
        return keep, log_prob, values.mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_policy.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add kvadmission/policy.py tests/test_policy.py
git commit -m "feat: admission policy head with shared value function"
```

---

### Task 4: Lagrangian occupancy controller

**Files:**
- Create: `kvadmission/lagrangian.py`
- Test: `tests/test_lagrangian.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LagrangianController(target: float, eta: float = 0.01, lam: float = 0.0, lam_max: float = 10.0)`
  - `LagrangianController.update(occupancy_frac: float) -> float` — one dual-ascent step, returns the new λ.
  - `LagrangianController.lam` — current multiplier, always ≥ 0.

Memory is a constraint, not a hand-tuned penalty. λ rises while occupancy sits above target and falls when it drops below, so the price of cache space finds its own level instead of being swept by hand.

`lam_max` exists because an unbounded multiplier can run away if the policy cannot reach the target at all — for example when the sink prefix alone exceeds it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lagrangian.py`:

```python
from kvadmission.lagrangian import LagrangianController


def test_lambda_rises_when_over_target():
    c = LagrangianController(target=0.5, eta=0.1)
    c.update(0.9)
    assert c.lam > 0.0


def test_lambda_falls_when_under_target():
    c = LagrangianController(target=0.5, eta=0.1, lam=1.0)
    before = c.lam
    c.update(0.1)
    assert c.lam < before


def test_lambda_never_goes_negative():
    c = LagrangianController(target=0.5, eta=0.5)
    for _ in range(50):
        c.update(0.0)
    assert c.lam == 0.0


def test_lambda_is_capped():
    c = LagrangianController(target=0.1, eta=1.0, lam_max=3.0)
    for _ in range(100):
        c.update(1.0)
    assert c.lam == 3.0


def test_lambda_drives_occupancy_to_target():
    """Closed loop: occupancy falls as the price of cache space rises."""
    c = LagrangianController(target=0.6, eta=0.2)
    occupancy = 1.0
    for _ in range(300):
        c.update(occupancy)
        occupancy = 1.0 / (1.0 + c.lam)  # more expensive space, less of it used
    assert abs(occupancy - 0.6) < 0.05, f"settled at {occupancy}"


def test_update_returns_the_new_lambda():
    c = LagrangianController(target=0.5, eta=0.1)
    assert c.update(0.9) == c.lam
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_lagrangian.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.lagrangian'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/lagrangian.py`:

```python
"""Dual ascent on the cache-occupancy constraint.

Treats memory as a constraint rather than a penalty term: lambda is a shadow
price on cache space that rises while occupancy is above target and falls when
it is below, so the trade-off tunes itself instead of being swept by hand.

Not novel - TriRoute (arXiv 2607.06601) applies the same dual ascent to KV
allocation. Used here for the practical benefit.
"""


class LagrangianController:
    """lambda_{t+1} = clip(lambda_t + eta * (occupancy - target), 0, lam_max)."""

    def __init__(self, target, eta=0.01, lam=0.0, lam_max=10.0):
        self.target = target
        self.eta = eta
        self.lam = lam
        self.lam_max = lam_max

    def update(self, occupancy_frac):
        """One dual-ascent step. Returns the new multiplier."""
        self.lam += self.eta * (occupancy_frac - self.target)
        # clipped below at 0 (a negative price would pay to fill the cache) and
        # above at lam_max, which stops runaway when the target is unreachable
        self.lam = max(0.0, min(self.lam_max, self.lam))
        return self.lam
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_lagrangian.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add kvadmission/lagrangian.py tests/test_lagrangian.py
git commit -m "feat: Lagrangian shadow price on cache occupancy"
```

---

### Task 5: Average-reward policy gradient

**Files:**
- Create: `kvadmission/learner.py`
- Test: `tests/test_learner.py`

**Interfaces:**
- Consumes: `AdmissionPolicy` from Task 3
- Produces:
  - `AverageRewardLearner(policy, lr: float = 3e-4, alpha: float = 0.01)`
  - `AverageRewardLearner.step(log_probs: list[Tensor], values: list[Tensor], rewards: list[float]) -> dict` — one update over a rollout of chunk-level timesteps. Returns `{"policy_loss", "value_loss", "r_bar", "mean_delta"}`.
  - `AverageRewardLearner.r_bar` — running average-reward estimate.

Differential TD, not discounted return: `δ_k = r_k − r̄ + V(s_{k+1}) − V(s_k)`, with `r̄ ← r̄ + α·δ_k`. In a continuing task with no episode boundary, a discount factor silently imposes a horizon — and if that horizon is shorter than the distance at which retention pays off, the optimal discounted policy is to drop everything.

Per spec §4.3: plain policy gradient with a value baseline. No PPO until this demonstrably fails.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_learner.py`:

```python
import torch

from kvadmission.learner import AverageRewardLearner
from kvadmission.policy import AdmissionPolicy


def _rollout(policy, rewards, gen):
    log_probs, values = [], []
    for _ in rewards:
        h = torch.randn(8, 64)
        extra = torch.zeros(8, 4)
        _, lp, v = policy.act(h, extra, generator=gen)
        log_probs.append(lp)
        values.append(v)
    return log_probs, values


def test_step_returns_diagnostics():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 0.5, 0.2], gen)
    out = learner.step(lp, v, [1.0, 0.5, 0.2])
    for key in ("policy_loss", "value_loss", "r_bar", "mean_delta"):
        assert key in out and isinstance(out[key], float)


def test_r_bar_tracks_the_reward_level():
    p = AdmissionPolicy(hidden_size=64)
    learner = AverageRewardLearner(p, alpha=0.2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        lp, v = _rollout(p, [2.0] * 4, gen)
        learner.step(lp, v, [2.0] * 4)
    assert 0.5 < learner.r_bar < 3.5, f"r_bar drifted to {learner.r_bar}"


def test_parameters_actually_move():
    p = AdmissionPolicy(hidden_size=64)
    before = p.proj.weight.detach().clone()
    learner = AverageRewardLearner(p, lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    lp, v = _rollout(p, [1.0, 1.0], gen)
    learner.step(lp, v, [1.0, 1.0])
    assert not torch.equal(before, p.proj.weight.detach())


def test_learns_a_toy_task_where_admitting_pays():
    """Reward proportional to how much was admitted - policy must trend to admit."""
    torch.manual_seed(0)
    p = AdmissionPolicy(hidden_size=8)
    learner = AverageRewardLearner(p, lr=5e-2, alpha=0.05)
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(32, 8)
    extra = torch.zeros(32, 4)

    def admit_rate():
        with torch.no_grad():
            logits, _ = p(h, extra)
            return torch.sigmoid(logits).mean().item()

    start = admit_rate()
    for _ in range(120):
        log_probs, values, rewards = [], [], []
        for _ in range(4):
            keep, lp, v = p.act(h, extra, generator=gen)
            log_probs.append(lp)
            values.append(v)
            rewards.append(keep.float().mean().item())
        learner.step(log_probs, values, rewards)
    assert admit_rate() > start + 0.05, f"{start:.3f} -> {admit_rate():.3f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_learner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.learner'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/learner.py`:

```python
"""Average-reward policy gradient with a value baseline.

Differential TD rather than discounted return. This is a continuing task with
no episode boundary, where a discount factor silently imposes a horizon - and
if that horizon is shorter than the distance at which retention pays off, the
optimal discounted policy is to drop everything.

Plain policy gradient per spec 4.3. No PPO until this demonstrably fails.
"""

import torch


class AverageRewardLearner:
    def __init__(self, policy, lr=3e-4, alpha=0.01, value_coef=0.5,
                 grad_clip=1.0):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.alpha = alpha
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.r_bar = 0.0

    def step(self, log_probs, values, rewards):
        """One update over a rollout of chunk-level timesteps."""
        n = len(rewards)
        assert len(log_probs) == len(values) == n, "ragged rollout"

        deltas, policy_terms, value_terms = [], [], []
        for k in range(n):
            # bootstrap off the next state; the final step has no successor
            v_next = values[k + 1].detach() if k + 1 < n else torch.zeros(())
            delta = rewards[k] - self.r_bar + v_next - values[k]
            deltas.append(delta.detach())
            policy_terms.append(-log_probs[k] * delta.detach())
            value_terms.append(delta.pow(2))

        mean_delta = torch.stack(deltas).mean()
        self.r_bar += self.alpha * mean_delta.item()

        policy_loss = torch.stack(policy_terms).mean()
        value_loss = torch.stack(value_terms).mean()
        loss = policy_loss + self.value_coef * value_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "r_bar": self.r_bar,
            "mean_delta": mean_delta.item(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_learner.py -v`
Expected: PASS, 4 passed. The toy-task test is the meaningful one — it proves the gradient has the right sign.

- [ ] **Step 5: Commit**

```bash
git add kvadmission/learner.py tests/test_learner.py
git commit -m "feat: average-reward policy gradient with value baseline"
```

---

### Task 6: Training loop

**Files:**
- Create: `kvadmission/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 3–5, `dual_prefill`, `build_code_stream`
- Produces:
  - `rollout(model, policy, controller, ids, device, budget, chunk=128, sinks=4, generator=None) -> dict` — one stream. Returns `{"log_probs", "values", "rewards", "occupancy", "lam", "kl"}`.
  - `train(model, tok, device, steps, budget, length, out_dir, lr, target, seed) -> list[dict]` — the loop, checkpointing the policy and appending a metrics row per step.
  - CLI: `python3 -m kvadmission.train --steps 300 --budget 2048 --length 8192`

**One timestep is one chunk**, not one token. State is the chunk's features; the action is its 128 admission decisions; the reward is

```
r_k = − mean(KL over chunk k)  −  λ_k · occupancy_frac_k
```

All decisions within a chunk share the timestep's δ. That is coarse credit assignment, and it is deliberate for v1 — per-token credit within a chunk needs per-token value estimates and a much noisier gradient. The value function bootstrapping across ~64 chunk-steps is what carries credit backwards to the decisions that caused a later divergence.

**The gradient problem to watch for:** `dual_prefill` is decorated `@torch.no_grad()`. The policy must still build a graph. The decider wraps its policy call in `torch.enable_grad()`, which re-enables gradients inside the no-grad region for the head only — the base model's forward stays gradient-free.

```
# ponytail: one delta per chunk shared by all 128 decisions. Per-token credit
# assignment only if the value function proves too coarse to learn from.
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
import torch

from kvadmission.lagrangian import LagrangianController
from kvadmission.policy import AdmissionPolicy
from kvadmission.runner import load_model
from kvadmission.signal import build_code_stream
from kvadmission.train import rollout

MODEL, TOK, DEVICE = load_model()


def _setup():
    policy = AdmissionPolicy(hidden_size=MODEL.config.hidden_size).to(DEVICE)
    controller = LagrangianController(target=0.9, eta=0.05)
    ids = build_code_stream(TOK, 1024, seed=0)
    return policy, controller, ids


def test_rollout_returns_one_entry_per_chunk():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert len(out["rewards"]) == 8
    assert len(out["log_probs"]) == 8
    assert len(out["values"]) == 8


def test_rollout_log_probs_carry_gradient():
    """dual_prefill runs under no_grad; the head must still build a graph."""
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert out["log_probs"][0].requires_grad
    out["log_probs"][0].backward()
    assert policy.proj.weight.grad is not None


def test_rollout_respects_the_budget():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert max(out["occupancy"]) <= 256


def test_rewards_are_finite_and_non_positive():
    """Reward is negative KL minus a non-negative occupancy price."""
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert all(r <= 1e-6 for r in out["rewards"])
    assert all(abs(r) < 1e6 for r in out["rewards"])


def test_lambda_moves_during_a_rollout():
    policy, controller, ids = _setup()
    out = rollout(MODEL, policy, controller, ids, DEVICE, budget=256, chunk=128)
    assert len(set(out["lam"])) > 1, "lambda never updated"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.train'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/train.py`:

```python
"""Training loop for the admission policy.

One timestep is one chunk: the action is that chunk's 128 admission decisions,
the reward is negative mean KL minus the priced occupancy, and the value
function bootstraps across the ~64 chunk-steps of a stream. That is what
carries credit back from a divergence to the decision that caused it.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache

from kvadmission.cache import AdmissionCache
from kvadmission.lagrangian import LagrangianController
from kvadmission.learner import AverageRewardLearner
from kvadmission.policy import AdmissionPolicy
from kvadmission.runner import load_model
from kvadmission.signal import build_code_stream


def _extra_features(keys_norm, n_new, occupancy_frac, lam, device):
    """Columns, in order: [k_norm, rel_pos, occupancy_frac, lam]."""
    k = keys_norm
    k = (k - k.mean()) / (k.std() + 1e-6)
    rel = torch.linspace(0, 1, n_new, device=device)
    occ = torch.full((n_new,), occupancy_frac, device=device)
    lm = torch.full((n_new,), lam, device=device)
    return torch.stack([k, rel, occ, lm], dim=-1)


@torch.no_grad()
def rollout(model, policy, controller, ids, device, budget, chunk=128,
            sinks=4, generator=None):
    """One stream. Chunk-level timesteps with per-chunk reward."""
    total = ids.shape[1]
    n_layers = model.config.num_hidden_layers
    ref = DynamicCache()
    comp = AdmissionCache(budget=budget, sinks=sinks)

    log_probs, values, rewards, occupancy, lams, kls = [], [], [], [], [], []

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
            output_hidden_states=True, use_cache=True,
        )

        lp_ref = torch.log_softmax(ref_out.logits[0].float(), dim=-1)
        lp_com = torch.log_softmax(comp_out.logits[0].float(), dim=-1)
        kl = (lp_ref.exp() * (lp_ref - lp_com)).sum(-1).clamp_min(0.0).mean()

        h = comp_out.hidden_states[-1][0].float()          # (n_new, hidden)
        k_norm = torch.stack([
            comp.layers[li].keys[0, :, -n_new:, :].float().norm(dim=-1).mean(0)
            for li in range(n_layers)
        ]).mean(0)

        occ_frac = comp.occupancy(0) / budget
        extra = _extra_features(k_norm, n_new, occ_frac, controller.lam, device)

        # dual_prefill-style no_grad covers the base model; the head still needs a graph
        with torch.enable_grad():
            keep, log_prob, value = policy.act(h, extra, generator=generator)

        for li in range(n_layers):
            comp.apply_admission(li, keep)
        occ = {comp.layers[li].keys.shape[-2] for li in range(n_layers)}
        if len(occ) > 1:
            raise RuntimeError(f"layers desynchronised: {sorted(occ)}")

        occ_now = comp.occupancy(0)
        lam = controller.update(occ_now / budget)
        reward = -kl.item() - lam * (occ_now / budget)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        occupancy.append(occ_now)
        lams.append(lam)
        kls.append(kl.item())
        del ref_out, comp_out, lp_ref, lp_com

    return {"log_probs": log_probs, "values": values, "rewards": rewards,
            "occupancy": occupancy, "lam": lams, "kl": kls}


def train(model, tok, device, steps, budget, length, out_dir, lr=3e-4,
          target=0.95, seed=0):
    policy = AdmissionPolicy(hidden_size=model.config.hidden_size).to(device)
    controller = LagrangianController(target=target, eta=0.02)
    learner = AverageRewardLearner(policy, lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []

    for step in range(steps):
        ids = build_code_stream(tok, length, seed=1000 + step)
        r = rollout(model, policy, controller, ids, device, budget,
                    generator=gen)
        stats = learner.step(r["log_probs"], r["values"], r["rewards"])
        stats.update({
            "step": step,
            "mean_kl": sum(r["kl"]) / len(r["kl"]),
            "mean_occupancy": sum(r["occupancy"]) / len(r["occupancy"]),
            "lam": r["lam"][-1],
        })
        history.append(stats)
        print(f"step {step:4d} kl={stats['mean_kl']:.4f} "
              f"occ={stats['mean_occupancy']:.0f} lam={stats['lam']:.3f} "
              f"r_bar={stats['r_bar']:.4f}", flush=True)

        if step % 20 == 0 or step == steps - 1:
            torch.save(policy.state_dict(), out / "policy.pt")
            (out / "history.json").write_text(json.dumps(history, indent=1))

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/policy")
    a = p.parse_args()

    model, tok, device = load_model(a.model)
    train(model, tok, device, a.steps, a.budget, a.length, a.out,
          lr=a.lr, target=a.target, seed=a.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_train.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Smoke-train briefly**

```bash
python3 -u -m kvadmission.train --steps 5 --budget 512 --length 2048 \
  --out results/policy-smoke
```

Expected: five step lines, `mean_kl` finite, `lam` moving, no crash. This checks the loop runs — five steps prove nothing about learning.

- [ ] **Step 6: Commit**

```bash
printf '\nresults/policy*/\n' >> .gitignore
git add kvadmission/train.py tests/test_train.py .gitignore
git commit -m "feat: chunk-level average-reward training loop"
```

---

### Task 7: Evaluate against the phase 1 baselines

**Files:**
- Create: `kvadmission/evaluate.py`
- Create: `docs/phase2-findings.md`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: everything above, plus `sweep.summarize` and `stream.query_needle` from phase 1
- Produces:
  - `learned_decider(policy, controller, model, budget, device) -> tuple[callable, bool]` — wraps a trained policy in the `decide(layer_idx, cache, attentions, n_new)` signature phase 1's harness expects, so it drops into `run_cell` beside `recent`, `h2o`, and the rest.
  - `evaluate(model, tok, device, checkpoint: str, budget: int, seeds: int, length: int, n_needles: int) -> list[dict]` — rows in phase 1's schema.
  - CLI: `python3 -m kvadmission.evaluate --checkpoint results/policy/policy.pt --budget 2048`

**The comparison must be exactly phase 1's.** Same needle harness, same seeds, same banding, same 8192-token streams over the Pride and Prejudice haystack that the policy has never seen. Anything else and the 0/192 baseline is not a baseline.

The policy's decider needs hidden states, so it runs its own forward rather than reading `attentions`. That makes evaluation slower than a heuristic cell — acceptable, since it runs once.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate.py`:

```python
import torch

from kvadmission.evaluate import learned_decider
from kvadmission.lagrangian import LagrangianController
from kvadmission.policy import AdmissionPolicy
from kvadmission.runner import load_model

MODEL, TOK, DEVICE = load_model()


def _decider(budget=256):
    policy = AdmissionPolicy(hidden_size=MODEL.config.hidden_size).to(DEVICE)
    controller = LagrangianController(target=0.95)
    return learned_decider(policy, controller, MODEL, budget, DEVICE)


def test_decider_has_the_phase1_signature():
    decide, need_attn = _decider()
    assert need_attn is False
    assert decide.__code__.co_argcount == 4


def test_decider_keeps_layers_aligned():
    from kvadmission.cache import AdmissionCache
    decide, _ = _decider(budget=64)
    cache = AdmissionCache(budget=64, sinks=4)
    hidden = MODEL.config.hidden_size
    for _ in range(3):
        for li in range(MODEL.config.num_hidden_layers):
            k = torch.zeros(1, MODEL.config.num_key_value_heads, 32, 128,
                            device=DEVICE)
            cache.update(k, k.clone(), li)
        decide(0, cache, None, 32)
        for li in range(1, MODEL.config.num_hidden_layers):
            decide(li, cache, None, 32)
        occ = {cache.occupancy(li)
               for li in range(MODEL.config.num_hidden_layers)}
        assert len(occ) == 1, f"desynchronised: {occ}"


def test_decider_respects_budget():
    from kvadmission.cache import AdmissionCache
    decide, _ = _decider(budget=64)
    cache = AdmissionCache(budget=64, sinks=4)
    for _ in range(4):
        for li in range(MODEL.config.num_hidden_layers):
            k = torch.zeros(1, MODEL.config.num_key_value_heads, 32, 128,
                            device=DEVICE)
            cache.update(k, k.clone(), li)
        for li in range(MODEL.config.num_hidden_layers):
            decide(li, cache, None, 32)
    assert cache.occupancy(0) <= 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.evaluate'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/evaluate.py`:

```python
"""Evaluate a trained policy on phase 1's needle benchmark.

Deliberately reuses phase 1's harness unchanged - same streams, same seeds,
same banding, same haystack the policy never trained on. Any deviation and the
0/192 middle-band baseline stops being a baseline.
"""

import argparse
import json
from pathlib import Path

import torch

from kvadmission.cache import AdmissionCache
from kvadmission.lagrangian import LagrangianController
from kvadmission.policy import AdmissionPolicy
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle
from kvadmission.sweep import summarize


def learned_decider(policy, controller, model, budget, device):
    """Wrap a trained policy in phase 1's decide(layer, cache, attn, n_new)."""
    state = {"keep": None}
    n_layers = model.config.num_hidden_layers

    @torch.no_grad()
    def decide(layer_idx, cache, attentions, n_new):
        # Decide once per chunk on layer 0, then apply the same mask everywhere:
        # per-layer decisions desynchronise layer lengths (spec 4.2).
        if layer_idx == 0:
            keys = cache.layers[0].keys[0, :, -n_new:, :].float()
            k_norm = keys.norm(dim=-1).mean(0)
            k_norm = (k_norm - k_norm.mean()) / (k_norm.std() + 1e-6)
            occ = cache.occupancy(0) / budget
            extra = torch.stack([
                k_norm,
                torch.linspace(0, 1, n_new, device=device),
                torch.full((n_new,), occ, device=device),
                torch.full((n_new,), controller.lam, device=device),
            ], dim=-1)
            h = state["hidden"][-n_new:]
            logits, _ = policy(h, extra)
            state["keep"] = (torch.sigmoid(logits) > 0.5).cpu()
        cache.apply_admission(layer_idx, state["keep"])
        if layer_idx == n_layers - 1:
            controller.update(cache.occupancy(0) / budget)

    decide.state = state
    return decide, False


@torch.no_grad()
def evaluate(model, tok, device, checkpoint, budget, seeds=5, length=8192,
             n_needles=8, chunk=128):
    policy = AdmissionPolicy(hidden_size=model.config.hidden_size).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()

    rows = []
    for seed in range(seeds):
        controller = LagrangianController(target=0.95, eta=0.02)
        decide, _ = learned_decider(policy, controller, model, budget, device)
        ids, needles = build_stream(tok, n_needles, length, seed=seed)
        cache = AdmissionCache(budget=budget, sinks=4)

        # hidden states must exist before decide runs, so capture them per chunk
        total = ids.shape[1]
        for start in range(0, total, chunk):
            piece = ids[:, start:start + chunk].to(device)
            n_new = piece.shape[1]
            out = model(
                piece, past_key_values=cache,
                position_ids=torch.arange(
                    start, start + n_new, device=device).unsqueeze(0),
                attention_mask=torch.ones(
                    1, cache.get_seq_length() + n_new, device=device,
                    dtype=torch.long),
                output_hidden_states=True, use_cache=True,
            )
            decide.state["hidden"] = out.hidden_states[-1][0].float()
            for li in range(model.config.num_hidden_layers):
                decide(li, cache, None, n_new)
            del out

        for n in needles:
            hit, text = query_needle(model, tok, cache, total, n, device)
            rows.append({
                "policy": "learned", "budget": budget, "seed": seed,
                "topic": n["topic"], "distance_back": total - n["pos"],
                "stream_len": total, "hit": bool(hit), "gen": text[:60],
            })
        print(f"seed={seed} learned B={budget} "
              f"{sum(r['hit'] for r in rows[-n_needles:])}/{n_needles}",
              flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint", default="results/policy/policy.pt")
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--needles", type=int, default=8)
    p.add_argument("--out", default="results/phase2-eval.json")
    a = p.parse_args()

    model, tok, device = load_model(a.model)
    rows = evaluate(model, tok, device, a.checkpoint, a.budget,
                    a.seeds, a.length, a.needles)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

    print("\nrecall by band (mean, 95% upper)")
    for key, cell in sorted(summarize(rows).items()):
        pol, b, band = key
        print(f"  {pol:8s} B={b:<6d} {band:7s} {cell['mean']:.2f} "
              f"(<={cell['upper95']:.2f}, n={cell['n']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_evaluate.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Train, then evaluate**

```bash
python3 -u -m kvadmission.train --steps 300 --budget 2048 --length 8192 \
  --out results/policy > results/train.log 2>&1

python3 -u -m kvadmission.evaluate --checkpoint results/policy/policy.pt \
  --budget 2048 --seeds 5 > results/phase2-eval.log 2>&1
```

- [ ] **Step 6: Write the findings**

Create `docs/phase2-findings.md` answering, with numbers:

1. Did training converge? Show `mean_kl` and `r_bar` over steps, and whether λ held occupancy near target.
2. **Middle-band recall against the 0/192 phase 1 baseline**, with the rule-of-three bound.
3. Near and far band recall — did the policy sacrifice the near band to reach the middle?
4. What admission rate did it settle at, and does it correlate with anything interpretable?
5. If middle-band recall is still zero: was the reward flat, the credit assignment too coarse, or the policy underpowered? Say which, from the logged metrics.

**A zero result here is a real result.** The dense KL reward has confirmed long-range signal (phase 2 signal check, ratio up to 12.5) — so if the policy still cannot convert that into mid-context recall, the interesting question becomes why, and that is worth reporting rather than burying under a hyperparameter search.

- [ ] **Step 7: Commit**

```bash
git add kvadmission/evaluate.py tests/test_evaluate.py docs/phase2-findings.md \
  results/phase2-eval.json
git commit -m "results: phase 2 learned policy evaluated against phase 1 baselines"
```


---

## Self-Review

**Spec coverage.** §4.2's policy features → Task 3 (`AdmissionPolicy`), with the attention-mass feature explicitly deferred and the reason recorded. §4.2's shared-decision constraint → asserted in `rollout` and `learned_decider`, and tested in both. §4.3's KL reward → Tasks 1, 2, 6. §4.3's Lagrangian λ → Task 4. §4.3's average-reward formulation → Task 5. §5's comparison against phase 1 baselines → Task 7. §4.4's WG-KV head-to-head is phase 3 and out of scope.

**Placeholder scan.** No TBDs. Every code step carries runnable code; every test step carries real assertions.

**Type consistency.** `decide(layer_idx, cache, attentions, n_new)` is identical across `policies.py`, `dual.py`, `train.py`, and `evaluate.py`, so phase 1 baselines and the learned policy are interchangeable in the same harness. `AdmissionPolicy.act` returns `(keep, log_prob, value)` in Tasks 3, 5, and 6 alike. `extra` columns are `[k_norm, rel_pos, occupancy_frac, lam]` in Task 3's docstring, Task 6's `_extra_features`, and Task 7's `learned_decider` — the same order in all three. `budget=None` means unbounded everywhere. Eval rows carry `stream_len`, matching phase 1's `summarize`.

**Two design choices worth challenging at review.**

*Chunk-level timesteps.* All 128 decisions in a chunk share one δ. Coarse, and marked with a `ponytail:` comment. The bet is that value-function bootstrapping across ~64 chunk-steps carries credit backwards well enough. If Task 7 reports zero middle-band recall with a flat value loss, this is the first thing to suspect.

*Greedy decisions at evaluation.* `learned_decider` thresholds at 0.5 rather than sampling, so evaluation is deterministic. That is right for measurement, but it means the eval distribution differs from the training distribution. If eval underperforms the training reward markedly, check this before anything else.

**Known cost.** `rollout` runs two forward passes per chunk plus `output_hidden_states`, so a 300-step run over 8192-token streams is on the order of several hours on the M4. The speedups identified earlier — SDPA where attention weights are unused, batched needle queries — are worth doing before the full run if training turns out to need more than a few hundred steps.
