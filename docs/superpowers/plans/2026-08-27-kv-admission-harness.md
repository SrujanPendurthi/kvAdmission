# KV Admission Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a budget-capped KV cache supporting both write-time admission and score-based eviction, plus the streaming harness and baselines needed to re-establish the mid-context dead zone at proper scale with error bars.

**Architecture:** A `Cache` subclass holds a per-layer token budget with a pinned sink prefix. The frozen model writes every token normally through `update()`; after each chunk the caller applies either an *admission* decision (keep-mask over just-written tokens, FIFO when over budget) or an *eviction* decision (top-B by score, for reproducing published baselines). Policies are pure functions over cached state, so they are testable without a model. A sweep driver runs the needle experiment across policies, budgets, and seeds.

**Tech Stack:** Python 3.13, PyTorch 2.8 (MPS), transformers 5.14.1, pytest. No accelerate, no training frameworks — phase 0–1 is inference only.

**Spec:** `docs/superpowers/specs/2026-08-27-kv-admission-middle-context-design.md`

## Global Constraints

- **The base model is frozen.** No gradients flow through it anywhere in this plan. All work is `torch.no_grad()` inference.
- **Sinks = first 4 tokens of the stream**, never evicted, in every policy.
- **FIFO** is the eviction rule for admission mode: over budget, the oldest non-sink resident goes first.
- **`position_ids` must be true stream position**, never derived from compressed cache length. This is the classic KV-eviction bug and it is silent.
- **Development model is `Qwen/Qwen3-1.7B`**, bf16, `attn_implementation="eager"` (required for `output_attentions`), device `mps`.
- **Code stays model-size-agnostic** — no hardcoded layer counts, head counts, or hidden sizes. Read everything from `model.config`.
- **`spike/` is frozen throwaway.** Do not import from it, do not modify it. It is the record of how we got here.
- Tests are plain `assert` statements run by pytest. No fixtures, no mocks, no parametrize unless a task says so.

---

### Task 1: AdmissionCache — budget, sinks, FIFO admission

**Files:**
- Create: `kvadmission/__init__.py`
- Create: `kvadmission/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `AdmissionCache(budget: int | None = None, sinks: int = 4, **kwargs)` — subclass of `transformers.DynamicCache`. `budget=None` means unbounded.
  - `AdmissionCache.occupancy(layer_idx: int = 0) -> int`
  - `AdmissionCache.apply_admission(layer_idx: int, keep: torch.Tensor) -> torch.Tensor` — `keep` is a 1-D bool tensor over the tokens written by the most recent `update()` for that layer. Returns a 1-D long tensor of the surviving positions *as indices into the pre-call cache*, so callers can reindex their own per-token bookkeeping.

- [ ] **Step 1: Install pytest**

```bash
cd /Users/srujanpendurthi/Projects/Current/kvAdmission
python3 -m pip install pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cache.py`:

```python
import torch

from kvadmission.cache import AdmissionCache


def _write(cache, layer_idx, n_tokens, heads=2, dim=4):
    """Append n_tokens of identifiable KV to a layer. Key value == token ordinal."""
    start = cache.occupancy(layer_idx)
    ords = torch.arange(start, start + n_tokens, dtype=torch.float32)
    k = ords.view(1, 1, n_tokens, 1).expand(1, heads, n_tokens, dim).contiguous()
    cache.update(k, k.clone(), layer_idx)
    return ords


def _resident_ordinals(cache, layer_idx):
    return cache.layers[layer_idx].keys[0, 0, :, 0].tolist()


def test_unbounded_budget_keeps_everything():
    cache = AdmissionCache(budget=None, sinks=4)
    _write(cache, 0, 10)
    cache.apply_admission(0, torch.zeros(10, dtype=torch.bool))
    assert cache.occupancy(0) == 10
    assert _resident_ordinals(cache, 0) == list(range(10))


def test_rejected_tokens_are_removed():
    cache = AdmissionCache(budget=100, sinks=0)
    _write(cache, 0, 6)
    keep = torch.tensor([True, False, True, False, True, False])
    cache.apply_admission(0, keep)
    assert _resident_ordinals(cache, 0) == [0.0, 2.0, 4.0]


def test_occupancy_never_exceeds_budget():
    cache = AdmissionCache(budget=8, sinks=4)
    for _ in range(10):
        n = cache.occupancy(0)
        _write(cache, 0, 5)
        cache.apply_admission(0, torch.ones(5, dtype=torch.bool))
        assert cache.occupancy(0) <= 8, f"overflow after starting from {n}"


def test_sinks_are_never_evicted():
    cache = AdmissionCache(budget=8, sinks=4)
    for _ in range(10):
        _write(cache, 0, 5)
        cache.apply_admission(0, torch.ones(5, dtype=torch.bool))
    assert _resident_ordinals(cache, 0)[:4] == [0.0, 1.0, 2.0, 3.0]


def test_eviction_is_fifo_oldest_non_sink_first():
    cache = AdmissionCache(budget=6, sinks=2)
    _write(cache, 0, 6)
    cache.apply_admission(0, torch.ones(6, dtype=torch.bool))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    _write(cache, 0, 2)
    cache.apply_admission(0, torch.ones(2, dtype=torch.bool))
    # sinks 0,1 pinned; oldest non-sinks 2,3 dropped; 6,7 admitted
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0]


def test_sinks_are_force_admitted_even_if_rejected():
    cache = AdmissionCache(budget=8, sinks=4)
    _write(cache, 0, 6)
    cache.apply_admission(0, torch.zeros(6, dtype=torch.bool))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0]


def test_returned_indices_reindex_parallel_bookkeeping():
    cache = AdmissionCache(budget=100, sinks=0)
    _write(cache, 0, 5)
    scores = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    keep = torch.tensor([True, False, True, False, True])
    surviving = cache.apply_admission(0, keep)
    assert scores[surviving].tolist() == [10.0, 30.0, 50.0]


def test_layers_are_independent():
    cache = AdmissionCache(budget=4, sinks=0)
    _write(cache, 0, 6)
    _write(cache, 1, 6)
    cache.apply_admission(0, torch.ones(6, dtype=torch.bool))
    assert cache.occupancy(0) == 4
    assert cache.occupancy(1) == 6
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/srujanpendurthi/Projects/Current/kvAdmission && python3 -m pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission'`

- [ ] **Step 4: Write the implementation**

Create `kvadmission/__init__.py`:

```python
"""Learned KV admission for the mid-context dead zone."""
```

Create `kvadmission/cache.py`:

```python
"""Budget-capped KV cache supporting write-time admission and score eviction."""

import torch
from transformers import DynamicCache


class AdmissionCache(DynamicCache):
    """KV cache with a per-layer token budget and a pinned sink prefix.

    The model writes every token through the normal `update()` path. After each
    forward pass the caller applies a decision per layer:

    - `apply_admission(layer_idx, keep)` drops the just-written tokens the policy
      rejected, then enforces the budget by evicting the oldest non-sink
      residents (FIFO). This is the mode our learned policy uses.
    - `apply_eviction(layer_idx, scores)` keeps the top-`budget` residents by
      score. This is the mode published baselines (H2O, SnapKV) use.

    Peak memory is `budget + chunk_size`, not the full stream length: tokens are
    written and then removed within a chunk rather than accumulating.

    `budget=None` disables both, making this behave exactly as DynamicCache. That
    is the identity mode the correctness test in Task 3 relies on.
    """

    def __init__(self, budget=None, sinks=4, **kwargs):
        super().__init__(**kwargs)
        self.budget = budget
        self.sinks = sinks

    def occupancy(self, layer_idx=0):
        """Number of tokens currently resident in a layer."""
        if layer_idx >= len(self.layers):
            return 0
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            return 0
        return layer.keys.shape[-2]

    def _reindex(self, layer_idx, surviving):
        layer = self.layers[layer_idx]
        layer.keys = layer.keys[:, :, surviving, :]
        layer.values = layer.values[:, :, surviving, :]

    def _enforce_budget_fifo(self, surviving):
        """Trim to budget by dropping the oldest non-sink entries."""
        if surviving.numel() <= self.budget:
            return surviving
        n_sink = min(self.sinks, surviving.numel())
        room = self.budget - n_sink
        if room <= 0:
            return surviving[:self.budget]
        return torch.cat([surviving[:n_sink], surviving[n_sink:][-room:]])

    def apply_admission(self, layer_idx, keep):
        """Drop rejected new tokens, then FIFO down to budget.

        `keep` is a bool tensor over the tokens written by the last update().
        Returns the surviving positions as indices into the pre-call cache, so
        callers can reindex parallel per-token state with `state[surviving]`.
        """
        n = self.occupancy(layer_idx)
        device = self.layers[layer_idx].keys.device
        idx = torch.arange(n, device=device)

        if self.budget is None:
            return idx

        n_new = keep.shape[0]
        n_old = n - n_new
        keep = keep.to(device)

        # The sink prefix is structural, not a policy choice - force it in.
        if n_old < self.sinks:
            keep = keep.clone()
            keep[: min(self.sinks - n_old, n_new)] = True

        surviving = torch.cat([idx[:n_old], idx[n_old:][keep]])
        surviving = self._enforce_budget_fifo(surviving)
        self._reindex(layer_idx, surviving)
        return surviving
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cache.py -v`
Expected: PASS, 8 passed

- [ ] **Step 6: Commit**

```bash
git add kvadmission/ tests/test_cache.py
git commit -m "feat: AdmissionCache with budget, sinks and FIFO admission"
```

---

### Task 2: Score-based eviction, for reproducing published baselines

**Files:**
- Modify: `kvadmission/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `AdmissionCache` from Task 1
- Produces: `AdmissionCache.apply_eviction(layer_idx: int, scores: torch.Tensor) -> torch.Tensor` — `scores` is a 1-D float tensor with one entry per *resident* token (length == `occupancy(layer_idx)`). Keeps the sink prefix plus the top `budget - sinks` by score, in original position order. Returns surviving indices, same contract as `apply_admission`.

Position order matters: RoPE is baked into cached keys, so residents must stay sorted by original position or attention geometry breaks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
def test_eviction_keeps_highest_scoring():
    cache = AdmissionCache(budget=3, sinks=0)
    _write(cache, 0, 6)
    scores = torch.tensor([1.0, 9.0, 2.0, 8.0, 3.0, 7.0])
    cache.apply_eviction(0, scores)
    assert _resident_ordinals(cache, 0) == [1.0, 3.0, 5.0]


def test_eviction_preserves_position_order():
    cache = AdmissionCache(budget=3, sinks=0)
    _write(cache, 0, 6)
    # highest scores are at the end, but output must stay position-sorted
    scores = torch.tensor([0.0, 0.0, 0.0, 7.0, 8.0, 9.0])
    cache.apply_eviction(0, scores)
    assert _resident_ordinals(cache, 0) == [3.0, 4.0, 5.0]


def test_eviction_pins_sinks_regardless_of_score():
    cache = AdmissionCache(budget=4, sinks=2)
    _write(cache, 0, 6)
    scores = torch.tensor([0.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    cache.apply_eviction(0, scores)
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 4.0, 5.0]


def test_eviction_is_noop_under_budget():
    cache = AdmissionCache(budget=10, sinks=0)
    _write(cache, 0, 4)
    cache.apply_eviction(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0]


def test_eviction_is_noop_when_unbounded():
    cache = AdmissionCache(budget=None, sinks=0)
    _write(cache, 0, 4)
    cache.apply_eviction(0, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert _resident_ordinals(cache, 0) == [0.0, 1.0, 2.0, 3.0]


def test_eviction_indices_reindex_scores():
    cache = AdmissionCache(budget=2, sinks=0)
    _write(cache, 0, 4)
    scores = torch.tensor([1.0, 9.0, 2.0, 8.0])
    surviving = cache.apply_eviction(0, scores)
    assert scores[surviving].tolist() == [9.0, 8.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cache.py -v -k eviction`
Expected: FAIL with `AttributeError: 'AdmissionCache' object has no attribute 'apply_eviction'`

- [ ] **Step 3: Write the implementation**

Add to `kvadmission/cache.py`, inside `AdmissionCache`:

```python
    def apply_eviction(self, layer_idx, scores):
        """Keep the sink prefix plus the top-scoring residents.

        This is the post-hoc mode published baselines use. `scores` has one
        entry per resident token. Survivors stay sorted by original position,
        because RoPE is already baked into the cached keys.
        """
        n = self.occupancy(layer_idx)
        device = self.layers[layer_idx].keys.device
        idx = torch.arange(n, device=device)

        if self.budget is None or n <= self.budget:
            return idx

        n_sink = min(self.sinks, n)
        room = self.budget - n_sink
        if room <= 0:
            surviving = idx[: self.budget]
        else:
            tail = idx[n_sink:]
            top = torch.topk(scores.to(device)[n_sink:], room).indices
            surviving = torch.cat([idx[:n_sink], tail[top.sort().values]])

        self._reindex(layer_idx, surviving)
        return surviving
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cache.py -v`
Expected: PASS, 14 passed

- [ ] **Step 5: Commit**

```bash
git add kvadmission/cache.py tests/test_cache.py
git commit -m "feat: score-based eviction mode for baseline reproduction"
```

---

### Task 3: Chunked prefill with correct positions, and the identity test

**Files:**
- Create: `kvadmission/runner.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: `AdmissionCache` from Tasks 1–2
- Produces:
  - `load_model(name: str = "Qwen/Qwen3-1.7B") -> tuple[model, tokenizer, str]` — returns model, tokenizer, device string. Model is in eval mode with `attn_implementation="eager"`.
  - `prefill(model, ids: torch.Tensor, cache: AdmissionCache, device: str, chunk: int = 128, decide=None, need_attn: bool = False) -> int` — runs chunked prefill, returns the stream length in tokens. `decide` is `None` (no compression) or a callable `decide(layer_idx, cache, attentions, n_new) -> None` invoked once per layer per chunk; it is responsible for calling `apply_admission` or `apply_eviction`.
  - `generate(model, tok, cache, stream_len: int, prompt: str, device: str, max_new: int = 24) -> str` — appends a prompt to the compressed cache, greedy-decodes, then rolls the cache back to its pre-prompt length.

This is the task that pins down `position_ids`. Everything downstream is silently wrong if it is wrong here.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_identity.py`:

```python
import torch

from kvadmission.cache import AdmissionCache
from kvadmission.runner import load_model, prefill, generate

MODEL, TOK, DEVICE = load_model()


def _ids(n_tokens=600):
    text = "The quick brown fox jumps over the lazy dog. " * 200
    ids = TOK(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids[:, :n_tokens]


def _final_logits(cache, chunk):
    ids = _ids()
    prefill(MODEL, ids, cache, DEVICE, chunk=chunk)
    with torch.no_grad():
        out = MODEL(
            ids[:, -1:].to(DEVICE),
            past_key_values=cache,
            position_ids=torch.tensor([[ids.shape[1]]], device=DEVICE),
            attention_mask=torch.ones(
                1, cache.occupancy(0) + 1, device=DEVICE, dtype=torch.long
            ),
            use_cache=True,
        )
    return out.logits[0, -1].float().cpu()


def test_unbounded_cache_is_bit_identical_to_dynamic_cache():
    """The identity test. Same chunking, only the cache class differs."""
    from transformers import DynamicCache

    a = _final_logits(AdmissionCache(budget=None), chunk=128)
    b = _final_logits(DynamicCache(), chunk=128)
    assert torch.equal(a, b), f"max abs diff {(a - b).abs().max().item()}"


def test_chunked_prefill_matches_single_pass():
    """Catches position_ids and attention-mask bugs. bf16 tolerance, not exact."""
    a = _final_logits(AdmissionCache(budget=None), chunk=128)
    b = _final_logits(AdmissionCache(budget=None), chunk=1024)
    assert torch.allclose(a, b, atol=0.05), \
        f"max abs diff {(a - b).abs().max().item()}"


def test_prefill_returns_stream_length():
    ids = _ids(600)
    cache = AdmissionCache(budget=None)
    assert prefill(MODEL, ids, cache, DEVICE, chunk=128) == 600


def test_generate_restores_cache_length():
    ids = _ids(600)
    cache = AdmissionCache(budget=256, sinks=4)

    def decide(layer_idx, cache, attentions, n_new):
        cache.apply_admission(layer_idx, torch.ones(n_new, dtype=torch.bool))

    stream_len = prefill(MODEL, ids, cache, DEVICE, chunk=128, decide=decide)
    before = cache.occupancy(0)
    generate(MODEL, TOK, cache, stream_len, "\n\nQuestion: What?\nAnswer:", DEVICE)
    assert cache.occupancy(0) == before


def test_generate_produces_text():
    ids = _ids(600)
    cache = AdmissionCache(budget=None)
    stream_len = prefill(MODEL, ids, cache, DEVICE, chunk=128)
    text = generate(MODEL, TOK, cache, stream_len, "\n\nThe fox is", DEVICE, max_new=8)
    assert isinstance(text, str) and len(text) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.runner'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/runner.py`:

```python
"""Chunked prefill and generation against a budget-capped cache.

The one thing to get right here is `position_ids`. Cached keys carry RoPE at
their original stream position, and compression makes cache length diverge from
stream position. Letting the model infer positions from cache length silently
corrupts every result, so positions are always passed explicitly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name="Qwen/Qwen3-1.7B"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    return model, tok, device


@torch.no_grad()
def prefill(model, ids, cache, device, chunk=128, decide=None, need_attn=False):
    """Chunked prefill. Returns stream length in tokens.

    `decide(layer_idx, cache, attentions, n_new)` is called once per layer per
    chunk and owns the apply_admission / apply_eviction call. None means no
    compression.
    """
    total = ids.shape[1]
    n_layers = model.config.num_hidden_layers

    for start in range(0, total, chunk):
        piece = ids[:, start:start + chunk].to(device)
        n_new = piece.shape[1]
        out = model(
            piece,
            past_key_values=cache,
            position_ids=torch.arange(
                start, start + n_new, device=device
            ).unsqueeze(0),
            attention_mask=torch.ones(
                1, cache.occupancy(0) + n_new, device=device, dtype=torch.long
            ),
            output_attentions=need_attn,
            use_cache=True,
        )
        if decide is not None:
            for li in range(n_layers):
                decide(li, cache, out.attentions if need_attn else None, n_new)
        del out

    return total


@torch.no_grad()
def generate(model, tok, cache, stream_len, prompt, device, max_new=24):
    """Greedy-decode a prompt against the compressed cache, then roll back."""
    keep = cache.occupancy(0)
    cur = tok(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)

    generated, pos = [], stream_len
    for _ in range(max_new):
        out = model(
            cur,
            past_key_values=cache,
            position_ids=torch.arange(
                pos, pos + cur.shape[1], device=device
            ).unsqueeze(0),
            attention_mask=torch.ones(
                1, cache.occupancy(0) + cur.shape[1], device=device, dtype=torch.long
            ),
            use_cache=True,
        )
        pos += cur.shape[1]
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(cur.item())

    cache.crop(keep)
    return tok.decode(generated)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_identity.py -v`
Expected: PASS, 5 passed. Takes a few minutes — it loads the model and runs several 600-token prefills.

If `test_unbounded_cache_is_bit_identical_to_dynamic_cache` fails, `AdmissionCache` is mutating state when `budget=None`. Fix that before continuing; every downstream number depends on it.

- [ ] **Step 5: Commit**

```bash
git add kvadmission/runner.py tests/test_identity.py
git commit -m "feat: chunked prefill with explicit positions, plus identity test"
```

---

### Task 4: Baseline policies

**Files:**
- Create: `kvadmission/policies.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: `AdmissionCache` from Tasks 1–2
- Produces: `make_decider(policy: str, budget: int, sinks: int = 4, window: int = 32, seed: int = 0) -> tuple[callable, bool]` — returns `(decide, need_attn)`. `decide` has the signature `prefill` expects. `policy` is one of `"full"`, `"recent"`, `"random"`, `"h2o"`, `"snapkv"`.

Policy semantics under this harness:

| policy | mode | rule |
|---|---|---|
| `full` | none | no compression |
| `recent` | admission | admit everything; FIFO keeps the newest B |
| `random` | admission | admit each new token with probability `budget/(budget+chunk)` |
| `h2o` | eviction | accumulated attention across chunks, reindexed through evictions |
| `snapkv` | eviction | attention from the last `window` query rows only |

`h2o` needs per-layer score state that survives eviction — that is what the returned indices from `apply_eviction` are for.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_policies.py`:

```python
import torch

from kvadmission.cache import AdmissionCache
from kvadmission.policies import make_decider


def _fake_attn(n_layers, n_kv, n_query, heads=2):
    """Attention with mass concentrated on kv position 0."""
    a = torch.full((1, heads, n_query, n_kv), 0.01)
    a[:, :, :, 0] = 10.0
    return [a for _ in range(n_layers)]


def test_full_policy_never_compresses():
    decide, need_attn = make_decider("full", budget=4)
    assert decide is None
    assert need_attn is False


def test_recent_admits_everything_and_fifo_trims():
    decide, need_attn = make_decider("recent", budget=6, sinks=2)
    assert need_attn is False
    cache = AdmissionCache(budget=6, sinks=2)
    k = torch.zeros(1, 2, 8, 4)
    k[0, 0, :, 0] = torch.arange(8, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, None, 8)
    assert cache.occupancy(0) == 6
    # sinks 0,1 pinned, newest four kept
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0]


def _run_random(seed):
    decide, _ = make_decider("random", budget=6, sinks=0, seed=seed)
    cache = AdmissionCache(budget=6, sinks=0)
    k = torch.zeros(1, 2, 10, 4)
    k[0, 0, :, 0] = torch.arange(10, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, None, 10)
    return cache.layers[0].keys[0, 0, :, 0].tolist()


def test_random_is_deterministic_under_seed():
    assert _run_random(7) == _run_random(7)


def test_random_respects_budget():
    assert len(_run_random(7)) <= 6


def test_h2o_keeps_the_high_attention_token():
    decide, need_attn = make_decider("h2o", budget=2, sinks=0)
    assert need_attn is True
    cache = AdmissionCache(budget=2, sinks=0)
    k = torch.zeros(1, 2, 4, 4)
    k[0, 0, :, 0] = torch.arange(4, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    decide(0, cache, _fake_attn(1, n_kv=4, n_query=4), 4)
    assert 0.0 in cache.layers[0].keys[0, 0, :, 0].tolist()


def test_h2o_scores_survive_across_chunks():
    """Score bookkeeping must be reindexed through eviction, not reset."""
    decide, _ = make_decider("h2o", budget=3, sinks=0)
    cache = AdmissionCache(budget=3, sinks=0)
    for step in range(3):
        k = torch.zeros(1, 2, 2, 4)
        k[0, 0, :, 0] = torch.arange(step * 2, step * 2 + 2, dtype=torch.float32)
        cache.update(k, k.clone(), 0)
        n_kv = cache.occupancy(0)
        decide(0, cache, _fake_attn(1, n_kv=n_kv, n_query=2), 2)
        assert cache.occupancy(0) <= 3
    # position 0 carried the attention mass every round, so it must still be there
    assert 0.0 in cache.layers[0].keys[0, 0, :, 0].tolist()


def test_snapkv_uses_only_the_observation_window():
    decide, need_attn = make_decider("snapkv", budget=2, sinks=0, window=1)
    assert need_attn is True
    cache = AdmissionCache(budget=2, sinks=0)
    k = torch.zeros(1, 2, 4, 4)
    k[0, 0, :, 0] = torch.arange(4, dtype=torch.float32)
    cache.update(k, k.clone(), 0)
    a = torch.full((1, 2, 4, 4), 0.01)
    a[:, :, -1, 3] = 99.0  # only the final query row attends to position 3
    decide(0, cache, [a], 4)
    assert 3.0 in cache.layers[0].keys[0, 0, :, 0].tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_policies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.policies'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/policies.py`:

```python
"""Baseline compression policies, as deciders for runner.prefill.

Two families. `recent` and `random` are admission policies: they choose which
newly-written tokens enter, and FIFO does the rest. `h2o` and `snapkv` are
eviction policies: they score every resident and keep the top B, which is how
the published methods actually work and how they must be reproduced.
"""

import torch

CHUNK_HINT = 128  # only used to set random's admit rate; see make_decider


def make_decider(policy, budget, sinks=4, window=32, seed=0, chunk=CHUNK_HINT):
    """Build a (decide, need_attn) pair for `runner.prefill`."""
    if policy == "full":
        return None, False

    if policy == "recent":
        def decide(layer_idx, cache, attentions, n_new):
            cache.apply_admission(
                layer_idx, torch.ones(n_new, dtype=torch.bool)
            )
        return decide, False

    if policy == "random":
        gen = torch.Generator().manual_seed(seed)
        rate = budget / (budget + chunk)

        def decide(layer_idx, cache, attentions, n_new):
            keep = torch.rand(n_new, generator=gen) < rate
            cache.apply_admission(layer_idx, keep)
        return decide, False

    if policy in ("h2o", "snapkv"):
        acc = {}

        def decide(layer_idx, cache, attentions, n_new):
            a = attentions[layer_idx].float()          # (1, H, n_query, n_kv)
            if policy == "h2o":
                cur = a.sum(dim=(0, 1, 2)).cpu()
                prev = acc.get(layer_idx)
                if prev is not None:
                    cur[: prev.shape[0]] += prev
                scores = cur
            else:
                scores = a[:, :, -window:, :].sum(dim=(0, 1, 2)).cpu()

            surviving = cache.apply_eviction(layer_idx, scores)
            acc[layer_idx] = scores[surviving.cpu()]
        return decide, True

    raise ValueError(f"unknown policy: {policy}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_policies.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add kvadmission/policies.py tests/test_policies.py
git commit -m "feat: baseline policies - recent, random, h2o, snapkv"
```

---

### Task 5: Needle stream and recall evaluation

**Files:**
- Create: `kvadmission/stream.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `runner.generate` from Task 3
- Produces:
  - `build_stream(tok, n_needles: int, target_len: int, seed: int = 0) -> tuple[torch.Tensor, list[dict]]` — returns `(ids, needles)` where each needle is `{"topic": str, "code": str, "pos": int}` and `pos` is the token index just past the planted fact.
  - `query_needle(model, tok, cache, stream_len: int, needle: dict, device: str) -> tuple[bool, str]` — returns `(recalled, generated_text)`.

Unlike the spike, needle placement is seed-varied so the sweep can produce error bars. Placement jitter is what stops a single unlucky layout from being mistaken for a horizon.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stream.py`:

```python
import torch
from transformers import AutoTokenizer

from kvadmission.stream import build_stream

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_stream_has_requested_length():
    ids, _ = build_stream(TOK, n_needles=4, target_len=2048)
    assert ids.shape == (1, 2048)


def test_all_needles_are_placed_and_distinct():
    ids, needles = build_stream(TOK, n_needles=6, target_len=4096)
    assert len(needles) == 6
    assert len({n["topic"] for n in needles}) == 6
    assert len({n["code"] for n in needles}) == 6


def test_needle_positions_are_increasing_and_in_range():
    ids, needles = build_stream(TOK, n_needles=6, target_len=4096)
    positions = [n["pos"] for n in needles]
    assert positions == sorted(positions)
    assert all(0 < p <= 4096 for p in positions)


def test_needle_text_is_actually_present_in_the_stream():
    ids, needles = build_stream(TOK, n_needles=3, target_len=2048)
    text = TOK.decode(ids[0])
    for n in needles:
        assert n["code"] in text
        assert n["topic"] in text


def test_seed_changes_placement_but_not_count():
    _, a = build_stream(TOK, n_needles=5, target_len=4096, seed=1)
    _, b = build_stream(TOK, n_needles=5, target_len=4096, seed=2)
    assert len(a) == len(b) == 5
    assert [n["pos"] for n in a] != [n["pos"] for n in b]


def test_seed_is_reproducible():
    _, a = build_stream(TOK, n_needles=5, target_len=4096, seed=3)
    _, b = build_stream(TOK, n_needles=5, target_len=4096, seed=3)
    assert [n["pos"] for n in a] == [n["pos"] for n in b]
    assert [n["code"] for n in a] == [n["code"] for n in b]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stream.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.stream'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/stream.py`:

```python
"""Needle-in-haystack stream construction and recall scoring.

Needles are planted at evenly spaced anchors with per-seed jitter, so repeated
seeds vary placement. Without jitter every seed measures the same layout and the
error bars would describe generation noise rather than placement sensitivity.
"""

import random
import re
import urllib.request
from pathlib import Path

import torch

from kvadmission.runner import generate

GUTENBERG = "https://www.gutenberg.org/files/1342/1342-0.txt"
CORPUS = Path(__file__).parent / "_haystack.txt"

TOPICS = [
    "oak", "harbor", "lantern", "quarry", "meridian", "thistle",
    "cobalt", "vellum", "cinder", "marrow", "fathom", "sable",
    "juniper", "ashen", "gallows", "wren", "pumice", "clover",
    "bramble", "onyx", "tallow", "sextant", "harrow", "plinth",
]


def _corpus():
    if not CORPUS.exists():
        raw = urllib.request.urlopen(GUTENBERG).read().decode("utf-8", "ignore")
        CORPUS.write_text(raw)
    return CORPUS.read_text()


def build_stream(tok, n_needles, target_len, seed=0):
    """Plant `n_needles` facts through `target_len` tokens of prose."""
    if n_needles > len(TOPICS):
        raise ValueError(f"only {len(TOPICS)} topics available")

    rng = random.Random(seed)
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", _corpus())
        if 40 < len(s.strip()) < 400
    ]
    rng.shuffle(sentences)

    topics = rng.sample(TOPICS, n_needles)
    needles = [
        {"topic": t, "code": f"{rng.randrange(100000, 999999)}"} for t in topics
    ]

    # even anchors, jittered within their own slot so seeds differ in placement
    slot = target_len / n_needles
    anchors = sorted(
        int(i * slot + rng.uniform(0.2, 0.8) * slot) for i in range(n_needles)
    )

    ids, si, placed = [], 0, 0
    while len(ids) < target_len:
        while placed < n_needles and len(ids) >= anchors[placed]:
            n = needles[placed]
            ids += tok(
                f" The secret code for {n['topic']} is {n['code']}. ",
                add_special_tokens=False,
            )["input_ids"]
            n["pos"] = len(ids)
            placed += 1
        ids += tok(" " + sentences[si % len(sentences)],
                   add_special_tokens=False)["input_ids"]
        si += 1

    # a needle anchored near the tail can be pushed past the cut - place it just inside
    for n in needles:
        if "pos" not in n or n["pos"] > target_len:
            n["pos"] = target_len

    return torch.tensor([ids[:target_len]]), needles


def query_needle(model, tok, cache, stream_len, needle, device):
    """Ask for one needle's code against the compressed cache."""
    prompt = (
        f"\n\nQuestion: What is the secret code for {needle['topic']}?\nAnswer:"
    )
    text = generate(model, tok, cache, stream_len, prompt, device)
    return needle["code"] in text, text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stream.py -v`
Expected: PASS, 6 passed. The first run downloads the corpus.

- [ ] **Step 5: Add the corpus to gitignore and commit**

```bash
cd /Users/srujanpendurthi/Projects/Current/kvAdmission
printf '\n# Downloaded corpus for the needle harness\nkvadmission/_haystack.txt\n' >> .gitignore
git add kvadmission/stream.py tests/test_stream.py .gitignore
git commit -m "feat: seeded needle stream with jittered placement"
```

---

### Task 6: Sweep driver with multiple seeds

**Files:**
- Create: `kvadmission/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5
- Produces:
  - `run_cell(model, tok, device, policy: str, budget: int, seed: int, length: int, n_needles: int, chunk: int = 128) -> list[dict]` — one policy/budget/seed combination. Each row is `{"policy", "budget", "seed", "topic", "distance_back", "hit", "gen"}`.
  - `summarize(rows: list[dict]) -> dict` — maps `(policy, budget, band)` to `{"mean": float, "n": int, "stderr": float}` where `band` is one of `"near"`, `"middle"`, `"far"` by distance tercile.
  - CLI: `python3 -m kvadmission.sweep --seeds 5 --length 8192 --needles 8`

The banding is the point. The spike's claim is specifically about the middle tercile, so the summary has to report it directly rather than leaving it to be eyeballed off a grid.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep.py`:

```python
from kvadmission.sweep import summarize


def _row(policy, budget, seed, d, hit):
    return {"policy": policy, "budget": budget, "seed": seed,
            "topic": "oak", "distance_back": d, "hit": hit, "gen": ""}


def test_summarize_bands_by_distance_tercile():
    rows = [
        _row("recent", 2048, 0, 100, True),    # near
        _row("recent", 2048, 0, 4000, False),  # middle
        _row("recent", 2048, 0, 7900, False),  # far
    ]
    out = summarize(rows)
    assert out[("recent", 2048, "near")]["mean"] == 1.0
    assert out[("recent", 2048, "middle")]["mean"] == 0.0
    assert out[("recent", 2048, "far")]["mean"] == 0.0


def test_summarize_averages_across_seeds():
    rows = [
        _row("h2o", 1024, 0, 100, True),
        _row("h2o", 1024, 1, 100, False),
    ]
    out = summarize(rows)
    cell = out[("h2o", 1024, "near")]
    assert cell["mean"] == 0.5
    assert cell["n"] == 2


def test_summarize_reports_stderr():
    rows = [_row("h2o", 1024, s, 100, s % 2 == 0) for s in range(4)]
    cell = summarize(rows)[("h2o", 1024, "near")]
    assert cell["n"] == 4
    assert 0.0 < cell["stderr"] < 0.5


def test_stderr_is_zero_when_unanimous():
    rows = [_row("recent", 512, s, 100, True) for s in range(4)]
    assert summarize(rows)[("recent", 512, "near")]["stderr"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kvadmission.sweep'`

- [ ] **Step 3: Write the implementation**

Create `kvadmission/sweep.py`:

```python
"""Sweep driver: policies x budgets x seeds, summarized by distance band."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from kvadmission.cache import AdmissionCache
from kvadmission.policies import make_decider
from kvadmission.runner import load_model, prefill
from kvadmission.stream import build_stream, query_needle

BANDS = ("near", "middle", "far")


def _band(distance, stream_len):
    third = stream_len / 3
    if distance < third:
        return "near"
    return "middle" if distance < 2 * third else "far"


def run_cell(model, tok, device, policy, budget, seed, length, n_needles,
             chunk=128):
    """One policy/budget/seed combination. Returns one row per needle."""
    ids, needles = build_stream(tok, n_needles, length, seed=seed)
    decide, need_attn = make_decider(
        policy, budget=budget, sinks=4, seed=seed, chunk=chunk
    )
    cache = AdmissionCache(
        budget=None if policy == "full" else budget, sinks=4
    )
    stream_len = prefill(
        model, ids, cache, device, chunk=chunk, decide=decide,
        need_attn=need_attn,
    )

    rows = []
    for n in needles:
        hit, text = query_needle(model, tok, cache, stream_len, n, device)
        rows.append({
            "policy": policy, "budget": budget, "seed": seed,
            "topic": n["topic"], "distance_back": stream_len - n["pos"],
            "hit": bool(hit), "gen": text[:60],
        })
    return rows


def summarize(rows):
    """Recall by (policy, budget, distance band), averaged across seeds."""
    stream_len = max(r["distance_back"] for r in rows) if rows else 1
    buckets = defaultdict(list)
    for r in rows:
        key = (r["policy"], r["budget"], _band(r["distance_back"], stream_len))
        buckets[key].append(1.0 if r["hit"] else 0.0)

    out = {}
    for key, vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        sd = statistics.stdev(vals) if n > 1 else 0.0
        out[key] = {"mean": mean, "n": n, "stderr": sd / (n ** 0.5) if n else 0.0}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--needles", type=int, default=8)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[256, 512, 1024, 2048])
    p.add_argument("--policies", nargs="+",
                   default=["full", "recent", "random", "h2o", "snapkv"])
    p.add_argument("--out", default="results/phase1.json")
    a = p.parse_args()

    model, tok, device = load_model(a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(a.seeds):
        for policy in a.policies:
            budgets = [a.length] if policy == "full" else a.budgets
            for b in budgets:
                rows += run_cell(model, tok, device, policy, b, seed,
                                 a.length, a.needles, a.chunk)
                hits = sum(r["hit"] for r in rows[-a.needles:])
                print(f"seed={seed} {policy:8s} B={b:<6d} "
                      f"{hits}/{a.needles}", flush=True)
                Path(a.out).write_text(json.dumps(rows, indent=1))

    print("\nrecall by band (mean +/- stderr)")
    for key, cell in sorted(summarize(rows).items()):
        policy, budget, band = key
        print(f"  {policy:8s} B={budget:<6d} {band:7s} "
              f"{cell['mean']:.2f} +/- {cell['stderr']:.2f}  (n={cell['n']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sweep.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 30 passed

- [ ] **Step 6: Commit**

```bash
git add kvadmission/sweep.py tests/test_sweep.py
git commit -m "feat: multi-seed sweep driver with distance-band summary"
```

---

### Task 7: Replicate the dead zone at scale

**Files:**
- Create: `results/phase1.json` (generated)
- Create: `docs/phase1-findings.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the `kvadmission.sweep` CLI from Task 6
- Produces: `docs/phase1-findings.md` recording whether the mid-context dead zone survives multi-seed replication.

This task has no TDD cycle — it runs an experiment and records the answer. Its gate is that the answer is written down honestly, including if the answer is no.

- [ ] **Step 1: Smoke-test the driver at small scale**

```bash
cd /Users/srujanpendurthi/Projects/Current/kvAdmission
python3 -m kvadmission.sweep --seeds 1 --length 2048 --needles 4 \
  --budgets 256 512 --policies full recent --out results/smoke.json
```

Expected: `full` recalls 4/4. If it does not, the task is not solvable at this length and nothing below is interpretable — stop and investigate before running the full sweep.

- [ ] **Step 2: Run the full sweep in the background**

```bash
python3 -u -m kvadmission.sweep --seeds 5 --length 8192 --needles 8 \
  --out results/phase1.json > results/phase1.log 2>&1 &
```

Expect roughly 5x the spike's runtime — order of 3–4 hours. It checkpoints `results/phase1.json` after every cell, so a crash loses at most one cell.

- [ ] **Step 3: Summarize when it finishes**

```bash
python3 -c "
import json
from kvadmission.sweep import summarize
rows = json.load(open('results/phase1.json'))
for key, cell in sorted(summarize(rows).items()):
    p, b, band = key
    print(f'{p:8s} B={b:<6d} {band:7s} {cell[\"mean\"]:.2f} +/- {cell[\"stderr\"]:.2f} (n={cell[\"n\"]})')
"
```

- [ ] **Step 4: Write the findings document**

Create `docs/phase1-findings.md` answering exactly these questions, with numbers:

1. Does `full` recall stay at or near 1.00 across all bands? (If not, nothing else is interpretable.)
2. What is each baseline's mean recall in the **middle** band, with stderr?
3. Is any baseline's middle-band recall distinguishable from zero at 2 stderr?
4. Does the near/far complementarity hold — `recent` strong in `near`, `h2o` strong in `far`?
5. Does the geometric-survival prediction `(B/(B+chunk))^(d/chunk)` still track `random`'s observed cutoff?

State plainly whether the dead zone replicated. **If any baseline achieves meaningful middle-band recall, the spec's central premise is weakened and phase 2 must be re-scoped before it starts.** Record that outcome rather than working around it.

- [ ] **Step 5: Commit results and findings**

```bash
printf '\n# Sweep logs - the JSON carries the data\nresults/*.log\nresults/smoke.json\n' >> .gitignore
git add results/phase1.json docs/phase1-findings.md .gitignore
git commit -m "results: phase 1 multi-seed replication of the mid-context dead zone"
```

---

## Self-Review

**Spec coverage.** §4.1 admission rule with FIFO and sinks → Task 1. Baseline reproduction requiring eviction mode → Task 2. §6 correctness checks: identity test → Task 3, occupancy bound and sink retention → Task 1, `position_ids` → Task 3. §5 baselines → Task 4. Needle evaluation → Task 5. §8 "phase 1 must re-establish the dead zone with error bars" → Tasks 6–7.

**Deliberately out of scope**, deferred to the phase 2 plan: the policy head (§4.2), the reward and Lagrangian (§4.3), the WG-KV head-to-head (§4.4), the λ-convergence check from §6, and RULER/LongBench/HELMET integration. None of those can be specified honestly until Task 7 reports.

**Type consistency.** `apply_admission` and `apply_eviction` both take `layer_idx` first and both return surviving indices into the pre-call cache. `decide(layer_idx, cache, attentions, n_new)` is identical across `policies.py` and `runner.py`. `budget=None` means unbounded everywhere. Needle dicts use `pos` in both `stream.py` and `sweep.py`.

**Note on peak memory.** The spec says admission saves peak memory. This harness writes tokens then removes them, so peak within a chunk is `budget + chunk`, not `budget`. That is still bounded and independent of stream length, but it is not the stronger claim — a true fused admission path would require patching attention internals, which is not worth it before the method is proven.
