# Learned KV Admission for the Mid-Context Dead Zone

**Date:** 2026-08-27
**Status:** design, pre-implementation
**Supersedes:** the initial "reward functions for continual RL on context compression" framing, retired after the literature review and spike below.

---

## 1. Problem

A long-context model writes every token into its KV cache. Under a memory budget, most of it must go. Existing methods decide *what to discard* by ranking tokens with attention-derived heuristics.

Those heuristics encode **position**, not utility. The spike in `spike/` measured this directly: across every published-style heuristic tested, at every budget tested, the middle of the context was unrecoverable. Methods succeed at one end of the stream or the other, according to which positional bias they happen to carry.

This design targets that gap: **a learned admission policy that decides, at write time, whether a token enters the cache at all, trained to recover information in the band where existing heuristics score zero.**

## 2. Motivating evidence

From `spike/report.html` — Qwen3-1.7B, 8192-token stream, 8 planted facts, recall by distance-back from the query.

| policy | budget | d=448 | d=1474 | d=2521 | d=3565 | d=4553 | d=5603 | d=6598 | d=7657 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 8192 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 8/8 |
| recent | 2048 | ✓ | ✓ | | | | | | | 2/8 |
| random | 2048 | ✓ | ✓ | | | | | | | 2/8 |
| snapkv | 2048 | ✓ | | | | | | | | 1/8 |
| h2o | 2048 | | | | | | | | ✓ | 1/8 |
| h2o (prune-once) | 2048 | | | | | | | ✓ | ✓ | 2/8 |

Three facts follow, and they set the whole design.

**2.1 The dead zone.** Distances 2521 through 5603 — half the needles — were recovered by *no policy at any budget*. `recent` covers the newest, `h2o` covers the oldest, and both score 2/8 by opposite accident. The headroom against full cache (8/8) is not marginal.

**2.2 Recency is emergent, not chosen.** `random` scored identically to `recent` at every budget. Because compression runs every chunk, the cache never exceeds `B + chunk`, so each round discards only `chunk` of `B + chunk` tokens. A token at distance `d` faces `d/chunk` rounds and survives with probability

```
P(survive) = ( B / (B + chunk) ) ^ (d / chunk)
```

At B=2048, chunk=128 this is `0.941^(d/128)`, predicting 0.81 at d=448, 0.50 at d=1474, 0.30 at d=2521 — which matches the observed cutoff exactly. **Any policy that does not actively protect distant tokens decays into recency for free.** The task is not ranking; it is rescuing tokens from an exponential already running.

**2.3 The compression schedule is not the story.** Streaming versus one-shot compression differed by a single needle, in both directions (h2o +1 at B=2048, snapkv −1 at both budgets). The episodic-vs-continuing gap this project originally proposed to exploit **is not demonstrated**, and no claim rests on it.

## 3. Prior work and the honest gap

| Work | What it does | Why this is not that |
|---|---|---|
| [WG-KV / KV Admission](https://arxiv.org/abs/2512.17452) (Dec 2025) | Write-time admission gate, frozen backbone, small MLP | Differentiable gating + supervised distillation loss; fixed λ swept by hand. No sequential decision-making. |
| [Learning to Evict](https://arxiv.org/abs/2602.10238) (Apple, ICML 2026) | RL, frozen LLM, per-head agents, REINFORCE | Post-hoc **eviction** after prefill, episodic, uniform fixed budgets, reward = future attention mass |
| [TriRoute](https://arxiv.org/pdf/2607.06601) (2026) | Online Lagrangian dual ascent for KV allocation | Budget allocation across resources, not admission |
| H2O, SnapKV, StreamingLLM, PyramidKV, Ada-KV | Attention/recency heuristics, layer budget allocation | Training-free; these are the baselines |
| [Hybrid sparse + learnable eviction](https://arxiv.org/pdf/2510.20787) | Hybrid linear+full attention, learned eviction | Supervised, eviction-side |

**The unoccupied cell:** RL applied to write-time *admission* rather than post-hoc eviction. Every neighbor is published; the intersection is not.

**Claims deliberately not made:** that RL beats supervised for cache decisions (Apple's ablation already shows this for eviction); that the Lagrangian shadow price is novel (TriRoute); that hybrid architectures are unexplored (2510.20787); that this is "continual RL" (it is streaming/non-episodic RL, and the continual-learning community means something else by the term).

**Primary contribution as stated:** a characterization of the mid-context dead zone, the geometric-survival mechanism that produces it, and an RL admission policy that recovers information in that band.

## 4. Method

### 4.1 Admission rule

Hard cap `B` tokens per full-attention layer. Each arriving token gets an admit/skip decision. **If admitted while the cache is full, it evicts the oldest resident (FIFO).**

FIFO is deliberate. It makes admissions *coupled* — admitting a low-value token physically displaces a high-value one — which is the sequential structure that justifies RL over per-token supervised labels. It also means the policy fights the geometric decay of §2.2 directly: refusing admission is how a distant token survives another round.

First 4 tokens are permanent sinks (StreamingLLM; settled, not re-derived).

> `ponytail:` FIFO eviction. Learned eviction only if an ablation shows FIFO is the binding constraint.

### 4.2 Policy

A small head reading what the frozen model already computes at each full-attention layer:

- hidden state `h_t^l`, projected to 128
- `‖K_t^l‖`, and the token's attention mass within its own prefill chunk
- recency, current occupancy, current `λ_t`
- layer embedding — one head shared across all full-attention layers

~1–3M parameters, shared value function. **The base model is frozen; no gradient flows through it.** Training cost is rollout inference plus gradients on the head.

#### The layer-alignment constraint

An earlier draft of this section specified *factored independent Bernoulli per
token per layer*. That is not implementable in the phase-1 harness, and the
constraint is structural rather than incidental:

**A single `attention_mask` is applied to every layer in one forward pass, so
all layers must hold the same number of tokens.** A policy deciding
independently per layer desynchronises their lengths, and the forward pass dies
with a shape mismatch inside the attention kernel. This was found the hard way —
the `random` baseline drew per layer and crashed at 215-vs-218 tokens.

The discovery is not purely mechanical. In the spike, `random` drew
independently per layer, which handed every token ~28 independent survival
draws per chunk and materially inflated its recall. Per-layer independence
changes the effective survival statistics, so it is a modelling choice with
empirical consequences, not a free implementation detail.

**Decision: one admission decision per token, shared across all full-attention
layers, for phase 2.** The layer embedding stays in the feature set — the head
still *sees* per-layer state and can weight it — but emits a single decision.

Rationale: the contribution is mid-context recall where baselines score zero.
Whether the decision is made once per token or independently per layer is an
orthogonal refinement, and forcing it onto the critical path buys engineering
risk before the central claim is tested.

Cost, stated plainly: WG-KV reports that token utility is head-specific, so a
shared decision is a genuine handicap and a reviewer may say so. That makes
per-layer decisions a **planned phase-4 ablation**, not an abandoned idea.

Path if per-layer proves necessary: transformers already computes per-layer mask
sizes for hybrid caches, where sliding-window and full-attention layers hold
different lengths by design (`DynamicLayer.get_mask_sizes`). Reusing that
machinery is the principled route, and it is far less work than a bespoke
attention implementation. It does forfeit the peak-memory benefit, since nothing
can be physically removed while layers must stay dimensionally uniform.

### 4.3 Reward

```
r_t = − KL( p_full ‖ p_compressed )  −  λ_t · (occupancy_t / B)

λ_{t+1} = max(0, λ_t + η · (occupancy_t − target))
```

Dense distributional matching against a full-cache reference pass. The Lagrangian multiplier is a standard technique used here for its practical benefit — memory as a constraint rather than a hand-swept penalty — not claimed as novel.

Average-reward (differential) TD rather than discounted return: `δ = r_t − r̄ + V(s_{t+1}) − V(s_t)`. This is the correct formulation for a continuing task, but §2.3 means it is a formulation choice, not a result.

Plain policy gradient with a value baseline first. No PPO until simple PG demonstrably fails.

### 4.4 Why RL rather than WG-KV's differentiable gate

The honest answer is that this must be *shown*, not asserted, and it is the project's main risk. The argument to test: admission decisions under a FIFO budget are not independent — whether token B is worth keeping depends on whether A was admitted — so a per-token differentiable gate optimizing a distillation loss is mis-specified in a way a sequential policy is not. **If the head-to-head against WG-KV loses, that is the finding, and it gets reported.**

## 5. Experiments

**Baselines:** full cache (ceiling) · random at matched budget · StreamingLLM · H2O · SnapKV · **WG-KV-style differentiable gate** (the one that matters) · Apple's RL evictor if reproducible.

**Benchmarks:** RULER, NIAH variants, LongBench-v2, HELMET.

**The headline experiment:** recall as a function of distance-back, at matched budget, versus all baselines. Success = nonzero recall in the band where every baseline is at zero.

**Ablations:** learned λ vs fixed · average-reward vs discounted · FIFO vs learned eviction · transfer of a policy trained on one model to another.

Two separate layer axes, easily confused:

- **Shared head vs per-layer heads** — a parameterization choice. One head with a layer embedding, versus one head per full-attention layer. Cheap either way, no harness change.
- **Shared decision vs per-layer decisions** — constrained by §4.2's layer-alignment requirement. Needs the hybrid-cache masking path before it can be run at all, and forfeits the peak-memory benefit. Phase 4.

## 6. Correctness checks

Assert-based, no framework:

- `B = ∞` → compressed logits bit-identical to full cache (catches every cache-plumbing bug)
- occupancy never exceeds `B`
- sinks always resident
- λ drives occupancy to target on a synthetic stream
- position_ids reflect true stream position, not compressed cache length (the classic eviction bug; already hit and handled in the spike)

## 7. Phases

| # | Work | Gate |
|---|---|---|
| 0 | `AdmissionCache` + identity test | logits bit-identical at `B=∞` |
| 1 | Reproduce StreamingLLM/H2O published numbers | harness validated against literature |
| 2 | Policy + reward, train on Qwen3-1.7B | nonzero mid-band recall |
| 3 | Head-to-head vs WG-KV differentiable gate | the go/no-go for the central claim |
| 4 | Ablations | |
| 5 | Scale-up on rented GPU | 27B, and the hybrid-architecture question |

Phase 3 is the decision point. If RL does not beat the differentiable gate, stop and write up the dead-zone characterization alone — that result stands on its own.

## 8. Constraints and risks

- **Hardware.** Development machine is an M4 with 24 GB. The 27B checkpoint at `../Qwen3.8-27B-CL/` is bf16 (~54 GB) and **missing shards 14–18**; the only complete local copy is a 15 GB Q4_K_M GGUF in the HF cache, which cannot be used because neither llama.cpp nor MLX exposes per-layer KV interception. All development runs on Qwen3-1.7B in HF transformers on MPS. Code stays model-size-agnostic so phase 5 is a config change.
- **Novelty risk.** The gap is narrow. If a 2026 paper lands on RL-for-admission mid-project, the dead-zone characterization and geometric-survival mechanism remain publishable independently.
- **Spike caveats.** §2 rests on one model, one seed, n=8 per cell, simplified H2O/SnapKV reimplementations, and a synthetic verbatim-recall task. **Phase 1 must re-establish the dead zone at proper scale with error bars before phase 2 depends on it.**

## 9. Out of scope

Quantization; retrieval over external stores; text-level prompt compression; training or finetuning the base model; the linear-attention state-gate analogue (revisit at phase 5).
