"""Graded eviction: an entry loses precision step by step before it leaves.

    fp16 -> int8 -> int4 -> int2 -> evicted

The budget is in bits, expressed as fp16-token equivalents so it compares
directly with `AdmissionCache(budget=B)`: B fp16 tokens, or ~1.8B int8 tokens,
~3.2B int4, ~5.3B int2 (each quantised group of 32 values carries an fp16
scale and zero, so int-b costs b+1 bits a value).

Which entry steps down next is the policy:

  uniform   every token is written at one level and FIFO-evicted. The control:
            it gets the whole "more tokens per byte" gain with no priority.
  age       the ladder, oldest chunk stepped first
  kl        the ladder, the chunk whose next step moves the model's prediction
            of the incoming text least stepped first

Steps act on whole written chunks and on every layer at once, so all layers
keep the same token count; the harness uses one global attention mask.
Quantisation is simulated (quantise, then dequantise in place): this measures
recall at a given size, not speed.
"""

import torch
import torch.nn.functional as F

from kvadmission.cache import AdmissionCache

BITS = (16, 8, 4, 2, 1)
EVICTED = len(BITS)
GROUP = 32
# bits per stored value at each level; evicted costs nothing
COST = torch.tensor([16.0] + [b + 32 / GROUP for b in BITS[1:]] + [0.0])


def fake_quant(x, bits, group=GROUP):
    """Asymmetric per-group quantise-dequantise along the last dim.

    `group=None` makes the whole last dim one group.
    """
    if bits >= 16:
        return x
    shape = x.shape
    group = group or shape[-1]
    g = x.float().reshape(*shape[:-1], shape[-1] // group, group)
    lo = g.amin(-1, keepdim=True)
    scale = ((g.amax(-1, keepdim=True) - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    q = ((g - lo) / scale).round().clamp(0, 2 ** bits - 1)
    return (q * scale + lo).reshape(shape).to(x.dtype)


class GradedCache(AdmissionCache):
    """Per-token precision levels, shared across layers, under a bit budget."""

    def __init__(self, budget, sinks=4, key_per_channel=True, **kwargs):
        super().__init__(budget=None, sinks=sinks, **kwargs)
        self.key_per_channel = key_per_channel
        self.capacity = budget * 16.0
        self.level = torch.zeros(0, dtype=torch.long)
        self.pos = torch.zeros(0, dtype=torch.long)

    def used(self):
        return COST[self.level].sum().item()

    def on_write(self, start, n_new, level=0, merge=1):
        """Record the chunk the forward pass just appended, at `level`.

        `merge` > 1 averages each run of `merge` consecutive tokens into one
        entry first (CaM/D2O-style), trading fidelity for tokens per byte. The
        chunk holding the sinks is never merged.

        ponytail: a merged entry is not up-weighted in softmax (it should count
        `merge` times, a log(merge) logit bias); add one if merging looks close.
        """
        if merge > 1 and start >= self.sinks:
            n = self.occupancy(0)
            for layer in self.layers:
                for name in ("keys", "values"):
                    t = getattr(layer, name)
                    new = t[:, :, n - n_new:]
                    new = new.reshape(*new.shape[:2], n_new // merge, merge,
                                      new.shape[-1]).mean(3)
                    setattr(layer, name, torch.cat([t[:, :, :n - n_new], new], 2))
            n_new //= merge
            positions = torch.arange(start, start + n_new * merge, merge)
        else:
            positions = torch.arange(start, start + n_new)
        self.level = torch.cat([self.level, torch.full((n_new,), level)])
        self.pos = torch.cat([self.pos, positions])
        if level:
            n = self.occupancy(0)
            idx = torch.arange(n - n_new, n)
            self._quantise(idx[self.pos[idx] >= self.sinks], BITS[level])

    def _quantise(self, idx, bits):
        """Values per token; keys per channel across the chunk, as KIVI does.

        Keys have outlier channels, so per-token int2 keys wreck attention (a
        run with them decoded "that that that" for every needle).
        """
        for layer in self.layers:
            dev_idx = idx.to(layer.keys.device)
            k = layer.keys[:, :, dev_idx]
            layer.keys[:, :, dev_idx] = (
                fake_quant(k.transpose(-1, -2), bits, None).transpose(-1, -2)
                if self.key_per_channel else fake_quant(k, bits))
            layer.values[:, :, dev_idx] = fake_quant(layer.values[:, :, dev_idx], bits)

    def step(self, idx):
        """Take the residents at cache indices `idx` one level down."""
        nxt = self.level[idx[0]].item() + 1
        if nxt < EVICTED:
            self._quantise(idx, BITS[nxt])
            self.level[idx] = nxt
        else:
            self.evict(idx)

    def evict(self, idx):
        keep = torch.ones(self.level.shape[0], dtype=torch.bool)
        keep[idx] = False
        surviving = keep.nonzero().squeeze(1)
        for li in range(len(self.layers)):
            self._reindex(li, surviving.to(self.layers[li].keys.device))
        self.level, self.pos = self.level[surviving], self.pos[surviving]

    def groups(self, chunk):
        """Written chunks still resident, oldest first, as {chunk_id: indices}.

        The sink prefix is never a candidate.
        """
        ids = self.pos // chunk
        live = self.pos >= self.sinks
        return {g: (live & (ids == g)).nonzero().squeeze(1)
                for g in ids[live].unique().tolist()}


@torch.no_grad()
def _probe_logprobs(model, cache, probe, start, device, mask_idx=None):
    """Log-probs of teacher-forced `probe` tokens against the cache, then undo."""
    n = cache.get_seq_length()
    mask = torch.ones(1, n + probe.shape[1], device=device, dtype=torch.long)
    if mask_idx is not None:
        mask[0, mask_idx.to(device)] = 0
    out = model(probe.to(device), past_key_values=cache,
                position_ids=torch.arange(start, start + probe.shape[1],
                                          device=device).unsqueeze(0),
                attention_mask=mask, use_cache=True)
    cache.crop(n)
    return F.log_softmax(out.logits[0].float(), -1)


def _kl(ref, logp):
    return (ref.exp() * (ref - logp)).sum(-1).mean().item()


@torch.no_grad()
def step_kl(model, cache, idx, probe, start, device, ref):
    """KL(prediction now || prediction with `idx` one level down), in nats/token.

    Eviction is tried by masking; a quantise step by quantising in place,
    probing and restoring the saved slice.
    """
    if cache.level[idx[0]].item() + 1 == EVICTED:
        return _kl(ref, _probe_logprobs(model, cache, probe, start, device, idx))
    saved = [(l.keys[:, :, idx.to(l.keys.device)].clone(),
              l.values[:, :, idx.to(l.keys.device)].clone()) for l in cache.layers]
    cache._quantise(idx, BITS[cache.level[idx[0]].item() + 1])
    kl = _kl(ref, _probe_logprobs(model, cache, probe, start, device))
    for l, (k, v) in zip(cache.layers, saved):
        d = idx.to(l.keys.device)
        l.keys[:, :, d], l.values[:, :, d] = k, v
    return kl


def compact(model, cache, target, policy, chunk, probe, start, device,
            eps=0.0, scores=None, kl_scores=None):
    """Step chunks down, lowest score first, until `used() <= target`.

    Past the target, keeps stepping any chunk whose KL is below `eps` (free
    memory). `scores` carries KL estimates between calls, keyed by
    (chunk_id, level); a chunk is re-probed only after it changes level.

    ponytail: KL of each chunk is measured alone and against the pre-round
    cache; interactions between chunks stepped in one round are ignored.
    """
    if policy == "uniform":
        # token-granular FIFO, oldest non-sink first, so small budgets are not
        # rounded down to whole chunks. With `scores`, lowest-scoring first
        # instead: uniform storage, but a signal choosing what to drop.
        live = (cache.pos >= cache.sinks).nonzero().squeeze(1)
        if scores is not None:
            live = live[torch.argsort(scores[cache.pos[live]])]
        spend = torch.cumsum(COST[cache.level[live]], 0)
        k = int((spend < cache.used() - target).sum().item()) + 1
        cache.evict(live[:min(k, live.numel())])
        return

    scores = {} if kl_scores is None else kl_scores
    ref = None
    while cache.used() > target or eps > 0:
        groups = cache.groups(chunk)
        order = []
        for g, idx in groups.items():
            key = (g, cache.level[idx[0]].item())
            if policy == "age":
                # whole cache down a rung before anyone drops two; oldest first
                # within a rung. Ranking on age alone walks the oldest chunk
                # through every rung each round: FIFO with a quantised tail.
                s = (key[1], g)
            else:
                if key not in scores:
                    if ref is None:
                        ref = _probe_logprobs(model, cache, probe, start, device)
                    scores[key] = step_kl(model, cache, idx, probe, start,
                                          device, ref)
                s = scores[key]
            order.append((s, g))
        order.sort()

        stepped = False
        for s, g in order:
            over = cache.used() > target
            if not over and not (policy == "kl" and s < eps):
                break
            idx = cache.groups(chunk)[g]
            scores.pop((g, cache.level[idx[0]].item()), None)
            cache.step(idx)
            stepped = True
        if not stepped:
            return
        ref = None  # the cache changed; later probes need a fresh reference


@torch.no_grad()
def graded_prefill(model, ids, cache, device, policy, chunk=128, probe=64,
                   write_level=0, eps=0.0, refresh=8, merge=1, scores=None):
    """Chunked prefill under a bit budget. Returns stream length in tokens.

    Before each chunk is written the cache is compacted to leave room for it,
    so `used() <= capacity` holds after every write. The probe is the first
    `probe` tokens of that incoming chunk: text the cache is about to be asked
    to predict, known at write time. KL scores are dropped every `refresh`
    chunks so they track the moving probe.
    """
    total = ids.shape[1]
    kl_scores = {}
    for i, start in enumerate(range(0, total, chunk)):
        piece = ids[:, start:start + chunk]
        n_new = piece.shape[1]
        target = cache.capacity - n_new / merge * COST[write_level].item()
        if refresh and i % refresh == 0:
            kl_scores.clear()
        if cache.get_seq_length() and cache.used() > target:
            compact(model, cache, target, policy, chunk, piece[:, :probe],
                    start, device, eps, scores, kl_scores)
        n = cache.get_seq_length()
        model(piece.to(device), past_key_values=cache,
              position_ids=torch.arange(start, start + n_new,
                                        device=device).unsqueeze(0),
              attention_mask=torch.ones(1, n + n_new, device=device,
                                        dtype=torch.long),
              use_cache=True)
        cache.on_write(start, n_new, write_level, merge)
    return total
