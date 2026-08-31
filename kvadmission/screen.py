"""Screen candidate write-time signals without running any policy.

§7.2's partial-oracle ablation measured recall against coverage over answer
tokens: 1.00 gave 7/7 in the middle band, 0.75 gave 1/7, 0.50 and below gave
0/7. Coverage is therefore a *test*. A signal that cannot put essentially
every answer token inside its top B is predicted to fail, and phase 10
confirmed that prediction end-to-end for surprisal.

This module computes several candidate signals over a stream so they can be
screened together. Each returns one score per token; higher means "keep".

The candidates are chosen to attack the failure from different directions:

  surprisal      information content - what the context cannot reconstruct
  norm_attention attention received per query row, rather than summed. This
                 is H2O with its diagnosed positional bias removed: the sum
                 favours early tokens because they are visible to more query
                 rows, so normalising by that count is the obvious repair.
  answer_shape   the FORM of an answer rather than its content - digits,
                 capitalised words, entity-looking tokens. Targets what a
                 question tends to ask for.
  rarity         inverse in-document frequency. A token appearing once is a
                 better index into this document than one appearing often.
  key_norm       ||k||, the cheap saliency proxy that several eviction papers
                 use as a tiebreak.
  lookahead      attention a token receives from the NEXT chunk's query rows.
                 The only candidate here with access to information about what
                 follows a token. It is not write-time-causal, but it is
                 implementable: delay compression by one chunk. Every other
                 signal must guess a token's future from its past; this one
                 gets to see 128 tokens of that future.
  random         chance control, so every number has a floor to beat.
"""

import math
import re

import torch

from kvadmission.cache import AdmissionCache


@torch.no_grad()
def candidate_scores(model, tok, ids, device, chunk=128, attn_layers=4,
                     seed=0):
    """Compute every candidate signal over `ids`. Returns {name: (n,) tensor}.

    Attention is accumulated from `attn_layers` evenly spaced layers rather
    than all of them, to keep the per-chunk attention tensor small enough to
    compute on a laptop. Documented rather than hidden: a full-layer version
    could rank differently, though H2O's own bias is a property of the sum and
    not of which layers it is summed over.
    """
    flat = ids[0]
    n = flat.shape[0]
    n_layers = model.config.num_hidden_layers
    keep_layers = sorted({int(i * (n_layers - 1) / max(1, attn_layers - 1))
                          for i in range(attn_layers)})

    surp = torch.zeros(n)
    attn_sum = torch.zeros(n)      # total attention received
    attn_rows = torch.zeros(n)     # how many query rows could see it
    knorm = torch.zeros(n)
    look = torch.zeros(n)          # attention from the FOLLOWING chunk only
    prev_start = None

    cache = AdmissionCache(budget=None, sinks=4)
    prev_row = None
    for start in range(0, n, chunk):
        piece = ids[:, start:start + chunk].to(device)
        n_new = piece.shape[1]
        out = model(piece, past_key_values=cache,
                    position_ids=torch.arange(start, start + n_new,
                                              device=device).unsqueeze(0),
                    attention_mask=torch.ones(1, cache.get_seq_length() + n_new,
                                              device=device, dtype=torch.long),
                    output_attentions=True, use_cache=True)

        lg = out.logits[0].float()
        rows, tgt = lg[:-1], flat[start:start + n_new].to(lg.device)
        if prev_row is not None:
            rows = torch.cat([prev_row.unsqueeze(0), rows], 0)
        else:
            tgt = tgt[1:]
        s = -torch.log_softmax(rows, -1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        if prev_row is None:
            s = torch.cat([s.new_zeros(1), s])
        prev_row = lg[-1]
        surp[start:start + n_new] = s.float().cpu()

        for li in keep_layers:
            a = out.attentions[li].float()              # (1, H, n_new, n_kv)
            recv = a.sum(dim=(0, 1, 2)).cpu()
            attn_sum[: recv.shape[0]] += recv
            if prev_start is not None:
                # what this chunk's queries paid to the PREVIOUS chunk's keys:
                # a one-chunk lookahead, available if compression is delayed
                # by a chunk rather than run immediately
                back = a[0, :, :, prev_start:start].sum(dim=(0, 1)).cpu()
                look[prev_start:prev_start + back.shape[0]] += back
        attn_rows[: start + n_new] += n_new * len(keep_layers)
        prev_start = start

        k = cache.layers[keep_layers[-1]].keys                # (1, kvH, T, D)
        kn = k.float().norm(dim=-1).mean(dim=1)[0].cpu()
        knorm[: kn.shape[0]] = kn
        del out

    # --- answer_shape: does this token look like the kind of thing a question
    # asks for? digits, capitalised words, standalone numbers.
    shape = torch.zeros(n)
    for i in range(n):
        piece = tok.decode([flat[i].item()])
        if any(c.isdigit() for c in piece):
            shape[i] = 2.0
        elif re.match(r"^\s*[A-Z][a-z]", piece):
            shape[i] = 1.0

    # --- rarity: inverse in-document frequency
    counts = torch.bincount(flat)
    rar = torch.tensor([math.log(n / max(1, counts[t].item())) for t in flat])

    g = torch.Generator().manual_seed(seed)
    jitter = torch.rand(n, generator=g) * 1e-3    # break ties in flat signals

    return {
        "surprisal": surp,
        "norm_attention": attn_sum / attn_rows.clamp(min=1),
        "answer_shape": shape + jitter,
        "rarity": rar + jitter,
        "key_norm": knorm,
        "lookahead": look,
        "random": torch.rand(n, generator=g),
    }


def coverage(scores, spans, budget, n_tokens):
    """Fraction of answer-span tokens inside the top-`budget` by score."""
    keep = set(torch.topk(scores, min(budget, n_tokens)).indices.tolist())
    covs = []
    for lo, hi in spans:
        toks = [t for t in range(lo, min(hi, n_tokens))]
        if toks:
            covs.append(sum(t in keep for t in toks) / len(toks))
    return covs
