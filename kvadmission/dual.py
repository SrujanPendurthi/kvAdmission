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
