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
                1, cache.get_seq_length() + n_new, device=device, dtype=torch.long
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
    keep = cache.get_seq_length()
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
                1, cache.get_seq_length() + cur.shape[1], device=device, dtype=torch.long
            ),
            use_cache=True,
        )
        pos += cur.shape[1]
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(cur.item())

    cache.crop(keep)
    return tok.decode(generated)
