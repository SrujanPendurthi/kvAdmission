"""Chunked prefill and generation against a budget-capped cache.

The one thing to get right here is `position_ids`. Cached keys carry RoPE at
their original stream position, and compression makes cache length diverge from
stream position. Letting the model infer positions from cache length silently
corrupts every result, so positions are always passed explicitly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    """CUDA if present, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name="Qwen/Qwen3-1.7B", device=None, attn="eager"):
    """Load a frozen base model.

    attn="eager" is required wherever output_attentions is used (h2o, snapkv).
    Policies that never read attention weights can pass attn="sdpa", which is
    substantially faster and numerically equivalent up to accumulation order -
    so the bit-identity test must stay on eager.
    """
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=attn
    ).to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)  # the base model is frozen, everywhere
    return model, tok, device


def _assert_layers_aligned(cache, n_layers):
    """Every layer must hold the same token count.

    A single global attention_mask is applied to all layers, so a policy that
    makes independent per-layer decisions desynchronises their lengths. Caught
    here it names the cause; left alone it surfaces as an opaque tensor-shape
    mismatch inside eager attention.
    """
    occ = {cache.layers[li].keys.shape[-2] for li in range(n_layers)}
    if len(occ) > 1:
        raise RuntimeError(
            f"layers desynchronised: occupancies {sorted(occ)}. This harness "
            "applies one attention mask to every layer, so a decider must keep "
            "layer lengths equal."
        )


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
            _assert_layers_aligned(cache, n_layers)
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
