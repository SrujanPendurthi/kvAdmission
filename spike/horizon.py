"""Spike: does a fixed-budget KV cache policy have a retention horizon?

Plants needles through a long stream, compresses the cache to budget B during
chunked prefill, then queries every needle from the end. Reports recall as a
function of distance-back. A cliff means episodic policies cannot see past
cache turnover; a uniform slope means the problem is budget, not formulation.

Throwaway: this answers a question, it is not the training harness.
"""

import argparse, json, random, re, urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

SINKS = 4  # StreamingLLM: first few tokens are load-bearing, never evict
HERE = Path(__file__).parent
GUTENBERG = "https://www.gutenberg.org/files/1342/1342-0.txt"


# ---------------------------------------------------------------- stream

def haystack_text() -> str:
    cache = HERE / "haystack.txt"
    if not cache.exists():
        raw = urllib.request.urlopen(GUTENBERG).read().decode("utf-8", "ignore")
        cache.write_text(raw)
    return cache.read_text()


def build_stream(tok, n_needles: int, target_len: int, seed: int = 0):
    """Interleave needles evenly through filler. Returns (ids, needles)."""
    rng = random.Random(seed)
    topics = ["oak", "harbor", "lantern", "quarry", "meridian", "thistle",
              "cobalt", "vellum", "cinder", "marrow", "fathom", "sable",
              "juniper", "ashen", "gallows", "wren"][:n_needles]
    assert len(topics) == n_needles, "add more topic words"

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", haystack_text())
             if 40 < len(s.strip()) < 400]
    rng.shuffle(sents)

    needles = [{"topic": t, "code": f"{rng.randrange(100000, 999999)}"} for t in topics]
    # place needle i at fraction (i+0.5)/n of the stream
    marks = [int(target_len * (i + 0.5) / n_needles) for i in range(n_needles)]

    ids, si, placed = [], 0, 0
    while len(ids) < target_len:
        while placed < n_needles and len(ids) >= marks[placed]:
            n = needles[placed]
            ids += tok(f" The secret code for {n['topic']} is {n['code']}. ",
                       add_special_tokens=False)["input_ids"]
            n["pos"] = len(ids)
            placed += 1
        ids += tok(" " + sents[si % len(sents)], add_special_tokens=False)["input_ids"]
        si += 1

    ids = ids[:target_len]
    needles = [n for n in needles if "pos" in n and n["pos"] < target_len]
    return torch.tensor([ids]), needles


# ---------------------------------------------------------------- policies

def keep_indices(policy: str, n: int, budget: int, h2o, snap, rng):
    """Which cache positions to retain. Always keeps the sink prefix."""
    if n <= budget:
        return None
    k = budget - SINKS
    tail = torch.arange(SINKS, n)

    if policy == "recent":                      # StreamingLLM
        sel = tail[-k:]
    elif policy == "h2o":                       # accumulated attention
        sel = tail[torch.topk(h2o[SINKS:], k).indices]
    elif policy == "snapkv":                    # recent-window attention vote
        sel = tail[torch.topk(snap[SINKS:], k).indices]
    elif policy == "random":
        sel = tail[torch.randperm(n - SINKS, generator=rng)[:k]]
    else:
        raise ValueError(policy)

    return torch.cat([torch.arange(SINKS), sel.sort().values])


# ---------------------------------------------------------------- run

@torch.no_grad()
def prefill(model, ids, policy, budget, chunk, window, device, seed=0, prune_once=False):
    """Chunked prefill. Compress after every chunk, or once at the end.

    prune_once=True reproduces how H2O/SnapKV are actually evaluated in their
    papers: score over the whole prefill, compress a single time. The control
    that separates 'repeated compression destroys needles' from 'my scoring is
    wrong'."""
    cache = DynamicCache()
    n_layers = model.config.num_hidden_layers
    acc = [None] * n_layers                       # accumulated attention per layer
    rng = torch.Generator().manual_seed(seed)
    need_attn = policy in ("h2o", "snapkv")
    total = ids.shape[1]

    for start in range(0, total, chunk):
        piece = ids[:, start:start + chunk].to(device)
        clen = cache.get_seq_length()
        out = model(
            piece,
            past_key_values=cache,
            # true stream positions, NOT cache offset - the cache is compressed
            position_ids=torch.arange(start, start + piece.shape[1],
                                      device=device).unsqueeze(0),
            attention_mask=torch.ones(1, clen + piece.shape[1], device=device,
                                      dtype=torch.long),
            output_attentions=need_attn,
            use_cache=True,
        )

        if policy == "full":
            continue
        # with prune_once we still accumulate scores every chunk, but only evict at the end
        do_prune = (not prune_once) or (start + chunk >= total)

        for li in range(n_layers):
            n = cache.layers[li].keys.shape[-2]
            h2o = snap = None
            if need_attn:
                a = out.attentions[li].float()                  # (1,H,C,kv)
                cur = a.sum(dim=(0, 1, 2)).cpu()                # per kv position
                win = a[:, :, -window:, :].sum(dim=(0, 1, 2)).cpu()
                prev = acc[li]
                if prev is not None:
                    cur[:prev.shape[0]] += prev
                acc[li], h2o, snap = cur, cur, win

            if not do_prune:
                continue
            idx = keep_indices(policy, n, budget, h2o, snap, rng)
            if idx is None:
                continue
            cache.layers[li].keys = cache.layers[li].keys[:, :, idx.to(device), :]
            cache.layers[li].values = cache.layers[li].values[:, :, idx.to(device), :]
            if acc[li] is not None:
                acc[li] = acc[li][idx]

        del out
    return cache, total


@torch.no_grad()
def ask(model, tok, cache, stream_len, needle, device, max_new=24):
    """Query one needle off the compressed cache, then restore the cache."""
    keep = cache.get_seq_length()
    q = tok(f"\n\nQuestion: What is the secret code for {needle['topic']}?\nAnswer:",
            add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    gen, pos = [], stream_len
    cur = q
    for _ in range(max_new):
        clen = cache.get_seq_length()
        out = model(
            cur,
            past_key_values=cache,
            position_ids=torch.arange(pos, pos + cur.shape[1],
                                      device=device).unsqueeze(0),
            attention_mask=torch.ones(1, clen + cur.shape[1], device=device,
                                      dtype=torch.long),
            use_cache=True,
        )
        pos += cur.shape[1]
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(nxt.item())
        cur = nxt

    cache.crop(keep)  # roll back the query so the next needle sees the same cache
    text = tok.decode(gen)
    return needle["code"] in text, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--len", type=int, default=8192)
    p.add_argument("--needles", type=int, default=8)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--window", type=int, default=32, help="SnapKV observation window")
    p.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--policies", nargs="+",
                   default=["full", "recent", "h2o", "snapkv", "random"])
    p.add_argument("--prune-once", action="store_true",
                   help="compress once after prefill (paper-style) instead of every chunk")
    p.add_argument("--out", default=str(HERE / "results.json"))
    a = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()

    ids, needles = build_stream(tok, a.needles, a.len)
    print(f"stream {ids.shape[1]} tok, {len(needles)} needles, device={device}", flush=True)

    rows = []
    for policy in a.policies:
        budgets = [a.len] if policy == "full" else a.budgets
        for b in budgets:
            cache, slen = prefill(model, ids, policy, b, a.chunk, a.window, device,
                                  prune_once=a.prune_once)
            for n in needles:
                ok, text = ask(model, tok, cache, slen, n, device)
                rows.append({"policy": policy, "budget": b, "prune_once": a.prune_once,
                         "topic": n["topic"],
                             "distance_back": slen - n["pos"], "hit": ok,
                             "gen": text[:60]})
            hits = sum(r["hit"] for r in rows[-len(needles):])
            tag = "once" if a.prune_once else "cont"
            print(f"{policy:8s} B={b:<6d} [{tag}] recall {hits}/{len(needles)}", flush=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))  # crash-safe
            del cache

    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
