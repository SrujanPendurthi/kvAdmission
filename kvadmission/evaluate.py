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
from kvadmission.runner import load_model
from kvadmission.stream import build_stream, query_needle
from kvadmission.sweep import summarize


def learned_decider(policy, controller, model, budget, device):
    """Wrap a trained policy in phase 1's decide(layer, cache, attn, n_new)."""
    state = {"keep": None, "hidden": None}
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
            # hidden states must be available before decide runs
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
