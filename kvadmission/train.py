"""Training loop for the admission policy.

One timestep is one chunk: the action is that chunk's admission decisions, the
reward is negative mean KL minus the priced occupancy, and the value function
bootstraps across the ~64 chunk-steps of a stream. That is what carries credit
back from a divergence to the decision that caused it.
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


def _extra_features(k_norm, n_new, occupancy_frac, lam, device):
    """Columns, in order: [k_norm, rel_pos, occupancy_frac, lam]."""
    k = (k_norm - k_norm.mean()) / (k_norm.std() + 1e-6)
    return torch.stack([
        k,
        torch.linspace(0, 1, n_new, device=device),
        torch.full((n_new,), occupancy_frac, device=device),
        torch.full((n_new,), lam, device=device),
    ], dim=-1)


@torch.no_grad()
def weighted_kl(kl_pos, beta):
    """Divergence-weighted mean over a chunk's positions.

    beta=0 recovers the plain mean, which is what phase 2 used and why it
    failed: long-dependency positions are ~1.5% of a stream, so even at 12.5x
    contrast they carry only ~15% of the KL mass. Averaging hands 85% of the
    gradient to local prediction, which recency already maximises.

    beta=1 weights each position by its own divergence, so positions where
    compression changed nothing contribute nothing and the signal survives.
    """
    if beta <= 0:
        return kl_pos.mean()
    w = kl_pos.detach() ** beta
    return (w * kl_pos).sum() / w.sum().clamp_min(1e-8)


def rollout(model, policy, controller, ids, device, budget, chunk=128,
            sinks=4, generator=None, beta=1.0):
    """One stream. Chunk-level timesteps with per-chunk reward."""
    total = ids.shape[1]
    n_layers = model.config.num_hidden_layers
    ref = DynamicCache()
    comp = AdmissionCache(budget=budget, sinks=sinks)

    log_probs, values, rewards, occupancy, lams, kls = [], [], [], [], [], []
    kls_mean = []

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
        kl_pos = (lp_ref.exp() * (lp_ref - lp_com)).sum(-1).clamp_min(0.0)
        kl = weighted_kl(kl_pos, beta)
        kl_mean = kl_pos.mean()  # logged unweighted so runs stay comparable

        h = comp_out.hidden_states[-1][0].float()
        k_norm = torch.stack([
            comp.layers[li].keys[0, :, -n_new:, :].float().norm(dim=-1).mean(0)
            for li in range(n_layers)
        ]).mean(0)

        occ_frac = comp.occupancy(0) / budget
        extra = _extra_features(k_norm, n_new, occ_frac, controller.lam, device)

        # the base model runs under no_grad; the head still needs a graph
        with torch.enable_grad():
            keep, log_prob, value = policy.act(h, extra, generator=generator)

        keep_cpu = keep.detach().cpu()
        for li in range(n_layers):
            comp.apply_admission(li, keep_cpu)
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
        kls_mean.append(kl_mean.item())
        del ref_out, comp_out, lp_ref, lp_com

    return {"log_probs": log_probs, "values": values, "rewards": rewards,
            "occupancy": occupancy, "lam": lams, "kl": kls,
            "kl_mean": kls_mean}


def train(model, tok, device, steps, budget, length, out_dir, lr=3e-4,
          target=0.95, seed=0, beta=1.0):
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
                    generator=gen, beta=beta)
        stats = learner.step(r["log_probs"], r["values"], r["rewards"])
        stats.update({
            "step": step,
            "mean_kl": sum(r["kl"]) / len(r["kl"]),
            "mean_kl_unweighted": sum(r["kl_mean"]) / len(r["kl_mean"]),
            "mean_occupancy": sum(r["occupancy"]) / len(r["occupancy"]),
            "lam": r["lam"][-1],
            "admit_rate": sum(r["occupancy"]) / (len(r["occupancy"]) * budget),
        })
        history.append(stats)
        print(f"step {step:4d} kl={stats['mean_kl']:.4f} "
              f"klraw={stats['mean_kl_unweighted']:.4f} "
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
    p.add_argument("--beta", type=float, default=1.0,
                   help="reward weighting exponent; 0 = plain mean (phase 2)")
    p.add_argument("--out", default="results/policy")
    a = p.parse_args()

    model, tok, device = load_model(a.model)
    print(f"device={device}", flush=True)
    train(model, tok, device, a.steps, a.budget, a.length, a.out,
          lr=a.lr, target=a.target, seed=a.seed, beta=a.beta)


if __name__ == "__main__":
    main()
