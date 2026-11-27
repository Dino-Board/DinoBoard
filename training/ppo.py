"""On-policy PPO for the selfplay pipeline (`training.algorithm = "ppo"`).

Selfplay runs with the engine's `policy_sampling` switch: every ply samples
its move from the network policy without search and records log pi(a) and
V(s) of the sampling network. Each step trains on that step's episodes only:

- advantages: GAE over each seat's own decision sequence, reward = the seat's
  terminal value at its last decision, gamma = 1;
- policy: clipped surrogate (Schulman et al. 2017) plus an entropy bonus and a
  reverse-KL penalty KL(pi || magnet) whose coefficient decays geometrically
  from `magnet.coef_initial` to `magnet.coef_final` over the PPO steps
  (Ataraxos' damped self-play: strong regularization early, weak late). The
  magnet is the network as it was at the first PPO step ("init", i.e. the
  heuristic-cloned start) or the uniform policy over legal moves;
- value: MSE to the lambda-returns, seat-rotated like the AlphaZero target
  (mover first, other seats share the negated return).

Steps that still play heuristic-guided moves are supervised (the pipeline's
replay path); PPO starts once the guidance ratio is 0, and the guidance
schedule must switch off in one step (steps == hold_steps + 1).
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

_PPO_KEYS = {
    "clip", "epochs", "minibatch_size", "gae_lambda", "value_loss_weight",
    "entropy_coef", "magnet", "policy_temperature", "target_kl",
}
_MAGNET_KEYS = {"policy", "coef_initial", "coef_final"}


@dataclass(frozen=True)
class PpoConfig:
    clip: float
    epochs: int
    minibatch_size: int
    gae_lambda: float
    value_loss_weight: float
    entropy_coef: float
    magnet_policy: str  # "init" | "uniform"
    magnet_coef_initial: float
    magnet_coef_final: float
    policy_temperature: float
    # Stop the epoch loop early once the mean approximate KL to the behavior
    # policy exceeds this (0 = never).
    target_kl: float


def parse_ppo_config(train_cfg: dict) -> PpoConfig | None:
    """`training.algorithm` = "alphazero" (default) or "ppo"; "ppo" needs a
    `training.ppo` block with exactly the _PPO_KEYS."""
    algorithm = train_cfg.get("algorithm", "alphazero")
    if algorithm not in ("alphazero", "ppo"):
        raise ValueError(f"training.algorithm must be 'alphazero' or 'ppo', got {algorithm!r}")
    if algorithm != "ppo":
        if "ppo" in train_cfg:
            raise ValueError("training.ppo is set but training.algorithm is not 'ppo'")
        return None
    block = train_cfg["ppo"]
    if not isinstance(block, dict) or set(block) != _PPO_KEYS:
        raise ValueError(f"training.ppo must have exactly {sorted(_PPO_KEYS)}, got {block!r}")
    magnet = block["magnet"]
    if not isinstance(magnet, dict) or set(magnet) != _MAGNET_KEYS:
        raise ValueError(f"training.ppo.magnet must have exactly {sorted(_MAGNET_KEYS)}, got {magnet!r}")
    if magnet["policy"] not in ("init", "uniform"):
        raise ValueError(f"training.ppo.magnet.policy must be 'init' or 'uniform', got {magnet['policy']!r}")
    cfg = PpoConfig(
        clip=float(block["clip"]),
        epochs=int(block["epochs"]),
        minibatch_size=int(block["minibatch_size"]),
        gae_lambda=float(block["gae_lambda"]),
        value_loss_weight=float(block["value_loss_weight"]),
        entropy_coef=float(block["entropy_coef"]),
        magnet_policy=str(magnet["policy"]),
        magnet_coef_initial=float(magnet["coef_initial"]),
        magnet_coef_final=float(magnet["coef_final"]),
        policy_temperature=float(block["policy_temperature"]),
        target_kl=float(block["target_kl"]),
    )
    if not 0.0 < cfg.clip < 1.0:
        raise ValueError(f"training.ppo.clip must be in (0, 1), got {cfg.clip}")
    if cfg.epochs < 1 or cfg.minibatch_size < 1:
        raise ValueError("training.ppo.epochs and minibatch_size must be >= 1")
    if not 0.0 <= cfg.gae_lambda <= 1.0:
        raise ValueError(f"training.ppo.gae_lambda must be in [0, 1], got {cfg.gae_lambda}")
    for name in ("value_loss_weight", "entropy_coef", "target_kl"):
        v = getattr(cfg, name)
        if not (math.isfinite(v) and v >= 0.0):
            raise ValueError(f"training.ppo.{name} must be a finite value >= 0, got {v}")
    if not (cfg.magnet_coef_initial >= cfg.magnet_coef_final >= 0.0):
        raise ValueError(
            "training.ppo.magnet needs coef_initial >= coef_final >= 0, got "
            f"{cfg.magnet_coef_initial}, {cfg.magnet_coef_final}")
    if not (math.isfinite(cfg.policy_temperature) and cfg.policy_temperature > 0.0):
        raise ValueError(f"training.ppo.policy_temperature must be > 0, got {cfg.policy_temperature}")
    return cfg


def gae_targets(values: list[float], final_reward: float, lam: float) -> tuple[list[float], list[float]]:
    """GAE with gamma = 1 over one seat's decisions: rewards are 0 except
    `final_reward` after the last decision, the value after it is 0.
    Returns (advantages, lambda-returns = advantages + values)."""
    n = len(values)
    adv = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < n else 0.0
        reward = final_reward if t == n - 1 else 0.0
        delta = reward + next_v - values[t]
        running = delta + lam * running
        adv[t] = running
    return adv, [a + v for a, v in zip(adv, values)]


def magnet_coef(cfg: PpoConfig, step: int, start_step: int, total_steps: int) -> float:
    """Geometric decay from coef_initial (first PPO step) to coef_final (last step)."""
    if cfg.magnet_coef_initial == 0.0:
        return 0.0
    span = max(1, total_steps - start_step)
    progress = min(1.0, max(0.0, (step - start_step) / span))
    if cfg.magnet_coef_final == 0.0:
        return cfg.magnet_coef_initial * (1.0 - progress)
    return cfg.magnet_coef_initial * (cfg.magnet_coef_final / cfg.magnet_coef_initial) ** progress


class PpoTrainer:
    """Holds the magnet and runs one PPO update per pipeline step."""

    def __init__(self, cfg: PpoConfig, *, num_players: int, action_space: int, device,
                 # Feature rows -> one float32 CPU tensor [N, F] (minibatches move to `device`).
                 features_to_tensor: Callable[[list], torch.Tensor],
                 score_head: torch.nn.Module | None, score_loss_weight: float,
                 score_targets: Callable[[dict, int], list[float]] | None):
        self.cfg = cfg
        self.num_players = num_players
        self.action_space = action_space
        self.device = device
        self.features_to_tensor = features_to_tensor
        self.score_head = score_head
        self.score_loss_weight = score_loss_weight
        self.score_targets = score_targets
        self.magnet_net: torch.nn.Module | None = None
        self.start_step: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "magnet_state_dict": self.magnet_net.state_dict() if self.magnet_net is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any], net: torch.nn.Module) -> None:
        self.start_step = state["start_step"]
        if state["magnet_state_dict"] is not None:
            self.magnet_net = copy.deepcopy(net).requires_grad_(False)
            self.magnet_net.load_state_dict(state["magnet_state_dict"])

    def _begin(self, net: torch.nn.Module, step: int) -> None:
        self.start_step = step
        if self.cfg.magnet_policy == "init":
            self.magnet_net = copy.deepcopy(net).requires_grad_(False)

    def build_rows(self, episodes: list[dict[str, Any]]) -> list[tuple]:
        """(features, legal_mask, action_id, behavior_logp, advantage, value_target, score_target)
        for every kept decision; pool episodes keep only the latest seat."""
        rows = []
        for ep in episodes:
            latest_seat = ep["latest_seat"]
            by_seat: dict[int, list[dict]] = defaultdict(list)
            for s in ep["samples"]:
                # Opening-exploration plies only move the game to a new start;
                # each seat's trajectory begins at its first policy-sampled ply.
                if s["opening_exploration"] or s.get("midgame_start"):
                    continue
                if not s["policy_sampled"]:
                    raise ValueError(
                        "PPO step got a sample that was not sampled from the policy "
                        "(heuristic-guided or searched); PPO steps need guidance ratio 0")
                if latest_seat is not None and s["player"] != latest_seat:
                    continue
                by_seat[s["player"]].append(s)
            z = ep["terminal_z_values"]
            if len(z) != self.num_players:
                raise ValueError(f"episode terminal_z_values has {len(z)} entries, expected {self.num_players}")
            score = None
            for seat, seq in by_seat.items():
                values = [float(s["behavior_value"]) for s in seq]
                adv, ret = gae_targets(values, float(z[seat]), self.cfg.gae_lambda)
                if self.score_targets is not None:
                    score = self.score_targets(ep, seat)
                for s, a, g in zip(seq, adv, ret):
                    other = -g / max(1, self.num_players - 1)
                    value_target = [g] + [other] * (self.num_players - 1)
                    rows.append((s["features"], s["legal_mask"], int(s["action_id"]),
                                 float(s["behavior_logp"]), a, value_target, score))
        return rows

    def update(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               episodes: list[dict[str, Any]], step: int, total_steps: int,
               grad_clip_norm: float) -> dict[str, float]:
        if self.start_step is None:
            self._begin(net, step)
        cfg = self.cfg
        coef_m = magnet_coef(cfg, step, self.start_step, total_steps)
        rows = self.build_rows(episodes)
        if not rows:
            raise ValueError(f"PPO step {step}: no decisions to train on")
        n = len(rows)
        feats_all = self.features_to_tensor([r[0] for r in rows])
        mask_all = torch.as_tensor(np.asarray([np.asarray(r[1], dtype=np.float32) for r in rows]),
                                   device=self.device)
        action_all = torch.as_tensor([r[2] for r in rows], dtype=torch.long, device=self.device)
        old_logp_all = torch.as_tensor([r[3] for r in rows], dtype=torch.float32, device=self.device)
        adv_all = torch.as_tensor([r[4] for r in rows], dtype=torch.float32, device=self.device)
        vt_all = torch.as_tensor([r[5] for r in rows], dtype=torch.float32, device=self.device)
        score_all = (torch.as_tensor([r[6] for r in rows], dtype=torch.float32, device=self.device)
                     if self.score_head is not None else None)
        legal_count = mask_all.sum(dim=-1).clamp_min(1.0)

        net.train()
        if self.score_head is not None:
            self.score_head.train()
        params = list(net.parameters()) + (list(self.score_head.parameters()) if self.score_head is not None else [])
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        stopped_early = False
        inv_t = 1.0 / cfg.policy_temperature
        for _epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            epoch_kl = 0.0
            epoch_batches = 0
            for b0 in range(0, n, cfg.minibatch_size):
                idx_cpu = perm[b0:b0 + cfg.minibatch_size]
                idx = idx_cpu.to(self.device)
                feats = feats_all[idx_cpu].to(self.device)
                mask = mask_all[idx]
                if self.score_head is not None:
                    logits, value_pred, embedding = net.forward_with_embedding(feats)
                else:
                    logits, value_pred = net(feats)
                logits = (logits * inv_t).masked_fill(mask == 0, -1e9)
                logp_all = torch.log_softmax(logits, dim=-1)
                probs = logp_all.exp()
                logp = logp_all.gather(1, action_all[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp - old_logp_all[idx])
                adv = adv_all[idx]
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                surr = torch.min(ratio * adv, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv)
                policy_loss = -surr.mean()
                plogp = torch.where(mask > 0, probs * logp_all, torch.zeros_like(probs))
                entropy = -plogp.sum(dim=-1).mean()
                if cfg.magnet_policy == "init":
                    with torch.no_grad():
                        m_logits, _ = self.magnet_net(feats)
                        m_logp = torch.log_softmax((m_logits * inv_t).masked_fill(mask == 0, -1e9), dim=-1)
                    kl_terms = torch.where(mask > 0, probs * (logp_all - m_logp), torch.zeros_like(probs))
                    magnet_kl = kl_terms.sum(dim=-1).mean()
                else:
                    magnet_kl = (plogp.sum(dim=-1) + torch.log(legal_count[idx])).mean()
                value_loss = torch.nn.functional.mse_loss(value_pred, vt_all[idx])
                loss = (policy_loss + cfg.value_loss_weight * value_loss
                        - cfg.entropy_coef * entropy + coef_m * magnet_kl)
                if self.score_head is not None:
                    score_loss = torch.nn.functional.mse_loss(self.score_head(embedding), score_all[idx])
                    loss = loss + self.score_loss_weight * score_loss
                    totals["score_loss"] += score_loss.item()
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
                optimizer.step()
                with torch.no_grad():
                    approx_kl = (old_logp_all[idx] - logp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["magnet_kl"] += magnet_kl.item()
                totals["approx_kl"] += approx_kl
                totals["clip_frac"] += clip_frac
                batches += 1
                epoch_kl += approx_kl
                epoch_batches += 1
            if cfg.target_kl > 0.0 and epoch_kl / max(1, epoch_batches) > cfg.target_kl:
                stopped_early = True
                break

        with torch.no_grad():
            returns = vt_all[:, 0]
            old_values = returns - adv_all
            var_r = returns.var(unbiased=False).item()
            explained = 1.0 - (returns - old_values).var(unbiased=False).item() / var_r if var_r > 0 else 0.0
        out = {k: v / max(1, batches) for k, v in totals.items()}
        out.update({
            "samples": float(n),
            "magnet_coef": coef_m,
            "explained_var": explained,
            "minibatches": float(batches),
            "stopped_early": float(stopped_early),
        })
        return out
