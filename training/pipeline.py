"""Self-play training pipeline: selfplay -> collect samples -> train -> export -> repeat."""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import random
import shutil
import signal
import threading
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    TimeoutError,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mcts_profile import MctsProfile, max_game_plies, profile_as_dict, resolve_profile
from .model import (
    AuxHead,
    PVNet,
    create_belief_model_from_config,
    convert_onnx_fp16,
    create_model_from_config,
    export_belief_onnx,
    export_onnx,
    padded_token_count,
)
from .symmetry import parse_symmetry_augmentation

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def atomic_write(path: Path):
    """Replace `path` atomically: yields a temp path in the same directory to
    write to; on normal exit the temp file is fsynced and renamed over `path`.
    A crash or kill at any point leaves the previous `path` intact (plus, at
    worst, a stale `<name>.tmp`, overwritten by the next write) — never a
    truncated file where `--resume` expects a complete one."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        fd = os.open(tmp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if os.name == "posix":
        # Make the rename itself durable (directory fds cannot be opened on Windows).
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _default_gating_threshold(num_players: int, eval_games: int,
                              victory_mode: str = "winner_takes") -> float:
    """Default gating win-rate threshold, scaled to the N-player null baseline.

    Scoring: winners split 1 point (score = max(0, z_candidate)); losers get 0.
    Null baseline is always 1/N. Threshold = baseline + z * sqrt(Var / games),
    z = 1.0 (one standard error above null; recovers 0.55 for 2p/100 games).

    winner_takes (1 winner, N-1 losers): score ∈ {0, 1}.
        Var = (1/N)(1-1/N) = (N-1)/N².
        2p/100g: 0.5500, 3p/100g: 0.3805, 4p/100g: 0.2933.

    loser_pays (1 loser, N-1 winners): score ∈ {0, 1/(N-1)}.
        Var = 1/(N²(N-1)).
        2p/100g: 0.5500, 3p/100g: 0.3610, 4p/100g: 0.2644.

    Override per game by setting `gating_accept_win_rate` in train.json.
    """
    N = max(num_players, 1)
    baseline = 1.0 / N
    z = 1.0  # (0.55 - 0.5) / sqrt(0.25 / 100)
    if victory_mode == "loser_pays" and N > 2:
        # share is 0 (loser) or 1/(N-1) (non-loser).
        # Var = p_lose * (0 - baseline)^2 + p_survive * (1/(N-1) - baseline)^2
        #     = (1/N) * (1/N)^2 + ((N-1)/N) * (1/(N-1) - 1/N)^2
        #     = 1/N^3 + ((N-1)/N) * (1/(N(N-1)))^2
        #     = 1/N^3 + 1/(N^3(N-1))
        #     = 1/N^3 * (1 + 1/(N-1)) = 1/N^3 * N/(N-1) = 1/(N^2(N-1))
        var_share = 1.0 / (N * N * (N - 1))
    else:
        # winner_takes: share ∈ {0, 1}. Var = baseline*(1-baseline) = (N-1)/N^2
        var_share = baseline * (1.0 - baseline)
    se = math.sqrt(var_share / max(eval_games, 1))
    return baseline + z * se


def _worker_selfplay_batch(args: tuple) -> list[dict[str, Any]]:
    """Run one selfplay episode in C++.

    args:
        (game_id, seeds, model_paths_per_seat, latest_seat, cfg,
         belief_paths_per_seat)

    `seeds` is a singleton list (one task = one episode). The C++
    `run_selfplay_episodes_batch` binding accepts a list, so the wrapper
    layer is identical for mirror and pool tasks.

    `latest_seat` is None for mirror tasks and an int for pool tasks.
    For pool tasks it is forwarded to C++ so the latest seat's MCTS
    installs a SplitPolicyValueEvaluator (opponent network for prior at
    opponent nodes, latest network for value everywhere). Mirror tasks
    forward `-1` and the C++ side skips the wrapper.

    `belief_paths_per_seat` is a list[str] of length num_players (or
    empty for the legacy mirror-belief path). Empty entries fall back to
    uniform sampling for that seat. For pool tasks the latest seat gets
    the latest belief and opponent seats get the belief paired with the
    chosen pool checkpoint.

    Sample filtering by latest_seat is caller-side; the worker just tags
    each episode dict so the caller knows which seat (if any) to keep.
    """
    import dinoboard_engine
    game_id, seeds, model_paths_per_seat, latest_seat, cfg, belief_paths_per_seat = args
    if cfg["temperature_schedule_enabled"]:
        t_initial = cfg["temperature_initial"]
        t_final = cfg["temperature_final"]
        t_decay = cfg["temperature_decay_plies"]
    else:
        t_initial = -1.0
        t_final = -1.0
        t_decay = 0
    out = dinoboard_engine.run_selfplay_episodes_batch(
        game_id=game_id,
        seeds=seeds,
        model_paths=model_paths_per_seat,
        simulations=cfg["simulations"],
        world_cap=cfg["world_cap"],
        c_puct=cfg["c_puct"],
        temperature=cfg["temperature"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_epsilon=cfg["dirichlet_epsilon"],
        dirichlet_on_first_n_plies=cfg["dirichlet_on_first_n_plies"],
        max_game_plies=cfg["max_game_plies"],
        tail_solve_enabled=cfg["tail_solve_enabled"],
        tail_solve_depth_limit=cfg["tail_solve_depth_limit"],
        tail_solve_node_budget=cfg["tail_solve_node_budget"],
        tail_solve_auxiliary_weight=cfg["tail_solve_auxiliary_weight"],
        temperature_initial=t_initial,
        temperature_final=t_final,
        temperature_decay_plies=t_decay,
        heuristic_guidance_ratio=cfg["heuristic_guidance_ratio"],
        heuristic_temperature=cfg["heuristic_temperature"],
        training_filter_ratio=cfg["training_filter_ratio"],
        opponent_selection=cfg["opponent_selection"],
        belief_model_path=cfg["belief_model_path"],
        heuristic_selector=cfg["heuristic_selector"],
        latest_seat=latest_seat if latest_seat is not None else -1,
        safe_mode=cfg["safe_mode"],
        top_k=cfg["top_k"],
        belief_model_paths=belief_paths_per_seat or [],
        fpu_reduction=cfg["fpu_reduction"],
        uct2_arrival_visits=cfg["uct2_arrival_visits"],
    )
    episodes = out["episodes"]
    for ep in episodes:
        ep["latest_seat"] = latest_seat
    return episodes


class PersistentPool:
    """Long-lived ProcessPoolExecutor with auto-rebuild on broken state.

    WSL randomly SIGSEGV/SIGKILLs worker processes; once that happens
    ProcessPoolExecutor enters its `_broken` state and every subsequent
    submit raises BrokenProcessPool. Rebuilding a pool every step (the
    old behavior) avoided this but spawned N×steps×3 processes per
    training run — sheer process churn was its own instability source.

    This wrapper keeps one executor across the whole training loop and
    only rebuilds when broken. Callers detect breakage via two paths:
      - `pool.get()` checks `_broken` proactively before returning the executor
      - `pool.mark_broken()` for callers that catch BrokenProcessPool from
        future.result()
    """

    def __init__(self, max_workers: int):
        self._max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

    def get(self) -> ProcessPoolExecutor:
        ex = self._executor
        if ex is None or getattr(ex, "_broken", False):
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            ex = ProcessPoolExecutor(max_workers=self._max_workers)
            self._executor = ex
        return ex

    def mark_broken(self) -> None:
        ex = self._executor
        self._executor = None
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def close(self) -> None:
        ex = self._executor
        self._executor = None
        if ex is not None:
            try:
                ex.shutdown(wait=True)
            except Exception:
                pass


def _collect_pool_paths(models_dir: Path) -> list[tuple[str, str]]:
    """All `model_step_*.onnx` checkpoints paired with their belief sibling.

    Returns (pv_path, belief_path) tuples sorted by step number.
    belief_path is "" when the corresponding belief_step_*.onnx does not
    exist (games without belief, or steps before belief was enabled).

    Excludes `model_latest.onnx`, `model_init.onnx`, `model_best.onnx`
    by glob pattern. Returns [] before the first checkpoint is saved.
    """
    pv_paths = sorted(models_dir.glob("model_step_*.onnx"))
    result: list[tuple[str, str]] = []
    for pv in pv_paths:
        belief = pv.parent / pv.name.replace("model_step_", "belief_step_")
        result.append((str(pv), str(belief) if belief.exists() else ""))
    return result


def _worker_arena(args: tuple) -> dict[str, Any]:
    """Run a single arena match in a worker process.

    Task tuple format: (game_id, seed, max_plies, seat_configs, belief_model_path, safe_mode).
    seat_configs is the per-seat AI config list — see profile_to_seat_config.
    """
    import dinoboard_engine
    game_id, seed, max_plies, seat_configs, belief_model_path, safe_mode = args
    return dinoboard_engine.run_arena_match(
        game_id=game_id, seed=seed,
        max_game_plies=max_plies,
        seat_configs=seat_configs,
        belief_model_path=belief_model_path,
        safe_mode=safe_mode,
    )


def _worker_eval_vs_heuristic(args: tuple) -> dict[str, Any]:
    """Run a single eval game vs heuristic in a worker process."""
    import dinoboard_engine
    (game_id, seed, model_path, simulations, world_cap, c_puct, temperature,
     model_is_player, constrained, h_temp, opp_sel,
     tail_solve_enabled, tail_solve_depth_limit, tail_solve_node_budget,
     tail_solve_auxiliary_weight,
     belief_model_path, heuristic_selector, safe_mode, top_k, max_game_plies,
     fpu_reduction, uct2_arrival_visits) = args
    return dinoboard_engine.run_constrained_eval_vs_heuristic(
        game_id=game_id,
        seed=seed,
        model_path=model_path,
        max_game_plies=max_game_plies,
        simulations=simulations,
        world_cap=world_cap,
        c_puct=c_puct,
        temperature=temperature,
        model_is_player=model_is_player,
        constrained=constrained,
        heuristic_temperature=h_temp,
        opponent_selection=opp_sel,
        tail_solve_enabled=tail_solve_enabled,
        tail_solve_depth_limit=tail_solve_depth_limit,
        tail_solve_node_budget=tail_solve_node_budget,
        tail_solve_auxiliary_weight=tail_solve_auxiliary_weight,
        belief_model_path=belief_model_path,
        heuristic_selector=heuristic_selector,
        safe_mode=safe_mode,
        top_k=top_k,
        fpu_reduction=fpu_reduction,
        uct2_arrival_visits=uct2_arrival_visits,
    )


def normalize_policy(action_ids: list[int], visits: list[int], action_space: int) -> list[float]:
    total = sum(max(0, v) for v in visits)
    policy = [0.0] * action_space
    if total <= 0:
        return policy
    for aid, v in zip(action_ids, visits):
        if 0 <= aid < action_space:
            policy[aid] = max(0, v) / total
    return policy


def sparse_policy(action_ids: list[int], visits: list[int], action_space: int) -> tuple[np.ndarray, np.ndarray]:
    """`normalize_policy` as (action ids, probabilities) over the visited
    actions only; `densify_policies` turns a batch of these back into rows."""
    total = sum(max(0, v) for v in visits)
    keep = [(a, max(0, v)) for a, v in zip(action_ids, visits) if 0 <= a < action_space and v > 0]
    if total <= 0 or not keep:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    ids = np.fromiter((a for a, _ in keep), dtype=np.int64, count=len(keep))
    probs = np.fromiter((v / total for _, v in keep), dtype=np.float32, count=len(keep))
    return ids, probs


def densify_policies(rows: list[tuple[np.ndarray, np.ndarray]], action_space: int) -> np.ndarray:
    out = np.zeros((len(rows), action_space), dtype=np.float32)
    for i, (ids, probs) in enumerate(rows):
        out[i, ids] = probs
    return out


def rows_to_tensor(rows: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(rows).to(device=device, dtype=torch.float32)


def valid_feature_length(n: int, feature_dim: int, variable_token_dim: int) -> bool:
    """Mirrors IFeatureEncoder::valid_feature_length: feature_dim floats, or a
    positive multiple of token_dim for variable-length token encoders."""
    if variable_token_dim:
        return n > 0 and n % variable_token_dim == 0
    return n == feature_dim


def pad_token_rows(rows, token_dim: int) -> np.ndarray:
    """Variable-length token encodings as one [B, n * token_dim] array,
    zero-padded to padded_token_count of the longest (as the C++ evaluator does)."""
    width = padded_token_count(max(len(r) for r in rows) // token_dim) * token_dim
    out = np.zeros((len(rows), width), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def parse_compact_replay(train_cfg: dict) -> np.dtype | None:
    """training.compact_replay = {"features_dtype": "float16" | "float32"}:
    replay rows hold ndarray features in that dtype, a uint8 legal mask and
    sparse policies instead of Python float lists. Absent = list rows."""
    block = train_cfg.get("compact_replay")
    if block is None:
        return None
    if not isinstance(block, dict) or set(block) != {"features_dtype"}:
        raise ValueError(f"training.compact_replay must be an object with exactly 'features_dtype', got {block!r}")
    if block["features_dtype"] not in ("float16", "float32"):
        raise ValueError(
            f"training.compact_replay.features_dtype must be 'float16' or 'float32', got {block['features_dtype']!r}")
    sc_block = train_cfg.get("selfplay_concurrency")
    if not (sc_block and sc_block.get("enabled")):
        raise ValueError("training.compact_replay requires training.selfplay_concurrency.enabled=true")
    return np.dtype(block["features_dtype"])


def train_step(
    net: PVNet,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
    grad_clip_norm: float = 1.0,
    *,
    score_head: AuxHead | None = None,
    score_targets: torch.Tensor | None = None,
    score_loss_weight: float = 0.0,
    opp_head: AuxHead | None = None,
    opp_targets: torch.Tensor | None = None,
    opp_weights: torch.Tensor | None = None,
    opp_loss_weight: float = 0.0,
) -> dict[str, float]:
    if (score_head is None) != (score_targets is None):
        raise ValueError("train_step: score_head and score_targets must be passed together")
    if not ((opp_head is None) == (opp_targets is None) == (opp_weights is None)):
        raise ValueError("train_step: opp_head, opp_targets and opp_weights must be passed together")
    net.train()
    aux_heads = [h for h in (score_head, opp_head) if h is not None]
    for h in aux_heads:
        h.train()
    if aux_heads:
        policy_logits, value_pred, embedding = net.forward_with_embedding(features)
    else:
        policy_logits, value_pred = net(features)

    if legal_mask is not None:
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

    value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)
    policy_loss = -(policy_targets * torch.nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    loss = policy_loss + value_loss
    score_loss = None
    if score_head is not None:
        score_loss = torch.nn.functional.mse_loss(score_head(embedding), score_targets)
        loss = loss + score_loss_weight * score_loss
    opp_loss = None
    if opp_head is not None:
        opp_ce = -(opp_targets * torch.nn.functional.log_softmax(opp_head(embedding), dim=-1)).sum(dim=-1)
        opp_loss = (opp_ce * opp_weights).sum() / opp_weights.sum().clamp_min(1.0)
        loss = loss + opp_loss_weight * opp_loss

    optimizer.zero_grad()
    loss.backward()
    if grad_clip_norm > 0:
        params = list(net.parameters())
        for h in aux_heads:
            params += list(h.parameters())
        torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
    optimizer.step()
    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }
    if score_loss is not None:
        metrics["score_loss"] = score_loss.item()
    if opp_loss is not None:
        metrics["opp_loss"] = opp_loss.item()
    return metrics


def score_margin_targets(final_scores: list[float], player: int, num_players: int,
                         scale: float) -> list[float]:
    """Seat-rotated (like the value target) score margins: each seat's final
    score minus the best other seat's, divided by `scale`."""
    if len(final_scores) != num_players:
        raise ValueError(
            f"final_scores has {len(final_scores)} entries, expected {num_players}")
    margins = []
    for p in range(num_players):
        best_other = max(final_scores[q] for q in range(num_players) if q != p)
        margins.append((final_scores[p] - best_other) / scale)
    return rotate_z_values(margins, player, num_players)


def next_opponent_indices(players: list[int]) -> list[int | None]:
    """For each ply, the index of the next ply acted by a different seat
    (None when no other seat moves again before the game ends)."""
    out: list[int | None] = [None] * len(players)
    nxt: int | None = None
    for i in range(len(players) - 1, -1, -1):
        if i + 1 < len(players) and players[i + 1] != players[i]:
            nxt = i + 1
        out[i] = nxt
    return out


@torch.no_grad()
def ema_update(ema_net: torch.nn.Module, net: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema_net.parameters(), net.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for e, b in zip(ema_net.buffers(), net.buffers()):
        e.copy_(b)


_LEGACY_GROUP_NAMES = ("pv", "score_head")


def build_pv_optimizer(net: torch.nn.Module, aux_heads: dict[str, torch.nn.Module],
                       learning_rate: float, weight_decay: float,
                       saved_state: dict | None) -> torch.optim.Optimizer:
    """AdamW with one named param group per module: "pv" for the PV net plus
    one per enabled auxiliary head (keyed by its config name).

    Saved groups are matched by name (unnamed groups from older checkpoints
    are positional: pv, score_head). A head enabled since the save starts
    with fresh moments, a head disabled since is dropped; any other mismatch
    raises in load_state_dict."""
    modules = {"pv": net, **aux_heads}
    if saved_state is None:
        return torch.optim.AdamW(
            [{"params": list(m.parameters()), "name": n} for n, m in modules.items()],
            lr=learning_rate, weight_decay=weight_decay)
    saved_groups: dict[str, dict] = {}
    for i, g in enumerate(saved_state["param_groups"]):
        name = g.get("name")
        if name is None:
            if i >= len(_LEGACY_GROUP_NAMES):
                raise ValueError(f"saved optimizer has unnamed param group #{i}")
            name = _LEGACY_GROUP_NAMES[i]
        saved_groups[name] = {**g, "name": name}
    if "pv" not in saved_groups:
        raise ValueError("saved optimizer state has no 'pv' param group")
    kept = [n for n in modules if n in saved_groups]
    kept_ids = {pid for n in kept for pid in saved_groups[n]["params"]}
    for n in saved_groups:
        if n not in modules:
            logger.info(f"Resume: dropped optimizer state of disabled '{n}'")
    optimizer = torch.optim.AdamW(
        [{"params": list(modules[n].parameters()), "name": n} for n in kept],
        lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict({
        "state": {k: v for k, v in saved_state["state"].items() if k in kept_ids},
        "param_groups": [saved_groups[n] for n in kept],
    })
    for n, m in modules.items():
        if n not in saved_groups:
            optimizer.add_param_group({"params": list(m.parameters()), "name": n})
            logger.info(f"Resume: '{n}' optimizer state freshly initialized")
    return optimizer


# Two-card multiset enumeration — must match
# `coup_net_adapter.cpp::TwoCardTable` exactly. m=0..14 with the canonical
# (R_a, R_b) order R_a ≤ R_b in role-id space (D=0..Con=4).
_BELIEF_BRANCH2_BASE = 0
_BELIEF_BRANCH2_SIZE = 15
_BELIEF_BRANCH1_BASE = 15
_BELIEF_BRANCH1_SIZE = 5


def _build_two_card_id_table(K_role: int) -> list[list[int]]:
    """id[a][b] = id[b][a] = multiset index for (a,b)."""
    table = [[0] * K_role for _ in range(K_role)]
    m = 0
    for a in range(K_role):
        for b in range(a, K_role):
            table[a][b] = m
            table[b][a] = m
            m += 1
    return table


def _hand_counts_to_multiset_id(
    hand_row: list[int], K_role: int, two_card_id: list[list[int]]
) -> tuple[int, int]:
    """Map a per-role count row into (alive, multiset_id) ∈
    ({0,1,2}, branch index). For alive=2 the id is a 0..14 two-card id;
    for alive=1 the id is the role itself (0..4); alive=0/≥3 → (alive,-1).
    """
    alive = sum(hand_row)
    if alive == 1:
        for r in range(K_role):
            if hand_row[r] == 1:
                return 1, r
        return 1, -1
    if alive == 2:
        roles = []
        for r in range(K_role):
            for _ in range(hand_row[r]):
                roles.append(r)
        if len(roles) != 2:
            return 2, -1
        return 2, two_card_id[roles[0]][roles[1]]
    return alive, -1


def _belief_feature_batch(batch: list, device) -> torch.Tensor:
    """Belief buffer rows hold features as a list (legacy) or an ndarray (compact)."""
    return torch.from_numpy(np.stack([np.asarray(s[0], dtype=np.float32) for s in batch])).to(device)


def _train_belief_step_coup_two_branch(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
) -> float:
    """Coup-specific belief training (BELIEF_NETWORK.md §6/§7).

    The belief net emits, per opp, a length-20 logit vector — `[0..14]`
    over two-card multisets and `[15..19]` over single roles. Targets are
    one-hot multiset ids: alive=2 → 15-way head; alive=1 → 5-way head;
    alive=0 / alive≥3 rows are skipped (alive=0 = opp eliminated, no
    belief to learn; alive≥3 = mid-Exchange bookkeeping not covered by
    either head). Loss is plain CE on the selected branch.

    `K` here is the per-opp output width (`output_logit_count() / (N-1)`)
    and must equal 20; the underlying role count is fixed at 5 (Coup).
    """
    if not buffer or net is None:
        return float("nan")
    n_opp = num_players - 1
    expected_K = _BELIEF_BRANCH2_SIZE + _BELIEF_BRANCH1_SIZE  # 20
    if n_opp <= 0 or K != expected_K:
        return float("nan")
    K_role = _BELIEF_BRANCH1_SIZE  # 5
    two_card_id = _build_two_card_id_table(K_role)

    device = next(net.parameters()).device
    buf = list(buffer)
    n = len(buf)
    net.train()
    total = 0.0
    log_softmax = torch.nn.functional.log_softmax
    for _ in range(batches_per_step):
        # feats + (target_id, target_branch) built for ONLY the sampled
        # minibatch (was: all n rows + an n×n_opp Python target loop every
        # step). target_id[b,i] = one-hot index inside the chosen branch (or
        # -1); target_branch[b,i] = 2/1/0 alive count for branch selection.
        # alive>=3 (mid-Exchange) rows fall through to the else and are skipped.
        idx = torch.randint(n, (min(batch_size, n),)).tolist()
        batch = [buf[i] for i in idx]
        B = len(batch)
        f = _belief_feature_batch(batch, device)
        tid_rows = []
        tbr_rows = []
        for s in batch:
            hand = s[1]
            alive = s[3]
            ids = []
            brs = []
            for oi in range(n_opp):
                row = hand[oi]
                a = alive[oi] if oi < len(alive) else 0
                if a == 2 or (a == 0 and sum(row) == 2):
                    a_eff, mid = _hand_counts_to_multiset_id(row, K_role,
                                                             two_card_id)
                    brs.append(a_eff)
                    ids.append(mid)
                elif a == 1:
                    _, mid = _hand_counts_to_multiset_id(row, K_role, two_card_id)
                    brs.append(1)
                    ids.append(mid)
                else:
                    brs.append(a)
                    ids.append(-1)
            tid_rows.append(ids)
            tbr_rows.append(brs)
        tid = torch.tensor(tid_rows, dtype=torch.long, device=device)   # [B, n_opp]
        tbr = torch.tensor(tbr_rows, dtype=torch.long, device=device)   # [B, n_opp]
        logits = net(f).view(B, n_opp, K)   # [B, n_opp, 20]

        # Branch 2 (two-card multiset) — alive == 2 rows.
        b2_logits = logits[..., _BELIEF_BRANCH2_BASE:
                                  _BELIEF_BRANCH2_BASE + _BELIEF_BRANCH2_SIZE]
        b2_log_pi = log_softmax(b2_logits, dim=-1)
        # Branch 1 (single-card) — alive == 1 rows.
        b1_logits = logits[..., _BELIEF_BRANCH1_BASE:
                                  _BELIEF_BRANCH1_BASE + _BELIEF_BRANCH1_SIZE]
        b1_log_pi = log_softmax(b1_logits, dim=-1)

        valid2 = (tbr == 2) & (tid >= 0)
        valid1 = (tbr == 1) & (tid >= 0)

        nll2 = torch.zeros((), dtype=b2_log_pi.dtype, device=device)
        nll1 = torch.zeros((), dtype=b1_log_pi.dtype, device=device)
        # gather indexes every position regardless of mask; clamp per-branch
        # to that branch's range so non-target rows (which we mask out
        # afterward) can't trigger out-of-bounds. b2 size=15, b1 size=5.
        if valid2.any():
            sel = tid.clamp(min=0, max=_BELIEF_BRANCH2_SIZE - 1).unsqueeze(-1)
            picked = b2_log_pi.gather(-1, sel).squeeze(-1)
            nll2 = -(picked[valid2]).sum()
        if valid1.any():
            sel = tid.clamp(min=0, max=_BELIEF_BRANCH1_SIZE - 1).unsqueeze(-1)
            picked = b1_log_pi.gather(-1, sel).squeeze(-1)
            nll1 = -(picked[valid1]).sum()

        n_valid = (valid2.sum() + valid1.sum()).clamp_min(1).float()
        loss = (nll2 + nll1) / n_valid

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    net.eval()
    return total / max(1, batches_per_step)


def _train_belief_step_flat_onehot(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
) -> float:
    """Flat per-opp categorical CE — one head per opp, K classes,
    target is the one-hot index inside `hand_counts[opp]`.

    Generic fallback for any belief whose target_extractor emits a single
    one-hot row of length K per opponent and whose net output factors as
    `[B, n_opp, K]`. No alive/branch logic — every buffer row contributes
    to the loss. (Liar's Dice uses the per-face variant below; Coup uses
    its own two-branch trainer.)
    """
    if not buffer or net is None:
        return float("nan")
    n_opp = num_players - 1
    if n_opp <= 0 or K <= 0:
        return float("nan")

    device = next(net.parameters()).device
    buf = list(buffer)
    n = len(buf)
    net.train()
    total = 0.0
    log_softmax = torch.nn.functional.log_softmax
    for _ in range(batches_per_step):
        # feats + one-hot target index built for ONLY the sampled minibatch
        # (was: all n rows + an n×n_opp Python target loop every step). Rows
        # whose row sum != 1 (corrupt one-hot) map to target id -1 (skipped).
        idx = torch.randint(n, (min(batch_size, n),)).tolist()
        batch = [buf[i] for i in idx]
        B = len(batch)
        f = _belief_feature_batch(batch, device)
        tid_rows = []
        for s in batch:
            rows = s[1]
            opp_ids = []
            for oi in range(n_opp):
                row = rows[oi] if oi < len(rows) else None
                hot = -1
                if row is not None and len(row) == K:
                    ones = 0
                    for j, v in enumerate(row):
                        if v == 1:
                            hot = j
                            ones += 1
                        elif v != 0:
                            ones = -1
                            break
                    if not (ones == 1 and 0 <= hot < K):
                        hot = -1
                opp_ids.append(hot)
            tid_rows.append(opp_ids)
        tid = torch.tensor(tid_rows, dtype=torch.long, device=device)  # [B, n_opp]
        logits = net(f).view(B, n_opp, K)   # [B, n_opp, K]
        log_pi = log_softmax(logits, dim=-1)

        valid = tid >= 0
        if not valid.any():
            continue
        sel = tid.clamp_min(0).unsqueeze(-1)
        picked = log_pi.gather(-1, sel).squeeze(-1)        # [B, n_opp]
        nll = -(picked[valid]).sum()
        n_valid = valid.sum().clamp_min(1).float()
        loss = nll / n_valid

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    net.eval()
    return total / max(1, batches_per_step)


_BELIEF_LIARSDICE_FACES = 6
_BELIEF_LIARSDICE_COUNT_BINS = 6  # n_f ∈ [0, kDicePerPlayer=5]
_BELIEF_LIARSDICE_K = (
    _BELIEF_LIARSDICE_FACES * _BELIEF_LIARSDICE_COUNT_BINS  # 36
)


def _train_belief_step_per_face_onehot(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
) -> float:
    """Per-face CE × 6 — Liar's Dice 36-dim per-opp head.

    Per-opp logits factor as 6 faces × 6 count bins. Target row is a flat
    length-36 vector, 6 concatenated 6-way one-hots (slot f * 6 + n_f =
    1). For each (batch, opp, face) we run an independent 6-way CE. Rows
    whose any face fails the one-hot check are skipped via target_id ==
    -1 on that face.
    """
    if not buffer or net is None:
        return float("nan")
    n_opp = num_players - 1
    F = _BELIEF_LIARSDICE_FACES
    C = _BELIEF_LIARSDICE_COUNT_BINS
    if n_opp <= 0 or K != F * C:
        return float("nan")

    device = next(net.parameters()).device
    buf = list(buffer)
    n = len(buf)
    net.train()
    total = 0.0
    log_softmax = torch.nn.functional.log_softmax
    for _ in range(batches_per_step):
        # feats + per-face target index built for ONLY the sampled minibatch
        # (was: all n rows + an n×n_opp×F Python target loop every step).
        # target_id[b, o, f] ∈ [0, C) or -1 (skip that face).
        idx = torch.randint(n, (min(batch_size, n),)).tolist()
        batch = [buf[i] for i in idx]
        B = len(batch)
        f = _belief_feature_batch(batch, device)
        tid_rows = []
        for s in batch:
            rows = s[1]
            opp_faces = []
            for oi in range(n_opp):
                row = rows[oi] if oi < len(rows) else None
                faces = [-1] * F
                if row is not None and len(row) == K:
                    for fi in range(F):
                        hot = -1
                        ones = 0
                        base = fi * C
                        ok = True
                        for cj in range(C):
                            v = row[base + cj]
                            if v == 1:
                                hot = cj
                                ones += 1
                            elif v != 0:
                                ok = False
                                break
                        if ok and ones == 1 and 0 <= hot < C:
                            faces[fi] = hot
                opp_faces.append(faces)
            tid_rows.append(opp_faces)
        tid = torch.tensor(tid_rows, dtype=torch.long, device=device)  # [B, n_opp, F]
        logits = net(f).view(B, n_opp, F, C)          # [B, n_opp, F, C]
        log_pi = log_softmax(logits, dim=-1)

        valid = tid >= 0                              # [B, n_opp, F]
        if not valid.any():
            continue
        sel = tid.clamp_min(0).unsqueeze(-1)
        picked = log_pi.gather(-1, sel).squeeze(-1)   # [B, n_opp, F]
        nll = -(picked[valid]).sum()
        n_valid = valid.sum().clamp_min(1).float()
        loss = nll / n_valid

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    net.eval()
    return total / max(1, batches_per_step)


def _train_belief_step_bce_multihot(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
) -> float:
    """Per-opp BCE-with-logits — multi-hot independent-bernoulli target.

    Each opp row is a length-K {0,1} vector, target = "opp holds card c".
    Cards are independent (no softmax constraint), so each of K logits is
    fed through sigmoid + BCE against its own bit. No row-validity skip
    since every multi-hot is admissible (including the empty row at
    end-of-game). 6 nimmt! uses K=104.
    """
    if not buffer or net is None:
        return float("nan")
    n_opp = num_players - 1
    if n_opp <= 0 or K <= 0:
        return float("nan")

    device = next(net.parameters()).device
    buf = list(buffer)
    n = len(buf)
    net.train()
    total = 0.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(batches_per_step):
        # Build feats + multi-hot target for ONLY the sampled minibatch. Was:
        # tensorize all n rows + an n×n_opp×K Python target loop (with per-cell
        # GPU writes) every step, so belief train time grew with buffer fill.
        idx = torch.randint(n, (min(batch_size, n),)).tolist()
        batch = [buf[i] for i in idx]
        B = len(batch)
        f = _belief_feature_batch(batch, device)
        tgt_np = np.zeros((B, n_opp, K), dtype=np.float32)
        for b, s in enumerate(batch):
            rows = s[1]
            for oi in range(min(n_opp, len(rows))):
                row = np.asarray(rows[oi])
                if row.shape == (K,):
                    tgt_np[b, oi] = row == 1
        tgt = torch.from_numpy(tgt_np).to(device)
        logits = net(f).view(B, n_opp, K)
        loss = bce(logits, tgt, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    net.eval()
    return total / max(1, batches_per_step)


def _train_belief_step(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: deque,
    *,
    num_players: int,
    K: int,
    batch_size: int,
    batches_per_step: int,
    grad_clip: float,
    loss: str = "ce",
) -> float:
    """Dispatch belief training by `loss` (from belief.json::training.loss).

    The belief target's shape is a per-game design choice (see each game's
    `<game>_net_adapter.cpp::BeliefLabelExtractor`). This entry picks the
    matching trainer:
      - loss == "bce" → BCE-with-logits multi-hot (6nimmit; independent
        per-card bernoulli — cards don't compete, opp holds many at once)
      - loss == "ce" + K == 20 → coup two-branch (15 two-card + 5 single)
      - loss == "ce" + K == 36 → liar's dice per-face CE × 6
      - loss == "ce" otherwise → flat one-hot CE
    """
    if loss == "bce":
        return _train_belief_step_bce_multihot(
            net, optimizer, buffer,
            num_players=num_players, K=K,
            batch_size=batch_size, batches_per_step=batches_per_step,
            grad_clip=grad_clip,
        )
    coup_K = _BELIEF_BRANCH2_SIZE + _BELIEF_BRANCH1_SIZE  # 20
    if K == coup_K:
        return _train_belief_step_coup_two_branch(
            net, optimizer, buffer,
            num_players=num_players, K=K,
            batch_size=batch_size, batches_per_step=batches_per_step,
            grad_clip=grad_clip,
        )
    if K == _BELIEF_LIARSDICE_K:
        return _train_belief_step_per_face_onehot(
            net, optimizer, buffer,
            num_players=num_players, K=K,
            batch_size=batch_size, batches_per_step=batches_per_step,
            grad_clip=grad_clip,
        )
    return _train_belief_step_flat_onehot(
        net, optimizer, buffer,
        num_players=num_players, K=K,
        batch_size=batch_size, batches_per_step=batches_per_step,
        grad_clip=grad_clip,
    )


def _run_selfplay_batch_concurrent(
    *,
    game_id: str,
    model_path: str,
    pool_paths: list[tuple[str, str]],
    num_episodes: int,
    base_seed: int,
    train_cfg: dict,
    sc_cfg: dict,
    num_players: int,
    n_self_workers: int,
    n_pool_workers: int,
    pool_size: int,
    pool_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Phase 1 batched/threaded selfplay runner.

    Single-process, multi-thread C++ binding (`run_concurrent_selfplay_step`).
    Replaces the legacy ProcessPoolExecutor path when
    train.json::training.selfplay_concurrency.enabled=true.

    Self/pool partition is reused unchanged: `n_self_workers + n_pool_workers`
    are interpreted as C++ thread counts (same number — meaning shifts from
    "process" to "thread"). Episodes are work-stolen by threads; each thread
    runs in its fixed mode (self/pool). EvaluatorPool dedups model paths so
    `latest` and each opp checkpoint share one BatchedEvaluator across all
    sampling threads.
    """
    import dinoboard_engine

    if train_cfg["temperature_schedule_enabled"]:
        t_initial = train_cfg["temperature_initial"]
        t_final = train_cfg["temperature_final"]
        t_decay = train_cfg["temperature_decay_plies"]
    else:
        t_initial = -1.0
        t_final = -1.0
        t_decay = 0

    seeds = list(range(base_seed, base_seed + num_episodes))
    ev_cfg = sc_cfg["evaluator"]

    # If pool is disabled or has no entries, force pool threads → 0.
    # pool_paths is list[(pv, belief)]; C++ wants list[str] for PV and
    # a separate list[str] for belief (paired by index).
    eff_pool_tuples = pool_paths if (pool_enabled and pool_paths) else []
    eff_pool_pv = [pv for pv, _belief in eff_pool_tuples]
    eff_pool_belief = [belief for _pv, belief in eff_pool_tuples]
    eff_pool_threads = n_pool_workers if eff_pool_pv else 0
    eff_self_threads = n_self_workers if eff_pool_pv else (n_self_workers + n_pool_workers)

    out = dinoboard_engine.run_concurrent_selfplay_step(
        game_id=game_id,
        seeds=seeds,
        latest_model_path=model_path,
        pool_paths=eff_pool_pv,
        num_self_threads=eff_self_threads,
        num_pool_threads=eff_pool_threads,
        opponent_pool_max_distinct=int(sc_cfg.get("opponent_pool_max_distinct", 3)),
        simulations=train_cfg["simulations"],
        world_cap=train_cfg["world_cap"],
        c_puct=train_cfg["c_puct"],
        temperature=train_cfg["temperature"],
        dirichlet_alpha=train_cfg["dirichlet_alpha"],
        dirichlet_epsilon=train_cfg["dirichlet_epsilon"],
        dirichlet_on_first_n_plies=train_cfg["dirichlet_on_first_n_plies"],
        max_game_plies=train_cfg["max_game_plies"],
        tail_solve_enabled=train_cfg["tail_solve_enabled"],
        tail_solve_depth_limit=train_cfg["tail_solve_depth_limit"],
        tail_solve_node_budget=train_cfg["tail_solve_node_budget"],
        tail_solve_auxiliary_weight=train_cfg["tail_solve_auxiliary_weight"],
        temperature_initial=t_initial,
        temperature_final=t_final,
        temperature_decay_plies=t_decay,
        # Guidance: forward the same fields the legacy ProcessPool path
        # forwarded to run_selfplay_episodes_batch. mcts_pv heuristics with
        # model_path get hot-swapped onto the shared BatchedEvaluator pool
        # inside the C++ binding so guidance MCTS calls batch with selfplay.
        heuristic_guidance_ratio=float(train_cfg["heuristic_guidance_ratio"]),
        heuristic_temperature=float(train_cfg["heuristic_temperature"]),
        heuristic_selector=str(train_cfg["heuristic_selector"]),
        training_filter_ratio=train_cfg["training_filter_ratio"],
        opponent_selection=train_cfg["opponent_selection"],
        belief_model_path=train_cfg["belief_model_path"],
        batch_size=int(ev_cfg["batch_size"]),
        batch_timeout_us=int(ev_cfg["batch_timeout_us"]),
        max_queue_size=int(ev_cfg["max_queue_size"]),
        intra_op_num_threads=int(ev_cfg["intra_op_num_threads"]),
        num_inference_threads=int(ev_cfg["num_inference_threads"]),
        providers=list(ev_cfg.get("providers", []) or []),
        devices=list(ev_cfg.get("devices", [])),
        safe_mode=train_cfg["safe_mode"],
        top_k=train_cfg["top_k"],
        pool_belief_paths=eff_pool_belief,
        leaf_batch_size=int(sc_cfg["leaf_batch_size"]),
        fast_simulations=int(train_cfg.get("fast_simulations", 0)),
        full_search_prob=float(train_cfg.get("full_search_prob", 1.0)),
        fpu_reduction=train_cfg["fpu_reduction"],
        compact_samples=bool(train_cfg.get("compact_samples", False)),
        uct2_arrival_visits=train_cfg["uct2_arrival_visits"],
    )
    episodes = out["episodes"]
    # Pipeline downstream expects latest_seat=None for mirror, int>=0 for pool.
    # The C++ binding emits -1 for mirror, normalize here.
    for ep in episodes:
        if ep.get("latest_seat", -1) < 0:
            ep["latest_seat"] = None
    n_self_done = sum(1 for ep in episodes if ep["latest_seat"] is None)
    n_pool_done = len(episodes) - n_self_done

    stats = {
        "n_self": n_self_done,
        "n_pool": n_pool_done,
        "n_self_workers": eff_self_threads,
        "n_pool_workers": eff_pool_threads,
        "pool_size": pool_size,
        "pool_enabled": pool_enabled,
        "dropped": num_episodes - len(episodes),
    }
    return episodes, stats


def run_selfplay_batch(
    game_id: str,
    model_path: str,
    num_episodes: int,
    base_seed: int,
    train_cfg: dict,
    max_workers: int,
    *,
    pool: PersistentPool | None = None,
    pool_paths: list[tuple[str, str]] | None = None,
    num_players: int | None = None,
    pool_enabled: bool = True,
    self_ratio: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run a step's worth of selfplay, split between mirror and pool modes.

    Workers are partitioned by mode, not episodes. `self_ratio` controls
    the **worker** split: `n_self_workers = round(max_workers *
    self_ratio)`, the rest are pool workers. Each worker keeps pulling
    same-mode episodes from a shared remaining-episode counter until the
    counter reaches zero. A slow mode (e.g. pool with split evaluator)
    naturally produces fewer episodes for the step — fast workers don't
    wait for slow ones, and pool latency doesn't dictate self-mode
    sample volume.

    Mirror episode = all seats use latest model. Pool episode = latest
    on a randomly chosen `latest_seat`, one frozen pool opponent on the
    rest, drawn from `random.Random(base_seed)` so two pool episodes in
    the same step don't share an opponent unless the RNG repeats.

    `pool_enabled=False` (training cfg `opponent_pool_enabled: false`)
    or empty `pool_paths` ⇒ every worker runs mirror.

    `self_ratio` ∈ [0, 1] is read from training cfg
    `opponent_pool_self_ratio`.

    Returns (episodes, stats) where stats = {"n_self": ..., "n_pool":
    ..., "pool_size": ..., "pool_enabled": ..., "dropped": ...}.
    `n_self` / `n_pool` are observed episode counts (not configured).
    Each episode dict carries `latest_seat: int | None` (None for
    mirror episodes; the chosen seat for pool episodes).
    """
    if pool_paths is None:
        pool_paths = []
    if not pool_enabled:
        pool_paths = []
    pool_size = len(pool_paths)

    if num_players is None:
        import dinoboard_engine
        num_players = dinoboard_engine.game_metadata(game_id)["num_players"]

    if not (0.0 <= self_ratio <= 1.0):
        raise ValueError(
            f"run_selfplay_batch: opponent_pool_self_ratio must be in [0, 1], "
            f"got {self_ratio}"
        )

    if num_episodes <= 0:
        return [], {
            "n_self": 0, "n_pool": 0,
            "n_self_workers": 0, "n_pool_workers": 0,
            "pool_size": pool_size, "pool_enabled": pool_enabled, "dropped": 0,
        }

    # Worker-mode partition: split the worker pool by `self_ratio`, not
    # episodes. With pool_size==0, every worker is a self worker.
    if pool_size == 0:
        n_self_workers = max_workers
        n_pool_workers = 0
    else:
        n_self_workers = round(max_workers * self_ratio)
        n_self_workers = max(0, min(max_workers, n_self_workers))
        # Edge cases: ratio > 0 but rounded to 0 → keep 1 self worker,
        # likewise for pool. Skip when total workers would exceed
        # max_workers (e.g. max_workers == 1 with mid ratio).
        if n_self_workers == 0 and self_ratio > 0.0 and max_workers >= 2:
            n_self_workers = 1
        n_pool_workers = max_workers - n_self_workers
        if n_pool_workers == 0 and self_ratio < 1.0 and max_workers >= 2:
            n_pool_workers = 1
            n_self_workers = max_workers - 1

    # ---------------------------------------------------------------
    # Phase 1 short-circuit: if train.json::training.selfplay_concurrency
    # has enabled=true, dispatch the entire step to the C++ batched runner
    # instead of the legacy ProcessPoolExecutor. Same self/pool worker split
    # is reused (now interpreted as thread counts inside one process).
    # ---------------------------------------------------------------
    sc_cfg = train_cfg.get("selfplay_concurrency") or {}
    if sc_cfg.get("enabled", False):
        return _run_selfplay_batch_concurrent(
            game_id=game_id,
            model_path=model_path,
            pool_paths=pool_paths,
            num_episodes=num_episodes,
            base_seed=base_seed,
            train_cfg=train_cfg,
            sc_cfg=sc_cfg,
            num_players=num_players,
            n_self_workers=n_self_workers,
            n_pool_workers=n_pool_workers,
            pool_size=pool_size,
            pool_enabled=pool_enabled,
        )

    rng = random.Random(base_seed)
    # Shared episode counters: each mode has its own remaining counter so
    # one mode burning through its share doesn't starve the other. We
    # don't pre-allocate per-mode budgets — total budget is `num_episodes`,
    # and the modes consume in proportion to their wallclock throughput.
    seed_counter = [0]  # mutable closure: monotonic seed per dispatched task

    # Belief routing for the legacy ProcessPool path:
    #  - mirror task: every seat shares the latest belief in `train_cfg
    #    ["belief_model_path"]`. Pass [] so C++ falls back to its single-
    #    evaluator legacy branch (cheaper than per-seat dedup with N
    #    identical paths).
    #  - pool task: latest seat → latest belief, opponent seats → the
    #    belief.onnx paired with the chosen pool checkpoint. Empty when
    #    the pool entry has no sibling belief file (older steps before
    #    belief was enabled, or games without belief).
    latest_belief_path = train_cfg["belief_model_path"]

    def make_self_task() -> tuple:
        s = base_seed + seed_counter[0]
        seed_counter[0] += 1
        return (
            game_id,
            [s],
            [model_path] * num_players,
            None,
            train_cfg,
            [],
        )

    def make_pool_task() -> tuple:
        s = base_seed + seed_counter[0]
        seed_counter[0] += 1
        latest_seat = rng.randrange(num_players)
        pool_pv, pool_belief = rng.choice(pool_paths)
        paths_per_seat = [
            model_path if seat == latest_seat else pool_pv
            for seat in range(num_players)
        ]
        # Per-seat belief routing. When the latest belief is empty (belief
        # disabled for this game) we still send [] so C++ skips the
        # per-seat path entirely.
        if latest_belief_path:
            belief_paths_per_seat = [
                latest_belief_path if seat == latest_seat else pool_belief
                for seat in range(num_players)
            ]
        else:
            belief_paths_per_seat = []
        return (
            game_id,
            [s],
            paths_per_seat,
            latest_seat,
            train_cfg,
            belief_paths_per_seat,
        )

    results: list[dict[str, Any]] = []
    n_self_done = 0
    n_pool_done = 0
    batch_timeout = float(train_cfg.get("selfplay_batch_timeout_seconds", 0.0) or 0.0)
    deadline = time.monotonic() + batch_timeout if batch_timeout > 0.0 else None
    timed_out = False
    pool_broken = False
    crashed = 0
    owned_pool = pool is None
    if owned_pool:
        pool = PersistentPool(max_workers)
    executor = pool.get()

    # future -> mode tag ("self"/"pool") so we know which factory to use
    # when refilling. Initial fill: one task per worker, mode-bound.
    in_flight: dict[Any, str] = {}
    remaining = num_episodes

    def submit(mode: str) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        task = make_self_task() if mode == "self" else make_pool_task()
        fut = executor.submit(_worker_selfplay_batch, task)
        in_flight[fut] = mode
        remaining -= 1

    for _ in range(n_self_workers):
        submit("self")
    for _ in range(n_pool_workers):
        submit("pool")

    try:
        while in_flight:
            timeout_left = None
            if deadline is not None:
                timeout_left = deadline - time.monotonic()
                if timeout_left <= 0:
                    raise TimeoutError()
            done, _pending = wait(
                list(in_flight.keys()),
                timeout=timeout_left,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                # wait() returned with empty done set ⇒ timeout
                raise TimeoutError()
            for future in done:
                mode = in_flight.pop(future)
                try:
                    eps = future.result()
                    results.extend(eps)
                    if mode == "self":
                        n_self_done += len(eps)
                    else:
                        n_pool_done += len(eps)
                except BrokenProcessPool:
                    pool_broken = True
                    break
                except Exception as exc:
                    crashed += 1
                    logger.warning(
                        "selfplay worker crashed; skipping episode (%s: %s)",
                        type(exc).__name__, exc,
                    )
                # Refill: same-mode replacement so the worker stays bound
                # to its mode for the rest of the step.
                submit(mode)
            if pool_broken:
                break
    except TimeoutError:
        timed_out = True
    finally:
        if timed_out or pool_broken:
            for future in list(in_flight.keys()):
                future.cancel()
            in_flight.clear()
        if pool_broken:
            pool.mark_broken()
        if owned_pool:
            pool.close()

    dropped = num_episodes - len(results)
    if timed_out:
        logger.warning(
            "selfplay batch timed out after %.1fs; keeping %d completed episodes, "
            "dropping %d unfinished (of which %d crashed)",
            batch_timeout, len(results), dropped, crashed,
        )
    elif pool_broken:
        logger.warning(
            "selfplay pool became broken; kept %d completed episodes, dropped %d "
            "(persistent pool will be rebuilt on next call)",
            len(results), dropped,
        )
    elif crashed > 0:
        logger.warning(
            "selfplay batch had %d crashed worker(s); kept %d completed episodes",
            crashed, len(results),
        )

    stats = {
        "n_self": n_self_done,
        "n_pool": n_pool_done,
        "n_self_workers": n_self_workers,
        "n_pool_workers": n_pool_workers,
        "pool_size": pool_size,
        "pool_enabled": pool_enabled,
        "dropped": dropped,
    }
    return results, stats


def run_eval_vs_heuristic(
    game_id: str,
    model_path: str,
    num_games: int,
    base_seed: int,
    simulations: int,
    world_cap: int,
    constrained: bool,
    heuristic_temperature: float,
    max_workers: int,
    opponent_selection: str,
    *,
    max_game_plies: int,
    c_puct: float = 1.4,
    temperature: float = 0.0,
    tail_solve_enabled: bool = False,
    tail_solve_depth_limit: int = 10,
    tail_solve_node_budget: int = 200_000,
    tail_solve_auxiliary_weight: float = 0.0,
    pool: PersistentPool | None = None,
    belief_model_path: str = "",
    heuristic_selector: str = "",
    concurrent_cfg: dict | None = None,
    safe_mode: bool = False,
    top_k: int = 0,
    fpu_reduction: float | None = None,
    uct2_arrival_visits: bool = True,
) -> dict[str, Any]:
    import dinoboard_engine
    num_players = dinoboard_engine.game_metadata(game_id)["num_players"]

    # Single-process batched path: run all games on `num_threads` OS threads
    # sharing ONE BatchedPolicyValueEvaluator (one ONNX session, batched leaf
    # inference). This avoids the ProcessPool path's fork-load of a full ONNX
    # stack per worker — the cause of the 30GB-box OOM at high worker counts —
    # and lets eval ride the same GPU batching as selfplay. The batched binding
    # now supports `constrained=true` (FilteredRulesWrapper for the model side)
    # and hot-swaps mcts_pv heuristic evaluators onto the shared pool too.
    if concurrent_cfg is not None:
        batched_tasks = [
            {"seed": base_seed + i, "model_is_player": i % num_players}
            for i in range(num_games)
        ]
        out = dinoboard_engine.run_concurrent_eval_vs_heuristic_batch(
            game_id=game_id,
            model_path=model_path,
            heuristic_selector=heuristic_selector,
            tasks=batched_tasks,
            simulations=simulations,
            world_cap=world_cap,
            c_puct=c_puct,
            temperature=temperature,
            heuristic_temperature=heuristic_temperature,
            max_game_plies=max_game_plies,
            tail_solve_enabled=tail_solve_enabled,
            tail_solve_depth_limit=tail_solve_depth_limit,
            tail_solve_node_budget=tail_solve_node_budget,
            tail_solve_auxiliary_weight=tail_solve_auxiliary_weight,
            opponent_selection=opponent_selection,
            belief_model_path=belief_model_path,
            num_threads=int(concurrent_cfg["num_threads"]),
            batch_size=int(concurrent_cfg["batch_size"]),
            batch_timeout_us=int(concurrent_cfg["batch_timeout_us"]),
            max_queue_size=int(concurrent_cfg["max_queue_size"]),
            intra_op_num_threads=int(concurrent_cfg["intra_op_num_threads"]),
            num_inference_threads=int(concurrent_cfg["num_inference_threads"]),
            providers=list(concurrent_cfg["providers"]),
            devices=list(concurrent_cfg["devices"]),
            constrained=constrained,
            safe_mode=safe_mode,
            top_k=top_k,
            leaf_batch_size=int(concurrent_cfg["leaf_batch_size"]),
            fpu_reduction=fpu_reduction,
            uct2_arrival_visits=uct2_arrival_visits,
        )
        # win_rate = mean candidate share over games, where each game's
        # candidate share = (z_candidate + 1) / N (see the selfplay log
        # block for the full rationale). For 2p winner-takes-all this
        # collapses to wins/(games) with draws contributing 0.5 each;
        # for liarsdice 3+ loser-pays a candidate that ducks every
        # challenge gets share 0.5 per game (non-loser bonus).
        # wins/losses/draws kept as descriptive integer columns: a "win"
        # is share > 1/N (above null baseline), "loss" is < 1/N, "draw"
        # is exactly = 1/N (true terminal draws + winner-takes-all draws).
        share_sum = 0.0
        wins = losses = draws = 0
        baseline = 1.0 / num_players
        for idx, r in enumerate(out["results"]):
            cand_seat = idx % num_players
            z = r["terminal_z_values"]
            if not z:
                # Defensive: shouldn't happen for terminal results.
                draws += 1
                share_sum += baseline
                continue
            # share = (z+1)/N per the per-seat candidate share formula.
            # For 2p winner_takes z∈{-1,0,+1} → share∈{0,0.5,1}; baseline=0.5.
            # For Np loser_pays a winner gets z=+1/(N-1) → share>baseline → W;
            # the loser gets z=-1 → share=0 < baseline → L. The earlier
            # `max(0.0, z)` clamp silently mapped every loss (z=-1) to
            # share=0 → counted as a draw, hiding all real losses.
            share = (float(z[cand_seat]) + 1.0) / num_players
            share_sum += share
            if share > baseline + 1e-6:
                wins += 1
            elif share < baseline - 1e-6:
                losses += 1
            else:
                draws += 1
        total = max(1, wins + losses + draws)
        win_rate = share_sum / total
        return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}

    tasks = []
    for i in range(num_games):
        # Rotate the model's seat across all N players so each seat carries
        # the same number of games. `i % 2` would only test seats 0 and 1,
        # leaving 3p/4p seats 2/3 unmeasured.
        model_side = i % num_players
        tasks.append((
            game_id, base_seed + i, model_path, simulations, world_cap,
            c_puct, temperature,
            model_side, constrained, heuristic_temperature,
            opponent_selection,
            tail_solve_enabled, tail_solve_depth_limit, tail_solve_node_budget,
            tail_solve_auxiliary_weight,
            belief_model_path, heuristic_selector, safe_mode, top_k,
            max_game_plies, fpu_reduction, uct2_arrival_visits,
        ))

    wins = losses = draws = 0
    share_sum = 0.0
    baseline = 1.0 / num_players
    owned_pool = pool is None
    if owned_pool:
        pool = PersistentPool(max_workers)
    try:
        executor = pool.get()
        futures = [executor.submit(_worker_eval_vs_heuristic, t) for t in tasks]
        pool_broken = False
        for idx, future in enumerate(futures):
            if pool_broken:
                future.cancel()
                continue
            try:
                r = future.result()
            except BrokenProcessPool:
                pool_broken = True
                pool.mark_broken()
                logger.warning(
                    "eval pool became broken; skipping remaining %d games",
                    len(futures) - idx,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "eval worker crashed; skipping game (%s: %s)",
                    type(exc).__name__, exc,
                )
                continue
            # tasks tuple layout (see tasks.append above / _worker_eval_vs_heuristic
            # unpack): [5]=c_puct, [6]=temperature, [7]=model_side. Index 7 is the
            # seat the model occupies — using [5] (c_puct, a float) made `w ==
            # model_side` never match, scoring every non-draw as a loss.
            model_side = tasks[idx][7]
            z = r["terminal_z_values"]
            if not z:
                draws += 1
                share_sum += baseline
                continue
            # share = (z+1)/N — see the matching block above (~L1237) for
            # why the earlier `max(0.0, z)` form hid every loss as a draw.
            share = (float(z[model_side]) + 1.0) / num_players
            share_sum += share
            if share > baseline + 1e-6:
                wins += 1
            elif share < baseline - 1e-6:
                losses += 1
            else:
                draws += 1
    finally:
        if owned_pool:
            pool.close()

    total = max(1, wins + losses + draws)
    win_rate = share_sum / total
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def run_eval_batch(
    game_id: str,
    candidate_path: str,
    opponent_path: str,
    num_games: int,
    base_seed: int,
    sims_candidate: int,
    sims_opponent: int,
    world_cap_candidate: int,
    world_cap_opponent: int,
    max_workers: int,
    max_plies: int,
    c_puct: float,
    temperature: float,
    opponent_selection: str,
    tail_solve_enabled: bool,
    tail_solve_depth_limit: int,
    tail_solve_node_budget: int,
    tail_solve_auxiliary_weight: float,
    *,
    pool: PersistentPool | None = None,
    belief_model_path: str = "",
    belief_model_path_opponent: str = "",
    concurrent_cfg: dict | None = None,
    safe_mode: bool = False,
    top_k: int = 0,
    fpu_reduction: float | None = None,
    uct2_arrival_visits: bool = True,
) -> dict[str, Any]:
    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(game_id)
    num_players = meta["num_players"]

    # Effective opponent belief: fall back to candidate's belief when not
    # specified, so callers that don't care about per-seat belief just pass
    # one path (the old behavior).
    eff_belief_opponent = belief_model_path_opponent or belief_model_path

    # Per-seat AI config: candidate vs opponent share every MCTS knob except
    # model_path / simulations / world_cap. tail_solve_enabled is identical
    # on both sides — A/B comparisons of tail_solve live in stand-alone tools
    # (scripts/eval_tail_solve.py), not in the gating eval batch.
    def _make_seat_config(
        model_path: str, sims: int, wcap: int,
        belief_model_path: str = "",
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "model_path": model_path,
            "simulations": sims,
            "world_cap": wcap,
            "c_puct": c_puct,
            "fpu_reduction": fpu_reduction,
            "uct2_arrival_visits": uct2_arrival_visits,
            "temperature": temperature,
            "opponent_selection": opponent_selection,
            "tail_solve_enabled": tail_solve_enabled,
            "tail_solve_depth_limit": tail_solve_depth_limit,
            "tail_solve_node_budget": tail_solve_node_budget,
            "tail_solve_auxiliary_weight": tail_solve_auxiliary_weight,
            "top_k": top_k,
            "safe_mode": safe_mode,
        }
        if belief_model_path:
            cfg["belief_model_path"] = belief_model_path
        return cfg

    # Single-process batched path (see run_eval_vs_heuristic for rationale):
    # all matches share one BatchedPolicyValueEvaluator pool across num_threads
    # OS threads — each distinct model is loaded ONCE, not once-per-worker.
    # This is the direct fix for the gating OOM: the ProcessPool path fork-loads
    # candidate + opponent + belief in every worker, and at 100 workers that
    # blew past a 30GB box.
    if concurrent_cfg is not None:
        batched_tasks = []
        candidate_seats = []
        for i in range(num_games):
            seat = i % num_players
            seat_configs = [
                _make_seat_config(opponent_path, sims_opponent, world_cap_opponent,
                                  belief_model_path=eff_belief_opponent)
                for _ in range(num_players)
            ]
            seat_configs[seat] = _make_seat_config(
                candidate_path, sims_candidate, world_cap_candidate,
                belief_model_path=belief_model_path)
            for sc in seat_configs:
                sc["leaf_batch_size"] = int(concurrent_cfg["leaf_batch_size"])
            # Per-task belief_model_path: use the candidate's belief for the
            # match. C++ `run_concurrent_arena_batch` uses one belief evaluator
            # per task (shared across all seats in that match).
            batched_tasks.append({
                "seed": base_seed + i,
                "seat_configs": seat_configs,
                "belief_model_path": belief_model_path,
            })
            candidate_seats.append(seat)
        out = dinoboard_engine.run_concurrent_arena_batch(
            game_id=game_id,
            tasks=batched_tasks,
            num_threads=int(concurrent_cfg["num_threads"]),
            max_game_plies=max_plies,
            batch_size=int(concurrent_cfg["batch_size"]),
            batch_timeout_us=int(concurrent_cfg["batch_timeout_us"]),
            max_queue_size=int(concurrent_cfg["max_queue_size"]),
            intra_op_num_threads=int(concurrent_cfg["intra_op_num_threads"]),
            num_inference_threads=int(concurrent_cfg["num_inference_threads"]),
            providers=list(concurrent_cfg["providers"]),
            devices=list(concurrent_cfg["devices"]),
            safe_mode=safe_mode,
        )
        share_sum = 0.0
        wins = losses = draws = 0
        baseline = 1.0 / num_players
        for idx, r in enumerate(out["results"]):
            z = r["terminal_z_values"]
            if not z:
                draws += 1
                share_sum += baseline
                continue
            # share = (z+1)/N — see L1237 block for why the prior
            # `max(0.0, z)` clamp counted every loss as a draw.
            share = (float(z[candidate_seats[idx]]) + 1.0) / num_players
            share_sum += share
            if share > baseline + 1e-6:
                wins += 1
            elif share < baseline - 1e-6:
                losses += 1
            else:
                draws += 1
        total = max(1, wins + losses + draws)
        win_rate = share_sum / total
        return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}

    tasks = []
    candidate_seats = []
    for i in range(num_games):
        seat = i % num_players
        seat_configs = [
            _make_seat_config(opponent_path, sims_opponent, world_cap_opponent,
                              belief_model_path=eff_belief_opponent)
            for _ in range(num_players)
        ]
        seat_configs[seat] = _make_seat_config(
            candidate_path, sims_candidate, world_cap_candidate,
            belief_model_path=belief_model_path)
        tasks.append((game_id, base_seed + i, max_plies, seat_configs, belief_model_path, safe_mode))
        candidate_seats.append(seat)

    wins = losses = draws = 0
    share_sum = 0.0
    baseline = 1.0 / num_players
    owned_pool = pool is None
    if owned_pool:
        pool = PersistentPool(max_workers)
    try:
        executor = pool.get()
        futures = [executor.submit(_worker_arena, t) for t in tasks]
        pool_broken = False
        for idx, future in enumerate(futures):
            if pool_broken:
                future.cancel()
                continue
            try:
                r = future.result()
            except BrokenProcessPool:
                pool_broken = True
                pool.mark_broken()
                logger.warning(
                    "arena pool became broken; skipping remaining %d games",
                    len(futures) - idx,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "arena worker crashed; skipping game (%s: %s)",
                    type(exc).__name__, exc,
                )
                continue
            z = r["terminal_z_values"]
            if not z:
                draws += 1
                share_sum += baseline
                continue
            # share = (z+1)/N — see L1237 block for why the prior
            # `max(0.0, z)` clamp counted every loss as a draw.
            share = (float(z[candidate_seats[idx]]) + 1.0) / num_players
            share_sum += share
            if share > baseline + 1e-6:
                wins += 1
            elif share < baseline - 1e-6:
                losses += 1
            else:
                draws += 1
    finally:
        if owned_pool:
            pool.close()

    total = max(1, wins + losses + draws)
    win_rate = share_sum / total
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def eval_benchmark_setup(game_id: str, eval_profile: MctsProfile, heuristic_name: str | None,
                         heuristic_temperature: float, leaf_batch_size: int) -> dict[str, Any]:
    """What each side of `eval vs heuristic` searches with. The candidate uses
    the eval MCTS profile (as does gating, on both sides); the opponent is the
    named heuristic as the engine instantiates it (None = no heuristic
    benchmark). The two need not match (simulation budget, tail solver, FPU),
    so the win rate is only comparable between runs with the same pair of
    settings."""
    import dinoboard_engine
    candidate = {
        "simulations": eval_profile.simulations,
        "world_cap": eval_profile.world_cap,
        "c_puct": eval_profile.c_puct,
        "fpu_reduction": eval_profile.fpu_reduction,
        "uct2_arrival_visits": eval_profile.uct2_arrival_visits,
        "temperature": eval_profile.temperature,
        "opponent_selection": eval_profile.opponent_selection,
        "tail_solve_enabled": eval_profile.tail_solve_enabled,
        "tail_solve_depth_limit": eval_profile.tail_solve_depth_limit,
        "tail_solve_node_budget": eval_profile.tail_solve_node_budget,
        "tail_solve_auxiliary_weight": eval_profile.tail_solve_auxiliary_weight,
        "top_k": eval_profile.top_k,
        "leaf_batch_size": leaf_batch_size,
    }
    opponent = None
    if heuristic_name is not None:
        opponent = dict(dinoboard_engine.heuristic_settings(game_id, heuristic_name))
        opponent["temperature"] = heuristic_temperature
    return {"candidate": candidate, "heuristic_opponent": opponent}


def executed_simulations(episodes: list[dict[str, Any]]) -> int:
    """MCTS simulations the selfplay search actually ran for these episodes:
    the engine's per-ply counts (full or playout-cap fast budget; 0 for forced
    and tail-solved plies). Heuristic-guided plies count 0 here; see
    heuristic_simulations."""
    return sum(s["simulations"] for ep in episodes for s in ep["samples"])


def heuristic_simulations(episodes: list[dict[str, Any]]) -> int:
    """Simulations run inside the guiding heuristic on heuristic-guided plies
    (an mcts_pv heuristic searches with its own network and budget; rule
    heuristics report 0)."""
    return sum(s["heuristic_simulations"] for ep in episodes for s in ep["samples"])


def outcome_label(terminal_value: float) -> str:
    """'win' / 'loss' / 'draw' of one seat from its own terminal value.
    Terminal values are zero-sum with 0 = all tied, so the sign is the seat's
    result whatever the number of players: sharing first place among 3-4
    players is a win (+1/nw), finishing behind tied leaders a loss (-1/nl).
    Same rule as the eval / gating tallies in this module, which compare the
    share (z + 1) / N with the 1 / N baseline."""
    if terminal_value > 1e-6:
        return "win"
    if terminal_value < -1e-6:
        return "loss"
    return "draw"


def rotate_z_values(z_values: list[float], player: int, num_players: int = 0) -> list[float]:
    if not z_values:
        raise ValueError("z_values is empty — C++ must always populate z_values for every sample")
    n = len(z_values)
    return [z_values[(player + i) % n] for i in range(n)]


def compute_schedule_ratio(
    step: int,
    decay_end_step: int,
    initial_ratio: float,
    hold_steps: int = 0,
) -> float:
    """Three-segment schedule: hold → linear decay → zero.

    - step <= hold_steps                       → initial_ratio
    - step >= decay_end_step                   → 0.0
    - hold_steps < step < decay_end_step       → linear interpolation

    decay_end_step <= 0 disables the schedule (always 0). Configuration
    error if 0 < decay_end_step <= hold_steps — caller must validate.
    """
    if decay_end_step <= 0:
        return 0.0
    if decay_end_step <= hold_steps:
        raise ValueError(
            f"compute_schedule_ratio: decay_end_step ({decay_end_step}) must be "
            f"> hold_steps ({hold_steps})"
        )
    if step <= hold_steps:
        return initial_ratio
    if step >= decay_end_step:
        return 0.0
    span = decay_end_step - hold_steps
    return initial_ratio * (decay_end_step - step) / span


def adaptive_guidance_ratio(
    step: int, trigger_step: int | None, initial_ratio: float, decay_steps: int,
) -> float:
    """heuristic_guidance.until_win_rate: initial_ratio until the eval trigger
    fires at trigger_step, then linear decay to 0 over decay_steps."""
    if trigger_step is None:
        return initial_ratio
    return compute_schedule_ratio(
        step, trigger_step + decay_steps, initial_ratio, hold_steps=trigger_step)


def compute_lr(step: int, total_steps: int, base_lr: float, schedule: dict | None) -> float:
    """LR schedule. Returns base_lr when schedule is None or disabled.

    schedule = {"type": "cosine"|"step"|"constant", "lr_min": float, ...}
    - cosine: half-cosine from base_lr at step=1 to lr_min at step=total_steps
    - step: divide by `gamma` every `step_size` steps (default gamma=10, step_size=total_steps/3)
    - constant / None: base_lr unchanged
    """
    if not schedule or schedule["type"] == "constant":
        return base_lr
    sched_type = schedule["type"]
    if sched_type == "cosine":
        lr_min = schedule["lr_min"]
        if total_steps <= 1:
            return base_lr
        progress = (step - 1) / (total_steps - 1)
        progress = max(0.0, min(1.0, progress))
        return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if sched_type == "step":
        gamma = schedule.get("gamma", 10.0)
        step_size = schedule.get("step_size", max(1, total_steps // 3))
        n_drops = (step - 1) // step_size
        return base_lr / (gamma ** n_drops)
    raise ValueError(f"unknown lr_schedule type: {sched_type!r}")


def run_training_loop(
    game_id: str,
    game_config: dict,
    train_config: dict,
    output_dir: Path,
    steps: int = 1000,
    episodes_per_step: int = 200,
    eval_every: int = 50,
    eval_games: int = 40,
    eval_benchmarks: list[str] | None = None,
    max_workers: int = 4,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    seed: int = 20260323,
    save_every: int = 0,
    init_from: str | None = None,
    resume_from: str | None = None,
    safe_mode: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    train_cfg = train_config["training"]
    action_space = game_config["action_space"]
    feature_dim = game_config["feature_dim"]
    if "num_players" not in game_config:
        raise KeyError(
            "game_config missing 'num_players'. Inject from engine.game_metadata(game_id) "
            "before calling run_training_loop — JSON 'players.max' is not authoritative."
        )
    num_players = game_config["num_players"]

    # MCTS profiles drive selfplay / arena (gating) / eval. Top-level
    # max_game_plies is a game property shared across profiles.
    selfplay_profile = resolve_profile(game_id, "selfplay")
    eval_profile = resolve_profile(game_id, "eval")
    game_max_plies = max_game_plies(game_id)
    # Selfplay seeds are seed + step*10000 + i; eval / gating seeds live in a
    # disjoint range so eval never replays a deal that was trained on.
    eval_seed_base = seed + 1_000_000_000

    if save_every <= 0:
        save_every = eval_every if eval_every > 0 else 50

    # Checkpoints stay fp32 (resume, deployment, pool); only the copies fed to
    # selfplay / eval / gating inference are converted.
    inference_precision = train_cfg.get("inference_precision", "fp32")
    if inference_precision not in ("fp32", "fp16"):
        raise ValueError(
            f"training.inference_precision must be 'fp32' or 'fp16', got {inference_precision!r}")

    def inference_model(path: str) -> str:
        if inference_precision == "fp32":
            return path
        return convert_onnx_fp16(Path(path), Path(path).parent / "fp16" / Path(path).name)

    # Pick training device. Apple M-series → mps; NVIDIA → cuda; otherwise
    # CPU. Selfplay runs on its own ONNX session (CPU/CoreMLEP), independent
    # of this device — they share the GPU only at runtime, not at compile.
    if torch.cuda.is_available():
        # Explicit index: the ORT CUDA EP switches the calling thread's current
        # device to the last selfplay device, which a bare "cuda" would follow.
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Training device: {device}")

    torch.manual_seed(seed)
    import dinoboard_engine as _engine
    token_layout = _engine.game_metadata(game_id)["token_layout"]
    variable_token_dim = token_layout["token_dim"] if token_layout and token_layout["variable_tokens"] else 0
    net = create_model_from_config(
        train_config["network"],
        input_dim=feature_dim,
        policy_dim=action_space,
        num_players=num_players,
        token_layout=token_layout,
    ).to(device)

    if init_from and resume_from:
        raise ValueError("init_from and resume_from are mutually exclusive")

    if init_from:
        init_path = Path(init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from path does not exist: {init_path}")
        ckpt = torch.load(str(init_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"init_from checkpoint must be a dict with 'model_state_dict' key; "
                f"got top-level: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        net.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Initialized weights from {init_path} (optimizer state reset)")

    # Resume bookkeeping. Filled in below either from checkpoint.pt (preferred)
    # or by scanning models/model_step_*.onnx (fallback). Belief restore lands
    # later, after belief_net / belief_optimizer are constructed.
    resume_pv_optimizer_state: dict | None = None
    resume_ckpt: dict | None = None
    resume_belief_path: Path | None = None
    start_step: int = 1

    if resume_from:
        resume_dir = Path(resume_from)
        ckpt_path = resume_dir / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            net.load_state_dict(ckpt["model_state_dict"])
            resume_pv_optimizer_state = ckpt.get("optimizer_state_dict")
            resume_ckpt = ckpt
            start_step = int(ckpt["step"]) + 1
            logger.info(
                f"Resume: PV restored from {ckpt_path} (last step={ckpt['step']}, "
                f"continuing at step={start_step})"
            )
        else:
            # Fallback: scan ONNX checkpoints, pick highest step. Note ONNX
            # carries no optimizer state — Adam moments are reset.
            scan_dir = resume_dir / "models"
            onnx_steps = []
            if scan_dir.exists():
                for p in scan_dir.glob("model_step_*.onnx"):
                    try:
                        onnx_steps.append(int(p.stem.split("_")[-1]))
                    except ValueError:
                        continue
            if not onnx_steps:
                raise FileNotFoundError(
                    f"--resume {resume_dir}: no checkpoint.pt and no "
                    f"models/model_step_*.onnx found — nothing to resume from."
                )
            last_step = max(onnx_steps)
            start_step = last_step + 1
            logger.warning(
                f"Resume: checkpoint.pt missing; recovering PV weights from "
                f"models/model_step_{last_step:05d}.onnx (optimizer state reset)"
            )
            # ONNX → state_dict round-trip via the export sibling.
            from training.model import import_pv_onnx_state_dict  # local import; helper added next
            net.load_state_dict(import_pv_onnx_state_dict(
                scan_dir / f"model_step_{last_step:05d}.onnx", net))
            logger.info(
                f"Resume: PV restored from ONNX (last step={last_step}, "
                f"continuing at step={start_step})"
            )
        # Belief side restore is deferred — belief_net is constructed below.
        belief_ckpt_path = resume_dir / "belief_checkpoint.pt"
        if belief_ckpt_path.exists():
            resume_belief_path = belief_ckpt_path

    # Score-margin auxiliary head (KataGo-style score target). Absent block =
    # disabled; the trunk then trains on policy + value only, as before.
    score_head_cfg = train_cfg.get("score_head")
    score_head: AuxHead | None = None
    score_loss_weight = 0.0
    score_scale = 1.0
    if score_head_cfg is not None:
        if not isinstance(score_head_cfg, dict) or set(score_head_cfg) != {"loss_weight", "scale"}:
            raise ValueError(
                f"training.score_head must be an object with exactly 'loss_weight' and "
                f"'scale', got {score_head_cfg!r}")
        score_loss_weight = float(score_head_cfg["loss_weight"])
        score_scale = float(score_head_cfg["scale"])
        if not (math.isfinite(score_loss_weight) and score_loss_weight > 0):
            raise ValueError(f"training.score_head.loss_weight must be > 0, got {score_loss_weight}")
        if not (math.isfinite(score_scale) and score_scale > 0):
            raise ValueError(f"training.score_head.scale must be > 0, got {score_scale}")
        if not _engine.game_metadata(game_id)["has_final_score_extractor"]:
            raise ValueError(
                f"training.score_head is set but game {game_id!r} registers no "
                f"final_score_extractor in its GameBundle")
        score_head = AuxHead(net.embedding_dim, num_players).to(device)
        if resume_ckpt is not None and "score_head_state_dict" in resume_ckpt:
            score_head.load_state_dict(resume_ckpt["score_head_state_dict"])
            logger.info("Resume: score head restored")
        else:
            logger.info("score_head: freshly initialized")

    # Opponent-next-move auxiliary head (KataGo-style): predicts the search
    # policy of the next ply acted by a different seat. Absent block = disabled.
    opp_head_cfg = train_cfg.get("opponent_policy_head")
    opp_head: AuxHead | None = None
    opp_loss_weight = 0.0
    if opp_head_cfg is not None:
        if not isinstance(opp_head_cfg, dict) or set(opp_head_cfg) != {"loss_weight"}:
            raise ValueError(
                f"training.opponent_policy_head must be an object with exactly "
                f"'loss_weight', got {opp_head_cfg!r}")
        opp_loss_weight = float(opp_head_cfg["loss_weight"])
        if not (math.isfinite(opp_loss_weight) and opp_loss_weight > 0):
            raise ValueError(
                f"training.opponent_policy_head.loss_weight must be > 0, got {opp_loss_weight}")
        opp_head = AuxHead(net.embedding_dim, action_space).to(device)
        if resume_ckpt is not None and "opponent_policy_head_state_dict" in resume_ckpt:
            opp_head.load_state_dict(resume_ckpt["opponent_policy_head_state_dict"])
            logger.info("Resume: opponent policy head restored")
        else:
            logger.info("opponent_policy_head: freshly initialized")
    aux_heads = {n: h for n, h in (("score_head", score_head),
                                   ("opponent_policy_head", opp_head)) if h is not None}

    # EMA of the PV weights. When enabled, the averaged net (not the raw
    # training weights) is what gets exported for selfplay / eval / gating.
    ema_cfg = train_cfg.get("ema")
    ema_net: torch.nn.Module | None = None
    ema_decay = 0.0
    if ema_cfg is not None:
        if not isinstance(ema_cfg, dict) or set(ema_cfg) != {"decay"}:
            raise ValueError(f"training.ema must be an object with exactly 'decay', got {ema_cfg!r}")
        ema_decay = float(ema_cfg["decay"])
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"training.ema.decay must be in (0, 1), got {ema_decay}")
        ema_net = copy.deepcopy(net).requires_grad_(False)
        if resume_ckpt is not None and "ema_state_dict" in resume_ckpt:
            ema_net.load_state_dict(resume_ckpt["ema_state_dict"])
            logger.info("Resume: EMA weights restored")
        else:
            logger.info("ema: initialized from the current PV weights")
    export_net = ema_net if ema_net is not None else net

    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = build_pv_optimizer(net, aux_heads, learning_rate, weight_decay,
                                   resume_pv_optimizer_state)
    if resume_pv_optimizer_state is not None:
        logger.info("Resume: PV optimizer state restored")
    lr_schedule = train_cfg.get("lr_schedule")  # None = constant lr

    initial_onnx = models_dir / "model_init.onnx"
    if resume_from:
        # Selfplay of the first resumed step must use the restored weights,
        # not the original run's random-init export.
        resumed_onnx = models_dir / "model_latest.onnx"
        export_onnx(export_net, resumed_onnx, feature_dim)
        current_model_path = str(resumed_onnx)
    else:
        export_onnx(export_net, initial_onnx, feature_dim)
        current_model_path = str(initial_onnx)
    best_model_path = current_model_path

    if resume_from:
        best_candidate = models_dir / "model_best.onnx"
        if best_candidate.exists():
            best_model_path = str(best_candidate)
            logger.info(f"Resume: best model loaded from {best_candidate}")

    # Scheduling parameters (three-segment: hold → linear decay → zero).
    # heuristic_guidance lives at train.json's top level (it is a training
    # schedule, not a runtime selector). Runtime selectors stay in
    # heuristic.json and are loaded into train_config["heuristic"].
    if "heuristic_guidance" not in train_config:
        raise KeyError(
            "train_config missing top-level 'heuristic_guidance' block. "
            "Pipeline expects training.cli to load games/<id>/config/"
            "train.json's heuristic_guidance into train_config before "
            "invoking run_training_loop.")
    if "heuristic" not in train_config:
        raise KeyError(
            "train_config missing 'heuristic' block. Pipeline expects "
            "training.cli to load games/<id>/config/heuristic.json into "
            "train_config['heuristic'] before invoking run_training_loop.")
    guidance_cfg = train_config["heuristic_guidance"]
    heuristic_guidance_hold_steps = int(guidance_cfg["hold_steps"])
    heuristic_guidance_steps = int(guidance_cfg["steps"])
    heuristic_guidance_initial = float(guidance_cfg["initial_ratio"])
    heuristic_guidance_temperature = float(guidance_cfg["temperature"])
    heuristic_guidance_name = guidance_cfg["name"]
    guidance_until = guidance_cfg.get("until_win_rate")
    guidance_trigger_step: int | None = None
    if guidance_until is not None:
        if eval_every <= 0:
            raise ValueError(
                "heuristic_guidance.until_win_rate needs eval_every > 0: the "
                "trigger reads the eval win rate.")
        if guidance_until["benchmark"] not in (eval_benchmarks or []):
            raise ValueError(
                f"heuristic_guidance.until_win_rate.benchmark "
                f"{guidance_until['benchmark']!r} is not in eval_benchmarks {eval_benchmarks}.")
        if resume_ckpt is not None:
            guidance_trigger_step = resume_ckpt.get("guidance_trigger_step")
    heuristic_cfg = train_config["heuristic"]
    selectors = heuristic_cfg["selectors"]
    selector_eval = selectors["eval_vs_heuristic"]["name"]
    eval_constrained_temperature = float(selectors["eval_vs_heuristic"].get("temperature", 0.0))
    eval_free_temperature = eval_constrained_temperature
    training_filter_hold_steps = train_cfg.get("training_filter_hold_steps", 0)
    training_filter_steps = train_cfg.get("training_filter_steps", 0)
    training_filter_initial = train_cfg.get("training_filter_initial_ratio", 0.5)

    simulations_start = train_cfg.get("simulations_start", selfplay_profile.simulations)
    simulations_full = selfplay_profile.simulations

    # Playout cap randomization (KataGo): fixed full-search size replaces the
    # profile's simulations and the simulations_start ramp.
    playout_cap = train_cfg.get("playout_cap")
    if playout_cap is not None:
        keys = {"full_simulations", "fast_simulations", "full_search_prob"}
        if set(playout_cap) != keys:
            raise ValueError(f"training.playout_cap must have exactly {sorted(keys)}, got {sorted(playout_cap)}")
        if "simulations_start" in train_cfg:
            raise ValueError("training.playout_cap and training.simulations_start are mutually exclusive")
        sc_block = train_cfg.get("selfplay_concurrency")
        if not (sc_block and sc_block.get("enabled")):
            raise ValueError("training.playout_cap requires training.selfplay_concurrency.enabled=true")
        pc_full = int(playout_cap["full_simulations"])
        pc_fast = int(playout_cap["fast_simulations"])
        pc_prob = float(playout_cap["full_search_prob"])
        if not 0 < pc_fast < pc_full:
            raise ValueError(f"training.playout_cap needs 0 < fast_simulations < full_simulations, got {pc_fast}, {pc_full}")
        if not 0.0 < pc_prob < 1.0:
            raise ValueError(f"training.playout_cap.full_search_prob must be in (0, 1), got {pc_prob}")
        simulations_start = simulations_full = pc_full

    # Opponent pool — opt-in via train.json. Default off (mirror selfplay) so
    # adding the feature didn't change any existing game's training behavior.
    # If `opponent_pool_enabled` is set without `opponent_pool_self_ratio`
    # we still throw — partial config is a bug.
    opponent_pool_enabled = train_cfg.get("opponent_pool_enabled", False)
    if opponent_pool_enabled:
        opponent_pool_self_ratio = train_cfg["opponent_pool_self_ratio"]
    else:
        opponent_pool_self_ratio = 1.0  # unused; full mirror

    replay_buffer_size = int(train_cfg.get("replay_buffer_size",
                                            episodes_per_step * 50 * 20))
    replay_buffer: deque[tuple[list, list, list, list]] = deque(maxlen=replay_buffer_size)
    compact_dtype = parse_compact_replay(train_cfg)
    augmenter = parse_symmetry_augmentation(
        train_cfg, game_id, feature_dim=feature_dim, action_space=action_space, token_layout=token_layout)
    augment_rng = np.random.default_rng(seed + 17)
    if augmenter is not None:
        logger.info(f"Symmetry augmentation: {augmenter.count} board symmetries, one drawn per sampled example")

    # Belief net plumbing (Plan §2.2 / §2.3). Activated only when the game
    # registers a belief feature extractor AND a 'belief' block is present
    # (CLI loads games/<id>/config/belief.json into train_config["belief"]).
    # Capability-driven — no runtime branch on game id.
    import dinoboard_engine as _engine
    meta = _engine.game_metadata(game_id)
    belief_enabled = meta["has_belief_extractor"] and "belief" in train_config
    belief_net = None
    belief_optimizer = None
    belief_buffer: deque[tuple] | None = None  # lists, or ndarrays under compact_replay
    belief_lr_schedule = None
    belief_batch_size = 1024
    belief_batches_per_step = 1
    belief_class_count = meta["belief_logit_count"]  # (N-1) * K — used for shape check
    current_belief_model_path = ""
    if belief_enabled:
        belief_cfg = train_config["belief"]
        belief_in = meta["belief_feature_dim"]
        belief_out = meta["belief_logit_count"]
        belief_net = create_belief_model_from_config(
            belief_cfg, input_dim=belief_in, output_dim=belief_out,
        ).to(device)
        b_train = belief_cfg["training"]
        # Belief training schedule mirrors PV — see find_belief_config: PV-side
        # knobs (batch_size / batches_per_step / lr_schedule / lr / weight_decay)
        # are forbidden in belief.json::training so the two networks can't
        # quietly fork. Only `loss` (belief-specific kind) is read here.
        belief_lr_init = learning_rate
        belief_lr_schedule = lr_schedule
        belief_batch_size = batch_size
        belief_batches_per_step = int(train_cfg["train_batches_per_step"])
        belief_loss_kind = str(b_train["loss"])
        belief_weight_decay = weight_decay
        belief_optimizer = torch.optim.AdamW(
            belief_net.parameters(),
            lr=belief_lr_init,
            weight_decay=belief_weight_decay,
        )
        belief_buffer = deque(maxlen=replay_buffer_size)
        if resume_belief_path is not None:
            b_ckpt = torch.load(str(resume_belief_path), map_location="cpu", weights_only=False)
            belief_net.load_state_dict(b_ckpt["model_state_dict"])
            belief_optimizer.load_state_dict(b_ckpt["optimizer_state_dict"])
            logger.info(
                f"Resume: belief restored from {resume_belief_path} "
                f"(last step={b_ckpt.get('step', '?')})"
            )
        elif resume_from:
            logger.warning(
                f"Resume: no belief_checkpoint.pt at {resume_from}; belief net "
                f"reinitialized from scratch (PV continues from its checkpoint)."
            )
        # Initial export so the next step's selfplay can use the net via path.
        # Random-init for fresh runs; restored weights on resume — either way
        # the file reflects belief_net's current state_dict.
        belief_init_path = models_dir / "belief_init.onnx"
        export_belief_onnx(belief_net, belief_init_path, belief_in)
        current_belief_model_path = str(belief_init_path)
        # Per-opp character count K (kCharacterCount in C++).
        # logit_count = (N-1) * K, so K = logit_count // (N-1).
        belief_K = belief_out // max(1, num_players - 1) if num_players > 1 else belief_out
        logger.info(
            f"  belief: enabled (input_dim={belief_in}, logits={belief_out}, "
            f"K={belief_K}, batch={belief_batch_size}, "
            f"batches/step={belief_batches_per_step})"
        )
    else:
        belief_K = 0
        belief_loss_kind = "ce"
        logger.info(f"  belief: disabled (no extractor or no 'belief' block)")

    logger.info(f"Starting training: game={game_id}, steps={steps}, episodes/step={episodes_per_step}")
    if guidance_until is not None:
        logger.info(
            f"  heuristic_guidance: initial_ratio={heuristic_guidance_initial} until "
            f"{guidance_until['benchmark']} win_rate >= {guidance_until['threshold']} "
            f"(min hold {heuristic_guidance_hold_steps}, max hold "
            f"{guidance_until['max_hold_steps']}), then decay over "
            f"{guidance_until['decay_steps']} steps; triggered at={guidance_trigger_step}"
        )
    else:
        logger.info(
            f"  heuristic_guidance: hold={heuristic_guidance_hold_steps}, "
            f"decay_end={heuristic_guidance_steps}, initial_ratio={heuristic_guidance_initial}"
        )
    logger.info(
        f"  training_filter: hold={training_filter_hold_steps}, "
        f"decay_end={training_filter_steps}, initial_ratio={training_filter_initial}"
    )
    logger.info(f"  simulations: start={simulations_start}, full={simulations_full}")
    logger.info(f"  uct2_arrival_visits: selfplay={selfplay_profile.uct2_arrival_visits}, "
                f"eval/gating={eval_profile.uct2_arrival_visits} "
                f"(mcts_pv heuristics are pinned to False, whatever this is)")
    logger.info(f"  tail_solve (selfplay): enabled={selfplay_profile.tail_solve_enabled}")
    logger.info(f"  opponent_selection (selfplay): {selfplay_profile.opponent_selection}")
    logger.info(
        f"  opponent_pool: enabled={opponent_pool_enabled}, "
        f"self_ratio={opponent_pool_self_ratio}"
    )
    logger.info(f"  replay_buffer: maxlen={replay_buffer_size}, "
                f"storage={'compact/' + compact_dtype.name if compact_dtype is not None else 'lists'}")
    logger.info(f"  score_head: {score_head_cfg if score_head is not None else 'disabled'}")
    logger.info(f"  opponent_policy_head: {opp_head_cfg if opp_head is not None else 'disabled'}")
    logger.info(f"  ema: {ema_cfg if ema_net is not None else 'disabled'}")
    if lr_schedule:
        logger.info(f"  lr_schedule: {lr_schedule} (base_lr={learning_rate})")
    else:
        logger.info(f"  lr: {learning_rate} (constant)")

    # Persistent worker pool — kept alive across all steps so we don't
    # spawn N×steps×3 processes per run (the old per-step
    # ProcessPoolExecutor was a stability risk on WSL where worker
    # processes occasionally SIGSEGV/SIGKILL during teardown).
    _sc_cfg = train_cfg.get("selfplay_concurrency") or {}
    pool_workers = max_workers
    worker_pool = PersistentPool(pool_workers)

    # Concurrent (single-process, multi-thread, shared-ONNX) eval routing. When
    # selfplay_concurrency is on, eval/gating go through the batched C++
    # bindings (run_concurrent_arena_batch / run_concurrent_eval_vs_heuristic_
    # batch) instead of the ProcessPool — one shared ONNX session per distinct
    # model rather than one stack per fork worker. num_threads can be large
    # (shared session, no per-thread ONNX cost), so it tracks selfplay
    # parallelism, NOT the memory-capped process-pool width. None → legacy
    # ProcessPool path (also used for constrained eval, which the batched
    # binding doesn't support).
    if _sc_cfg.get("enabled", False):
        _ev_eval = _sc_cfg["evaluator"]
        concurrent_eval_cfg = {
            "num_threads": max_workers,
            "batch_size": int(_ev_eval["batch_size"]),
            "batch_timeout_us": int(_ev_eval["batch_timeout_us"]),
            "max_queue_size": int(_ev_eval["max_queue_size"]),
            "intra_op_num_threads": int(_ev_eval["intra_op_num_threads"]),
            "num_inference_threads": int(_ev_eval["num_inference_threads"]),
            "providers": list(_ev_eval.get("providers", []) or []),
            "devices": list(_ev_eval.get("devices", [])),
            "leaf_batch_size": int(_sc_cfg["leaf_batch_size"]),
        }
    else:
        concurrent_eval_cfg = None

    # Benchmark transparency: state once what eval / gating search with. The
    # heuristic opponent is resolved only when a heuristic benchmark will run.
    logger.info(f"  eval: benchmarks={eval_benchmarks or []}, every={eval_every} steps, "
                f"games={eval_games}")
    if eval_every > 0:
        vs_heuristic = any(b in ("heuristic_constrained", "heuristic_free")
                           for b in (eval_benchmarks or []))
        eval_setup = eval_benchmark_setup(
            game_id, eval_profile, selector_eval if vs_heuristic else None,
            eval_constrained_temperature,
            concurrent_eval_cfg["leaf_batch_size"] if concurrent_eval_cfg is not None else 1)
        logger.info(f"  eval candidate (mcts profile 'eval'; gating uses it on both sides): "
                    f"{eval_setup['candidate']}")
        if vs_heuristic:
            logger.info(f"  eval heuristic opponent (heuristic.json selector "
                        f"'eval_vs_heuristic'): {eval_setup['heuristic_opponent']}")

    # Graceful stop: finish the current step (incl. eval + checkpoint), then
    # exit. SIGINT alone is not enough — a process started with `cmd &` from a
    # non-interactive shell inherits SIGINT ignored and Python keeps it so.
    stop_signal: list[int] = []
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, lambda s, _f: stop_signal.append(s))

    try:
        for step in range(start_step, steps + 1):
            t0 = time.perf_counter()

            current_lr = compute_lr(step, steps, learning_rate, lr_schedule)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Compute scheduled ratios (three-segment: hold → decay → zero)
            if guidance_until is None:
                heuristic_ratio = compute_schedule_ratio(
                    step,
                    heuristic_guidance_steps,
                    heuristic_guidance_initial,
                    hold_steps=heuristic_guidance_hold_steps,
                )
            else:
                if guidance_trigger_step is None and step > guidance_until["max_hold_steps"]:
                    guidance_trigger_step = step - 1
                    logger.info(
                        f"  guidance: max_hold_steps={guidance_until['max_hold_steps']} reached "
                        f"without win_rate >= {guidance_until['threshold']}; decaying from step {step}")
                heuristic_ratio = adaptive_guidance_ratio(
                    step, guidance_trigger_step, heuristic_guidance_initial,
                    int(guidance_until["decay_steps"]))
            filter_ratio = compute_schedule_ratio(
                step,
                training_filter_steps,
                training_filter_initial,
                hold_steps=training_filter_hold_steps,
            )

            # Ramp simulations
            sim_frac = min(1.0, step / max(1, steps * 0.3))
            current_sims = int(simulations_start + (simulations_full - simulations_start) * sim_frac)

            # Action filter is bool-gated by profile; the schedule's effective
            # ratio is zeroed when ai_use_action_filter=false (allows ablation
            # without deleting the schedule config).
            effective_filter_ratio = (
                filter_ratio if selfplay_profile.ai_use_action_filter else 0.0
            )

            selfplay_cfg = profile_as_dict(selfplay_profile)
            selfplay_cfg["simulations"] = current_sims
            selfplay_cfg["max_game_plies"] = game_max_plies
            selfplay_cfg["heuristic_guidance_ratio"] = heuristic_ratio
            # Selfplay heuristic branch uses guidance temperature (high = diverse
            # exploration during warm period). Eval vs heuristic uses the
            # eval_vs_heuristic selector's temperature (low = strength benchmark).
            selfplay_cfg["heuristic_temperature"] = heuristic_guidance_temperature
            selfplay_cfg["heuristic_selector"] = heuristic_guidance_name
            selfplay_cfg["training_filter_ratio"] = effective_filter_ratio
            selfplay_cfg["belief_model_path"] = current_belief_model_path
            selfplay_cfg["safe_mode"] = safe_mode
            if playout_cap is not None:
                selfplay_cfg["fast_simulations"] = pc_fast
                selfplay_cfg["full_search_prob"] = pc_prob
            # Phase 1 batched/threaded path opt-in: forward the
            # selfplay_concurrency block from train.json so run_selfplay_batch
            # can short-circuit to the C++ batched runner when enabled=true.
            sc_block = train_cfg.get("selfplay_concurrency")
            if sc_block is not None:
                selfplay_cfg["selfplay_concurrency"] = sc_block
            selfplay_cfg["compact_samples"] = compact_dtype is not None

            # Recompute the pool every step — newly saved checkpoints automatically
            # enrol the step after they land. Empty pool ⇒ all episodes mirror.
            # Disabled via train.json `opponent_pool_enabled: false` ⇒ same.
            pool_paths = _collect_pool_paths(models_dir)
            t_selfplay_start = time.perf_counter()
            episodes, sp_stats = run_selfplay_batch(
                game_id, inference_model(current_model_path), episodes_per_step,
                seed + step * 10000, selfplay_cfg, max_workers,
                pool_paths=pool_paths, num_players=num_players,
                pool_enabled=opponent_pool_enabled,
                self_ratio=opponent_pool_self_ratio,
                pool=worker_pool)
            selfplay_elapsed = time.perf_counter() - t_selfplay_start
            logger.info(
                f"  pool: enabled={sp_stats['pool_enabled']}, "
                f"|P|={sp_stats['pool_size']}, "
                f"workers(self/pool)={sp_stats['n_self_workers']}/"
                f"{sp_stats['n_pool_workers']}, "
                f"episodes(self/pool)={sp_stats['n_self']}/"
                f"{sp_stats['n_pool']}, "
                f"dropped={sp_stats['dropped']}"
            )

            # `step_samples` counts what actually entered the replay buffer
            # (after pool-mode filtering), not the raw episode-output count.
            step_samples = 0
            belief_step_samples = 0
            for ep in episodes:
                latest_seat = ep["latest_seat"]  # int for pool, None for mirror
                # Belief samples (Plan §2.2). One per ply per observer; pool-mode
                # filter applies — only keep observer == latest_seat. (The latest
                # net's belief is what we're training; freezing the pool's net
                # means its belief samples would train against the frozen weights.)
                if belief_enabled:
                    for bs in ep.get("belief_samples", []):
                        if latest_seat is not None and bs["observer"] != latest_seat:
                            continue
                        belief_buffer.append((
                            bs["features"],
                            bs["hand_counts"],   # [N-1][K]
                            bs["remaining"],     # [K]
                            bs["alive_per_opp"], # [N-1]
                        ))
                        belief_step_samples += 1
                if score_head is not None and "final_scores" not in ep:
                    raise ValueError(
                        f"step {step}: training.score_head is enabled but a selfplay "
                        f"episode carries no 'final_scores'")
                ep_samples = ep["samples"]
                next_opp = (next_opponent_indices([s["player"] for s in ep_samples])
                            if opp_head is not None else None)
                for si, sample in enumerate(ep_samples):
                    # Pool-mode: keep only the latest model's seat samples — the
                    # opp's seat was driven by a frozen historical net and would
                    # train against itself if we kept it.
                    if latest_seat is not None and sample["player"] != latest_seat:
                        continue
                    if not sample["full_search"]:
                        continue
                    feats = sample["features"]
                    if not valid_feature_length(len(feats), feature_dim, variable_token_dim):
                        raise ValueError(
                            f"step {step}: feature length {len(feats)} is not a whole encoding "
                            f"(feature_dim {feature_dim}, variable token_dim {variable_token_dim})")
                    to_policy = sparse_policy if compact_dtype is not None else normalize_policy
                    if compact_dtype is not None:
                        feats = feats.astype(compact_dtype)
                    policy = to_policy(
                        sample["policy_action_ids"],
                        sample["policy_action_visits"],
                        action_space,
                    )
                    z_vals = sample["z_values"]
                    player = sample["player"]
                    z_rotated = rotate_z_values(z_vals, player, num_players)
                    mask = sample["legal_mask"]
                    score_target = (
                        score_margin_targets(ep["final_scores"], player, num_players, score_scale)
                        if score_head is not None else None)
                    opp_target = None
                    if opp_head is not None:
                        j = next_opp[si]
                        if j is None:
                            opp_target = (to_policy([], [], action_space), 0.0)
                        else:
                            opp_policy = to_policy(
                                ep_samples[j]["policy_action_ids"],
                                ep_samples[j]["policy_action_visits"],
                                action_space,
                            )
                            has_target = (len(opp_policy[0]) > 0 if compact_dtype is not None
                                          else any(opp_policy))
                            opp_target = (opp_policy, 1.0 if has_target else 0.0)
                    replay_buffer.append((feats, policy, z_rotated, mask, score_target, opp_target))
                    step_samples += 1

            # `pX=...` log column: per-seat win-share accumulated over
            # the step's episodes. Each episode contributes share =
            # (z + 1) / N from terminal_z_values to that seat's tally —
            # this turns any zero-sum z (winner-takes-all OR liarsdice
            # loser-pays) into a share ∈ [0, 1] per seat that sums to 1
            # over seats. Sum of pX over the step then equals episode
            # count (e.g. 100 for 100 episodes) and a balanced model
            # produces ~equal pX. Draws (winner < 0, z all zeros) self-
            # split at 1/N per seat, so we don't report a separate draw
            # column — it would be folded into pX evenly anyway.
            per_player_wins = [0.0] * num_players
            inv_n = 1.0 / num_players
            for ep in episodes:
                z = ep["terminal_z_values"]
                if not z:
                    continue
                for p, v in enumerate(z):
                    per_player_wins[p] += (v + 1.0) * inv_n
            plies_list = [ep["total_plies"] for ep in episodes]
            avg_plies = sum(plies_list) / max(1, len(plies_list))
            max_plies_hit = sum(1 for p in plies_list if p >= game_max_plies)

            ts_attempts = sum(ep["tail_solve_attempts"] for ep in episodes)
            ts_completed = sum(ep["tail_solve_completed"] for ep in episodes)
            ts_successes = sum(ep["tail_solve_successes"] for ep in episodes)

            if not replay_buffer:
                logger.warning(f"Step {step}: no training samples, skipping")
                continue

            # Sample-then-tensorize: convert ONLY the sampled minibatch to
            # tensors each step, not the whole replay buffer. Tensorizing all
            # ~replay_buffer_size rows per step — when only train_batches×
            # batch_size of them are ever used — made step time scale with
            # buffer occupancy (near-instant while the buffer was small,
            # seconds once it filled to maxlen). `list(buffer)` is a cheap
            # shallow copy that gives O(1) indexing (deque indexing is O(n)).
            buf = list(replay_buffer)
            n = len(buf)
            grad_clip = train_cfg.get("grad_clip_norm", 1.0)
            train_batches = train_cfg["train_batches_per_step"]
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_score_loss = 0.0
            total_opp_loss = 0.0
            t_train_start = time.perf_counter()
            for b in range(train_batches):
                idx = torch.randint(n, (min(batch_size, n),)).tolist()
                batch = [buf[i] for i in idx]
                # One random board symmetry per example, applied to its
                # features and to every action-indexed row (policy, legal
                # mask, opponent policy); values and scores are invariant.
                sym = augmenter.sample(augment_rng, len(batch)) if augmenter is not None else None
                if variable_token_dim:
                    feats_np = pad_token_rows([s[0] for s in batch], variable_token_dim)
                elif compact_dtype is not None:
                    feats_np = np.stack([s[0] for s in batch])
                elif sym is not None:
                    feats_np = np.asarray([s[0] for s in batch], dtype=np.float32)
                if variable_token_dim or compact_dtype is not None or sym is not None:
                    if sym is not None:
                        feats_np = augmenter.features(feats_np, sym)
                    feat_tensor = rows_to_tensor(feats_np, device)
                else:
                    feat_tensor = torch.tensor(
                        [s[0] for s in batch], dtype=torch.float32, device=device)
                if compact_dtype is not None or sym is not None:
                    if compact_dtype is not None:
                        policy_np = densify_policies([s[1] for s in batch], action_space)
                        mask_np = np.stack([s[3] for s in batch])
                    else:
                        policy_np = np.asarray([s[1] for s in batch], dtype=np.float32)
                        mask_np = np.asarray([s[3] for s in batch], dtype=np.float32)
                    if sym is not None:
                        policy_np = augmenter.actions(policy_np, sym)
                        mask_np = augmenter.actions(mask_np, sym)
                    policy_tensor = rows_to_tensor(policy_np, device)
                    mask_tensor = rows_to_tensor(mask_np, device)
                else:
                    policy_tensor = torch.tensor(
                        [s[1] for s in batch], dtype=torch.float32, device=device)
                    mask_tensor = torch.tensor(
                        [s[3] for s in batch], dtype=torch.float32, device=device)
                value_tensor = torch.tensor(
                    [s[2] for s in batch], dtype=torch.float32, device=device)
                score_tensor = None
                if score_head is not None:
                    score_tensor = torch.tensor(
                        [s[4] for s in batch], dtype=torch.float32, device=device)
                opp_tensor = opp_weight_tensor = None
                if opp_head is not None:
                    opp_rows = [s[5][0] for s in batch]
                    if compact_dtype is not None or sym is not None:
                        opp_np = (densify_policies(opp_rows, action_space) if compact_dtype is not None
                                  else np.asarray(opp_rows, dtype=np.float32))
                        if sym is not None:
                            opp_np = augmenter.actions(opp_np, sym)
                        opp_tensor = rows_to_tensor(opp_np, device)
                    else:
                        opp_tensor = torch.tensor(opp_rows, dtype=torch.float32, device=device)
                    opp_weight_tensor = torch.tensor(
                        [s[5][1] for s in batch], dtype=torch.float32, device=device)
                metrics = train_step(
                    net, optimizer,
                    feat_tensor, policy_tensor, value_tensor,
                    legal_mask=mask_tensor,
                    grad_clip_norm=grad_clip,
                    score_head=score_head,
                    score_targets=score_tensor,
                    score_loss_weight=score_loss_weight,
                    opp_head=opp_head,
                    opp_targets=opp_tensor,
                    opp_weights=opp_weight_tensor,
                    opp_loss_weight=opp_loss_weight,
                )
                if ema_net is not None:
                    ema_update(ema_net, net, ema_decay)
                total_loss += metrics["loss"]
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_score_loss += metrics.get("score_loss", 0.0)
                total_opp_loss += metrics.get("opp_loss", 0.0)
            avg_loss = total_loss / max(1, train_batches)
            avg_opp_loss = total_opp_loss / max(1, train_batches)
            avg_policy_loss = total_policy_loss / max(1, train_batches)
            avg_value_loss = total_value_loss / max(1, train_batches)
            avg_score_loss = total_score_loss / max(1, train_batches)

            latest_onnx = models_dir / "model_latest.onnx"
            export_onnx(export_net, latest_onnx, feature_dim)
            current_model_path = str(latest_onnx)

            if step % save_every == 0:
                # Atomic: a truncated model_step_*.onnx would be enrolled in
                # the opponent pool (and the ONNX resume fallback) on resume.
                with atomic_write(models_dir / f"model_step_{step:05d}.onnx") as tmp:
                    export_onnx(export_net, tmp, feature_dim)

            # Belief training step — multiset categorical CE
            # (BELIEF_NETWORK.md §7). Per opp the network emits 20 logits split
            # into a 15-way two-card-multiset branch and a 5-way single-card
            # branch; alive_per_opp[i] selects the branch (alive=2 → 15-way,
            # alive=1 → 5-way, alive=0 → skip in loss). Target is one-hot at
            # the multiset id derived from hand_counts; remaining[R] is unused
            # by the trainer (only needed at sim entry for feasibility masking).
            belief_loss_val = None
            if belief_enabled and belief_buffer:
                belief_lr = compute_lr(step, steps, belief_lr_init, belief_lr_schedule)
                for pg in belief_optimizer.param_groups:
                    pg["lr"] = belief_lr
                belief_loss_val = _train_belief_step(
                    belief_net, belief_optimizer, belief_buffer,
                    num_players=num_players, K=belief_K,
                    batch_size=belief_batch_size,
                    batches_per_step=belief_batches_per_step,
                    grad_clip=train_cfg.get("grad_clip_norm", 1.0),
                    loss=belief_loss_kind,
                )
                if not math.isfinite(belief_loss_val):
                    raise RuntimeError(
                        f"belief loss diverged: {belief_loss_val} — abort to "
                        "prevent training-on-NaN drift (Plan §2.4)"
                    )
                belief_latest = models_dir / "belief_latest.onnx"
                export_belief_onnx(belief_net, belief_latest, meta["belief_feature_dim"])
                current_belief_model_path = str(belief_latest)
                if step % save_every == 0:
                    with atomic_write(models_dir / f"belief_step_{step:05d}.onnx") as tmp:
                        export_belief_onnx(belief_net, tmp, meta["belief_feature_dim"])

            elapsed = time.perf_counter() - t0
            train_elapsed = time.perf_counter() - t_train_start
            log_parts = [
                f"Step {step}/{steps}: p_loss={avg_policy_loss:.4f}, "
                f"v_loss={avg_value_loss:.4f}"
                + (f", s_loss={avg_score_loss:.4f}" if score_head is not None else "")
                + (f", o_loss={avg_opp_loss:.4f}" if opp_head is not None else ""),
                f"episodes={len(episodes)}, samples={n}",
                # pX = (z+1)/N share accumulated over episodes; total over p
                # equals len(episodes). One decimal — share is fractional.
                ", ".join(f"p{p}={per_player_wins[p]:.1f}" for p in range(num_players)),
                f"plies={avg_plies:.1f} (cap_hits={max_plies_hit}/{len(episodes)})",
                f"sims={current_sims}",
            ]
            if lr_schedule:
                log_parts.append(f"lr={current_lr:.2e}")
            if heuristic_ratio > 0:
                log_parts.append(f"h_ratio={heuristic_ratio:.2f}")
            if filter_ratio > 0:
                log_parts.append(f"f_ratio={filter_ratio:.2f}")
            if ts_attempts > 0:
                log_parts.append(f"ts={ts_successes}/{ts_completed}/{ts_attempts}")
            # smp/s and mcts/s normalize against selfplay wall-clock only —
            # they're meant to express "how fast does selfplay generate
            # samples", and folding the train phase into the divisor would
            # silently make a faster network look like slower selfplay.
            # mcts/s counts the simulations every ply really ran: samples *
            # sims would miss the playout-cap fast plies (not samples, fewer
            # sims) and overcount forced / tail-solved ones. It is the total
            # search work of the step, including what the guiding heuristic
            # simulated on guided plies; that share is shown in brackets
            # because it runs another network at another cost per simulation.
            sps = step_samples / max(0.01, selfplay_elapsed)
            guided_sims = heuristic_simulations(episodes)
            mps = (executed_simulations(episodes) + guided_sims) / max(0.01, selfplay_elapsed)
            rates = f"{sps:.0f}smp/s, {mps:.0f}mcts/s"
            if guided_sims > 0:
                rates += f" (heuristic {guided_sims / max(0.01, selfplay_elapsed):.0f})"
            log_parts.append(rates)
            if belief_enabled:
                buf_n = len(belief_buffer) if belief_buffer is not None else 0
                if belief_loss_val is not None:
                    log_parts.append(
                        f"b_loss={belief_loss_val:.4f}, b_buf={buf_n}, "
                        f"b_step={belief_step_samples}"
                    )
                else:
                    log_parts.append(f"b_buf={buf_n}, b_step={belief_step_samples}")
            # Game-defined per-episode stats (averaged over the step's
            # episodes). Splendor's extractor returns {"turns": main_actions
            # / num_players}, which excludes return-token sub-actions and
            # noble-pick sub-actions so the count reflects real game rounds,
            # not raw plies.
            custom_keys: dict[str, list[float]] = {}
            for ep in episodes:
                cs = ep.get("custom_stats")
                if not cs:
                    continue
                for k, v in cs.items():
                    custom_keys.setdefault(k, []).append(float(v))
            for k, vs in sorted(custom_keys.items()):
                avg = sum(vs) / len(vs)
                # Rate-like keys (averages already in [0,1]) need more precision
                # than count-like keys; the convention is `_rate` suffix.
                fmt = ".3f" if k.endswith("_rate") else ".1f"
                log_parts.append(f"{k}={avg:{fmt}}")
            log_parts.append(
                f"time={elapsed:.1f}s (selfplay={selfplay_elapsed:.1f}s, "
                f"train={train_elapsed:.1f}s)"
            )
            logger.info(", ".join(log_parts))

            if eval_every > 0 and step % eval_every == 0:
                eval_model = current_model_path
                eval_infer = inference_model(eval_model)
                h_temp = eval_constrained_temperature
                free_h_temp = eval_free_temperature
                benchmarks = eval_benchmarks or []
                eval_win_rates: dict[str, float] = {}

                for bench in benchmarks:
                    if bench == "heuristic_constrained":
                        r = run_eval_vs_heuristic(
                            game_id, eval_infer, eval_games, eval_seed_base + step * 100000,
                            eval_profile.simulations, eval_profile.world_cap,
                            True, h_temp, max_workers,
                            opponent_selection=eval_profile.opponent_selection,
                            max_game_plies=game_max_plies,
                            c_puct=eval_profile.c_puct,
                            temperature=eval_profile.temperature,
                            tail_solve_enabled=eval_profile.tail_solve_enabled,
                            tail_solve_depth_limit=eval_profile.tail_solve_depth_limit,
                            tail_solve_node_budget=eval_profile.tail_solve_node_budget,
                            tail_solve_auxiliary_weight=eval_profile.tail_solve_auxiliary_weight,
                            belief_model_path=current_belief_model_path,
                            pool=worker_pool,
                            heuristic_selector=selector_eval,
                            concurrent_cfg=concurrent_eval_cfg,
                            safe_mode=safe_mode,
                            top_k=eval_profile.top_k,
                            fpu_reduction=eval_profile.fpu_reduction,
                            uct2_arrival_visits=eval_profile.uct2_arrival_visits)
                        eval_win_rates[bench] = r["win_rate"]
                        logger.info(
                            f"  eval vs heuristic (constrained): win_rate={r['win_rate']:.1%} "
                            f"(W={r['wins']}, L={r['losses']}, D={r['draws']})")
                    elif bench == "heuristic_free":
                        r = run_eval_vs_heuristic(
                            game_id, eval_infer, eval_games, eval_seed_base + step * 100000 + 50000,
                            eval_profile.simulations, eval_profile.world_cap,
                            False, free_h_temp, max_workers,
                            opponent_selection=eval_profile.opponent_selection,
                            max_game_plies=game_max_plies,
                            c_puct=eval_profile.c_puct,
                            temperature=eval_profile.temperature,
                            tail_solve_enabled=eval_profile.tail_solve_enabled,
                            tail_solve_depth_limit=eval_profile.tail_solve_depth_limit,
                            tail_solve_node_budget=eval_profile.tail_solve_node_budget,
                            tail_solve_auxiliary_weight=eval_profile.tail_solve_auxiliary_weight,
                            belief_model_path=current_belief_model_path,
                            pool=worker_pool,
                            heuristic_selector=selector_eval,
                            concurrent_cfg=concurrent_eval_cfg,
                            safe_mode=safe_mode,
                            top_k=eval_profile.top_k,
                            fpu_reduction=eval_profile.fpu_reduction,
                            uct2_arrival_visits=eval_profile.uct2_arrival_visits)
                        eval_win_rates[bench] = r["win_rate"]
                        logger.info(
                            f"  eval vs heuristic (free): win_rate={r['win_rate']:.1%} "
                            f"(W={r['wins']}, L={r['losses']}, D={r['draws']})")
                    else:
                        bench_result = run_eval_batch(
                            game_id, eval_infer, bench,
                            eval_games, eval_seed_base + step * 100000 + 90000,
                            eval_profile.simulations, eval_profile.simulations,
                            eval_profile.world_cap, eval_profile.world_cap,
                            max_workers,
                            max_plies=game_max_plies,
                            c_puct=eval_profile.c_puct,
                            temperature=eval_profile.temperature,
                            opponent_selection=eval_profile.opponent_selection,
                            tail_solve_enabled=eval_profile.tail_solve_enabled,
                            tail_solve_depth_limit=eval_profile.tail_solve_depth_limit,
                            tail_solve_node_budget=eval_profile.tail_solve_node_budget,
                            tail_solve_auxiliary_weight=eval_profile.tail_solve_auxiliary_weight,
                            belief_model_path=current_belief_model_path,
                            pool=worker_pool,
                            concurrent_cfg=concurrent_eval_cfg,
                            safe_mode=safe_mode,
                            top_k=eval_profile.top_k,
                            fpu_reduction=eval_profile.fpu_reduction,
                            uct2_arrival_visits=eval_profile.uct2_arrival_visits,
                        )
                        eval_win_rates[bench] = bench_result["win_rate"]
                        logger.info(
                            f"  eval vs {Path(bench).stem}: win_rate={bench_result['win_rate']:.1%} "
                            f"(W={bench_result['wins']}, L={bench_result['losses']}, D={bench_result['draws']})")

                if (guidance_until is not None and guidance_trigger_step is None
                        and step >= heuristic_guidance_hold_steps
                        and eval_win_rates[guidance_until["benchmark"]] >= guidance_until["threshold"]):
                    guidance_trigger_step = step
                    logger.info(
                        f"  guidance: {guidance_until['benchmark']} win_rate "
                        f"{eval_win_rates[guidance_until['benchmark']]:.1%} >= "
                        f"{guidance_until['threshold']:.0%} at step {step}; decaying over "
                        f"{guidance_until['decay_steps']} steps")

                # Gating: latest vs best (always runs) — shares the eval profile
                # since gating and benchmark eval are the same MCTS behavior
                # (pair-wise tournament for win-rate).
                # Per-seat belief: candidate (latest) gets belief_latest,
                # opponent (best) gets belief_best. Falls back to latest when
                # belief_best doesn't exist yet (first gating pass).
                belief_best_path = str(models_dir / "belief_best.onnx")
                if not Path(belief_best_path).exists():
                    belief_best_path = current_belief_model_path
                gating_result = run_eval_batch(
                    game_id, eval_infer, inference_model(best_model_path),
                    eval_games, eval_seed_base + step * 100000 + 80000,
                    eval_profile.simulations, eval_profile.simulations,
                    eval_profile.world_cap, eval_profile.world_cap,
                    max_workers,
                    max_plies=game_max_plies,
                    c_puct=eval_profile.c_puct,
                    temperature=eval_profile.temperature,
                    opponent_selection=eval_profile.opponent_selection,
                    tail_solve_enabled=eval_profile.tail_solve_enabled,
                    tail_solve_depth_limit=eval_profile.tail_solve_depth_limit,
                    tail_solve_node_budget=eval_profile.tail_solve_node_budget,
                    tail_solve_auxiliary_weight=eval_profile.tail_solve_auxiliary_weight,
                    belief_model_path=current_belief_model_path,
                    belief_model_path_opponent=belief_best_path,
                    pool=worker_pool,
                    concurrent_cfg=concurrent_eval_cfg,
                    safe_mode=safe_mode,
                    top_k=eval_profile.top_k,
                    fpu_reduction=eval_profile.fpu_reduction,
                    uct2_arrival_visits=eval_profile.uct2_arrival_visits,
                )
                gating_wr = gating_result["win_rate"]
                logger.info(
                    f"  gating vs best: win_rate={gating_wr:.1%} "
                    f"(W={gating_result['wins']}, L={gating_result['losses']}, D={gating_result['draws']})")
                gating_threshold = train_cfg.get(
                    "gating_accept_win_rate",
                    _default_gating_threshold(num_players, eval_games,
                                              game_config.get("victory_mode", "winner_takes")))
                if gating_wr >= gating_threshold:
                    best_onnx = models_dir / "model_best.onnx"
                    with atomic_write(best_onnx) as tmp:
                        shutil.copy2(eval_model, tmp)
                    best_model_path = str(best_onnx)
                    if belief_enabled:
                        with atomic_write(models_dir / "belief_best.onnx") as tmp:
                            shutil.copy2(belief_latest, tmp)
                    logger.info(f"  new best model: step {step}, win_rate={gating_wr:.1%}")

            checkpoint = {
                "step": step,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "guidance_trigger_step": guidance_trigger_step,
            }
            if score_head is not None:
                checkpoint["score_head_state_dict"] = score_head.state_dict()
            if opp_head is not None:
                checkpoint["opponent_policy_head_state_dict"] = opp_head.state_dict()
            if ema_net is not None:
                checkpoint["ema_state_dict"] = ema_net.state_dict()
            # Rewritten every step and required by --resume: never write in place.
            with atomic_write(output_dir / "checkpoint.pt") as tmp:
                torch.save(checkpoint, tmp)
            if belief_enabled:
                belief_checkpoint = {
                    "step": step,
                    "model_state_dict": belief_net.state_dict(),
                    "optimizer_state_dict": belief_optimizer.state_dict(),
                }
                with atomic_write(output_dir / "belief_checkpoint.pt") as tmp:
                    torch.save(belief_checkpoint, tmp)
            if stop_signal:
                logger.info(f"Stop signal {stop_signal[0]} received: stopped after step {step} "
                            f"(checkpoint saved; --resume continues at step {step + 1})")
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        worker_pool.close()

    logger.info(f"Training complete. Best model: {best_model_path}")
