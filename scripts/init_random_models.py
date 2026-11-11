"""Random-init policy/value + belief ONNX models for a given game variant.

Used after an encoder dim change (e.g. Coup belief-net plan §16: feature_dim
141/188/235) to produce shipped placeholder ONNX files that load cleanly
through the AI / web stack so 2p/3p/4p rooms can open while real training
proceeds. No game playing strength implied.

Usage:
    python3 scripts/init_random_models.py --game coup
    python3 scripts/init_random_models.py --game coup --variants 2p,3p,4p
    python3 scripts/init_random_models.py --game coup --seed 42
    python3 scripts/init_random_models.py --game harmonies --variants tok --no-belief \
        --train-config games/harmonies/config/train_transformer.json

Files go where platform/model_paths.py looks (e.g. games/harmonies/model/harmonies_tok_2p.onnx).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine  # noqa: E402
from model_paths import expected_belief_model_path, expected_model_path  # noqa: E402
from training.game_ids import base_game_id as _base_game_id  # noqa: E402
from training.model import (  # noqa: E402
    create_belief_model_from_config,
    create_model_from_config,
    export_belief_onnx,
    export_onnx,
)


def _load_train_config(game_id: str, cfg_path: Path | None) -> dict:
    if cfg_path is None:
        cfg_path = PROJECT_ROOT / "games" / _base_game_id(game_id) / "config" / "train.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "belief" in cfg:
        raise ValueError(
            f"{cfg_path} contains a top-level 'belief' block — that block has moved "
            f"to games/<base>/config/belief.json. Move it there and remove from train.json.")
    return cfg


def _load_belief_config(game_id: str) -> dict | None:
    base = _base_game_id(game_id)
    cfg_path = PROJECT_ROOT / "games" / base / "config" / "belief.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _init_one(game_id: str, out_path: Path, belief_out_path: Path | None,
              seed: int, train_config: Path | None) -> None:
    meta = dinoboard_engine.game_metadata(game_id)
    train_cfg = _load_train_config(game_id, train_config)

    torch.manual_seed(seed)
    pv = create_model_from_config(
        train_cfg["network"],
        input_dim=meta["feature_dim"],
        policy_dim=meta["action_space"],
        num_players=meta["num_players"],
        token_layout=meta["token_layout"],
    )
    export_onnx(pv, out_path, meta["feature_dim"])
    print(f"  policy/value: {out_path} (input_dim={meta['feature_dim']}, "
          f"num_players={meta['num_players']})")

    if belief_out_path is not None and meta.get("has_belief_extractor"):
        belief_cfg = _load_belief_config(game_id)
        if belief_cfg is None:
            print(f"  [skip belief] {game_id}: no belief.json under config/")
            return
        in_dim = meta["belief_feature_dim"]
        out_dim = meta["belief_logit_count"]
        torch.manual_seed(seed + 1)
        bnet = create_belief_model_from_config(
            belief_cfg, input_dim=in_dim, output_dim=out_dim,
        )
        export_belief_onnx(bnet, belief_out_path, in_dim)
        print(f"  belief:       {belief_out_path} (input_dim={in_dim}, "
              f"logits={out_dim})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True,
                        help="Base game id, e.g. 'coup'")
    parser.add_argument("--variants", default="2p,3p,4p",
                        help="Comma-separated variants, e.g. '2p,3p,4p'. "
                             "If a variant has no registered game id "
                             "(e.g. tictactoe_3p) it is silently skipped.")
    parser.add_argument("--seed", type=int, default=0xDEADBEEF)
    parser.add_argument("--no-belief", action="store_true",
                        help="Skip belief.onnx export.")
    parser.add_argument("--train-config", type=Path,
                        help="Train config whose `network` to use (default "
                             "games/<game>/config/train.json), e.g. a "
                             "train_transformer.json for a _tok variant.")
    args = parser.parse_args()

    base = args.game
    available = set(dinoboard_engine.available_games())
    model_dir = PROJECT_ROOT / "games" / base / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    for v in args.variants.split(","):
        v = v.strip()
        # Try variant id first ('coup_3p'), fall back to base ('coup' for 2p).
        gid = f"{base}_{v}"
        if gid not in available:
            if v == "2p" and base in available:
                gid = base
            else:
                print(f"[skip] {gid} not registered")
                continue
        out = expected_model_path(gid)
        belief_out = None if args.no_belief else expected_belief_model_path(gid)
        print(f"[{gid}] →")
        _init_one(gid, out, belief_out, args.seed, args.train_config)


if __name__ == "__main__":
    main()
