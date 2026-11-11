"""Unified training CLI: python -m training.cli --game splendor --output runs/splendor_001"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .game_ids import base_game_id as _base_game_id
from .pipeline import atomic_write, run_training_loop
from .model import validate_belief_config


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Nested dicts are merged key-by-key; non-dict values in `override` replace
    the corresponding key in `base` wholesale (lists are not concatenated).
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_game_config(game_id: str) -> dict:
    """Load the game-level config (variants / max_game_plies).

    `game.json` is shared across variants. Per-variant dims live under
    `cfg["variants"][variant]`; cross-checked against engine.game_metadata.
    MCTS profiles are no longer here — they live in `mcts.json` and are
    read via `training.mcts_profile.resolve_profile`.
    Cross-validated dims are also injected at the top level for back-compat
    with any code still reading `cfg["feature_dim"]` etc.
    """
    root = Path(__file__).resolve().parents[1]
    base_id = _base_game_id(game_id)
    config_path = root / "games" / base_id / "config" / "game.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Game config not found at {config_path} (base id '{base_id}' from variant '{game_id}')")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "variants" not in cfg:
        raise KeyError(
            f"{config_path} is missing required 'variants' block. Each game.json "
            f"must declare per-variant dims (feature_dim/action_space/num_players "
            f"and belief_* if applicable) under variants.<variant>.")
    if game_id not in cfg["variants"]:
        raise KeyError(
            f"{config_path} 'variants' has no entry for '{game_id}'. "
            f"Known variants: {sorted(cfg['variants'].keys())}")
    variant_cfg = cfg["variants"][game_id]

    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(game_id)
    for k in ("feature_dim", "action_space", "num_players"):
        if variant_cfg[k] != meta[k]:
            raise ValueError(
                f"{config_path} variants.{game_id}.{k} = {variant_cfg[k]!r} "
                f"disagrees with engine.game_metadata({game_id!r})[{k!r}] = {meta[k]!r}. "
                f"game.json is the source of truth declaration; C++ side must agree.")
    if meta.get("has_belief_extractor"):
        for k in ("belief_feature_dim", "belief_logit_count"):
            if variant_cfg[k] != meta[k]:
                raise ValueError(
                    f"{config_path} variants.{game_id}.{k} = {variant_cfg[k]!r} "
                    f"disagrees with engine.game_metadata({game_id!r})[{k!r}] = {meta[k]!r}.")

    cfg["feature_dim"] = variant_cfg["feature_dim"]
    cfg["action_space"] = variant_cfg["action_space"]
    cfg["num_players"] = variant_cfg["num_players"]
    return cfg


def validate_guidance_until(gs: dict, path) -> None:
    """heuristic_guidance.until_win_rate: hold initial_ratio until the eval
    benchmark win rate reaches threshold (not before hold_steps, at the latest
    after max_hold_steps), then decay linearly over decay_steps. The trigger
    owns the decay, so `steps` must be 0."""
    until = gs.get("until_win_rate")
    if until is None:
        return
    keys = {"benchmark", "threshold", "decay_steps", "max_hold_steps"}
    if not isinstance(until, dict) or set(until) != keys:
        raise ValueError(
            f"{path} heuristic_guidance.until_win_rate must have exactly the keys "
            f"{sorted(keys)}, got {until!r}.")
    if gs["steps"] != 0:
        raise ValueError(
            f"{path} heuristic_guidance.steps must be 0 with until_win_rate "
            f"(the win-rate trigger starts the decay), got {gs['steps']}.")
    if not 0.0 < float(until["threshold"]) <= 1.0:
        raise ValueError(f"{path} until_win_rate.threshold must be in (0, 1], got {until['threshold']}.")
    for k in ("decay_steps", "max_hold_steps"):
        if not isinstance(until[k], int) or isinstance(until[k], bool) or until[k] < 1:
            raise ValueError(f"{path} until_win_rate.{k} must be int >= 1, got {until[k]!r}.")
    if until["max_hold_steps"] < gs["hold_steps"]:
        raise ValueError(
            f"{path} until_win_rate.max_hold_steps ({until['max_hold_steps']}) must be "
            f">= hold_steps ({gs['hold_steps']}).")


def find_train_config(game_id: str) -> tuple[dict, Path]:
    """Load the per-game training recipe (network + training).

    `train.json` is shared across all variants of a game (only dims differ,
    and dims live in game.json). The file is mandatory; missing → raise.
    Belief network config lives in a separate `belief.json` (only present
    for games that train one) — see `find_belief_config`.
    """
    root = Path(__file__).resolve().parents[1]
    base_id = _base_game_id(game_id)
    path = root / "games" / base_id / "config" / "train.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Train config not found at {path} (base id '{base_id}' from variant '{game_id}'). "
            f"Every game must ship a games/<base>/config/train.json.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "belief" in cfg:
        raise ValueError(
            f"{path} contains a top-level 'belief' block — that block has moved to "
            f"games/<base>/config/belief.json. Move it there and remove from train.json.")
    forbidden = (
        "heuristic_guidance_hold_steps",
        "heuristic_guidance_steps",
        "heuristic_guidance_initial_ratio",
        "heuristic_guidance_ratio",
        "heuristic_guidance_temperature",
        "heuristic_temperature",
        "free_heuristic_temperature",
    )
    training_block = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    present = [k for k in forbidden if k in training_block]
    if present:
        raise ValueError(
            f"{path}: keys {present} under training have moved to "
            f"games/{base_id}/config/train.json::heuristic_guidance "
            f"(top-level) and games/{base_id}/config/heuristic.json::"
            f"selectors. Remove them from training.")
    if "heuristic_guidance" not in cfg:
        raise KeyError(
            f"{path} is missing top-level 'heuristic_guidance' block. "
            f"Schema: {{'name': str, 'hold_steps': int, 'steps': int, "
            f"'initial_ratio': float, 'temperature': float}}.")
    gs = cfg["heuristic_guidance"]
    for k in ("name", "hold_steps", "steps", "initial_ratio", "temperature"):
        if k not in gs:
            raise KeyError(
                f"{path} heuristic_guidance is missing required field "
                f"'{k}'. Required fields: name, hold_steps, steps, "
                f"initial_ratio, temperature.")
    validate_guidance_until(gs, path)
    # Cross-check the named heuristic exists in heuristic.json.
    h_cfg, h_path = find_heuristic_config(game_id)
    if gs["name"] not in h_cfg["named"]:
        raise KeyError(
            f"{path} heuristic_guidance.name = {gs['name']!r} is not "
            f"present in {h_path}'s 'named'. Known: "
            f"{sorted(h_cfg['named'].keys())}.")

    # Optional: selfplay_concurrency block (Phase 1 batched / threaded path).
    # Plan v0.7 §1.5: missing block ≡ enabled=false → old multi-process path.
    # When present, validate per §4.3 safety rails.
    sc = cfg.get("training", {}).get("selfplay_concurrency")
    if sc is not None:
        if not isinstance(sc, dict):
            raise ValueError(
                f"{path} training.selfplay_concurrency must be an object")
        if "enabled" not in sc:
            raise KeyError(
                f"{path} training.selfplay_concurrency missing 'enabled' "
                f"(must be bool; false ≡ block absent ≡ legacy path).")
        enabled = bool(sc["enabled"])
        if not enabled:
            # Rail 1: enabled=false → no other keys allowed (prevents
            # 'thought I changed it but it's silently inactive' confusion).
            extra = sorted(set(sc.keys()) - {"enabled"})
            if extra:
                raise ValueError(
                    f"{path} training.selfplay_concurrency.enabled=false but "
                    f"other keys present: {extra}. Either set enabled=true to "
                    f"activate them, or remove them so the block reads as "
                    f"pure legacy path.")
        else:
            # When enabled, validate the few knobs we expose.
            opp_distinct = sc.get("opponent_pool_max_distinct", 3)
            if not isinstance(opp_distinct, int) or opp_distinct < 1:
                raise ValueError(
                    f"{path} selfplay_concurrency.opponent_pool_max_distinct "
                    f"must be int >= 1, got {opp_distinct}.")
            ev = sc.get("evaluator", {})
            if not isinstance(ev, dict):
                raise ValueError(
                    f"{path} selfplay_concurrency.evaluator must be an object")
            for k, default in (("batch_size", 32),
                                ("batch_timeout_us", 2000),
                                ("max_queue_size", 256),
                                ("intra_op_num_threads", 2)):
                v = ev.get(k, default)
                if not isinstance(v, int) or v < 1:
                    raise ValueError(
                        f"{path} selfplay_concurrency.evaluator.{k} must be "
                        f"int >= 1, got {v!r}.")
            if ev.get("max_queue_size", 256) < ev.get("batch_size", 32):
                raise ValueError(
                    f"{path} selfplay_concurrency.evaluator: "
                    f"max_queue_size must be >= batch_size.")
            for where, block, key in (
                    ("selfplay_concurrency", sc, "leaf_batch_size"),
                    ("selfplay_concurrency.evaluator", ev, "num_inference_threads")):
                if key not in block:
                    raise KeyError(f"{path} {where} missing '{key}' (int >= 1; 1 = previous behavior).")
                v = block[key]
                if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                    raise ValueError(f"{path} {where}.{key} must be int >= 1, got {v!r}.")
            devices = ev.get("devices", [])
            if (not isinstance(devices, list)
                    or any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in devices)
                    or len(set(devices)) != len(devices)):
                raise ValueError(
                    f"{path} selfplay_concurrency.evaluator.devices must be a list of "
                    f"distinct GPU ids >= 0, got {devices!r}.")
            if devices and ev["num_inference_threads"] % len(devices):
                raise ValueError(
                    f"{path} selfplay_concurrency.evaluator.num_inference_threads "
                    f"({ev['num_inference_threads']}) must be a multiple of the "
                    f"{len(devices)} devices.")
    return cfg, path


def find_heuristic_config(game_id: str) -> tuple[dict, Path]:
    """Load the per-game heuristic config (named pool + selectors).

    `heuristic.json` is shared across all variants of a game. Mandatory:
    every game must declare at least one named heuristic and the two
    selectors (`eval_vs_heuristic`, `web`). The training-time guidance
    schedule lives in `train.json::heuristic_guidance`, validated against
    this file's `named` pool by `find_train_config`.
    """
    root = Path(__file__).resolve().parents[1]
    base_id = _base_game_id(game_id)
    path = root / "games" / base_id / "config" / "heuristic.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Heuristic config not found at {path} (base id '{base_id}' "
            f"from variant '{game_id}'). Every game must ship a "
            f"games/<base>/config/heuristic.json.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "guidance_schedule" in cfg:
        raise KeyError(
            f"{path} contains a top-level 'guidance_schedule' block — that "
            f"block has moved to games/{base_id}/config/train.json::"
            f"heuristic_guidance. Move it there and remove from heuristic.json.")
    for key in ("named", "selectors"):
        if key not in cfg:
            raise KeyError(
                f"{path} is missing required top-level key '{key}'. "
                f"Schema: {{'named': {{...}}, 'selectors': {{...}}}}.")
    for sel in ("eval_vs_heuristic", "web"):
        if sel not in cfg["selectors"]:
            raise KeyError(
                f"{path} selectors block is missing '{sel}'. Both "
                f"eval_vs_heuristic and web are required.")
        sel_cfg = cfg["selectors"][sel]
        if "name" not in sel_cfg:
            raise KeyError(
                f"{path} selectors.{sel} must have a 'name' field.")
        if sel_cfg["name"] not in cfg["named"]:
            raise KeyError(
                f"{path} selectors.{sel}.name = {sel_cfg['name']!r} is not "
                f"present in 'named'. Known: {sorted(cfg['named'].keys())}.")
    for name, node in cfg["named"].items():
        if not isinstance(node, dict):
            raise ValueError(
                f"{path} named.{name} must be an object, got "
                f"{type(node).__name__}.")
        h_type = node.get("type")
        if h_type == "rule":
            extra = sorted(set(node.keys()) - {"type"})
            if extra:
                raise ValueError(
                    f"{path} named.{name} type='rule' has unsupported fields "
                    f"{extra}. Rule heuristics are C++-registered by name.")
        elif h_type == "mcts_pv":
            allowed = {"type", "simulations", "c_puct", "model_path", "world_cap",
                       "leaf_batch_size"}
            extra = sorted(set(node.keys()) - allowed)
            if extra:
                raise ValueError(
                    f"{path} named.{name} type='mcts_pv' has unsupported "
                    f"fields {extra}. Allowed: {sorted(allowed)}.")
            for req in ("simulations", "c_puct"):
                if req not in node:
                    raise KeyError(
                        f"{path} named.{name} type='mcts_pv' missing "
                        f"required field {req!r}.")
            if "leaf_batch_size" in node:
                lb = node["leaf_batch_size"]
                if not isinstance(lb, int) or isinstance(lb, bool) or lb < 1:
                    raise ValueError(
                        f"{path} named.{name}.leaf_batch_size must be int >= 1, "
                        f"got {lb!r}.")
        else:
            raise ValueError(
                f"{path} named.{name}.type must be 'rule' or 'mcts_pv', "
                f"got {h_type!r}.")
    return cfg, path


def find_belief_config(game_id: str) -> tuple[dict | None, Path | None]:
    """Load the per-game belief network recipe, if the game ships one.

    Returns (cfg, path) when `games/<base>/config/belief.json` exists, else
    (None, None). Caller decides whether to enable the belief net based on
    engine capability (`game_metadata(...)["has_belief_extractor"]`) and
    whether this returns non-None.

    Belief training schedule (batch_size / batches_per_step / lr / weight_decay)
    is intentionally NOT configurable here — it always mirrors the PV training
    schedule so the two networks can't quietly fork. Belief.json carries only
    the belief-specific knobs: network shape and loss kind.
    """
    root = Path(__file__).resolve().parents[1]
    base_id = _base_game_id(game_id)
    path = root / "games" / base_id / "config" / "belief.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    validate_belief_config(cfg, where=str(path))

    # Forbid PV-side training knobs in belief.json. These have moved to
    # train.json::training and are inherited by the belief loop verbatim;
    # writing them in belief.json was the entry point for accidental
    # PV/belief schedule divergence.
    forbidden_in_training = {
        "batch_size", "steps_per_pv_step", "lr_schedule",
        "lr", "learning_rate", "weight_decay",
    }
    b_train = cfg.get("training", {})
    bad = forbidden_in_training & set(b_train.keys())
    if bad:
        raise ValueError(
            f"{path}: belief.json::training may not contain {sorted(bad)}. "
            f"These knobs are inherited from train.json::training so PV and "
            f"belief always train on the same schedule. Remove them.")

    return cfg, path


def main() -> int:
    parser = argparse.ArgumentParser(description="DinoBoard training CLI")
    parser.add_argument("--game", type=str, required=True, help="Game ID / variant (e.g., liarsdice, liarsdice_3p)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=0, help="Override training steps (0 = use config)")
    parser.add_argument("--episodes", type=int, default=0, help="Override episodes per step (0 = use config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (override config)")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated ONNX execution providers (tried in "
                             "order) for concurrent selfplay/eval, e.g. 'cuda,cpu' "
                             "on NVIDIA or 'coreml,cpu' on Apple. Overrides "
                             "selfplay_concurrency.evaluator.providers at runtime so "
                             "train.json stays machine-agnostic. Omitted = use the "
                             "json value (default: CPU).")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated GPU ids for the cuda/tensorrt providers, "
                             "e.g. '0,1,2'. Inference thread i runs on device i mod n, so "
                             "num_inference_threads must be a multiple of n. Overrides "
                             "selfplay_concurrency.evaluator.devices at runtime. Omitted = "
                             "use the json value (default: device 0).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (override config)")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N steps (0 = disable; override config)")
    parser.add_argument("--save-every", type=int, default=None, help="Save model checkpoint every N steps (0 = same as eval-every; override config)")
    parser.add_argument("--eval-games", type=int, default=None, help="Number of evaluation games (override config)")
    parser.add_argument("--batch-size", type=int, default=0, help="Training batch size (0 = use config)")
    parser.add_argument("--lr", type=float, default=0.0, help="Learning rate (0 = use config)")
    parser.add_argument("--eval-benchmark", type=str, nargs="+", default=None,
                        help='Benchmarks: heuristic_constrained, heuristic_free, or ONNX path')
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to a .pt checkpoint to initialize weights from. "
                             "Optimizer state is reset (Adam moments do not survive "
                             "architecture changes). The checkpoint must be a dict with "
                             "'model_state_dict' whose keys / shapes match the configured "
                             "PVNet.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume an interrupted run from its output directory. "
                             "Reads <dir>/checkpoint.pt for PV weights+optimizer+step "
                             "and <dir>/belief_checkpoint.pt for belief (if present). "
                             "Falls back to scanning <dir>/models/model_step_*.onnx for "
                             "the highest step when checkpoint.pt is missing. The new "
                             "run continues at step=last+1 with the same numbering, "
                             "appends to train.log, and re-uses the existing models/ "
                             "directory as the opponent pool. Mutually exclusive with "
                             "--init-from.")
    parser.add_argument("--train-config", type=str, default=None,
                        help="Path to a train.json that overrides games/<base>/config/train.json. "
                             "Use this to launch a run with a different network architecture / "
                             "training hyperparameters without editing the game's default recipe.")
    parser.add_argument("--config-override", type=str, default=None,
                        help="Path to a JSON file whose contents are deep-merged onto the "
                             "loaded train.json (overrides any field, including nested ones "
                             "like training.lr_schedule).")
    parser.add_argument("--belief-config", type=str, default=None,
                        help="Path to a belief.json that overrides games/<base>/config/belief.json. "
                             "Only meaningful for games whose engine binding registers a belief "
                             "extractor (e.g. Coup, LiarsDice).")
    parser.add_argument("--safe-mode", action="store_true", default=False,
                        help="Enable MCTS safe mode: validate DAG edges at every "
                             "descent step. Slower but catches hash/schema bugs.")
    args = parser.parse_args()

    import faulthandler
    faulthandler.enable()

    if args.resume and args.init_from:
        raise SystemExit("--resume and --init-from are mutually exclusive")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise SystemExit(f"--resume path does not exist: {resume_dir}")
        if Path(args.output) != resume_dir:
            raise SystemExit(
                f"--resume requires --output to point at the same directory "
                f"({resume_dir}); got --output={args.output}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format=log_fmt, datefmt=log_datefmt)
    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt=log_datefmt))
    logging.getLogger().addHandler(fh)

    game_config = find_game_config(args.game)

    if args.train_config:
        train_path = Path(args.train_config)
        if not train_path.exists():
            raise SystemExit(f"--train-config path does not exist: {train_path}")
        with open(train_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        if "belief" in train_config:
            raise SystemExit(
                f"--train-config {train_path} contains a top-level 'belief' block — "
                f"belief config has moved to belief.json. Pass it via --belief-config "
                f"or place it next to the train.json under config/belief.json.")
        if "mcts" in train_config:
            base_cfg, base_path = find_train_config(args.game)
            if train_config["mcts"] != base_cfg["mcts"]:
                raise SystemExit(
                    f"--train-config {train_path} has an 'mcts' block that differs from "
                    f"{base_path}. MCTS profiles are always resolved from the base "
                    f"train.json (training.mcts_profile.resolve_profile); drop the block "
                    f"from {train_path.name} or edit {base_path} instead.")
        logging.getLogger().info(f"Loaded train.json from {train_path}")
    else:
        train_config, train_path = find_train_config(args.game)
        logging.getLogger().info(f"Loaded train.json from {train_path}")

    if args.config_override:
        override_path = Path(args.config_override)
        if not override_path.exists():
            raise SystemExit(f"--config-override path does not exist: {override_path}")
        with open(override_path, "r", encoding="utf-8") as f:
            override = json.load(f)
        if not isinstance(override, dict):
            raise SystemExit(f"--config-override JSON must be an object at top level: {override_path}")
        train_config = deep_merge(train_config, override)
        logging.getLogger().info(f"Applied config override from {override_path}")
    validate_guidance_until(train_config["heuristic_guidance"], train_path)

    # Belief network config (optional — only for games that train one).
    if args.belief_config:
        belief_path = Path(args.belief_config)
        if not belief_path.exists():
            raise SystemExit(f"--belief-config path does not exist: {belief_path}")
        with open(belief_path, "r", encoding="utf-8") as f:
            belief_cfg = json.load(f)
        validate_belief_config(belief_cfg, where=str(belief_path))
        logging.getLogger().info(f"Loaded belief.json from {belief_path}")
    else:
        belief_cfg, belief_path = find_belief_config(args.game)
        if belief_cfg is not None:
            logging.getLogger().info(f"Loaded belief.json from {belief_path}")
    if belief_cfg is not None:
        # pipeline.py reads train_config["belief"] — keep that contract by
        # injecting after load so existing readers don't need to change.
        train_config["belief"] = belief_cfg

    heuristic_cfg, heuristic_path = find_heuristic_config(args.game)
    logging.getLogger().info(f"Loaded heuristic.json from {heuristic_path}")
    train_config["heuristic"] = heuristic_cfg

    # Persist the effective train.json into the run dir so resume / fine-tune
    # paths can rebuild the architecture without ambiguity. Belief and
    # heuristic blocks are written separately, mirroring the source layout.
    effective_train = {
        k: v for k, v in train_config.items() if k not in ("belief", "heuristic")
    }
    # Written atomically: a resumed run may be launched with --train-config /
    # --belief-config pointing at these very files, which are rewritten here.
    to_persist = [("train.json", effective_train), ("heuristic.json", heuristic_cfg)]
    if belief_cfg is not None:
        to_persist.append(("belief.json", belief_cfg))
    for name, content in to_persist:
        with atomic_write(output_dir / name) as tmp:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)

    train_cfg = train_config["training"]

    # --providers overrides the ONNX execution providers for the concurrent
    # selfplay/eval path at runtime. Applied AFTER persisting effective
    # train.json above, so the on-disk copy stays portable (no hardcoded
    # "cuda" that would break a CPU-only machine on resume); each launch
    # re-supplies the EPs via CLI, falling back to the json value (default CPU)
    # when omitted. Both selfplay (sc_cfg.evaluator) and the batched eval path
    # read this same dict, so one injection covers both.
    if args.providers is not None:
        prov = [p.strip() for p in args.providers.split(",") if p.strip()]
        sc = train_cfg.setdefault("selfplay_concurrency", {})
        ev = sc.setdefault("evaluator", {})
        ev["providers"] = prov
        logging.getLogger().info(
            f"selfplay_concurrency.evaluator.providers <- CLI --providers {prov}")
    if args.devices is not None:
        devices = [int(d) for d in args.devices.split(",") if d.strip()]
        ev = train_cfg.setdefault("selfplay_concurrency", {}).setdefault("evaluator", {})
        ev["devices"] = devices
        logging.getLogger().info(
            f"selfplay_concurrency.evaluator.devices <- CLI --devices {devices}")

    steps = args.steps if args.steps > 0 else train_cfg["steps"]
    episodes = args.episodes if args.episodes > 0 else train_cfg["episodes_per_step"]
    batch_size = args.batch_size if args.batch_size > 0 else train_cfg["batch_size"]
    lr = args.lr if args.lr > 0 else train_cfg["learning_rate"]
    eval_benchmarks = args.eval_benchmark if args.eval_benchmark else train_cfg.get("eval_benchmarks")

    runtime_cfg = train_config["runtime"]
    workers = args.workers if args.workers is not None else runtime_cfg.get("workers", 4)
    seed = args.seed if args.seed is not None else runtime_cfg.get("seed", 20260323)
    eval_every = args.eval_every if args.eval_every is not None else runtime_cfg["eval_every"]
    save_every = args.save_every if args.save_every is not None else runtime_cfg["save_every"]
    eval_games = args.eval_games if args.eval_games is not None else runtime_cfg["eval_games"]

    run_training_loop(
        game_id=args.game,
        game_config=game_config,
        train_config=train_config,
        output_dir=Path(args.output),
        steps=steps,
        episodes_per_step=episodes,
        eval_every=eval_every,
        eval_games=eval_games,
        eval_benchmarks=eval_benchmarks,
        max_workers=workers,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        save_every=save_every,
        init_from=args.init_from,
        resume_from=args.resume,
        safe_mode=args.safe_mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
