"""Evaluate trained models / heuristics: run arena matches, report win rates, save replays.

**Defaults match the strength a web user faces**: model side uses the
`web_expert` MCTS profile; the bare `heuristic` actor spec resolves to the
`web` selector's `name` and `temperature` (the game's web-hint baseline). Use
`--profile eval` + `--heuristic-selector eval_vs_heuristic` to mirror the
training pipeline's vs-heuristic benchmark instead.

Both `--model` and `--opponent` accept three forms:
  - an ONNX path                     → MCTS-driven model with the chosen --profile
  - "heuristic"                      → the named heuristic + temperature referenced
                                       by heuristic.json::selectors[<--heuristic-selector>]
                                       (default 'web')
  - "heuristic:<name>"               → a specific named heuristic from the
                                       game's heuristic.json::named pool
                                       (e.g. "heuristic:lkf_v1", "heuristic:rule_v1")

Three combinations are supported:
  - model vs model       → run_arena_match (per-seat MCTS configs)
  - model vs heuristic   → run_constrained_eval_vs_heuristic (model on test seat,
                           named heuristic on opponent seat — or the reverse,
                           when --model is the heuristic side)
  - heuristic vs heuristic → run_heuristic_vs_heuristic (per-seat heuristic name)

Number of seats is resolved from the game_id (`game_metadata(game).num_players`).
The "model under test" rotates across all seats via `game_idx % num_players`.

Usage:
  # 2p model-vs-default-eval-heuristic
  python scripts/eval_model.py --game quoridor \\
    --model runs/quoridor_v14/models/model_best.onnx \\
    --opponent heuristic --games 40 --workers 4 --no-save -o /tmp/eval

  # 2p model-vs-specific-heuristic
  python scripts/eval_model.py --game 6nimmit_2p \\
    --model runs/sixnimmit/models/model_best.onnx \\
    --opponent heuristic:lkf_v1 --games 40 --no-save -o /tmp/eval

  # 2p heuristic-vs-heuristic (no model needed, just compare heuristics)
  python scripts/eval_model.py --game 6nimmit_2p \\
    --model heuristic:lkf_v1 --opponent heuristic:rule_v1 \\
    --games 40 --no-save -o /tmp/eval

  # 2p model-vs-model
  python scripts/eval_model.py --game quoridor \\
    --model new.onnx --opponent old.onnx --games 40 -o replays/v14
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "platform"))
from game_service.replay import build_replay_dict
from training.cli import find_heuristic_config, find_train_config
from training.mcts_profile import max_game_plies, resolve_profile
from training.pipeline import outcome_label

_HEURISTIC = "heuristic"
_HEURISTIC_PREFIX = "heuristic:"

# Above this many --workers, the legacy per-process ONNX path is an OOM
# hazard: ProcessPoolExecutor forks one worker per --workers and each worker
# loads its OWN evaluator(s) (model + belief, plus opponent for m_vs_m), so
# the model footprint multiplies by --workers. The --batched path shares one
# evaluator pool across N threads, so it scales safely. h_vs_h has no ONNX
# and is exempt from this guard.
_SAFE_PROCESS_WORKERS = 16


def _parse_providers(raw: str | None) -> list[str] | None:
    """Parse CLI provider override.

    None means "not specified, use train.json". An empty string means
    "explicitly CPU/default provider only".
    """
    if raw is None:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def _default_eval_batch_config(game_id: str) -> dict | None:
    train_cfg, train_path = find_train_config(game_id)
    if "training" not in train_cfg:
        raise KeyError(f"{train_path} is missing required top-level 'training'")
    training_cfg = train_cfg["training"]
    if "selfplay_concurrency" not in training_cfg:
        return None
    sc_cfg = training_cfg["selfplay_concurrency"]
    if "enabled" not in sc_cfg:
        raise KeyError(
            f"{train_path} training.selfplay_concurrency missing 'enabled'")
    if not bool(sc_cfg["enabled"]):
        return None
    if "evaluator" not in sc_cfg:
        raise KeyError(
            f"{train_path} training.selfplay_concurrency missing 'evaluator'")
    ev_cfg = sc_cfg["evaluator"]
    for key in ("batch_size", "batch_timeout_us", "max_queue_size",
                "intra_op_num_threads"):
        if key not in ev_cfg:
            raise KeyError(
                f"{train_path} training.selfplay_concurrency.evaluator "
                f"missing '{key}'")
    if "providers" not in ev_cfg:
        raise KeyError(
            f"{train_path} training.selfplay_concurrency.evaluator "
            f"missing 'providers'")
    providers = ev_cfg["providers"]
    if not isinstance(providers, list):
        raise ValueError(
            f"{train_path} training.selfplay_concurrency.evaluator.providers "
            f"must be a list, got {type(providers).__name__}")
    return {
        "batch_size": int(ev_cfg["batch_size"]),
        "batch_timeout_us": int(ev_cfg["batch_timeout_us"]),
        "max_queue_size": int(ev_cfg["max_queue_size"]),
        "intra_op_num_threads": int(ev_cfg["intra_op_num_threads"]),
        "providers": list(providers),
        "source": str(train_path),
    }


def _resolve_eval_batch_config(args, mode: str) -> dict | None:
    if mode not in ("m_vs_m", "m_vs_h"):
        return None
    if args.no_batched:
        return None

    cfg = _default_eval_batch_config(args.game)
    cli_requests_batch = (
        args.batched or args.providers is not None or
        args.batch_size is not None or args.batch_timeout_us is not None or
        args.max_queue_size is not None or
        args.intra_op_num_threads is not None
    )
    if cfg is None:
        if not cli_requests_batch:
            return None
        cfg = {
            "batch_size": 32,
            "batch_timeout_us": 50,
            "max_queue_size": 256,
            "intra_op_num_threads": 2,
            "providers": [],
            "source": "CLI defaults",
        }

    providers_override = _parse_providers(args.providers)
    if providers_override is not None:
        cfg["providers"] = providers_override
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.batch_timeout_us is not None:
        cfg["batch_timeout_us"] = args.batch_timeout_us
    if args.max_queue_size is not None:
        cfg["max_queue_size"] = args.max_queue_size
    if args.intra_op_num_threads is not None:
        cfg["intra_op_num_threads"] = args.intra_op_num_threads

    if cfg["batch_size"] < 1:
        raise SystemExit("--batch-size must be >= 1")
    if cfg["batch_timeout_us"] < 1:
        raise SystemExit("--batch-timeout-us must be >= 1")
    if cfg["max_queue_size"] < cfg["batch_size"]:
        raise SystemExit("--max-queue-size must be >= --batch-size")
    if cfg["intra_op_num_threads"] < 1:
        raise SystemExit("--intra-op-num-threads must be >= 1")
    return cfg


def _parse_actor(spec: str, default_heuristic_name: str) -> tuple[str, str]:
    """Parse an actor spec into ('heuristic', name) or ('model', path).

    'heuristic' alone resolves to default_heuristic_name (eval_vs_heuristic
    selector). 'heuristic:<name>' resolves to that explicit named heuristic.
    Anything else is treated as a model path.
    """
    if spec == _HEURISTIC:
        return ("heuristic", default_heuristic_name)
    if spec.startswith(_HEURISTIC_PREFIX):
        name = spec[len(_HEURISTIC_PREFIX):]
        if not name:
            raise SystemExit(
                f"Invalid actor spec '{spec}': expected 'heuristic:<name>' "
                f"with a non-empty name.")
        return ("heuristic", name)
    return ("model", spec)


def test_side_outcome(result: dict, seat: int) -> str:
    """'win' / 'loss' / 'draw' of the seat under test.

    Finished game (zero-sum terminal values): the sign of that seat's own
    value, so a shared first place among 3-4 players is a win and finishing
    behind tied leaders a loss (`winner` is -1 in both cases and would read as
    a draw). Game cut off at the ply cap (the engine reports -1 for every
    seat, or no values): the adjudicator's verdict in `winner` / `draw`, as
    before. 2-player results are labelled exactly as they always were."""
    z = result["terminal_z_values"]
    if z and abs(sum(z)) < 1e-6:
        return outcome_label(z[seat])
    if result["draw"]:
        return "draw"
    return "win" if result["winner"] == seat else "loss"


def _run_one_game(task: dict) -> dict:
    import dinoboard_engine

    mode = task["mode"]
    if mode == "h_vs_h":
        # Both seats are heuristics. Per-seat names + per-seat temperatures.
        result = dinoboard_engine.run_heuristic_vs_heuristic(
            game_id=task["game"],
            seed=task["seed"],
            max_game_plies=task["max_plies"],
            heuristic_names=task["heuristic_names"],
            temperatures=task["heuristic_temps"])
    elif mode == "m_vs_h":
        # Exactly one seat runs MCTS over an ONNX model; the other(s) play a
        # named heuristic. The binding parameterizes "which seat is the model".
        result = dinoboard_engine.run_constrained_eval_vs_heuristic(
            game_id=task["game"], seed=task["seed"],
            model_path=task["model_path"],
            simulations=task["sims"],
            world_cap=task["world_cap"],
            c_puct=task["c_puct"],
            temperature=task["temp"],
            model_is_player=task["model_seat_in_engine"],
            constrained=False,
            heuristic_temperature=task["heuristic_temp"],
            opponent_selection=task["opp_sel"],
            tail_solve_enabled=task["model_tail_solve_enabled"],
            tail_solve_depth_limit=task["tail_solve_depth_limit"],
            tail_solve_node_budget=task["tail_solve_node_budget"],
            tail_solve_auxiliary_weight=task["tail_solve_auxiliary_weight"],
            belief_model_path=task.get("belief_path", ""),
            heuristic_selector=task["heuristic_selector"],
            safe_mode=task.get("safe_mode", False),
            top_k=task["top_k"],
            fpu_reduction=task["fpu_reduction"],
            uct2_arrival_visits=task["uct2_arrival_visits"])
    else:  # m_vs_m
        n = task["num_players"]
        def _seat(model_path: str, ts_enabled: bool) -> dict:
            return {
                "model_path": model_path,
                "simulations": task["sims"],
                "world_cap": task["world_cap"],
                "c_puct": task["c_puct"],
                "fpu_reduction": task["fpu_reduction"],
                "uct2_arrival_visits": task["uct2_arrival_visits"],
                "temperature": task["temp"],
                "opponent_selection": task["opp_sel"],
                "tail_solve_enabled": ts_enabled,
                "tail_solve_depth_limit": task["tail_solve_depth_limit"],
                "tail_solve_node_budget": task["tail_solve_node_budget"],
                "tail_solve_auxiliary_weight": task["tail_solve_auxiliary_weight"],
                "top_k": task["top_k"],
            }
        seat_configs = [_seat(task["opponent_path"], task["opponent_tail_solve_enabled"]) for _ in range(n)]
        seat_configs[task["model_seat"]] = _seat(task["model_path"], task["model_tail_solve_enabled"])
        result = dinoboard_engine.run_arena_match(
            game_id=task["game"], seed=task["seed"],
            max_game_plies=task["max_plies"],
            belief_model_path=task.get("belief_path", ""),
            seat_configs=seat_configs,
            safe_mode=task.get("safe_mode", False))

    return {
        "game_idx": task["game_idx"],
        "seed": task["seed"],
        "model_seat": task["model_seat"],
        "winner": result["winner"],
        "draw": result["draw"],
        "terminal_z_values": list(result["terminal_z_values"]),
        "total_plies": result["total_plies"],
        "action_history": list(result["action_history"]),
    }


def main():
    p = argparse.ArgumentParser(description="Run matches and save replay JSON")
    p.add_argument("--game", default="quoridor",
                   help="Game id (e.g. 'quoridor', 'loveletter_4p', 'azul_3p').")
    p.add_argument("--model", required=True,
                   help="Test side: ONNX path, 'heuristic', or 'heuristic:<name>'. "
                        "Rotates across all seats via game_idx %% num_players.")
    p.add_argument("--opponent", default=_HEURISTIC,
                   help="Opponent for non-test seats: ONNX path, 'heuristic', or "
                        "'heuristic:<name>'.")
    p.add_argument("--name-model", default=None, help="Display name for the test side")
    p.add_argument("--name-opponent", default=None, help="Display name for opponent")
    p.add_argument("--profile", default="web_expert",
                   help="MCTS profile name for model seats (default 'web_expert'; "
                        "= the strength a web user faces). Other valid values: "
                        "'eval' (gating + benchmark, lower sims), 'web_casual', "
                        "'analysis', 'selfplay'.")
    p.add_argument("--heuristic-selector", default="web",
                   choices=["web", "eval_vs_heuristic"],
                   help="Which heuristic.json::selectors entry drives the default "
                        "heuristic name and temperature for the 'heuristic' actor "
                        "spec (default 'web'). Use 'eval_vs_heuristic' to match "
                        "the training pipeline's vs-heuristic benchmark.")
    p.add_argument("--heuristic-temp", type=float, default=None,
                   help="Override heuristic temperature (default: read from the "
                        "chosen --heuristic-selector entry).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seed-stride", type=int, default=1,
                   help="Seed increment between game seeds. With "
                        "--paired-seats the stride applies per paired seed, "
                        "so one seed block can sample non-adjacent deals.")
    p.add_argument("--games", type=int, default=1, help="Number of games (rotates seat).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p.add_argument("--output", "-o", required=True, help="Output dir or file path")
    p.add_argument("--no-save", action="store_true", help="Only print stats, skip saving replay JSON")
    p.add_argument("--sims", type=int, default=None,
                   help="Override profile simulations count (both model seats)")
    p.add_argument("--world-cap", type=int, default=None,
                   help="Override profile world_cap (both model seats)")
    p.add_argument("--c-puct", type=float, default=None,
                   help="Override profile c_puct (both model seats)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override profile temperature for action selection "
                        "(both model seats; heuristic temperature is separate, "
                        "see --heuristic-temp)")
    p.add_argument("--belief", default=None,
                   help="Override belief-net ONNX path. Empty default lets the "
                        "session factory resolve games/<base>/model/<base>_belief_<N>p.onnx.")
    p.add_argument("--model-tail-solve", choices=["on", "off"], default=None,
                   help="Override tail_solve for the test model seat (default: profile)")
    p.add_argument("--opponent-tail-solve", choices=["on", "off"], default=None,
                   help="Override tail_solve for the opponent seat (default: profile)")
    p.add_argument("--uct2-arrival-visits", choices=["on", "off"], default=None,
                   help="Override the profile's uct2_arrival_visits for all model "
                        "seats (default: profile; only train.json's selfplay / "
                        "eval profiles can set it, so web profiles are always on). "
                        "'off' = the UCT2 parent count used before the switch "
                        "existed. mcts_pv heuristics are not affected: they "
                        "always search with 'off' (see 'Heuristic ... settings').")
    p.add_argument("--fixed-seat", type=int, default=None,
                   help="Pin the test side (--model) to this seat for every game "
                        "instead of rotating game_idx %% num_players. Use when "
                        "the game has no first/second-mover bias and you want a "
                        "stable per-seat number")
    p.add_argument("--paired-seats", action="store_true",
                   help="Pair every seed across all seats so deal-luck cancels "
                        "out: --games becomes pair count × num_players (the "
                        "same seed is replayed once per seat with the model "
                        "rotated). 2p only for now; mutually exclusive with "
                        "--fixed-seat.")
    # Phase 2 batched/threaded path. Model-involved eval defaults to the same
    # train.json::training.selfplay_concurrency.evaluator block used by
    # self-play, so workers share a BatchedEvaluator pool. --batched is kept as
    # a compatibility alias that forces the batched path even when the game has
    # no enabled selfplay_concurrency block; --no-batched opts back into the
    # legacy ProcessPool / synchronous path.
    p.add_argument("--batched", action="store_true",
                   help="Force the single-process batched eval runner. This is "
                        "usually unnecessary when train.json enables "
                        "selfplay_concurrency; model-involved eval uses that "
                        "path automatically.")
    p.add_argument("--no-batched", action="store_true",
                   help="Disable automatic batched eval and use the legacy "
                        "ProcessPool/synchronous path.")
    p.add_argument("--providers", type=str, default=None,
                   help="Comma-separated ONNX providers tried in order, "
                        "e.g. 'coreml,cpu' on Apple, 'cuda,cpu' on NVIDIA. "
                        "Overrides train.json selfplay_concurrency.evaluator."
                        " Omitted = use the json value; empty string = CPU/default.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override BatchedEvaluator batch size (default: "
                        "train.json selfplay_concurrency.evaluator.batch_size).")
    p.add_argument("--batch-timeout-us", type=int, default=None,
                   help="Override BatchedEvaluator timeout in microseconds "
                        "(default: train.json selfplay_concurrency.evaluator."
                        "batch_timeout_us).")
    p.add_argument("--max-queue-size", type=int, default=None,
                   help="Override BatchedEvaluator max queue size (default: "
                        "train.json selfplay_concurrency.evaluator."
                        "max_queue_size).")
    p.add_argument("--intra-op-num-threads", type=int, default=None,
                   help="Override ONNX intra-op SIMD threads per evaluator "
                        "(default: train.json selfplay_concurrency.evaluator."
                        "intra_op_num_threads).")
    p.add_argument("--safe-mode", action="store_true", default=False,
                   help="Enable MCTS safe mode: validate DAG edges at every "
                        "descent step. Slower but catches hash/schema bugs.")
    args = p.parse_args()

    import dinoboard_engine
    meta = dinoboard_engine.game_metadata(args.game)
    num_players = meta["num_players"]

    profile = resolve_profile(args.game, args.profile)
    game_max_plies = max_game_plies(args.game)

    # Resolve defaults from the chosen selector. The 'heuristic' actor spec
    # (without :name suffix) resolves to selector.name; heuristic_temp falls
    # back to selector.temperature unless --heuristic-temp overrides.
    heuristic_cfg, _ = find_heuristic_config(args.game)
    selector = heuristic_cfg["selectors"][args.heuristic_selector]
    default_heuristic_name = selector["name"]
    default_heuristic_temp = float(selector.get("temperature", 0.0))
    heuristic_temp = (args.heuristic_temp if args.heuristic_temp is not None
                      else default_heuristic_temp)
    available_heuristics = set(heuristic_cfg["named"].keys())

    model_kind, model_ref = _parse_actor(args.model, default_heuristic_name)
    opp_kind, opp_ref = _parse_actor(args.opponent, default_heuristic_name)

    # Validate heuristic names exist; validate model paths exist.
    for kind, ref, label in [(model_kind, model_ref, "--model"),
                             (opp_kind, opp_ref, "--opponent")]:
        if kind == "heuristic":
            if ref not in available_heuristics:
                raise SystemExit(
                    f"{label}: heuristic '{ref}' is not in "
                    f"games/{args.game}/config/heuristic.json::named "
                    f"(known: {sorted(available_heuristics)}).")
        else:
            if not Path(ref).exists():
                raise SystemExit(f"{label} path does not exist: {ref}")

    if args.belief and not Path(args.belief).exists():
        raise SystemExit(f"--belief path does not exist: {args.belief}")

    # Mode dispatch.
    if model_kind == "model" and opp_kind == "model":
        mode = "m_vs_m"
    elif model_kind == "model" and opp_kind == "heuristic":
        mode = "m_vs_h"
    elif model_kind == "heuristic" and opp_kind == "model":
        mode = "m_vs_h"  # same engine call; we just swap which seat the model occupies.
    else:
        mode = "h_vs_h"

    def _default_name(kind: str, ref: str) -> str:
        return ref if kind == "heuristic" else Path(ref).stem

    name_model = args.name_model or _default_name(model_kind, model_ref)
    name_opp = args.name_opponent or _default_name(opp_kind, opp_ref)

    out_path = Path(args.output)

    def _resolve_ts_override(flag):
        if flag is None:
            return profile.tail_solve_enabled
        return flag == "on"

    model_ts = _resolve_ts_override(args.model_tail_solve)
    opp_ts = _resolve_ts_override(args.opponent_tail_solve)

    if args.paired_seats and args.fixed_seat is not None:
        raise SystemExit("--paired-seats and --fixed-seat are mutually exclusive")
    if args.seed_stride < 1:
        raise SystemExit("--seed-stride must be >= 1")
    if args.paired_seats and args.games % num_players != 0:
        raise SystemExit(
            f"--paired-seats requires --games divisible by num_players "
            f"({num_players}); got --games={args.games}")

    tasks = []
    for game_idx in range(args.games):
        if args.paired_seats:
            # Pair every seed across all seats: seed bumps every num_players
            # games; within a pair the model rotates 0..N-1.
            seed = args.seed + (game_idx // num_players) * args.seed_stride
            model_seat = game_idx % num_players
        elif args.fixed_seat is not None:
            if not (0 <= args.fixed_seat < num_players):
                raise SystemExit(
                    f"--fixed-seat={args.fixed_seat} out of range for "
                    f"{num_players}p game")
            seed = args.seed + game_idx * args.seed_stride
            model_seat = args.fixed_seat
        else:
            seed = args.seed + game_idx * args.seed_stride
            model_seat = game_idx % num_players  # which seat the "test side" occupies

        common = {
            "game_idx": game_idx,
            "game": args.game,
            "seed": seed,
            "num_players": num_players,
            "model_seat": model_seat,
            "sims": args.sims if args.sims is not None else profile.simulations,
            "world_cap": args.world_cap if args.world_cap is not None else profile.world_cap,
            "c_puct": args.c_puct if args.c_puct is not None else profile.c_puct,
            "fpu_reduction": profile.fpu_reduction,
            "uct2_arrival_visits": (profile.uct2_arrival_visits
                                    if args.uct2_arrival_visits is None
                                    else args.uct2_arrival_visits == "on"),
            "belief_path": args.belief or "",
            "temp": args.temperature if args.temperature is not None else profile.temperature,
            "max_plies": game_max_plies,
            "opp_sel": profile.opponent_selection,
            "model_tail_solve_enabled": model_ts,
            "tail_solve_depth_limit": profile.tail_solve_depth_limit,
            "tail_solve_node_budget": profile.tail_solve_node_budget,
            "tail_solve_auxiliary_weight": profile.tail_solve_auxiliary_weight,
            "safe_mode": args.safe_mode,
            "top_k": profile.top_k,
        }

        if mode == "h_vs_h":
            heuristic_names = [opp_ref] * num_players
            heuristic_names[model_seat] = model_ref
            # Same temperature for every heuristic seat (fine for benchmarks;
            # split into two flags only if needed in future).
            common.update({
                "mode": "h_vs_h",
                "heuristic_names": heuristic_names,
                "heuristic_temps": [heuristic_temp] * num_players,
            })
        elif mode == "m_vs_h":
            # The C++ binding parameter `model_is_player` is which seat runs
            # the ONNX. If --model is the model side, that's `model_seat`. If
            # --model is the heuristic side, the model lives on the OTHER seat.
            if model_kind == "model":
                model_path_for_engine = model_ref
                heuristic_name_for_engine = opp_ref
                model_seat_in_engine = model_seat
            else:  # heuristic vs model — model is the opponent here
                model_path_for_engine = opp_ref
                heuristic_name_for_engine = model_ref
                # The "model side under test" rotates to model_seat, so the
                # ONNX (the actual opponent) sits on the OTHER seat. For 2p
                # this is (1 - model_seat); for >2p we pick the first non-model
                # seat. (The eval-vs-heuristic binding is single-model anyway,
                # so n>2 already implies n-1 heuristic seats with one model.)
                model_seat_in_engine = (model_seat + 1) % num_players
            common.update({
                "mode": "m_vs_h",
                "model_path": model_path_for_engine,
                "model_seat_in_engine": model_seat_in_engine,
                "heuristic_temp": heuristic_temp,
                "heuristic_selector": heuristic_name_for_engine,
            })
        else:  # m_vs_m
            # tail_solve fields are already in common; m_vs_m adds opponent_ts.
            common.update({
                "mode": "m_vs_m",
                "model_path": model_ref,
                "opponent_path": opp_ref,
                "opponent_tail_solve_enabled": opp_ts,
            })

        tasks.append(common)

    # Benchmark transparency: what each side really searches with (profile
    # plus CLI overrides; heuristics as the engine instantiates them). Printed
    # once and stored in every saved replay's config. A model and an mcts_pv
    # heuristic need not search alike (budget, tail solver, FPU).
    model_search = None
    if mode != "h_vs_h" and tasks:
        t0 = tasks[0]
        model_search = {
            "simulations": t0["sims"],
            "world_cap": t0["world_cap"],
            "c_puct": t0["c_puct"],
            "fpu_reduction": t0["fpu_reduction"],
            "uct2_arrival_visits": t0["uct2_arrival_visits"],
            "temperature": t0["temp"],
            "opponent_selection": t0["opp_sel"],
            "tail_solve_enabled": model_ts,
            "tail_solve_depth_limit": t0["tail_solve_depth_limit"],
            "tail_solve_node_budget": t0["tail_solve_node_budget"],
            "tail_solve_auxiliary_weight": t0["tail_solve_auxiliary_weight"],
            "top_k": t0["top_k"],
        }
        if mode == "m_vs_m":
            model_search["opponent_tail_solve_enabled"] = opp_ts
        print(f"Model search settings (profile '{args.profile}' + overrides): {model_search}",
              flush=True)
    heuristic_settings = {}
    for kind, ref in ((model_kind, model_ref), (opp_kind, opp_ref)):
        if kind == "heuristic" and ref not in heuristic_settings:
            heuristic_settings[ref] = {
                **dinoboard_engine.heuristic_settings(args.game, ref),
                "temperature": heuristic_temp,
            }
            print(f"Heuristic '{ref}' settings: {heuristic_settings[ref]}", flush=True)

    per_seat = [{"w": 0, "l": 0, "d": 0} for _ in range(num_players)]
    running_w = running_l = running_d = 0

    if args.games > 1 and not args.no_save:
        out_path.mkdir(parents=True, exist_ok=True)

    def _stream_print(r: dict, n_done: int) -> None:
        nonlocal running_w, running_l, running_d
        model_seat = r["model_seat"]
        outcome = test_side_outcome(r, model_seat)
        if outcome == "draw":
            winner_name = "draw"
            per_seat[model_seat]["d"] += 1
            running_d += 1
        elif outcome == "win":
            winner_name = name_model
            per_seat[model_seat]["w"] += 1
            running_w += 1
        else:
            winner_name = name_opp
            per_seat[model_seat]["l"] += 1
            running_l += 1
        running_total = running_w + running_l + running_d
        wr = ((running_w + 0.5 * running_d) / running_total * 100) if running_total else 0.0
        print(f"[{r['game_idx']+1}/{args.games}] {name_model}(seat{model_seat}) vs {name_opp} "
              f"seed={r['seed']}: {winner_name} wins, {r['total_plies']} plies  "
              f"({n_done}/{args.games}, {name_model} {running_w}-{running_l}-{running_d}={wr:.0f}%)",
              flush=True)

    results: list[dict] = []
    n_done = 0
    # Phase 2 batched path: m_vs_m and m_vs_h supported. h_vs_h has no ONNX
    # so falls back to legacy ProcessPool.
    batch_cfg = _resolve_eval_batch_config(args, mode)
    use_batched = batch_cfg is not None
    if (not use_batched and mode in ("m_vs_m", "m_vs_h")
            and args.workers > _SAFE_PROCESS_WORKERS):
        raise SystemExit(
            f"--workers {args.workers} on the legacy multiprocess path would "
            f"fork {args.workers} processes, each loading its own ONNX "
            f"evaluator(s) (model + belief"
            f"{' + opponent' if mode == 'm_vs_m' else ''}) — this multiplies "
            f"the model footprint by {args.workers} and OOM-kills the run. "
            f"Use the default batched path (single process, {args.workers} "
            f"threads sharing one evaluator pool with batched inference) — "
            f"and --providers "
            f"cuda,cpu / coreml,cpu to use the GPU — or lower --workers to "
            f"<= {_SAFE_PROCESS_WORKERS}.")
    if use_batched:
        import dinoboard_engine
        common_kwargs = dict(
            num_threads=max(1, args.workers),
            max_game_plies=tasks[0]["max_plies"],
            batch_size=batch_cfg["batch_size"],
            batch_timeout_us=batch_cfg["batch_timeout_us"],
            max_queue_size=batch_cfg["max_queue_size"],
            intra_op_num_threads=batch_cfg["intra_op_num_threads"],
            providers=batch_cfg["providers"],
            safe_mode=args.safe_mode,
        )
        print(
            "Using batched eval: "
            f"threads={common_kwargs['num_threads']}, "
            f"batch_size={common_kwargs['batch_size']}, "
            f"batch_timeout_us={common_kwargs['batch_timeout_us']}, "
            f"max_queue_size={common_kwargs['max_queue_size']}, "
            f"intra_op_num_threads={common_kwargs['intra_op_num_threads']}, "
            f"providers={common_kwargs['providers']} "
            f"(source: {batch_cfg['source']})",
            flush=True,
        )
        if mode == "m_vs_m":
            batched_tasks = []
            for t in tasks:
                n = t["num_players"]
                def _seat(model_path: str, ts_enabled: bool) -> dict:
                    return {
                        "model_path": model_path,
                        "simulations": t["sims"],
                        "world_cap": t["world_cap"],
                        "c_puct": t["c_puct"],
                        "fpu_reduction": t["fpu_reduction"],
                        "uct2_arrival_visits": t["uct2_arrival_visits"],
                        "temperature": t["temp"],
                        "opponent_selection": t["opp_sel"],
                        "tail_solve_enabled": ts_enabled,
                        "tail_solve_depth_limit": t["tail_solve_depth_limit"],
                        "tail_solve_node_budget": t["tail_solve_node_budget"],
                        "tail_solve_auxiliary_weight": t["tail_solve_auxiliary_weight"],
                        "top_k": t["top_k"],
                    }
                seat_configs = [_seat(t["opponent_path"], t["opponent_tail_solve_enabled"]) for _ in range(n)]
                seat_configs[t["model_seat"]] = _seat(t["model_path"], t["model_tail_solve_enabled"])
                batched_tasks.append({
                    "seed": t["seed"],
                    "seat_configs": seat_configs,
                    "belief_model_path": t.get("belief_path", ""),
                })
            out = dinoboard_engine.run_concurrent_arena_batch(
                game_id=args.game, tasks=batched_tasks, **common_kwargs)
        else:  # m_vs_h
            # All tasks share model_path / heuristic_selector / sims / etc.
            t0 = tasks[0]
            batched_tasks = [
                {"seed": t["seed"], "model_is_player": t["model_seat_in_engine"]}
                for t in tasks
            ]
            out = dinoboard_engine.run_concurrent_eval_vs_heuristic_batch(
                game_id=args.game,
                model_path=t0["model_path"],
                heuristic_selector=t0["heuristic_selector"],
                tasks=batched_tasks,
                simulations=t0["sims"],
                world_cap=t0["world_cap"],
                c_puct=t0["c_puct"],
                temperature=t0["temp"],
                heuristic_temperature=t0["heuristic_temp"],
                tail_solve_enabled=t0["model_tail_solve_enabled"],
                tail_solve_depth_limit=t0["tail_solve_depth_limit"],
                tail_solve_node_budget=t0["tail_solve_node_budget"],
                tail_solve_auxiliary_weight=t0["tail_solve_auxiliary_weight"],
                opponent_selection=t0["opp_sel"],
                belief_model_path=t0.get("belief_path", ""),
                top_k=t0["top_k"],
                fpu_reduction=t0["fpu_reduction"],
                uct2_arrival_visits=t0["uct2_arrival_visits"],
                **common_kwargs,
            )
        for idx, r in enumerate(out["results"]):
            t = tasks[idx]
            r2 = {
                "game_idx": t["game_idx"],
                "seed": t["seed"],
                "model_seat": t["model_seat"],
                "winner": r["winner"],
                "draw": r["draw"],
                "terminal_z_values": list(r["terminal_z_values"]),
                "total_plies": r["total_plies"],
                "action_history": list(r["action_history"]),
            }
            results.append(r2)
            n_done += 1
            _stream_print(r2, n_done)
    elif args.workers <= 1:
        for t in tasks:
            r = _run_one_game(t)
            results.append(r)
            n_done += 1
            _stream_print(r, n_done)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one_game, t): t for t in tasks}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                n_done += 1
                _stream_print(r, n_done)

    results.sort(key=lambda r: r["game_idx"])

    for r in results:
        game_idx = r["game_idx"]
        model_seat = r["model_seat"]

        if args.no_save:
            continue

        players = {}
        for k in range(num_players):
            if k == model_seat:
                players[f"player_{k}"] = {
                    "name": name_model,
                    "type": "heuristic" if model_kind == "heuristic" else "model",
                }
            else:
                players[f"player_{k}"] = {
                    "name": name_opp,
                    "type": "heuristic" if opp_kind == "heuristic" else "model",
                }

        replay = build_replay_dict(
            game_id=args.game,
            seed=r["seed"],
            action_history=r["action_history"],
            players=players,
            result={
                "winner": r["winner"],
                "draw": r["draw"],
                "total_plies": r["total_plies"],
            },
            config={
                "profile": args.profile,
                "simulations": profile.simulations,
                "world_cap": profile.world_cap,
                "temperature": profile.temperature,
                "opponent_selection": profile.opponent_selection,
                "tail_solve_enabled": profile.tail_solve_enabled,
                "max_game_plies": game_max_plies,
                "heuristic_temperature": (
                    heuristic_temp
                    if (model_kind == "heuristic" or opp_kind == "heuristic")
                    else None),
                "model_seat": model_seat,
                "num_players": num_players,
                "match_mode": (
                    "h_vs_h" if (model_kind == "heuristic" and opp_kind == "heuristic")
                    else "m_vs_h" if (model_kind == "heuristic") != (opp_kind == "heuristic")
                    else "m_vs_m"),
                "model_kind": model_kind,
                "opponent_kind": opp_kind,
                "model_ref": model_ref,
                "opponent_ref": opp_ref,
                "model_search": model_search,
                "heuristic_settings": heuristic_settings,
            },
        )

        if args.games == 1:
            if out_path.suffix == ".json":
                dest = out_path
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                dest = out_path / f"game_seed{r['seed']}.json"
        else:
            dest = out_path / f"game_{game_idx:03d}_seed{r['seed']}_seat{model_seat}.json"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(replay, ensure_ascii=False))
        print(f"  Saved to {dest}")

    if args.games > 1:
        total_w = sum(s["w"] for s in per_seat)
        total_l = sum(s["l"] for s in per_seat)
        total_d = sum(s["d"] for s in per_seat)
        # Win rate counts draws as half wins (Elo convention).
        wr_pct = (total_w + 0.5 * total_d) / args.games * 100
        # Standard error for 95% CI on a paired win rate.
        # Var per game ≈ p*(1-p), SE = sqrt(Var/N), 95% half-width ≈ 1.96*SE.
        p = wr_pct / 100
        import math
        se_pct = math.sqrt(p * (1 - p) / args.games) * 100
        ci_half = 1.96 * se_pct
        print(f"\n=== {name_model} vs {name_opp} ({num_players}p) ===")
        print(f"Overall: {total_w}W-{total_l}L-{total_d}D / {args.games} games "
              f"({wr_pct:.2f}%, 95% CI ±{ci_half:.2f}pp)")
        for k in range(num_players):
            n_k = per_seat[k]["w"] + per_seat[k]["l"] + per_seat[k]["d"]
            if n_k == 0:
                continue
            seat_wr = (per_seat[k]['w'] + 0.5 * per_seat[k]['d']) / n_k * 100
            print(f"  As seat{k}: {per_seat[k]['w']}W-{per_seat[k]['l']}L-{per_seat[k]['d']}D / {n_k} "
                  f"({seat_wr:.2f}%)")


if __name__ == "__main__":
    main()
