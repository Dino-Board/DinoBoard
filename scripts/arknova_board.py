"""Play Ark Nova on ark-nova-ai's board against DinoBoard.

The board, the game engine that runs the game and the logins are ark-nova-ai's
(github.com/collinfeng/ark-nova-ai, MIT; see games/arknova/LICENSE-ark-nova-ai). This script starts
its server with two opponents only, on Map 0:
  - dinoboard: DinoBoard's token-transformer MCTS (search from the web.json profile). Without
    --model it plays games/arknova/model/arknova_2p.onnx, for now a randomly initialised network
    (scripts/init_random_models.py --game arknova --variants 2p --no-belief).
  - beam_3x8: ark-nova-ai's strongest rule-based search.

    python scripts/arknova_board.py --ref ~/ref/ark-nova-ai [--model m.onnx] [--profile web_expert]

Then open http://127.0.0.1:8765 once the page is built (in the ark-nova-ai checkout:
cd web && pnpm install && pnpm build), or run `pnpm dev` there and open the address it prints.
Saves, logins and the server's config.json go to runs/arknova_board/data (ARKNOVA_DATA), not into
the ark-nova-ai checkout; a browser on this machine is signed in as `player` without a password.

Each DinoBoard decision loads ark-nova-ai's state into a truth GameSession, which searches on the
mover's masked view (it never reads the opponent's hidden cards), one atom at a time, until the
chosen atoms name one of the board's options. The process follows ark-nova-ai's rules where they
depart from the rulebook (set_external_rules), so both engines list the same options.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "games" / "arknova" / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine  # noqa: E402
from diff_test import canon, dumps, export_state  # noqa: E402
from model_paths import expected_model_path  # noqa: E402
from training.mcts_profile import resolve_profile  # noqa: E402

GAME = "arknova"
DINOBOARD = "dinoboard"
BEAM = "beam_3x8"
RUN_DIR = PROJECT_ROOT / "runs" / "arknova_board"


class DinoBoardAI:
    """ark-nova-ai's AI interface (analyse(game) -> Analysis) over a DinoBoard GameSession."""

    def __init__(self, analysis_type, model: Path, profile: str):
        self.Analysis = analysis_type
        self.name = self.checkpoint = DINOBOARD
        p = resolve_profile(GAME, profile)
        self.search = dict(simulations=p.simulations, world_cap=p.world_cap, c_puct=p.c_puct,
                           temperature=p.temperature, opponent_selection=p.opponent_selection,
                           top_k=p.top_k, fpu_reduction=p.fpu_reduction)
        self.session = engine.GameSession(GAME, 0, str(model))
        self.lock = threading.Lock()  # one session serves every game of the server

    def analyse(self, game):
        return self.analyse_state(game.state, game.options())

    def analyse_state(self, state, options):
        keys = [dumps(canon(o.key)) for o in options]
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"ark-nova-ai lists an option twice: {keys}")
        if len(keys) == 1:
            return self._result(keys, keys[0], 0.0)
        with self.lock:
            s = self.session
            s.load_external_state(dumps(export_state(state)))
            listed = self._options()
            if sorted({k for k, _ in listed}) != sorted(keys):
                only_ours = sorted({k for k, _ in listed} - set(keys))
                only_theirs = sorted(set(keys) - {k for k, _ in listed})
                raise RuntimeError(f"the engines list different options ({state.stack[-1].kind}): "
                                   f"DinoBoard only {only_ours[:5]}, ark-nova-ai only {only_theirs[:5]}")
            chosen, value = 0, None
            while True:
                r = s.get_ai_action(**self.search)
                if value is None:
                    value = float(r["stats"]["best_value"])
                atom = r["action"]
                named = {k for k, atoms in listed if atoms[chosen] == atom}
                if len(named) == 1:
                    return self._result(keys, named.pop(), value)
                if not named:
                    raise RuntimeError(f"DinoBoard chose atom {atom}, which names no option")
                s.apply_action(atom)
                chosen += 1
                listed = self._options()

    def _options(self):
        return [(dumps(json.loads(k)), atoms) for k, atoms in self.session.external_options()]

    def _result(self, keys, key, value):
        best = keys.index(key)
        # A searched move with no distribution over the options: shown like a rule policy's (no odds).
        return self.Analysis([1.0 if i == best else 0.0 for i in range(len(keys))], round(value, 4), best,
                             {"heuristic": True})


def configure(server, ai_module, dinoboard):
    """Map 0, and only the DinoBoard network and beam_3x8 as opponents and hint models."""
    HTTPException = server.HTTPException
    server.MAPS[:] = ["map0"]
    server.OFFERED_NETWORKS[:] = []
    server.ANY_MAP_NETWORKS[:] = []
    builtin = ai_module.BUILTIN_OPPONENTS
    beam = builtin[BEAM]
    builtin.clear()
    builtin[DINOBOARD] = None  # listed by /api/checkpoints; _ai below makes the opponent
    builtin[BEAM] = beam

    def opponent(checkpoint=None, map_name="map0"):
        if map_name != "map0":
            raise HTTPException(400, "this board plays Map 0 only")
        if checkpoint in (None, "", DINOBOARD):
            return dinoboard
        if checkpoint == BEAM:
            return ai_module.HeuristicAI(BEAM)  # its own copy: a beam keeps a plan per game
        raise HTTPException(404, f"no opponent {checkpoint}: this board offers {DINOBOARD} and {BEAM}")

    def hint_model(name):
        return None if name in (None, "", "default") else opponent(name)

    server._ai = opponent
    server._hint_ai = hint_model
    server.ctx["default"] = DINOBOARD


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="path of the ark-nova-ai checkout")
    ap.add_argument("--model", help="DinoBoard ONNX model (default: games/arknova/model/arknova_2p.onnx)")
    ap.add_argument("--profile", default="web_expert", choices=["web_expert", "web_casual"],
                    help="MCTS profile of games/arknova/config/web.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    ref = Path(args.ref).expanduser().resolve()
    if not (ref / "arknova" / "play" / "server.py").is_file():
        raise SystemExit(f"{ref} is not an ark-nova-ai checkout")
    data = RUN_DIR / "data"
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "config.json").exists():
        (data / "config.json").write_text(json.dumps({"skip_login": "player"}, indent=1) + "\n")
    os.environ["ARKNOVA_DATA"] = str(data)  # read when ark-nova-ai's modules are imported
    sys.path.insert(0, str(ref))
    import uvicorn  # noqa: PLC0415
    from arknova.play import ai as ai_module, auth, server  # noqa: PLC0415

    model = Path(args.model).resolve() if args.model else expected_model_path(GAME)
    if not model.is_file():
        raise SystemExit(f"no model at {model}")
    engine.set_external_rules(GAME, True)
    configure(server, ai_module, DinoBoardAI(ai_module.Analysis, model, args.profile))
    if auth.ensure():
        print(f"made the logins: see {auth.AUTH_DIR / 'credentials.txt'}")
    if not server.DIST.is_dir():
        print(f"{server.DIST} is not built: run `pnpm install && pnpm build` in {ref / 'web'}, "
              f"or `pnpm dev` there and open its address")
    print(f"opponents: {DINOBOARD} ({model.name}, {args.profile}) and {BEAM}")
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
