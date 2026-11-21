"""Play Ark Nova on ark-nova-ai's board against DinoBoard.

The board and the logins are ark-nova-ai's (github.com/collinfeng/ark-nova-ai, MIT; see
games/arknova/LICENSE-ark-nova-ai); the game is refereed by DinoBoard's C++ engine (its table is
games/arknova/tools/host_table.HostTable, so the truth, the options, their texts and the activity feed
come from C++). This script starts the server with two opponents only, on Map 0:
  - dinoboard: DinoBoard's token-transformer MCTS (search from the web.json profile). Without
    --model it plays games/arknova/model/arknova_2p.onnx, for now a randomly initialised network
    (scripts/init_random_models.py --game arknova --variants 2p --no-belief).
  - beam_3x8 (the default): the rule-based search of that name (DinoBoard's C++ port of ark-nova-ai's
    strongest one, heuristic.json `named`); the human's hints are then its own choice too, unless the
    page's hint model setting names the network.

    git clone -b rulebook https://github.com/Zhiqi-Wang/ark-nova-ai.git ~/ark-nova-fork
    python scripts/arknova_board.py --ref ~/ark-nova-fork [--model m.onnx] [--profile web_expert]

Then open http://127.0.0.1:8765 once the page is built (in the ark-nova-ai checkout:
cd web && pnpm install && pnpm build), or run `pnpm dev` there and open the address it prints.
Saves, logins and the server's config.json go to runs/arknova_board/data (ARKNOVA_DATA), not into
the ark-nova-ai checkout; a browser on this machine is signed in as `player` without a password.

Each DinoBoard decision loads the table's state into a truth GameSession, which searches on the
mover's masked view (it never reads the opponent's hidden cards), one atom at a time, until the
chosen atoms name one of the board's options. --ref must be a checkout with the board's table and
rules hooks (arknova/play/table.py, arknova/play/rules.py; the rulebook branch of
github.com/Zhiqi-Wang/ark-nova-ai, see docs/games/arknova_api.md): the script points both at C++
(host_table.HostTable, HostRules), so the board's hints follow the same rules as the game.
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
    """ark-nova-ai's AI interface (analyse(game) -> Analysis) over a DinoBoard GameSession: the network's
    MCTS, or with `heuristic` one of the game's registered heuristics (heuristic.json `named`), which also
    pick from the mover's masked view."""

    def __init__(self, analysis_type, model: Path | None, profile: str | None, heuristic: str | None = None):
        self.Analysis = analysis_type
        self.name = self.checkpoint = heuristic or DINOBOARD
        self.heuristic = heuristic
        if heuristic is None:
            p = resolve_profile(GAME, profile)
            self.search = dict(simulations=p.simulations, world_cap=p.world_cap, c_puct=p.c_puct,
                               temperature=p.temperature, opponent_selection=p.opponent_selection,
                               top_k=p.top_k, fpu_reduction=p.fpu_reduction)
        self.session = engine.GameSession(GAME, 0, "" if heuristic else str(model))
        self.lock = threading.Lock()  # one session serves every game of the server

    def analyse(self, game):
        return self.analyse_state(game.state, game.options())

    def _pick(self):
        """The next atom and (for the network) its search value."""
        if self.heuristic is not None:
            return self.session.get_heuristic_action(self.heuristic)["action"], 0.0
        r = self.session.get_ai_action(**self.search)
        return r["action"], float(r["stats"]["best_value"])

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
            value = None
            while True:
                atom, atom_value = self._pick()
                if value is None:
                    value = atom_value
                # The session applies atoms that have a single continuation by itself, so the next atom's
                # place is the prefix's length, not the number of atoms chosen here.
                at = len(s.get_state_dict()["prefix"].split())
                named = {k for k, atoms in listed if atoms[at] == atom}
                if len(named) == 1:
                    return self._result(keys, named.pop(), value)
                if not named:
                    raise RuntimeError(f"DinoBoard chose atom {atom}, which names no option")
                s.apply_action(atom)
                listed = self._options()

    def _options(self):
        return [(dumps(json.loads(k)), atoms) for k, atoms in self.session.external_options()]

    def _result(self, keys, key, value):
        best = keys.index(key)
        # A searched move with no distribution over the options: shown like a rule policy's (no odds).
        return self.Analysis([1.0 if i == best else 0.0 for i in range(len(keys))], round(value, 4), best,
                             {"heuristic": True})


def configure(server, ai_module, dinoboard):
    """Map 0, and only the DinoBoard network and beam_3x8 (the default) as opponents and hint models."""
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
            return DinoBoardAI(dinoboard.Analysis, None, None, heuristic=BEAM)
        raise HTTPException(404, f"no opponent {checkpoint}: this board offers {DINOBOARD} and {BEAM}")

    def hint_model(name):
        if name == BEAM:
            return DinoBoardAI(dinoboard.Analysis, None, None, heuristic=BEAM)  # its own choice is the hint
        return None if name in (None, "", "default") else opponent(name)

    server._ai = opponent
    server._hint_ai = hint_model
    server.ctx["default"] = BEAM


def setup(ref: Path, model: Path | None, profile: str, data: Path):
    """ark-nova-ai's server app with DinoBoard as referee and opponent (see the module docstring);
    `data` holds its saves, logins and config.json."""
    ref = ref.expanduser().resolve()
    if not (ref / "arknova" / "play" / "server.py").is_file():
        raise SystemExit(f"{ref} is not an ark-nova-ai checkout")
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "config.json").exists():
        (data / "config.json").write_text(json.dumps({"skip_login": "player"}, indent=1) + "\n")
    os.environ["ARKNOVA_DATA"] = str(data)  # read when ark-nova-ai's modules are imported
    sys.path.insert(0, str(ref))
    from arknova.play import ai as ai_module, auth, rules, server, table  # noqa: PLC0415
    from host_table import HostRules, HostTable  # noqa: PLC0415

    table.TABLE = HostTable
    rules.RULES = HostRules()

    model = model.resolve() if model else expected_model_path(GAME)
    if not model.is_file():
        raise SystemExit(f"no model at {model}")
    configure(server, ai_module, DinoBoardAI(ai_module.Analysis, model, profile))
    if auth.ensure():
        print(f"made the logins: see {auth.AUTH_DIR / 'credentials.txt'}")
    if not server.DIST.is_dir():
        print(f"{server.DIST} is not built: run `pnpm install && pnpm build` in {ref / 'web'}, "
              f"or `pnpm dev` there and open its address")
    print(f"opponents: {DINOBOARD} ({model.name}, {profile}) and {BEAM}")
    return server.app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="path of the ark-nova-ai checkout")
    ap.add_argument("--model", help="DinoBoard ONNX model (default: games/arknova/model/arknova_2p.onnx)")
    ap.add_argument("--profile", default="web_expert", choices=["web_expert", "web_casual"],
                    help="MCTS profile of games/arknova/config/web.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app = setup(Path(args.ref), Path(args.model) if args.model else None, args.profile, RUN_DIR / "data")
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
