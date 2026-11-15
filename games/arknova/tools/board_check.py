"""End-to-end check of scripts/arknova_board.py: ark-nova-ai's server with the C++ engine as referee.

Starts the server in this process on a free port with a scratch data directory and plays whole games
through its HTTP API (always the first option, one undo and redo, ending each turn) against each
opponent. Then checks that no module of ark-nova-ai's rules was ever imported: the game, the option
texts, the activity feed, the hints and the opponents all came from C++.

    python games/arknova/tools/board_check.py --ref ~/ark-nova-fork --model m.onnx
"""
import argparse
import json
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import arknova_board  # noqa: E402

RULES_MODULES = ("arknova.engine", "arknova.effects", "arknova.actions", "arknova.abilities",
                 "arknova.abilities_basic", "arknova.abilities_action", "arknova.sponsor_rules",
                 "arknova.sponsor_effects", "arknova.map_rules", "arknova.final_scoring", "arknova.bot")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Board:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy for localhost

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, method="GET" if body is None else "POST",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Lang": "zh-CN"})
        with self.opener.open(req, timeout=900) as r:
            return json.loads(r.read())


def play(board, opponent, seed, max_moves):
    b = board.call("/api/new", {"seed": seed, "checkpoint": opponent, "map": "map0"})
    moves = 0
    while not b["over"]:
        if moves >= max_moves:
            raise AssertionError(f"{opponent} seed {seed}: no end after {max_moves} human moves")
        if b["decision"] is None:
            b = board.call("/api/end_turn", {})
            continue
        if not b["decision"]["options"]:
            raise AssertionError(f"{opponent} seed {seed}: a decision without options")
        b = board.call("/api/move", {"index": 0})
        moves += 1
        if moves == 20:
            before = json.dumps(b["decision"])
            board.call("/api/undo", {})
            b = board.call("/api/redo", {})
            if json.dumps(b["decision"]) != before:
                raise AssertionError(f"{opponent} seed {seed}: undo and redo changed the decision")
    if not b["winners"] or not b["activity_history"] or not b["log"]:
        raise AssertionError(f"{opponent} seed {seed}: the finished game shows no winners, feed or log")
    return moves, b["scores"], b["winners"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="ark-nova-ai checkout with the board's table and rules hooks")
    ap.add_argument("--model", required=True, help="DinoBoard ONNX model for the network opponent")
    ap.add_argument("--profile", default="web_casual", choices=["web_expert", "web_casual"])
    ap.add_argument("--max-moves", type=int, default=400, help="human moves before a game counts as stuck")
    args = ap.parse_args()

    import uvicorn

    app = arknova_board.setup(Path(args.ref), Path(args.model), args.profile, Path(tempfile.mkdtemp()))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    board = Board(port)
    for opponent, seed in ((arknova_board.BEAM, 11), (arknova_board.DINOBOARD, 12)):
        t0 = time.time()
        moves, scores, winners = play(board, opponent, seed, args.max_moves)
        print(f"{opponent} seed {seed}: {moves} human moves, scores {scores}, winners {winners} "
              f"({time.time() - t0:.0f}s)")
    server.should_exit = True
    loaded = sorted(m for m in sys.modules if any(m == r or m.startswith(r + ".") for r in RULES_MODULES))
    if loaded:
        raise AssertionError(f"ark-nova-ai's rules were imported: {loaded}")
    print("no ark-nova-ai rules module was imported")


if __name__ == "__main__":
    main()
