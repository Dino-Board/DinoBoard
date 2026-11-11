#!/usr/bin/env python3
"""External-GT 测试 runner for 6 nimmt!.

把 DinoBoard 自己的 AI（通过 REST API）放进**别人写的 6 nimmt! 评测框架**里玩，
用来端到端验证：
  1. AI 服务的接入协议（initial_observation / observe / decide）真的能跑
  2. 我们的 AI 强度大致符合预期（vs 同目录下编译好的 `external_ai.out`）

整个 runner 是「自带 GT」的：用 `engine.GameSession` 当 ground truth，按 6 nimmt!
判官（`judge.cpp`，Apache LZTHW 在课程作业里的 reference judge）的 stdio 协议**直接调用**
同目录下的 `external_ai.out`（`./external_ai`, `./easiest` 等任何符合协议的可执行文件），
不需要走原来那个 fork-exec 的 judge 进程；同时通过 REST API 把我们的 AI 当成另一个玩家。

DESIGN

    +--------+        +-------------------+        +-------------------+
    | runner | ---->  | engine.GameSession | <----  | uvicorn AI service |
    | (truth)|  GT    +-------------------+        +-------------------+
    |        | ──fork──▶ ./external_ai.out (per-move stdio, 短命)
    +--------+

**runner 是真值**。每个 ply：
    1. 从 truth session 抽公开局面 → 按"判官输入格式"喂给当前玩家
    2. 当前玩家是外部 AI → fork 一个 `./external_ai.out` 子进程，stdin 写局面，stdout 读出 card / row
                       → 我们的 AI → 调 `POST /ai/sessions/{sid}/decide`
    3. 把动作 apply 进 truth；同时给所有其它玩家的 AI session 调
       `POST /ai/sessions/{sid}/observe`，喂 `(action_id, events, public_snapshot)`

为什么不直接用原 `judge.cpp`：
  - 原 judge 一手一 fork 启动玩家，**没有跨手的进程**——这正合外部 AI 的胃口
    (`external_ai.cpp` 读完 stdin 就 exit)，但**完全不适合我们的 AI**（要维护 belief
    tracker 跨 ply 的 session 状态）
  - 把我们的 AI 拆成"每次启动 → 重建 session → 喂全部历史 → 决策 → 退出"会让一场
    10 ply 的游戏跑成 O(N²) 通信，并且要一份单独的"重放历史"逻辑——脆弱、慢、
    多余
  - 直接用 GameSession 当 truth，外部 AI 作为短命子进程，按它的协议喂它就行；
    我们的 AI 走 REST 完全合协议——结构对称、调试友好

USAGE

    # 0) 编译外部 AI（只需做一次）
    cd games/6nimmit/external_ground_truth
    g++ -O2 -o external_ai.out external_ai.cpp
    g++ -O2 -o easiest.out easiest.cpp

    # 1) 启动 DinoBoard AI 服务
    uvicorn platform.web.app:app --port 8000 &

    # 2) 跑一场 2 人对局：我方 seat 0 vs 外部 AI seat 1
    python3 runner.py --num-players 2 --me 0 --external ./external_ai.out --games 1

    # 3) 跑 4 人混战：我方 seat 0（DinoBoard），其余 3 家用外部 AI 混搭
    python3 runner.py --num-players 4 --me 0 \
        --external ./external_ai.out,./external_ai.out,./easiest.out --games 5

    # 注：当前 runner 只支持单 DinoBoard 座位（apply_action_with_trace 推进
    # truth 时只能 serialize 一个 perspective）。多座位支持需要单独暴露一个
    # snapshot-only binding。

OUTPUTS

每场结束打印：
    - 终局牛头分（越少越好，DinoBoard 的胜负判定）
    - 我方动作历史 / 对手动作历史（card 编号）
    - 每方平均 decide 用时
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make sure we can import engine when this script is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import dinoboard_engine as engine  # noqa: E402  (after sys.path edit)

DEFAULT_BASE_URL = os.environ.get("DINOBOARD_AI_API", "http://127.0.0.1:8000")

NUM_ROWS = 4
ROW_CAP = 5
NUM_ROUNDS = 10
NUM_CARDS = 104
HAND_SIZE = 10
TIME_LIMIT_SECONDS = 5.0


# ---------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------


def _http_post(base: str, path: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _http_delete(base: str, path: str, timeout: float = 5.0) -> None:
    req = urllib.request.Request(f"{base}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------
# Snapshot / move helpers
# ---------------------------------------------------------------------


def _hand_cards(state_dict: dict, seat: int) -> list[int]:
    return list(state_dict["players"][seat]["hand"])


def _table(state_dict: dict) -> list[list[int]]:
    rows = state_dict["table"]
    out = []
    for r in rows:
        row = list(r)
        if len(row) < ROW_CAP:
            row += [0] * (ROW_CAP - len(row))
        out.append(row[:ROW_CAP])
    return out


def _action_to_card(action_id: int) -> Optional[int]:
    if 0 <= action_id < NUM_CARDS:
        return action_id + 1
    return None


def _action_to_row(action_id: int) -> Optional[int]:
    if NUM_CARDS <= action_id < NUM_CARDS + NUM_ROWS:
        return action_id - NUM_CARDS
    return None


def _card_to_action(card: int) -> int:
    return card - 1


def _row_to_action(row: int) -> int:
    return NUM_CARDS + row


# ---------------------------------------------------------------------
# External-AI subprocess (one fork per move, judge.cpp protocol)
# ---------------------------------------------------------------------


@dataclass
class ExternalAi:
    cmd: str
    seat: int
    history_plays: list[list[int]] = field(default_factory=list)  # one row per finished round
    history_tables: list[list[list[int]]] = field(default_factory=list)  # table after each finished round
    initial_table: Optional[list[list[int]]] = None
    decide_time: float = 0.0
    decide_count: int = 0

    def _build_input(self, hand: list[int], cur_round: int, scores: list[int],
                     num_players: int, num_cards: int, small: bool,
                     this_round_plays: Optional[list[int]] = None) -> str:
        """See `judge.cpp::run_program` for the canonical protocol."""
        lines: list[str] = []
        lines.append(f"{self.seat} {cur_round}")
        lines.append(" ".join(str(c) for c in sorted(hand)))
        lines.append(f"{num_players} {num_cards}")
        # Initial 4 rows × 5 cells
        for row in self.initial_table or [[0] * ROW_CAP for _ in range(NUM_ROWS)]:
            lines.append(" ".join(str(c) for c in row))
        # Past full rounds: plays line, then 4 table lines
        for plays, table in zip(self.history_plays, self.history_tables):
            lines.append(" ".join(str(c) for c in plays))
            for row in table:
                lines.append(" ".join(str(c) for c in row))
        # SMALL mode: append this round's plays line (judge has already
        # entered next round's small-pick stage).
        if small:
            assert this_round_plays is not None
            lines.append(" ".join(str(c) for c in this_round_plays))
        # Scores
        lines.append(" ".join(str(s) for s in scores))
        return "\n".join(lines) + "\n"

    def ask(self, hand: list[int], cur_round: int, scores: list[int],
            num_players: int, num_cards: int, small: bool,
            this_round_plays: Optional[list[int]] = None) -> int:
        argv = [self.cmd]
        if small:
            argv.append("-small")
        payload = self._build_input(
            hand, cur_round, scores, num_players, num_cards, small,
            this_round_plays,
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            argv, input=payload, capture_output=True, text=True,
            timeout=TIME_LIMIT_SECONDS, check=False,
        )
        self.decide_time += time.perf_counter() - t0
        self.decide_count += 1
        out = (proc.stdout or "").strip().split()
        if not out:
            raise RuntimeError(f"external AI {self.cmd} produced empty stdout: {proc.stderr!r}")
        return int(out[0])


# ---------------------------------------------------------------------
# DinoBoard AI session (REST)
# ---------------------------------------------------------------------


@dataclass
class RestAi:
    base: str
    game_id: str
    seat: int
    session_id: str = ""
    decide_time: float = 0.0
    decide_count: int = 0

    def open(self, initial_observation: dict, seed: int) -> None:
        body = {
            "game_id": self.game_id,
            "seed": seed,
            "my_seat": self.seat,
            "initial_observation": initial_observation,
        }
        resp = _http_post(self.base, "/ai/sessions", body)
        self.session_id = resp["session_id"]

    def close(self) -> None:
        if self.session_id:
            _http_delete(self.base, f"/ai/sessions/{self.session_id}")

    def observe(self, action_id: int, events: list, public_snapshot: dict) -> None:
        _http_post(self.base, f"/ai/sessions/{self.session_id}/observe", {
            "action_id": int(action_id),
            "events": events,
            "public_snapshot": public_snapshot,
        })

    def decide(self) -> int:
        t0 = time.perf_counter()
        resp = _http_post(self.base, f"/ai/sessions/{self.session_id}/decide", {})
        self.decide_time += time.perf_counter() - t0
        self.decide_count += 1
        return int(resp["action_id"])


# ---------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------


def play_one_game(*, num_players: int, my_seats: set[int], external_cmds: list[str],
                  base: str, seed: int, verbose: bool = False) -> dict:
    if num_players not in (2, 3, 4):
        raise ValueError("6 nimmt! supports 2-4 players only")

    game_id = f"6nimmit_{num_players}p"

    # Truth session (in-process; not the same object as any AI session).
    truth = engine.GameSession(game_id=game_id, seed=seed)

    # External players (one ExternalAi per non-self seat).
    externals: dict[int, ExternalAi] = {}
    ext_seat_iter = iter(external_cmds)
    for p in range(num_players):
        if p in my_seats:
            continue
        externals[p] = ExternalAi(cmd=next(ext_seat_iter), seat=p)

    # REST sessions for our seats.
    rest_ais: dict[int, RestAi] = {}
    try:
        for p in my_seats:
            init_obs = truth.extract_initial_observation(p)
            ai = RestAi(base=base, game_id=game_id, seat=p)
            ai.open(init_obs, seed=secrets.randbits(64))
            rest_ais[p] = ai

        # Initialise the external-AI history buffer with the starting table.
        sd = truth.get_state_dict()
        starting_table = _table(sd)
        for ext in externals.values():
            ext.initial_table = starting_table

        # Per-round play buffer (filled as 4 commits land; flushed at round end).
        plays_this_round: list[int] = [0] * num_players

        cur_round = 0
        scores_for_external = [0] * num_players  # 6nimmt judge expects external scores

        while not truth.is_terminal:
            sd = truth.get_state_dict()
            actor = sd["current_player"]
            small_phase = bool(sd["small_phase_active"])
            hand = _hand_cards(sd, actor)

            if actor in externals:
                ext = externals[actor]
                if small_phase:
                    raw = ext.ask(
                        hand=hand, cur_round=cur_round + 1,
                        scores=scores_for_external,
                        num_players=num_players, num_cards=NUM_CARDS,
                        small=True,
                        this_round_plays=plays_this_round,
                    )
                    if not (0 <= raw < NUM_ROWS):
                        raw = 0
                    action_id = _row_to_action(raw)
                else:
                    raw_card = ext.ask(
                        hand=hand, cur_round=cur_round,
                        scores=scores_for_external,
                        num_players=num_players, num_cards=NUM_CARDS,
                        small=False,
                    )
                    if raw_card not in hand:
                        raw_card = min(hand)
                    action_id = _card_to_action(raw_card)
            else:
                action_id = rest_ais[actor].decide()

            # Apply to truth ONCE per ply. `apply_action_with_trace` advances
            # truth as a side effect, so we cannot call it per-perspective.
            # The current binding emits a trace for one chosen perspective;
            # that's also the perspective whose REST session we feed below.
            # Multi-DinoBoard-seat support would need a separate
            # "snapshot-only" binding — not exposed today, so the runner
            # restricts to a single DinoBoard seat (enforced in `main`).
            ai_seat = next(iter(rest_ais))
            trace = truth.apply_action_with_trace(action_id, perspective=ai_seat)
            rest_ais[ai_seat].observe(
                action_id, trace.get("events", []), trace.get("public_snapshot", {}),
            )

            # Track plays / round flips for the external-judge protocol.
            if 0 <= action_id < NUM_CARDS:
                plays_this_round[actor] = action_id + 1

            new_sd = truth.get_state_dict()
            if new_sd["current_round"] != cur_round and not new_sd["small_phase_active"]:
                # Round just settled — record the now-public table + plays.
                for ext in externals.values():
                    ext.history_plays.append(list(plays_this_round))
                    ext.history_tables.append(_table(new_sd))
                cur_round = new_sd["current_round"]
                plays_this_round = [0] * num_players
                # Update bullhead-derived "scores" the way 6nimmt judge does
                # (it tracks negative bullhead totals).
                for p in range(num_players):
                    scores_for_external[p] = -int(new_sd["players"][p]["bullhead_count"])

            if verbose:
                print(f"  ply {sd['ply']} actor={actor} action_id={action_id}")

        final = truth.get_state_dict()
        return {
            "winner": final["winner"],
            "bullhead_counts": [int(final["players"][p]["bullhead_count"]) for p in range(num_players)],
            "rest_decide_avg_ms": {p: 1000.0 * ai.decide_time / max(1, ai.decide_count) for p, ai in rest_ais.items()},
            "ext_decide_avg_ms": {p: 1000.0 * ext.decide_time / max(1, ext.decide_count) for p, ext in externals.items()},
        }
    finally:
        for ai in rest_ais.values():
            ai.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--me", default="0", help="comma-separated seats DinoBoard plays (e.g. '0' or '0,2')")
    ap.add_argument("--external", default="./external_ai.out",
                    help="comma-separated external AI executables for the OTHER seats; cycles if shorter than needed")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--base", default=DEFAULT_BASE_URL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    my_seats = {int(s) for s in args.me.split(",")}
    if len(my_seats) != 1:
        ap.error("--me currently must specify exactly one seat (the runner's "
                 "apply path serializes one perspective per ply)")
    external_cmds = [s.strip() for s in args.external.split(",") if s.strip()]
    other_count = args.num_players - len(my_seats)
    if other_count <= 0:
        ap.error("at least one seat must NOT be in --me")
    while len(external_cmds) < other_count:
        external_cmds.append(external_cmds[-1])
    external_cmds = external_cmds[:other_count]

    rng = secrets.SystemRandom() if args.seed == 0 else None

    wins = {p: 0 for p in range(args.num_players)}
    for g in range(args.games):
        seed = rng.randint(1, 2**63 - 1) if rng else (args.seed + g)
        result = play_one_game(
            num_players=args.num_players, my_seats=my_seats,
            external_cmds=external_cmds, base=args.base,
            seed=seed, verbose=args.verbose,
        )
        bh = result["bullhead_counts"]
        winner = result["winner"]
        if 0 <= winner < args.num_players:
            wins[winner] += 1
        print(f"Game {g+1}/{args.games} seed={seed} bullheads={bh} winner=seat#{winner}")
        for p, ms in result["rest_decide_avg_ms"].items():
            print(f"  DinoBoard seat#{p} decide avg {ms:.1f} ms")
        for p, ms in result["ext_decide_avg_ms"].items():
            print(f"  External seat#{p} decide avg {ms:.1f} ms")

    print("\n=== Final ===")
    for p, w in wins.items():
        tag = "DinoBoard" if p in my_seats else "External"
        print(f"  seat#{p} ({tag}): {w} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
