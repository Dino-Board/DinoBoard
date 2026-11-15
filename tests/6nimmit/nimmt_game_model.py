"""A 6 nimmt! game (one hand), written from the rules alone, as a table of the game would keep it.

The model is told the deal (all hands, the four row starters) and every
action, and keeps everything else itself: rows, bullheads, the cards that
have been shown, whose card is face down or face up, who has to choose a
row, and when the game is over. It never asks the engine for any of it, so
comparing the two at every ply checks the engine, and views built from it
(`view`) check the encoder against what a player could have written down.
"""
from __future__ import annotations

NUM_CARDS, NUM_ROWS, ROW_CAPACITY, HAND_SIZE = 104, 4, 5, 10
MARGIN_SCALE = 25.0
GAMES = {"6nimmit": 2, "6nimmit_3p": 3, "6nimmit_4p": 4}


def deal_size(players: int) -> int:
    return NUM_ROWS + players * HAND_SIZE


def heads_of(card: int) -> int:
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 5 == 0:
        return 2
    return 1


def margin(heads: list, p: int) -> float:
    """The other players' average bullheads minus p's, in units of MARGIN_SCALE, clipped to +-1."""
    n = len(heads)
    others = (sum(heads) - heads[p]) / (n - 1)
    return max(-1.0, min(1.0, (others - heads[p]) / MARGIN_SCALE))


class Game:
    def __init__(self, hands, starters):
        self.n = len(hands)
        dealt = [c for hand in hands for c in hand] + list(starters)
        assert all(len(h) == HAND_SIZE for h in hands) and len(starters) == NUM_ROWS
        assert len(set(dealt)) == deal_size(self.n) and all(1 <= c <= NUM_CARDS for c in dealt)
        self.hands = [sorted(h) for h in hands]
        self.table = [[c] for c in starters]
        self.gone = set(starters)  # shown so far
        self.heads = [0] * self.n
        self.round = 0
        self.face_down = {}  # seat -> card played, the other seats cannot see it
        self.face_up = {}    # seat -> card turned over, not placed yet
        self.chooser = -1    # seat choosing a row to take
        self.terminal = False

    # ---- whose move ----
    def to_move(self, seat: int) -> bool:
        if self.terminal:
            return False
        if self.chooser >= 0:
            return seat == self.chooser
        return seat not in self.face_down

    def actor(self) -> int:
        """The seats play in seat order; the engine asks them one at a time."""
        if self.chooser >= 0:
            return self.chooser
        return next(p for p in range(self.n) if p not in self.face_down)

    def legal(self) -> list:
        if self.terminal:
            return []
        if self.chooser >= 0:
            return [NUM_CARDS + r for r in range(NUM_ROWS)]
        return [c - 1 for c in self.hands[self.actor()]]

    # ---- an action of the seat to move ----
    def apply(self, action: int):
        assert action in self.legal()
        if self.chooser >= 0:
            seat = self.chooser
            self._take(seat, action - NUM_CARDS, self.face_up.pop(seat))
            self.chooser = -1
        else:
            seat = self.actor()
            self.hands[seat].remove(action + 1)
            self.face_down[seat] = action + 1
            if len(self.face_down) < self.n:
                return
            self.face_up, self.face_down = self.face_down, {}
            self.gone |= set(self.face_up.values())
        self._place()

    def _take(self, seat, row, card):
        self.heads[seat] += sum(map(heads_of, self.table[row]))
        self.table[row] = [card]

    def _place(self):
        while self.face_up:
            seat = min(self.face_up, key=self.face_up.get)
            card = self.face_up[seat]
            below = [r for r in range(NUM_ROWS) if self.table[r][-1] < card]
            if not below:
                self.chooser = seat
                return
            row = max(below, key=lambda r: self.table[r][-1])
            del self.face_up[seat]
            if len(self.table[row]) == ROW_CAPACITY:
                self._take(seat, row, card)
            else:
                self.table[row].append(card)
        self.round += 1
        self.terminal = self.round == HAND_SIZE

    # ---- the result ----
    @property
    def losers(self) -> list:
        """The players with the most bullheads; nobody when all are level."""
        if not self.terminal or len(set(self.heads)) == 1:
            return []
        return [p for p in range(self.n) if self.heads[p] == max(self.heads)]

    @property
    def winners(self) -> list:
        return [p for p in range(self.n) if p not in self.losers] if self.losers else []

    @property
    def winner(self) -> int:
        """The only player with the fewest bullheads, else -1."""
        if not self.terminal:
            return -1
        best = [p for p in range(self.n) if self.heads[p] == min(self.heads)]
        return best[0] if len(best) == 1 else -1

    def values(self) -> list:
        return [margin(self.heads, p) for p in range(self.n)]

    # ---- what one seat can see ----
    def view(self, me: int) -> dict:
        shown = dict(self.face_up)
        if me in self.face_down:
            shown[me] = self.face_down[me]
        return {
            "me": me, "players": self.n, "round": self.round, "choosing_row": self.chooser >= 0,
            "chooser": self.chooser, "to_move": self.to_move(me), "hand": list(self.hands[me]),
            "table": [list(r) for r in self.table], "gone": set(self.gone), "shown": shown,
            "heads": list(self.heads),
        }


def deal_of(d: dict):
    """(hands, starters) of a freshly dealt truth state dict."""
    return [p["hand"] for p in d["players"]], [row[0] for row in d["table"]]


def drive(engine, game: str, seed: int, policy, max_plies: int = 200):
    """Play one game on the engine with the model alongside.

    Yields (session, model, state dict) before every action and once at the
    end. `policy(session, state dict, legal, rng)` returns the action.
    """
    import random
    gs = engine.GameSession(game, seed=seed)
    rng = random.Random(seed)
    d = gs.get_state_dict()
    model = Game(*deal_of(d))
    for _ in range(max_plies):
        yield gs, model, d
        if gs.is_terminal:
            return
        action = policy(gs, d, gs.get_legal_actions(), rng)
        gs.apply_action(action)
        model.apply(action)
        d = gs.get_state_dict()
    raise AssertionError(f"game not over after {max_plies} plies")


def random_policy(gs, d, legal, rng):
    return rng.choice(legal)


def heuristic_policy(name: str, epsilon: float = 0.0):
    def policy(gs, d, legal, rng):
        if epsilon and rng.random() < epsilon:
            return rng.choice(legal)
        return gs.get_heuristic_action(name, 0.0)["action"]
    return policy
