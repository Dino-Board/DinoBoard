"""Independent recomputation of the 6 nimmt! token encoding.

Works from what one player can see (`view`, see nimmt_game_model.Game.view),
written against the rulebook rather than the C++ encoder: sets and direct
counts instead of prefix sums, the printed bullhead table, a replay of the
round for the row previews.

    view = {
        "me": seat,
        "players": 2..4,
        "round": 0..10,                # rounds finished
        "choosing_row": bool,          # a player is choosing a row to take
        "chooser": seat or -1,
        "to_move": bool,               # I have a decision to make now
        "hand": [cards],               # mine
        "table": [[cards], ...],       # four rows, first card first
        "gone": {cards},               # every card shown
        "shown": {seat: card},         # played cards I can see, not yet placed
        "heads": [by seat],            # bullheads taken
    }
"""
from __future__ import annotations

import numpy as np

from nimmt_game_model import HAND_SIZE, NUM_CARDS, ROW_CAPACITY, heads_of, margin

CARDS = range(1, NUM_CARDS + 1)


def _onehot(i: int, n: int) -> list:
    return [1.0 if k == i else 0.0 for k in range(n)]


def _play_out(table: dict, taken_row: int, chooser: int, shown: dict, players: int) -> list:
    """Bullheads each player takes in the rest of the round if the chooser takes `taken_row`."""
    table = {i: list(cards) for i, cards in table.items()}
    gain = [0] * players
    gain[chooser] += sum(map(heads_of, table[taken_row]))
    table[taken_row] = [shown[chooser]]
    for card, owner in sorted((c, p) for p, c in shown.items() if p != chooser):
        i = max((i for i, cards in table.items() if cards[-1] < card), key=lambda i: table[i][-1])
        if len(table[i]) == ROW_CAPACITY:
            gain[owner] += sum(map(heads_of, table[i]))
            table[i] = [card]
        else:
            table[i].append(card)
    return gain


def expected_tokens(view: dict) -> list:
    """Tokens in encoder order: dicts {type, anchor, pos, feats}."""
    me, n = view["me"], view["players"]
    hand = sorted(view["hand"])
    table = {i: list(cards) for i, cards in enumerate(view["table"]) if cards}
    shown = dict(view["shown"])
    choosing = view["choosing_row"]
    heads = view["heads"]

    # Cards that can still turn up: not shown, not mine.
    seen = set(view["gone"]) | set(hand) | set(shown.values())
    unseen = [c for c in CARDS if c not in seen]
    each = max(0, HAND_SIZE - view["round"] - (1 if choosing else 0))  # cards each other player holds face down
    density = (n - 1) * each / len(unseen) if unseen else 0.0

    def unseen_in(lo, hi):
        return [c for c in unseen if lo < c < hi]

    def f_unseen(k):
        return min(k, 20) / 20

    def f_expected(k):
        return min(k * density, 5) / 5

    tails = sorted(cards[-1] for cards in table.values())
    row_of_tail = {cards[-1]: i for i, cards in table.items()}
    row_heads = {i: sum(map(heads_of, cards)) for i, cards in table.items()}
    cheapest = min(row_heads.values())

    tokens = []
    low = len(unseen_in(0, tails[0]))
    tokens.append({"type": "global", "anchor": 0, "pos": None, "feats": {
        "round": view["round"] / HAND_SIZE,
        "choosing_row": float(choosing),
        "to_move": float(view["to_move"]),
        "unseen": len(unseen) / NUM_CARDS,
        "opp_cards": each / HAND_SIZE,
        "density": density,
        "low_unseen": f_unseen(low),
        "low_expected": f_expected(low),
        "min_row_heads": cheapest / 10,
    }})

    for offset in range(n):
        p = (me + offset) % n
        tokens.append({"type": "player", "anchor": 0, "pos": None, "feats": {
            "seat": _onehot(offset, 4),
            "hand_heads": min(heads[p], 30) / 30,
            "margin": margin(heads, p),
            "shown": float(p in shown),
            "chooser": float(choosing and view["chooser"] == p),
        }})

    for i, cards in table.items():
        tail = cards[-1]
        order = tails.index(tail)
        nxt = tails[order + 1] if order + 1 < len(tails) else NUM_CARDS + 1
        reach = unseen_in(tail, nxt)
        feats = {
            "tail": tail / NUM_CARDS,
            "length": _onehot(len(cards) - 1, 5),
            "space": (ROW_CAPACITY - len(cards)) / 4,
            "heads": row_heads[i] / 10,
            "cheapest": float(row_heads[i] == cheapest),
            "order": _onehot(order, 4),
            "reach_unseen": f_unseen(len(reach)),
            "reach_expected": f_expected(len(reach)),
            "reach_heads": min(sum(map(heads_of, reach)), 40) / 40,
            "reach_mine": sum(tail < c < nxt for c in hand) / 5,
            "reach_shown": sum(tail < c < nxt for c in shown.values()) / 3,
            "if_taken_mine": 0.0,
            "if_taken_others": 0.0,
        }
        if choosing and view["chooser"] in shown:
            gain = _play_out(table, i, view["chooser"], shown, n)
            feats["if_taken_mine"] = gain[me] / 10
            feats["if_taken_others"] = (sum(gain) - gain[me]) / 10
        tokens.append({"type": "row", "anchor": NUM_CARDS + 1 + i, "pos": (len(unseen_in(0, tail)), order + 1),
                       "feats": feats})

    def card_token(card, owner, held, rank):
        below = len(unseen_in(0, card))
        row_class = sum(t < card for t in tails)
        feats = {
            "in_hand": float(held),
            "my_shown": float(not held and owner == me),
            "opp_shown": float(not held and owner != me),
            "owner": _onehot((owner - me) % n, 4),
            "resolving": float(not held and choosing and view["chooser"] == owner),
            "value": card / NUM_CARDS,
            "heads": heads_of(card) / 7,
            "hand_rank": rank / 9,
            "percentile": below / len(unseen) if unseen else 0.0,
            "below_expected": f_expected(below),
            "small": 0.0, "small_cost": 0.0, "gap_safe": 0.0, "gap_unseen": 0.0, "gap_expected": 0.0,
            "gap_mine": 0.0, "gap_shown": 0.0, "row_length": [0.0] * 5, "row_space": 0.0, "row_heads": 0.0,
            "sixth": 0.0, "cost_now": 0.0,
        }
        if row_class == 0:
            feats.update(small=1.0, small_cost=cheapest / 10, cost_now=cheapest / 10)
        else:
            tail = tails[row_class - 1]
            cards = table[row_of_tail[tail]]
            gap = len(unseen_in(tail, card))
            full = len(cards) == ROW_CAPACITY
            feats.update(
                gap_safe=float(gap == 0), gap_unseen=f_unseen(gap), gap_expected=f_expected(gap),
                gap_mine=sum(tail < c < card for c in hand) / 5,
                gap_shown=sum(tail < c < card for c in shown.values()) / 3,
                row_length=_onehot(len(cards) - 1, 5), row_space=(ROW_CAPACITY - len(cards)) / 4,
                row_heads=row_heads[row_of_tail[tail]] / 10, sixth=float(full),
                cost_now=row_heads[row_of_tail[tail]] / 10 if full else 0.0)
        return {"type": "card", "anchor": card if held else 0, "pos": (below, row_class), "feats": feats}

    for rank, card in enumerate(hand):
        tokens.append(card_token(card, me, True, rank))
    for card, owner in sorted((c, p) for p, c in shown.items()):
        tokens.append(card_token(card, owner, False, 0))
    return tokens


def vectorize(token: dict, layout: dict) -> np.ndarray:
    """The token as the encoder writes it: header, then its type's feature table."""
    names = [t["name"] for t in layout["types"]]
    kind = names.index(token["type"])
    out = np.zeros(layout["token_dim"], dtype=np.float64)
    out[0] = 1.0
    out[2] = token["anchor"]
    out[3] = kind
    if token["pos"] is not None:
        out[4], out[5], out[6] = 1.0, token["pos"][0], token["pos"][1]
    table = {name: (off, width) for name, off, width in layout["types"][kind]["features"]}
    assert set(table) == set(token["feats"]), (token["type"], set(table) ^ set(token["feats"]))
    for name, value in token["feats"].items():
        off, width = table[name]
        out[layout["header"] + off:layout["header"] + off + width] = value
    return out


def differences(x: np.ndarray, tokens: list, layout: dict, tol: float = 1e-5) -> list:
    """Human-readable mismatches between an encoding and the expected tokens."""
    if len(x) != len(tokens):
        return [f"{len(x)} tokens, expected {len(tokens)}"]
    problems = []
    fields = layout["header_fields"]
    for i, (got, token) in enumerate(zip(x, tokens)):
        want = vectorize(token, layout)
        kind = [t["name"] for t in layout["types"]].index(token["type"])
        for j in np.flatnonzero(np.abs(got - want) > tol):
            if j < layout["header"]:
                name = fields[j]
            else:
                name = next((f"{n}[{j - layout['header'] - o}]" for n, o, w in layout["types"][kind]["features"]
                             if o <= j - layout["header"] < o + w), f"unused[{j}]")
            problems.append(f"token {i} ({token['type']}, anchor {token['anchor']}) {name}: "
                            f"got {got[j]:.5f}, expected {want[j]:.5f}")
    return problems
