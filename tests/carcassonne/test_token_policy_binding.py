"""Where each policy logit of the token transformer comes from, on real
Carcassonne encodings: every legal action is scored by exactly one pointer
head of one token (or by the global head for "no meeple"), and no two
actions read the same logit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from training.model import create_model_from_config, padded_token_count

HEADS = 4 + 13
MEEPLE_PASS = 1600
GLOBAL, CURRENT, TILE, FRONTIER = range(4)
NET = {"type": "token_transformer", "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64,
       "relative_position_radius": 4, "anchor_embedding": False}


GAME = "carcassonne_tok"


def test_every_legal_action_has_its_own_logit():
    meta = engine.game_metadata(GAME)
    lay = meta["token_layout"]
    td = lay["token_dim"]
    torch.manual_seed(3)
    net = create_model_from_config(NET, input_dim=meta["feature_dim"],
                                   policy_dim=meta["action_space"], num_players=2, token_layout=lay).eval()
    captured = {}
    net.pointer_head.register_forward_hook(lambda m, i, o: captured.__setitem__("ptr", o.detach()))
    net.global_policy.register_forward_hook(lambda m, i, o: captured.__setitem__("glob", o.detach()))

    # Layout: one binding per action.
    anchor_of = {a: i for i, a in enumerate(lay["anchor_action"]) if a >= 0}
    static = {a for a, t in enumerate(lay["action_token"]) if t >= 0}
    assert len(anchor_of) == sum(1 for a in lay["anchor_action"] if a >= 0), "an action bound to two anchors"
    assert not static & set(anchor_of)
    assert set(range(meta["action_space"])) - static - set(anchor_of) == {MEEPLE_PASS}

    ep = engine.run_heuristic_episode(GAME, 17, 0.0, 200, "rule")
    states = sources_seen = 0
    for smp in ep["samples"][::7]:
        x = np.asarray(smp["features"], dtype=np.float32).reshape(-1, td)
        padded = np.zeros((padded_token_count(len(x)), td), dtype=np.float32)
        padded[:len(x)] = x
        x = padded
        with torch.no_grad():
            policy = net(torch.from_numpy(x.reshape(1, -1)))[0][0].numpy()
        ptr, glob = captured["ptr"][0].numpy(), captured["glob"][0].numpy()
        token_at_anchor = {}
        for i, t in enumerate(x):
            if t[0] == 1 and t[2] > 0:
                assert int(t[2]) not in token_at_anchor, "two valid tokens share an anchor"
                token_at_anchor[int(t[2])] = i
        used = set()
        for a in np.nonzero(np.asarray(smp["legal_mask"]))[0].tolist():
            if a == MEEPLE_PASS:
                assert policy[a] == glob[a]
                source = ("global",)
            elif a in static:
                token = lay["action_token"][a]
                assert int(x[token, 3]) == CURRENT and x[token, 0] == 1
                source = (token, lay["action_head"][a])
                assert policy[a] == ptr[1 + token, lay["action_head"][a]]
            else:
                anchor, head = anchor_of[a] // HEADS + 1, anchor_of[a] % HEADS
                assert anchor in token_at_anchor, f"legal action {a} has no token to score it"
                token = token_at_anchor[anchor]
                assert int(x[token, 3]) == (FRONTIER if a < MEEPLE_PASS else CURRENT)
                source = (token, head)
                assert policy[a] == ptr[1 + token, head]
            assert source not in used, f"two legal actions read logit {source}"
            used.add(source)
        states += 1
        sources_seen += len(used)
    assert states >= 15 and sources_seen > 150
