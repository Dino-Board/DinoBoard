"""Ark Nova factored actions: one meaning per atom, order-free setup discards, long public lists, and
where the network reads each logit from."""
from __future__ import annotations

import collections
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from conftest import get_test_model

sys.path.insert(0, str(Path(__file__).parent))
from arknova_checks import binding  # noqa: E402
from arknova_reference import GAME, atom_index, atom_names, card_index, cards, decode, own_slots  # noqa: E402

TAKE, GIVE, PICK = "take", "give", "pick"  # gain / play / keep it; give it up; a choice of neither kind
# Every atom class (the part of the atom's name before ':') that can be legal in each decision
# frame, and what choosing it means there. An atom class is one family of logits.
MEANING = {
    "setup_keep": {"hand_discard": GIVE},
    "turn": {"act": PICK, "xtoken": PICK},
    "extra_action": {"act": PICK, "skip": PICK},
    "multiplier_repeat": {"repeat": PICK, "skip": PICK},
    "act_cards": {"deck": TAKE, "display": TAKE, "snap": TAKE},
    "take_card": {"deck": TAKE, "display": TAKE},
    "discard_hand": {"hand_discard": GIVE},
    "act_build": {"build": PICK, "engineer": PICK, "cell": PICK, "done": PICK},
    "free_build": {"place": PICK, "cell": PICK, "skip": PICK},
    "sponsor_building": {"cell": PICK, "place": PICK},  # place: a choice only next to the harbour
    "act_animals": {"hand": TAKE, "display": TAKE, "enclosure": PICK, "done": PICK},
    "act_sponsors": {"hand": TAKE, "display": TAKE, "break": PICK, "done": PICK},
    "paid_sponsor": {"hand": TAKE, "skip": PICK},
    "act_association": {"reputation": PICK, "partner_zoo": TAKE, "university": TAKE, "project": PICK, "hand": TAKE,
                        "display": TAKE, "position": PICK, "map_token": PICK, "zoo_give": GIVE, "donate": PICK,
                        "done": PICK},
    "gain_partner_zoo": {"partner_zoo": TAKE},
    "gain_university": {"university": TAKE},
    "upgrade": {"upgrade": PICK},
    "upgrade_or_worker": {"upgrade": PICK, "worker": PICK},
    "conservation_tile": {"money": TAKE, "tile": TAKE},
    "clever": {"clever": PICK, "skip": PICK},
    "place_multiplier": {"multiplier": PICK},
    "discard_final": {"final_discard": GIVE},
    "ab_sun": {"hand_discard": GIVE, "done": PICK},
    "ab_pouch": {"hand_discard": GIVE, "done": PICK},
    "ab_dig": {"hand_discard": GIVE, "display_discard": GIVE, "done": PICK},
    "ab_snap": {"display": TAKE, "replenish": PICK, "skip": PICK},
    "ab_pick": {"private": TAKE},
    "ab_resist": {"private": TAKE},
    "ab_hunter": {"list": TAKE, "skip": PICK},
    "ab_project": {"list": TAKE, "skip": PICK},
    "ab_boost": {"boost": PICK, "skip": PICK},
    "ab_pilfer": {"give_card": GIVE, "give_money": GIVE},
    "move_animals": {"zoo": PICK, "done": PICK},
    "okapi_stable": {"yes": PICK, "no": PICK},
    "waza_kind": {"kind": PICK},
}
# The harbour (Map 4) is offered on top of any decision of the player whose turn it is: "harbor",
# then the hand card to discard for 3 money.
HARBOR = {"harbor": PICK, "hand_discard": GIVE}


def _core(gs):
    return json.loads(gs.get_state_dict()["core_json"])


def test_no_atom_means_both_taking_and_giving():
    """F1. The declared meanings never put "take" and "give" on one atom class, and the table is the
    complete list of what the rules offer: over 75 random games (on random maps) exactly these
    (frame, atom class) pairs are legal."""
    by_class = collections.defaultdict(set)
    for classes in list(MEANING.values()) + [HARBOR]:
        for cls, meaning in classes.items():
            by_class[cls].add(meaning)
    mixed = {cls: m for cls, m in by_class.items() if {TAKE, GIVE} <= m}
    assert not mixed, f"atom classes used for both taking and giving: {mixed}"

    names = atom_names()
    seen, harbor = set(), set()
    for seed in range(75):
        gs = engine.GameSession(GAME, seed=seed)
        rng = random.Random(seed)
        while not gs.is_terminal:
            state = gs.get_state_dict()
            frame = json.loads(state["core_json"])["stack"][-1]["kind"]
            after_harbor = state["prefix"].split() == ["harbor"]
            legal = gs.get_legal_actions()
            for a in legal:
                cls = names[a].split(":")[0]
                if cls == "harbor" or after_harbor:
                    assert cls in HARBOR, f"atom {names[a]} after the harbour"
                    harbor.add(cls)
                    continue
                assert cls in MEANING[frame], f"atom {names[a]} is legal in frame {frame}: declare its meaning"
                seen.add((frame, cls))
            gs.apply_action(rng.choice(legal))
    declared = {(frame, cls) for frame, classes in MEANING.items() for cls in classes}
    assert seen == declared, f"declared but never legal in these games: {sorted(declared - seen)}"
    assert harbor == set(HARBOR), f"the harbour was never offered in these games: {sorted(harbor)}"


def test_card_atoms_do_what_their_class_says():
    """F1, behaviour: after a "take" atom of a card the card is in my hand, my zoo or in play; after
    a "give" atom it is in the discard pile or tucked; `money` gains money and `give_money` loses it."""
    names, idx, index = atom_names(), card_index(), atom_index()
    done = collections.Counter()
    for seed in range(12):
        gs = engine.GameSession(GAME, seed=seed)
        rng = random.Random(seed)
        pending = []  # (atom class, frame, mover, card index or None, facts before)
        while not gs.is_terminal:
            core = _core(gs)
            me, frame = gs.current_player, core["stack"][-1]
            action = rng.choice(gs.get_legal_actions())
            if frame["kind"] == "ab_pilfer":  # rare: take both answers in turn
                action = index["give_money" if done["give_money"] <= done["give_card"] else "give_card"]
                assert action in gs.get_legal_actions()
            cls, _, arg = names[action].partition(":")
            own = {p: own_slots(gs, p) for p in (0, 1)}
            card = None
            if cls in ("hand", "hand_discard", "zoo_give"):
                card = int(arg)
            elif cls in ("display", "display_discard", "snap"):
                card = idx[core["display"][int(arg)]]
            elif cls == "final_discard":
                card = own[me]["final"][int(arg)]
            elif cls == "private":
                k = sum(f["kind"] in ("ab_pick", "ab_resist") and f["player"] == me for f in core["stack"][:-1])
                card = own[me]["private"][k][int(arg)]
            elif cls == "list" and frame["kind"] == "ab_hunter":
                card = idx[frame["data"]["cards"][int(arg)]]
            pending.append((cls, frame["kind"], me, card, core, own))
            gs.apply_action(action)
            if gs.get_state_dict()["prefix"]:
                continue  # the option is not complete yet
            after = _core(gs)
            own_after = {p: own_slots(gs, p) for p in (0, 1)}
            reshuffled = len(after["deck"]) > len(core["deck"])
            harbor = pending[0][0] == "harbor"  # the card goes to the discard pile whatever the decision
            for cls, kind, p, card, before, own_before in pending:
                mine, cid = after["players"][p], cards()[card]["id_str"] if card is not None else None
                in_play = card is not None and (cid in mine["zoo_cards"] or cid in after["projects"])
                if cls in ("hand", "display", "snap", "private", "list") and card is not None:
                    kept = card in own_after[p]["hand"] or card in own_after[p]["final"] or in_play
                    assert kept, f"{cls} in {kind}: card {cid} was not gained (seed {seed})"
                    assert cid not in after["discard"] and card not in own_after[p]["tucked"]
                elif cls in ("hand_discard", "display_discard"):
                    assert card not in own_after[p]["hand"] and not in_play
                    if kind == "ab_pouch" and not harbor:
                        assert card in own_after[p]["tucked"]
                    elif not reshuffled:
                        assert cid in after["discard"], f"{cls} in {kind}: card {cid} is not on the discard pile"
                elif cls == "final_discard":
                    assert card not in own_after[p]["final"]
                    assert len(after["final_deck"]) == len(before["final_deck"]) + 1
                elif cls == "zoo_give":
                    tokens = before["players"][p]["card_tokens"].get(cid)
                    if cid in mine["zoo_cards"]:  # a breeding card: one of its tokens is spent
                        assert mine["card_tokens"][cid] == tokens - 1
                    else:  # a released animal
                        assert cid in after["discard"] or reshuffled
                elif cls in ("money", "give_money") and after["turn"]:
                    # Still inside the turn, so nothing but the choice itself moved money.
                    gain = mine["money"] - before["players"][p]["money"]
                    if cls == "money":
                        assert gain == 5
                    else:
                        paid = min(5, before["players"][p]["money"])
                        assert gain == -paid
                        assert after["players"][1 - p]["money"] - before["players"][1 - p]["money"] == paid
                elif cls == "give_card":
                    lost = own_before[p]["hand"] - own_after[p]["hand"]
                    assert len(lost) == 1 and lost <= own_after[1 - p]["hand"]
                else:
                    continue
                done[cls] += 1
            pending = []
    for cls in ("hand", "hand_discard", "display", "display_discard", "snap", "private", "list", "final_discard",
                "money", "give_money", "give_card"):
        assert done[cls], f"no {cls} atom was exercised"


def test_setup_discards_can_be_named_in_any_order():
    """F6. The 4 setup discards are a set: every hand card is a legal discard at once, any order of
    the same 4 gives the same states on the way and the same hand at the end, and every 4 of the 8
    cards can be discarded."""
    names, index = atom_names(), atom_index()
    seed = 7
    gs = engine.GameSession(GAME, seed=seed)
    p = gs.current_player
    hand = sorted(own_slots(gs, p)["hand"])
    assert len(hand) == 8 and _core(gs)["stack"][-1]["kind"] == "setup_keep"

    def legal(g):
        return {names[a] for a in g.get_legal_actions()}

    assert legal(gs) == {f"hand_discard:{c}" for c in hand}

    discards = [hand[6], hand[1], hand[4], hand[3]]
    partial = collections.defaultdict(set)  # the discards so far -> the states reached with them
    finals = set()
    for order in itertools.permutations(discards):
        g = engine.GameSession(GAME, seed=seed)
        for k, c in enumerate(order):
            assert legal(g) == {f"hand_discard:{x}" for x in hand if x not in order[:k]}
            g.apply_action(index[f"hand_discard:{c}"])
            if k < 3:
                state = g.get_state_dict()
                partial[frozenset(order[:k + 1])].add((g.state_hash_for_perspective(p), state["prefix"]))
        assert own_slots(g, p)["hand"] == set(hand) - set(discards)
        assert {cards()[c]["id_str"] for c in discards} <= set(_core(g)["discard"])
        finals.add((g.state_hash_for_perspective(0), g.state_hash_for_perspective(1), g.get_state_dict()["core_json"]))
    assert len(finals) == 1, "the order of the setup discards changed the result"
    assert all(len(states) == 1 for states in partial.values()), "the same discards in another order are another state"

    kept = set()
    for combo in itertools.combinations(hand, 4):
        g = engine.GameSession(GAME, seed=seed)
        for c in reversed(combo):
            g.apply_action(index[f"hand_discard:{c}"])
        kept.add(frozenset(own_slots(g, p)["hand"]))
        assert own_slots(g, p)["hand"] == set(hand) - set(combo)
    assert len(kept) == 70


def _pack_deck_index(pack):
    """Index of the deck size in the packed public vector (the layout of ArknovaState::pack_public)."""
    i = 3                       # after the version, the map and the harbour flag
    for _ in range(2):
        i += 7 + 5 + 4          # track values, action card slots, workers per task
        i += 1 + pack[i]        # partner zoos
        i += 1 + pack[i]        # universities
        i += 7 + 1              # map tokens, hand size
        i += 1 + pack[i]        # hand cards shown to the opponent
        i += 1 + pack[i]        # maybe tucked
        i += 1 + pack[i]        # zoo cards
        i += 1                  # number of final cards
        i += 1 + 8 * pack[i]    # buildings
        i += 1 + 3 * pack[i]    # housing
        for _ in range(5):
            i += 1 + pack[i]    # tokens on each action card
        i += 1 + 2 * pack[i]    # card tokens
        i += 3
    return i + 1                # after the player to move


FRAME_INTS = 2 + 32  # kind, player, kFrameData values


def _hunter_position(max_seed=40):
    """A random game played until a Hunter reveal is the pending decision."""
    for seed in range(max_seed):
        gs = engine.GameSession(GAME, seed=seed)
        rng = random.Random(seed)
        while not gs.is_terminal:
            if _core(gs)["stack"][-1]["kind"] == "ab_hunter":
                return gs
            gs.apply_action(rng.choice(gs.get_legal_actions()))
    raise AssertionError("no Hunter reveal in these games")


def test_public_list_longer_than_twelve_cards():
    """F8. A Hunter reveal of 16 cards (Spotted Hyena Compound with many predators): every animal of
    the list can be taken and the search runs. The position is a real Hunter decision whose list is
    extended, for the mover's view, by cards that are in the deck."""
    names, index, table, idx = atom_names(), atom_index(), cards(), card_index()
    truth = _hunter_position()
    p = truth.current_player
    core = _core(truth)
    frame = core["stack"][-1]
    revealed = [idx[c] for c in frame["data"]["cards"]]
    # Cards nobody holds and no public zone shows: they are in the deck.
    located = set(revealed) | {idx[c] for c in core["display"] + core["discard"] + core["base_projects"] +
                               core["base_project_deck"] + core["projects"] if c is not None}
    for q in (0, 1):
        own = own_slots(truth, q)
        located |= own["hand"] | set(own["final"]) | set(own["tucked"]) | {c for f in own["private"] for c in f}
        located |= {idx[c] for c in core["players"][q]["zoo_cards"]}
    in_deck = [c for c in range(len(table)) if c not in located and table[c]["type"] != "endgame"]
    assert len(in_deck) == len(core["deck"])
    animals = [c for c in in_deck if table[c]["type"] == "animal"]
    others = [c for c in in_deck if table[c]["type"] != "animal"]
    extra = (others[:4] + animals)[:16 - len(revealed)]
    cards_now = revealed + extra
    assert len(cards_now) == 16 and table[cards_now[15]]["type"] == "animal"

    obs = truth.extract_initial_observation(p)
    pack = obs["public_snapshot"]["public"][1]
    assert pack[0] == 0xA45, "the packed public state changed: update _pack_deck_index"
    deck_at = _pack_deck_index(pack)
    assert pack[deck_at] == len(core["deck"])
    n_prefix = len(truth.get_state_dict()["prefix"].split())
    top = len(pack) - 1 - n_prefix - FRAME_INTS
    assert pack[top - FRAME_INTS * (len(core["stack"]) - 1) - 1] == len(core["stack"]) and pack[top + 1] == p
    assert pack[top + 2] == len(revealed) and pack[top + 3:top + 3 + len(revealed)] == revealed
    pack[top + 2] = len(cards_now)
    pack[top + 3:top + 3 + len(cards_now)] = cards_now
    pack[deck_at] -= len(extra)

    view = engine.GameSession(GAME, seed=4242, model_path=get_test_model(GAME))
    view.apply_initial_observation(p, obs)
    legal = {names[a] for a in view.get_legal_actions()}
    assert legal == {f"list:{j}" for j, c in enumerate(cards_now) if table[c]["type"] == "animal"} | {"skip"}
    assert "list:15" in legal
    choice = view.get_ai_action(simulations=24, temperature=0.0)
    assert choice["action_info"]["name"] in legal
    view.apply_action(index["list:15"])
    after = _core(view)
    assert table[cards_now[15]]["id_str"] in after["players"][p]["public_hand"]
    assert {table[c]["id_str"] for c in cards_now[:15]} <= set(after["discard"])


def test_network_reads_each_logit_from_its_token():
    """The network built the way the training CLI builds it (train.json's network block, the
    engine's token layout) reads every legal pointer atom from the pointer head of the one token
    carrying its anchor, every other atom from the global head, and no two legal atoms from the same
    place."""
    from training.cli import find_train_config
    from training.model import create_model_from_config
    from training.pipeline import pad_token_rows

    meta = engine.game_metadata(GAME)
    train_cfg, _ = find_train_config(GAME)
    torch.manual_seed(3)
    net = create_model_from_config(train_cfg["network"], input_dim=meta["feature_dim"],
                                   policy_dim=meta["action_space"], num_players=meta["num_players"],
                                   token_layout=meta["token_layout"]).eval()
    seen = {}
    net.pointer_head.register_forward_hook(lambda m, i, out: seen.__setitem__("pointer", out))
    net.global_policy.register_forward_hook(lambda m, i, out: seen.__setitem__("global", out))
    bind, names = binding(), atom_names()
    ep = engine.run_heuristic_episode(game_id=GAME, seed=9, temperature=0.0, max_game_plies=3000,
                                      heuristic_selector="random")
    pointer_atoms = global_atoms = 0
    for s in ep["samples"][::7]:
        row = np.asarray(s["features"], dtype=np.float32)
        tokens = decode(row)
        x = torch.from_numpy(pad_token_rows([row], meta["token_layout"]["token_dim"]))
        with torch.no_grad():
            policy, _ = net(x)
        position = {t.anchor: i for i, t in enumerate(tokens) if t.anchor}
        sources = set()
        for a in np.nonzero(np.asarray(s["legal_mask"]))[0].tolist():
            if a in bind:
                anchor, head = bind[a]
                source = ("token", position[anchor], head)
                want = seen["pointer"][0, 1 + position[anchor], head]  # row 0 is the CLS token
                pointer_atoms += 1
            else:
                source = ("global", a)
                want = seen["global"][0, a]
                global_atoms += 1
            assert policy[0, a] == want, f"logit of {names[a]} does not come from {source}"
            sources.add(source)
        assert len(sources) == int(np.count_nonzero(s["legal_mask"]))
    assert pointer_atoms > 100 and global_atoms > 100


def _belief_model(tmp_path, seed=11):
    from training.model import create_belief_model_from_config, export_belief_onnx

    meta = engine.game_metadata(GAME)
    cfg = json.loads((Path(__file__).parents[2] / "games/arknova/config/belief.json").read_text())
    torch.manual_seed(seed)
    net = create_belief_model_from_config(cfg, input_dim=meta["belief_feature_dim"],
                                          output_dim=meta["belief_logit_count"]).eval()
    path = tmp_path / f"belief_{seed}.onnx"
    export_belief_onnx(net, path, meta["belief_feature_dim"])
    return str(path)


def test_search_samples_worlds_from_its_own_posterior(tmp_path):
    """The other half of F10: at the root it was computed for, the posterior is used. Two belief
    nets with different weights make the same search sample different worlds, so some visit counts
    differ; without a belief net they are reproducible."""
    model = get_test_model(GAME)
    sessions = [engine.GameSession(GAME, seed=5, model_path=model, belief_model_path=_belief_model(tmp_path, s))
                for s in (11, 12)]
    plain = [engine.GameSession(GAME, seed=5, model_path=model) for _ in range(2)]
    rng = random.Random(5)
    differ = searches = 0
    for ply in range(120):
        legal = sessions[0].get_legal_actions()
        if ply % 6 == 0 and len(legal) > 1:
            visits = [g.get_ai_action(simulations=48, temperature=0.0)["stats"]["root_action_visits"]
                      for g in sessions + plain]
            assert visits[2] == visits[3]
            differ += visits[0] != visits[1]
            searches += 1
        action = rng.choice(legal)
        for g in sessions + plain:
            g.apply_action(action)
    assert searches >= 10 and differ > 0, "the belief net's posterior never changed a search"


def test_belief_posterior_is_not_reused_for_a_later_position(tmp_path):
    """F10. A search caches the belief net's posterior on the seat's tracker. A later heuristic
    decision of that seat must sample its world as if no search had happened: two sessions playing
    the same moves, one of which searched once on the way, choose the same heuristic moves."""
    model, belief = get_test_model(GAME), _belief_model(tmp_path)
    seed = 22  # with the posterior reused, 7 of this game's beam_2x3 decisions change (first at ply 214)
    searched = engine.GameSession(GAME, seed=seed, model_path=model, belief_model_path=belief)
    plain = engine.GameSession(GAME, seed=seed, model_path=model, belief_model_path=belief)
    rng = random.Random(seed)
    compared = 0
    for ply in range(350):
        legal = searched.get_legal_actions()
        if ply >= 20 and ply % 10 in (0, 1):  # both seats search now and then; the result is not played
            searched.get_ai_action(simulations=4, temperature=0.0)
        elif ply > 21 and len(legal) > 1:
            a = searched.get_heuristic_action("beam_2x3")["action"]
            b = plain.get_heuristic_action("beam_2x3")["action"]
            assert a == b, f"ply {ply}: the heuristic sampled a different world after an earlier search"
            compared += 1
        action = rng.choice(legal)
        searched.apply_action(action)
        plain.apply_action(action)
    assert compared > 200
