"""Ark Nova's final score breakdown: per seat appeal and conservation from play, conservation from
the Final Scoring card, conservation and appeal from the other end-of-game cards.

* The categories rebuild each seat's final score: (appeal) minus the target appeal of (conservation).
* Each Final Scoring card a seat holds scores 0-4 (Resistance can add cards to the one kept at 10
  conservation points); the end-of-game parts are never negative, and greedy games reach non-zero
  values in each of them.
* The Ark Nova training config enables the breakdown head.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dinoboard_engine as engine

sys.path.insert(0, str(Path(__file__).parent))
from arknova_reference import GAME, ROOT, target_appeal  # noqa: E402

MAX_FINAL_CARDS = 6  # kMaxFinalCards


def test_breakdown_rebuilds_the_final_scores():
    assert engine.game_metadata(GAME)["score_breakdown_categories"] == 5
    seen = [0] * 5
    for seed in (2, 5, 11):
        ep = engine.run_heuristic_episode(game_id=GAME, seed=seed, temperature=0.0, max_game_plies=3000,
                                          heuristic_selector="greedy")
        bd, final = ep["final_score_breakdown"], ep["final_scores"]
        assert len(bd) == 10
        for p in range(2):
            appeal, conservation, final_cp, card_cp, card_appeal = bd[5 * p:5 * p + 5]
            # Resistance can add final scoring cards: each scores at most 4.
            assert 0 <= final_cp <= 4 * MAX_FINAL_CARDS and card_cp >= 0 and card_appeal >= 0, f"seed {seed} seat {p}: {bd}"
            vp = appeal + card_appeal - target_appeal(int(conservation + final_cp + card_cp))
            assert vp == final[p], f"seed {seed} seat {p}: breakdown {bd[5 * p:5 * p + 5]} gives {vp}, score {final[p]}"
            seen = [s + bool(v) for s, v in zip(seen, bd[5 * p:5 * p + 5])]
    assert seen[2] and (seen[3] or seen[4]), f"categories never non-zero: {seen}"


def test_training_config_enables_the_breakdown_head():
    train = json.loads((ROOT / "games/arknova/config/train.json").read_text(encoding="utf-8"))["training"]
    assert train["score_breakdown_head"]["loss_weight"] > 0
