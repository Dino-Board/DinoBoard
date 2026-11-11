"""Splendor belief sampling must reproduce the public deck sizes exactly.

`randomize_unseen` rebuilds each tier's deck from the cards the tracker has
not seen yet. Every card leaving the hidden pool (deck flip, opponent buying
a blind-reserved card) must be retired from that pool, otherwise sampled
decks come out larger than `deck_sizes` and the search plays in impossible
worlds.
"""
import dinoboard_engine


def test_sampled_deck_size_matches_truth():
    for seed in range(60):
        out = dinoboard_engine.test_belief_tracker("splendor", seed, 60, 3)
        true_n = len(out["original_deck"])
        for deck in out["trial_decks"]:
            assert len(deck) == true_n, (
                f"seed={seed}: sampled deck has {len(deck)} cards, truth has {true_n}")
