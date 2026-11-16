"""Autoregressive belief over sorted hands (engine/core/belief_sequence.h, belief.json type
"autoregressive"): the hidden hands are written smallest element first, one segment after
another; the net's logits are added to the uniform deal's log-probabilities of the next element.

Expected values are recomputed here by enumerating the possible assignments, not read back from
the engine.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, deque

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from training.model import (BELIEF_SEQUENCE_LOSS, create_belief_model_from_config, export_belief_onnx,
                            validate_belief_config)
from training.pipeline import _train_belief_step_sequence

CONFIG = {"type": "autoregressive", "architecture": [64, 64], "head": [64], "activation": "relu",
          "batch_norm": False, "residual": False, "block_size": 2, "training": {"loss": BELIEF_SEQUENCE_LOSS}}


def _layout(classes, segments):
    return {"classes": classes, "segments": segments, "prefix_dim": segments * classes + segments + classes}


def _assignments(pool, sizes):
    """Every way to give the segments sorted hands out of the pool (a 0/1 list)."""
    def rec(left, rest):
        if not rest:
            yield ()
            return
        for hand in itertools.combinations(left, rest[0]):
            for tail in rec([c for c in left if c not in hand], rest[1:]):
                yield (hand,) + tail
    return list(rec([c for c, v in enumerate(pool) if v], list(sizes)))


def _rows(pool, hands):
    """Training rows of one position: hands as lists of elements per segment."""
    k = len(pool)
    onehot = np.zeros((1, len(hands), k), dtype=np.int32)
    for s, hand in enumerate(hands):
        onehot[0, s, list(hand)] = 1
    return engine.belief_sequence_training_rows(k, len(hands), np.asarray([pool], dtype=np.int32), onehot)


def _sample(pool, sizes, n, seed, **kw):
    kw.setdefault("logits", [0.0] * len(pool))
    return engine.belief_sequence_sample(len(pool), len(sizes), pool, list(sizes), n, seed, **kw)


CASES = [
    ([1, 1, 1, 0, 1, 1], (2, 1)),
    ([1, 1, 1, 1, 1], (2, 3)),       # the pool is exactly the hands
    ([0, 1, 1, 1, 1, 1, 0, 1], (1, 0, 3)),  # a segment with nothing hidden
    ([1, 0, 1, 1], (0, 0)),
]


@pytest.mark.parametrize("pool,sizes", CASES)
def test_the_prior_is_the_uniform_deal(pool, sizes):
    """Walking any possible assignment, the prior's log-probabilities of its elements sum to
    -log(number of assignments): every assignment is equally likely before the net speaks."""
    every = _assignments(pool, sizes)
    for hands in every:
        rows = _rows(pool, hands)
        assert rows["target"].tolist() == [c for hand in hands for c in hand]
        total = sum(rows["log_prior"][i, t] for i, t in enumerate(rows["target"]))
        assert total == pytest.approx(-math.log(len(every)), abs=1e-4)
        # Each step's prior is a distribution over the elements that can come next.
        for row in rows["log_prior"]:
            assert np.exp(row[np.isfinite(row)]).sum() == pytest.approx(1.0, abs=1e-5)


def test_the_prior_allows_exactly_the_elements_that_can_come_next():
    """Pool {0,1,2,4,5}, hands (1,4) then (2): not an element the pool lacks, not one at or below
    the segment's previous element, not one already given out, and not one so high that the
    segment could not be completed."""
    rows = _rows([1, 1, 1, 0, 1, 1], [(1, 4), (2,)])
    allowed = [sorted(np.flatnonzero(np.isfinite(r)).tolist()) for r in rows["log_prior"]]
    assert allowed == [[0, 1, 2, 4],   # 5 would leave nothing above it for the second card
                       [2, 4, 5],      # above 1
                       [0, 2, 5]]      # second segment: what is left
    assert rows["position"].tolist() == [0, 0, 0]


def test_the_prefix_tells_what_has_been_written():
    """[segments x classes] elements given so far, [segments] the segment being filled, [classes]
    its previous element."""
    k, pool, hands = 6, [1, 1, 1, 0, 1, 1], [(1, 4), (2,)]
    prefix = _rows(pool, hands)["prefix"]
    assert prefix.shape == (3, 2 * k + 2 + k)
    given, segment, last = prefix[:, :2 * k].reshape(3, 2, k), prefix[:, 2 * k:2 * k + 2], prefix[:, 2 * k + 2:]
    assert given[0].sum() == 0 and segment[0].tolist() == [1, 0] and last[0].sum() == 0
    assert np.flatnonzero(given[1].ravel()).tolist() == [1] and segment[1].tolist() == [1, 0]
    assert np.flatnonzero(last[1]).tolist() == [1]
    assert np.flatnonzero(given[2].ravel()).tolist() == [1, 4] and segment[2].tolist() == [0, 1]
    assert last[2].sum() == 0, "a new segment starts without a previous element"


def test_rows_of_a_batch_name_their_position():
    pool = np.asarray([[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=np.int32)
    hands = np.zeros((3, 1, 4), dtype=np.int32)
    hands[0, 0, [0, 3]] = 1
    hands[2, 0, [1, 2, 3]] = 1  # the middle position hides nothing
    rows = engine.belief_sequence_training_rows(4, 1, pool, hands)
    assert rows["position"].tolist() == [0, 0, 2, 2, 2]
    assert rows["target"].tolist() == [0, 3, 1, 2, 3]


@pytest.mark.parametrize("pool,sizes", CASES)
def test_a_silent_net_deals_uniformly(pool, sizes):
    """Zero logits: every sample is a possible assignment and all of them come up equally often."""
    every = _assignments(pool, sizes)
    n = 400 * len(every)
    seen = Counter(tuple(tuple(h) for h in s) for s in _sample(pool, sizes, n, seed=7))
    assert set(seen) == set(every)
    for count in seen.values():
        assert abs(count - 400) < 5 * math.sqrt(400)


def test_the_net_tilts_the_deal_and_never_breaks_it():
    """Logits favouring element 4: it is dealt whenever an assignment can hold it (here always,
    three of five elements are dealt), and every sample is still a possible assignment."""
    pool, sizes = [1, 1, 1, 0, 1, 1], (2, 1)
    every = set(_assignments(pool, sizes))
    samples = _sample(pool, sizes, 2000, seed=3, logits=[0, 0, 0, 0, 12.0, 0])
    assert {tuple(tuple(h) for h in s) for s in samples} <= every
    assert sum(any(4 in h for h in s) for s in samples) > 1900
    # A logit on an element that cannot be dealt changes nothing.
    assert _sample(pool, sizes, 50, seed=3, logits=[0, 0, 0, 30.0, 0, 0]) == _sample(pool, sizes, 50, seed=3)


def test_sampling_is_seeded():
    pool, sizes = [1] * 12, (3, 4)
    assert _sample(pool, sizes, 20, seed=5) == _sample(pool, sizes, 20, seed=5)
    assert _sample(pool, sizes, 20, seed=5) != _sample(pool, sizes, 20, seed=6)


def test_impossible_problems_are_refused():
    with pytest.raises(ValueError, match="pool holds 2 elements for hands of 3"):
        _sample([1, 1, 0], (2, 1), 1, seed=1)
    with pytest.raises(ValueError, match="at most once"):
        _sample([2, 1, 0], (1, 1), 1, seed=1)
    with pytest.raises(ValueError, match="cannot come next"):
        _rows([1, 1, 0, 1], [(2,)])  # the hand holds an element the pool lacks
    with pytest.raises(ValueError, match="cannot come next"):
        _rows([1, 1, 1, 1], [(1,), (1,)])  # two segments hold the same element


def test_config_ties_the_type_to_its_loss_and_layout():
    validate_belief_config(CONFIG)
    with pytest.raises(ValueError, match="go together"):
        validate_belief_config({**CONFIG, "training": {"loss": "ce"}})
    with pytest.raises(ValueError, match="go together"):
        validate_belief_config({**CONFIG, "type": "mlp"})
    with pytest.raises(ValueError, match="needs \\['head'\\]"):
        validate_belief_config({k: v for k, v in CONFIG.items() if k != "head"})
    layout = _layout(6, 2)
    net = create_belief_model_from_config(CONFIG, input_dim=9, output_dim=6, sequence_layout=layout)
    assert net.feature_dim == 9 and net.prefix_dim == 20
    with pytest.raises(ValueError, match="sequence layout"):
        create_belief_model_from_config(CONFIG, input_dim=9, output_dim=6)
    with pytest.raises(ValueError, match="disagrees"):
        create_belief_model_from_config(CONFIG, input_dim=9, output_dim=7, sequence_layout=layout)
    mlp = {**CONFIG, "type": "mlp", "training": {"loss": "ce"}}
    create_belief_model_from_config(mlp, input_dim=9, output_dim=6)
    with pytest.raises(ValueError, match="must be 'autoregressive'"):
        create_belief_model_from_config(mlp, input_dim=9, output_dim=6, sequence_layout=layout)


def test_an_untrained_net_deals_uniformly(tmp_path):
    """A new net's last layer is zero, so its ONNX export samples like the uniform deal."""
    pool, sizes, features = [1, 1, 1, 0, 1, 1], (2, 1), [0.3, -1.0, 2.0]
    torch.manual_seed(1)
    net = create_belief_model_from_config(CONFIG, input_dim=len(features), output_dim=6, sequence_layout=_layout(6, 2))
    path = tmp_path / "belief.onnx"
    export_belief_onnx(net, path, len(features))
    got = _sample(pool, sizes, 300, seed=11, model_path=str(path), features=features)
    assert got == _sample(pool, sizes, 300, seed=11)


def test_the_exported_model_is_the_trained_one(tmp_path):
    """The export takes one position and any number of prefixes (the position's part of the net
    runs once); its logits equal the training-time forward on the same rows."""
    import onnxruntime as ort

    k, layout = 7, _layout(7, 2)
    torch.manual_seed(4)
    net = create_belief_model_from_config({**CONFIG, "batch_norm": True}, input_dim=5, output_dim=k,
                                          sequence_layout=layout).eval()
    torch.nn.init.normal_(net.head[-1].weight, std=0.5)
    path = tmp_path / "belief.onnx"
    export_belief_onnx(net, path, 5)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert {i.name: i.shape[1] for i in session.get_inputs()} == {"features": 5, "prefix": layout["prefix_dim"]}
    features = torch.randn(3, 5)
    for rows in (1, 2, 9):
        prefix = torch.randn(rows, layout["prefix_dim"])
        for position in range(3):
            want = net(features, prefix, torch.full((rows,), position)).detach().numpy()
            got = session.run(None, {"features": features[position:position + 1].numpy(), "prefix": prefix.numpy()})[0]
            assert got.shape == (rows, k)
            assert np.allclose(got, want, atol=1e-5)


def test_samples_in_the_same_state_share_one_row():
    """The sampler asks the net once per distinct prefix: the result is the same deal-by-deal
    distribution, and with a single hidden element one row serves every sample."""
    pool, sizes = [1, 1, 1, 0, 1, 1], (1,)
    seen = Counter(s[0][0] for s in _sample(pool, sizes, 5000, seed=2, logits=[0, 0, 2.0, 0, 0, 0]))
    weights = {0: 1.0, 1: 1.0, 2: math.e ** 2, 4: 1.0, 5: 1.0}
    total = sum(weights.values())
    assert set(seen) == set(weights)
    for c, w in weights.items():
        assert abs(seen[c] - 5000 * w / total) < 5 * math.sqrt(5000 * w / total)


def test_training_learns_what_the_features_tell(tmp_path):
    """Positions whose features name one element the first segment always holds and one the second
    never holds: the trained net's loss drops below the uniform deal's, and its samples (through
    the exported model, as a tracker would draw them) follow both hints."""
    k, sizes = 10, (3, 2)
    layout = _layout(k, 2)
    rng = np.random.default_rng(0)

    def position():
        pool = np.ones(k, dtype=np.int32)
        pool[rng.choice(k, 2, replace=False)] = 0
        deal = rng.permutation(np.flatnonzero(pool))
        first, second, rest = deal[:3], deal[3:5], deal[5:]
        held, absent = int(rng.choice(first)), int(rng.choice(np.concatenate([first, rest])))
        hands = np.zeros((2, k), dtype=np.int32)
        hands[0, first] = 1
        hands[1, second] = 1
        features = np.zeros(2 * k, dtype=np.float32)
        features[held] = 1.0
        features[k + absent] = 1.0
        return features, hands, pool, held, absent

    buffer = deque((f, h, p, None) for f, h, p, _, _ in (position() for _ in range(4000)))
    torch.manual_seed(2)
    in_dim = 2 * k
    net = create_belief_model_from_config(CONFIG, input_dim=in_dim, output_dim=k, sequence_layout=layout)
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3)
    first_loss, uniform = _train_belief_step_sequence(
        net, optimizer, buffer, layout=layout, batch_size=256, batches_per_step=1, grad_clip=1.0)
    assert first_loss == pytest.approx(uniform, abs=1e-4), "an untrained net is the uniform deal"
    loss, uniform = _train_belief_step_sequence(
        net, optimizer, buffer, layout=layout, batch_size=256, batches_per_step=2000, grad_clip=1.0)
    assert loss < uniform - 0.1

    path = tmp_path / "belief.onnx"
    export_belief_onnx(net, path, in_dim)
    follows_held = follows_absent = uniform_held = total = 0
    for seed in range(40):
        features, _, pool, held, absent = position()
        kw = dict(n=50, seed=seed)
        learned = engine.belief_sequence_sample(k, 2, pool.tolist(), list(sizes), model_path=str(path),
                                                features=features.tolist(), logits=[], **kw)
        blind = engine.belief_sequence_sample(k, 2, pool.tolist(), list(sizes), logits=[0.0] * k, **kw)
        for s in learned:
            assert len(s[0]) == sizes[0] and len(s[1]) == sizes[1]
            assert all(pool[c] for h in s for c in h) and len({c for h in s for c in h}) == 5
            follows_held += held in s[0]
            follows_absent += absent not in s[1]
        uniform_held += sum(held in s[0] for s in blind)
        total += len(learned)
    assert follows_held > 0.9 * total and follows_absent > 0.9 * total, (follows_held, follows_absent, total)
    assert uniform_held < 0.6 * total, "the uniform deal gives the first segment that element 3 times in 8"
