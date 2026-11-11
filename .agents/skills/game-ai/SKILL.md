---
description: "Wire AI components — encoder, tracker, registration, event protocol, belief, training features, API (Phase 2 of adding a new game)"
---

# Game AI Components

Read the sub-document matching your current task:

- **IFeatureEncoder** (encode_features, masked_state, perspective): read [encoder.md](encoder.md)
- **History chain encoding** (bounded per-round play history → fixed-size reverse+pad feature block): read [history-chain-encoding.md](history-chain-encoding.md) — Maskmen
- **IBeliefTracker** (init, observe, randomize_unseen): read [tracker.md](tracker.md)
- **Belief sampling precision** (choosing randomize_unseen tier — exact presence / expected counts / joint; Coup/Maskmen/Liar's Dice): read [belief-sampling.md](belief-sampling.md)
- **GameBundle + GameRegistrar** (wire components, install_event_protocol): read [registration.md](registration.md)
- **Learned belief network** (bluff games — Coup/LiarsDice/6nimmit): read [belief-net.md](belief-net.md)
- **Optional training features** (heuristic, tail solver, filter, adjudicator, stats, opponent pool): read [training.md](training.md)
- **REST AI API** (observation-only session lifecycle, endpoints): read [api.md](api.md)

Only read the file relevant to the current task.
