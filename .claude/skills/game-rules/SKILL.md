---
description: "Implement game state, rules, schema, and build integration (Phase 1 of adding a new game)"
---

# Game Rules Development

Before implementing a game mechanic (bag draw, hidden deck, endgame solving,
viz reveals, intra-turn phases), check the **mechanic recipes** below for
which existing game to mirror — each recipe states the pattern, the worked
example, and the invariants. Do not re-derive a pattern an existing game
already solved. (Belief-sampling precision — Coup/Maskmen/Liar's Dice — is a
tracker concern; see the `game-ai` skill.)

## Mechanic recipes (which game to mirror)

- **Bag draw with per-token counts** (public count reservoir, weighted draw): read [bag-draw.md](bag-draw.md) — Azul, Harmonies
- **Hidden deck maintenance** (secret contents, public size, belief resample): read [deck-maintenance.md](deck-maintenance.md) — Splendor, Harmonies
- **Tail solving in stochastic games** (alpha-beta endgame without peeking random): read [tail-solve.md](tail-solve.md) — Splendor, Azul
- **Viz tricks** (precision reveals that raise AI strength): read [viz-tricks.md](viz-tricks.md) — Love Letter
- **Intra-turn phase state machine** (multi-stage sub-protocol within one turn): read [phase-state-machine.md](phase-state-machine.md) — Coup
- **Enumerated state + transition table** (finite legal combinations -> type enum + lookup table): read [enumerated-state-transition.md](enumerated-state-transition.md) — Harmonies
- **Simultaneous actions** (acts_mask + pending_commits + cursor scan, no framework change): read [simultaneous.md](simultaneous.md) — 6 nimmt!
- **Factored actions** (compound options as atom sequences over a public prefix; pointer atoms for the transformer): read [factored-actions.md](factored-actions.md) — Ark Nova
- **Count form + effect-based action ids** (unordered hands/markets hash equivalent multisets to one DAG node; actions by effect not by slot): read [count-and-effect-encoding.md](count-and-effect-encoding.md) — Jaipur

## Interface reference

Read the sub-document matching your current task:

- **IGameState** (schema, viz fields, hash dispatchers, clone): read [state.md](state.md)
- **IGameRules** (legal_actions, do_action_fast, viz maintenance, undo): read [rules.md](rules.md)
- **Build integration** (manifest.json, CMakeLists, file layout, invariants): read [build.md](build.md)

Only read the file relevant to the current task.
