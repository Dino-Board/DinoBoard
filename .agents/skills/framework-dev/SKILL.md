---
description: "Modify MCTS / search / framework internals — algorithm understanding, safety red lines, performance hot paths"
when_to_use: "When changing engine/search/, engine/core/, engine/runtime/, engine/infer/, or any framework-level C++ code"
---
**STOP. Before touching any framework code, read reference.md completely.** Framework changes affect every game simultaneously. One wrong line can silently weaken all AI or leak hidden information with no test failure.
Read reference.md for the algorithm, red lines, and hot paths.
