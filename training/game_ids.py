"""Game id variants. A registered id is <base>[_<side>][_tok][_<N>p]: board side (`_a`), token
encoder (`_tok`), player count (`_3p`). Variants share the base game's configs and model folder."""
from __future__ import annotations

import re

_VARIANT_RE = re.compile(r"(_[a-z])?(_tok)?(_\d+p)?$")


def base_game_id(game_id: str) -> str:
    """'splendor_tok' -> 'splendor', 'harmonies_a_3p' -> 'harmonies', 'azul' -> 'azul'."""
    return _VARIANT_RE.sub("", game_id)
