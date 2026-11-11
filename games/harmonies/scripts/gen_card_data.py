"""Generate games/harmonies/harmonies_card_data.h and harmonies_board_data.h.

harmonies_card_data.h (from animal_cards.json):
- kTransforms[6]: six 60-degree axial rotation matrices applied to (q,r).
- kCards[32]: per-card data with `pattern_size`, `anchor_index`, `scores`,
  `pieces[8]` where each piece has `(dq, dr, tower_mask, is_anchor)`.
- `tower_mask` (uint16): bit i set iff tower_type=i satisfies `levels`.

harmonies_board_data.h (from BOARD_SIDES below), one BoardData<side> per
personal-board side:
- hex ids column-major, axial (q, r) per hex, neighbours per direction,
  an axial -> hex lookup padded so rotated pattern probes never leave it;
- the board's symmetry group: every hex-lattice rotation / reflection (plus
  translation) mapping the board onto itself, as hex permutation, direction
  permutation and axial affine map. A reflection is only emitted after
  checking that every animal pattern is achiral (its mirror image is one of
  the 6 rotations the rules allow), so the whole group preserves the rules.

Run from repo root:  python3 games/harmonies/scripts/gen_card_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "animal_cards.json"
OUT = ROOT / "harmonies_card_data.h"
OUT_BOARD = ROOT / "harmonies_board_data.h"

# Personal boards: flat-top hexes in columns, left to right; odd columns sit
# half a hex lower than even ones. Side A: rules p.2 (setup) / p.8; side B:
# the board drawn next to "Water - side B: the islands" on p.8.
BOARD_SIDES = {
    "A": [5, 4, 5, 4, 5],
    "B": [4, 3, 4, 3, 4, 3, 4],
}
# Axial directions, index order = the engine's direction ids.
DIRS = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]  # N NE SE S SW NW
DIR_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]
AXIAL_PAD = 8

# Tower type indices, matching harmonies_state.h:
#   0=empty, 1=Blue, 2=Gray1, 3=Brown, 4=Green, 5=Yellow, 6=Red1,
#   7=房(Red on something layer2), 8=木木 (Brown+Brown),
#   9=树2 (Brown+Green), 10=山2 (Gray+Gray), 11=树3 (Brown+Brown+Green),
#   12=山3 (Gray+Gray+Gray).
# Each tower_type defines a fixed (lv0, lv1, lv2) color stack.
TOWER_LAYERS = {
    0: (None, None, None),
    1: ("Blue", None, None),
    2: ("Gray", None, None),
    3: ("Brown", None, None),
    4: ("Green", None, None),
    5: ("Yellow", None, None),
    6: ("Red", None, None),
    # tower 7: 房 = base in {Brown, Gray, Red} with Red on top (layer 1).
    # Three concrete stacks; matched via per-stack expansion below.
    7: ("__BUILDING__", "Red", None),
    8: ("Brown", "Brown", None),
    9: ("Brown", "Green", None),
    10: ("Gray", "Gray", None),
    11: ("Brown", "Brown", "Green"),
    12: ("Gray", "Gray", "Gray"),
}

# Building base alternatives (tower_type 7 layer 0).
BUILDING_BASE = {"Brown", "Gray", "Red"}


def level_matches(required, actual: str | None) -> bool:
    """Does a piece's `levels[i]` requirement match this tower's layer color?"""
    if required is None:
        return actual is None  # layer must be empty
    if actual is None:
        return False  # piece needs a color here, tower has nothing
    if isinstance(required, list):
        return actual in required
    return required == actual


def tower_mask_for(piece_levels: list) -> int:
    """Compute uint16 mask: bit i set iff tower_type=i satisfies all 3 levels."""
    mask = 0
    for tower_id, layers in TOWER_LAYERS.items():
        # tower 7 special-cases: layer 0 may be Brown OR Gray OR Red.
        if tower_id == 7:
            ok_any = False
            for base in BUILDING_BASE:
                concrete = (base, "Red", None)
                if all(level_matches(req, act)
                       for req, act in zip(piece_levels, concrete)):
                    ok_any = True
                    break
            if ok_any:
                mask |= (1 << tower_id)
            continue
        if all(level_matches(req, act) for req, act in zip(piece_levels, layers)):
            mask |= (1 << tower_id)
    return mask


def rot(q, r):
    """60-degree rotation in axial coordinates."""
    return (-r, q + r)


def mirror(q, r):
    """Reflection swapping the cube x and z axes."""
    return (r, q)


def lattice_maps():
    """The 12 point symmetries of the hex lattice as axial 2x2 matrices
    (a, b, c, d): (q', r') = (a q + b r, c q + d r). Identity first."""
    maps = []
    for m in (False, True):
        for k in range(6):
            def f(q, r, m=m, k=k):
                if m:
                    q, r = mirror(q, r)
                for _ in range(k):
                    q, r = rot(q, r)
                return q, r
            cq, cr = f(1, 0), f(0, 1)
            maps.append(((cq[0], cr[0], cq[1], cr[1]), m, k))
    return maps


def build_board(col_heights):
    """Hex ids column-major; axial r = i - q // 2 puts odd columns half a hex
    lower (screen y = r + q / 2)."""
    axial = []
    starts = []
    for q, h in enumerate(col_heights):
        starts.append(len(axial))
        for i in range(h):
            axial.append((q, i - q // 2))
    return axial, starts


def card_patterns(cards):
    out = []
    for c in cards:
        out.append([(tuple(p["offset"]), tower_mask_for(p["levels"]), i == c["anchor_index"])
                    for i, p in enumerate(c["pattern"])])
    return out


def norm_pattern(pp):
    ax = next(p for p, _, a in pp if a)
    return frozenset(((q - ax[0], r - ax[1]), m, a) for (q, r), m, a in pp)


def pattern_is_achiral(pat):
    rots = set()
    cur = pat
    for _ in range(6):
        rots.add(norm_pattern(cur))
        cur = [(rot(*p), m, a) for p, m, a in cur]
    return norm_pattern([(mirror(*p), m, a) for p, m, a in pat]) in rots


def board_symmetries(axial, patterns):
    cells = set(axial)
    index = {p: h for h, p in enumerate(axial)}
    syms = []
    for (a, b, c, d), mirrored, k in lattice_maps():
        img = [(a * q + b * r, c * q + d * r) for q, r in axial]
        found = None
        for iq, ir in img:
            tq, tr = axial[0][0] - iq, axial[0][1] - ir
            if {(q + tq, r + tr) for q, r in img} == cells:
                found = (tq, tr)
                break
        if found is None:
            continue
        if mirrored:
            chiral = [i + 1 for i, pat in enumerate(patterns) if not pattern_is_achiral(pat)]
            assert not chiral, (
                f"cards {chiral} are chiral: the board reflection {(a, b, c, d)} would not preserve "
                f"pattern matching (which only tries rotations)")
        hex_map = [index[(q + found[0], r + found[1])] for q, r in img]
        dir_map = [DIRS.index((a * dq + b * dr, c * dq + d * dr)) for dq, dr in DIRS]
        name = ("mirror+" if mirrored else "") + f"rot{60 * k}"
        syms.append((name, hex_map, dir_map, (a, b, c, d, found[0], found[1])))
    assert syms[0][0] == "rot0" and syms[0][1] == list(range(len(axial)))
    return syms


def emit_board_header(patterns):
    lines = [
        "// AUTO-GENERATED by games/harmonies/scripts/gen_card_data.py",
        "// Do not edit by hand. Regenerate after changing BOARD_SIDES there.",
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace board_ai::harmonies {",
        "",
        "constexpr int kSideA = 0;",
        "constexpr int kSideB = 1;",
        "",
        "// Directions: " + ", ".join(f"{i}={n} {DIRS[i]}" for i, n in enumerate(DIR_NAMES))
        + " in axial (dq, dr).",
        "constexpr std::int8_t kDirDq[6] = {" + ", ".join(str(d[0]) for d in DIRS) + "};",
        "constexpr std::int8_t kDirDr[6] = {" + ", ".join(str(d[1]) for d in DIRS) + "};",
        "",
        "template <int Side>",
        "struct BoardData;",
        "",
    ]
    for side_idx, (side, heights) in enumerate(BOARD_SIDES.items()):
        axial, starts = build_board(heights)
        n = len(axial)
        index = {p: h for h, p in enumerate(axial)}
        nbrs = [[index.get((q + dq, r + dr), -1) for dq, dr in DIRS] for q, r in axial]
        syms = board_symmetries(axial, patterns)
        qs = [p[0] for p in axial]
        rs = [p[1] for p in axial]
        qmin, qmax = min(qs) - AXIAL_PAD, max(qs) + AXIAL_PAD
        rmin, rmax = min(rs) - AXIAL_PAD, max(rs) + AXIAL_PAD
        width, height = qmax - qmin + 1, rmax - rmin + 1
        flat = [-1] * (width * height)
        for h, (q, r) in enumerate(axial):
            flat[(q - qmin) * height + (r - rmin)] = h
        ns = len(syms)
        lines += [
            f"// Side {side}: columns of {'/'.join(map(str, heights))} hexes ({n} spaces).",
            "// Symmetries: " + ", ".join(s[0] for s in syms) + ".",
            "template <>",
            f"struct BoardData<{['kSideA', 'kSideB'][side_idx]}> {{",
            f"  static constexpr char kName = '{side}';",
            f"  static constexpr int kHexes = {n};",
            f"  static constexpr int kColumns = {len(heights)};",
            f"  static constexpr std::array<std::int8_t, {len(heights)}> kColumnHeight = {{"
            + ", ".join(map(str, heights)) + "};",
            f"  static constexpr std::array<std::int8_t, {len(heights)}> kColumnStart = {{"
            + ", ".join(map(str, starts)) + "};",
            f"  static constexpr std::array<std::int8_t, {n}> kAxialQ = {{" + ", ".join(map(str, qs)) + "};",
            f"  static constexpr std::array<std::int8_t, {n}> kAxialR = {{" + ", ".join(map(str, rs)) + "};",
            f"  static constexpr std::array<std::array<std::int8_t, 6>, {n}> kNeighbors = {{{{",
        ]
        for h in range(n):
            lines.append("      {{" + ", ".join(f"{v:2d}" for v in nbrs[h]) + f"}}}},  // hex {h} {axial[h]}")
        lines += [
            "  }};",
            f"  static constexpr int kAxialQOffset = {-qmin};",
            f"  static constexpr int kAxialROffset = {-rmin};",
            f"  static constexpr int kAxialWidth = {width};",
            f"  static constexpr int kAxialHeight = {height};",
            f"  static constexpr std::array<std::int8_t, {width * height}> kAxialToHex = {{",
        ]
        for i in range(0, len(flat), 24):
            lines.append("      " + ", ".join(str(v) for v in flat[i:i + 24]) + ",")
        lines += [
            "  };",
            f"  static constexpr int kNumSymmetries = {ns};",
            "  // Symmetry g maps hex h to kSymHex[g][h], direction d to kSymDir[g][d] and",
            "  // axial (q, r) to (a q + b r + e, c q + d r + f) with kSymAxial[g] = {a, b, c, d, e, f}.",
            f"  static constexpr std::array<std::array<std::int8_t, {n}>, {ns}> kSymHex = {{{{",
        ]
        for name, hex_map, _, _ in syms:
            lines.append("      {{" + ", ".join(map(str, hex_map)) + f"}}}},  // {name}")
        lines.append("  }};")
        lines.append(f"  static constexpr std::array<std::array<std::int8_t, 6>, {ns}> kSymDir = {{{{")
        for name, _, dir_map, _ in syms:
            lines.append("      {{" + ", ".join(map(str, dir_map)) + f"}}}},  // {name}")
        lines.append("  }};")
        lines.append(f"  static constexpr std::array<std::array<std::int8_t, 6>, {ns}> kSymAxial = {{{{")
        for name, _, _, aff in syms:
            lines.append("      {{" + ", ".join(map(str, aff)) + f"}}}},  // {name}")
        lines.append("  }};")
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace board_ai::harmonies")
    lines.append("")
    OUT_BOARD.write_text("\n".join(lines))
    print(f"wrote {OUT_BOARD} ({len(lines)} lines)")


def main():
    data = json.loads(JSON.read_text())
    cards = data["cards"]
    assert len(cards) == 32, f"expected 32 cards, got {len(cards)}"
    # The json legend: position 4 = S = (0, 1), the engine's direction 3.
    assert data["_position_to_offset"]["4"] == list(DIRS[3])

    # Emit header.
    lines = []
    lines.append("// AUTO-GENERATED by games/harmonies/scripts/gen_card_data.py")
    lines.append("// Do not edit by hand. Regenerate after changing animal_cards.json.")
    lines.append("#pragma once")
    lines.append("#include <array>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace board_ai::harmonies {")
    lines.append("")
    lines.append("constexpr int kMaxPiecesPerCard = 8;")
    lines.append("constexpr int kNumCards = 32;")
    lines.append("constexpr int kMaxScoresPerCard = 6;")
    lines.append("")
    lines.append("struct CardPiece {")
    lines.append("  std::int8_t dq;")
    lines.append("  std::int8_t dr;")
    lines.append("  std::uint16_t tower_mask;  // bit i set iff tower_type=i matches levels")
    lines.append("  std::uint8_t is_anchor;    // 0/1")
    lines.append("};")
    lines.append("")
    lines.append("struct CardData {")
    lines.append("  std::uint8_t pattern_size;")
    lines.append("  std::uint8_t anchor_index;")
    lines.append("  std::uint8_t num_cubes;       // == len(scores)")
    lines.append("  std::int16_t scores[kMaxScoresPerCard];  // score at each cube placement; scores[num_cubes-1] = on completion")
    lines.append("  CardPiece pieces[kMaxPiecesPerCard];")
    lines.append("};")
    lines.append("")

    lines.append("// 6 rotations (60-deg increments). Each row: (a, b, c, d) acting as")
    lines.append("//   (q', r') = (a*q + b*r, c*q + d*r)")
    lines.append("constexpr std::int8_t kTransforms[6][4] = {")
    cur_q = (1, 0)
    cur_r = (0, 1)
    rows = []
    for _ in range(6):
        rows.append((cur_q[0], cur_r[0], cur_q[1], cur_r[1]))
        cur_q = rot(*cur_q)
        cur_r = rot(*cur_r)
    for a, b, c, d in rows:
        lines.append(f"    {{{a}, {b}, {c}, {d}}},")
    lines.append("};")
    lines.append("")

    # Card pieces.
    lines.append("constexpr CardData kCards[kNumCards] = {")
    for c in cards:
        pieces = c["pattern"]
        assert len(pieces) <= 8
        anchor_index = c["anchor_index"]
        scores = c["scores"]
        assert len(scores) <= 6
        scores_padded = list(scores) + [0] * (6 - len(scores))
        lines.append(f"    // id={c['id']} {c['animal']} ({c['habitat']})")
        lines.append(f"    {{ {len(pieces)}, {anchor_index}, {len(scores)},")
        lines.append(f"      {{ {', '.join(str(s) for s in scores_padded)} }},")
        lines.append("      {")
        for i, p in enumerate(pieces):
            dq, dr = p["offset"]
            mask = tower_mask_for(p["levels"])
            is_anchor = 1 if i == anchor_index else 0
            lines.append(f"        {{ {dq}, {dr}, 0x{mask:04X}, {is_anchor} }},  // {p['levels']}")
        # pad
        for _ in range(8 - len(pieces)):
            lines.append("        { 0, 0, 0x0000, 0 },")
        lines.append("      }")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace board_ai::harmonies")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines, {len(cards)} cards)")
    emit_board_header(card_patterns(cards))


if __name__ == "__main__":
    main()
