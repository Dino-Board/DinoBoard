"""Neural network model for policy-value prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_residual_layers(hidden_layers: Sequence[int], block_size: int,
                              *, where: str) -> None:
    if not hidden_layers:
        raise ValueError(f"{where}: hidden_layers empty under residual=true")
    if block_size < 1:
        raise ValueError(f"{where}: block_size must be >= 1, got {block_size}")
    if len(hidden_layers) < block_size:
        raise ValueError(
            f"{where}: residual=true requires len(hidden_layers) >= block_size; "
            f"got len={len(hidden_layers)}, block_size={block_size}")
    if len(hidden_layers) % block_size != 0:
        raise ValueError(
            f"{where}: residual=true requires len(hidden_layers) % block_size == 0; "
            f"got len={len(hidden_layers)}, block_size={block_size}")
    width = hidden_layers[0]
    if any(h != width for h in hidden_layers):
        raise ValueError(
            f"{where}: residual=true requires uniform width across hidden_layers; "
            f"got {list(hidden_layers)}")


_TOKEN_TRANSFORMER_KEYS = ("type", "d_model", "num_layers", "num_heads", "ffn_dim")
_TOKEN_TRANSFORMER_OPTIONAL = ("attention", "relative_position_radius", "anchor_embedding")
_ATTENTION_EXPORTS = ("matmul", "fused")


def _validate_pv_network_config(network_cfg: dict) -> None:
    net_type = network_cfg.get("type", "mlp")
    if net_type == "token_transformer":
        missing = [k for k in _TOKEN_TRANSFORMER_KEYS if k not in network_cfg]
        unknown = sorted(set(network_cfg) - set(_TOKEN_TRANSFORMER_KEYS) - set(_TOKEN_TRANSFORMER_OPTIONAL))
        if missing or unknown:
            raise ValueError(
                f"PV network.type='token_transformer' needs {list(_TOKEN_TRANSFORMER_KEYS)} "
                f"(optional {list(_TOKEN_TRANSFORMER_OPTIONAL)}); missing={missing}, unknown={unknown}")
        attention = network_cfg.get("attention", "matmul")
        if attention not in _ATTENTION_EXPORTS:
            raise ValueError(f"PV network.attention must be one of {list(_ATTENTION_EXPORTS)}, got {attention!r}")
        radius = network_cfg.get("relative_position_radius")
        if radius is not None and (isinstance(radius, bool) or not isinstance(radius, int) or radius < 1):
            raise ValueError(f"PV network.relative_position_radius must be a positive int, got {radius!r}")
        if not isinstance(network_cfg.get("anchor_embedding", True), bool):
            raise ValueError(f"PV network.anchor_embedding must be a bool, got {network_cfg['anchor_embedding']!r}")
        for k in _TOKEN_TRANSFORMER_KEYS[1:]:
            v = network_cfg[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"PV network.{k} must be a positive int, got {v!r}")
        if network_cfg["d_model"] % network_cfg["num_heads"] != 0:
            raise ValueError(
                f"PV network.d_model={network_cfg['d_model']} must be divisible by "
                f"num_heads={network_cfg['num_heads']}")
        return
    if net_type != "mlp":
        raise ValueError(
            f"PV network.type={net_type!r} is not supported; expected 'mlp' or "
            f"'token_transformer'.")
    activation = network_cfg.get("activation", "relu")
    if activation != "relu":
        raise ValueError(
            f"PV network.activation={activation!r} is not supported; only "
            f"'relu' is implemented today.")


def validate_belief_config(belief_config: dict, *, where: str = "belief.json") -> None:
    if "enabled" in belief_config:
        raise ValueError(
            f"{where}: 'enabled' is not a supported belief config field. "
            f"Belief is enabled by engine capability plus the presence of "
            f"belief.json; remove this field.")
    if "temperature" in belief_config:
        raise ValueError(
            f"{where}: belief.json may not contain 'temperature'. Belief logit "
            f"temperature is fixed at 1.0 in the engine; calibrate the belief "
            f"model during training instead.")
    activation = belief_config["activation"]
    if activation != "relu":
        raise ValueError(
            f"{where}: activation={activation!r} is not supported; only 'relu' "
            f"is implemented today.")


class ResidualBlock(nn.Module):
    """Pre-add residual block: width-uniform stack of Linear (+optional BN) +
    ReLU between, with skip across the whole stack and ReLU after the add.

    Forward:
        h = ReLU(BN(Linear(x)))                          # block_size-1 times
        h = BN(Linear(h))                                # last linear, no ReLU
        out = ReLU(h + x)
    """

    def __init__(self, width: int, block_size: int, batch_norm: bool):
        super().__init__()
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        layers: list[nn.Module] = []
        for i in range(block_size):
            layers.append(nn.Linear(width, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            if i < block_size - 1:
                layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.tail = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(x) + x)


def _build_backbone(input_dim: int, hidden_layers: Sequence[int],
                    *, residual: bool, block_size: int, batch_norm: bool,
                    where: str) -> tuple[nn.Sequential, int]:
    """Returns (backbone, out_width). Plain path mirrors the legacy MLP shape;
    residual path projects to the uniform width once, then stacks blocks."""
    layers: list[nn.Module] = []
    if not residual:
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        return nn.Sequential(*layers), prev

    _validate_residual_layers(hidden_layers, block_size, where=where)
    width = hidden_layers[0]
    layers.append(nn.Linear(input_dim, width))
    if batch_norm:
        layers.append(nn.BatchNorm1d(width))
    layers.append(nn.ReLU())
    num_blocks = len(hidden_layers) // block_size
    for _ in range(num_blocks):
        layers.append(ResidualBlock(width, block_size, batch_norm=batch_norm))
    return nn.Sequential(*layers), width


class PVNet(nn.Module):
    def __init__(self, input_dim: int, policy_dim: int, hidden_layers: list[int],
                 num_players: int = 2,
                 *, residual: bool = False, block_size: int = 2):
        super().__init__()
        self.num_players = num_players
        # PVNet historically has no BN; keep that under both paths.
        self.backbone, out_width = _build_backbone(
            input_dim, hidden_layers,
            residual=residual, block_size=block_size, batch_norm=False,
            where="PVNet")
        self.policy_head = nn.Linear(out_width, policy_dim)
        self.value_head = nn.Sequential(nn.Linear(out_width, num_players), nn.Tanh())
        self.embedding_dim = out_width

    def forward_with_embedding(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h), h

    def forward(self, x: torch.Tensor):
        policy, value, _ = self.forward_with_embedding(x)
        return policy, value


class _OrtMultiHeadAttention(torch.autograd.Function):
    """Export-only: emits onnxruntime's fused com.microsoft::MultiHeadAttention
    (q/k/v as [B, S, H*dh], additive bias [B, 1, S, S]). forward() only runs
    while tracing. The TorchScript exporter cannot emit skipped optional
    inputs, so the bias is traced as input 3 and `_place_mha_attention_bias`
    moves it to its real slot (5) after export."""

    @staticmethod
    def forward(ctx, q, k, v, attn_bias, num_heads):
        b, s, d = q.shape
        split = lambda t: t.reshape(b, s, num_heads, d // num_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=attn_bias)
        return out.transpose(1, 2).reshape(b, s, d)

    @staticmethod
    def symbolic(g, q, k, v, attn_bias, num_heads):
        return g.op("com.microsoft::MultiHeadAttention", q, k, v, attn_bias, num_heads_i=num_heads)


def _place_mha_attention_bias(path: Path) -> None:
    import onnx
    model = onnx.load(str(path))
    nodes = [n for n in model.graph.node if n.op_type == "MultiHeadAttention" and n.domain == "com.microsoft"]
    if not nodes:
        return
    for n in nodes:
        if len(n.input) != 4:
            raise ValueError(f"{n.name}: expected the 4 traced inputs (q, k, v, attn_bias), got {list(n.input)}")
        q, k, v, bias = n.input
        del n.input[:]
        n.input.extend([q, k, v, "", "", bias])
    onnx.save(model, str(path))


class _BucketGather(torch.autograd.Function):
    """F.embedding(index, table) for a small table read by a huge index: the
    gradient is one index_add_, ~3x faster than embedding's sort-based backward."""

    @staticmethod
    def forward(ctx, index, table):
        ctx.save_for_backward(index)
        ctx.table_shape = table.shape
        return F.embedding(index, table)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        rows, cols = ctx.table_shape
        table_grad = grad.new_zeros(rows, cols, dtype=torch.float32).index_add_(
            0, index.reshape(-1), grad.reshape(-1, cols).float())
        return None, table_grad.to(grad.dtype)


class _TransformerBlock(nn.Module):
    """Pre-LN self-attention block over a token set of `num_tokens` tokens, or
    of any length when `num_tokens` is None. Trains with
    scaled_dot_product_attention; `attention` picks the exported graph:
    "matmul" (plain ONNX ops) or "fused" (onnxruntime contrib op, needs a
    [B, 1 or H, S, S] bias)."""

    def __init__(self, num_tokens: int | None, d_model: int, num_heads: int, ffn_dim: int,
                 attention: str = "matmul"):
        super().__init__()
        if attention not in _ATTENTION_EXPORTS:
            raise ValueError(f"attention must be one of {list(_ATTENTION_EXPORTS)}, got {attention!r}")
        self.attention = attention
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        n = x.shape[1] if self.num_tokens is None else self.num_tokens
        h, dh = self.num_heads, self.head_dim
        qkv = self.qkv(self.ln1(x))
        if torch.onnx.is_in_onnx_export() and self.attention == "fused":
            q, k, v = qkv.split(h * dh, dim=-1)
            out = _OrtMultiHeadAttention.apply(q, k, v, attn_bias.expand(-1, -1, n, -1), h)
        else:
            qkv = qkv.reshape(-1, n, 3, h, dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if torch.onnx.is_in_onnx_export():
                scores = torch.matmul(q, k.transpose(-1, -2)) * (dh ** -0.5) + attn_bias
                out = torch.matmul(torch.softmax(scores, dim=-1), v)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype))
            out = out.permute(0, 2, 1, 3).reshape(-1, n, h * dh)
        x = x + self.proj(out)
        return x + self.ff2(self.act(self.ff1(self.ln2(x))))


class TokenTransformerPVNet(nn.Module):
    """Set transformer over the encoder's TokenLayout (engine/core/feature_encoder.h).

    Input stays the flat `[B, num_tokens * token_dim]` feature vector, so the
    C++ inference path is unchanged. Each slot gets a per-type projection plus
    an entity-id embedding and no positional embedding, which keeps outputs
    equivariant to interchangeable slots. A learned CLS token feeds the value
    head and the global policy head; entity-bound actions are scored by a
    pointer head on their token and scattered back into the flat action space.
    With anchors, a token's pointer heads score the actions bound to its
    anchor id instead of a fixed slot, and the anchor adds a learned
    position embedding.

    A variable-length layout (`variable_tokens`) takes `[B, n * token_dim]`
    for any n: each token names its own type, every pointer action is bound
    by anchor, and all-zero padding tokens are masked out.

    With `token_positions`, attention adds a learned per-head bias for each
    pair of positioned tokens by their offset (dx, dy) clipped to
    `relative_position_radius`, and one bias for pairs where either token (or
    CLS) has no position. One table serves all blocks (as in T5): the bias
    is [B, H, S, S] per batch, so a table per block would multiply the
    dominant memory traffic by num_layers. `anchor_embedding=False` drops the
    absolute anchor embedding, so outputs depend on positions only through
    offsets.
    """

    def __init__(self, token_layout: dict, policy_dim: int, num_players: int, *,
                 d_model: int, num_layers: int, num_heads: int, ffn_dim: int,
                 attention: str = "matmul", relative_position_radius: int | None = None,
                 anchor_embedding: bool = True):
        super().__init__()
        self.num_players = num_players
        self.variable_tokens = bool(token_layout.get("variable_tokens", False))
        self.token_positions = bool(token_layout.get("token_positions", False))
        if self.token_positions and not self.variable_tokens:
            raise ValueError("token_positions requires a variable_tokens layout")
        if self.token_positions != (relative_position_radius is not None):
            raise ValueError(
                "network.relative_position_radius must be set exactly when the token layout has "
                f"token_positions (token_positions={self.token_positions}, "
                f"relative_position_radius={relative_position_radius!r})")
        self.rel_radius = relative_position_radius
        self.num_tokens = None if self.variable_tokens else int(token_layout["num_tokens"])
        self.token_dim = int(token_layout["token_dim"])
        num_types = int(token_layout["num_token_types"])
        num_entities = int(token_layout["num_entity_ids"])
        num_ptr_heads = int(token_layout["num_pointer_heads"])
        token_types = list(token_layout["token_types"])
        action_token = list(token_layout["action_token"])
        action_head = list(token_layout["action_head"])
        if self.variable_tokens:
            if any(t >= 0 for t in action_token) or any(hd >= 0 for hd in action_head):
                raise ValueError("variable_tokens layouts bind actions by anchor only "
                                 "(action_token / action_head must all be -1)")
        elif len(token_types) != self.num_tokens:
            raise ValueError(f"token_types has {len(token_types)} entries, expected {self.num_tokens}")
        if len(action_token) != policy_dim or len(action_head) != policy_dim:
            raise ValueError(
                f"action_token/action_head lengths {len(action_token)}/{len(action_head)} "
                f"!= policy_dim {policy_dim}")

        self.num_anchors = int(token_layout.get("num_anchor_ids", 0))
        anchor_action = list(token_layout.get("anchor_action", []))
        if len(anchor_action) != self.num_anchors * num_ptr_heads:
            raise ValueError(
                f"anchor_action has {len(anchor_action)} entries, expected "
                f"num_anchor_ids*num_pointer_heads = {self.num_anchors * num_ptr_heads}")

        if not anchor_embedding and not self.num_anchors:
            raise ValueError("network.anchor_embedding=false needs a layout with anchors")
        self.anchor_embedding = bool(self.num_anchors) and anchor_embedding
        self.feat_offset = 7 if self.token_positions else 4 if self.variable_tokens else 3 if self.num_anchors else 2
        self.reads_anchor = self.variable_tokens or bool(self.num_anchors)
        feat_dim = self.token_dim - self.feat_offset
        if self.variable_tokens:
            # One-hot rows gathered by type id: onnxconverter_common's fp16 pass
            # mistypes a bool -> float Cast, so the one-hot must not come from a comparison.
            self.register_buffer("type_onehot", torch.eye(num_types), persistent=False)
        else:
            # One matmul per token type over its slots, then permute back to slot
            # order: a per-slot weight gather lowers to num_tokens tiny matmuls.
            type_slots = [[s for s, t in enumerate(token_types) if t == k] for k in range(num_types)]
            if any(not 0 <= t < num_types for t in token_types):
                raise ValueError(f"token_types must lie in [0, {num_types})")
            self.present_types = [k for k in range(num_types) if type_slots[k]]
            for k in self.present_types:
                self.register_buffer(f"type_slots_{k}", torch.tensor(type_slots[k], dtype=torch.long),
                                     persistent=False)
            grouped = [s for k in self.present_types for s in type_slots[k]]
            slot_order = [0] * self.num_tokens
            for pos, s in enumerate(grouped):
                slot_order[s] = pos
            self.register_buffer("slot_order", torch.tensor(slot_order, dtype=torch.long), persistent=False)
        self.type_weight = nn.Parameter(torch.randn(num_types, feat_dim, d_model) * feat_dim ** -0.5)
        self.type_bias = nn.Parameter(torch.zeros(num_types, d_model))
        self.entity_embed = nn.Embedding(num_entities, d_model, padding_idx=0)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # onnxruntime's memory-efficient attention kernel takes an additive
        # bias only when the fp16 sequence length is a multiple of 8; the
        # padding tokens are masked keys and their outputs are never read.
        # Variable-length inputs arrive padded that way (padded_token_count).
        self.seq_pad = -(self.num_tokens + 1) % 8 if attention == "fused" and not self.variable_tokens else 0
        block_tokens = None if self.variable_tokens else self.num_tokens + 1 + self.seq_pad
        self.blocks = nn.ModuleList(
            _TransformerBlock(block_tokens, d_model, num_heads, ffn_dim, attention)
            for _ in range(num_layers))
        if self.token_positions:
            self.rel_bias = nn.Parameter(torch.zeros((2 * self.rel_radius + 1) ** 2 + 1, num_heads))
        self.final_norm = nn.LayerNorm(d_model)
        self.pointer_head = nn.Linear(d_model, num_ptr_heads)
        self.global_policy = nn.Linear(d_model, policy_dim)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_players), nn.Tanh())
        self.embedding_dim = d_model

        ptr_index, ptr_mask = [], []
        for a, (t, hd) in enumerate(zip(action_token, action_head)):
            if (t < 0) != (hd < 0):
                raise ValueError(f"action {a}: action_token={t} / action_head={hd} must both be -1 or both set")
            if (self.num_tokens is not None and t >= self.num_tokens) or hd >= num_ptr_heads:
                raise ValueError(f"action {a}: token {t} / head {hd} out of range")
            ptr_mask.append(t >= 0)
            ptr_index.append((t + 1) * num_ptr_heads + hd if t >= 0 else 0)
        self.register_buffer("ptr_index", torch.tensor(ptr_index, dtype=torch.long), persistent=False)
        self.register_buffer("ptr_mask", torch.tensor(ptr_mask, dtype=torch.bool), persistent=False)

        if self.anchor_embedding:
            self.anchor_embed = nn.Embedding(self.num_anchors + 1, d_model, padding_idx=0)
        if self.num_anchors:
            anchor_index = [0] * policy_dim
            anchor_mask = [False] * policy_dim
            for i, a in enumerate(anchor_action):
                if a < 0:
                    continue
                if a >= policy_dim:
                    raise ValueError(f"anchor_action[{i}] = {a} out of range")
                if ptr_mask[a] or anchor_mask[a]:
                    raise ValueError(f"action {a} is bound more than once (static and/or anchor)")
                anchor_mask[a] = True
                anchor_index[a] = i
            self.register_buffer("anchor_index", torch.tensor(anchor_index, dtype=torch.long), persistent=False)
            self.register_buffer("anchor_mask", torch.tensor(anchor_mask, dtype=torch.bool), persistent=False)

    def forward(self, x: torch.Tensor):
        policy, value, _ = self.forward_with_embedding(x)
        return policy, value

    def forward_with_embedding(self, x: torch.Tensor):
        if self.variable_tokens:
            tok = x.reshape(x.shape[0], -1, self.token_dim)
            n = tok.shape[1]
        else:
            tok = x.reshape(-1, self.num_tokens, self.token_dim)
            n = self.num_tokens
        valid = tok[:, :, 0]
        entity = tok[:, :, 1].long()
        feats = tok[:, :, self.feat_offset:]
        if self.variable_tokens:
            kinds = self.type_onehot.to(tok.dtype)[tok[:, :, 3].long()]
            h = ((feats.unsqueeze(2) * kinds.unsqueeze(3)).flatten(2) @ self.type_weight.flatten(0, 1)
                 + kinds @ self.type_bias)
        else:
            h = torch.cat([
                feats.index_select(1, getattr(self, f"type_slots_{k}")) @ self.type_weight[k] + self.type_bias[k]
                for k in self.present_types], dim=1).index_select(1, self.slot_order)
        h = h + self.entity_embed(entity)
        if self.reads_anchor:
            anchor = tok[:, :, 2].long()
        if self.anchor_embedding:
            h = h + self.anchor_embed(anchor)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        key_valid = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        if self.seq_pad:
            h = torch.cat([h, h.new_zeros(h.shape[0], self.seq_pad, h.shape[2])], dim=1)
            key_valid = torch.cat([key_valid, valid.new_zeros(valid.shape[0], self.seq_pad)], dim=1)
        attn_bias = ((1.0 - key_valid) * -1e4)[:, None, None, :]
        if self.token_positions:
            # SDPA is ~10x slower on a non-contiguous bias, hence the contiguous().
            gather = F.embedding if torch.onnx.is_in_onnx_export() else _BucketGather.apply
            attn_bias = (gather(self._relative_index(tok), self.rel_bias).permute(0, 3, 1, 2)
                         + attn_bias).contiguous()
        for blk in self.blocks:
            h = blk(h, attn_bias)
        h = self.final_norm(h)
        g = h[:, 0]
        ptr = self.pointer_head(h)
        ptr_logits = ptr.flatten(1).index_select(1, self.ptr_index)
        policy = torch.where(self.ptr_mask, ptr_logits, self.global_policy(g))
        if self.num_anchors:
            # Anchors are unique per state, so the scatter is well-defined; all
            # anchorless tokens collide on row 0, which is dropped.
            heads = ptr.shape[-1]
            anchor_logits = ptr.new_zeros(ptr.shape[0], self.num_anchors + 1, heads).scatter(
                1, anchor.unsqueeze(-1).expand(-1, -1, heads), ptr[:, 1:1 + n])[:, 1:].flatten(1)
            policy = torch.where(self.anchor_mask, anchor_logits.index_select(1, self.anchor_index), policy)
        return policy, self.value_head(g), g

    def _relative_index(self, tok: torch.Tensor) -> torch.Tensor:
        """[B, S, S] bucket per (query, key) pair over CLS + tokens: 0 when
        either side has no position, else 1 + the clipped (dx, dy) cell.
        Arithmetic stays in float (exact for small integers, also in fp16)
        so the exported graph has no bool or integer ops besides the final cast."""
        r = float(self.rel_radius)
        pos = torch.cat([tok.new_zeros(tok.shape[0], 1, 3), tok[:, :, 4:7]], dim=1)
        has, px, py = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]
        dx = torch.clamp(px[:, :, None] - px[:, None, :], -r, r) + r
        dy = torch.clamp(py[:, :, None] - py[:, None, :], -r, r) + r
        both = has[:, :, None] * has[:, None, :]
        return ((dx * (2.0 * r + 1.0) + dy + 1.0) * both).long()


class AuxHead(nn.Module):
    """Training-only auxiliary head on the PV net's shared embedding (score
    margin per seat, or the opponent's next policy logits). Never exported to
    ONNX — it only shapes the trunk through its gradient."""

    def __init__(self, embedding_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, out_dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


def create_model_from_config(cfg: dict, *,
                              input_dim: int | None = None,
                              policy_dim: int | None = None,
                              num_players: int | None = None,
                              token_layout: dict | None = None) -> nn.Module:
    """Build a PVNet. Two calling conventions:

    1. **New (preferred)** — `cfg` is the train.json `network` block; dims
       are passed as kwargs from game.json + engine metadata::

           create_model_from_config(train_cfg["network"],
                                     input_dim=feature_dim,
                                     policy_dim=action_space,
                                     num_players=num_players)

    2. **Legacy / test helper** — `cfg` is a full dict containing
       `feature_dim` / `action_space` / `num_players` at the top level,
       and optionally a `network` block::

           create_model_from_config(game_config)

       Used by tests that load via `load_game_config` and don't separately
       parse train.json. If no `network` block is present, defaults
       (plain MLP, 256×2) are used.
    """
    if input_dim is None and policy_dim is None and num_players is None:
        input_dim = cfg["feature_dim"]
        policy_dim = cfg["action_space"]
        num_players = cfg["num_players"]
        network_cfg = cfg.get("network", {})
    else:
        if input_dim is None or policy_dim is None or num_players is None:
            raise ValueError(
                "create_model_from_config: pass all of input_dim/policy_dim/"
                "num_players together, or none (legacy single-arg form)"
            )
        network_cfg = cfg
    _validate_pv_network_config(network_cfg)
    if network_cfg.get("type", "mlp") == "token_transformer":
        if token_layout is None:
            raise ValueError(
                "network.type='token_transformer' requires an encoder with a token "
                "layout (engine.game_metadata(game_id)['token_layout'] is None); "
                "use a token variant such as splendor_tok.")
        if token_layout.get("variable_tokens"):
            if input_dim != 0:
                raise ValueError(f"variable_tokens layout needs feature_dim 0, got input_dim {input_dim}")
        elif token_layout["num_tokens"] * token_layout["token_dim"] != input_dim:
            raise ValueError(
                f"token layout {token_layout['num_tokens']}x{token_layout['token_dim']} "
                f"!= input_dim {input_dim}")
        return TokenTransformerPVNet(
            token_layout, policy_dim, num_players,
            d_model=network_cfg["d_model"], num_layers=network_cfg["num_layers"],
            num_heads=network_cfg["num_heads"], ffn_dim=network_cfg["ffn_dim"],
            attention=network_cfg.get("attention", "matmul"),
            relative_position_radius=network_cfg.get("relative_position_radius"),
            anchor_embedding=network_cfg.get("anchor_embedding", True))
    hidden_layers = network_cfg.get("hidden_layers", [256, 256])
    residual = bool(network_cfg.get("residual", False))
    block_size = int(network_cfg.get("block_size", 2))
    return PVNet(input_dim, policy_dim, hidden_layers, num_players=num_players,
                 residual=residual, block_size=block_size)


class BeliefNet(nn.Module):
    """Belief-network: MLP (optionally residual) with optional BatchNorm.

    Input: flat feature vector (size = belief_feature_dim).
    Output: flat logits (size = (N-1) * K), reshaped client-side to (N-1, K).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: Sequence[int],
                 batch_norm: bool = True,
                 *, residual: bool = False, block_size: int = 2):
        super().__init__()
        backbone, out_width = _build_backbone(
            input_dim, hidden_layers,
            residual=residual, block_size=block_size, batch_norm=batch_norm,
            where="BeliefNet")
        self.net = nn.Sequential(*backbone, nn.Linear(out_width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_belief_model_from_config(belief_config: dict, *,
                                    input_dim: int,
                                    output_dim: int) -> BeliefNet:
    validate_belief_config(belief_config)
    hidden = belief_config["architecture"]
    batch_norm = belief_config["batch_norm"]
    residual = bool(belief_config["residual"])
    block_size = int(belief_config["block_size"])
    return BeliefNet(input_dim, output_dim, hidden, batch_norm=batch_norm,
                     residual=residual, block_size=block_size)


def export_belief_onnx(net: BeliefNet, path: Path, input_dim: int) -> str:
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    # BN with batch=1 in eval mode reads running stats — safe. The exported
    # graph keeps a batch axis for runtime convenience even though the C++
    # evaluator always feeds batch=1.
    saved_device = next(net.parameters()).device
    if saved_device.type != "cpu":
        net.to("cpu")
    try:
        dummy = torch.zeros((1, input_dim), dtype=torch.float32)
        torch.onnx.export(
            net, dummy, str(path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
            dynamo=False,
        )
    finally:
        if saved_device.type != "cpu":
            net.to(saved_device)
    return str(path)


def import_pv_onnx_state_dict(path: Path, target_net: nn.Module) -> dict:
    """Decode ONNX initializers back into a state_dict matching `target_net`.

    Used by `--resume` when `checkpoint.pt` is missing — recovers PV weights
    from the latest `model_step_*.onnx`. Only weights survive: optimizer
    state (Adam moments, LR schedule progress) is unrecoverable from ONNX
    and is reset by the caller.

    The legacy TorchScript-based ONNX exporter we use in `export_onnx`
    preserves PyTorch parameter names as initializer names, except for
    bias-free matmul weights (e.g. transformer projections), which become
    anonymous `onnx::MatMul_*` initializers, possibly transposed. Those are
    located by exporting a probe of the same architecture with distinct
    weights: the exporter names initializers deterministically for a given
    architecture, so the probe's name ↔ parameter correspondence carries over.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    onnx_tensors: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    target_sd = target_net.state_dict()
    unnamed = [k for k in target_sd if k not in onnx_tensors]
    anonymous = _locate_anonymous_initializers(model, target_net, unnamed) if unnamed else {}
    out: dict = {}
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for k, v in target_sd.items():
        if k in onnx_tensors:
            arr = onnx_tensors[k]
        elif k in anonymous and all(n in onnx_tensors for n in anonymous[k][0]):
            names, decode = anonymous[k]
            arr = decode(*(onnx_tensors[n] for n in names))
        else:
            missing.append(k)
            continue
        if tuple(arr.shape) != tuple(v.shape):
            shape_mismatch.append(
                f"{k}: onnx={tuple(arr.shape)} vs net={tuple(v.shape)}"
            )
            continue
        out[k] = torch.from_numpy(arr.copy()).to(dtype=v.dtype)
    if missing or shape_mismatch:
        raise ValueError(
            f"import_pv_onnx_state_dict({path}): cannot reconstruct net.\n"
            f"  missing keys: {missing}\n"
            f"  shape mismatch: {shape_mismatch}\n"
            f"The ONNX file does not match the configured network architecture. "
            f"This usually means train.json was changed after the run started."
        )
    return out


def _locate_anonymous_initializers(model, target_net: nn.Module, keys: list[str]) -> dict:
    """Map state_dict keys absent from the initializer names to
    (initializer names, decode) by exporting a probe of `target_net`'s
    architecture whose weights are distinct random values. Recognised forms:
    verbatim, transposed, row-gathered (`param[index_buffer]` constant-folded
    by the exporter; every row of the parameter must appear), and split
    (each `param[i]` its own initializer). Anything else is left out (the
    caller reports it as missing)."""
    import copy
    import tempfile
    import numpy as np
    import onnx
    from onnx import numpy_helper

    input_dim = 0
    if not getattr(target_net, "variable_tokens", False):
        input_dim = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        if input_dim <= 0:
            raise ValueError("ONNX input has no static feature dimension")
    probe = copy.deepcopy(target_net).to("cpu")
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in probe.parameters():
            p.copy_(torch.randn(p.shape, generator=gen))
    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "probe.onnx"
        export_onnx(probe, probe_path, input_dim)
        probe_inits = {i.name: numpy_helper.to_array(i)
                       for i in onnx.load(str(probe_path)).graph.initializer}
    probe_sd = probe.state_dict()
    found: dict = {}
    for k in keys:
        want = probe_sd[k].numpy()
        for name, arr in probe_inits.items():
            if arr.shape == want.shape and np.array_equal(arr, want):
                found[k] = ((name,), lambda a: a)
                break
            if want.ndim == 2 and arr.shape == want.T.shape and np.array_equal(arr, want.T):
                found[k] = ((name,), lambda a: a.T)
                break
            if arr.ndim == want.ndim and arr.shape[1:] == want.shape[1:]:
                rows = _gathered_rows(arr, want)
                if rows is not None:
                    found[k] = ((name,), lambda a, rows=rows: a[rows])
                    break
        if k not in found and want.ndim >= 2:
            names = [next((n for n, arr in probe_inits.items()
                           if arr.shape == want.shape[1:] and np.array_equal(arr, want[i])), None)
                     for i in range(want.shape[0])]
            if None not in names:
                found[k] = (tuple(names), lambda *parts: np.stack(parts))
    return found


def _gathered_rows(arr, want):
    """If every row of `arr` equals a row of `want` and every row of `want`
    occurs, return, for each row of `want`, one index into `arr` holding it."""
    import numpy as np
    first = {}
    for i in range(arr.shape[0]):
        hits = [j for j in range(want.shape[0]) if np.array_equal(arr[i], want[j])]
        if len(hits) != 1:
            return None
        first.setdefault(hits[0], i)
    if len(first) != want.shape[0]:
        return None
    return [first[j] for j in range(want.shape[0])]


def convert_onnx_fp16(src: Path, dst: Path) -> str:
    """fp16 copy of an exported fp32 model with fp32 inputs/outputs, so the C++
    evaluator feeds and reads it unchanged."""
    import onnx
    from onnxconverter_common import float16
    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return str(dst)


def padded_token_count(n: int) -> int:
    """Token count a batch of variable-length encodings is padded to; mirrors
    padded_token_count in engine/core/feature_encoder.h."""
    return (n + 8) // 8 * 8 - 1


def export_onnx(net: nn.Module, path: Path, input_dim: int) -> str:
    """`input_dim` is the feature length; variable-length token nets ignore it
    and export a dynamic feature axis."""
    net.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ONNX export must happen on CPU regardless of training device. Move the
    # net to CPU for the export call, then back. The state_dict is CPU-side
    # the whole time on Apple mps anyway (mps weights live in unified memory),
    # so the move is cheap.
    saved_device = next(net.parameters()).device
    if saved_device.type != "cpu":
        net.to("cpu")
    try:
        variable = getattr(net, "variable_tokens", False)
        if variable:
            input_dim = padded_token_count(1) * net.token_dim
        dummy = torch.zeros((1, input_dim), dtype=torch.float32)
        output_names = ["policy", "value"]
        dynamic_axes = {
            "features": {0: "batch", 1: "features"} if variable else {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
        torch.onnx.export(
            net, dummy, str(path),
            input_names=["features"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )
        _place_mha_attention_bias(path)
    finally:
        if saved_device.type != "cpu":
            net.to(saved_device)
    return str(path)
