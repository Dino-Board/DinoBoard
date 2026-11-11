# Jaipur — AI API 接入文档

## 概览

- **game_id**：`jaipur`
- **玩家数**：2
- **范围**：只打**一轮**（不是规则书的三局两胜）；0 号位先手；卖出时一次卖光手里该种货物
- **动作空间**：35（按效果编码，见下）
- **隐藏信息**：对手手牌的构成、牌堆的构成、面朝下奖励标记的面值（对手拿到的和堆里剩下的）
- **公开事件**：`goods_taken`、`goods_sold`、`exchange_resolved`（都是公开事件，所有视角相同）

---

## 动作编码

货物类型 `g`：`0=钻石 1=黄金 2=白银 3=香料 4=布料 5=皮革`。

| 范围 | 阶段 | 含义 |
|------|------|------|
| 0–5 | 空闲 | 卖出手里全部 `g` 类货物（钻石/黄金/白银至少 2 张） |
| 6–11 | 空闲 | 从市场拿 1 张 `g` 类货物（手牌 < 7） |
| 12 | 空闲 | 拿走市场上全部骆驼 |
| 13 | 空闲 | 进入交换（之后必须完成，没有取消动作） |
| 14–19 | 交换·选市场牌 | 再选 1 张 `g` 类市场货物 |
| 20 | 交换·选市场牌 | 确认市场选择（至少 2 张且付得起） |
| 21–28 | 交换·选骆驼 | 用 `action − 21` 只骆驼支付 |
| 29–34 | 交换·选手牌 | 交出 1 张 `g` 类手牌（不能与拿的类型相同） |

交换是一个回合内的多步子协议：子步骤期间 `current_player` 不变。

---

## 随机性：没有预存的牌序

牌堆和奖励标记堆**都不保存顺序**：

- 牌堆是每类牌的张数 `deck_count[7]`（隐藏），抽牌是在抽的那一刻用调用方传入的 rng 按张数加权抽样。
- 每档奖励堆是每种面值的个数 `bonus_pile[3][3]`（隐藏），卖出 3/4/5+ 张时同样用 rng 加权抽一个。
- 因此任何 state（真值、会话、MCTS 模拟世界）里都不存在"下一张牌是什么"。真值对局可由 episode seed 复现（GT 的 step rng 由 seed 决定）；模拟世界用模拟自己的 rng，从 `randomize_unseen` 采样出的构成里抽。
- 规则在抽牌时会校验隐藏构成与公开张数一致；对一个没有经过 `randomize_unseen` 的观察者 state 直接执行抽牌动作会抛异常，而不是编造一张牌。

---

## 公开事件

| kind | payload | 何时发 |
|------|---------|--------|
| `goods_taken` | `{player, good_type}` | 动作 6–11 |
| `goods_sold` | `{player, good_type}` | 动作 0–5 |
| `exchange_resolved` | `{player, take_types: [g...], give_types: [g...]}` | 交换完成的那一步（`give_types` 可能为空：全部用骆驼支付） |

事件只喂给 tracker。玩家自己抽到的奖励面值**不走事件**，而是作为 owner-only 的 state 槽位 `bonus_taken` 随快照下发给本人。

---

## 快照字段（schema）

全公开：`current_player`、`winner`、`terminal`、`plies`、`market_count[7]`（6 类货物 + 骆驼）、`deck_pos`（已抽张数，牌堆余量 = 55 − `deck_pos`）、`camel_count[2]`、`rupees[2]`（货物标记分 + 骆驼标记）、`value_tokens[6]`（各类货物标记剩余个数）、`bonus_tokens_3/4/5`（三档奖励堆剩余个数）、`camel_bonus_token`、`piles_exhausted`、`bonus_count[2][3]`、`goods_tokens_taken[2]`、`discard_count[6]`（已卖出弃掉的各类货物张数）、交换过程的 `exch_*`。

仅本人可见（owner-only）：`hand_count[p][6]`、`bonus_total[p]`、`bonus_taken[p][3][3]`（本人抽到的各档各面值个数）。

对双方都隐藏：`deck_count[7]`、`bonus_pile[3][3]`。

`get_state_dict()` 是真值侧的完整转储（含对手手牌、`deck_count`、`bonus_pile`），只用于 GT / 测试 / 本地前端，不要转发给对手。

---

## 初始观察与中途接入

`extract_initial_observation(seat)` 的 `tracker_init` 携带：

| key | 内容 |
|-----|------|
| `own_hand` | 本人 6 类货物张数 |
| `sold` | 12 个 0/1：`[p*6+g]`，玩家 p 是否卖出过 g 类货物 |
| `net_in` | 12 个整数：`[p*6+g]`，玩家 p 对 g 类货物的净流入（已卖出类型即精确持有数，否则为下界） |

`sold` / `net_in` 是 tracker 的全部记忆，可以在**任意一步**取出：新会话用它初始化后，与从第 0 步开始逐步接收事件的会话信念完全一致。

---

## 终局

- 三种货物标记堆取空，或补市场时牌堆不够 → 本轮结束。
- 骆驼多者 +5（相同则都不加）。
- 胜者：`rupees + bonus_total` 高者；相同比奖励标记个数；再相同比货物标记个数；仍相同为平局（`winner = -1`，双方价值 0）。
- 达到 `max_game_plies` 仍未结束：走 adjudicator，双方价值都是 −1（拖延惩罚，有意的非零和）。

---

## 常见踩坑

- **事件里的空列表**：`exchange_resolved.give_types` 为空时，经 JSON/Python 往返后仍必须作为列表传回，tracker 会按"没有交出货物"处理；不要省略这个 key。
- **不要对观察者会话调用 `apply_action`**：观察者会话只通过 `apply_observation` 推进。
