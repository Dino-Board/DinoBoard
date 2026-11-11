# Harmonies — AI API 接入文档

## 概览

- **game_id**：
  - B 面（岛屿计分，默认）：`harmonies` / `harmonies_2p` / `harmonies_3p` / `harmonies_4p`
  - A 面（河流计分）：`harmonies_a` / `harmonies_a_2p` / `harmonies_a_3p` / `harmonies_a_4p`
  - B 面 2 人、token 编码器（Transformer）：`harmonies_tok`
- **玩家数**：2–4
- **范围**：个人版图 A/B 两面；不含 Nature's Spirit 牌和单人模式；0 号位先手
- **动作空间**：B 面 988，A 面 912（与人数无关）
- **隐藏信息**：动物牌堆里剩哪些牌对所有人隐藏（`animal_deck`），张数公开。由于市场、各家手里和已完成的牌都是公开的，牌堆成员其实可以完全推出
- **公开事件**：无（全部靠 `public_snapshot`）

### 两面版图

版图几何由 `scripts/gen_card_data.py` 生成到 `harmonies_board_data.h`（每面：轴坐标、邻接、对称群），规则、编码器、启发式和网页都从这里取，不手写坐标。平顶六边形按列排布，hex 编号按列从左到右、列内从上到下；奇数列比偶数列低半格。轴坐标 `(q, r)`：`q` = 列号，`r` = 列内序号 − ⌊q/2⌋。

| 面 | 列高 | 格数 | 水的计分 | 来源 |
|----|------|------|----------|------|
| A | 5-4-5-4-5 | 23 | 最佳河流 | 规则书 p.2 布置图 |
| B | 4-3-4-3-4-3-4 | 25 | 岛屿 | 规则书 p.8「Water – side B」旁的 B 面版图 |

状态字典带 `board_side`（`"A"`/`"B"`）、`board_columns`（每列的 hex id）、`action_layout`（各动作段起点），网页据此画版图、解码动作。

### 模型兼容

- 随附模型改名为 `games/harmonies/model/harmonies_a_2p.onnx`（A 面 MLP，输入 1190 维，输出 912）。它只用于 `harmonies_a` / `harmonies_a_2p`；B 面变体的输入维度（1250）和动作空间（988）都不同，误加载会在推理时直接报维度错误，而且 B 面按约定的文件名 `harmonies_2p.onnx` 现在不存在，平台不会自动找到 A 面模型。
- 平台（`platform/model_paths.py`）只认 `_<N>p` 后缀，`harmonies_a_2p` 的模型不会被网页自动发现；需要时显式传 `model_path`。训练脚本（`training/cli.py`、`training/mcts_profile.py`）把 `_a`、`_tok` 都归到 `games/harmonies/config/`。
- B 面的 MLP 与 token 网络都需要重新训练。

---

## 动作编码

`H` = 格数（A 面 23，B 面 25）。

| 范围 | 含义 |
|------|------|
| 0–4 | 选中央版图第 `slot` 格，拿走其上 3 个令牌（回合第一步） |
| 5–36 | 从市场拿动物牌 `card_id = action − 5 + 1`（每回合至多一次，手里至多 4 张未完成的牌） |
| 37 … 37+6H−1 | 放令牌：`rel = action − 37`，`hex = rel // 6`，`color = rel % 6 + 1`（1 蓝 2 灰 3 棕 4 绿 5 黄 6 红）。A 面 37–174，B 面 37–186 |
| 37+6H … 37+38H−1 | 放动物方块：`rel = action − (37+6H)`，`hex = rel // 32`，`card_id = rel % 32 + 1`。A 面 175–910，B 面 187–986 |
| 37+38H | 结束回合（3 个令牌都放完后才合法）。A 面 911，B 面 987 |

一个回合是多步子协议：选格 → 放 3 个令牌 → 结束回合；拿牌和放方块可以穿插在选格之后的任意时刻。

---

## 规则要点

- 堆叠：灰可叠灰（山，最高 3）；棕可叠棕（最多 2 棕）；绿可放在 1–2 个棕上（树）；红可放在单个棕/灰/红上（建筑）；蓝、黄不可堆叠；有动物方块的格子不能再放令牌。
- 计分：树 1/3/7；与其它山相邻的山 1/3/7；每片 ≥2 的田 5；周围至少 3 种颜色的建筑 5；动物牌按已放方块数取牌面分值。
- 水，A 面：最长河流（同一片蓝色中两格间最短路径的格数，含两端）0/2/5/8/11/15，超过 6 每格 +4。
- 水，B 面：每个岛 5 分。岛 = 被蓝色令牌隔开的一格或一片相连的格（空格也算，规则书原文 "each space or group of spaces separated from each other by blue tokens"，示例里的岛几乎全是空格）；至少算 1 个岛，所以空版图也有 5 分（开局 `landscape_score` 即为 5）。
- 结束触发：补中央版图时袋子不够，或某玩家版图空格 ≤ 2；打完当前这一轮（所有人回合数相同）后结束。
- **终局排名（规则书）**：总分高者胜；同分比**已放置的动物方块数**；仍相同则共享胜利。

### 终局结果与价值

| 终局情形 | `winner` | `winners` | `losers` | 各家价值 |
|----------|----------|-----------|----------|----------|
| 唯一胜者（含靠方块数决胜） | 该玩家 | `[w]` | 其余所有人 | 胜者 +1，其余各 −1/(N−1) |
| 3–4 人局部分玩家并列第一（方块数也相同） | `-1` | 所有并列第一者 | 其余所有人 | 并列者各 +1/nw，其余各 −1/nl |
| 所有人并列（含 2 人平局） | `-1` | `[]` | `[]` | 全 0（平局） |

### 步数上限

`game.json::max_game_plies` 不低于一局可能的最大动作数，因此对局不会被截断：袋子最多支撑 36 个回合，每回合 5 个必做动作；32 张动物牌各最多被拿一次；每个方块占自己版图上不同的一格（每人至多 H 个，且不超过牌上方块总数 96）。上界 = `36×5 + 32 + min(H×N, 96)`，B 面 4 人为 308。

---

## 版图对称（训练数据增强）

两面的对称群都是 4 个元素：恒等、旋转 180°、左右镜像、上下镜像（生成脚本枚举六边形格点的 12 个旋转/反射并检查哪些把版图映到自身）。规则只按 6 个旋转匹配动物图案，镜像之所以也合法，是因为 32 张牌的图案全部是非手性的（镜像等于某个旋转，生成脚本会断言这一点）。所有地形计分只依赖邻接关系。`tests/harmonies/test_symmetry.py` 用实验验证：一局棋按对称 g 变换每个动作重放，每一步合法动作一一对应、版图逐格对应、分数和胜负相同；两个编码器满足 `encode(g s) = transform_g(encode(s))`，合法掩码同样对应。

编码器通过 `IFeatureEncoder::symmetries()` 给出每个对称下样本怎么变（MLP：特征下标置换；token：锚点置换、位置的轴坐标仿射变换、对手格 `neighbour` 按方向的列置换；两者共用动作置换）。`train.json` 里 `training.symmetry_augmentation: true` 时，训练在取批时给每个样本随机选一个对称（`training/symmetry.py`），回放缓冲不变大。

---

## Token 编码器（`harmonies_tok`）

变长 token、带位置（`TokenLayout::variable_tokens` / `token_positions`）。每个 token 的头部 7 列：`valid, entity, anchor, type, has_position, q, r`，特征从第 7 列开始，`token_dim = 86`（最宽的 `opp_cell` 有 79 个特征）。"我" = 编码视角（查询策略时就是行动方）。

- `global` ×1，`player` ×N（我在前，其余按座位顺序），`slot` ×5，`card`（市场里每张 + 每家手里每张，entity = 牌 id），`cell` ×25（我的版图，位置 = 该格轴坐标），`opp_cell` ×25×(N−1)（对手版图，**不带位置**，所以相对位置偏置只在我自己的格子之间起作用），`cube`（每个当前合法的放方块动作一个，位置 = 该格，entity = 牌 id）。
- 策略：6 个指针头。我的格子的头 c = 在此放颜色 c；`slot` 的头 0 = 选这格；市场牌的头 0 = 拿这张；`cube` 的头 0 = 放这个方块；结束回合用全局头。放方块绑定到 `cube` token 而不是格子上的 32 个"牌 id 头"：锚点绑定必须是静态的（锚点 × 头 → 动作），而手牌的位置不固定，格子上只能按牌 id 开头，那 32 个头每个都只在拿着那张牌时才有训练信号；`cube` token 坐在目标格上，带上牌的信息，一个共享的头就能给所有 (格, 牌) 打分。
- 我的手牌按牌 id 从小到大排名（`rank`），格子上的 `card_*` 第 k 组就是排名 k 的那张牌。
- "最佳部分匹配"：把图案按规则（锚点块放在某格、6 个旋转之一）放到版图上，方块格不能有动物、每个未满足的块还能继续叠到要求（不能叠在有动物或不兼容的塔上）；所需令牌数 = 各块还差的令牌数之和；最佳 = 所需令牌数最少的放法（`missing`）。`card_advance` = 在此格放这个颜色后最少所需令牌数减少多少（变多为负，放法全部失效按 +3 计），÷3 截到 [−1, 1]。

#### `global`

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0 | 1 | `turn_started` | 已选中央格 |
| 1 | 1 | `took_card` | 本回合已拿牌 |
| 2 | 1 | `end_triggered` | 终局已触发 |
| 3 | 1 | `my_turn` | 轮到我 |
| 4-9 | 6 | `pool` | 行动方手里待放的各色令牌 ÷3 |
| 10 | 1 | `pool_total` | 待放令牌总数 ÷3 |
| 11-16 | 6 | `pouch` | 袋中各色 ÷23 |
| 17 | 1 | `pouch_total` | 袋中总数 ÷120 |
| 18 | 1 | `deck` | 牌堆张数 ÷27 |
| 19 | 1 | `market_cards` | 市场张数 ÷5 |
| 20 | 1 | `refills_left` | ⌊袋中数/3⌋ ÷36 |
| 21 | 1 | `min_spaces` | 各家最少空格 ÷25 |

#### `player`

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0-3 | 4 | `seat` | 相对座位 one-hot（0 = 我） |
| 4 | 1 | `to_move` | 是行动方 |
| 5 | 1 | `first_player` | 是起始玩家 |
| 6 | 1 | `animal_score` | 动物分 ÷50 |
| 7 | 1 | `landscape_score` | 地形分 ÷50 |
| 8 | 1 | `total_score` | 总分 ÷100 |
| 9 | 1 | `margin` | 总分 − 其他人最高分，÷50 |
| 10 | 1 | `trees` | 树分 ÷20 |
| 11 | 1 | `mountains` | 山分 ÷20 |
| 12 | 1 | `fields` | 田分 ÷20 |
| 13 | 1 | `buildings` | 建筑分 ÷20 |
| 14 | 1 | `water` | 水分（岛）÷20 |
| 15 | 1 | `islands` | 岛数 ÷8 |
| 16 | 1 | `spaces` | 空格 ÷25 |
| 17 | 1 | `cubes_placed` | 已放方块 ÷25 |
| 18 | 1 | `cards_held` | 手牌数 ÷4 |
| 19 | 1 | `card_points_open` | 手牌上还能拿到的分 ÷40 |

#### `slot`

锚点 = `slot` 段 + 格号（5 格都有 token，空格计数为 0）。

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0-5 | 6 | `tokens` | 各色令牌数 ÷3 |
| 6 | 1 | `total` | 令牌数 ÷3 |
| 7 | 1 | `best_landscape` | 这些颜色在我版图上单放一枚能得到的最大地形分变化 ÷10 |
| 8 | 1 | `advancing` | 颜色能让某张手牌所需令牌数减少（在某格）的令牌数 ÷3 |
| 9 | 1 | `completing` | 颜色能让某张手牌立刻可放方块的令牌数 ÷3 |

#### `card`

市场牌锚点 = `market_card` 段 + 牌 id − 1（头 0 = 拿牌）；手牌无锚点。进度相对持牌人版图（市场牌相对我的版图）。

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0 | 1 | `market` | 在市场 |
| 1 | 1 | `my_hand` | 在我手里 |
| 2 | 1 | `opp_hand` | 在对手手里 |
| 3-6 | 4 | `holder_seat` | 持牌人相对座位（市场牌全 0） |
| 7-10 | 4 | `rank` | 我的手牌按 id 的排名（对应格子上的 `card_*` 组） |
| 11 | 1 | `pattern_size` | 图案格数 ÷4 |
| 12-17 | 6 | `need` | 从空格搭出整个图案要的各色令牌 ÷4（建筑算 2 红） |
| 18-23 | 6 | `habitat` | 方块格的顶色 one-hot |
| 24 | 1 | `cubes_total` | 方块总数 ÷5 |
| 25 | 1 | `cubes_left` | 剩余方块 ÷5 |
| 26 | 1 | `points_now` | 当前得分 ÷20 |
| 27 | 1 | `next_points` | 下一个方块的增量分 ÷10 |
| 28 | 1 | `full_points` | 完成时得分 ÷20 |
| 29 | 1 | `missing` | 最佳部分匹配还差的令牌数，截到 6 再 ÷6（无可行放法为 1） |
| 30 | 1 | `no_fit` | 没有可行放法 |
| 31 | 1 | `ready` | 现在能放方块的格数 ÷25 |
| 32 | 1 | `opp_missing` | 仅市场牌：对手里最少的 `missing` |

#### `cell`

锚点 = `cell` 段 + hex，位置 = 轴坐标。

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0-12 | 13 | `tower` | 塔类型 one-hot |
| 13 | 1 | `height` | 层数 ÷3 |
| 14 | 1 | `animal` | 有动物方块 |
| 15 | 1 | `points` | 这格的树/山/建筑当前得分 ÷7 |
| 16 | 1 | `field_group` | 所在黄色片的格数，截到 5 再 ÷5 |
| 17 | 1 | `island` | 非蓝格：所在岛的格数 ÷25 |
| 18 | 1 | `island_split` | 蓝格：相邻的不同岛数 ÷3 |
| 19 | 1 | `nb_colours` | 邻格顶色种数 ÷6 |
| 20 | 1 | `nb_empty` | 空邻格数 ÷6 |
| 21 | 1 | `edge` | 出界方向数 ÷6 |
| 22-27 | 6 | `can_place` | 颜色 c 能叠在这里（不看手里有没有） |
| 28-33 | 6 | `legal` | 现在放颜色 c 到这里合法 |
| 34-39 | 6 | `delta` | 放颜色 c 后我的地形分变化 ÷10 |
| 40-43 | 4 | `card_part` | 排名 k 的手牌：这格属于某个最佳部分匹配 |
| 44-47 | 4 | `card_missing` | 这格是某个最佳部分匹配里还没满足的块 |
| 48-51 | 4 | `card_ready` | 现在能把这张牌的方块放在这格 |
| 52-75 | 24 | `card_advance` | 第 k×6+c 列：放颜色 c 后这张牌的最少所需令牌数减少量 ÷3 |

#### `opp_cell`

无锚点、无位置、无 entity。

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0-2 | 3 | `owner_seat` | 所属对手的相对座位 − 1 |
| 3-15 | 13 | `tower` | 塔类型 one-hot |
| 16 | 1 | `height` | 层数 ÷3 |
| 17 | 1 | `animal` | 有动物方块 |
| 18 | 1 | `points` | 同 `cell` |
| 19 | 1 | `field_group` | 同 `cell` |
| 20 | 1 | `island` | 同 `cell` |
| 21 | 1 | `island_split` | 同 `cell` |
| 22 | 1 | `nb_colours` | 同 `cell` |
| 23 | 1 | `nb_empty` | 同 `cell` |
| 24 | 1 | `edge` | 同 `cell` |
| 25-30 | 6 | `delta` | 放颜色 c 后该对手的地形分变化 ÷10 |
| 31-78 | 48 | `neighbour` | 方向 d（N NE SE S SW NW）邻格的类别 one-hot，第 d×8+k 列：k=0 出界，1 空，2–7 顶色蓝…红 |

#### `cube`

锚点 = `cube` 段 + hex×32 + 牌 id − 1，位置 = 该格，entity = 牌 id；只在我回合中（已选格）对合法的放方块动作出现。

| 偏移 | 宽 | 名称 | 含义 |
|------|----|------|------|
| 0-3 | 4 | `rank` | 牌在我手牌中的排名 |
| 4 | 1 | `points` | 这个方块的增量分 ÷10 |
| 5 | 1 | `completes` | 放完这张牌就完成 |
| 6 | 1 | `cubes_left` | 放后剩余方块 ÷5 |
| 7 | 1 | `blocks` | 放这里会让我另外几张手牌的 `missing` 变大 ÷4 |

#### `anchors`

| 段 | 首个锚点 id | 数量 |
|----|-------------|------|
| `cell` | 1 | 25 |
| `slot` | 26 | 5 |
| `market_card` | 31 | 32 |
| `cube` | 63 | 800 |

#### `entities`

| 段 | 首个 id | 数量 |
|----|---------|------|
| `card` | 1 | 32 |

`tests/harmonies/test_token_encoder.py` 检查这些表与引擎报告的布局一致，并用 `tests/harmonies/harmonies_reference.py`（独立实现的堆叠规则、图案匹配、计分）逐个特征核对真实对局。

训练：`games/harmonies/config/train_transformer.json`（`relative_position_radius` 6 覆盖整个 B 面，`anchor_embedding: false` 让网络只通过相对位置看格子，`symmetry_augmentation: true`）：

```bash
python -m training.cli --game harmonies_tok --output runs/harmonies_tok \
    --train-config games/harmonies/config/train_transformer.json
```

---

## 快照字段（schema）

全公开：`current_player`、`game_first_player`、`winner`、`terminal`、`end_triggered`、`turn_started`、`took_card_this_turn`、`selected_tokens[6]`、`central_board_counts[5][6]`、`animal_market[5]`（牌 id，0 = 空）、`animal_deck_count`、`pouch_counts[6]`、`tower_type[N][H]`、`has_animal[N][H]`、`held_cards[N][4]`、`cube_remaining[N][4]`、`cube_initial[N][4]`、`space_remaining[N]`、`animal_scores[N]`、`landscape_scores[N]`、`completed_cards[32]`（1 = 该牌方块已放完、已离手）。

隐藏：`animal_deck[32]`（1 = 还在牌堆里）。`randomize_unseen` 用"32 张牌 − 市场 − 各家手里 − 已完成"重建它；补市场时用 rng 从中均匀抽一张，牌堆没有顺序。

`tower_type` 取值：`0` 空，`1..6` 单层（蓝/灰/棕/绿/黄/红），`7` 建筑，`8` 双棕，`9` 棕+绿，`10` 双灰，`11` 棕+棕+绿，`12` 三灰。
