# Azul — AI 接入文档

## 概览

- **game_id**：`azul` / `azul_2p` / `azul_3p` / `azul_4p`（扁平编码 + MLP）；`azul_tok` / `azul_tok_3p` / `azul_tok_4p`（token 编码 + transformer，规则与动作 id 完全相同，见「Token 编码」一节）
- **玩家数**：2–4
- **动作空间**：`(factories + 1) × 5 × 6`，2p/3p/4p 下的工厂数不同
  - 2p: 5 factories → 180
  - 3p: 7 factories → 240
  - 4p: 9 factories → 300
- **隐藏信息**：袋中瓷砖顺序是**对称随机**——所有玩家都不知道下一批翻到工厂上是什么颜色，但知道袋里各色的总数
- **公开事件**：`factory_refill`（post）——round 边界翻新工厂时发

---

## 动作编码

每个动作是一个三元组 `(source, color, target)`：

```python
factories_count = 5 if num_players == 2 else 7 if num_players == 3 else 9
COLORS = 5           # 蓝/黄/红/黑/白（索引 0..4）
TARGETS = 6          # 5 条 pattern line + 1 floor (索引 5)

source = action_id // (COLORS * TARGETS)     # 0..factories_count-1 = 工厂号；factories_count = center
color = (action_id // TARGETS) % COLORS       # 0..4
target = action_id % TARGETS                  # 0..4 = pattern line 0..4；5 = floor
```

含义：「从 `source`（某个工厂或 center）拿所有 `color` 色的瓷砖，放到 `target`（自己的 pattern line 或 floor）」。

规则简介：
- 每回合玩家选一个工厂（或 center）取某色全部瓷砖。工厂剩余的瓷砖推到 center。
- 拿到的瓷砖放到自己对应长度的 pattern line 上（line 0 长度 1，line 4 长度 5）。
- Pattern line 满员后在 wall 上成墙得分。
- 瓷砖溢出（已满 line、或颜色已在 wall 上）或 floor 都扣分。
- Round 在**工厂和 center 上没有瓷砖**时结束（先手标记不是瓷砖，不参与判定），随后结算并从袋中补给工厂。下一 round 由拿走先手标记的玩家先手。
- **先手标记整轮没人拿**（每次取砖都取空一个单色工厂、center 从未有砖）：标记留在 center，没有人为它扣分，下一 round 仍由本 round 的先手玩家先手。
- **一块瓷砖都补不出来**（袋和 box-lid 同时为空）：下一 round 无法进行，游戏立即按正常终局计分结束（行/列/同色奖励照算）。
- 终局排名：分数高者胜；同分比完成的横行数；仍相同则共享胜利（见下方 `winner` / `winners`）。
- floor 满 7 格后溢出的瓷砖直接移出游戏（不进 box-lid）——这是本实现采用的规则。

---

## 公开事件

### `factory_refill`（post）

当一个 round 结束，游戏从袋子和弃盘补给工厂时发一次。

```json
{
  "kind": "factory_refill",
  "payload": {
    "factories": [[...], [...], [...], [...], [...]]
  }
}
```

- `factories`：长度 = factories_count 的二维数组，`factories[i][c]` = 工厂 i 上色 c 的瓷砖数

袋中剩余的逐色计数（`bag_counts`）随 `public_snapshot` 整体覆写，不在事件 payload 里重复。

**何时发**：当 `round_index` 增加时，即新一 round 刚开始。普通动作（取瓷砖）之间**不发**。

---

## state_dict 字段（全部公开，Azul 无玩家私有字段）

`state_serializer` 输出（`games/azul/azul_register.cpp::serialize_azul`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 唯一胜者；`-1` = 还没结束，或终局时没有唯一胜者（共享胜利） |
| `num_players` | int | 2 / 3 / 4 |
| `round_index` | int | 当前 round（0 起） |
| `first_player_token_in_center` | bool | 先手标记是否还在 center |
| `scores` | `int[N]` | 每位玩家当前总分（与 `players[i].score` 同步） |
| `factories` | `int[F][5]` | F = 5/7/9，每个工厂上每色瓷砖数 |
| `center` | `int[5]` | center 上每色瓷砖数 |
| `bag_counts` | `int[5]` | 袋中每色剩余 |
| `bag_total` | int | `sum(bag_counts)` |
| `box_counts` | `int[5]` | 已弃盘（box-lid）每色累计 |
| `players[i].score` | int | 当前分数 |
| `players[i].pattern_lines` | `[{length, color, capacity}, ...]` | 长度 5；capacity = row+1（1..5） |
| `players[i].wall` | `int[5][5]` | 0/1 标记墙上哪些格子已成墙 |
| `players[i].floor` | `int[]` | floor 上瓷砖序列（颜色 id） |
| `players[i].floor_count` | int | floor 张数 |

### 终局结果与价值

`GameSession.winners` / `GameSession.losers`（即 `IGameState::winners()` / `losers()`）：

| 终局情形 | `winner` | `winners` | `losers` | 各家价值 |
|----------|----------|-----------|----------|----------|
| 唯一胜者 | 该玩家 | `[w]` | 其余所有人 | 胜者 +1，其余各 −1/(N−1) |
| 3–4 人局部分玩家并列第一（比完横行数仍相同） | `-1` | 所有并列第一者 | 其余所有人 | 并列者各 +1/nw，其余各 −1/nl |
| 所有人并列（含 2 人平局） | `-1` | `[]` | `[]` | 全 0（平局） |

---

## Token 编码（`azul_tok` / `azul_tok_3p` / `azul_tok_4p`）

`AzulTokenEncoder`（`games/azul/azul_net_adapter.{h,cpp}`）把局面编码成一组 token，供 `token_transformer` 网络使用。规则、动作 id、state_dict、事件都与扁平变体相同；扁平变体（`azul` 等）和它的 `AzulFeatureEncoder` 没有改动，旧模型照常可用。

**工厂顺序不影响决策**：一个「堆」token = 某个工厂（或 center）上某一种颜色的全部瓷砖，也就是一次动作拿走的东西。token 里没有任何「这是第几号工厂」的信息（只列出同一工厂上其余颜色各几块），网络也没有位置编码，所以把工厂任意重排后，网络给出的局面价值不变，各个动作的策略值跟着对应的堆走。`tests/azul/test_token_encoder.py` 用真实局面验证了这一点。

### 槽位

每个 token 是 `[valid, entity_id, 特征…]`，共 43 维（`entity_id` 恒为 0；不存在的 token 全 0）。按玩家排列的块都以视角玩家为第 0 位、按座次往后排。

| 块 | 数量 | 类型 id | 说明 |
|----|------|---------|------|
| 全局 | 1 | 0 | 轮次、袋子、弃盘 |
| 玩家 | P | 1 | 分数、floor、墙的汇总 |
| 图案行 | 5P | 2 | 每位玩家的每条 pattern line 及对应的墙行 |
| 堆 | (F+1)×5 | 3 | 槽位 = `source × 5 + color`；该颜色数量为 0 时 token 不存在 |

P = 玩家数，F = 工厂数。2p / 3p / 4p 分别是 43 / 59 / 75 个 token，`feature_dim` = 1849 / 2537 / 3225。

### 动作绑定

所有动作都绑定到堆 token：动作 `(source, color, target)` 的策略值 = 堆 `(source, color)` 的第 `target` 个指针头（6 个头：5 条 pattern line + floor）。没有走全局策略头的动作。

### 特征

全局（16）：

| 偏移 | 含义 |
|------|------|
| 0 | `min(round_index, 10) / 10` |
| 1–5 | 袋中各色 / 20 |
| 6–10 | 弃盘各色 / 20 |
| 11 | 袋中总数 / 100 |
| 12 | 先手标记还在 center |
| 13 | 轮到视角玩家 |
| 14 | 桌面（工厂 + center）剩余瓷砖 / (4F) |
| 15 | 桌面剩余堆数 / 20（本轮最多还能取几次） |

玩家（28）：

| 偏移 | 含义 |
|------|------|
| 0–3 | 相对视角玩家的座次 one-hot |
| 4 | 轮到该玩家 |
| 5 | 已拿到先手标记（下一轮先手） |
| 6 | 分数 / 100 |
| 7 | floor 张数 / 7 |
| 8 | floor 目前的扣分 / 14 |
| 9 | 下一块落到 floor 的扣分 / 3 |
| 10–14 | floor 上各色块数 / 7 |
| 15 | 本轮结算时已满的行将得的分 / 20 |
| 16 | 结算后会凑满一个墙行（游戏结束） |
| 17 | 以当前墙计的终局奖励 / 50 |
| 18–22 | 墙上每列块数 / 5 |
| 23–27 | 墙上每种颜色块数 / 5 |

图案行（33）：

| 偏移 | 含义 |
|------|------|
| 0–3 | 所属玩家的相对座次 one-hot |
| 4–8 | 第几行 one-hot |
| 9–13 | 行内颜色 one-hot（空行全 0） |
| 14 | 已放块数 / 容量 |
| 15 | 空位数 / 5 |
| 16 | 已满 |
| 17–21 | 该墙行上已有颜色 c |
| 22–26 | 该墙行第 k 列已有块 |
| 27–31 | 颜色 c 在这一行成墙时的得分 / 10（该色已在墙上则为 0） |
| 32 | 这一行本轮结算的得分 / 10（未满为 0） |

「成墙得分」按结算顺序算：各行从上到下成墙，所以某一行计分时，上方已满的行已经在墙上。

堆（41）：

| 偏移 | 含义 |
|------|------|
| 0 | 在 center |
| 1 | 拿它会同时拿到先手标记 |
| 2–6 | 颜色 one-hot |
| 7 | `min(数量, 16) / 8` |
| 8–12 | 同一工厂上其余颜色各几块 / 4（拿走后它们进 center；center 的堆全 0） |
| 13 + 5r … 17 + 5r | 对视角玩家的第 r 条 pattern line：可放、放进去的块数 / 5、放后该行满、满后成墙得分 / 10、溢出到 floor（含先手标记）的扣分 / 14 |
| 38 | 全部放 floor（含先手标记）的扣分 / 14 |
| 39 | 下家有图案行能放这种颜色 |
| 40 | 下家拿它能放满某一行 |

「可放」与规则的合法动作判定一致，所以轮到视角玩家时，堆 token 上的「可放」标志（加上 floor 总是可放）恰好等于合法动作掩码。

### 旧网络与热启动

旧的 MLP 模型（`games/azul/model/azul_{2,3,4}p.onnx`）吃的是扁平编码，和 transformer 结构不同，权重不能直接搬过去。热启动的方式与 `splendor_tok` 相同：在 token 变体里，`pv_512x4` 启发式仍然用旧模型 + 扁平编码器做搜索（配置按对应的扁平变体 id 读取 `heuristic.json`），训练前期由它来下棋，transformer 学它的搜索结果，之后逐步过渡到自我对弈；评估时它也是对手。旧模型在 `azul_tok` 里选的动作与在 `azul` 里完全一样（有测试）。

`final_score_extractor` 已注册（各家终局分数），所以 `training.score_head` 可用。

### 训练配方（`train_transformer.json`，20000 步）

网络和优化与 Splendor 的 `train_transformer_kata_guided.json` 相同（网络 d128×4、每步 400 局、批 1024×12、学习率 2e-4 余弦降到 4e-6、回放池 20 万、fp16 推理），KataGo 的四项也全部打开。不同的是：旧网络是强老师，引导全程 1.0 保持到第 500 步，再线性退到第 1000 步（最初定的 1000 / 2000 在前 100 步就看出学得很快，缩短了）；每 100 步评估 200 局（对手就是 `pv_512x4`，一次评估的误差约 ±7%）。

| 项 | 配置 | 值 |
|----|------|----|
| 随机搜索量（playout cap） | `training.playout_cap` | 完整 800 / 快速 150 / 完整概率 0.25 |
| 分差辅助头 | `training.score_head` | 权重 0.5，`scale` 40 |
| 权重滑动平均 | `training.ema` | 0.99 |
| FPU 降低 | `train.json::mcts.selfplay` / `mcts.eval` 的 `fpu_reduction` | 0.2 |

吞吐（`.agents/skills/training/performance.md`，4070 Ti SUPER + 32 线程，400 局并发）：引导阶段由旧网络搜索，MLP 很小，瓶颈在凑批：`heuristic.json` 里 `pv_512x4.leaf_batch_size` 从 8 提到 32，每步自我对弈 36.6 s → 16.5 s（56 万 → 125 万次模拟/秒），再提到 64 只快 10%，但 1000 次模拟里同时展开的叶子太多，老师变弱。之后由 transformer 搜索时显卡跑满（约 20 万次模拟/秒、210 W），`leaf_batch_size` 16、批 1024 或 2048 都在 ±3% 内；`evaluator` 取批 2048、队列 16384，两个阶段共用。回放池用 `compact_replay` 存 fp16 特征。

自我对弈的探索（`.agents/skills/training/exploration.md`）：

| 项 | 配置 | 值 |
|----|------|----|
| 开局随机走 | `training.opening_exploration` | 25% 的对局先按网络策略不搜索地走 1–6 步（温度 1.2，15% 完全随机），之后正常搜索；二人局一般 50–60 步 |
| 中盘切入分叉 | `training.midgame_start` | 15% 的对局先按温度 0.7 快速推演 8–22 步，末步做 `3..8` 候选锦标赛分叉后进入正常搜索 |
| 自适应塑形根噪声 | `train.json::mcts.selfplay.dirichlet_total_alpha` / `dirichlet_shaped` | `10.83` + `dirichlet_shaped: true`（`dirichlet_alpha` 0）：50% 均匀 + 50% 超额对数先验塑形 |
| 根节点策略温度 | `train.json::mcts.selfplay.root_policy_temperature` | `1.25 → 1.1`，按 `halflife_plies: 4.5` 指数衰减（与 `temperature_schedule` `0.8 → 0.1` 半衰期一致） |

唯一和 Splendor 数字不同的是 `score_head.scale`：它的单位是游戏分数。Splendor 的分差标准差约 4.5 分，除以 15 后目标方差约 0.09；Azul 旧网络对局的分差标准差约 11 分（带噪声的自我对弈约 21 分），要得到同样量级的目标需要除以 40（方差 0.08–0.29）。照搬 15 的话分差损失会比胜负损失还大。

为了和 Splendor 一致，Azul 的公共配置也改了三处，它们同样作用于扁平变体 `azul` 的训练和评估（网页对局读 `web.json`，不受影响）：

- `train.json::mcts`：`selfplay` 和 `eval` 加 `fpu_reduction` 0.2；`eval.simulations` 400 → 2000。改动前后的评估胜率不能直接比较。
- `heuristic.json`：`pv_512x4` 加 `leaf_batch_size` 32（引导阶段旧网络的搜索按批送显卡，见上面的吞吐一段）。

```bash
python3 -m training.cli --game azul_tok \
    --train-config games/azul/config/train_transformer.json \
    --output runs/azul_tok_guided20k --providers cuda,cpu
# 中断后继续：同一条命令加 --resume runs/azul_tok_guided20k
```

---

## 常见踩坑

- **忘传 `public_snapshot`**：observer 路径上不重放规则，所有公开字段（factories、center、各家面板、丢弃区、袋中色数）每步都要从 snapshot 整体覆写。漏传一个公开 slot，observer 会跟 truth 立刻偏
- **没有 belief tracker**：Azul 全部公开（袋子和 box_lid 在 schema 里以 per-color counts 体现），不注册 tracker；GT 端仍会按规则发 `factory_refill` 事件,observer 收到后没有 tracker 消费(events 走空消费路径),snapshot 是 observer 同步公开局面的唯一有效通道
- **不需要单独 `take_tiles` 事件**：动作本身（action_id）已经表达了"从哪个 source 拿什么色"，public_snapshot 把对应槽位刷新即可
