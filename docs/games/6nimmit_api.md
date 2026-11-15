# 6 nimmt! — AI API 接入文档

## 概览

- **game_id**：`6nimmit`（= `6nimmit_2p`）/ `6nimmit_3p` / `6nimmit_4p`
- **玩家数**：2–4
- **动作空间**：108（固定，与人数无关）
- **一局就是一手牌**：每人 10 张，打 10 轮，局终。吃到牛头最少的人赢。
- **66 分制按一手一局来打**：Board Game Arena 的 66 分制每手都把 104 张重洗，手与手之间除了分数什么都不继承，所以每一手都当成一局新游戏。AI 只追求这一手的牛头差，不考虑整场比分（例如最后一手领先时求稳）。
- **价值 = 牛头差**：终局时每个座位的价值是 `(其他人平均牛头 − 我的牛头) / 25`，截断到 ±1（`SixNimmtMarginValueModel`）。训练标签和搜索里的终局价值都用它，不是输赢。
- **隐藏信息**：每个玩家的手牌 + 本轮已出但还没翻开的牌（`pending_commits`）；别人的这两样都看不到。
- **记牌**：`used[104]`（公开）记录露过面的牌（行首牌和已翻开的牌）；网络和搜索都只从没露过面的牌里猜别人的手牌。
- **公开事件**：唯一一种 `card_played {player, card}`。

## 规则

- 104 张牌（牌号 1..104），牛头数：5 的倍数 2 个，10 的倍数 3 个，11 的倍数 5 个，55 是 7 个，其余 1 个。
- **发牌**：104 张洗匀，发 4 张行首牌 + 每人 10 张（2 / 3 / 4 人局用 24 / 34 / 44 张），`used` 记下 4 张行首牌。
- **一轮**：所有人各出一张牌（同时、暗扣）→ 全部翻开 → 按牌号从小到大依次落桌：接在行尾比它小且最接近的那一行后面；如果那一行已有 5 张，出牌人收走这 5 张（吃掉上面的牛头），自己的牌成为新的行首；如果牌比所有行尾都小，出牌人**自己选一行**收走。
- **结束**：10 轮打完。
- **胜负**：牛头最多的人输（并列都算输），其余人赢；所有人牛头一样是平局。`winner` 字段 = 唯一的牛头最少的玩家，否则 `-1`。训练和评估用的是价值（牛头差），胜负只用于显示和统计。
- 步数上限 `max_game_plies = 100`（正常一局 2 人约 25 步，4 人约 45 步，碰不到）。到上限时 `winner` 记为唯一的牛头最少的玩家，但训练目标按框架惯例给所有座位记负（防拖延）。

---

## 动作编码

| 范围 | 类型 | 说明 |
|------|------|------|
| `0 .. 103` | `commit` | 暗扣某张牌；`card_number = action_id + 1` |
| `104 .. 107` | `small_choose_row` | 牌比所有行尾都小时选择一行收走；`row = action_id - 104` |
| `108` | （观察值）hidden commit | 不是合法动作、不在策略头里。**其他人还没出完牌时**的 commit 在观察流里的替身：`public_action` 把它投影成 108，`apply_action_with_trace(...)["action"]`、selfplay 的 `observation_trace`、各座位 tracker、网页轮询拿到的都是 108，牌号不随动作外传。本轮最后一个 commit 会在同一步把所有暗扣翻开，因此原样透传 |

**`/observe` 要传投影后的 id**：GT 端用 `trace["action"]` 回灌 `/observe`。

合法约束：
- `small_phase_active == true` 时仅 `small_choose_row` 合法（4 行都可选），由 `small_player` 决定。
- 否则仅 `commit` 合法，只能出自己手里的牌；本轮已经出过牌的人不会再轮到（`current_player` 跳过）。
- 出牌顺序只是实现上的串行化（座位 0 先出），先出的人看不到后出的人出了什么。

---

## 公开事件

| kind | payload | 何时触发 |
|------|---------|----------|
| `card_played` | `{player, card}` | 这张牌在这一步落到桌面上。每个座位每张出过的牌恰好一条；牌还扣着的时候没有事件 |

其余状态变化都靠 `public_snapshot` 携带：

- 公开 scalar：`current_player` / `terminal` / `winner` / `ply` / `current_round` / `small_phase_active` / `small_player` / `apply_cursor`
- 公开数组：`hand_heads[N]`（已吃的牛头）、`acts_mask[N]`、`table[4][5]`（槽值 = 牌号，0 = 空）、`used[104]`
- 私有（owner_only）：`hand[N][104]`；`pending_commits[N]`（本轮已出未翻的牌）

`pending_commits` 的可见性会变：别人的牌扣着时不可见；本轮最后一张牌出完的那一步全部翻开；牌落桌后槽位清 0 但**保持公开**（这样只靠观察同步的会话也能知道它已经空了）；该座位下一次出牌时重新变成只有自己可见。snapshot 带一个键 `__viz_all_viewers__["pending_commits"]`（该字段整张可见性张量）来同步这一点。


---

## state_dict 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 这一局是否结束 |
| `winner` | int | 唯一的牛头最少的玩家；没结束或并列是 `-1` |
| `num_players` | int | 2 / 3 / 4 |
| `ply` | int | 累计步数 |
| `current_round` | int | 已打完的轮数（0..10） |
| `num_committed_this_round` | int | 本轮已出牌人数 |
| `small_phase_active` | bool | 是否有人正在选行 |
| `small_player` | int | 选行的人；`-1` 表示无 |
| `apply_cursor` | int | 本轮已落桌的牌数 |
| `table` | `int[4][5]` | 行内的牌，0 = 空槽 |
| `used` | `int[]` | 露过面的牌号（升序） |
| `apply_queue_card` / `apply_queue_player` | `int[]` | 已翻开、还没落桌的牌及其主人（牌号升序） |
| `players[p].hand` | `int[]` | 手牌（升序） |
| `players[p].hand_count` | int | 手牌张数 |
| `players[p].hand_heads` | int | 已吃的牛头数 |
| `players[p].committed` | bool | 本轮是否已出牌 |
| `players[p].pending_card` | int | 本轮已出、还没落桌的牌；0 = 没有 |
| `encoder_layout` | str(JSON) | 特征编码布局（下文各表由它核对） |

`final_scores`（selfplay / arena 结果）= 各人吃的牛头数取负（越高越好）。`terminal_z_values` = 各人的牛头差价值。`custom_stats`：`most_heads`、`fewest_heads`。

---

## 网络特征编码

Token 编码 + transformer（`config/train.json`），2 / 3 / 4 人局布局完全相同，`feature_dim` 为 0（变长）。

**任何 token 都不说自己是 104 张里的哪一张。** 一张牌只有牌号大小和牛头数；真正决定打法的是它相对各行行尾、相对"还没露面的牌"的位置。"未见牌"= 不在 `used` 里、不在我手里、也不是我看得到的已出牌，也就是别人的手牌、别人扣着的牌和牌堆里的牌。已经露面的牌在 `used` 里，所以记牌的效果直接体现在所有"未见牌张数"特征上。

Token（变长，只有存在的才出现；2 / 3 / 4 人局最多 18 / 20 / 22 个），顺序固定：

- `global` ×1：轮次、现在在决定什么、未见牌有多少、其中别人手里占多大比例。
- `player` ×N：我在前，其余按座位顺序；已吃的牛头、目前的牛头差。
- `row` ×4：按桌面行号；行尾、长度、牛头数、会落到这一行的未见牌 / 我的手牌 / 已翻开的牌；选行阶段还有"拿走这一行，这一轮剩下的牌会怎么落"的推演结果。
- `card`：我的每张手牌（牌号升序），然后是我看得到、还没落桌的已出牌（我自己扣着的牌；翻牌后所有人还没落下的牌），同样升序。

每个 token = `[1, 0, anchor, type, has_position, x, y, 29 个特征]` = 36 维（特征不足 29 个的类型尾部补 0）。entity 恒为 0。

**位置**（只有 `row` 和 `card` 有）：`x` = 牌号比它（行尾）小的未见牌张数，`y` = 行尾比它小的行数（`row` 把自己的行尾也算上，所以是 1–4；`card` 是 0–4，0 表示比所有行尾都小）。网络只通过相对位置偏置看到两个 token 的差 `(dx, dy)`：一张牌和一行 `dy = 0` 就是"这张牌会接在这一行后面"，这时 `dx` 正好是可能插到它们中间的未见牌张数。绝对牌号不进入位置。`relative_position_radius` 是 8，差值超过 8 的都算"很远"。

**策略**：2 个指针头。手牌 token 的 `play` 头 = 出这张牌，`row` token 的 `take` 头 = 收走这一行。108 个动作全部绑在锚点上，全局头不用。网络配置必须 `"anchor_embedding": false`：锚点号就是牌号，给它配可学习的嵌入等于把 104 维 one-hot 又加了回来。

#### `anchors`

| 名称 | 起始 id | 个数 | 绑定 |
|------|---------|------|------|
| `card` | 1 | 104 | 手牌 token，锚点 = 牌号；`play` 头 → 动作 `牌号 − 1` |
| `row` | 105 | 4 | 行 token，锚点 = 105 + 行号；`take` 头 → 动作 `104 + 行号` |

归一化：未见牌张数 `min(n, 20) / 20`；"期望"= 张数 × `density`（别人手里的牌占未见牌的比例），`min(e, 5) / 5`；一行里的牛头或一轮里吃到的牛头 `/ 10`；已吃的牛头 `min(h, 30) / 30`；牛头差和价值同一尺度（`/ 25`，截断到 ±1）。

#### `global`

| 下标 | 宽度 | 名称 | 含义 |
|------|------|------|------|
| 0 | 1 | `round` | 已打完的轮数 / 10 |
| 1 | 1 | `choosing_row` | 正有人在选要收走的行 |
| 2 | 1 | `to_move` | 现在该我做决定 |
| 3 | 1 | `unseen` | 未见牌张数 / 104 |
| 4 | 1 | `opp_cards` | 别人每人手里（含扣着的牌）还有几张 / 10 |
| 5 | 1 | `density` | 别人手里的牌总数 / 未见牌张数 |
| 6 | 1 | `low_unseen` | 比所有行尾都小的未见牌张数 |
| 7 | 1 | `low_expected` | 其中别人手里的期望张数 |
| 8 | 1 | `min_row_heads` | 牛头最少的那一行的牛头数（出小牌的代价） |

#### `player`

| 下标 | 宽度 | 名称 | 含义 |
|------|------|------|------|
| 0-3 | 4 | `seat` | 相对座位 one-hot（0 = 我） |
| 4 | 1 | `hand_heads` | 已吃的牛头 |
| 5 | 1 | `margin` | （其他人平均牛头 − 他的牛头）/ 25，截断到 ±1：现在就结束时他的价值 |
| 6 | 1 | `shown` | 他出的牌我看得到且还没落桌（对应一个 `card` token） |
| 7 | 1 | `chooser` | 正在选行的人 |

#### `row`

| 下标 | 宽度 | 名称 | 含义 |
|------|------|------|------|
| 0 | 1 | `tail` | 行尾牌号 / 104 |
| 1-5 | 5 | `length` | 行长 1–5 的 one-hot（5 = 再来一张就要收） |
| 6 | 1 | `space` | 还能放几张 / 4 |
| 7 | 1 | `heads` | 这一行的牛头数 |
| 8 | 1 | `cheapest` | 牛头最少的行（并列也算） |
| 9-12 | 4 | `order` | 行尾在四行里从小到大排第几 |
| 13 | 1 | `reach_unseen` | 会落到这一行的未见牌张数（牌号在本行行尾和下一个更大的行尾之间） |
| 14 | 1 | `reach_expected` | 其中别人手里的期望张数 |
| 15 | 1 | `reach_heads` | 这些未见牌的牛头总数，`min(h, 40) / 40` |
| 16 | 1 | `reach_mine` | 我的手牌里会落到这一行的张数 / 5 |
| 17 | 1 | `reach_shown` | 已翻开待落的牌里会落到这一行的张数 / 3 |
| 18 | 1 | `if_taken_mine` | 选行阶段：选行的人收走这一行后，这一轮剩下的牌按规则落完，我一共吃多少牛头（收行本身也算） |
| 19 | 1 | `if_taken_others` | 同上，其他人合计吃多少牛头 |

#### `card`

| 下标 | 宽度 | 名称 | 含义 |
|------|------|------|------|
| 0 | 1 | `in_hand` | 我的手牌 |
| 1 | 1 | `my_shown` | 我已经出的牌，还没落桌 |
| 2 | 1 | `opp_shown` | 别人已翻开的牌，还没落桌 |
| 3-6 | 4 | `owner` | 牌主的相对座位 one-hot（0 = 我） |
| 7 | 1 | `resolving` | 正在等牌主选行的那张牌 |
| 8 | 1 | `value` | 牌号 / 104 |
| 9 | 1 | `heads` | 这张牌的牛头数 / 7 |
| 10 | 1 | `hand_rank` | 在我手牌里从小到大排第几 / 9（非手牌为 0） |
| 11 | 1 | `percentile` | 比它小的未见牌 / 全部未见牌 |
| 12 | 1 | `below_expected` | 别人手里比它小的牌的期望张数 |
| 13 | 1 | `small` | 比所有行尾都小：出它就要选一行收走 |
| 14 | 1 | `small_cost` | `small` 时牛头最少那一行的牛头数 |
| 15 | 1 | `gap_safe` | 它和目标行行尾之间没有未见牌：没人能插进来 |
| 16 | 1 | `gap_unseen` | 它和目标行行尾之间的未见牌张数 |
| 17 | 1 | `gap_expected` | 其中别人手里的期望张数 |
| 18 | 1 | `gap_mine` | 我的手牌里夹在中间的张数 / 5 |
| 19 | 1 | `gap_shown` | 已翻开待落的牌里夹在中间的张数 / 3 |
| 20-24 | 5 | `row_length` | 目标行行长 one-hot |
| 25 | 1 | `row_space` | 目标行还能放几张 / 4 |
| 26 | 1 | `row_heads` | 目标行的牛头数 |
| 27 | 1 | `sixth` | 目标行已满 5 张：现在落下去就是第 6 张 |
| 28 | 1 | `cost_now` | 没人干扰时出这张牌立刻吃到的牛头（`sixth` 时是目标行，`small` 时是最便宜的行，否则 0） |

"目标行"= 行尾比这张牌小的行里行尾最大的那一行。`small` 的牌从 `gap_safe` 到 `sixth` 全为 0。

验证（`tests/6nimmit/test_token_encoder.py`）：2 / 3 / 4 人整局（随机和两个启发式）里每一步、每个座位视角的编码，都和 `nimmt_token_reference.py` 只凭该座位能记下来的信息（自己的手牌、桌面、各人吃的牛头、自己记的"露过面的牌"）独立重算的结果逐项相同；参考实现不读别人的手牌和扣着的牌。

---

## 搜索里怎么猜别人的手牌

没有信念网络。`SixNimmtBeliefTracker` 不记任何东西：每次采样时，从"不在 `used`、不在我手里、不是我看得到的已出牌"的牌里均匀抽，给其他每个人补满手牌（`10 − 轮次`，他的牌已翻开就再少一张）。别人扣着的牌在采样里视为还在他手上，搜索里由他从采样出的手牌里出。发牌之后规则里没有随机。

## 启发式

`rule_v1`（规则决策树）和 `lkf_v1`（一份人类启发式的移植，每步约 150 ms）。两者看桌面、手牌和 `used`。`heuristic.json::selectors` 让 `eval_vs_heuristic` / `web` 用 `lkf_v1`；`train.json` 用 `rule_v1` 做开局引导（warm start）。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# 4 人局，AI 扮演 2 号玩家
initial_observation = gt_session.extract_initial_observation(2)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "6nimmit_4p",
    "seed": 1234,
    "my_seat": 2,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]:
        break

    if status["current_player"] == 2:
        action_id = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()["action_id"]
    else:
        action_id = opp_pick()

    # GT 端推进并取 2 号视角的 trace：events 是 card_played，public_snapshot 是
    # 这一步之后 2 号看得到的所有槽位。
    trace = gt_session.apply_action_with_trace(action_id, 2)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": trace["action"],   # 公开投影：未翻开的 commit 在这里是 108
        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 常见踩坑

- **出牌是暗扣，不是直接落桌**：`commit` 之后牌进入 `pending_commits[me]`。所有人都出完才翻开，按牌号升序落桌。
- **每个动作都要 `observe`**：每轮 "N 个 commit + 0 或 1 个 small_choose_row" 都是独立动作。（一轮里最多只有一次选行：只有本轮最小的那张牌可能比所有行尾都小。）
- **轮到谁不是简单轮转**：出牌阶段轮到还没出牌的人；选行阶段是 `small_player`。
- **`hand_heads` 越少越好**；`final_scores` 是它取负。
- **66 分制的下一手要开新会话**：一局只有一手，局终后不会再发牌。
- **没有已训练模型**：玩法改了以后旧模型全部作废。训练出新模型放到 `games/6nimmit/model/` 之前，AI 会话只能用启发式难度。
