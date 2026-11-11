# Azul — AI API 接入文档

## 概览

- **game_id**：`azul` / `azul_2p` / `azul_3p` / `azul_4p`
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

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

initial_observation = gt_session.extract_initial_observation(2)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "azul_3p", "seed": 42, "my_seat": 2,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 2:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
        info = r["action_info"]
        print(f"AI: 从{'工厂'+str(info['source']) if info['source']<7 else 'center'}拿色{info['color']}到{'line'+str(info['target']) if info['target']<5 else 'floor'}")
    else:
        action_id = opp_pick()

    # GT 端:`apply_action_with_trace` 一行包揽 do_action_fast 推进 +
    # `azul_events::extract_events`(注册时 wired 到 `b.public_event_extractor`)
    # diff 出 `factory_refill` / `round_settled` 等事件 + perspective 视角的
    # public_snapshot。
    trace = gt_session.apply_action_with_trace(action_id, perspective=2)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id,
        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 常见踩坑

- **忘传 `public_snapshot`**：observer 路径上不重放规则，所有公开字段（factories、center、各家面板、丢弃区、袋中色数）每步都要从 snapshot 整体覆写。漏传一个公开 slot，observer 会跟 truth 立刻偏
- **没有 belief tracker**：Azul 全部公开（袋子和 box_lid 在 schema 里以 per-color counts 体现），不注册 tracker；GT 端仍会按规则发 `factory_refill` 事件,observer 收到后没有 tracker 消费(events 走空消费路径),snapshot 是 observer 同步公开局面的唯一有效通道
- **不需要单独 `take_tiles` 事件**：动作本身（action_id）已经表达了"从哪个 source 拿什么色"，public_snapshot 把对应槽位刷新即可
