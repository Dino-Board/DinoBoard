# 6 nimmt! — AI API 接入文档

## 概览

- **game_id**：`6nimmit` / `6nimmit_2p` / `6nimmit_3p` / `6nimmit_4p`
- **玩家数**：2–4
- **动作空间**：108（固定）
- **隐藏信息**：有（每个玩家手牌 + 本轮"已选但未翻"的暗扣牌 `pending_commits`；其它玩家的两者都不可见）
- **公开事件**：唯一一种 `card_played {player, card}` —— 某玩家 `pending_commits` 从一张牌（1..104）变 0 时发出，告诉所有人"这张暗扣牌已经被翻出来"。其它结构性变化（`apply_queue_card` / `bullhead_pile` / `committed` / `table` 等公开槽位的更新）由 `public_snapshot` 携带，不走 event
- **特点**：本框架**只实现 1 个版本** —— 104 张牌、4 行、行容量 5、10 轮、每人手牌 10 张。无 Tactics / Logic / Professional / Tournament / 长牌局变体

---

## 动作编码

| 范围 | 类型 | 说明 |
|------|------|------|
| `0 .. 103` | `commit` | 暗扣某张牌；`card_number = action_id + 1` |
| `104 .. 107` | `small_choose_row` | "数字太小没法接"时选择一行收走；`row = action_id - 104` |
| `108` | （观察值）hidden commit | 不是合法动作、不在策略头里。**其他人还没扣完牌时**的 commit 在观察流里的替身：`SixNimmtRules::public_action` 把它投影成 108，`apply_action_with_trace(...)["action"]`、selfplay 的 `observation_trace`、各座位 tracker、网页轮询拿到的都是 108，牌号不随动作外传。本轮最后一个 commit 会在同一步把所有暗扣翻开，因此原样透传 |

**`/observe` 要传投影后的 id**：GT 端用 `trace["action"]` 回灌 `/observe`。被翻开的牌通过 `card_played` 事件和 snapshot（`table` / 已公开的 `pending_commits`）到达。

辅助：
- `kCommitOffset = 0`，`kSmallChooseRowOffset = 104`
- 一局回合结构：每轮所有玩家**并发**`commit`一张暗扣 → 4 张牌按数字升序逐张应用 → 期间若某张数字小于所有行尾就触发 `small_phase_active=true`，由该玩家用 `small_choose_row` 选一行收走

合法约束：
- `small_phase_active == true` 时仅 `small_choose_row` 合法（4 个动作 ID 全合法，玩家可选哪一行）
- 否则仅 `commit` 合法，且只能 commit 自己手牌里**还没出过**的牌
- 该玩家本轮已 `committed[p] == 1` 后不会再轮到他做决定（`current_player` 跳过）

---

## 公开事件

唯一一种 `card_played`：

| kind | payload | 何时触发 |
|------|---------|----------|
| `card_played` | `{player, card}` | 该玩家的暗扣牌在这一步被 apply 到桌面的那一刻（`pending_commits[player]` 被消费）。**每个座位每张出过的牌恰好一条**：本轮最后一个扣牌的座位，其 commit 和 apply 发生在同一步（槽位前后都是 0），事件从动作本身取牌号——修复前这个座位只有在出现 SMALL 暂停的轮次才会被记录 |

剩下的状态变化都靠 `public_snapshot` 携带，不走 event：

- 公开 scalar：`current_player` / `is_terminal` / `winner` / `ply` / `current_round` / `num_committed_this_round` / `small_phase_active` / `small_player` / `apply_cursor`
- `table[4][5]`：4 行各 5 个槽，槽值 = card_number，0 = 空
- `bullhead_pile[N][104]`：每位玩家"被吃了"收的牌池（公开）
- `committed[N]`：本轮每位玩家是否已扣牌（公开 bool）
- `apply_queue_card[N]` / `apply_queue_player[N]`：4 张暗扣翻牌后按升序排好的同时揭示队列（公开）—— 一旦进入应用阶段，每张暗扣的真值通过这里公开

私有（owner_only）：
- `hand[N][104]`：自己 viz=1，对手 placeholder
- `pending_commits[N]`：本轮"自己已选好但还没翻"的那一张；自己 viz=1，对手 placeholder

snapshot 还带一个可选键 `__viz_all_viewers__["pending_commits"]`（该字段整张 viz 张量，含所有观察者列；翻牌是公开事件，所以"谁能看到哪张暗扣"是公开信息）。手工构造 snapshot 的外部客户端可以不带它。

信念网络输入（`SixNimmtBeliefFeatureExtractor`，维度不变 325+2N）：[A]"未见池"现在同时扣除桌面和各牛头堆里的牌——之前开局 4 张行首牌和未被记录的出牌一直被标成"未见"；[B]/[C]"己方/对方已出"因 `card_played` 补全而对所有座位对称完整。

整个流程没有任何**对手手牌反向揭示**——揭示总是在应用阶段从 `pending_commits` 流向公开 `apply_queue_card` + `card_played` event。

---

## state_dict 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 胜者；`-1` 还没结束 |
| `num_players` | int | 2 / 3 / 4 |
| `ply` | int | 累计步数 |
| `current_round` | int | 当前轮次（0..9） |
| `num_committed_this_round` | int | 本轮已扣牌人数 |
| `small_phase_active` | bool | 当前是否处于"选行收走"阶段 |
| `small_player` | int | 触发 small phase 的玩家；`-1` 表示无 |
| `apply_cursor` | int | 升序应用队列已处理到第几个 |
| `table` | `int[4][5]` | 行内的牌（card_number），0 = 空槽 |
| `apply_queue_card` | `int[N]` | 本轮揭示队列（按 card 升序） |
| `apply_queue_player` | `int[N]` | 揭示队列对应的玩家 |
| `players[p].hand` | `int[]` | 玩家 p 当前手牌（自己看得到；对手在 state_dict 上是空列表，因为 owner_only viz） |
| `players[p].hand_count` | int | 手牌张数 |
| `players[p].bullhead_pile` | `int[]` | 玩家 p 已收的牌（公开） |
| `players[p].bullhead_count` | int | 玩家 p 已累计的牛头总数（积分用） |
| `players[p].committed` | bool | 本轮是否已扣牌 |
| `players[p].pending_card` | int | 玩家 p 本轮已选未翻的暗扣（owner_only；自己看到非 0，对手看到 0） |

胜负：10 轮结束时**牛头数最少**的玩家获胜。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# 4 人局，AI 扮演 2 号玩家
initial_observation = gt_session.extract_initial_observation(my_seat=2)
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
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
        # action_id 可能是 commit (0..103) 或 small_choose_row (104..107)
    else:
        action_id = opp_pick()

    # GT 端:`apply_action_with_trace` 推进 GT 并取 perspective=2 的 trace
    # (events 含 6 nimmt! 自己的 `card_played` event,public_snapshot 是
    # apply 后 perspective 看到的所有 viz=1 槽位)。
    trace = gt_session.apply_action_with_trace(action_id, perspective=2)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": trace["action"],   # 公开投影：未翻开的 commit 在这里是 108

        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 常见踩坑

- **commit 是"暗扣"，不是直接落桌**：`commit` 之后牌进入 `pending_commits[me]`（owner_only），`committed[me] = 1`（公开）。N 个玩家都 commit 完才进入升序应用阶段，这时 `apply_queue_card` 公开
- **多阶段回合要逐 action 喂**：每轮里"4 个 commit + 0..4 个 small_choose_row"都是独立 action，每个都要 `observe`
- **轮到的人不一定是 ply+1 的人**：commit 阶段是按"还没扣的玩家"轮转；小阶段是 `small_player` 接管；apply_cursor 推进到下一张牌时若触发 SMALL，`current_player` 会跳到那张牌的所有者
- **牛头分越少越好**：`bullhead_count` 最少的玩家获胜
- **action_space 跟 N 无关**：固定 108；feature_dim 跟 N 走（2p=341 / 3p=448 / 4p=555）
- **两份 heuristic**：注册了 `rule_v1`（直接规则）和 `lkf_v1`（一份 9 年前的人类启发式 port），`heuristic.json::selectors` 默认让 `eval_vs_heuristic` / `web` 都用 `lkf_v1`
