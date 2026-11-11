# Coup — AI API 接入文档

## 概览

- **game_id**：`coup` / `coup_2p` / `coup_3p` / `coup_4p`
- **玩家数**：2–4
- **动作空间**：32
- **隐藏信息**：有（每人 2 张未揭示的 influence 卡 + deck 剩余卡序列）
- **公开事件**：Coup 是少数**会发 events** 的游戏（六种 kind，详见下文"公开事件"）。tracker 同时消费 events 和 `(actor, action_id)`：events 提供"哪张牌被揭开 / 换牌完成"等结构性信号，action 流提供 claim/challenge/allow 的语义。被揭示的角色身份本身也通过 `viz::reveal_slot` + `public_snapshot` 同步（不依赖 event）
- **特点**：AI 用**启发式概率 sampler**（而非均匀采样）来采样对手暗牌——tracker 综合 claim/challenge action + `card_revealed` / `exchange_complete` event 累积每位对手对每个角色的信号

---

## 动作编码

固定 32 个动作 ID：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0 | Income | +1 币 |
| 1 | Foreign Aid | +2 币（可被 Duke 挡） |
| 2–5 | Coup target 0..3 | 花 7 币强制 target 丢一张 influence |
| 6 | Tax | +3 币（claim Duke，可被挑战） |
| 7–10 | Assassinate target 0..3 | 花 3 币让 target 丢 influence（claim Assassin，可被挑战或 Contessa 挡） |
| 11–14 | Steal target 0..3 | 从 target 偷 2 币（claim Captain，可被挑战或 Captain/Ambassador 挡） |
| 15 | Exchange | claim Ambassador，从 deck 抽 2 张换手牌 |
| 16 | Challenge | 质疑当前 claim |
| 17 | Allow | 不挑战（普通动作） |
| 18 | Block Duke | claim Duke 挡 Foreign Aid |
| 19 | Block Contessa | claim Contessa 挡 Assassinate |
| 20 | Block Ambassador | claim Ambassador 挡 Steal |
| 21 | Block Captain | claim Captain 挡 Steal |
| 22 | Allow No Block | （不挡 Foreign Aid，默认流程） |
| 23 | Reveal Slot 0 | 被挑战后揭示 slot 0 影响 |
| 24 | Reveal Slot 1 | 被挑战后揭示 slot 1 |
| 25 | Lose Slot 0 | 被 coup / 挑战失败时选择丢 slot 0 |
| 26 | Lose Slot 1 | 同上，slot 1 |
| 27–31 | Return 角色 | Ambassador exchange 最后把某角色牌还回 deck。27=Duke, 28=Assassin, 29=Captain, 30=Ambassador, 31=Contessa。**只有换牌者自己知道**——见下 |
| 32 | （观察值）Hidden Return | 不是合法动作、不在策略头里。还牌动作在观察流里的替身：`CoupRules::public_action` 把 27–31 一律投影成 32，`apply_action_with_trace(...)["action"]`、selfplay 的 `observation_trace`、各座位 tracker、网页轮询拿到的都是 32。还的是哪个角色不出现在任何发给别人的字段里 |

**`/observe` 要传投影后的 id**：GT 端用 `trace["action"]`（而不是自己手里的真实 action_id）回灌 `/observe`，这样 AI session 永远收不到对手还了哪张牌。对绝大多数动作两者相等。

**多阶段回合**：Coup 的一个"回合"可能产生多个 action_id（出招 → 被挑战 → 揭示 → 丢牌 等）。AI 每收到一个 action_id 就更新一步内部 state，直到 `current_player` 走到下一真正的决策者。

---

## 公开事件

Coup 的 `events` 由 `extract_events_only`（`games/coup/coup_register.cpp::coup_events::extract_events_only`）通过 diff `(state_before, state_after)` 产出，共 6 种 kind，所有 payload 字段都是 int / bool（不会泄露暗牌）：

| kind | payload | 何时触发 |
|------|---------|----------|
| `card_revealed` | `{player, slot, role}` | 任意 `(p, slot)` 的 `revealed` 从 false 翻 true（lose-influence / 挑战失败 reveal） |
| `claim_resolved_truthful` | `{claimer, role}` | `kResolveChallengeAction` 走出，`action_challenge_succeeded == false`（claim 是真，牌洗回 deck） |
| `block_resolved_truthful` | `{blocker, role}` | `kResolveChallengeCounter` 走出，`counter_challenge_succeeded == false`（block 是真） |
| `claim_unchallenged` | `{claimer, role}` | `kChallengeAction` 走出且未进入 `kResolveChallengeAction`（所有人 Allow） |
| `block_unchallenged` | `{blocker, role}` | `kChallengeCounter` 走出且未进入 `kResolveChallengeCounter` |
| `exchange_complete` | `{player}` | `kExchangeReturn2` 走出（Ambassador 换牌结束，先前所有 claim 信号失效） |

被揭示的角色**身份**也同时走 `public_snapshot`：rules 在 `do_action_fast` 里调 `reveal_slot_to(viewer=everyone, "influences", {p, slot})`，walker 把 `influences[p][slot].revealed/character` 写进 snapshot；observer 整体覆写后那些槽位 viz=1。

接入方需要:(1) 让 GT 端的 rules 在 `do_action_fast` 里维护好 `revealed[]` / `influence[]` / `viz_`;(2) 用框架自带的 `install_event_protocol(extract_events_only, schema_provider)`(`coup_register.cpp` 已经这么做)让 framework 自动 diff 出上面 6 种 event 并 broadcast。**不要自己手写 events**——直接调 `gt_session.apply_action_with_trace(action, perspective)` 把 framework diff 出来的 trace 透传即可。

---

## state_dict 字段

`state_serializer` 输出（`games/coup/coup_register.cpp::serialize_coup`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 胜者；`-1` 还没结束 |
| `num_players` | int | 2 / 3 / 4 |
| `ply` | int | 累计步数 |
| `stage` | int | 当前阶段（CoupStage 枚举值） |
| `active_player` | int | 出招方 |
| `declared_action` | int | 当前正在结算的 action_id |
| `action_target` | int | 出招的 target seat（无则 -1） |
| `blocker` | int | 当前提出 block 的玩家（无则 -1） |
| `challenger` | int | 当前提出挑战的玩家（无则 -1） |
| `deck_size` | int | 牌堆剩余张数 |
| `exchange_drawn` | `int[]` | 仅在 `kExchangeReturn1` / `kExchangeReturn2` 阶段填充：active_player 抽到的 2 张换牌（自己可见，对手 placeholder=0；-1 表示 Return1 已选完） |
| `players[p].alive` | bool | 是否还在场 |
| `players[p].coins` | int | 金币数 |
| `players[p].influences[s].character` | int | 角色 id（0..4 = Duke/Assassin/Captain/Ambassador/Contessa）。`revealed=true` 的对所有 viz=1；未揭示的对自己 viz=1 |
| `players[p].influences[s].character_name` | string | 角色名（同上 viz） |
| `players[p].influences[s].revealed` | bool | 是否已揭开（公开） |

| `players[p].deck_known` | `int[5]` | 座位 p **确定知道**在牌堆（或下面被标记的槽位）里的各角色张数——主要来自它自己 Exchange 还回去的牌，以及质疑失败后公开洗回的那张。owner-only，仅供测试/守恒检查；AI 只通过 masked state 读自己的那一行 |
| `players[p].deck_known_taint` | `int[2N]` | `[q*2+s]=1` 表示槽位 (q, s) 在 p 有上述知识期间从牌堆暗抽过牌（那张已知的牌可能跑到这里） |
| `deck_count` | `int[5]` | 牌堆各角色张数。对所有玩家隐藏，仅供测试做守恒检查 |

私有：自己的两张未揭示 `influences[my_seat][s].character`、Ambassador 阶段的 `exchange_drawn`、自己的 `deck_known` / `deck_known_taint` 行。对手的暗牌不可见，AI 靠 sampler 采样；采样出的世界保证满足 `deck_known[observer] ⊆ 牌堆 ⊎ 被标记槽位`（不会把观察者自己还回牌堆的牌发给对手），其余座位的知识行在采样世界里清零。

**snapshot 新增字段**：`deck_known`、`deck_known_taint`（owner-only，只含接收者自己那一行）以及可选的 `__viz_all_viewers__["influence"]`（`influence` 的整张 viz 张量，含所有观察者列——"谁能看到哪张牌"在 Coup 里是公开信息，接收者的搜索需要它来正确模拟对手视角）。`exchange_drawn` 的 viz 不整张同步：哪个抽牌槽先被还掉是换牌者的私有信息。手工构造 snapshot 的外部客户端可以不带 `__viz_all_viewers__`（退回只同步自己那一列的旧行为），但必须带齐所有 schema 字段的 `__viz__` 切片（含上面两个新字段）。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

initial_observation = gt_session.extract_initial_observation(3)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "coup_4p", "seed": 555, "my_seat": 3,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

# AI 需要所有玩家的 claim / challenge / reveal 事件才能正确建模
while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 3:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick()

    # Ground truth 端:`apply_action_with_trace(action, perspective)` 一行
    # 包揽——内部 do_action_fast 推进 GT、然后调 framework 自动派生的
    # `public_event_extractor`(由 `install_event_protocol(extract_events_only, ...)`
    # 装好)diff 出 events 并算 public_snapshot,返回 {events, public_snapshot}。
    trace = gt_session.apply_action_with_trace(action_id, perspective=3)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": trace["action"],   # 公开投影：还牌动作在这里是 32

        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## Coup 特有：启发式 belief

Coup 的 tracker (`games/coup/coup_net_adapter.cpp::CoupBeliefTracker`) 不是 uniform 采样。它综合 `(actor, action_id)` 流和上面 6 种 event 在内部维护每个对手对每个角色的**信号计数**：

- 对手**claim 某角色且未被挑战**（来自 `claim_unchallenged` / `block_unchallenged` event 或对应 action 流）→ +1 信号
- 对手**真实 claim 被对手挑战且自证成功**（`claim_resolved_truthful` / `block_resolved_truthful`）→ 相关信号清零（牌已经洗回 deck）
- 对手**挑战别人的某 claim** → 该对手对该角色 +1 信号（他敢挑战，说明他自己可能持有）
- 对手**揭示某角色**（`card_revealed` event）→ 清空对那角色的信号；该角色身份也通过 viz=1 进 snapshot
- 对手**完成 Ambassador exchange**（`exchange_complete` event）→ 清空所有信号（手牌重洗）

采样时权重 = `pool_remaining × (1 + 0.5 × signal_count)`，加硬约束保证全局守恒（每角色总共 ≤ 3 张）。

**接入方什么都不用做**——只要 GT 端通过 `install_event_protocol` 让 framework 自动 diff 出 events，tracker 会自己消费它们。

---

## 常见踩坑

- **多阶段回合要逐 action 喂**：一次出招 → 挑战 → reveal → 丢牌 是 4 个连续的 action。每个 action 都必须 `observe` 一次，每次都带最新的 `public_snapshot`，不能跳过
- **被揭示的牌靠 `viz` + snapshot 自动同步**：rules 在 `do_action_fast` 里调 `reveal_slot_to(everyone, "influences", {p, slot})` 后，walker 把 `influences[p][slot].character/revealed` 写进 snapshot；observer 整体覆写后 viz=1，无需任何额外事件
- **不要自己构造 `card_revealed` / `exchange_complete` payload**：framework 的 `install_event_protocol` 已经包了 `extract_events_only`，调它 diff 即可，不要绕过去手写
- **`exchange_drawn` 槽位"是否有牌"以 viz 为准**：第一次还牌后，被清空的抽牌槽在换牌者自己的 session 里仍留着旧角色（snapshot 只写可见槽位）。rules 的换牌路径（合法动作和状态转移）一律通过 `exchange_slot_live`（槽位对换牌者可见且值 ≥ 0）判断，`randomize_unseen` 也会把不可见的抽牌槽归位成 -1
- **`exchange_drawn` 字段只在换牌阶段存在**：`kExchangeReturn1` / `kExchangeReturn2` 之外这个字段是空 list；不要把任何阶段的 `exchange_drawn` 当成 ambassador 暗牌池的稳定快照
