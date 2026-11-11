# Liar's Dice — AI API 接入文档

## 概览

- **game_id**：`liarsdice` / `liarsdice_2p` / `liarsdice_3p` / `liarsdice_4p`
- **玩家数**：2–4（每人 5 颗骰子，单回合制：第一次 challenge 即终局）
- **动作空间**：`5N × 5 + 1`（2p=51 / 3p=76 / 4p=101）
- **隐藏信息**：有（每个玩家自己骰子的面值分布；其它玩家不可见）
- **公开事件**：`events` 始终是 `[]`。Liarsdice 不发任何 event payload —— bid 和 challenge 的语义完全由 `(actor, action_id)` 表达，tracker 端在 `observe_public_event` 内自己累积"每位玩家最近 5 次叫牌"的滑窗作为 belief net 的输入特征
- **特点**：内置**网络化 belief**（`liarsdice_belief_<N>p.onnx`）—— MCTS 在每个 sim 入口为每位对手从该 net 输出的后验中按 252 种 Σ=5 多重集采样，而不是均匀采样。接入方什么都不用做

---

## 动作编码

总动作空间 `5N × 5 + 1`，分两类：

| 范围 | 类型 | 说明 |
|------|------|------|
| `0 .. 5N×5 - 1` | 叫牌 (bid) | `X = action_id / 5 + 1`（叫的数量，1..5N），`Y = action_id % 5 + 2`（叫的面值，2..6） |
| `5N × 5` | challenge | 质疑上一位玩家的叫牌，立即结算终局 |

合法约束：
- 第一手（`current_bid_count == 0`）只能 bid，不能 challenge
- bid 必须**严格大于**当前 bid：`(X', Y') > (X, Y)` 当且仅当 `X' > X` 或 `(X' == X 且 Y' > Y)`
- 面值 1 是百搭（wild），在 challenge 结算时计入任何被叫的面值；但 1 不能作为 bid 的 `Y` 值

辅助函数（C++ 端，接入方可参考语义）：
- `encode_bid(x, y) = (x - 1) * 5 + (y - 2)`
- `bid_count_from_action(a) = a / 5 + 1`
- `bid_face_from_action(a) = a % 5 + 2`
- `is_challenge(a, total_dice) = (a == total_dice * 5)`

---

## 公开事件

`events` 永远是 `[]`。`public_snapshot` 携带的公开 schema 槽位：

- `current_bid_count`（int）：当前叫的数量
- `current_bid_face`（int）：当前叫的面值（0 = 还没人叫）
- `current_player`（int）：下一个要决策的玩家
- `is_terminal`（bool）：是否终局
- `winner`（int）：终局后是 challenger 还是 bidder（`-1` 还没结束）

**`dice_count[player]` 的 viz**：
- 进行中：`viz[p, *, p] = 1`，对自己可见，其它玩家全是 placeholder
- challenge 之后：rules 调 `viz::reveal_slot(s, "dice_count", {p, f})` 把每个玩家每个面值的 count 翻给所有人，walker 把真值序列化进 snapshot（作为终局结算的事实）

---

## state_dict 字段

`state_serializer` 输出（`platform/web` 和 `state_dict` API 都用这份）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 胜者；`-1` 未结束 |
| `num_players` | int | 2 / 3 / 4 |
| `current_bid_count` | int | 当前 bid 的 X（0 = 还没人叫） |
| `current_bid_face` | int | 当前 bid 的 Y（0 = 还没人叫） |
| `players[p].dice_count` | `int[6]` | 玩家 p 每个面值的骰子数；对自己之外的玩家在非终局时全 placeholder（实际行为是 owner-only viz：调用方读 `state_dict` 也只看到自己） |
| `players[p].dice` | `int[]` | 平铺的骰子面值列表（按面值升序）；`dice_count` 的 expand-form |

私有：自己的 `dice_count[my_seat]` / `dice[my_seat]` 始终可见；对手的在非终局期间是不可见的。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# AI 扮演 2 人局的 1 号玩家
initial_observation = gt_session.extract_initial_observation(my_seat=1)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "liarsdice_2p",
    "seed": 42,
    "my_seat": 1,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]:
        break

    if status["current_player"] == 1:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick()  # 对手选 bid 或 challenge

    # GT 端:`apply_action_with_trace` 推进 GT 并取 perspective=1 的 trace。
    # LiarsDice 注册了 `no_events_extractor`,events 永远是 [],信息全靠
    # public_snapshot。
    trace = gt_session.apply_action_with_trace(action_id, perspective=1)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id,
        "events": trace["events"],  # 总是 []
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## Liarsdice 特有：网络化 belief

每个变体在训练阶段同时产出两份 ONNX：
- `liarsdice_<N>p.onnx`：策略 + 价值（PV net）
- `liarsdice_belief_<N>p.onnx`:belief net,输入是"每个 seat 最近 5 次 bid 的滑窗",形状 `30N` 维(N 个 seat × 每 seat 5 条 bid × 每条 bid 6 维 = X 标量 + Y one-hot 5)。输出是 **per-opp per-face** 的边缘分布:`(N-1) × 6 × 6 = 36(N-1)` 维 logits(每个对手 6 个面值 × 每面 0..5 张骰的 6 个 count bin)。`game.json::variants.<v>.belief_feature_dim` / `belief_logit_count` 跟这两个值对应

MCTS 流程:
1. 决策树根节点处:`prepare_for_root` 调 belief net 取 per-face 边缘,然后在 252 个 Σ=5 outcome 上按 `∏_f p(n_f|f)` 拼成 joint 后验、归一化、softmax 缓存到 `pi[opp][252]`(252 = `C(10, 5)` 即 5 颗骰子 × 6 面值的 Σ=5 多重集枚举数,见 `games/liarsdice/outcome_index.h`)
2. 每个 world 入口(world_cap split,详见 ALGORITHM_OVERVIEW §9.4):`randomize_unseen` 按 `pi[opp]` 抽一个具体 outcome,写进 world_state 的 `dice_count[opp]`,M 个内层 sim 共享这个采样
3. 没装载 belief net 时退化为均匀采样(252 种等概率)

**接入方不需要做任何事** —— belief net 只在 AI session 内部跑，对外接口完全不变。

---

## 常见踩坑

- **单回合制，没有"丢骰子"**：和 Perudo 不同，Liarsdice 这里实现的是单回合规则——一次 challenge 直接终局判胜负，不会进入下一轮 reroll。`is_terminal` 触发后整局结束
- **face 1 是 wild**：challenge 结算时 face 1 计入任何被叫的面值；但 face 1 不能作为 bid 的 `Y`
- **action_space 跟 N 走**：2p=51 / 3p=76 / 4p=101；不要硬编码
- **bid history 不在 schema**：滑窗只在 tracker 内（`bids_by_player_`），不写进 state；schema 只放当前 outstanding bid（legal_actions 需要它）
- **对手骰子在非终局期间永远不可见**：你的 GT 端如果想 echo 对手骰子，必须 wait until `is_terminal == true` 才在 snapshot 里能拿到 `dice_count[opp]` 的真值
