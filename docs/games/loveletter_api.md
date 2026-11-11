# Love Letter — AI API 接入文档

## 概览

- **game_id**：`loveletter` / `loveletter_2p` / `loveletter_3p` / `loveletter_4p`
- **玩家数**：2–4
- **动作空间**：47
- **隐藏信息**：有（每人手牌 + 轮到的人抽牌 + 牌堆顺序）
- **公开事件**：LL **不发任何 `events`**（`out.events` 始终为空）。所有"reveal 对手手牌"语义（Priest peek、Baron 比较、King 交换、Prince 弃手）都由 GT 端 rules 调 `viz::reveal_slot_to` / `swap_slot_owned` 写进 `state.viz_`，schema-driven walker 把这些 viz=1 槽位的真值塞进 `public_snapshot` 的 `hand[*]` / `drawn_card`，observer 用 snapshot 整体覆写后那些槽位天然 viz=1 + 真值。**接入方只发 `public_snapshot`**，`events` 传 `[]`。

---

## 动作编码

Love Letter 的动作覆盖所有"打出某张牌 + 目标（若该牌需要目标）+ 猜测（Guard 专用）"的组合：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0–27 | Guard 猜 | `target × 7 + (guess - 2)`。target 0..3，guess 值 2..8（7 种猜测；不能猜 Guard=1 因为规则不允许） |
| 28–31 | Priest | target 0..3（偷看该玩家手牌） |
| 32–35 | Baron | target 0..3（和该玩家比牌） |
| 36 | Handmaid | 自己保护一轮 |
| 37–40 | Prince | target 0..3（让该玩家弃手牌并重抽，可选自己） |
| 41–44 | King | target 0..3（交换手牌） |
| 45 | Countess | 自己打出 Countess（无效果，但规则强制与 King/Prince 同手时必须打） |
| 46 | Princess | 自己打出 Princess（立刻自我淘汰） |

**target 是绝对座位号**（0..N-1），不是相对偏移。例如 `kPriestOffset + 2 = 30` 永远表示"对 2 号玩家用 Priest"，无论 actor 是谁。`add_targeted_actions` 直接 `actions.push_back(offset + t)`（`games/loveletter/loveletter_rules.cpp::add_targeted_actions`）。无效 target（自己 / 已淘汰 / 受 Handmaid 保护）不会出现在 `legal_actions`；唯一例外是当所有合法目标都被排除时，规则强制塞 `offset + me` 作为"无效果自打"占位。2p 游戏里 N=2，合法 target 集合永远只有 1 个非己玩家。

**Guard 猜牌的 `guess` 值**：
| 值 | 牌 |
|----|----|
| 2 | Priest |
| 3 | Baron |
| 4 | Handmaid |
| 5 | Prince |
| 6 | King |
| 7 | Countess |
| 8 | Princess |

---

## 公开事件

LL 协议下 `events` 始终是 `[]`——所有 perspective-private 的 reveal 都通过 viz schema + walker 走 `public_snapshot` 通道，**不再有平行私人事件**。

`public_snapshot` 由 GT 端的 schema walker（`viz::serialize_public_snapshot(state, schema, perspective, snap)`）每个 perspective 各产出一份。它走全字段集，对每个 viz[..., perspective]=1 的槽位写一对 `(idx, value)` 进 `snap[field_name]`，并把该视角的 viz 切片打包到 `snap["__viz__"][field_name]`；接收端 `viz::apply_public_snapshot` 整张 viz 切片字节级覆写后再回填值。它包含该视角下 viz=1 槽位的所有真值：

- **自己手牌** (`hand[my_seat]`)：始终 viz=1
- **自己当前抽到的牌** (`drawn_card`)：在自己回合 reveal 后 viz=1
- **被 Priest peek 的对手手牌**：peek 后 viz=1 给 actor，第三方仍 viz=0
- **King 交换后双方的新手牌**：swap 后双方对各自手牌 viz=1，并相互对对方 viz=1
- **Baron 比较后**：分支 1（牌不等）loser 全员 viz=1；分支 2（平手）actor 与 target 互相 viz=1
- **Prince 让 target 弃手**：target 原手牌进 discard pile，公开
- **牌堆抽完的摊牌**：所有仍在场玩家的手牌对所有人 viz=1（"all players still in the round reveal their hands"），因此任何座位的视图都能算出胜负

**谁看过谁的手牌（`__viz_all_viewers__`）**：snapshot 带一个可选键 `__viz_all_viewers__`，里面是某些字段的整张 viz 张量（含所有观察者列），让接收者的搜索知道对手能看到什么。LL 里 `drawn_card` 总是同步（谁手里有刚抽的牌是公开的）；`hand` **只在 2 人局同步**——2 人局里"对手是否看过我的手牌"双方都能推出。3–4 人局不同步：q 偷看过 p 之后 p 出一张牌，q 的知识是否还有效取决于 p 出的是不是被看过的那张，第三个座位无从得知，把 q 那一位发给它等于泄露 p 手牌的信息。手工构造 snapshot 的外部客户端可以不带这个键（退回只同步自己那一列的旧行为）。

接入方只需要：rules 端按规则正确调用 `reveal_slot_to(viewer)` / `swap_slot_owned(a, b)` / `reset_to_base(slot)`，walker 自动把 viz=1 槽位序列化进 snapshot——所有 perspective-private 的真值都跟随 `__viz__` 切片 + `(idx, value)` 对一起在线上传输，没有任何平行的私人事件协议。

---

## state_dict 字段

`state_serializer` 输出（`games/loveletter/loveletter_register.cpp::serialize_loveletter`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 下一个决策者 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 胜者；`-1` 还没结束 |
| `num_players` | int | 2 / 3 / 4 |
| `ply` | int | 累计步数 |
| `deck_size` | int | 牌堆剩余张数 |
| `drawn_card` | int | 当前 actor 抽到的牌；对其他 perspective 是 0 |
| `drawn_card_name` | string | 牌名（同上 viz） |
| `face_up_count` | `int[9]` | 2p 开局移走的 3 张按牌型计数（idx 0 不用，1..8 = Guard..Princess） |
| `face_up_removed` | `int[]` | 同上 expand-form（只在终局或测试里有用） |
| `set_aside_card` | int | 牌堆顶移走的 1 张；任何 perspective 都看不到（仅供测试做守恒断言） |
| `players[p].alive` | bool | 是否还在场 |
| `players[p].protected` | bool | Handmaid 保护状态 |
| `players[p].hand` | int | 当前手牌 cid（自己的恒可见；通过 Priest peek / Baron 比较 / King 交换合法看到的对手 cid 也 viz=1） |
| `players[p].hand_name` | string | 手牌名（同上 viz） |
| `players[p].hand_seen_by` | `int[]` | 当前能看到这张手牌的座位（owner、Priest peek、Baron 平手、King 交换、摊牌）——运行时 viz 行，供测试 / UI 用 |
| `players[p].discard_count` | `int[9]` | 玩家 p 已弃牌按牌型计数（idx 0 不用） |
| `players[p].discards` | `int[]` | 同上 expand-form，UI 用 |

玩家私有：自己的 `hand[my_seat]` 始终可见；`drawn_card` 在自己回合 reveal 后可见。被 Priest peek / Baron 比较 / King 交换合法看到的对手手牌也以 viz=1 出现在 my_seat 的 session state 里——不需要 AI 端单独维护 `known_hand` 字典，直接读 session state 上对应 `players[opp].hand` 即可。

**胜负与平局**：单轮制。只剩一人或牌堆抽完时终局；摊牌比手牌点数，再比弃牌点数和。两者都相同的玩家**并列第一**：`winner` 为 `-1`，`winners` 返回所有并列者，`losers` 返回其余所有人（含已淘汰者），终局价值为并列者各 `+1/nw`、其余各 `-1/nl`；所有玩家全部并列才是和局（全 0）。

**特征编码（维度 70 / 99 / 128 不变）**：两个曾经失效的输入现在是真实值——牌堆张数（下标 `13N`，`deck_size/16`；以前恒为 -64）和"我的手牌已被别人看到"（下标 3；以前在搜索根节点恒为 0）。后者只在 2 人局有意义：3–4 人局里别人是否看过我不是公开信息，该位恒为 0，其余座位的同名位表示"我看过他的手牌"。**在修复前训练的网络**这两个输入上的权重从未被训练过，直接喂真实值会扰动输出；用 `python games/loveletter/migrate_model_features.py <model.onnx | checkpoint.pt>` 迁移（把牌堆列乘 -64 折进首层偏置后清零该列，并清零下标 3 的列；带标记，重复运行无副作用）。仓库自带的三个模型已迁移。

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# AI 扮演 3 人局的第 1 号玩家。
# initial_observation 必须由 GT 端通过
# `gt_session.extract_initial_observation(my_seat=1)` 提供——
# 它两节式 `{public_snapshot, tracker_init}` 携带初始 viz=1 槽位
# (公共牌堆大小、discard 等) 加 perspective-private bootstrap (自己的初始手牌)。
initial_observation = gt_session.extract_initial_observation(1)
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "loveletter_3p", "seed": 99, "my_seat": 1,
    "initial_observation": initial_observation,
}).json()
sid = sess["session_id"]

# Ground truth 端:每个动作 apply 之后,framework 自动算 perspective 视角的
# events + public_snapshot。LL register 用 `viz::no_events_extractor`,所以
# events 永远是空,信息全靠 public_snapshot 走 viz=1 → walker 同传。

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]: break

    if status["current_player"] == 1:
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
    else:
        action_id = opp_pick()

    # GT 端 apply 动作并提取 perspective=1 的 trace。
    # `apply_action_with_trace(action, perspective)` 内部会 do_action_fast
    # 推进 GT,然后调 `public_event_extractor` 算这一步的 events +
    # public_snapshot;返回 {events, public_snapshot}。
    trace = gt_session.apply_action_with_trace(action_id, perspective=1)
    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id,
        "events": trace["events"],
        "public_snapshot": trace["public_snapshot"],
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 常见踩坑

- **observer 路径不重放规则**：动作的具体后果（弃牌、King 后双方的新手牌、谁淘汰、Priest peek 的对手 cid）全部由 `public_snapshot` 携带；接入方只需保证 rules 端正确调 `reveal_slot_to` / `swap_slot_owned` / `reset_to_base` 维护 viz，walker 自动把每个 perspective 的 viz=1 槽位序列化进 snapshot
- **Priest peek 是 per-perspective 的 viz reveal**：`reveal_slot_to(actor)` 只让 actor 这一个 perspective 的 viz 转 1，第三方 perspective 的 snapshot 里那个槽位仍是 placeholder——信息隔离结构上保证
- **淘汰时手牌自然公开**：被 Guard 猜中、Baron 比输、Prince 弃的 Princess 等情况，被淘汰的玩家手牌进入 discard pile，AI 通过 discard_piles state 自然知道
- **target 是绝对座位号**：3/4 人局里同一 action_id 对所有 actor 指向同一玩家——`kPriestOffset + 2` 永远是"对 2 号玩家用 Priest"，不是"对自己之后的第 3 个玩家"。AI 端如果想做"对下家用 Priest"必须自己算 `(me + 1) % N` 后再 +offset
