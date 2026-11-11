# DinoBoard 算法概览 — ISMCTS over DAG with per-field visibility

> 新读者从这里入门：先看名词表，再读一句话主线，然后按 §1–§8 顺读。
> 不变量速查 & 开发者职责清单已移至 skill `game-rules`。
> 配置字段参考见 skill `game-config`。

---

## 名词表

**数据**

| 名词 | 一句话 |
|------|--------|
| **schema** | 每个字段的 name + 数据 shape + base viz tensor；hash / encoder / snapshot 都按 schema 序遍历 |
| **state** | schema 实例化后的字段集合；每个字段额外挂一个 viz |
| **viz** | 与字段同 shape 的可见性张量；`viz[i, p] = 1` 表示玩家 p 能看到该槽位真值 |

**角色**

| 名词 | 一句话 |
|------|--------|
| **GT** | 持有完整 state 的"上帝视角"，按 rules 推进、抽真随机、广播 snapshot |
| **AI session** | 一个 perspective 的客户端，收 snapshot，只持有自己能看见的 state |
| **sim** | MCTS 内一次模拟，从 session 克隆出 state 后 determinize 再往下走 |

**派生**

| 名词 | 一句话 |
|------|--------|
| **walker** | schema 驱动的遍历器，对每个槽位看 viz 决定"该露还是该藏" |
| **masked clone** | walker 的产物——viz=0 槽位被替换成 placeholder 的副本 |
| **snapshot** | GT 每 ply 发出的 wire 消息，包含公开字段 + viz 切片 |
| **hash** | 按 perspective 哈希 masked clone + step_count，作为 MCTS DAG key |
| **encoder** | 把 masked clone 转成 `vector<float>` 作网络输入 |

**Belief**

| 名词 | 一句话 |
|------|--------|
| **tracker** | 维护观察记忆 + 承载 `randomize_unseen`；不读 state 真值 |
| **belief** | `randomize_unseen` 的产物——viz=0 槽位被填上具体值的完整 state |

**算法**

| 名词 | 一句话 |
|------|--------|
| **ISMCTS** | Information Set MCTS：每个 world 入口从 belief 采一个具体世界，用 perspective hash 共享 DAG 节点 |

---

## 一句话主线

GT 和 AI 两端共享 schema + rules + walker，但角色不同：

> **GT 端**：rules 推进 state、同时维护 viz → walker 按每个 perspective
> mask → 打包成 snapshot 广播给对应 AI session。
>
> **AI 端**：收 snapshot 整张替换 session state → 维护 tracker → MCTS 在
> 每个 world 入口从 belief 采样具体世界 → 同一份 rules 推进、同一份 walker
> mask → hash / encoder 消费 masked clone → ISMCTS 搜索决策。

---

## §1 核心概念：Per-field visibility

### Schema 声明

每款游戏在 `<game>_state.cpp::schema()` 中声明所有 state 字段的元数据：
name + 数据 shape + base viz tensor。

- 字段 shape 为 `[D1, …, Dk]`，可见性张量 shape = `[D1, …, Dk, NPlayers]`
  ——末尾多挂一根 viewer 轴
- **Base viz** 是静态默认值：`all_public`（全公开）、`owner_only_first_axis`
  （手牌）、`all_hidden`（对所有人不可见）
- Schema 是 hash / encoder / snapshot scope 的**单一事实源**——没有绕开
  schema 的后门

### Rules 维护 viz

`do_action_fast_impl` 同时推进业务字段和 viz。框架提供 helper：

- `reveal_slot` — 把指定槽位 viz 设为全 1
- `reveal_slot_to(viewer)` — 只对 viewer 可见（Priest peek 风格）
- `reset_to_base` — 重置回 schema 声明的 base（槽位内容变化时用）
- `swap_slot` / `swap_slot_owned` — viz 跟着内容一起对换

Rules 是 viz 的**唯一游戏侧 writer**。AI 端不重算 viz——GT 算一次、协议
整张同传。

### Viz 不单调

Viz 不是累积量。槽位内容变化时 rules 主动 `reset_to_base`（例：Love Letter
出牌后手牌槽清空 → reset → 摸新牌后重新变 owner-only）。AI 端是**整张替换**
not OR-merge。

### Masked clone 是派生水线的 fulcrum

```cpp
auto masked = make_masked_state(state, schema, perspective);
// viz=1 → 复制真值; viz=0 → 写 kPlaceholder sentinel
```

Snapshot / hash / encoder 都由同一份 schema + viz 驱动，不绕过可见性边界。
Snapshot 直接从 live state 按 schema 序列化公开槽位；MCTS sim descent 则每步
materialize 一份 masked clone，hash 和 encoder 共享这同一份副本——先 hash
决定是否新节点，是新节点才 encode。

Encoder 接口入参锁 `const IGameState& masked_state`；非虚入口先 mask 再转
发到虚函数。Viz=0 槽位读到 placeholder——忘记处理最多 emit 占位符特征，结构
上不可能泄漏真值。

---

## §2 GT / AI 分离

GT 持有完整 state（含隐藏字段），按 rules 推进、判终局、抽真随机，每 ply 给
每个 perspective 广播一份 `(actor, action_id, events, public_snapshot)`。GT 不
做 belief、不做搜索。

AI session 只看消息流。收到 observe 后：(1) `public_snapshot` 整张替换 session
公开字段（含 viz 切片），(2) `events` 喂给 tracker 维护观察记忆。Viz=0 hidden
槽位 session 不维护——决策侧物理上读不到（hash 用 sentinel、encoder 用
placeholder、sim 在克隆上 determinize）。

**这不是约定——是结构性不可能泄漏**：

1. **Tracker 的观察入口拿不到 truth state**——`observe_public_event` 只收事件流；
   `init` 只在 session state 上消费 GT 侧打包的 bootstrap payload；`randomize_unseen`
   拿 state 也是 MCTS world clone 上的**写**端（填 viz=0 槽位），不是读真值的入口。
2. **Observer session 不跑 `do_action_fast`**（snapshot-path 游戏）——
   公开字段从 snapshot 整张覆盖，hidden 槽不动。
3. **决策侧读 session state 碰不到 viz=0 槽位**——hash 混
   `kHiddenHashSentinel`、encoder 读 masked clone 上的 placeholder、MCTS sim
   入口在克隆上走 `randomize_unseen`。
4. **Selfplay / web / API 三路对称**——都走同一个 AI session + tracker 栈，
   MCTS root state 永远是 session state，不是 truth。

---

## §3 ISMCTS 算法

### 搜索核心属性

1. **Root 采样 determinization** — 每个 world 入口用独立 RNG 调
   `tracker->randomize_unseen` 采一个完整世界
2. **Per-acting-player 节点 keying** — `state_hash_for_perspective
   (state.current_player())` 作 key
3. **博弈 DAG**（不是 tree）— 同一 acting player 视角下 masked clone +
   step_count 一致就是同一个节点
4. **UCT2 UCB** — `sqrt()` 分子底用"经刚走过的入边到达本节点"的次数，不是
   node 全局 visit_count
5. **无 chance node** — 物理随机由 root determinization 和 sim_rng 处理，
   observer 可见的随机自动通过 hash 差异分叉、不可见的自动合并
6. **Step counter 防环** — `step_count_` 由框架自动递增，纳入 hash，保证
   DAG 结构性 acyclic
7. **Encoder 对齐 hash scope** — encoder 和 hash 读同一组 viz=1 槽位

### RNG 管理

State 上没有 RNG。运行时三类 RNG 互不派生、互不共享：

| RNG | 持有者 | 用途 |
|-----|--------|------|
| gt_rng | GT runner | 整局真随机 |
| session_rng | 每个 AI session | session 内部推导 |
| sim_rng | sim 局部栈变量 | randomize_unseen + descent 中 do_action_fast |

### DAG 数据结构

```cpp
struct Edge { ActionId action; float prior; int child; int visit_count; float value_sum; };
struct Node { int to_play; bool expanded; int visit_count; float value_sum; vector<Edge> edges; };
vector<Node> nodes;                              // 线性存储
unordered_map<StateHash64, int> node_index;      // per-search DAG 查找表
```

`nodes[0]` 是根。Edge 里没有 chance_children——DAG 查找统一走全局表。

### search_root 流程

```
输入: root_state, rules, evaluator, config, search_seed

初始化:
  创建根节点, expand, 加 Dirichlet 噪声
  // 根只有一个合法动作(forced move)且不是分析模式(cover_root_edges):
  // 不跑任何 sim,直接返回该动作——visits one-hot, simulations_done = 0,
  // root_values 取根节点的网络评估。tail solver 照常先跑。

world_cap split:
  K = (world_cap > 0) ? clamp(world_cap, 1, sims) : sims
  M = ceil(sims / K)
  // world_cap == 0(默认) → 每 sim 一个 world
  // world_cap > 0 → K 个 determinization,每个内 M 个 sim

每个 world wk = 0..K-1:
  world_state  = root_state.clone()
  world_tracker = session.belief_tracker.clone()
  world_tracker.randomize_unseen(world_state, observer, world_rng)

  每个 sim wm = 0..M-1:
    sim_state = world_state.clone()
    sim_tracker = world_tracker.clone()

    descent: 选边(UCT2) → do_action_fast → hash → 新节点则 expand
    backup:  沿 path 反向更新 edge/node 统计(value_clip 裁剪)

选 root 最优: argmax(root edges visit_count), 平局 uniform tie-break
```

世界容量（`world_cap`）控制搜索粒度：默认每 sim 独立采一个世界（经典 ISMCTS）；
`world_cap > 0` 让 M 个 sim 共享同一个 determinization——在隐藏信息空间巨大的
游戏（6 nimmt!、Liar's Dice）上避免 bandit 退化成纯 prior 拍板。

### UCT2

标准 PUCT 在 DAG 里有多入边聚合的 exploration 偏差。UCT2（Childs, Brodeur,
Kocsis 2008）把 `sqrt()` 分子底换成"这次 sim 刚经过的那条入边的
visit_count"，反映"我这条路径上到达当前节点多少次"，而不是所有路径算上。
Root 没有入边，用 `nodes[0].visit_count` 替代。

因为没有 chance node，同一条边在不同 sim 里可能通向不同的子节点（observer
可见的随机结果，或 determinization 采到的下家私有信息）。这时入边自己的
visit_count 是它所有结果的总和，直接用它会把概率为 p 的结果节点上的计数
高估 1/p。所以分子底取的是"经这条入边**到达本节点**的次数"
（`NetMctsConfig::uct2_arrival_visits`，默认 true；每搜索一张
(parent, edge, child) → 次数的哈希表，与边的 visit_count / virtual_loss 同步
更新）：边只有一个结果时它就是边的 visit_count，节点只有一条入边时它就是节点
自己的 visit_count（普通 PUCT）。设为 false 回到"入边总 visit_count"的旧口径，
只用于 A/B 对比（arena 的 `seat_configs` 可按座位设置）。

未访问边的 Q 默认取 0。配置 `fpu_reduction` 后改为 Leela/KataGo 的 FPU：
`Q(父节点) − r·sqrt(已访问子边先验之和)`。训练侧的其他 KataGo 技巧
（playout cap、辅助头、EMA）不改变搜索，见 `/training` 技能的
`katago-techniques.md`。

### Step counter 防环

`IGameState::step_count_` 由框架 non-virtual wrapper 管理——`do_action_fast`
自动 +1、`undo_action` 自动 -1。Step count 纳入 hash，parent→child 永远
step 增加，不可能回到同 hash，DAG 结构性 acyclic。游戏作者唯一要求：
`reset_with_seed` 第一行调 `reset_step_count_base()`。

### 物理随机 = sampled world

不需要显式 chance node。Root 采样把"整局所有未来随机"固定成一条世界路径：

- Observer 不可见的随机（opp 抽牌）：不同 sim 抽不同牌，回到 observer 节点时
  masked clone 相同 → hash 合并
- Observer 可见的随机（Splendor 翻新卡）：公开槽位不同 → hash 不同 → 自然
  分叉
- N 次 sim = N 条可能世界轨迹，Q 统计近似真实期望

---

## §4 Tracker 与 belief

### Tracker 边界

State 由 rules 维护（推进游戏所需的全部信息）。Tracker 维护"GT 不 care、
但 NN 想要的公开衍生特征"——纯网络输入辅助。判别准则：

- GT 规则引擎需要 → 进 state
- GT 不需要、NN 想要 → 进 tracker
- 都不需要 → 不存在

Tracker 的读边界是：`observe_public_event` 只能拿事件流，不能读 truth state。
`init` 会在 session state 上消费 GT 侧打包的 `AnyMap` bootstrap（例如自己的起手
牌），并可写入 perspective-private 槽位；`randomize_unseen` 是 MCTS world 入口
的写端，在 cloned state 上向 viz=0 槽填具体世界。

### randomize_unseen 契约

唯一调用点是 MCTS world 入口（在 cloned world_state 上）。契约：产出世界中
所有公开槽位（viz=1）的内容只取决于 tracker 的观察历史，与输入 state 的
hidden 内容、调用者 RNG 无关。

### 两种实现策略

| 情况 | 做法 | 示例 |
|------|------|------|
| 隐藏信息空间小 | 从剩余池均匀采样 | Love Letter / Splendor |
| Bluff 游戏（均匀采样退化） | 信念网络产出后验分布，按后验采样 | Coup / Liar's Dice / 6 nimmt! |

每个隐藏信息游戏都要自己写 `randomize_unseen`。没有框架 default。

**信念网络路径**：游戏注册 `IBeliefFeatureExtractor`（输入特征）+
`belief_model_path`（运行时加载 ONNX evaluator）+ `IBeliefLabelExtractor`（GT 侧
label 产出）。框架在 MCTS root 处跑一次推理得到对手手牌后验 `pi`，缓存结果；
world 入口的 `randomize_unseen` 直接读缓存按后验采样。缺少 extractor 或 model
path 时，`prepare_for_root` 等价 no-op，tracker 回到均匀 / multiset-derived 采样。

---

## §5 Wire 协议

### Snapshot 格式

GT 每 ply 给每个 perspective 发一份 `PublicEventTrace`：

- **`events`**：这次 transition 的公开事实序列（只喂 tracker，不驱动 state
  mutation）
- **`public_snapshot`**：整张 post-action 公开字段 dict + viz 切片

`events` 的 kind / payload 由游戏自定。没有 phase 概念——observer 不跑
`do_action_fast`，事件无"动作前 vs 动作后"之分。

### 统一实现

`viz::serialize_public_snapshot` / `viz::apply_public_snapshot` 走同一条
schema-walker 路径：对每个 `viz[..., perspective]=1` 的 slot 序列化/反序列化，
并把整张 viz 切片以扁平 vector 同传到 `__viz__` 键下。所有 hidden-info 游戏
共用，无 per-game 分支。

`__viz__` 只含接收者自己那一列。搜索里对手决策节点的 mask / hash / encode 用的
是对手那一列，所以"谁能看到哪个槽位"本身是公开信息的字段，要在 schema 里声明
`VizSync::kAllViewers`：这些字段的整张 viz 张量（所有观察者列）走可选键
`__viz_all_viewers__` 同传（Coup 的 `influence`、6 nimmt! 的 `pending_commits`、
Love Letter 的 `drawn_card` 和 2 人局的 `hand`）。可见性模式依赖隐藏值的字段
（Love Letter 3–4 人局的 `hand`、Coup 的 `exchange_drawn`）必须保持默认的
`kReceiverOnly`，否则模式本身会泄露信息。不带该键的 snapshot（旧发送端、外部
手工构造）照常应用，只同步自己那一列。

### 开局观察

`initial_observation` 和逐 ply snapshot 同构——public 部分共享
`serialize/apply_public_snapshot`，perspective-private 部分由
`pack_init_payload` ↔ `tracker.init` 承担。

### 协议字段限制

`CreateSessionRequest` 不接受 `simulations` / `temperature` /
`opponent_selection` 入参——AI 强度 server-controlled。`decide()` 不改 session
状态——只跑 search 返 action_id，GT 收到后 commit + 抽真随机 + 走 observe
回灌给 AI session。

### 游戏接入协议

每个 hidden-info 游戏通过 `GameBundle::install_event_protocol(events_only_diff,
schema_provider)` 注册事件协议。框架自动派生三个钩子：

- `events_only_extractor` — sim 内路径（只产事件增量，不产 snapshot）
- `public_event_extractor` — selfplay / wire 路径（事件 + 框架的
  `viz::serialize_public_snapshot`）
- `public_state_applier` — receiver 路径（框架的
  `viz::apply_public_snapshot`）

游戏只写 events diff，wholesale snapshot 由框架包办。

---

## §6 Opponent selection

非 root 的对手决策节点上，ISMCTS 可选两种选边策略：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `"puct"` (默认) | 照常跑 PUCT bandit | 确定性 / 弱隐藏信息游戏 |
| `"prior"` | 从 policy head 的冻结先验做 multinomial sample | 重 bluff 隐藏信息游戏 |

Frozen-prior 缓解 ISMCTS 的策略融合 / 对手全知偏差——bandit 在不同
determinization 之间积累统计会让对手节点"跨世界共享"，变得不合理地精准。
Acting player 自己的 root selection 始终是 PUCT。

设在 `mcts.json` baseline 里（所有 profile 继承），REST API 不接受 wire 传入。

---

## §7 三类游戏

| 类型 | 示例 | schema 特征 | 需要 belief sample | 需要 tracker |
|------|------|------------|-------------------|-------------|
| 完全公开 + 确定 | TicTacToe, Quoridor | 全 `all_public` | 否 | 可选 |
| 完全公开 + 物理随机 | Azul | 全 `all_public` | 否(rng 在 do_action_fast 即时消费) | 可选 |
| 信息不对称 | Love Letter, Splendor, Coup | 有 owner-only 字段 | 是(sim 入口 randomize_unseen) | 通常需要 |

同一套 MCTS 代码——区别只在 walker 物化 masked clone 时哪些槽位被置
placeholder，以及有没有调用 `randomize_unseen`。Tracker 始终可选——实现质量
决定 AI 强度，不决定能不能跑。

---

## §8 do_action_fast vs do_action_deterministic

两个 action-application 入口，区别只在 undo 支持：

| 入口 | 用途 | 支持 undo |
|------|------|----------|
| `do_action_fast(state, action, rng)` | MCTS / selfplay / arena / web | 否 — 用完丢弃 state |
| `do_action_deterministic(state, action)` | tail solver | 是 — push undo_stack |

**不要在 `do_action_fast_impl` 里 push undo_stack**——sims 跑成百万次，多余
clone 是纯浪费。`undo_action` 的唯一调用点在 `tail_solver.cpp`，永远和
`do_action_deterministic` 配对。

---

## 进一步阅读

- **[FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md)** — 功能概览速查地图
- **[docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md)** — 设计动机
- Skills: `/game-rules`（不变量 + 开发者清单）、`/game-config`（配置字段）、`/game-ai`（AI 组件接入）

## 参考文献

1. **Cowling, Powley, Whitehouse (2012)** — "Information Set Monte Carlo Tree Search". ISMCTS 原始论文
2. **Childs, Brodeur, Kocsis (2008)** — "Transpositions and Move Groups in MCTS". UCT2
3. **Silver et al. (2018)** — "AlphaZero". PUCT + 网络先验
4. **Paolini et al. (2024)** — "Learning to Play 7 Wonders Duel Without Human Supervision" (ZeusAI)
5. **Silver & Veness (2010)** — "Monte-Carlo Planning in Large POMDPs" (POMCP)
