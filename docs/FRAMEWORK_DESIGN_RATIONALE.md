# 框架设计动机与定位

## 0. 立项动机

DinoBoard 的立项动机是一条具体的工程判断：

> **在 LLM 辅助开发（vibe coding）的默认假设下，让规则引擎 + AI
> 行为对人类玩家可信。**

把这条动机拆开：

- **规则引擎可信**——LLM 写隐藏信息游戏的规则代码时，最容易出
  silent bug 的是"什么字段对哪个玩家可见"这条线。这条契约写在文档
  里、靠 reviewer 抓错位、靠 round-trip 测试兜底——传统人类小心写
  代码的项目这套 contractual safety 工作得不错（OpenSpiel 用了多年
  没爆过 bug），LLM 写代码不会"小心"，它会按表面合理的模式生成代
  码，silent drift 不报警。所以 viz 必须从作者契约提到框架结构上：
  作者写错 viz 一格才能泄漏，比作者每写一份序列化都要正确容易抓得
  多。
- **AI 行为可信**——训练 metric（win rate / loss curve / policy
  entropy）正常、AI 在第 7 回合做了一个明显在送的决策，这种 bug 在
  命令行里基本看不出来，**人类对战是唯一可靠的发现渠道**。LLM 改
  了某个 reward shaping、encoder 字段、tracker 加权——metric 看起
  来正常甚至更好，但棋风变诡异——人类对战 30 秒能感觉到不对，
  metric 一周都不会触发警报。所以 web 前端是**强制**而非可选项。

派生出的三条 mechanism：

1. **per-field viz tensor**（§1-2）——把"哪个字段对谁可见"从作者代
   码逻辑提到 state 字段，框架持有这条事实而非作者持有。
2. **GT 与 AI session 物理分离**（§3）——AI 决策路径（belief tracker
   / encoder / MCTS）的接口签名里就没有 `IGameState*` truth，物理上
   拿不到、不需要 reviewer 检查。
3. **强制 web 前端 + 复盘工具**（§4）——人类对战是 AI 行为可信度的
   最终判据，命令行 metric 永远不能替代。

这三条都是 **mechanism 服务于 motivation**——不是各自独立的设计偏
好。后续章节论证每条 mechanism 解决什么具体 silent bug 模式、代价
是什么、什么情况下不值得。


---

## 1. OpenSpiel 的做法

OpenSpiel 有两条路径——legacy 四方法 API 和现代 Observer API（2020
年前后引入）。

**Legacy 四方法 API**（`open_spiel/spiel.h`）：

```cpp
class State {
    virtual std::string InformationStateString(Player p) const;
    virtual void InformationStateTensor(Player p, absl::Span<float>) const;
    virtual std::string ObservationString(Player p) const;
    virtual void ObservationTensor(Player p, absl::Span<float>) const;
    virtual std::unique_ptr<State> ResampleFromInfostate(
        int player_id, std::function<double()> rng) const;
};
```

每份序列化各自决定"哪些字段对玩家 P 可见、按什么 layout 写出"，
多份之间靠作者保证语义对齐。

**现代 Observer API**（`open_spiel/observer.h`）：

```cpp
class Observer {
  virtual void WriteTensor(const State& state, int player,
                           Allocator* allocator) const;
  virtual std::string StringFrom(const State& state, int player) const;
};

std::shared_ptr<Observer> obs = game.MakeObserver(
    IIGObservationType{
      .public_info = true,
      .perfect_recall = false,
      .private_info = PrivateInfoType::kSinglePlayer,
    },
    /*params=*/{});
```

作者实现一次 observer，框架按 `IIGObservationType` 的 flag 组合派
生不同观察类型；`WriteTensor` 拿到的 `Allocator*` 让作者按命名子张
量写。同一个 observer 同时供应 string（info-set key）和 tensor
（network 输入），两者从同一份代码派生——**对齐由实现共享而非作者
契约保证**。

**两条路径共有的事实**：

- 没有 per-slot 结构化的可见性元数据——具体到字段的"现在 viz=1 还是
  viz=0"是 observer 实现内部的代码逻辑。Observer API 把可见性折叠
  成 observation-type 级别（`public_info` / `private_info` /
  `perfect_recall` 三档枚举），更细的粒度由作者代码决定。
- belief sample 由作者写 `ResampleFromInfostate`——这个方法是 `State`
  的成员，作者**有能力**读真值，契约要求只用 info-set 信息（contractual，
  非 structural）。
- 适用范围远比 DinoBoard 宽。对照
  [FEATURES_OVERVIEW.md §框架局限性](../FEATURES_OVERVIEW.md)，
  OpenSpiel 直接 cover 而 DinoBoard 不 cover 的形态包括：需要混合
  策略均衡的扑克类（CFR / Deep CFR / NFSP，自带 `kuhn_poker` /
  `leduc_poker` / `universal_poker`）、动作空间组合爆炸的斗地主类
  （自带 `dou_dizhu`）、合作 / 非零和博弈（自带 `hanabi` /
  `bargaining` / `negotiation`）、单人游戏（自带 `solitaire` /
  `twenty_forty_eight`）、长历史依赖（`hanabi`）、非传递循环 / 对
  手池训练（PSRO）；以及因子化观察随机博弈（FOSG）/ 不完美回忆这
  些形态化博弈论假设。

OpenSpiel 是 game AI 圈的事实标准，覆盖最广、活跃度最高、是审稿人
认得的 benchmark 平台。DinoBoard 在适用范围、算法广度、社区支撑、
学术认可度上都不如 OpenSpiel——只是在隐藏信息桌游 + LLM 协助开发
这条窄路径上做了不同的工程取舍。

---

## 2. DinoBoard 怎么换形

把可见性从作者代码逻辑提到 state 字段：

```
schema 声明字段 + base viz
       ↓
rules 在 do_action_fast 里维护 viz（reveal_slot / reset_to_base）
       ↓
walker make_masked_state(state, schema, perspective) 派一份 masked clone
       ↓
三家消费者（snapshot 序列化 / hash digest / encoder tensor）
入参锁 const IGameState& masked_state —— viz=0 槽位结构性读到 placeholder
```

| 维度 | OpenSpiel（Observer API） | DinoBoard |
|---|---|---|
| 可见性元数据 | observation-type 级（flag），具体字段散在 observer 实现里 | per-slot per-perspective viz tensor |
| Belief sample | `ResampleFromInfostate(State*)`——作者**有能力**读真值，契约只用 info-set 信息 | tracker 接口签名里没有 `IGameState*`，`randomize_unseen(state, rng)` 的 state 是已经 resample 过的 session state |
| Hash / encoder / sampler 一致性 | observer 派生 string + tensor（一份实现）；ResampleFromInfostate 是另一条独立实现 | walker 派一份 masked clone，三家消费者共享同一对象 |

### 2.1 Worked example：假如要在 OpenSpiel 实现 Love Letter

OpenSpiel 现有仓库（截至 2026-05）没有 Love Letter。下面对比"假如
要接进去的最优做法"和"DinoBoard 实际做法"。

场景：p 用 Priest 看了 q 的手牌 = X。下一回合 p 出 Guard，AI 应当
稳定地猜 X。这条信息流要"用得起来"必须同时满足三件：(1) info-set
key 区分"已知 q=X"和"未知 q"，(2) belief sample 锚住 q=X，(3)
network tensor 包含"已知 q=X"特征。

#### OpenSpiel 上的最优做法

不是 observer 内部记 peek_record + resample 里手动 force 那种把可见
性散在三处的写法。最优做法是把可见性物化成 state 字段：

```cpp
class LoveLetterState : public State {
  std::array<Card, kNumPlayers> hands_;
  std::array<std::array<bool, kNumPlayers>, kNumPlayers> hand_known_;
};

void DoApplyAction(Action action) override {
  if (action.kind == kPriest) hand_known_[target][viewer] = true;
  if (action.kind == kDraw && drawer == q) {
    for (int v = 0; v < kNumPlayers; ++v) hand_known_[q][v] = false;
  }
}

void WriteTensor(const State& s, int player, Allocator* alloc) const {
  auto opp_hand = alloc->Get("opp_hand", {kNumPlayers, kCardTypes});
  for (int opp = 0; opp < kNumPlayers; ++opp) {
    if (s.hand_known_[opp][player]) {
      opp_hand[opp][s.hands_[opp]] = 1.0;
    }
  }
}

std::unique_ptr<State> ResampleFromInfostate(int player, ...) const {
  auto sampled = /* replay history */;
  for (int opp = 0; opp < kNumPlayers; ++opp) {
    if (hand_known_[opp][player]) {
      sampled->ForceOpponentHand(opp, hands_[opp]);
    }
  }
  return sampled;
}
```

`hand_known_` 这个字段让 observer / resample 都从同一个 bit 派生。
Priest action 翻 1、q 抽新牌翻 0，rules 一处改、observer 和 resample
自动跟着。

#### DinoBoard 上的写法

```cpp
// rules：
viz::reveal_slot_to(state, "hand", {target}, /*viewer=*/viewer);
viz::reset_to_base(state, "hand", State::schema(), {q});

// encoder 不知道有 peek：
void encode_features(const IGameState& masked_state, int perspective,
                     const IBeliefTracker* /*tracker*/,
                     vector<float>* out) {
  const auto& m = static_cast<const State&>(masked_state);
  for (int seat = 0; seat < num_players; ++seat) {
    encode_card(m.hand[seat], out);
  }
}
// m.hand[seat] viz=1 时是真值、viz=0 时是 kPlaceholder。

// tracker.randomize_unseen 看 viz=1 跳过，只填 viz=0 的格子。
```

#### 客观对比

OpenSpiel 上"最优做法"和 DinoBoard 的核心机制其实是同一个想法。
两边的实际差异：

1. **谁强制**。OpenSpiel 的 `hand_known_` 是作者自己想到要加的——
   API 不要求、framework 不提供、其它游戏不一定这样写。DinoBoard
   把这条模式提到 IGameState 基类强制走。**对自觉、有经验的作者
   这条差异不存在**——他写 OpenSpiel 自然会加 `hand_known_`，写
   DinoBoard 自然走 viz。差异只在两种情形显现：(a) 作者经验不足、
   想不到要这样写，(b) 作者是 LLM、不会"想到"什么、只会按表面合
   理的模式生成代码。
2. **CI 集中**。DinoBoard 的 viz 在基类、所有游戏共享，能写一组
   round-trip 测试跑遍所有游戏。OpenSpiel 即使每款游戏都按最优做
   法写，每家的 visibility 字段命名 / 形状不同，集中 CI 不现实——
   要么每款游戏单独写 round-trip 测试、要么靠各家作者自觉。
3. **encoder 端代码量**。OpenSpiel `WriteTensor` 仍然要写 per-game
   的 if + 字段命名 + layout 设计（"opp_hand" 还是 "known_opponent"？
   要不要加 `did_peek` mask bit 区分"没看过"和"看过 0号"？）。
   DinoBoard `encode_card(m.hand[seat], ...)` 和"有没有 peek"完全
   无关。这条差异在动态揭示频繁的游戏（Priest peek、Coup exchange）
   里显著；在 Bridge 这种发完牌就基本固定的游戏里不太能体现。

**所以 viz tensor 不是 OpenSpiel 做不到的事，是 DinoBoard 把它
做成了默认形状**。如果你是熟练的 OpenSpiel 用户，按最优做法写
LL，简洁度和 DinoBoard 接近。如果你是 LLM、或者经验不足的作者，
DinoBoard 强制结构帮你避开"observer 内部 if-else 散开"那种偷懒
写法导致的 silent bug。

### 2.2 viz 写错的 blast radius

| 写错形式 | 后果 |
|---|---|
| 该 reveal 没 reveal（viz=0 但其实公开了） | observer 多 sample 不可能的世界，强度↓；游戏正常 |
| 高阶推理漏写 | 该分支多 sample，强度↓；游戏正常 |
| **viz 把 0 写成 1** | walker 按 viz 拷真值进 wire snapshot，把不该看的真值传给 observer = 信息泄露 |

写漏 = AI 弱一点；把 0 写成 1 = 正确性塌方。CI 强制跑
`test_public_snapshot_round_trip`、`test_encoder_respects_hash_scope`、
`test_public_hash_excludes_internal_rng`——把已知可枚举的泄漏路径
前置到 CI。

OpenSpiel 那边 observer 内部"判断字段对 perspective 是否可见"的逻
辑写错也是 silent 的，定位方式同样靠 round-trip 测试。两边都靠测试
兜底；DinoBoard 的差异是把可见性从"observer 内部代码逻辑"提到
"state 上的 viz tensor"——契约表面积更小（一格 viz 而非一段 if-
else），但不是 0。doc 早期版本用过 "structurally impossible to leak"
这种措辞，更准确的说法是**契约表面积更小、CI 可枚举的泄漏路径前
置覆盖**，作者写错 viz 仍然能泄漏。

---

## 3. GT 与 AI session 物理分离

第二条 mechanism。motivation 是同一个：vibe coding 下"AI 偷读 truth
的隐藏字段"是 silent bug，要从结构上让它写不出来。

### 3.1 OpenSpiel 上的形状

`Bot::Step(const State& state)` 接受任意一个 State&——bot 没法分辨
传进来的是 truth 还是 sample。Runner 喂什么是 runner 的事。所以做
GT/AI 分离不需要改基类，写一个 per-seat runner 就行：

```cpp
class SeparatedRunner {
  std::unique_ptr<State> truth_;
  std::vector<std::unique_ptr<State>> per_seat_;

  void Step() {
    int actor = truth_->CurrentPlayer();
    Action a = bots_[actor]->Step(*per_seat_[actor]);
    truth_->ApplyAction(a);
    for (int p = 0; p < n; ++p) {
      per_seat_[p] = truth_->ResampleFromInfostate(p, rng_p_[p]);
    }
  }
};
```

bot 拿到的 `*per_seat_[actor]` 是 resample 出来的 fresh state，没有
truth 的指针、没有 truth 的隐藏字段。OpenSpiel 上"做到 GT/AI 分
离"完全可行，工作量是写一个几百行的 runner + 每款游戏审一遍
`ResampleFromInfostate` 实现没读 truth 隐藏字段。

### 3.2 DinoBoard 强制走法

`IBeliefTracker::randomize_unseen(state, observer, rng)` 接口签名里没
有 `truth`：`state` 是 session（runtime 喂进来的是
`per_seat_states[observer]`），`observer` 是要填谁视角下 viz=0 槽位
的 perspective。MCTS 的入口同构——root state 永远是当前 acting
player 的 session state，runner 物理上没把 truth 指针递给搜索。作
者**没有能力**在这条路径上读 truth——不是作者克制，是接口里没
传。

### 3.3 客观对比

OpenSpiel 路线和 DinoBoard 路线在**功能正确性上完全等价**。差异
只在两点：

1. **审计成本**。OpenSpiel 上每款游戏的 `ResampleFromInfostate` 实
   现要审一遍——`*this` 是 truth、作者写实现时有能力读隐藏字段。
   DinoBoard 这条免审——接口里没传，物理上读不到。
2. **新游戏的"想到"成本**。OpenSpiel 上写新游戏时作者要主动想
   "我这个 ResampleFromInfostate 不能读 hidden 字段"——经验作者
   想得到，新手可能想不到。DinoBoard 这条不需要"想到"——接口里
   没传。

跟 §2 viz 一节同样的结论：**对自觉的、有经验的人类作者，这条架
构差异接近 0**。差异在 vibe coding / 新手作者上才显现。

如果你的项目是单维护者 + 自己每行都看代码，OpenSpiel 那种 contractual
safety 完全够用——审一次 `ResampleFromInfostate` 不是负担。如果是
LLM 协助开发，contractual safety 的可信度大幅下降，把契约从作者
搬到接口签名上才有意义。

### 3.4 真实代价

GT/AI 分离不是免费的：

- **per-seat session 是 N 倍内存**——每个 seat 一份 state。在 8 款
  桌游 roster 里 state 不大、可接受；在状态极大的游戏里这条成本不
  能忽略。
- **接口契约更严**——`do_action_fast` 不能依赖外部 history，所有
  rules / viz 用得到的历史信息必须物化进 state。这条在 §3.5 详
  细说。
- **session 的 viz=0 字段每 ply 重抽**——session state 的 hidden 字
  段不是 truth 的拷贝，是 tracker 信息集里的一个 sample。下游代
  码不能依赖"session.hidden == truth.hidden"——这条在 OpenSpiel 上
  也成立（resample 出来的 state 同样不是 truth），但 DinoBoard 强
  制每 ply 重抽，作者更容易踩到这条。

### 3.5 viz 必须是 (state, action, next_state) 的纯函数

rules 在 `do_action_fast` 里算 viz 时只能拿到当前 state、刚执行的
action——**不能依赖外部 history buffer**。任何 rules / viz 用得到
的历史信息必须物化进 state。

合理的桌游基本都满足。**真正不能表达的是闭眼游戏**——狼人杀夜
晚、密写动作这种"viz 本身不是公开规则"的形态。框架假设"哪个槽位
对谁可见"是 god-view 下唯一确定的游戏规则：每个玩家都知道"自己
看得到 X、看不到 Y、对手能看到 Z"，只是看不到 Y、Z 的内容。闭眼
环节违反这条——平民不知道狼人之间夜里有没有互认、其他玩家不知道
法官有没有私下告知某玩家牌面。这种"viz 嵌套在 hidden 上"的高阶
不确定性既不能写成确定的 base viz、也不能写成 `do_action_fast` 里
的 deterministic reveal。这是范式硬冲突，不是补 helper 能解决的，
详见 [FEATURES_OVERVIEW.md §框架局限性](../FEATURES_OVERVIEW.md)
第 6 条。

---

## 4. 强制 web 前端

第三条 mechanism。motivation 不是"用户体验"，是 **AI 行为可信度**。

### 4.1 命令行 metric 看不出来的 bug

vibe coding 下 LLM 改了某段代码——可能是 reward shaping、可能是
encoder 字段顺序、可能是 tracker 的加权策略——跑训练，metric 全部
正常：

- win rate 没掉
- value loss 在下降
- policy entropy 在合理区间
- 对照组打不过 latest

但如果你坐下来跟 AI 玩一局：

- AI 在第 7 回合明显送一张能赢的牌
- AI 永远不出某张特定的卡（feature 编码错了，那张卡的特征向量是 0）
- AI 在残局阶段开始随机走（value head 在 terminal 附近发散）
- AI 总是先攻击某个特定 seat（rotation 哪里搞错了，AI 把所有对手都
  当成 seat 0）

这些 bug 在 metric 上全部不报警——win rate 不掉是因为对照组也是同
一份代码、同样有 bug，**互相对消**。policy entropy 正常是因为 AI
还在做"决定性"决策，只是决策错了。

发现这些 bug 唯一可靠的方式是**人类对战 + 复盘**。命令行 dashboard
做不到。

### 4.2 web 前端不是 nice-to-have

DinoBoard 把 web 前端列为强制项——每款游戏接进来的同时必须有可玩
前端、可视化的 AI 决策、回放工具（设计原则见 `/web-dev` skill）。
列出硬性原则（spatial anchoring / 动画过渡 / current player indicator
/ heavy compute 不锁主线程）。这些不是 UI 美学要求，是确保人类对
战时能看清 AI 在做什么——AI 行为不对劲时第一时间能定位到哪一手。

### 4.3 这条对自审型开发者过分

如果你不用 LLM 协助、自己每行代码都看、能审 reward shaping / encoder
/ tracker 加权——你不需要强制 web 前端。你 review 代码时就能抓住
"feature 编码错了"、"rotation 搞错了"这种 bug。命令行 metric 加
单元测试足够。

DinoBoard 把 web 列为强制项是为 vibe coding 场景做的 trade——单维
护者 + LLM 协助 + 没有 reviewer，web 对战是最后一道防线。这条 trade
对**不在这个场景里的人**是过度投资。alpha-zero-general / OpenSpiel
都没有强制前端，他们的目标用户不需要这道防线。

---

## 5. 和其它框架的对比

### DinoBoard 不适合的场景

- **不用 LLM 写代码** — viz/schema 的抽象成本对手工审代码的人是负价值，OpenSpiel 更合身
- **只做一款游戏** — 抽象成本要 3+ 款游戏才摊得回来，单游戏用 KataGo / alpha-zero-general 更划算
- **做研究** — DinoBoard 不是学术 benchmark，选 OpenSpiel
- **完全公开信息博弈**（围棋、国象）— viz/GT分离都是空抽象，OpenSpiel 直接够用
- **范式硬冲突** — 精确混合均衡 / 动作爆炸 / 合作博弈 / 单人游戏 / 闭眼环节（详见 [FEATURES_OVERVIEW.md](../FEATURES_OVERVIEW.md)）

DinoBoard 的目标受众很窄：LLM 协助开发 / 3+ 款隐藏信息桌游 / 需要 web 前端闭环。

| 框架 | 游戏类型 | 隐藏信息? | Web 前端? | 状态 |
|---|---|---|---|---|
| Stable-Baselines3 / Tianshou / RLlib | 通用 RL（非博弈） | 不擅长 | 无 | 活跃 |
| PettingZoo / Gymnasium | 仅 env 接口 | env only | 无 | 活跃 |
| RLCard | 卡牌 | ✓ | 有 | 活跃 |
| Polygames | 完全信息博弈 | ✗ | 无 | 2022 起无更新|
| alpha-zero-general | 完全信息博弈 | ✗ | 无 | 最后更新 2025-01 |
| KataGo / Leela Chess Zero | Go / Chess 单游戏 | ✗ | 有 | 活跃 |
| OpenSpiel | 通用博弈 | ✓ | 无 | 活跃 |
| **DinoBoard** | 通用博弈（适用范围内） | ✓ | 强制 | 活跃 |

### 5.1 通用 RL 框架（SB3 / Tianshou / RLlib）

model-free RL（PPO / DQN / SAC）的轮子。对完美信息棋类游戏 model-
free 远比 model-based search 弱。对隐藏信息游戏更糟——这些框架根
本没 ISMCTS / determinization 概念。工具错了。

### 5.2 PettingZoo / Gymnasium

只是 env 接口。不提供训练 pipeline、不提供搜索、不提供 belief
tracking、不提供前端。把 DinoBoard 拆开看，PettingZoo 只对应到
IGameState / IGameRules 那一层。

### 5.3 RLCard

定位是扑克 / 斗地主等 mixed-strategy 重的卡牌做的 model-free RL
benchmark。算法栈：DQN / NFSP / DMC / CFR-family——没有 search-
based 算法。

### 5.4 Polygames

Facebook 2019-2021 年的 AlphaZero 研究框架。多人零和博弈侧只支持
完美信息（Go / Chess / Hex / Amazons / Connect6 / Othello 等
AlphaZero 系标准曲目）。2022 年起无人维护——生产路径上等于不可用。

### 5.5 alpha-zero-general

AlphaZero 算法的通用实现。Scope 限定在完全信息博弈，没有隐藏信息
支持。

### 5.6 KataGo / Leela Chess Zero

Go / Chess 单游戏 SOTA。代码深度优化、社区活跃、有可玩前端。**单
游戏（如果是 Go / Chess）开发者的最优选择**——比 DinoBoard 强得
多，没有第二种合理选择。

### 5.7 OpenSpiel

Game AI 圈事实标准，覆盖最广。算法库完整（AlphaZero / ISMCTS / CFR / Deep CFR / NFSP / PSRO / DQN）。

与 DinoBoard 的核心差异只有三条（§1-§3 已详述）：viz 事实在 observer 代码内 vs state 字段上；GT/AI 分离靠契约 vs 靠接口签名；没有 web 前端。算法层面两边等价（SO-ISMCTS + DAG），selection 差异是 UCB1 vs PUCT+UCT2。

对熟练作者，差异接近零。对 LLM 写代码，DinoBoard 的结构性防线比契约性防线更可靠。

---

## 6. 总结

DinoBoard 把隐藏信息桌游的可见性契约从"作者小心写对"提到"接口结构上没有写错的位置"，并强制每款游戏带 web 前端——训练指标骗不过人类 30 秒的试玩。

在 LLM 协助开发的场景下，这条 trade 是正价值：

- **结构性安全** — viz tensor + GT/AI 物理分离让 LLM 写的代码没有泄漏隐藏信息的路径
- **端到端自动化** — 从规则实现到训练到 web 对战，一条 pipeline 跑通
- **LLM 原生工作流** — Claude Code skill 按任务自动加载正确的接口参考和已知坑点，OpenSpiel 没有这套（它面向手写代码的研究者，不需要）

八款游戏已经验证了这条路线：从完全信息（TicTacToe / Quoridor）到对称随机（Azul）到非对称隐藏（Splendor / Love Letter）到 bluff（Coup / Liar's Dice）到同时行动（6 nimmt!），每种类型都有训练好的 AI 和可玩的 web 前端。
