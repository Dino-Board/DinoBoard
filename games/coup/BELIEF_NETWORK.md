# Coup 信念网络（Learned Belief Posterior）

本文档介绍 Coup 上启用的网络化 belief 实现。框架级接口/调用流见
[ALGORITHM_OVERVIEW §8.4](../../ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior)；
配置字段见 [CONFIG_REFERENCE belief.json](../../docs/guide/CONFIG_REFERENCE.md#beliefjson可选仅启用网络化-belief-的游戏)。
本文只讲 Coup 自己的特征 / 输出 / tracker 状态机 / 采样公式。

---

## 1. 为什么 Coup 需要它

Coup 的 root determinization 必须给每个对手抽一组具体的影响牌。最朴素
的 `randomize_unseen` 是从 `remaining[R]`（公开池子里 R 角色还剩几张）
按 R 的张数 weighted 抽：

```
weight[R] = remaining[R]    // uniform-by-pool
```

这条在 Splendor 这种"袋里有什么我都看得到"的游戏里没问题。**Coup 上
有结构性 bug**：

1. **诈唬游戏的纳什均衡是混合策略**——但 ISMCTS 训练的是确定性策略。
   均衡上"该 claim Duke 的频率"是一条软概率；网络只能拟合 argmax。
2. **如果 belief 也 uniform**，对手早期 claim Tax 的信号在 sim 里完全
   消失——sim 入口随手把对手抽成 Captain / Ambassador 都和"实际 claim
   过 Duke"一样。结果：网络收敛到**永远 Challenge**——因为它在 sim
   世界里看到的对手手牌跟 claim 无关，挑战的胜率被 inflate。

**Coup 实际想要的后验**：

```
P(opp 持有手牌 H | 公开历史)        // H 是 multiset，alive=2 时 |H|=2，alive=1 时 |H|=1
```

**网络化 belief 的解法**：训练一个独立的小网络，输入 perspective 视角
下的 claim/challenge 公开历史，**直接输出对手完整手牌 multiset 的后
验**——不是 5 维 marginal、不是边缘分布的拼凑，而是 `15 + 5 = 20` 维
categorical：

- 15 维 = `C(5,2) + 5 = 10 + 5` 个**二张 multiset**（包含同名对子如
  `(Duke, Duke)`），用于 `alive=2` 的对手；
- 5 维 = 5 个**单张** R，用于 `alive=1` 的对手（一张已被 reveal）。

每个对手按当前 `alive` 切到 15 维或 5 维分支，分支内是普通
categorical。**label 空间 = 采样空间 = multiset，一次 categorical 就
到位，没有任何无放回组合的概念。**

对 deck 槽位（无 belief 网络 row）保留 uniform—— deck 是公开剩余池
本身，slot 间无顺序差别，剩余 `remaining[R]` 直接填入即可。

**没有 belief 网络时的 fallback**：直接把所有 viz=0 槽位（不分 opp /
deck）当作从 `remaining[R]` 计数加权的 multinomial without
replacement——逐 slot 抽一张、抽到的角色 `remaining` 减一。语义上等价
"把剩余池洗一次然后顺序发给所有 viz=0 卡牌"。

---

## 2. 整体数据流

```
                  GT (truth)                          AI session
                  ────────                            ──────────
                                                      tracker
                                                      ├── observe_public_event(...)  ← claim/challenge 事件流
                                                      └── prepare_for_root(masked_root, root_player,
                                                                            extractor, evaluator)
                                                            │
                                                            ▼
                                          ┌─── extractor.extract_from_state ───┐
                                          │   (输入 6 + 28×(N-1) 维 features)    │
                                          └────────────┬───────────────────────┘
                                                       ▼
                                         ┌── OnnxBeliefEvaluator.evaluate ──┐
                                         │      (N-1) × 20 logits           │
                                         │   per opp = [15 multiset | 5 单张]│
                                         └────────────┬─────────────────────┘
                                                      ▼
                                 按 alive 切分支 → softmax(logits / T) 单分支内
                                                      ▼
                                       tracker.pi_hand_[opp] 缓存（20 维）
                                                      ▼
       ┌─ MCTS sim 入口 ───────────────────────────────┐
       │ sim_tracker = session.tracker.clone()         │ ← pi_hand_ 一起 clone
       │ sim_tracker.randomize_unseen(sim_state, ...)  │
       │   ↳ 对每 opp：取 alive 对应分支               │
       │      → feasibility mask (need[h][R] ≤         │
       │         remaining_after_alloc[R]) → renorm    │
       │      → 一次 categorical 抽完整手牌            │
       │   ↳ deck slot：从公开剩余池均匀抽             │
       └───────────────────────────────────────────────┘

  selfplay 训练时另一条 GT-side label 流：
  ────────────────────────────────────────
  GT runner 每 ply 调 belief_label_extractor.extract(truth, observer)
    → (hand_multiset_id[opp], remaining[R], alive[opp])
  连同 observer 视角 features 一起 emit 成 BeliefSample
  下一个 PV-step 训练循环里：
    target = one-hot at hand_multiset_id（按 alive 选 15 维或 5 维分支）
    loss   = CE(softmax(logits[opp][branch] / T), target)，dead opp 不算
```

四条物理屏障保证决策侧不读真值（详见
[ALGORITHM_OVERVIEW §8.4](../../ALGORITHM_OVERVIEW.md#84-可选网络化-belieflearned-posterior)
"关键性质"）：

1. `extract_from_state` 入参锁 `const IGameState& masked_state`，非虚
   入口先 `make_masked_state` 再 forward，viz=0 永远是 placeholder；
2. `belief_label_extractor` 物理上只在 GT runner 持有，`AI session /
   tracker / wire / 训练循环之外` 都拿不到；
3. `randomize_unseen` 在 sim_tracker（clone）上调，session tracker 永远
   不被 sim 改写；
4. `pi_observer_` 校验：sim 里要用网络化路径必须 `pi_valid_ == true` 且
   `pi_observer_ == observer`，否则走 §6 的 uniform-from-pool fallback。
   多 multiset 分布 `pi_hand_` 与 `pi_observer_` 共生命周期，clone 一并
   复制。

---

## 3. Tracker 维护的状态机

Tracker 的 raw 字段（`coup_net_adapter.h:`）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `pre_claim_counts_[p][r]` | `N × 5` | **跨过最近一次洗牌**累计的 claim 次数 |
| `post_claim_counts_[p][r]` | `N × 5` | 自最近一次洗牌**之后**新的 claim 次数 |
| `pre_challenge_initiated_[p][r]` | `N × 5` | 同上，发起挑战的次数 |
| `post_challenge_initiated_[p][r]` | `N × 5` | 同上 |
| `last_reshuffle_kind_[p]` | `N` | `{None, Exchange, RevealTruthful}` 之一 |
| `last_revealed_role_[p]` | `N` | 上次洗牌对应被亮出的 role（Exchange 时 -1） |
| `pending_claimer_` / `pending_claim_role_` / `pending_challenged_` | scalars | 正在解决的 claim cycle 状态 |

Tracker 不维护任何"分数"或加权和——只累加纯公共事件计数。

### 3.1 claim/challenge 事件如何累加

每收到一个 public event，处理在 `observe_public_event`
（`coup_net_adapter.cpp:264-393`）：

- **Action-level（claim opening）**：`actor` 出 Tax / Steal / Assassinate
  / Exchange / Block 之一 → `post_claim_counts_[actor][claimed_role] += 1`，
  并把 `(pending_claimer_, pending_claim_role_)` 设到这个 actor。
- **Action-level（challenge declared）**：`actor` 出 Challenge →
  `post_challenge_initiated_[actor][pending_claim_role_] += 1`。
- **Event `claim_unchallenged` / `block_unchallenged`**：通过 → 清
  pending 状态。**不动 pre/post**——没洗牌。
- **Event `claim_resolved_truthful` / `block_resolved_truthful`**：
  挑战失败、claimer 把那张牌洗回牌堆（Coup 规则）→
  `promote_post_to_pre(c)` +
  `last_reshuffle_kind_[c] = RevealTruthful` + `last_revealed_role_[c] = r`。
- **Event `card_revealed`**：lose-influence / 挑战成功导致那张影响牌
  公开亮出。**不 promote**——那张牌物理离场而非回牌堆，pre/post 不动。
- **Event `exchange_complete`**：Ambassador 换完牌 →
  `promote_post_to_pre(p)` + `last_reshuffle_kind_[p] = Exchange`。

### 3.2 promote_post_to_pre：累加，不替换

关键的洗牌处理（`coup_net_adapter.h:114`）：

```cpp
void promote_post_to_pre(int player) {
  for (int r = 0; r < kCharacterCount; ++r) {
    pre_claim_counts_[player][r] += post_claim_counts_[player][r];   // 加上
    post_claim_counts_[player][r] = 0;                                // post 清零
    pre_challenge_initiated_[player][r] +=
        post_challenge_initiated_[player][r];
    post_challenge_initiated_[player][r] = 0;
  }
}
```

语义上 `pre + post = 这个玩家全游戏累计 claim 次数`。**洗牌前的
claim 不会被丢弃**，只是被打上"已经经过一次洗牌、信息含金量降低"的
标记——下一轮 claim 落到 fresh `post`，belief 网络可以根据这两个数
组的 split 学到"刚 claim 的更可信" vs "claim 过但已经洗过牌、可能不
持有了"的区别。

### 3.3 pending claim cycle

Coup 一个 claim 跨多个 ply（claim → 对手 challenge / 对手 allow → 揭
牌 / 跳过）。tracker 用 `pending_claimer_ / pending_claim_role_ /
pending_challenged_` 三个 scalar 跟踪这个 cycle：

- 出牌阶段（任一 claim-bearing action） → 设 pending；
- Challenge → 设 `pending_challenged_ = true`；
- claim_resolved_truthful / block_resolved_truthful / card_revealed
  对上 pending → 清 pending；
- claim_unchallenged / block_unchallenged 对上 pending → 清 pending。

pending 三元组直接进 §4 的特征向量——通过 raw 计数的 pre/post 切分以及
`last_reshuffle_kind` / `last_revealed_role` one-hot，belief 网络可以
学到"刚 claim 还没 resolve 的角色更可信"这类时序结构。fallback 路径
（§6.2）不读 pending。

---

## 4. Feature 输入（28×(N-1) + 6 维）

由 `CoupBeliefFeatureExtractor::extract`（`coup_net_adapter.cpp:748` 起）
emit。布局：

### 4.1 Global block（6 维）

| 维度 | 含义 |
|---|---|
| 5 | `remaining[R]` for `R in {Duke, Assassin, Captain, Ambassador, Contessa}`，observer 视角公开剩余池 |
| 1 | `ply_count / 200`，进度 |

`remaining[R]` 的算法是：3（每角色总数）− 公开 revealed 的 R 张数 −
observer 自己未 reveal 的 R 张数。其它玩家未 reveal 的槽位是 viz=0
→ kPlaceholder，**结构性贡献为 0**——这正是 belief 网络要估算的部分。

### 4.2 Per-opp block（28 × (N-1) 维）

按 `opp_to_player(observer, i, N) = (observer + 1 + i) % N` 的 i 顺序
排列——这与 belief net 输出的 `(N-1) × 5` 行索引对齐。每个对手 28 维：

| 维度 | 含义 |
|---|---|
| 5 | `pre_claim_counts[R] / 4` clamp [0, 1] |
| 5 | `post_claim_counts[R] / 4` clamp [0, 1] |
| 5 | `pre_challenge_initiated[R] / 4` clamp [0, 1] |
| 5 | `post_challenge_initiated[R] / 4` clamp [0, 1] |
| 3 | `last_reshuffle_kind` one-hot ∈ {None, Exchange, RevealTruthful} |
| 5 | `last_revealed_role` one-hot（kind != RevealTruthful 时全 0） |

**所有维度都来自 tracker 的公开事件累积**——没有任何字段读 truth state
的 `influence[]`。这是网络无法泄漏的物理保证。

---

## 5. 网络结构与输出

| 项 | 值 |
|---|---|
| 输入维度 | 6 + 28 × (N-1) = `34/62/90`（2p/3p/4p） |
| 输出 | `(N-1) × 20` logits per opp，分两段：`[0..14]` 二张 multiset、`[15..19]` 单张 |
| 网络 | MLP `[256, 256, 128]`（默认）+ ReLU + BatchNorm |
| 温度 | 框架固定 `board_ai::kBeliefTemperature = 1.0f` |
| 后处理 | 按 `alive` 切分支，分支内 softmax(logits) → `pi_hand_[opp]` |

### 5.1 二张 multiset 枚举（15 维）

按字典序固定下标 `m = 0..14`，对应 `(R_a, R_b)`，`R_a ≤ R_b`，
角色编号 `D=0, As=1, Cap=2, Amb=3, Con=4`：

```
m=0  (D, D)        m=5  (As, As)      m=10 (Cap, Amb)
m=1  (D, As)       m=6  (As, Cap)     m=11 (Cap, Con)
m=2  (D, Cap)      m=7  (As, Amb)     m=12 (Amb, Amb)
m=3  (D, Amb)      m=8  (As, Con)     m=13 (Amb, Con)
m=4  (D, Con)      m=9  (Cap, Cap)    m=14 (Con, Con)
```

`C(5+2-1, 2) = 15`。编号由 `coup_net_adapter.cpp` 中的
`enumerate_two_card_multisets()` 表静态生成。`need[m][R]` =
multiset m 中 R 的张数（0/1/2），是 deck feasibility mask 的查表。

### 5.2 单张分支（5 维）

下标 `r = 0..4` 对应 `{Duke, Assassin, Captain, Ambassador, Contessa}`，
`alive=1` 的对手用此分支。

### 5.3 温度

Belief logit temperature 不可配置，框架固定为 1.0。网络输出什么分布，
tracker 就按原始 logits 做分支内 softmax；如果需要更尖锐或更平滑的后验，
应该通过训练校准模型，而不是在运行时加隐藏温度变换。

---

## 6. Multiset categorical 采样

`randomize_unseen`（`coup_net_adapter.cpp:411-...`）在 sim 入口被
clone 后的 sim_tracker 调用一次，把 viz=0 槽位填上具体 role。两条路径：

### 6.1 Belief-net 路径（`pi_valid_ && pi_observer_ == observer`）

**Per-opp 一发抽完整手牌 + deck 兜底**：

```
1. 收集所有 viz=0 slot；按 owner 分组：opp slots[oi] / deck slots
2. 全局 pool 起点：global_remaining[R] = remaining[R]   // observer 视角公开剩余池
3. 用 sim_rng `std::shuffle` opp 顺序，遍历每个还有 viz=0 slot 的 opp：
     a = alive[opp(oi)]                          // 1 或 2，由 viz=0 slot 数量推出
     if a == 2: prob = pi_hand_[oi][0..14];  need_table = need_2card[m]
     else:      prob = pi_hand_[oi][15..19]; need_table = need_1card[r] (one-hot)
     # feasibility mask：need[h][R] ≤ global_remaining[R] 才合法
     for h in branch:
         prob[h] *= 1 if all_R need[h][R] <= global_remaining[R] else 0
     renormalize prob; if sum==0 → throw（pool 与 alive 矛盾，coup 规则不应出现）
     h* = categorical_sample(prob, sim_rng)
     # 把抽到的 multiset 落到 opp 的 viz=0 slot 上（slot 内顺序无所谓，二张多元集对称）
     for R in need_table[h*]:
         take one viz=0 slot of this opp, write R; global_remaining[R] -= 1
4. deck slot：直接按剩下的 global_remaining 公平抽（uniform without replacement）。
```

注意点：

1. **池一致性硬约束**——开始时若 `sum(remaining) != viz=0 slot 总数` 直接
   throw `std::logic_error`，按 CLAUDE.md "No Fallbacks"。
2. **observer 自己的槽位**结构性 viz=1，不应进 `slots`；如果走到这里
   说明 viz mis-tagged，throw。
3. **label 空间 = 采样空间 = multiset**——单次 categorical 即得到完整
   手牌，没有 marginal 拼装也没有无放回组合。
4. **遍历顺序由 sim_rng 随机化**——多个 opp 的采样通过 `global_remaining`
   feasibility mask 耦合（前一个 opp 抽掉某 role 后，后面的 opp 在该 role
   上的 candidate 可能被 mask 掉），固定 oi 升序会让 sim 之间产生系统性
   偏置。`std::shuffle(opp_order, sim_rng)` 让顺序成为 sim 噪声的一部分，
   每个 sim 的耦合方向独立，整体期望就是真后验。
5. **softmax 在根上做一次**：`pi_hand_[opp][...]` 在 `prepare_for_root`
   阶段按固定温度 1.0 做了分支内归一化（branch 2 [0..14] 与
   branch 1 [15..19] 各自独立 softmax），sim 里直接读概率、做掩码、
   重归一化即可，无需再调网络。

### 6.2 No-belief fallback：uniform-from-pool

belief.onnx 没加载（训练初期）/ 没 `prepare_for_root` / `pi_observer_
!= observer` 时走这条降级路径。把所有 viz=0 槽位（不分 opp / deck）
当成"剩余池洗一次后顺序发牌"：

```
remaining[R] = observer 视角剩余池
for slot in 所有 viz=0 槽位（按 owner 升序、slot index 升序）:
    total = sum(remaining)
    if total <= 0: throw std::logic_error（池耗尽，必为 bug）
    u = uniform_int(0, total - 1, sim_rng)
    R = the role whose cumulative count covers u
    write R to slot
    remaining[R] -= 1
```

这等价于 multinomial without replacement——每张抽到的概率正比于剩余张
数。语义上和 §1 那条"如果 belief 也 uniform" 一致——MCTS 会退化到
"永远 Challenge"，所以 belief.onnx 训出后必须接管。fallback 的作用是
让 belief.onnx 缺位时 selfplay / 测试不 crash。

---

## 7. 训练数据与损失

`CoupBeliefLabelExtractor`（`coup_net_adapter.cpp:850`）在 selfplay
runner 的 emit 路径调用，**只在 GT 端**：

```cpp
struct BeliefSample {
  std::vector<float> features;          // 6 + 28×(N-1)
  // 二选一：alive==2 时填 hand_multiset_2card_id（0..14），否则 hand_single_id（0..4）
  std::vector<int> hand_multiset_id;     // (N-1)，对应 alive 分支内的下标
  std::vector<int> remaining;            // 5
  std::vector<int> alive_per_opp;        // (N-1)，0/1/2
};
```

label 来自 truth：
- `hand_multiset_id[i]` = `enumerate_two_card_multisets` 索引（alive=2）或
  单张 R 索引（alive=1）；alive=0 的 opp 标 `-1` 表示无 label。
- `remaining[R]` 同 §4.1 公式但走真值。
- `alive_per_opp[i]` ∈ {0, 1, 2} = opp i 当前未 reveal 的影响牌张数。

**Loss = cross-entropy on multiset one-hot**（`coup/config/belief.json
::training.loss = "ce"`，per-opp branch 内 CE）：

```python
for opp in opps:
    a = alive[opp]
    if a == 0:                              # 全 reveal 完了，无 belief 可学
        continue
    branch = logits[opp][0..14] if a == 2 else logits[opp][15..19]
    target = one_hot(hand_multiset_id[opp], depth=15 if a == 2 else 5)
    loss += cross_entropy(softmax(branch / T), target)
loss /= num_opps_with_alive_gt_0
```

target 天然在概率单纯形上（peak=1.0），无需除以 alive，没有
"marginal/multiset 不一致"的拼装。dead opps 不贡献 loss。

训练循环（`training/pipeline.py`）：每 PV step 跑 `belief.training.
steps_per_pv_step` 次 belief mini-batch（默认 1）；独立 cosine LR
schedule（默认 `lr_max=3e-4 → lr_min=3e-5`）；独立 replay buffer
`b_buf=200000`。日志里 `b_loss` 是此项 CE。完全 uniform 起点下：
- alive=2 分支理论上限 `log(15) ≈ 2.71`
- alive=1 分支理论上限 `log(5)  ≈ 1.61`

---

## 8. Config 参考

`games/coup/config/belief.json`（独立文件，不写在 train.json 顶层）：

```json
{
  "architecture": [256, 256, 128],
  "activation": "relu",
  "batch_norm": true,
  "training": {
    "lr_schedule": {"type": "cosine", "lr_max": 3e-4, "lr_min": 3e-5},
    "batch_size": 1024,
    "steps_per_pv_step": 1,
    "loss": "ce"
  }
}
```

Belief temperature 不写入配置，框架固定为 1.0。

详见 [CONFIG_REFERENCE belief.json](../../docs/guide/CONFIG_REFERENCE.md#beliefjson可选仅启用网络化-belief-的游戏)。

---

## 9. 文件索引

| 文件 | 内容 |
|---|---|
| `coup_net_adapter.h:85-170` | tracker 字段 + `promote_post_to_pre` |
| `coup_net_adapter.h:171-220` | `CoupBeliefFeatureExtractor` 接口 |
| `coup_net_adapter.cpp:264-393` | `observe_public_event` 状态机 |
| `coup_net_adapter.cpp:411-646` | `randomize_unseen` multiset categorical 采样 |
| `coup_net_adapter.cpp:649-700` | `prepare_for_root`（softmax + cache） |
| `coup_net_adapter.cpp:741-840` | `extract`（28×(N-1)+6 维 features） |
| `coup_net_adapter.cpp:850-908` | `CoupBeliefLabelExtractor::extract` |
| `coup_register.cpp` | GameBundle 三字段注册 |
| `model/coup_belief_2p.onnx` 等 | ship 出去的 ONNX |

新人接手时，按 §3 → §4 → §5 → §6 → §7 顺序读这些文件即可。
