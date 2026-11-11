# DinoBoard — 功能概览

给开发者的速查地图。算法见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 核心组件

一个完整的游戏 = `state.cpp` + `rules.cpp` + `net_adapter.cpp` +
`register.cpp` + 一组 `config/*.json` + `web/`。

| 文件 | 接口 | 职责 |
|------|-----|------|
| `<game>_state.cpp` | `IGameState` | 状态 + visibility schema（hash / encoder / snapshot 的单一事实源） |
| `<game>_rules.cpp` | `IGameRules` | 合法动作、`do_action_fast`（含 viz 维护） |
| `<game>_net_adapter.cpp` | `IFeatureEncoder` + `IBeliefTracker` | 特征编码（入参锁 masked_state）+ belief tracker（隐藏信息游戏） |
| `<game>_register.cpp` | `GameBundle` | 组件打包注册 |
| `config/*.json` | — | game / mcts / train / web / heuristic / 可选 belief |
| `web/<game>.js` | `createApp(...)` | 玩家交互界面 |

---

## 架构：GT 推进 + AI session 各自独立

- **GT 端**持真值 state，每步产出公开消息流（events + public_snapshot）
- **AI session**（每 perspective 一份）只吃消息流，public 字段整张覆盖，hidden 槽位不维护
- **MCTS / encoder** 在 selfplay / web / API 三条路径下都吃 AI session state，结构性不读真值

详见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 搜索：ISMCTS over DAG

神经网络引导 MCTS，原生 2-4 人。Root 采样 determinization，DAG 共享节点，UCT2，step counter 防环。没有 chance node——物理随机统一由 root determinization 处理。

隐藏信息 + bluff 游戏需要额外机制（"四板斧"）：

| 机制 | 解决什么 | 哪些游戏用 |
|------|---------|-----------|
| 信念网络 | uniform 采样退化（bluff 信号被一抹平） | Coup, Liar's Dice |
| 冻结对手 | 对手全知化（sim 内 opp 太强） | Coup, 6 nimmt!, Liar's Dice |
| 对手池 | mirror selfplay 策略坍缩 | Coup |
| 世界容量 | 隐藏空间大导致 bandit 深度=0 | 6 nimmt!, Liar's Dice |

算法详解见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 现有游戏

| 游戏 | 类型 | 特点 |
|------|------|------|
| TicTacToe | 完全信息 | 最小闭环参考 |
| Quoridor | 完全信息 | tail solver + 启发式全配齐 |
| Azul | 对称随机 | 无 tracker，物理随机由 sim_rng 在 do_action_fast 里即时处理 |
| Splendor | 对称随机 + 轻量隐藏 | 最简 randomize_unseen（均匀采样暗压牌） |
| Love Letter | 非对称隐藏 | viz reveal 范例——规则多写 reveal_slot_to 可显著提升 AI 强度 |
| Coup | bluff | 四板斧范例（信念网络 + 冻结对手 + 对手池） |
| 6 nimmt! | 同时行动 | 同时行动范式 + 世界容量 |
| Liar's Dice | bluff | 四板斧范例（信念网络 + 世界容量） |
| Ark Nova | 随机 + 隐藏手牌 | 分解式动作 + 变长 token 编码 + 公开动作投影（给 belief 用） |

---

## 训练

`selfplay → 收集样本 → SGD → ONNX export → gating eval → 循环`

全流程 C++（selfplay / 搜索 / eval），Python 只做训练循环和 SGD。配置驱动，不改代码调参。

Value head N 维 perspective-relative（`values[0]`=当前玩家）。网络可选 MLP 或实体 token 的 set transformer；KataGo 式可选项（playout cap、FPU、分差/对手策略辅助头、EMA）和吞吐调优见 `/training` 技能。Selfplay 用叶子批处理 + 多线程批量推理，可跨多张 GPU（`--devices`）；启发式引导可按步数衰减或按胜率提前退出。

---

## Web 前端

每个游戏必须有可玩的 web 前端——这是验证 AI 行为的最终标尺。

设计原则：自然交互（不要每动作一按钮）、空间锚定、动作动画、视觉可辨。

---

## AI API

`platform/ai_service/` 提供 observation-only REST API。调用方只传 action_id + 公开事件，session 内部自维护 state。每游戏的动作编码见 `docs/games/<game>_api.md`。

---

## 框架局限性

### 范式硬冲突（不可解）

1. **精确混合均衡**（德州扑克）— AlphaZero 训确定性策略，无法收敛精确 Nash。近似混合（Coup / Liar's Dice 的 bluff）在框架范围内
2. **动作空间爆炸**（斗地主 27k+、万智牌 C(N,K)）— 固定 action_space 不适用，需分层/自回归解码
3. **合作博弈**（Hanabi）— self-play 会收敛到只对训练自己有效的私有约定，zero-shot coordination 问题
4. **单人游戏**（纸牌接龙、2048）— 无对手 → 无胜负信号 → 训练起不来
5. **闭眼环节**（狼人杀夜晚）— viz 是 god-view 函数，不支持"不知道对手是否获得了信息"的高阶不确定性
