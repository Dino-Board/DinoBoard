# 外部 Ground Truth — 6 nimmt! REST API 端到端验证

## 目的

把 DinoBoard 的 AI 通过 REST API 接入一个**别人写的** 6 nimmt! 评测框架，端到端验证：

1. **接入协议真能跑**：`POST /ai/sessions`（带 `initial_observation`）+ `/observe`（per-action 喂 `events` + `public_snapshot`）+ `/decide` 这一整套，对一个 6 nimmt! 这种回合并发提交、有 small-phase 选择的游戏，是不是真的没漏 corner case。
2. **强度符合预期**：让 DinoBoard AI 跟同目录里编译好的学院级别 reference player（`external_ai.out`，LKF 课程作业里成绩较好的版本）以及 baseline（`easiest.out`）对打，看胜率。

## 文件清单

| 文件 | 说明 |
|------|------|
| `judge.cpp` | 6 nimmt! 课程作业里那份 reference judge 源码。**仅作协议参考**，runner 不直接调用它 |
| `external_ai.cpp` | LKF 课程作业里的 reference player（基于规则 + 简单搜索）。runner 把它当作非 DinoBoard 的对手玩家 |
| `easiest.cpp` | 最简单的 baseline 玩家（出最小牌） |
| `runner.py` | 我们写的 orchestrator：用 `engine.GameSession` 当 ground truth，按 judge 协议 fork 外部 AI，按 REST API 调 DinoBoard AI |

## 架构

```
+---------------------+        +-----------------------+        +-------------------+
| runner.py           | -GT->  | engine.GameSession    | <----  | uvicorn AI service |
| (Python orchestr.)  |        | (DinoBoard 内 truth)   |        | (REST endpoints)   |
+---------------------+        +-----------------------+        +-------------------+
   |  per-move fork
   v
   ./external_ai.out / ./easiest.out  (短命子进程, judge.cpp 协议 stdin/stdout)
```

为什么 runner 直接当 truth、不用原 `judge.cpp`：

- 原 judge **每一手 fork 一次玩家**（短命进程）——这正合外部 AI 的胃口（`external_ai.cpp` 读完 stdin 一次就 exit）。
- 但完全不适合 DinoBoard：我们的 AI 要在跨 ply 之间维护 belief tracker / session state，不能每次重启。如果硬塞进 judge 流程，会变成「每次启动 → 重建 session → 重放整个历史 → decide → 退出」，O(N²) 通信 + 单独一份「重放历史」逻辑，慢且脆弱。
- 拆开看：把 `engine.GameSession` 自己当真值，外部 AI 作为短命子进程沿用 judge 的 stdin 协议，DinoBoard AI 走标准 REST，结构对称、谁都不用妥协。

## 准备

### 1. 编译外部 AI

```bash
cd games/6nimmit/external_ground_truth
g++ -O2 -std=c++14 -o external_ai.out external_ai.cpp   # 旧代码用了 random_shuffle，需要 C++14
g++ -O2 -std=c++17 -o easiest.out easiest.cpp
```

### 2. 启动 DinoBoard AI 服务

```bash
# 仓库根目录
uvicorn platform.web.app:app --port 8000 &
```

`runner.py` 会读环境变量 `DINOBOARD_AI_API`，默认 `http://127.0.0.1:8000`。

## 用法

### 2 人对局：我方 seat 0 vs 外部 AI seat 1

```bash
python3 runner.py --num-players 2 --me 0 --external ./external_ai.out --games 5
```

### 4 人混战：我方 seat 0，其余三家用外部 AI（同一个可执行文件）

```bash
python3 runner.py --num-players 4 --me 0 --external ./external_ai.out --games 5
```

### 4 人混合外部 AI（不同对手）

```bash
python3 runner.py --num-players 4 --me 0 \
    --external ./external_ai.out,./external_ai.out,./easiest.out --games 5
```

`--external` 是逗号分隔的可执行文件列表，给非 DinoBoard 的座位用。如果短于需要的座位数，会循环复用最后一个。

> **目前只支持 1 个 DinoBoard 座位**：`apply_action_with_trace` 是单 perspective 的（调一次会推进 truth），所以一个 ply 只能给一个 AI session 喂 trace。要支持多座位需要单独暴露一个 "snapshot-only" binding，眼下没必要。

### 选项

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num-players` | 2 | 玩家数，2 / 3 / 4 |
| `--me` | `0` | 我方座位（DinoBoard）。当前必须是单座位 |
| `--external` | `./external_ai.out` | 外部 AI 可执行文件，逗号分隔 |
| `--games` | 1 | 游戏局数 |
| `--base` | `http://127.0.0.1:8000` | DinoBoard AI 服务 URL |
| `--seed` | 0 | 0 = 用 `secrets.SystemRandom()`；非 0 = `seed + game_index` |
| `-v, --verbose` | off | 打印每个 ply 的 actor / action_id |

## 输出

每场结束打印：

```
Game 1/5 seed=12345 bullheads=[12, 28] winner=seat#0
  DinoBoard seat#0 decide avg 215.3 ms
  External seat#1 decide avg 8.2 ms

=== Final ===
  seat#0 (DinoBoard): 4 wins
  seat#1 (External): 1 wins
```

- `bullheads`：终局每人累计牛头数（**越少越好**，DinoBoard 的胜负判定）
- `winner`：胜者 seat（`engine.GameSession.winner`）
- `decide avg`：每方平均 decide 用时（毫秒）

## 协议参考

外部 AI 的 stdin/stdout 协议见 `judge.cpp::run_program`。`runner.py::ExternalAi._build_input` 是它的 Python 实现，按以下顺序写入 stdin：

```
seat round
hand_card_1 hand_card_2 ... (sorted)
num_players num_cards
initial_table_row_0  (5 ints, 0=空)
initial_table_row_1
initial_table_row_2
initial_table_row_3
[for each completed round:]
  plays_this_round  (num_players ints, 玩家 seat 顺序)
  table_after_row_0
  table_after_row_1
  table_after_row_2
  table_after_row_3
[if -small mode:]
  this_round_plays  (num_players ints)
scores  (num_players ints, 负的累计 bullhead)
```

stdout：单个 int。常规出牌阶段是要打的牌号 1..104；`-small` 模式下是要踩的行号 0..3。超时 5 秒。

## 这个 harness 测了什么 / 没测什么

✅ **测**：
- REST 协议在 6 nimmt! 这种回合并发 + small-phase 的复杂状态机里能走通
- DinoBoard AI 对一个看得见的、独立设计的对手能赢多少（强度回归）
- belief tracker 跨 ply 维护、`public_snapshot` 整体覆写、`events` 流是否齐全

❌ **不测**：
- 牛头王评测框架的对局规则（runner 用 DinoBoard 自己的 `engine.GameSession`，规则一致性由 `tests/6nimmit/test_checklist.py` 保证）
- DinoBoard 内部的搜索 / 训练正确性（那些有专门的 framework 测试）

如果 `runner.py` 跑挂了但 `tests/6nimmit/test_checklist.py` 通过，问题大概率在 REST 适配层（事件提取、初始观察、snapshot），而不是规则。
