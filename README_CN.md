# DinoBoard

**English version:** [README.md](README.md)

> **Drop a rulebook. Ship a superhuman AI.**
>
> 给 LLM 一句话「加上 Azul」，它读规则书、写 C++ 规则引擎、跑全量测试、
> 训练出 AI、生成 Web 前端。你只负责验收。

一个面向**通用桌游**的 AlphaZero 风格框架，把"规则引擎 + AI 决策 + Web 对战"做成端到端闭环，在 LLM 协助开发下保持可信。

---

## 核心机制

三条机制让规则引擎和 AI 决策路径在 vibe coding 下也对人类玩家可信：

1. **字段级可见性标记** — 每个游戏声明每个槽位"谁能看"。框架统一遍历，产出 snapshot / hash / encoder 三家共用的 masked 观测视图。隐藏槽位结构性读出 placeholder，真值无法泄漏。

2. **GT/AI 物理分离** — 真值端推进游戏，每个玩家的 AI session 只吃消息流。AI 没有接口获取真值指针——结构性安全，非约定。

3. **强制 Web 前端** — 训练指标可能看着正常但 AI 在关键时刻送子。人类玩 30 秒就能发现。

算法详解见 [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md)。

---

## 游戏

| 游戏 | 类型 | 2p | 3p | 4p |
|------|------|----|----|-----|
| TicTacToe | 完全信息 | 强/✓ | — | — |
| Quoridor | 完全信息 | 强/✓ | — | — |
| Azul | 随机 | 中等/✓ | 中等/✗ | 中等/✗ |
| Splendor | 随机 + 轻量隐藏信息 | 中等/✓ | 中等/✗ | 中等/✗ |
| Love Letter | 随机 + 隐藏信息 | 随机/✓ | 随机/✓ | 随机/✓ |
| Coup | 诈唬 | 随机/✓ | 随机/✗ | 随机/✓ |
| 6 nimmt! | 同时行动 + 隐藏信息 | 强/✓ | 强/✗ | 强/✗ |
| Liar's Dice | 诈唬 | 中等/✓ | 中等/✓ | 中等/✓ |
| Harmonies | 随机 | 中等/✗ | 中等/✗ | 中等/✗ |
| Jaipur | 随机 + 隐藏信息 | 中等/✓ | — | — |
| Carcassonne | 随机 + 隐藏信息 | 中等/✗ | — | — |
| Maskmen | 隐藏信息 | — | 随机/✓ | 随机/✗ |
| Ark Nova | 随机 + 隐藏信息 | 未评估/✗ | — | — |

> 格式：**启发式强度/网络**。✓ = 已训练网络，✗ = 无训练网络，— = 不支持该变体。
>
> Token transformer 网络：Ark Nova 默认使用；Splendor、Carcassonne、Harmonies 通过 `games/<game>/config/train_transformer*.json` 启用。
> 这四个游戏的 2 人网页对局使用 transformer 变体（`game.json` 的 `web_variants`）：Splendor 和 Carcassonne 是训练好的模型，Harmonies 和 Ark Nova 是随机初始化的模型。

---

## 快速开始

### 依赖

- **C++17 编译器** — Mac: `xcode-select --install`；Linux: `apt install build-essential`
- **Python >= 3.9**，`pybind11` + `torch`
- **ONNX Runtime** — Mac: `brew install onnxruntime`；Linux/Windows: 仓库内自带或从 [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) 下载

### 编译

```bash
python3 -m pip install -e .
python3 -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

### 训练

```bash
GAME=splendor
RUN_DIR=runs/${GAME}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

# CPU（--workers 按核心数调整）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python3 -m training.cli --game "$GAME" --output "$RUN_DIR" \
    --workers 4 > "$RUN_DIR/train.log" 2>&1 &

# GPU/CoreML：加 --providers cuda,cpu 或 --providers coreml,cpu
# Token transformer 多卡训练：--game splendor_tok \
#   --train-config games/splendor/config/train_transformer_kata_guided.json \
#   --providers cuda,cpu --devices 0,1
```

### Web 对战

```bash
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

### 在 ark-nova-ai 的界面上玩 Ark Nova

DinoBoard 没有自己的 Ark Nova 页面。`scripts/arknova_board.py` 启动
[ark-nova-ai](https://github.com/collinfeng/ark-nova-ai)（MIT，需单独 clone）的服务器，只提供 Map 0
和两个对手：`dinoboard`（DinoBoard 的 token transformer + MCTS，默认用
`games/arknova/model/arknova_2p.onnx`，目前是随机初始化的网络，`--model` 可换）和 ark-nova-ai 的 `beam_3x8` 搜索。

```bash
python scripts/arknova_board.py --ref ~/ref/ark-nova-ai   # [--model m.onnx] [--profile web_casual]
# 页面：cd ~/ref/ark-nova-ai/web && pnpm install && pnpm build，然后打开 http://127.0.0.1:8765
```

### AI API

```
POST /ai/sessions                  → 创建 session
POST /ai/sessions/{id}/observe     → 喂公开事件
POST /ai/sessions/{id}/decide      → 获取 AI 动作
DELETE /ai/sessions/{id}           → 结束 session
```

调用方只需 action ID + 公开事件。

---

## 加新游戏

1. 镜像最近的现有游戏实现
2. 复制 `tests/<最近的>/test_checklist.py`，改 `GAME = "..."`
3. 迭代测试 — `pytest tests/<新游戏>/` 全绿 = 完成
4. `python -m training.cli --game <id>` 启动训练
5. Web UI 验证

---

## 框架局限性

基于 AlphaZero / ISMCTS 范式，以下不适用：

1. **精确混合均衡**（德州扑克）— 近似混合（Coup / Liar's Dice 的 bluff）在框架范围内
2. **动作空间爆炸**（斗地主 27k+、万智牌构筑）
3. **合作博弈**（Hanabi）— self-play 会收敛到私有约定
4. **单人游戏**（纸牌接龙、2048）— 无对手 = 无训练信号
5. **闭眼环节**（狼人杀夜晚）— viz 是 god-view 函数

---

## 文档

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) — 功能概览
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) — 算法介绍
- [docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md) — 设计动机
- `docs/games/<game>_api.md` — 各游戏 API 集成文档
