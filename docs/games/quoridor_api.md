# Quoridor — AI API 接入文档

## 概览

- **game_id**：`quoridor`
- **玩家数**：固定 2
- **动作空间**：209
- **隐藏信息**：无
- **公开事件**：无

---

## 动作编码

动作空间分成三段：

| 范围 | 类型 | 说明 |
|------|------|------|
| 0–80 | 移动 | 移动到指定格子（9×9 棋盘，行优先索引） |
| 81–144 | 水平墙 | 在 (row, col) 放水平墙，8×8 墙槽 = 64 个位置 |
| 145–208 | 垂直墙 | 在 (row, col) 放垂直墙，8×8 墙槽 |

### 移动

目标格子 id：`action_id` 直接对应 9×9 棋盘的线性索引。

```python
target_row = action_id // 9
target_col = action_id % 9
```

合法移动包括：四方向单步、跳过对手棋子（如果对手相邻且跳过后不越界）、侧跳（被墙挡住时）。框架计算 `legal_actions`，第三方只需相信它。

### 墙

水平墙：起始 offset `kActionHWallStart = 81`

```python
h_wall_id = action_id - 81   # 0..63
wall_row = h_wall_id // 8    # 0..7
wall_col = h_wall_id % 8     # 0..7
```

水平墙占据 `(wall_row, wall_col)` 和 `(wall_row, wall_col+1)` 两格的边界，阻挡两行之间的纵向通行。

垂直墙：起始 offset `kActionVWallStart = 145`

```python
v_wall_id = action_id - 145
wall_row = v_wall_id // 8
wall_col = v_wall_id % 8
```

垂直墙占据 `(wall_row, wall_col)` 和 `(wall_row+1, wall_col)` 两格的边界。

**合法性约束**：墙不能与已有墙重叠或交叉，墙不能完全堵死任一玩家通往自己终点的路径（Quoridor 规则）。框架 `legal_actions` 已计算好。

---

## 多人变体

无。Quoridor 只有 2 人版。

---

## state_dict 字段

`state_serializer` 输出（`games/quoridor/quoridor_register.cpp::serialize_quoridor`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_player` | int | 0 或 1 |
| `is_terminal` | bool | 是否终局 |
| `winner` | int | 胜者；`-1` 还没结束 |
| `move_count` | int | 累计步数 |
| `board_size` | int | 棋盘边长（固定 9） |
| `scores` | `int[2]` | 胜者 +1 败者 -1；未结束都是 `[0, 0]` |
| `pawns` | `[{player, row, col}, ...]` | 长度 2，每项 0..8 |
| `walls_remaining` | `int[2]` | 每人开局 10 堵 |
| `horizontal_walls` / `vertical_walls` | `[{row, col}, ...]` | 已放置的墙（8×8 槽位） |
| `goal_rows` | `int[2]` | 玩家 0 / 1 的终点行（默认 `[8, 0]`） |

---

## 完整示例（Python）

```python
import requests
BASE = "http://localhost:8000"

# AI 扮演玩家 1（先让对手走）
sess = requests.post(f"{BASE}/ai/sessions", json={
    "game_id": "quoridor", "seed": 123, "my_seat": 1,
    # 注:body 里的 simulations/temperature 会被 pydantic 丢弃,
    # 强度由服务端从 web.json::web_expert 解析(见 AI_API.md)
}).json()
sid = sess["session_id"]

while True:
    status = requests.get(f"{BASE}/ai/sessions/{sid}").json()
    if status["is_terminal"]:
        print(f"Game over. Winner: {status.get('winner')}")
        break

    if status["current_player"] == 1:
        # AI 回合
        r = requests.post(f"{BASE}/ai/sessions/{sid}/decide").json()
        action_id = r["action_id"]
        info = r["action_info"]
        if info["type"] == "move":
            print(f"AI 移到 ({info['row']},{info['col']})")
        else:
            print(f"AI 放{'水平' if info['type']=='hwall' else '垂直'}墙 at ({info['row']},{info['col']})")
    else:
        # 对手输入
        action_id = int(input("Your move (0-208): "))

    requests.post(f"{BASE}/ai/sessions/{sid}/observe", json={
        "action_id": action_id, "events": [], "public_snapshot": {},
    })

requests.delete(f"{BASE}/ai/sessions/{sid}")
```

---

## 可选特性：Tail Solver

Quoridor 注册了 `tail_solver`（α-β 残局求解）。AI 在局末（通常 ply ≥ 30）会尝试精确求解必胜路线。接入方不需要额外操作——`decide` 返回的 `stats.tail_solved=true` 表示本步是 tail solve 的结果（已证明胜/负，而非 MCTS 估计）。

---

## 常见踩坑

- **墙的坐标系容易混**：水平墙编号 `wall_row` 指墙"下面一行"的 row index。对照 legal_actions + action_info 反推比手算安全
- **分支因子大**：Quoridor 每步合法动作常在 40-100 之间，比 TicTacToe 大很多。simulations 建议 ≥ 600 才能玩出对局感
