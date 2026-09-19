# OC2StateBridge — BepInEx 游戏内状态桥

替代原视觉感知方案。直接在 Overcooked! 2 进程内运行，读取游戏对象的真实状态
（玩家/锅/订单/计时/分数/网格），通过本地 TCP 以 JSON 提供给 Python agent，
并接收动作指令注入玩家输入。

## 架构

```
Overcooked2.exe (Unity 2017.4, Mono, x86)
 └─ BepInEx 5.4.23.2 (win_x86)
     └─ plugins/OC2StateBridge.dll
         ├─ StateServer      TCP 127.0.0.1:8765（后台线程，只发快照字符串）
         ├─ StateExtractor   主线程每帧采集 → JSON（玩家/锅/订单/计时/分数/网格）
         ├─ InputInjector    替换 ControlSchemeData 的输入槽位，驱动厨师
         └─ Discovery        F9 导出对象普查 → <游戏目录>/oc2bridge_discovery.log

Python: src/bridge/state_client.py
         ├─ GET    → 一行 JSON 状态
         └─ ACTION → 移动/拾取/切菜/冲刺
```

## 协议（换行分隔文本）

| 方向 | 内容 | 说明 |
|---|---|---|
| C→S | `GET` | S→C 一行 JSON 状态 |
| C→S | `ACTION {"player":0,"move":[x,y],"pickup":true,"use":false,"dash":false}` | S→C `OK`；缺失的键保持不变；按钮为按住语义 |
| C→S | `PING` | S→C `PONG` |

## 状态 JSON 字段

- `scene` / `in_round` — 当前场景 / 是否在关卡中
- `round` — `time_elapsed` / `time_limit` / `time_remaining` / `score` / `tips`
- `grid` — `active` / `half_size`（游戏原生 GridManager，比视觉标定精确）
- `players[]` — `pos` / `fwd` / `grid`（世界坐标 + 网格坐标）/ `player_id` / `held{name,is_plate,contents[]}`
- `cookers[]` — `grid` / `station_type` / `progress` / `cook_time` / `state` / `is_cooked` / `is_burning` / `contents[]`
- `orders[]` — `id` / `recipe` / `remaining` / `lifetime`

## 构建与部署

```bash
# 游戏路径可通过 OC2Dir 覆盖（默认 E:\SteamLibrary\...\Overcooked! 2）
dotnet build mod/OC2StateBridge
# 若游戏目录已装 BepInEx，构建后自动复制到 BepInEx\plugins\
```

游戏侧 BepInEx 安装（已完成一次，无需重复）：解压 `BepInEx_win_x86_5.4.23.2.zip`
到游戏根目录（Steam 版 OC2 是 x86；Epic 版需 x64 包）。

## 使用方法

1. 启动游戏（插件自动加载，见 `BepInEx/LogOutput.log` 中 `OC2 State Bridge`）
2. 进入一个关卡（任意厨房）
3. `python src/bridge/state_client.py` —— 4 Hz 打印状态摘要
4. 游戏内按 **F9** 导出 discovery 普查，验证字段假设
5. `python src/bridge/state_client.py --drive` —— 随机游走演示（驱动玩家 0）

## 数据来源对照（反编译验证）

| 数据 | 游戏类/API |
|---|---|
| 玩家位置/朝向 | `tag=="Player"` → `transform` |
| 玩家手持物 | `IPlayerCarrier.InspectCarriedItem()` |
| 网格坐标 | `GridManager.GetActive(0).GetGridLocationFromPos()` |
| 锅状态 | `ServerCookingHandler.GetCookingHandlers()` → 进度/时长/状态 |
| 内容物 | `IIngredientContents.GetContents()` → `AssembledDefinitionNode` 树 |
| 订单 | `ServerTeamMonitor.OrdersController` → `m_activeOrders`（反射） |
| 分数 | `ServerTeamMonitor.Score` |
| 计时 | `ServerKitchenFlowControllerBase.RoundTimer` |
| 输入注入 | `PlayerControls.ControlScheme` 公有字段替换 `ILogicalValue`/`ILogicalButton` |

## 已知边界

- 移动轴 (move_x, move_y) 是游戏输入轴，方向映射可能随相机角度变化，需逐关标定
- `player` 槽位按角色名排序分配，与游戏内 chef 编号的对应关系以 discovery 日志为准
- 联机模式未测试（单机本地服务器架构已验证）
