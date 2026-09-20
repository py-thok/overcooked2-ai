# 真实游戏 RL 管线 — 现状与操作手册

> 最后更新：2026-09-19。本文档记录 BepInEx 状态桥路线的实施现状。
> 视觉识别路线已废弃（归档分支 `archive/vision-pipeline`）。

## 目标

- **主目标**：寿司关（s_sushi_1_4，战役 1-4）RL 达到极高分（参照排行榜：世界纪录 2136，前五 ~1900）
- **次目标**：观测/动作表示布局无关，为其他关卡泛化打底

## 架构

```
Overcooked2.exe (Unity 2017.4, Mono, x86)
 └─ BepInEx 5.4.23.2 (win_x86)
     └─ plugins/OC2StateBridge.dll
         ├─ StateExtractor    每帧(主线程)采集 ground-truth 状态 → JSON
         ├─ StateServer       TCP 127.0.0.1:8765（换行分隔协议）
         ├─ InputInjector     移动轴/按钮注入（drive 模式）或只读回显（sniff 模式）
         ├─ SemanticActions   语义动作：直接调 ChefEventMessage(PickUp/Place/Throw)
         ├─ EnvControl        RESET / TIMESCALE / LOADLEVEL / SETPOS
         └─ Discovery         F9 对象普查 → <游戏目录>/oc2bridge_discovery.log

Python (src/bridge/)
 ├─ state_client.py    TCP 客户端（所有命令的 Python 封装）
 ├─ encoder.py         状态 JSON → 14通道网格张量 + 全局向量（布局无关）
 ├─ env.py             OC2Env：gym 风格 reset/step，shaping 奖励
 ├─ record_demos.py    人类示范录制（sniff 模式）
 ├─ bc_train.py        行为克隆（CNN 双头：移动9类 + 按钮4类）
 ├─ ppo_train.py       PPO 自我对弈微调（GAE 按 agent 拆分、全指标 TB）
 └─ panel.py           BetterGI 风格悬浮控制面板
```

## 协议速查（127.0.0.1:8765，换行分隔）

| 命令 | 说明 |
|---|---|
| `GET` | → 一行状态 JSON（玩家/锅/订单/物品/计时/分数/网格/输入回显） |
| `STATIONS` | → 当前关卡静态站点普查（2s 缓存） |
| `ACTION {"player":0,"move":[x,y],"pickup":bool,"use":bool,"dash":bool}` | 原始输入注入 |
| `INTERACT <p>` | 语义拾取/放下（游戏自身交互事件，无竞态） |
| `THROW <p>` | 投掷手持物 |
| `RESET` | 回合中途立即重开（暂停菜单同款路径） |
| `TIMESCALE <0.25-8>` | 游戏加速（LateUpdate 钉住，跨重开保持） |
| `LOADLEVEL <scene>` | 加载关卡（无会话时经 WorldMap 中转） |
| `SETPOS <p> <x> <y> <z>` | 传送玩家 |
| `MODE drive\|sniff` | AI 控制 / 人类控制+回显 |
| `PING` | 探活 |

关键事实（源码实证）：移动输入**直接映射世界轴** `world_dir = (signX·x, -signY·y)`，
`sign` 由每关 `MovementData` 反转标志决定并经状态 `move_sign` 字段下发。**不是相机相对，无需标定。**

## 已验证（实测通过）

- [x] 状态提取：双玩家 pos/grid/fwd/held、3 锅 progress/state/contents、订单 recipe+remaining、分数、计时
- [x] 移动注入：闭环误差 <5°（开阔处）
- [x] RESET：2 秒内重开、计时归零、TIMESCALE 跨重开保持
- [x]  sniff 录制：3 局有效（002/003/004，~7000 帧，分数 196/261/198）
- [x]  BC：`checkpoints/bc_sushi.pt`（按钮时序好，移动方向弱 — 意图不可观测所致，正常）
- [x]  PPO 管线：rollout→GAE→更新全链路，~23-26 transitions/s @ 3x
- [x]  LOADLEVEL：大地图/关卡内可用；主菜单需经 WorldMap 中转（见未决问题 2）

## 未决问题（按优先级）

### 1. ✅ 已修复：drive 模式按钮失效（2026-09-20）

根因：`ILogicalButton.JustPressed()` 单次消费语义，按键事件被读取链上的
第一个读者消费。修复：`SemanticActions.Interact/ThrowItem` 绕过按钮层，直接发
`ChefEventMessage(PickUp/Place/Throw)`，目标由游戏自身邻近扫描解析。
实测：米箱拾取✓ 砧板放置✓ 切菜(use-hold 原始注入)✓ 入锅 shaping 奖励✓。

### 2. ✅ 已修复：主菜单直进关卡黑屏（2026-09-20）

根因有两层：(a) StartScreen 等待"按任意键"engagement，(b) 无会话时跳过
`SetupCoopSession` 导致厨师不分配。修复：新增 `ENGAGE` 命令（程序化
`StartGameownerEngagement` + 反射调 `StartScreenFlow.OnEngagementFinished`），
LOADLEVEL 无会话时自动复刻故事模式流程（`StartEmptySession(-1)` +
`SetupCoopSession` + `ServerGameSetup.Mode=Campaign`）。
实测：全新启动 → ENGAGE → LOADLEVEL → 双厨师生成，全程无人工。

### 3. 🔴 500k 步零分（2026-09-20 run6 结果，已加修复待验证）

run6 完整跑完 500k 步 / 318 回合 / 全部零分，锅事件仅 6/489 rollout。根因分析：

1. **拾取延迟自抵消**：语义拾取结果 ~0.6s 后才进状态（网络往返），策略 67ms/步，
   拿起米后自己不知道，继续按 pickup（每步 25% 概率）→ 9 步内又放下。
   → 修复：INTERACT 加 0.7s 墙钟冷却。
2. **策略极少走到箱子**（BC 移动头坍缩 + 稀疏奖励无法引导导航）。
   → 修复：课程式出生点——每回合 P0 放在米箱 (8.4,-2.4) 前、P1 放在砧板前。
3. 米箱位置实测：黄瓜箱=(13.2,-2.4)，**米箱=(8.4,-2.4)**。

另修：回合边界检测（timer 钉 0=结算画面、timer 消失/跳变=跨回合，3 次重读确认防闪烁）；
跨回合比分归零不再产生伪负奖励；结算画面死时间跳过（样本效率 +25%）。

### 4. 🟢 小项

- `push` 偶发 SSL 失败（代理问题），本地提交不受影响
- 面板浮窗与游戏并行时曾误判（浮窗只读，非原因）
- BC 移动头坍缩为"站稳"（意图不可观测 + 类不平衡），PPO 负责用奖励补

## 训练操作

```bash
# 1. 启动游戏（BepInEx 自动加载插件），手动点一次"故事模式"
# 2. 正式训练（3 倍速，50 万步 ≈ 5-6 小时）
python src/bridge/ppo_train.py --timescale 3 --total-steps 500000

# 续训
python src/bridge/ppo_train.py --resume checkpoints/ppo_sushi_20480.pt

# 监控
tensorboard --logdir logs/ppo_sushi --port 6007   # http://localhost:6007
python src/bridge/panel.py                        # 悬浮面板（可选）

# 录制更多示范（可提升 BC 起点）
python src/bridge/record_demos.py --rounds 3 --start 7 --out demos/sushi
```

## 数据与产物

| 路径 | 内容 |
|---|---|
| `demos/sushi/` | 人类示范（round_002/003/004 有效；不入库） |
| `demos/sushi/stations.json` | 寿司关站点普查（编码器依赖） |
| `checkpoints/bc_sushi.pt` | BC 初始权重 |
| `checkpoints/ppo_sushi_*.pt` | PPO 存档（含 `_final.pt`） |
| `logs/ppo_sushi/` | TensorBoard 事件 |
| `work/decomp/AssemblyCSharp/` | 游戏反编译代码（2365 类，不入库） |

## 目标分数参照（overcooked.greeny.dev 排行榜，1-4 单人）

| 分数 | 水平 |
|---|---|
| 2136 | 世界纪录 |
| 1952~2044 | 前三 |
| 1860 | 前五 |
| ~1300 | 中上 |
| 196~261 | 当前人类示范水平 |

## 路线图

1. ~~修按钮~~ ✅ / ~~全自动进关~~ ✅ / ~~回合边界~~ ✅ → **run7 训练中**（冷却+课程出生点）← 当前位置
2. 寿司关冲分（若 run7 仍零分：检查 BC 移动头质量、加距离势 shaping、考虑 1x 减速训练）
3. 泛化阶段：模拟器网格观测对齐 + A100 CNN 预训练 → 权重迁移
