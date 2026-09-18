# 图像识别 → 96 维向量：技术方案详设

## 0. 关键先行结论（来自对训练向量的实测）

对开局状态 `featurize_state_mdp` 的 96 维输出做统计分析：

- **取值域极小**：所有维度 ∈ {-2, -1, 0, 1, 2}（ cramped_room 是 5×4 网格，坐标差最多 ±3，A* 代价最多个位数；one-hot 是 0/1；cook_time 是 0-20 整数但开局为 -1/0）
- **稀疏**：开局 96 维里只有 27 维非零
- **语义离散**：没有需要"连续回归"的维度。所有量要么是 one-hot 分类，要么是小整数坐标/计数

> **设计含义**：这不是一个"图像 → 连续嵌入"的表示学习问题（不需要训练 CNN encoder），
> 而是一个"图像 → 结构化离散状态 → 查表/计算生成向量"的**感知+符号计算**问题。
> 检测器只需输出正确的**离散类别和小整数**，向量由 feature_builder.py 确定性生成。
> 这决定了我们不用训任何神经网络来做视觉，经典 CV（模板匹配/颜色分割/OCR）就够。

## 1. 96 维规格书（逐段， cramped_room, 2玩家, num_pots=2）

偏移量是累加的（每玩家段 44 维 = 24 + 2×10）：

| 维度 | 名称 | 类型 | 视觉来源 | 检测方法 |
|---|---|---|---|---|
| 0-3 | pi_orientation (N/S/E/W one-hot) | 分类×4 | 角色精灵朝向 | 模板匹配4方向sprite |
| 4-7 | pi_obj (onion/tomato/dish/soup, 空手=全0) | 分类×4 | 角色头顶携带图标 | 模板匹配小图标 |
| 8-11 | pi_wall_{j} (4方向是否贴墙) | 0/1 | 地图+坐标推算 | **查表**（地图固定，坐标已知即可算） |
| 12-13 | closest_onion (dx,dy) | A*代价 | 洋葱堆位置 | 颜色分割+网格定位+A* |
| 14-15 | closest_tomato | A*代价 | 番茄堆（cramped_room无→(0,0)） | 布局固定则无 |
| 16-17 | closest_dish | A*代价 | 盘子堆位置 | 颜色分割+网格定位 |
| 18-19 | closest_soup | A*代价 | 煮好的汤 | 锅状态推导 |
| 20-21 | closest_serving | A*代价 | 上菜口位置 | **查表**（固定） |
| 22-23 | closest_empty_counter | A*代价 | 空台面 | 台面占用检测 |
| 24-25 | soup_n_onions / n_tomatoes | 计数0-3 | 锅气泡图标 | 小图标计数 |
| 26-35 | pot_0: exists/empty/full/cooking/ready/n_onions/n_tomatoes/cook_time/(dx,dy) | 混合 | 锅1 | 见难点2 |
| 36-45 | pot_1: 同上 | 混合 | 锅2 | 见难点2 |
| 46-89 | other_player 的 44 维（重复上述） | — | 队友（另一颜色角色） | 同 player_i |
| 90-91 | dist_to_other (dx,dy) | 整数 | 两角色坐标差 | 坐标相减 |
| 92-95 | pi_position (x,y) 及padding | 整数 | 角色网格坐标 | 颜色分割+网格吸附 |

（注：确切偏移以 `mdp.featurize_state` 的 ordered 输出为准，feature_builder.py 已封装，
检测器只需填 PerceivedState，偏移对齐由 builder 保证——已验证 exact-match。）

## 2. 检测器技术选型（按维度类别）

### A. 网格定位（一切的基础）
OC2 的厨房是固定网格布局，角色/物体都在格子上。
- **标定**：一次性确定截图中厨房网格的原点像素 (ox,oy)、格间距 cell_px、网格行列数。
  cramped_room 5×4，同一分辨率下永远不变。
- **吸附**：检测到的像素质心 (px,py) → `grid = (round((px-ox)/cell_px), round((py-oy)/cell_py))`
- **输出**：所有实体的离散网格坐标。这是 PerceivedState 的主要字段。

### B. 角色检测（玩家坐标+朝向）
- **方法**：颜色分割。chef1/chef2 的主色（如蓝/绿）在 HSV 空间阈值分割 → 连通域 → 质心。
- **朝向**：取角色 sprite 区域，与 4 个朝向模板做 `cv2.matchTemplate`，取最高分。
  - 缓解抖动：对朝向做时序中值滤波（连续3帧一致才更新）。
- **风险**：角色部分被遮挡（手里举锅/菜）时颜色面积变化 → 用最大连通域+面积下限兜底。

### C. 静态物体（洋葱堆/盘子堆/上菜口/台面）
- cramped_room 这些**位置固定** → 大部分可直接**查表硬编码**进 grid_calib。
- 只有"empty_counter"是动态的（台面被占用与否）→ 对每个台面格子做 occupancy 检测
  （该格中心区域颜色是否=空台面底色）。

### D. 锅状态机（难点1）
4 态：empty / filled(加了料没煮) / cooking(煮中,有进度条) / ready(煮好)。
- **状态分类**：对锅格子区域模板匹配 4 个状态 sprite。
- **cook_time（剩余秒数）**：cooking 态时有进度条。进度条**像素长度 → 剩余时间**线性映射。
  - 标定：录一段煮汤全程，逐帧记录进度条像素宽度与真实剩余秒数，拟合斜率。
  - 特征只需整数秒 → 允许 ±1 误差。
- **配料计数 n_onions**：锅上方气泡里的洋葱小图标个数。
  - cramped_room 只做洋葱汤 → 只需数洋葱，0-3 个。
  - 方法：气泡区域做洋葱模板的多尺度 matchTemplate，非极大抑制数峰值个数。
  - **这是最脆弱环节**，图标小且可能重叠。缓解：限定只在 cooking/filled 态检测。

### E. 手持物（pi_obj 4类）
- 角色头顶有小图标表示手持物（洋葱/番茄/盘子/汤）。
- 模板匹配 4 类 + "无"（头顶无图标）。
- 风险：图标紧贴角色头顶，可能被 HUD 遮挡 → 检测窗口取角色格子上方半格。

## 3. A* 距离的重建（难点2，已解决）

`closest_*` 的 (dx,dy) 是 `MotionPlanner.min_cost_to_feature` 的 **A* 路径代价**，
不是直线距离。**必须复用模拟器的规划器**：

- feature_builder.py 已内置 `OvercookedEnv.mlam`（含 MotionPlanner）；
- 检测器只需给出目标的**格子坐标**，builder 调 `featurize_state` 自动算 A* 代价；
- 已验证：随机 rollout 200 步，重建向量与 ground truth **逐位相等**。

> 所以"图像识别"的边界止于"给出每个实体在哪个格子/什么状态"，
> 距离计算完全由 builder 用与训练一致的代码完成，无需在视觉端实现 A*。

## 4. 端到端数据流

```
真实截图 (1280×720)
   │  capture.py (mss 抓屏)
   ▼
网格标定 (一次性, grid_calib)
   │
   ├── 颜色分割 → 角色质心 → 网格吸附 → pi_position, dist_to_other
   ├── 模板匹配 → 朝向 / 手持物 / 锅状态
   ├── 进度条测量 → cook_time
   ├── 图标计数 → n_onions
   └── 查表 → wall / serving / 静态物
   ▼
PerceivedState (离散结构化状态)
   │  feature_builder.build()  ← 复用 MotionPlanner
   ▼
96-dim vector (取值∈小整数集)
   │  MLP.predict()
   ▼
action (0-5) → 按键注入
```

## 5. 验证策略（无 X display 的服务器上先行）

1. **离线对齐（当前可做）**：用模拟器渲染器生成状态图 + SimDetector 输出 PerceivedState，
   验证 builder 与 ground truth 一致（✅ 已通过，exact-match）。
2. **检测器单测（需真实截图）**：在能跑 OC2 的机器上，对每个检测器（角色/朝向/锅/计数）
   采集标注样本，算分类准确率。目标：网格定位 100%，状态分类 >98%，计数 >95%。
3. **端到端一致性**：真实截图 → 检测 → PerceivedState → 向量，与"同局面下模拟器
   重建的向量"比对（需人工对齐局面，或录制视频帧+日志）。

## 6. 风险与缓解汇总

| 风险 | 等级 | 缓解 |
|---|---|---|
| 锅配料计数不准（小图标重叠） | 高 | 只数洋葱(≤3)、多帧投票、限制在 cooking 态检测 |
| cook_time 进度条标定依赖分辨率 | 中 | 标定随分辨率存配置；±1s 误差可接受 |
| 角色遮挡导致定位漂移 | 中 | 时序滤波 + 网格吸附 + 面积兜底 |
| 朝向抖动 | 低 | 连续3帧一致才更新 |
| A* 距离不一致 | ~~高~~ 已解决 | 复用 MotionPlanner，exact-match 验证通过 |
| 光照/画质变化（录制vs实时） | 中 | 模板匹配对亮度敏感 → 用归一化相关系数 TM_CCOEFF_NORMED |

## 7. 不在范围内
- 订单识别（96维特征不含订单，策略本身不看订单）→ 无需检测
- 分数识别（奖励用 `_score_signature` 像素变化检测，与特征解耦）
- 其他布局：方案按 cramped_room 设计，换布局需重新标定 grid_calib 和静态物查表
