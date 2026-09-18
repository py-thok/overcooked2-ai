# 方案 B 技术可行性分析：视觉感知 → 96 维 MLP 向量

## 目标
从 Overcooked! 2 真实游戏截图，重建 overcooked-ai `featurize_state` 输出的 96 维向量，
使模拟器训练的 MLP 策略（`MlpPolicy` net_arch=[256,256]）能直接用于真实游戏。

## 96 维特征的真实构成（已逆向 `mdp.featurize_state` 源码）

以 cramped_room（2玩家, num_pots=2）为例，玩家 i 视角：

| 段 | 长度 | 内容 | 视觉可获取性 |
|---|---|---|---|
| pi_orientation | 4 | 朝向 one-hot | ✅ 角色精灵朝向（模板匹配4个方向） |
| pi_obj | 4 | 手持物 one-hot (onion/tomato/dish/soup) | ✅ 角色头顶图标 |
| pi_wall_{j} | 4 | 四方向是否贴墙 | ✅ 由地图+坐标推算（地图固定） |
| pi_closest_{onion\|tomato\|dish\|soup\|serving\|empty_counter} | 12 | 6类目标的 (dx,dy) **A*路径代价** | ⚠️ 需目标检测+A*规划 |
| pi_closest_soup_n_{onions\|tomatoes} | 2 | 最近汤的配料数 | ⚠️ 锅上方UI气泡，需计数识别 |
| pi_closest_pot_{j}_* (×2锅) | 20 | 锅存在/空/满/煮中/好/配料数/剩余时间/(dx,dy) | ⚠️ 锅状态+倒计时数字OCR |
| other_player_features | 44 | 队友整套特征 | ✅ 同上（换角色颜色） |
| dist_to_other | 2 | 队友相对坐标差 | ✅ 两个角色坐标相减 |
| pi_position | 2 | 自身 (x,y) 网格坐标 | ✅ 网格定位 |
| **合计** | **96** | | |

## 核心难点（按风险排序）

### 难点1：dx/dy 是 A* 路径代价，不是欧氏/曼哈顿距离 ❗
`get_deltas_to_closest_location` 调用 `mlam.motion_planner.min_cost_to_feature`，
是**考虑墙壁网格的可达路径长度**。
- 好消息：cramped_room 地图固定（5×4），motion_planner 可在感知层内置重建
- 做法：感知层只需给出"目标在哪个格子"，A* 距离用与模拟器相同的代码离线计算
- **结论：可解决，但必须复用 overcooked_ai_py 的 MotionPlanner，不能自己算欧氏距离**

### 难点2：玩家网格坐标定位
- 角色是连续移动的（格子间平滑过渡），特征却是**离散网格坐标**
- 需要"像素坐标 → 网格坐标"的吸附逻辑（取最近格心）
- 角色 sprite 大、颜色独特（chef1 蓝/chef2 绿），模板匹配或颜色分割可行
- **风险：中。移动中朝向判定可能抖动**

### 难点3：锅的状态机与倒计时
- 锅有 空/加了料/煮中(有进度条)/煮好 四态，sprite 不同 → 模板匹配可区分
- `cook_time` 是**剩余秒数整数**：煮中时有进度条，进度条长度→剩余时间线性映射
- **风险：中高。进度条像素测量需要标定，且不同分辨率要重新标定**

### 难点4：soup 配料计数
- 锅上方气泡显示已投入的洋葱/番茄数（小图标堆叠）
- 图标小、可能重叠 → 最脆弱的环节
- **缓解：cramped_room 菜谱固定（洋葱汤），只需数洋葱，且最多3个**

### 难点5：订单/分数（不进 96 维，但进奖励）
- 分数 HUD 已有 `_score_signature`（像素变化检测），不依赖感知层
- 96 维特征**不含订单信息**（overcooked-ai 特征设计如此）→ 策略本身不看订单

## 服务器限制
当前服务器无 X display，无法运行真实 OC2。**验证策略**：
用 overcooked-ai 的 `OvercookedGridworld` 渲染器生成仿真截图，
在其上走通"截图→检测→96维向量→MLP推理"全链路，并与
`featurize_state_mdp` 的 ground truth 逐维比对（误差报告）。

## 实现架构

```
src/perception/
├── FEASIBILITY.md          ← 本文档
├── capture.py              ← 截图获取（真实=mss / 仿真=mdp渲染器）
├── detectors.py            ← 各类检测器（颜色分割/模板匹配/OCR占位）
├── feature_builder.py      ← 检测结果 → 96维向量（复用 MotionPlanner 算A*距离）
├── pipeline.py             ← 端到端: 截图 → vector → PPO.predict → action
└── assets/templates/       ← 模板图像库（待标定）
```

## 结论
**技术可行，但工作量集中在"锅状态+配料计数"的视觉识别，以及 A* 距离的正确重建。**
 cramped_room 是最简单的布局（固定地图、单一菜谱、2锅），作为首个目标合理。
建议先在仿真渲染图上把 feature_builder 与 ground truth 对齐到 >99% 一致，
再投入真实截图的模板标定。
