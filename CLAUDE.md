# CLAUDE.md — 项目运行手册

## 目标
RL 在 Overcooked! 2 寿司关(s_sushi_1_4,campaign 1-4)拿极高分(世界纪录 2136,前五 ~1900),之后泛化到其他关卡。

## 架构
- `mod/OC2StateBridge/` — BepInEx 5 插件(Unity 2017.4 Mono, net35),每帧采集状态、TCP 服务(127.0.0.1,端口由 `OC2BRIDGE_PORT` 环境变量决定,默认 8765)、注入输入、语义动作(INTERACT/THROW)、关卡控制(RESET/LOADLEVEL/TIMESCALE/SETPOS)。`dotnet build -c Release` 自动部署到**原安装目录**的 plugins;实例 2 要手动 copy dll。
- `src/bridge/` — Python 侧:`state_client.py`(TCP 客户端)、`encoder.py`(观测编码)、`env.py`(gym 壳:lifecycle/step/边界检测)、`reward.py`(reward shaping,权重全在 `REWARD_WEIGHTS`)、`nav.py`(菜谱路由)、`level_config.py`(关卡坐标)、`ppo_train.py`(训练入口)、`bc_train.py`(BC 预训练,BCPolicy 被 PPO 复用)。
- `docs/REAL_GAME_RL_STATUS.md` — 详细状态文档;`work/` — 探针(probes/)、测试(tests/)、日志(logs/)、工具(tb_trend.py)。

## 双实例并行 RL(关键,踩坑都解决了)
1. 实例 2 是完整拷贝 `E:\SteamLibrary\steamapps\common\Overcooked2_inst2\`;**两个目录都必须有 `steam_appid.txt`(内容 728880)**,否则 Steam 把拷贝重定向回原安装 / 拒绝第二个进程。
2. 端口 8766 被 Cursor 占用 → 实例 2 用 **8767**。
3. 启动顺序(先 2 后 1,PowerShell):
   ```powershell
   $env:OC2BRIDGE_PORT='8767'; Start-Process "E:\...\Overcooked2_inst2\Overcooked2.exe" -WorkingDirectory "E:\...\Overcooked2_inst2"; Remove-Item Env:OC2BRIDGE_PORT
   # 必须等 8767 LISTENING 再起实例 1!间隔太短 Steam 仍会把实例 2 重定向到原路径
   # (两个 Overcooked2 进程同路径 = 失败了,全杀重来)
   Start-Process "E:\...\Overcooked! 2\Overcooked2.exe" -WorkingDirectory "E:\...\Overcooked! 2"
   ```
   不要用 ProcessStartInfo UseShellExecute=$false(游戏立刻 exit 53)。
4. 重启训练:
   ```powershell
   # 先快照!checkpoint 文件名跨 run 碰撞,按 LastWriteTime 取最新
   Copy-Item (最新 ppo_sushi_*.pt) checkpoints\ppo_sushi_runN_end.pt
   python -u src/bridge/ppo_train.py --ports 8765 8767 --resume checkpoints\ppo_sushi_runN_end.pt --timescale 3 --decision-hz 5 --total-steps 1000000 *>&1 | Tee-Object work/logs/runM.log
   ```
   `-u` 必须(否则管道块缓冲,日志几小时看不到)。

## 探针/调试教训(血泪)
- 探针第一步必须 `send_action(player=p, move=(0,0))` 零化**两个**玩家的移动输入——注入的移动输入会永久残留。
- **探针必须 `set_mode("drive")`**——没有 drive 模式,move/use 注入被完全忽略(传送和语义事件照常,极具迷惑性:切菜静默无进度、fwd 永远不变)。
- 拾取动画 ~1s:INTERACT 后**至少等 1.3s 再传送**,否则动画被取消、物品被弹飞(切好的黄瓜飞了 7m)。
- 3x timescale 下一回合 ~70 wall 秒;探针跨回合边界会拿到 no-player/WorldMap。
- 箱子身份(fetch 实测):(8.4,-2.4)=米饭、(8.4,-1.2)=海带、(13.2,-2.4)=黄瓜、**(13.2,-1.2)=鱼**(长期误标为灭火器——灭火器是 (14.4,-2.4) 的散件)。
- 上菜口 (15,-12) 正对北站位 (15.0,-10.9) INTERACT 即交付(实测 +63 分);站偏会被 plate_return (13.2,-12) 抢先。
- 盘子站 (15,-12) 就是上菜口;切菜板在 (15.6,-2.4)/(20.4,-2.4)。
- 订单只有 Sushi_Cucumber 和 Sushi_Fish——都需要米饭+海带+切好的料,**不切菜永远 0 分**。
- items[] 的 PhysicalAttachment 扫描**包含手持物**(手持=挂点在玩家身上)——奖励核算必须按名去重,否则拿起/放下产生幻影增减。
- 盘子 contents 是嵌套复合节点(真名在 children);锅里熟米饭仍是 ingredient 节点(Cooked* 只在盘子上)。

## 监控
- `python work\tb_trend.py` — 最近 50 个 rollout 的 reward parts 正率 + episode 分数。
- 游戏日志:`<安装目录>\BepInEx\LogOutput.log`。

## Commit 规范
英文、尽量短,结尾 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
