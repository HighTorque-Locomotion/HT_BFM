# Flat AMP 播放可视化面板

非 headless 的 `python -m humanoidverse.flat_amp play ...` 默认显示 **IsaacLab** 面板，
支持 AMP 和 BFM/BFM-phase 播放。训练和 headless 播放不创建面板及额外的足底力传感器。

```bash
python -m humanoidverse.flat_amp play \
  --task walk_pi_waq_12 --controller amp \
  --checkpoint /path/to/model_39000.pt \
  --num-envs 1 --device cuda:0 --keyboard-control --steps 100000
```

把 checkpoint 替换为实际文件。`--checkpoint` 和 `--steps` 都需要参数值。
使用项目现有的 Python 3.11 / Isaac Sim 5.1 环境，无需安装或导入 HT_lab。

## 面板操作

面板通常停靠在右侧 Property 附近。展开 **Scene Debug Visualization**，勾选项目，再展开对应曲线。

| 选项 | 内容 |
| --- | --- |
| Simulation Settings | 渲染模式、动画录制；和原版一样，使用 Fabric 时动画录制按钮禁用 |
| Viewer Settings | 环境编号（从 1 开始）、World/Env/Robot 跟随模式、相机位置和目标 |
| 场景对象和传感器 | Isaac Lab 原生调试显示；没有实现调试绘制的项目会禁用 |
| Actions | 网络输出、延迟和裁剪后的动作、实际关节位置目标（rad）、关节力矩（Nm） |
| Observations | 当前 actor 帧的角速度、重力投影、速度指令、关节位置/速度、上一动作、相位；critic 额外的线速度和足接触 |
| Commands | 目标速度以及测得的机体线速度、角速度 |
| Rewards | 原生 RewardManager 缓存的各奖励项（加权值，遵循 Isaac Lab 原生显示口径） |
| Curriculum | 有生成地形时显示地形等级；平地任务显示无活动课程的提示 |
| Termination | 接触超限、倾斜、姿态、高度和超时终止信号 |
| Command Markers | 目标速度和实际速度的场景箭头 |
| Foot Contact Forces | 左右脚世界坐标系 Fx/Fy/Fz，单位 N；地面法向力与摩擦力之和 |

曲线沿用原版 `LiveLinePlot`：图例控制单条曲线显示、自动缩放/手动范围、滤波、暂停。
每条曲线保留约 200 个控制步；横轴是采样序号。切换环境编号会清空旧曲线，防止混入另一环境的数据。
W/S、A/D、Q/E、X 的键盘操作保持原有行为；控制机器人时将焦点移回仿真视口。

观测显示策略实际使用的当前帧，保留归一化缩放和噪声语义，不会为绘图重新计算或追加历史。
BFM 模式下这些观测和动作对应底层物理环境，不包含高层 BFM latent 编码器的内部信号。
奖励曲线是环境奖励项，不包含训练时由 AMP 判别器额外计算的奖励。

## 实现来源与接入

HT_lab 的 `scripts/instinct_rl/play.py` 通过环境创建窗口，窗口主体来自 Isaac Lab 的
`ManagerBasedRLEnvWindow` / `BaseEnvWindow`，示波器来自 `ManagerLiveVisualizer` / `LiveLinePlot`。
HT_lab 的 `source/HT_lab/HT_lab/tasks/pi12_walk_amp/ui.py` 扩展了双脚三轴力面板。

本项目直接复用安装环境中的原生窗口、相机和绘图组件：

- `humanoidverse/flat_amp/play_ui.py`：窗口适配、图表生命周期和环境选择。
- `humanoidverse/flat_amp/play_ui_data.py`：只读信号适配，不导入 Kit。
- vendored `legged_lab/utils/env_utils/scene.py`：仅 GUI 播放增加双脚独立地面过滤传感器；
  使用 64 条接触记录容量，沿用 HT_lab 对足部冲击缓冲溢出的处理。
- vendored `legged_lab/envs/pi/pi_env.py`：在奖励/终止计算后、自动重置前每控制步采样一次，
  防止渲染刷新重复采样，也保留终止前接触力和终止信号。
- 两个播放入口负责创建面板，并在 Isaac Sim 关闭前释放窗口和相机订阅。

面板通过独立的 UI facade 提供管理器接口，不替换训练环境中的管理器，不改动策略输入、动作或奖励。

## 验证

```bash
python -m pytest tests/flat_amp/test_play_ui_data.py tests/flat_amp/test_keyboard_control.py -q
python tests/flat_amp/smoke_play_ui.py
```

第二条需要 GPU 和图形桌面，会创建两个 Pi12 环境，检查曲线更新、面板开关、切换回环境 1、
相机跟随、键盘指令、足底力、重置前的终止信号和资源清理；成功打印 `PLAY_UI_SMOKE_OK`。
