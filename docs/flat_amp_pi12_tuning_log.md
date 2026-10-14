# walk_pi_waq_12 调参日志

同日有追加调整；同一项以最后一条修改记录为准，前面的记录保留为修改历史。

## 2026-09-25：放宽前进转弯时的姿态和落脚约束

问题：当前策略在同时给定前进线速度与 z 轴角速度时行走不稳。
本次修改允许转弯时调整身体倾斜、髋部姿态和步宽，同时加强横滚/俯仰角速度抑制。
以下记录的是代码与配置变更，尚无本次变更后的训练或策略表现结论。

### 四元数与终止条件

- `mdp/rewards.py::get_euler_rpy` 改为读取 Isaac Lab 的 **wxyz** 四元数。
  `get_euler_xyz_tensor` 返回带符号的 `(roll, pitch, yaw)`，单位为弧度。
- `body_orientation_euler` 的欧拉角奖励选取 `body_euler[:, :2]`，明确约束 roll/pitch，
  不约束世界航向。原实现误用 xyzw，导致旧变量第三项实际对应 roll。
- `RobotCfg.max_body_yaw_angle_rad` 重命名为 `max_body_roll_angle_rad`，值仍为 `0.8`。
  `check_reset` 改用第 0 项 roll，与第 1 项 pitch 一起检查。
- 终止条件：`abs(roll)>0.8 rad` 或 `abs(pitch)>1.0 rad`；原有根部整体倾斜超过
  45° 的条件也保留。yaw 不触发此姿态终止条件。接触、超时及高度条件沿用原实现。
- 同步更新旧配置覆盖文件时，需将 `max_body_yaw_angle_rad` 改名为
  `max_body_roll_angle_rad`；策略网络及观测维度没有改变。

### 奖励门控

所有判断使用当前指令，不使用实际速度；沿用线速度阈值 `0.1 m/s`、角速度阈值 `0.1 rad/s`。

- **纯前进**：`vx_cmd >= 0.1` 且 `abs(vy_cmd) < 0.1` 且 `abs(wz_cmd) < 0.1`。
  指正向 x 行走，不包含后退；侧向或转向分量在死区内允许存在。
- **站立**：`norm([vx_cmd, vy_cmd]) < 0.1` 且 `abs(wz_cmd) < 0.1`。
  与已有 `command_activity_masks` 一致；例如 `(0.09, 0.09, 0)` 不属于站立。

| 奖励项 | 修改前 | 修改后 |
|---|---|---|
| `joint_deviation_hip_walk` | 除原地转向及侧移豁免外，惩罚左右 thigh/hip_roll 相对默认角度的平方偏移；前进转弯也受罚 | **仅纯前进**时惩罚，权重仍为 `-1.0`；站立、后退、侧移、原地转向、前进转弯均豁免 |
| `feet_y_distance` | 只要 `abs(vy_cmd)<0.1` 就惩罚 | **仅纯前进或站立**时惩罚；权重仍为 `-2.0`，保持 `abs((y_left_body-y_right_body)-0.16)` 的计算方式 |

`joint_deviation_hip`、`joint_deviation_legs`、`stand_still` 的站立惩罚没有改变。
站立时虽然不再有 `joint_deviation_hip_walk`，上述站立约束仍然存在。

### 奖励权重

| 奖励项 | 修改前 | 修改后 |
|---|---:|---:|
| `body_orientation_euler` | `+1.0` | `0.0`，禁用 |
| `flat_orientation_l2` | `-1.0` | `-1.0`，保留 |
| `ang_vel_xy_l2` | `-0.05` | `-0.3` |
| `ang_vel_xy_l2_body` | `-0.05` | `-0.05`，保留 |
| `lin_vel_z_l2` | `-1.0` | `-0.2` |

以上是配置权重。任务奖励每步仍乘 `dt=0.02 s`，之后按
`0.65 * task_reward + 0.35 * amp_reward` 混合；AMP 内部系数仍为 `0.3`。

### 当前环境/指令采样比例（本次未修改）

默认 `num_envs=4096`。每次重置/指令重采样时按以下概率分组，不是固定数量划分：

| 分组 | 概率 | 指令后处理 |
|---|---:|---|
| 站立 `rel_standing_envs` | 10% | vx、vy、wz 全部置零 |
| 侧移 `rel_onestep_envs` | 15% | vx、wz 置零，仅保留随机 vy；该实现不是单步动作任务 |
| 原地转向 `rel_turn_only_envs` | 15% | vx、vy 置零，仅保留 wz |
| 其余混合运动 | 60% | 保留采样的 vx、vy 和航向控制产生的 wz |

每 10 秒重采样；vx 范围 `[-0.6,1.0] m/s`、vy 范围 `[-0.5,0.5] m/s`、
wz 限幅 `[-1.57,1.57] rad/s`。`heading_command=True`、`rel_heading_envs=1.0`，
所有分组先按 `clip(0.5*wrap(target_heading-current_heading), -1.57, 1.57)`
更新 wz，再应用上表的置零规则。100% 航向控制不是额外的互斥分组。

**没有独立的纯前进固定配额**。纯前进由混合组当前指令是否落入门控范围决定；
实际站立/转向/纯前进的时刻占比也受死区与航向收敛影响，不能直接等同于采样组比例。

### 判断与后续观察

- 关闭衰减很快的直立指数奖励，并在转弯时解除髋关节和步宽约束，方向上符合本次问题。
  保留 `flat_orientation_l2`，仍会温和地鼓励直立。
- 当前 URDF 的根部和躯干固定且旋转对齐，两项机体 x/y 角速度惩罚的合计效果约由
  `-0.1` 增至 `-0.35`。这可抑制晃动，也可能压制倾斜转弯时的角运动，应观察
  roll/pitch、分项奖励、速度跟踪误差和跌倒频率。
- 当前训练仍主要是转到目标航向，持续固定 vx+wz 绕圈的覆盖问题没有随本次修改解决。
- 本次多项参数一起修改；需要训练后比较直行与左右持续转弯，才能判断整体是否改善。

### 验证

- `python -m pytest tests/flat_amp/test_pi12_turn_rewards.py tests/flat_amp/test_pi12.py -q`：
  **42 项通过**（HT_BFM 环境）。新增 29 项覆盖 WXYZ 轴顺序、四元数正负号等价、
  姿态奖励对 yaw 不敏感、前进/站立/后退/侧移/转向及门限边界、roll/pitch/倾斜终止。
- 修改的 Python 文件与新增测试通过 Ruff 检查；`git diff --check` 通过。
- 未启动 GPU 训练、未验证训练收敛或实际行走改善。需重新启动训练才能应用本次变更，
  已运行的训练进程不会自动读取这些修改。

## 2026-09-25 追加：新增前进/后退直行指令组

### 奖励门控

- 直行条件调整为 `abs(vx_cmd)>=0.1`、`abs(vy_cmd)<0.1`、`abs(wz_cmd)<0.1`，
  同时支持前进和后退。共享函数改名为 `straight_command_mask`。
- `joint_deviation_hip_walk`：仅上述直行条件下生效，权重保持 `-1.0`。
- `feet_y_distance`：上述直行条件或站立条件下生效，权重保持 `-2.0`；保留站立约束。
- 站立仍按 `norm([vx_cmd,vy_cmd])<0.1` 且 `abs(wz_cmd)<0.1` 判断。
  本次没有继续修改姿态奖励、角速度和竖直速度惩罚的权重。

### 分组与速度范围

新增配置 `rel_straight_envs=0.15`，从混合组分配 15%，其他分组比例保持不变。
新增 `is_straight_env`，完成重采样与环境初始化接线；直行组在每次航向控制之后
将 vy、wz 置零，避免后续控制步重新写入转向命令。

| 指令组 | 采样概率 | vx（m/s） | vy（m/s） | wz（rad/s） |
|---|---:|---|---|---|
| 站立 | 10% | 0 | 0 | 0 |
| 纯侧移 | 15% | 0 | `[-0.5,0.5]` | 0 |
| 原地转向 | 15% | 0 | 0 | `[-1.57,1.57]`，由航向误差计算 |
| 纯前进/后退 | 15% | `[-0.6,1.0]` | 0 | 0 |
| 混合运动 | 45% | `[-0.6,1.0]` | `[-0.5,0.5]` | `[-1.57,1.57]`，由航向误差计算 |

范围中的线速度均为连续均匀采样；前进/后退没有额外按 1:1 分配，沿用 x 速度范围。
这里的 `abs(vx)>=0.1` 是奖励门控，不是采样下限：直行组可采到 `(-0.1,0.1)`，
此时视为站立死区，髋部直行惩罚关闭，步宽站立惩罚仍生效。

`heading_command=True`、`rel_heading_envs=1.0` 和每 10 秒重采样不变。
原地转向和混合组实际使用
`wz=clip(0.5*wrap(target_heading-current_heading), -1.57, 1.57)`，
目标航向范围为 `[-pi,pi]`；并非持续独立均匀采样固定 wz。
分组比例为随机采样概率，默认仍为 4096 个环境；实际数量和死区后的运动状态比例可能不同。

固定速度/键盘回放关闭新增直行分组，并在键盘应用指令时清除 `is_straight_env`，
确保手动输入的侧移和转向不会被训练用分组覆盖。

### 验证

- `python -m pytest tests/flat_amp/test_pi12_turn_rewards.py tests/flat_amp/test_pi12.py tests/flat_amp/test_keyboard_control.py -q`：
  **57 项通过**。覆盖正负 vx 门限、后退转向豁免、五组比例与互斥关系、速度范围、
  连续航向更新后直行组仍保持零 vy/wz、部分环境重采样及键盘控制。
- 修改文件通过 Ruff 检查和 `git diff --check`。
- 未运行本次修改后的 GPU 训练，尚无策略表现结论；重启训练后生效。

## 2026-09-25 追加：速度命令生成倾斜参考的高斯奖励

### 目标与公式

新增 `track_command_tilt_exp`，作用于 `torso_link`，在所有指令模式下计算。
采用水平航向坐标系 x 向前、y 向左、z 向上，ZYX 欧拉角。
仅使用当前速度命令的旋转耦合项，不对重采样造成的跳变指令求导：

```text
ax = -wz_cmd * vy_cmd
ay =  wz_cmd * vx_cmd
pitch_ref = atan2(ax, g)
roll_ref  = atan2(-ay, sqrt(g*g + ax*ax))
g = 9.81 m/s²
```

例如 `vx=1 m/s, vy=0, wz=1 rad/s` 时目标 roll 约为 -5.82°，pitch 为 0；
匀速直行、纯侧移、原地转向和站立的目标均为零倾角。
这是一种稳态转弯的软参考，不包含加减速引起的倾斜，也不等于唯一可行的双足姿态。

### 奖励形式与参数

roll/pitch 分别计算相对目标的最短角度误差，并分别扣除 3° 容差：

```text
e_roll  = max(abs(wrap(roll  - roll_ref))  - tolerance, 0)
e_pitch = max(abs(wrap(pitch - pitch_ref)) - tolerance, 0)
r = exp(-0.5 * ((e_roll/std)**2 + (e_pitch/std)**2))
```

- 新增项权重暂设 `+1.0`，原始奖励范围 `[0,1]`；不是负惩罚，误差增大时减少正奖励。
- 两个轴分别在 ±3° 容差内时奖励为 1，包括两个轴同时各偏差 3°；超出后连续衰减。
- `std` 暂取 **5°**（向用户推荐的试验起点，尚非训练验证后的最优值）；用户尚未另选数值。
- 配置位于 `PiDof12RewardCfg.track_command_tilt_exp.params`，使用 `math.radians(5.0)`
  和 `math.radians(3.0)` 转换为弧度；可独立调整 `std`、`tolerance` 和 `gravity`。
- 这里明确采用标准高斯的 `exp(-e²/(2*std²))`，与已有速度奖励的 `exp(-e²/std²)`
  约定不同。std 衡量的是**超出容差后的误差**。

仅一轴有误差、另一轴在容差内时：

| 相对目标的总角度误差 | 超出容差的误差 | std=3° | std=5° | std=8° |
|---:|---:|---:|---:|---:|
| ≤3° | 0° | 1.000 | 1.000 | 1.000 |
| 5° | 2° | 0.801 | 0.923 | 0.969 |
| 8° | 5° | 0.249 | 0.607 | 0.823 |
| 13° | 10° | 0.004 | 0.135 | 0.458 |

两个轴都超出容差时，奖励为两个单轴因子的乘积。
权重乘法、每步 `dt=0.02` 和 AMP 混合规则沿用现有实现。

### 与已有配置的关系

- `body_orientation_euler` 继续关闭；`flat_orientation_l2=-1.0` 按此前要求保留。
  因此 3° 容差只保证新奖励不扣分，整体奖励仍有轻微的零倾角偏好。
- 角速度、竖直速度、关节及步宽奖励、环境分组比例均未继续修改。
- 新函数不保存历史状态，也不做指令求导，因此没有差分尖峰或跨重置的历史污染。
  指令跳变仍会使目标倾角跳变；本次未增加滤波、变化率限制或目标倾角限幅。

### 验证

- `python -m pytest tests/flat_amp/test_pi12_turn_rewards.py tests/flat_amp/test_pi12.py -q`：
  **71 项通过**。新增覆盖左/右转、后退转向、侧移转向与两轴耦合的参考方向，
  yaw 不敏感性、逐轴 3° 容差、高斯衰减值、批量输入、指令突变无历史依赖及参数校验。
- 修改文件通过 Ruff 检查和 `git diff --check`。
- 尚未运行新奖励下的 GPU 训练；需重新启动训练，观察转弯稳定性与分项奖励。

## 2026-09-28：删除冗余姿态终止条件

按用户要求，姿态终止仅保留总倾角严格大于 45°（`max_tilt_angle_deg=45.0`，
`max_tilt_angle_rad` 非 None 时优先）。删除 `max_body_roll_angle_rad`、
`max_body_pitch_angle_rad`、独立欧拉角判定及 `bad_euler` 终止原因统计。
环境的 `body_euler` 缓存及 reset/step 中的重复转换不再使用，一并删除；
奖励函数所需的 WXYZ 转换仍保留。接触、高度、超时条件保持原样。

当前 45° 总倾角比旧 roll=0.8 rad / pitch=1 rad 更严格，因此默认判定等价。
以后放宽总倾角时，不会再有独立轴阈值生效。旧配置覆盖中应删除上述两个轴阈值字段，
而不是继续沿用历史记录中的 yaw→roll 改名方式。策略网络和观测维度保持。

验证：`python -m pytest tests/flat_amp/test_pi12_turn_rewards.py tests/flat_amp/test_pi12.py -q`
共 75 项通过，覆盖 45° 附近边界、严格大于、复合倾斜、航向无关性及 10000 个随机姿态
与原判据等价；相关 Python 文件 Ruff 与 `git diff --check` 通过。
HT_lab 的 35 项 CPU 回归也通过，包括旧/新 BFM 与 pi_soccer 姿态判定对照；
日志 `HT_lab/logs/setup/pi_bfm_tilt_only_20260928.log`。此次未重跑 GPU 仿真或训练。


## 2026-10-12 纯侧移滑动窗口速度居中惩罚

参照 HT_lab hi_walk_23dof 的 waist_yaw_center_symmetry_penalty，新增
lateral_vx_center / lateral_wz_center 两项，各权重 -2.0。仅指令
abs(vy)>=0.1、abs(vx)<0.1、abs(wz)<0.1 时累计；窗口1.0秒，控制周期0.02秒时50帧，
填满后分别输出 abs(max(vx)+min(vx)) 和 abs(max(wz)+min(wz))，不平方、无额外死区。
vx为根部世界线速度转换至机器人yaw水平坐标系的x分量；wz为世界z角速度，沿用现有跟踪定义。
退出纯侧移、左右切换或reset清空历史；仅改变速度幅值且不改变侧移方向时窗口继续。
两项由RewardManager乘dt，AMP混合不变。原29项有效BFM奖励与零权重脚位项保持；有效奖励增至31项。
HT_lab同步实现，walk角色门控另行保留；Actor观测、网络、随机化、已有模型不改。
CPU验证覆盖满窗、持续滚动、方向/模式切换、局部坐标、分环境reset及两项相互抵消问题；未进行GPU训练或GUI回放。
新奖励需要重新训练/续训才能影响策略；极值居中不等于平均速度为零，评估同时关注平均vx/wz及实际位移。

## 2026-10-13 停用idle_penalty，与HT_lab同步

按用户要求，将`PiDof12RewardCfg.idle_penalty.weight`从-5.0改为0.0，
与HT_lab的`WalkRewardsCfg.idle_penalty`同步。旧权重注释保留，函数和阈值保留。
该项的平移与转向怠动惩罚全部停用；其余奖励、AMP、专家和网络配置保持。
Pi12配置的非零权重项减少一项。

原因：瞬时平面速度模长判据允许纯侧移时通过前后漂移越过0.1m/s阈值逃罚。
HT_lab已完成该奖励冲突的诊断，本次先关闭此项进行后续训练对照；未据此宣称BFM策略已验证改善。
新启动或续训加载新配置后生效；已有训练进程不会自动更新。本次未启动训练。

验证：`python -m pytest tests/flat_amp/test_pi12.py::test_all_pi12_reward_selectors_match_urdf -q`
1项通过；该测试补齐此前新增`LateralVelocityWindow`的模拟定义，解决配置求值时的NameError。
HT_lab的14项CPU同步检查通过。两份配置AST核对权重均为0.0、语法检查通过，
本项目修改文件Ruff与两项目`git diff --check`通过；未进行GPU训练或策略效果验证。
