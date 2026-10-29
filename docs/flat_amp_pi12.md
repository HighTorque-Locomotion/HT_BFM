# Pi 12 自由度平地 AMP 训练

任务名为 `walk_pi_waq_12`，controller 为 `amp`。本项目仅保留此任务的
ActorCriticWaq、PPO、AMP 判别器、历史编码器与镜像损失。

## 配置与调参入口

逐次变更、指令比例与验证结果见 [Pi12 调参日志](flat_amp_pi12_tuning_log.md)。

任务配置集中在
`humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/pi_12dof_cfg.py`。
参照独立 Pi12 delivery 版本展开，奖励、环境和训练配置不再继承 Pi22/Pi23
任务类；展开时保留原任务的参数值，只复用通用配置类型与计算函数。

| 修改内容 | 配置位置 |
|---|---|
| 奖励函数、权重、阈值和关节/刚体选择器 | `PiDof12RewardCfg` 各项的 `func`、`weight`、`params` |
| 步态周期、左右相位、腾空比例 | `GaitCfg` |
| 站立/行走门限 | `PI_LIN_COMMAND_THRESHOLD`、`PI_YAW_COMMAND_THRESHOLD`（仅用于引用它们的配置项） |
| 摩擦、编码器偏置、质量/质心、重置和推力 | `PiEventCfg` |
| 环境数、终止条件、历史帧数、速度范围、噪声和仿真步长 | `PiDof12WalkFlatEnvCfg` |
| 网络、学习率、PPO、镜像损失、AMP 奖励混合及保存间隔 | `PiDof12WalkAgentCfg` |

例如修改 `track_lin_vel_xy_exp.weight` 对应的定义可调整线速度跟踪权重；
修改 `commands.ranges.lin_vel_x` 可调整速度课程最终的前后速度范围。
奖励项设为 `None` 或 `weight=0.0` 会禁用该项；新增奖励可直接添加 `RewTerm`。
奖励计算函数位于 `humanoidverse/flat_amp/_vendor/legged_lab/mdp/rewards.py`，
若只给 Pi12 添加新计算逻辑，可在该模块新增函数并仅在 Pi12 配置中引用；
修改计算函数后需重新运行 Pi12 回归测试。

电机参数和初始姿态仍在 `.../legged_lab/assets/pi/pi_12dof.py`。
命令行参数（如 `--num-envs`、`--max-iterations`、`--amp-preload`）优先于配置默认值。
修改配置需重新启动训练；回放脚本会覆盖环境数、噪声、推力和速度指令等设置。

2026-09-20 配置展开验证：在真实 Isaac Lab 配置实例上逐项比较环境和 agent
的 `to_dict()`，与改写前的继承配置、delivery 配置均一致；修改 Pi12 实例的
奖励、指令范围和学习率不会污染新实例或已注册的 Pi22 配置。
`test_pi12.py` 与 `test_integration.py` 共 19 项测试及改动 Python 文件的 Ruff
检查通过。RTX 4090 上完成 16 环境、2 次训练迭代、2048 AMP 预载的短跑，
保存 checkpoint 后重新创建策略、严格加载权重并在同一训练环境推理 50 步，
动作、观测和奖励均为有限值；本次未验证 GUI 回放或长跑收敛。

## 速度课程（AMP / BFM 共用）

`PiDof12WalkFlatEnvCfg.velocity_curriculum` 默认启用，参数类及计算实现位于
`humanoidverse/flat_amp/velocity_curriculum.py`。参考 HT_lab 的
`hi_walk_23dof@701e2e031c35199a67caec52fbf346212420bd79` 分支中
`hi_walk_amp/mdp/curriculums.py::velocity_ranges_curriculum`，将按地形分组适配为
平地任务的混合、纯侧移、原地转向和直行/后退四种指令模式。

| 模式 | 初始 vx（m/s） | 初始 vy（m/s） | 初始 wz（rad/s） | 最终范围 |
| --- | --- | --- | --- | --- |
| 混合 | [-0.15, 0.30] | [-0.15, 0.15] | [-0.30, 0.30] | vx [-1.0, 1.5]，vy ±0.8，wz ±2.0 |
| 纯侧移 | 0 | [-0.15, 0.15] | 0 | vy ±0.8 |
| 原地转向 | 0 | 0 | [-0.50, 0.50] | wz ±2.0 |
| 直行/后退 | [0.10, 0.50] | 0 | 0 | vx [-1.0, 1.5] |

每个环境独立保存四个 `[0,1]` 难度等级。实际采样范围为
`initial + level * (target - initial)`；episode reset 时更新等级，10 秒指令重采样只换
指令/模式，不额外升级。原有站立 10%、侧移/转向/直行各 15%、混合 45% 比例保持。

评分沿用参考实现的 `exp(-error² / 0.5²)`：平移模式使用机身坐标系 XY 速度误差平方和，
纯转向使用机身 z 轴角速度误差平方。为适应 episode 内切换模式，按模式分别累计均值，
再乘实际执行步数占完整 episode 步数的比例（上限 1）；因此提前跌倒不会仅凭短暂的
准确跟踪获得高分。站立不计分，未出现过的模式不改变等级。

- 混合、侧移、直行：评分 `>0.70` 升级，`<0.55` 降级。
- 原地转向：评分 `>0.50` 升级，`<0.35` 降级。
- 每次升级 `+0.05`，降级 `-0.02`，中间区间和恰好等于阈值时保持。

航向控制增益从 0.5 调为 1.0，使角速度能达到 ±2 rad/s；每步计算的航向指令仍受当前
环境、当前模式课程范围限幅。速度上限是各分量上限，不是平面速度模长上限。

TensorBoard 的 `Curriculum/<模式>/level` 和各轴 `*_min` / `*_max` 记录全体环境的
平均等级及平均范围。AMP/BFM checkpoint 保存各环境课程等级，续训恢复；环境数变化时
用原各模式的平均等级初始化。旧 checkpoint 无该字段时从初始难度开始。
续训启动时已采样的低速指令保留到下一次正常采样，随后使用恢复的课程等级。
不恢复旧 episode 的分数和累计步数。

固定速度和键盘回放均关闭课程，直接使用指定速度；键盘限幅采用最终范围。
BFM 部署导出的 `interface.json` 同样记录最终指令范围。设置
`velocity_curriculum.enabled=False` 可关闭训练课程并直接从最终范围采样。

## 环境实现入口

### 2026-09-21 横移、纯转向采样与接触力调整

以下为相对调整前配置的变更记录，当前源码已核对：

| 位置 / 参数 | 调整前 | 调整后 |
|---|---|---|
| `PiDof12WalkFlatEnvCfg.commands.rel_onestep_envs` | `0.05` | `0.15`（纯横移命令采样比例） |
| `PiDof12WalkFlatEnvCfg.commands.rel_turn_only_envs` | `0.05` | `0.15`（纯转向命令采样比例） |
| `PiDof12RewardCfg.joint_deviation_hip_walk.params` | 仅豁免纯转向 | 新增 `exempt_lateral=True`，同时豁免横移 |
| `PiDof12RewardCfg.feet_force.params.threshold` | `200` | `170`（取建议范围 160～180 的中间值） |
| `PiDof12RewardCfg.feet_force.params.max_reward` | `200` | `400` |

`joint_deviation_hip_walk.weight` 仍为 `-1.0`，`feet_force.weight` 仍为 `-0.03`。
共享函数 `mdp/rewards.py::joint_deviation_l2` 增加可选参数
`exempt_lateral=False`，由 Pi12 单独启用。横移豁免条件为
`abs(vy_cmd) >= lin_command_threshold` 且
`abs(vx_cmd) < lin_command_threshold`，不额外限制 yaw 命令；
前进/后退和明显含前后速度的斜向行走仍保留原惩罚，原有纯转向豁免保留。
其他未启用该参数的任务保持原有行为。

本轮未调整 `foot_gait.params.threshold`，其值仍为 `0.56`。
上述训练参数需重启训练或在新配置下恢复训练后生效，不会自动改变已运行的训练进程。

`humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/pi_12dof_env.py` 中的
`PiDof12Env` 已展开为独立环境，只继承 RSL-RL 的 `VecEnv` 抽象接口，
不再继承或导入 `PiDof23Env`。Pi12 的环境逻辑可直接在此文件修改：

| 修改内容 | 方法 |
|---|---|
| 场景、机器人、指令及奖励管理器初始化 | `__init__`、`init_buffers` |
| 动作缩放、延迟和物理步进、奖励计算调用 | `_resolve_action_scale`、`step` |
| Actor/Critic 单帧观测、噪声与历史缓存 | `compute_current_observations`、`init_obs_buffer`、`compute_observations` |
| AMP 36 维特征及双脚坐标变换 | `get_amp_obs_for_expert_trans` |
| 接触、姿态、高度和超时终止 | `check_reset` |
| 重置、日志及终止前 AMP/Actor 状态保存 | `reset` |

2026-09-28：姿态终止仅保留总倾角 >45°，删除冗余 roll/pitch 阈值及 `bad_euler` 统计。
总倾角由根部重力投影计算，yaw 自由；接触、高度及 20 s 超时规则保持。
详细变更与验证见 [调参日志](flat_amp_pi12_tuning_log.md)。

| 站立门控、相位和左右脚步态更新 | `_calculate_gait_para` |

展开保留当前项目的回放面板更新，以及 `step` 内自动重置前保存终止状态的
处理；正常返回或异常时均恢复采集标志。奖励项仍由 `pi_12dof_cfg.py` 配置，
计算函数仍复用 `mdp/rewards.py`，环境中通过 `reward_manager.compute` 调用。

2026-09-20 环境展开验证：22 项相关 CPU 测试和 Ruff 检查通过；GPU 上完成
16 环境、2 次训练迭代、2048 AMP 预载。新生成和展开前的 checkpoint 均严格
加载成功，各推理 50 步且动作、观测、奖励均为有限值。强制全部环境超时后，
正确返回 `(16, 36)` 的终止 AMP 状态和 `(16, 47)` 的终止 Actor 观测。
本次验证为无界面短跑，未验证 GUI 回放或长期收敛。

## 输入资产与专家数据

URDF 使用相邻资产项目，专家数据默认使用 HT_BFM 项目内的副本及镜像：

- URDF：`../ht_urdf/ht_urdf/Pi_S_12L0A0G0H0W/urdf/Pi_S_12L0A0G0H0W_260909.urdf`
- 专家目录：`humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/datasets/motion_amp_expert/pi_12_amp/`
- 专家选择：上述目录下的 `Pi_S_12L0A0G0H0W_LSE_lafan_walk.yaml`，当前启用 11 对、共 22 条。

可用 `PI12_URDF_PATH` 和 `PI12_AMP_MOTION_YAML` 环境变量覆盖路径。保留 URDF
相邻的 meshes 目录。训练不再依赖外部 HT_lab 数据目录，无需安装或导入 HT_lab。

按 YAML 的 `selected_files` 读取全部 NPZ，路径可相对 YAML 目录或其父目录
（项目内选择文件使用相对 YAML 所在目录的路径）。文件包含：
`joint_names`、`joint_pos`、`base_pos_w`、`base_quat_w`（wxyz）、`framerate`。
源轨迹虽然含上肢，但双腿几何基本相同，加载器按名称取出腿部 12 个关节，
保留时间、根运动和关节角；通过 MuJoCo 对指定 URDF 做离线 FK，重新计算
根坐标系内的脚踝位置，不模拟、不写回原数据。速度采用前向差分，最后一帧
沿用前一帧导数；四元数差使用最短旋转以避免符号翻转造成角速度尖峰。

### 专家左右对称扩充

现有 AMP 加载器不在线生成镜像专家。PPO 的 `use_mirror_loss=True` 约束策略的
左右对称性；`use_data_augmentation` 同样针对策略训练数据，不会扩充专家库。
Pi12 的 `pi_12_amp/` 保留 27 条原件和 27 条离线镜像，其中包含原先清单
漏记的反向行走动作对。当前 YAML 选择其中 11 对、共 22 条片段，按片段等权采样。
清理任务代码时没有修改 NPZ 内容或 YAML 选择，帧数、帧率不变。
镜像规则为矢状面反射（world Y=0）：

- 按关节名称交换左右；hip_pitch/calf/ankle_pitch 保持符号，hip_roll/thigh/ankle_roll 取反。
- 根位置 Y 取反；WXYZ 根四元数变为 `(w,-x,y,-z)`，不是直接交换四元数分量。
- 原始命名上肢列也进行相应镜像，训练仍只取腿部 12 关节。
- 镜像文件只保留上述五个原始姿态字段；不复制源机器人缓存的刚体位置/速度。
  Pi12 加载器重新差分求速度，并使用目标 URDF 重算脚端 FK。

`manifest.json` 记录原件与镜像的 SHA256、帧数、帧率和省略的缓存字段。
原始动作数据保持不变。复现镜像命令如下（输出必须是
不存在的新目录；已附带生成好的数据，日常训练不需要执行）：

```bash
python -m humanoidverse.flat_amp.augment_pi12_data \
  --source-yaml ../HT_lab/Datasets/pi12_walk/Pi_S_12L0A0G0H0W_LSE_lafan_walk.yaml \
  --output /tmp/pi12_amp_augmented
```

如曾设置 `PI12_AMP_MOTION_YAML`，该变量仍优先于项目内默认路径；要使用
项目当前默认选择，先 `unset PI12_AMP_MOTION_YAML`。

镜像数据生成后，44 项 CPU 测试通过，覆盖全部 52 条轨迹的加载/采样、
镜像两次还原、四元数反射、AMP 速度/关节特征及目标脚端 FK 对称关系，
以及已有 PPO 更新与导出回归。本次数据扩充后未重新运行 GPU 训练。

## 观测、输出与训练

| 接口 | 维度 / 内容 |
|---|---|
| 单帧 actor 观测 | 47：角速度 3、重力 3、指令 3、关节位置偏差 12、关节速度 12、上一动作 12、相位 sin/cos 2 |
| 历史 | 10 帧，共 470 维，最旧到最新 |
| Actor MLP 输入 | 当前帧 47 + 估计线速度 3 + 历史隐变量 16 = 66 |
| Critic 输入 | 52：当前帧 47 + 真值线速度 3 + 双脚接触 2 |
| 动作 | 12 维，运行时 articulation 关节顺序，全部用于腿部位置控制 |
| AMP 单帧 | 36：根局部线/角速度 6、关节角 12、关节速度 12、双脚局部位置 6 |
| AMP 判别器输入 | 前后两帧拼接，共 72 维 |

AMP 内部关节顺序固定为左腿六关节、右腿六关节；策略动作和观测使用仿真
运行时关节顺序。二者按名称映射，不假设 URDF 文本顺序等于 PhysX 顺序。
脚端顺序为左、右。无上肢、头部、腰部的观测、动作屏蔽或对应奖励选择器。
保留 `torso_link` 固定刚体供姿态奖励和质量/质心随机化使用。

Actor/critic 隐藏层 `[512,256,128]`，历史编码器及 VAE 辅助损失沿用现有配置；
AMP 判别器 `[1024,512,256]`。50 Hz 控制（200 Hz 物理仿真，decimation=4）。
默认 PPO rollout=24、epochs=5、minibatches=4、lr=1e-3（KL 自适应）、γ=0.99、λ=0.95。
奖励为 `0.65 * task_reward + 0.35 * amp_reward`，AMP 系数为 0.3。
指令域沿用行走任务：vx `[-0.6,1.0]`、vy `[-0.5,0.5]`、yaw `[-1.57,1.57]`。
虽然专家 YAML 含跑步片段，本任务仍使用现有行走指令和步态奖励。

Pi12 实机电机确认为 5036，参数对齐 HT_lab 的 `assets/PiPlus_S_12L8A0G2H0W_5036.py`
及 `HTMotorCfg_5036` 配置：armature=0.013212，Kp≈52.15889、Kd≈3.32054，
effort_limit_sim=24 Nm、velocity_limit_sim=60 rad/s。
力速曲线为 `|ω| = -0.006667|T|² - 0.113990|T| + 7.732552`，
max_torque=23.7 Nm、max_velocity=7.95 rad/s；后者与仿真关节速度限值含义不同。
驱动按曲线限幅，制动按 23.7 Nm 限幅。动作 scale 为 `0.25 * 24 / Kp`（约 0.115033 rad）。
默认姿态为 hip_pitch=-0.25、calf=0.65、ankle_pitch=-0.4，其余为 0，根高度 0.351 m。
电机延迟沿用 HT_lab Pi12 训练配置的 0..4 个物理步，使用 delayed PD 路径；
不改变已有 PiPlus 任务的电机实现。`--actuator-profile legacy` 不适用于本任务。

## 命令

在已有 Python 3.11 / Isaac Sim 5.1 / Isaac Lab 环境中，从 HT_BFM 根目录执行：

```bash
python -m humanoidverse.flat_amp train \
  --task walk_pi_waq_12 --controller amp \
  --num-envs 4096 --device cuda:0 --headless
```

先进行短跑检查：

```bash
python -m humanoidverse.flat_amp train \
  --task walk_pi_waq_12 --controller amp \
  --num-envs 16 --max-iterations 2 --amp-preload 2048 \
  --device cuda:0 --headless
```

输出在 `logs/flat_amp/walk_pi_waq_12/`，包含 checkpoint 与实际训练配置。
Pi12 的保存间隔在 `humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/pi_12dof_cfg.py`
的 `PiDof12WalkAgentCfg.save_interval = 1000` 配置，不影响其他任务。
Runner 使用从 0 开始的迭代编号，周期保存 `model_0.pt`、`model_1000.pt`、
`model_2000.pt` 等，正常训练结束还会额外保存最后一次迭代。
修改对下一次启动的训练生效，已运行的进程不会自动更新。
恢复训练追加 `--resume /path/to/model_N.pt`，回放使用：

```bash
python -m humanoidverse.flat_amp play \
  --task walk_pi_waq_12 --controller amp \
  --checkpoint /path/to/model_N.pt --velocity 0.4 0 0
```

通用 `flat_amp export` 支持此任务的 470 维历史输入与 12 维输出。
普通 AMP 回放支持可选键盘调速，按键映射参考 HT_lab 的 `hi_walk_amp/play.py`：

```bash
python -m humanoidverse.flat_amp play \
  --task walk_pi_waq_12 --controller amp \
  --checkpoint /path/to/model_N.pt --num-envs 1 --device cuda:0 \
  --keyboard-control --steps 100000
```

先点击 Isaac Sim 窗口使其获得键盘焦点：`W/S` 增减前进速度，`A/D` 增减
侧向速度（A 左移，D 右移），`Q/E` 增减转向角速度（Q 左转，E 右转），
`X` 将全部速度指令归零。松开按键后保持目标速度，长按使用系统重复事件继续调节。
归零不是物理急停，也不保证机器人瞬间停止。每次调节会在终端打印目标速度。

默认从 `(0,0,0)` 开始，可用 `--velocity VX VY YAW` 设置初始值；线速度和
角速度默认每次增减 0.1 m/s、0.1 rad/s，可通过 `--keyboard-linvel-step` 和
`--keyboard-angvel-step` 调整，以上键盘参数也接受 HT_lab 风格的下划线写法。
指令按任务配置限幅（Pi12：vx `[-0.6,1.0]`，vy `[-0.5,0.5]`，yaw `[-1.57,1.57]`），
所有并行环境使用同一指令；重置和定时重采样保留最新键盘指令。
键盘模式按控制周期节流，不快于实时；不支持 `--headless`。
BFM controller 也已接入相同键盘操作，命令见 [BFM 训练说明](flat_amp_pi12_bfm.md)。
`--steps` 仍控制回放步数上限。未加键盘开关时，固定速度回放行为不变。




