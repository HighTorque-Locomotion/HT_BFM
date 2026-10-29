# Pi12 command-to-z：冻结 BFM + AMP/PPO

## 训练方案与配置

参考 `origin/bmf-amp-lyp`和 `0c234a2`；两者的
`bfm_bridge.py`、`bfm_training.py`、`BFMCommandActorCritic`、`AMPPPO` 核心相同。
本实现将该方案接到当前 `walk_pi_waq_12`，保留独立的普通 AMP/DreamWaQ 路径。

- 物理环境直接使用 `PiDof12Env` / `PiDof12WalkFlatEnvCfg`，机器人资产、5036 电机、PD、
  动作延迟、奖励、命令分布、噪声、随机化、终止条件均沿用现有配置。
- 速度课程与 AMP 共用：按跟踪评分逐步扩展到 vx `[-1.0,1.5] m/s`、vy `±0.8 m/s`、
  wz `±2.0 rad/s`，课程等级随 checkpoint 保存；[参数和升降级规则](flat_amp_pi12.md#速度课程amp--bfm-共用)。
- 训练参数从 `PiDof12WalkAgentCfg` 深拷贝；复用 `AMPLoaderPi12`、当前 22 条专家轨迹、
  AMP 特征和在线判别器。任务奖励比例 0.65、AMP 比例 0.35，AMP reward coefficient 为 0.3。
- PPO 优化上层 command-to-z 和特权 critic，同时训练 AMP 判别器。BFM actor 与其
  BatchNorm 参数/统计量固定，排除在优化器之外。不训练 BFM 的 backward encoder。
- command-to-z 沿用参考分支的 `Linear → LayerNorm → Tanh → Linear`（本权重对应隐藏层 256）。
  输出高斯分布的均值，初始标准差 `exp(-1.5)`；`log_std` 约束在 `[-5, 1]`。
- PPO 存储与计算概率使用原始 256 维 latent，送入 BFM 前投影为
  `z = sqrt(256) * normalize(raw_z)`。确定性播放使用均值后再投影。
- 使用普通 AMP/PPO，不使用 DreamWaQ 的 VAE 辅助损失。沿用参考方案关闭关节动作镜像损失，
  因为 12 维关节镜像排列不能直接用于 256 维 latent；专家数据的镜像轨迹照常使用。

## 观测和动作接口

给定权重 `pi12dof_ufogpu0123_20260913_125031` 的接口：

| 输入 | 维度 | 顺序 / 处理 |
| --- | ---: | --- |
| command | 3 | 当前任务的 vx、vy、yaw 命令 |
| state | 30 | 12 维关节位置偏移、12 维关节速度、3 维重力投影、3 维角速度 |
| last_action | 12 | BFM 输出乘 32 后裁剪到 ±32，保持 BFM 训练时的反馈定义 |
| history_actor | 168 | 4 帧，按 actions、base_ang_vel、dof_pos、dof_vel、projected_gravity 分组；新帧在前，不含当前帧 |
| phase | 2 | 当前 Pi12 环境的 sin/cos 步态相位 |

command-to-z 默认共 **215 维**，critic 使用当前任务的 52 维特权观测拼接编码器观测，共 **267 维**。
`--no-phase` 对应 213 / 265 维。速度指令和相位只输入新编码器；底层 BFM 仍接收
210 维本体观测和 256 维 z。参考方案在编码器侧额外把关节速度及其历史乘 0.05；
底层 BFM 保持原有速度尺度，角速度乘 0.25。

桥接器按关节名称在 BFM 顺序和 Isaac 关节顺序之间转换，先根据 BFM 保存的默认姿态、
动作缩放算出绝对关节目标，再换算成当前 Pi12 环境的动作坐标。
BFM 的动作缩放来自其 `config.yaml`，本权重为 `0.25 * 20 / 52.15889 ≈ 0.09586094`；
物理环境继续使用当前 Pi12 的 PD 和 action scale。BFM 的旧电机参数不会覆盖当前机器人配置。

部分环境终止时只清除对应 BFM 历史；AMP 使用物理环境自动重置前保存的终止帧，避免将
重置姿态误当作动作转移的下一状态。

## 权重加载

`--bfm-weights` 接受 run、checkpoint 或 model 目录，查找 `model.safetensors`、
`config.json` 和 `init_kwargs.json`，并向上查找 `config.yaml`；必要时可显式传
`--bfm-robot-config`。加载器只实例化 residual actor 和其使用的归一化器，不加载
BFM 的 forward/backward、critic、判别器或训练 buffer，不需要 `--bfm-source`。

严格核查 12 维动作、256 维 z、归一化方式、观测维度、actor/normalizer 参数完整性和有限性。
当前提供的 ONNX 导出与原生 checkpoint 的参数不同，因此训练以原生 safetensors 为准。
数值回归在内存中将原生参数写入既有 ONNX 图，然后对照两套计算实现；不会修改权重文件。


## 部署导出

与普通 AMP 共用 `python -m humanoidverse.flat_amp export`，根据 checkpoint 中的
`infos.bfm` 自动识别控制器。可显式传 `--controller bfm`，与 checkpoint 不符时报错。
在 HT_BFM 环境和仓库根目录运行，不依赖 ROS、Isaac Sim 或 `instinct_onboard` 源码：

```bash
python -m humanoidverse.flat_amp export \
  --checkpoint /path/to/bfm_run/model_1000.pt \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --output exported/
```

输出目录必须尚不存在，生成以下配对文件：

| 文件 | 内容 |
| --- | --- |
| `encoder.onnx` | command-to-z 均值网络，215 → 256；无相位 checkpoint 为 213 → 256 |
| `bfm.onnx` | 与训练一致的原生冻结 actor 和归一化器，466 → 12 |
| `interface.json` | 模型哈希、关节/动作约定、相位、控制周期、PD 与限位 |

BFM 原始权重不包含在上层 checkpoint 中，因此必须提供 `--bfm-weights`；
导出校验权重哈希与控制契约，不能用预训练目录已有的旧 ONNX 替代。
默认读取 `<checkpoint所在目录>/params/env.yaml` 和其中记录的 URDF。
文件迁移后可用 `--env-config`、`--urdf`、`--bfm-robot-config` 指定对应文件。
无相位 checkpoint 自动识别，不需要 `--no-phase`。

只校验原生模型与配置：

```bash
python -m humanoidverse.flat_amp export \
  --checkpoint /path/to/bfm_run/model_1000.pt \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --check-only
```

ONNX 使用 opset 17，导出时通过 ONNX checker，并用 ONNX Runtime CPU 在 batch 1/2/4 下
对照原生网络数值结果；两模型均通过后才发布整个目录。
将完整目录复制到机器人，交给 `instinct_onboard/scripts/pi_bfm_walk.py --logdir exported/`。
机载端负责按接口组装观测、历史和相位，并将 raw_z 归一化为长度 16 的 latent；
critic 和 AMP 判别器不参与部署。运行端不需要 Torch 或训练仓库。

## 仿真输入记录与 ONNX 对比

BFM 回放支持 `--record-policy <新目录>`，可与 `--keyboard-control` 一起使用：

```bash
python -m humanoidverse.flat_amp play --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --checkpoint logs/flat_amp/walk_pi_waq_12/2026-10-15_10-48-23_020338_bfm/model_34000.pt \
  --num-envs 1 --keyboard-control --steps 100000 \
  --record-policy logs/diagnostics/bfm_34000_keyboard

python -m humanoidverse.flat_amp.compare_bfm_onnx \
  --trace logs/diagnostics/bfm_34000_keyboard \
  --export-dir logs/flat_amp/walk_pi_waq_12/2026-10-15_10-48-23_020338_bfm/exported
```

记录器捕获回放实际执行的 encoder 输入/输出、BFM 输入/输出、目标环境动作、关节目标和
done 标志；不额外执行策略推理。每 512 步保存一个 `chunk_*.npz`，数组前两维为
`[step, env]`，正常退出或 Python 异常退出时保存剩余完整帧。`metadata.json` 保存
checkpoint 哈希、BFM 权重签名、关节顺序及动作转换参数。输出目录必须尚不存在。

对比工具检查 checkpoint/BFM/ONNX 哈希和控制接口，默认以 ONNX Runtime CPU、batch=1
逐帧运行：分别比较 encoder 原始 latent、归一化 latent、相同 BFM 输入的底层输出，以及
encoder→归一化→BFM 串联输出。结果写入采集目录的 `onnx_comparison.json`，包含最大误差、
平均绝对误差、RMSE、P99、逐维最大误差和最差样本位置。关节目标为延迟和 encoder bias
之前的目标角度，采用目标机器人关节顺序；不会把两条独立闭环轨迹的状态发散混入推理误差。

2026-10-22 对 `model_34000.pt` 与同目录 `exported/` 的实测：CUDA checkpoint 回放，
1 个环境、`velocity=0.4 0 0`、1,000 步（20 秒），无 reset；ONNX Runtime 1.26.0 CPU
逐帧使用保存的相同输入。最大绝对误差如下：

| 对比位置 | 最大绝对误差 |
| --- | --- |
| encoder raw z | 2.0981e-5 |
| 归一化 z | 2.6226e-6 |
| BFM，相同 466 维输入 | 1.3858e-6 |
| encoder + BFM 串联动作 | 1.4156e-6 |
| 串联关节目标 | 4.3511e-6 rad（0.0002493°） |

串联关节目标平均绝对误差为 1.0109e-6 rad，RMSE 为 1.2756e-6 rad。
原始数据与报告位于 `logs/diagnostics/bfm_onnx_34000_forward/`。该结果验证本地配对 ONNX
在这段仿真数据上的数值一致性，不涉及机载 TensorRT/FP16、传感器预处理或控制时序。

## 实现位置

- `legged_lab/scripts/train_bfm_amp.py`：训练、续训、验证和播放入口。
- `legged_lab/scripts/export_policy.py`：AMP/BFM 导出统一分派，保留原 AMP 导出接口。
- `legged_lab/scripts/export_bfm_policy.py`：配对 ONNX 导出、训练配置解析与数值核查。
- `legged_lab/utils/bfm_deployment.py`：部署 JSON schema 校验，与机载端约定保持兼容。
- `legged_lab/utils/bfm_actor.py`：原生冻结 actor 的独立加载和推理。
- `legged_lab/utils/bfm_bridge.py`：Pi12 观测历史、关节顺序、动作坐标及 latent 环境适配。
- `legged_lab/utils/bfm_training.py`：配置派生、冻结约束、checkpoint 接口校验。
- `rsl_rl/modules/bfm_command_actor_critic.py`：上层 latent actor / critic。
- `rsl_rl/algorithms/amp_ppo.py`、`rsl_rl/runners/amp_on_policy_runner.py`：参考分支的 AMP/PPO 训练路径。

上述路径均位于 `humanoidverse/flat_amp/_vendor/`。


