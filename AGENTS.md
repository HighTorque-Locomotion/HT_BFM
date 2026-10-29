# HT_BFM 项目指南

## 沟通与工作习惯

- 默认用中文，按具备工程技术基础的工科硕士同行交流；先给结论，再给必要依据。
- 回复精准、简洁，不重复解释常识，不机械附加“需要续训才能影响策略”“重启后生效”等提醒。
- 简单修改完成后说明修改项、关键数值和核查结果；详细分析按需展开。
- 开始修改前看 `git status --short`，保留用户已有改动和未跟踪文件，不覆盖或回退无关工作。

## 项目定位与快速入口

当前项目专注于 **Pi 12 自由度双足机器人平地行走**，唯一注册任务为 `walk_pi_waq_12`，
统一入口是 `python -m humanoidverse.flat_amp`，支持 `amp` 和 `bfm` 两种控制器。
训练与物理回放使用 Isaac Lab / Isaac Sim；MuJoCo 用于专家动作的离线正向运动学计算，
当前 CLI 没有 `--simulator mujoco` 回放选项。

不要沿用历史 HumanoidVerse 的 Hydra 配置、`humanoidverse.train` 或 tracking inference 入口。
当前任务配置是 Python `configclass`，主要实现位于 `humanoidverse/flat_amp/_vendor/`。

建议按任务需要阅读：

1. `README.md`：使用概览。
2. `humanoidverse/flat_amp/__main__.py`：CLI 参数、默认值和 AMP/BFM 分派。
3. `humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/pi_12dof_cfg.py`：奖励、指令、随机化和训练配置。
4. `docs/flat_amp_pi12.md`、`docs/flat_amp_pi12_bfm.md`：两种控制器的接口与实现。
5. `docs/flat_amp_pi12_tuning_log.md`：调参历史；`docs/flat_amp_play_ui.md`：回放面板。

历史文档中的参数和数据数量可能对应旧版本，当前行为以源码和实际选择文件为准。

## 环境与外部依赖

在项目根目录使用现有环境：

```bash
conda activate HT_BFM
python -m humanoidverse.flat_amp list-tasks
python -m humanoidverse.flat_amp train --help
```

- Python 3.11、Isaac Sim 5.1、Isaac Lab、CUDA PyTorch；具体依赖见 `environment.yml` 和 `pyproject.toml`。
- 优先复用配置好的 `HT_BFM`，需要安装本项目时用 `python -m pip install --no-deps -e .`。
  Isaac Lab 是相邻源码的 editable 安装，不要通过依赖解析器随意替换 Isaac Lab / Isaac Sim / Torch。
- `flat_amp.activate()` 将本项目 `_vendor` 放到导入路径前部，保留 `legged_lab`、`rsl_rl`
  模块名以兼容 checkpoint；同一进程不能混用外部版本。不需要安装或导入 `HT_lab`。
- 机器人 URDF 默认：
  `../ht_urdf/ht_urdf/Pi_S_12L0A0G0H0W/urdf/Pi_S_12L0A0G0H0W_260909.urdf`，
  必须保留相邻 meshes；可用 `PI12_URDF_PATH` 覆盖。
- 专家数据位于 `_vendor/legged_lab/envs/pi/datasets/motion_amp_expert/pi_12_amp/`
  （相对 `humanoidverse/flat_amp/`）。选择文件为 `Pi_S_12L0A0G0H0W_LSE_lafan_walk.yaml`，
  当前选择 11 对原始/镜像动作，共 22 条；可用 `PI12_AMP_MOTION_YAML` 覆盖。
- BFM 预训练权重默认使用项目根目录的 `pi12dof_ufogpu0123_20260913_125031/`，
  CLI 必须显式指定该目录。路径解析见 `pi12_paths.py` 和 `utils/bfm_actor.py`。

## 两种控制器与正式训练命令

### AMP：直接学习关节动作

`ActorCriticWaq` + `AMPWAQPPO` + `AmpWaqOnPolicyRunner`，包含 DreamWaQ 历史编码/VAE
辅助损失、在线 AMP 判别器和关节动作镜像损失；策略输出 12 维腿部动作。

```bash
python -m humanoidverse.flat_amp train \
  --task walk_pi_waq_12 --controller amp \
  --num-envs 4096 --device cuda:0 --headless
```

### BFM：冻结底层策略，训练 command-to-z

PPO 学习上层 command-to-z 编码器和 critic，同时训练 AMP 判别器；预训练 BFM actor
及其归一化统计固定，不训练 BFM backward encoder。复用当前 Pi12 的物理环境、奖励、
电机、随机化和专家数据；使用 `BFMCommandActorCritic` / `AMPPPO`，
不使用 DreamWaQ VAE 辅助损失和关节动作镜像损失。

```bash
python -m humanoidverse.flat_amp train \
  --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --num-envs 4096 --device cuda:0 --headless
```

任务默认是 `walk_pi_waq_12`，控制器默认是 `amp`。完整无界面参数为 `--headless`。
两者默认训练 50000 次迭代，`--max-iterations` 指定本次执行的更新次数，续训时也是追加次数。
日志默认写到 `logs/flat_amp/walk_pi_waq_12/<timestamp>_amp/` 或 `<timestamp>_bfm/`，
包含 `model_*.pt`、`params/env.yaml`、`params/agent.yaml` 和 TensorBoard 日志。
`--log-dir` 可指定输出目录，但该目录必须尚不存在。

## 续训、回放与导出

```bash
# AMP 续训
python -m humanoidverse.flat_amp train --controller amp \
  --resume /path/to/model_1000.pt --max-iterations 1000 \
  --num-envs 4096 --device cuda:0 --headless

# BFM 续训：同时提供上层 checkpoint 和原始 BFM 权重
python -m humanoidverse.flat_amp train --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --resume /path/to/model_1000.pt --max-iterations 1000 \
  --num-envs 4096 --device cuda:0 --headless

# AMP 固定速度回放
python -m humanoidverse.flat_amp play --controller amp \
  --checkpoint /path/to/model_1000.pt --num-envs 1 \
  --velocity 0.4 0 0 --steps 1000 --device cuda:0 --headless

# BFM 固定速度回放
python -m humanoidverse.flat_amp play --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --checkpoint /path/to/model_1000.pt --num-envs 1 \
  --velocity 0.4 0 0 --steps 1000 --device cuda:0 --headless

# AMP 导出，无需启动 Isaac Sim；输出目录必须尚不存在
python -m humanoidverse.flat_amp export \
  --checkpoint /path/to/model_1000.pt --output /path/to/new_export_directory

# BFM+AMP 导出：自动识别 checkpoint，额外提供原始冻结权重
python -m humanoidverse.flat_amp export \
  --checkpoint /path/to/bfm_run/model_1000.pt \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --output /path/to/new_bfm_export_directory
```

两种控制器均支持图形回放：去掉 `--headless`，可加 `--keyboard-control`。
`W/S` 前后、`A/D` 横移、`Q/E` 转向、`X` 归零；松键保持指令。
AMP 导出生成 `policy.pt`、`policy.onnx`、`interface.json`。
BFM 导出生成 `encoder.onnx`、`bfm.onnx`、`interface.json`，默认读取 checkpoint 同级
`params/env.yaml`；可用 `--env-config`、`--urdf`、`--bfm-robot-config` 覆盖文件位置。
`--check-only` 仅校验 BFM 原生模型与配置，不创建输出目录；相位维度从 checkpoint 自动识别。
导出完全在 HT_BFM 内实现，不依赖 `instinct_onboard`；机载端只需完整导出目录。
BFM checkpoint 通过哈希和接口元数据绑定原始权重，需一起保留；
使用 `--no-phase` 训练的 checkpoint，续训和播放也必须传入该标志。

## 源码导航与调参位置

以下 `legged_lab/` 和 `rsl_rl/` 均相对 `humanoidverse/flat_amp/_vendor/`：

| 要修改的内容 | 入口 |
| --- | --- |
| CLI、参数转换、默认输出目录 | `humanoidverse/flat_amp/__main__.py` |
| 奖励权重/阈值、步态、随机化、指令范围、PPO/AMP 参数 | `legged_lab/envs/pi/pi_12dof_cfg.py` |
| 观测、动作缩放/延迟、物理步进、reset、终止状态 | `legged_lab/envs/pi/pi_12dof_env.py` |
| PD、5036 电机模型、默认姿态、action scale | `legged_lab/assets/pi/pi_12dof.py` |
| 奖励计算、命令采样、随机化、横移窗口统计 | `legged_lab/mdp/{rewards,commands,events,lateral_window}.py` |
| AMP 训练/回放/导出 | `legged_lab/scripts/{train,play,export_policy}.py` |
| AMP 网络、算法、runner | `rsl_rl/modules/actor_critic_waq.py`、`algorithms/amp_waq_ppo.py`、`runners/amp_waq_on_policy_runner.py` |
| 专家 NPZ 加载、关节映射、离线 FK、AMP 特征 | `rsl_rl/utils/motion_loader_forpi_12dof.py` |
| BFM 训练、续训、dry-run、回放 | `legged_lab/scripts/train_bfm_amp.py` |
| BFM 部署导出、接口 schema 校验 | `legged_lab/scripts/export_bfm_policy.py`、`utils/bfm_deployment.py`；统一由 `scripts/export_policy.py` 分派 |
| 原生冻结 BFM 加载与推理 | `legged_lab/utils/bfm_actor.py` |
| BFM 观测历史、关节映射、动作坐标、latent 环境 | `legged_lab/utils/bfm_bridge.py` |
| BFM 配置派生、冻结检查、checkpoint 契约 | `legged_lab/utils/bfm_training.py` |
| BFM 上层网络与 PPO | `rsl_rl/modules/bfm_command_actor_critic.py`、`algorithms/amp_ppo.py`、`runners/amp_on_policy_runner.py` |
| 回放 UI、键盘和显示数据 | `humanoidverse/flat_amp/{play_ui,keyboard_control,play_ui_data}.py` |

配置类分工：`PiDof12RewardCfg` 管奖励，`GaitCfg` 管步态，`PiEventCfg` 管随机化，
`PiDof12WalkFlatEnvCfg` 管环境，`PiDof12WalkAgentCfg` 管训练。
命令行覆盖相应配置默认值。公共环境/奖励修改通常同时影响两种控制器，
BFM 特有的网络和算法配置由 `make_bfm_train_cfg()` 派生。

## 关键接口约束

- 当前控制频率 50 Hz，物理步长 `0.005 s`、decimation `4`。
- AMP actor 单帧 47 维，10 帧历史共 470 维，历史从旧到新；critic 52 维，动作 12 维。
  AMP 特征单帧 36 维，判别器输入前后两帧共 72 维。
- 给定 BFM 权重的 command-to-z 默认输入 215 维（含相位 sin/cos）、critic 267 维，
  输出 256 维 latent；`--no-phase` 对应 213 / 265 维。
  底层 BFM 接收 210 维本体观测和 256 维 z，输出 12 维动作。
- PPO 保存原始 latent 并计算其概率；BFM 推理前使用 `sqrt(256) * normalize(raw_z)`。
  BFM 历史为分组排列的 4 帧、新帧在前且不含当前帧，不能套用 AMP 历史顺序。
- 关节顺序必须按名称映射，不假设 URDF、PhysX、AMP 专家和 BFM 的顺序相同。
  BFM 动作经其默认姿态/action scale 转成绝对目标，再换算到当前 Pi12 动作坐标。
- BFM 加载原生 `checkpoint/model/model.safetensors`、模型 JSON 和 `config.yaml`。
  当前随附 ONNX 与原生 checkpoint 参数不同，训练以原生权重为准。
- 保留自动 reset 前的终止 AMP 状态；部分环境 reset 时只清除对应的 BFM 历史。
- 当前任务奖励/AMP 混合比例为 `0.65 / 0.35`，AMP reward coefficient 为 `0.3`；
  修改观测、动作或 AMP 特征时，同步核对环境、加载器、网络、checkpoint 和测试。

## 验证与修改规范

项目已有 `tests/flat_amp/`，覆盖 CLI、AMP 更新/导出、Pi12 配置/奖励/数据、
BFM 加载/冻结/历史/映射/续训、键盘和 UI 数据。优先运行相关测试，再按改动范围扩大：

```bash
python -m pytest tests/flat_amp -q
ruff check humanoidverse tests
git diff --check

# BFM 真实权重与接口验证，不启动 Isaac Sim
python -m humanoidverse.flat_amp train --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --dry-run --device cpu --num-envs 2

# AMP 最小训练短跑；需要 Isaac Sim 和 GPU
python -m humanoidverse.flat_amp train --controller amp \
  --num-envs 16 --max-iterations 2 --amp-preload 2048 --device cuda:0 --headless

# BFM 最小训练短跑
python -m humanoidverse.flat_amp train --controller bfm \
  --bfm-weights ./pi12dof_ufogpu0123_20260913_125031 \
  --num-envs 16 --max-iterations 2 --amp-preload 2048 --device cuda:0 --headless
```

- 文档或轻量修改不必启动训练；需要验证训练路径时才运行相应短跑，不能用短跑推断长期收敛。
- 测试不要求全部启动 Isaac Sim，但资产/FK 和真实权重测试需要相应本地文件。
- Python 3.11、四空格缩进、显式导入、snake_case / PascalCase；Ruff 行宽 140，启用导入排序。
  `_vendor` 默认被 Ruff 排除，修改其中代码时需显式对改动文件运行 `ruff check --no-force-exclude <文件>`。
- `test_integration.py` 会校验 `_vendor/source_manifest.json` 的文件哈希。
  修改清单覆盖的源码时，同步相应 `local_sha256`，保留来源信息；不要为消除失败刷新无关条目。
  修改专家数据时同步数据 `manifest.json`、选择 YAML 和相关测试。
- 保留第三方许可证及来源清单。不要提交或重生成无关权重、日志、缓存和资产。
  `logs/`、`model/`、`outputs/`、`exported/` 和预训练权重目录属于运行产物/数据。
- 可复用说明放在 `docs/`，参数演进记入调参日志。提交应聚焦，说明影响的控制器、行为和验证结果。
