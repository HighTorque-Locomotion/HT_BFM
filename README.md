# HT_BFM · walk_pi_waq_12

本项目仅保留 Pi 12 自由度平地 AMP 任务 `walk_pi_waq_12`，包括训练、断点恢复、播放、键盘调速、Isaac Lab 播放面板，以及 TorchScript / ONNX 导出。

## 环境与资产

在项目根目录使用现有环境：

```bash
conda activate HT_BFM
python -m humanoidverse.flat_amp list-tasks
```

运行环境为 Python 3.11、Isaac Sim 5.1、Isaac Lab 与 CUDA PyTorch。
`environment.yml` 记录依赖和相邻 Isaac Lab 源码路径；已经配置好的 `HT_BFM` 无需重装。
需要安装本项目时运行 `python -m pip install --no-deps -e .`。

机器人默认读取相邻资产仓库：
`../ht_urdf/ht_urdf/Pi_S_12L0A0G0H0W/urdf/Pi_S_12L0A0G0H0W_260909.urdf`，并需要其相邻 meshes。
可用 `PI12_URDF_PATH` 覆盖 URDF 路径。
专家数据位于本项目 `humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/datasets/motion_amp_expert/pi_12_amp/`，
当前 YAML 选择 11 对、共 22 条动作。`PI12_AMP_MOTION_YAML` 可覆盖选择文件。
无需安装或导入 `HT_lab`。

## 训练与恢复

```bash
python -m humanoidverse.flat_amp train \
  --task walk_pi_waq_12 --controller amp \
  --num-envs 4096 --device cuda:0 --headless

python -m humanoidverse.flat_amp train \
  --resume /path/to/model_1000.pt --max-iterations 1000 --headless
```

任务默认就是 `walk_pi_waq_12`，controller 默认 `amp`，可以省略。
日志和 checkpoint 默认写入 `logs/flat_amp/walk_pi_waq_12/`；`--log-dir` 可指定新的输出目录。
短跑验证可使用 `--num-envs 16 --max-iterations 2 --amp-preload 2048`。

## 播放

```bash
python -m humanoidverse.flat_amp play \
  --checkpoint /path/to/model_1000.pt --num-envs 1 --steps 100000

python -m humanoidverse.flat_amp play \
  --checkpoint /path/to/model_1000.pt --keyboard-control --steps 100000

python -m humanoidverse.flat_amp play \
  --checkpoint /path/to/model_1000.pt --headless --steps 100
```

键盘：`W/S` 前后、`A/D` 横移、`Q/E` 转向、`X` 归零。
固定速度可传入 `--velocity VX VY YAW`。图形播放需要可用的 NVIDIA/Vulkan 图形环境。
播放面板见 [面板说明](docs/flat_amp_play_ui.md)。

## 导出

```bash
python -m humanoidverse.flat_amp export \
  --checkpoint /path/to/model_1000.pt --output /path/to/new_export_directory
```

导出目录必须尚不存在。生成 `policy.pt`、`policy.onnx` 和 `interface.json`，无需启动 Isaac Sim。
输入为从旧到新的 10 帧观测，共 470 维；输出为 12 维归一化关节动作。
部署时按任务配置应用 action scale 和默认姿态。

## 实现与验证

- 任务环境：`humanoidverse/flat_amp/_vendor/legged_lab/envs/pi/pi_12dof_env.py`
- 奖励、随机化和训练配置：`.../envs/pi/pi_12dof_cfg.py`
- Pi12 电机与机器人：`.../assets/pi/pi_12dof.py`
- 独立动作加载器：`humanoidverse/flat_amp/_vendor/rsl_rl/utils/motion_loader_forpi_12dof.py`

环境、任务配置、动作加载器及电机不再继承其他任务实现，仍使用 Isaac Lab / RSL-RL 的通用接口。
详细配置见 [Pi12 文档](docs/flat_amp_pi12.md)。

```bash
python -m pytest tests/flat_amp -q
ruff check humanoidverse tests
```

按 `.gitignore` 忽略的历史数据、日志、模型、缓存和环境目录保留，不参与任务注册。
