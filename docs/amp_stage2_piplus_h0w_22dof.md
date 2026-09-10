# PiPlus H0W 22-DoF AMP Stage 2

本配置将 command-conditioned AMP Stage 2 迁移到无腰关节的
`PiPlus_S_12L8A0G2H0W`，第一阶段 BFM 完全冻结，不启用 LoRA。

## 迁移合同

| 合同 | 第一阶段来源 | Stage 2 目标 | 复用方式 | 变更 |
|---|---|---|---|---|
| BFM | `huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint` | 同一 22-DoF actor，`z_dim=256` | unchanged | `--disable-lora` 保证 BFM 无可训练适配器 |
| 机器人 | checkpoint `config.yaml` 中的 `PiPlus_S_12L8A0G2H0W` | `humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml` | configured | Stage 2 根据 `robot_type` 自动选择 H0W Hydra robot |
| 控制 | checkpoint 的 PD、action scale/clip、HTMotor、限位、armature、friction | H0W robot config | unchanged | 启动前逐字段校验，不一致直接失败 |
| 资产 | H0W URDF + canonical H0W MotionLib XML | Isaac Sim + MuJoCo FK | unchanged | Isaac Sim 使用 URDF；AMP FK 使用现有无后缀 XML |
| AMP 数据 | `/home/sunteng/Project/HT_LAB_AMP_CHECKPOINT/piplus_nowaist_walk_run` | `dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl` | adapted | 读取 YAML 选中的 15 条 NPZ，按名称重排 22 个关节，四元数 `wxyz -> xyzw` |
| AMP 特征 | 局部 root velocity、5 个 key body、8 帧关节历史 | `3 + 5*3 + 8*22 = 194` | unchanged | 仅维度随 DoF 合同变化 |
| 控制频率 | checkpoint Isaac Sim `200 Hz / decimation 4` | 50 Hz | unchanged | 无变化 |
| 部署/export | 本次未要求 | 未修改 | not applicable | 后续导出需保留 22-DoF joint order |

电机实现核对了 `/home/sunteng/Project/HT_lab_pipeline/source/HT_lab/HT_lab/actuators/HT_motor.py`、`HT_motor_cfg.py` 和 `assets/PiPlus_S_12L8A0G2H0W.py`。本项目复用已有的 HTMotor 接入，没有复制新的电机实现。需要注意：当前 pipeline 的 H0W asset 已切到 5047 腿部电机，而这个第一阶段 checkpoint 明确记录的是 5036 对应的 armature/PD（腿部 `0.013212 / 52.15889 / 3.32054`）。本迁移按用户要求以 checkpoint 为准，并通过启动时合同校验防止静默切换到 5047。

数据转换：

```bash
uv run python data_process/convert_piplus_nowaist_amp_dataset.py --overwrite
```

纯 CPU 合同检查：

```bash
uv run python -m humanoidverse.amp_stage2 \
  --dry-run --device cpu \
  --bfm-checkpoint 'huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint' \
  --expert-dataset dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml
```

训练入口：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_amp_stage2_piplus_h0w.sh \
  --device cuda:0 --gpu-ids single --num-envs 4096
```

`scripts/train_amp_stage2_piplus_h0w.sh` 固定传入 `--disable-lora`。额外参数可以从命令行覆盖训练规模和优化参数，但不应覆盖机器人、BFM 或 AMP 数据路径。

## 数据说明

- 原始目录保持不变；本项目中的 `dataset/piplus_nowaist_walk_run` 是数据副本。
- YAML 使用 `PiPlus_S_12L8A0G2H0W_mocap_walk_run.yaml`。
- 转换产物包含 15 条 motion、2,210 帧；默认 8 帧 AMP history 生成 2,105 个窗口。
- 数据类别为 forward 456、backward 427、lateral 550、turn 672 个 8 帧窗口；该源数据没有独立 stand motion。
- NPZ 中存在多种 joint ordering，转换器按 joint name 显式重排到 H0W XML/actor 顺序，禁止按列位置猜测。
- NPZ 的 `base_quat_w` 是 `wxyz`，MotionLib `root_rot` 是 `xyzw`。
