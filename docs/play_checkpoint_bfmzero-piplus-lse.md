# PiPlus LSE 专家动作与 Checkpoint 播放

## 播放专家动作

`--motion 0` 表示播放数据集中的第 0 条动作，可改为其他动作编号。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.visualize_motion \
  --robot piplus_lse \
  --data-path dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run.pkl \
  --motion 0 \
  --max-frames 300
```

## PiPlus H0W 22DoF 冻结 BFM 的 AMP 专家动作

当前 22DoF H0W Stage2 使用的专家数据不是 LSE 23DoF 数据，而是：

```text
dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl
```

该数据由 `/home/sunteng/Project/HT_LAB_AMP_CHECKPOINT/piplus_nowaist_walk_run` 转换而来，机器人使用 `piplus_h0w`。推荐的完整播放和方向检查命令见 [播放PiPlus-H0W-22DoF-AMP专家动作.md](播放PiPlus-H0W-22DoF-AMP专家动作.md)。最小 MuJoCo 播放命令如下：

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion backward__walk_backward_edited_retargetted \
  --max-frames 434 \
  --viewer
```

确认专家数据与冻结 H0W BFM 的 22DoF/194维 AMP 合同：

```bash
uv run python -m humanoidverse.amp_stage2 \
  --dry-run --disable-lora --device cpu \
  --bfm-checkpoint 'huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint' \
  --expert-dataset dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml
```

## Play Checkpoint

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.tracking_inference \
  --model-folder 'huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)' \
  --data-path dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped.pkl \
  --simulator isaacsim \
  --robot piplus_lse \
  --motion-list 0 \
  --headless \
  --save-mp4 \
  --disable-dr \
  --disable-obs-noise \
  --device cpu
```

## Play Stage2 AMP Checkpoint

### 当前 GPU11 23DoF checkpoint（46200）

已从 GPU11 拉取并校验到本地的最新 23DoF frozen-BFM + command encoder AMP Stage2 checkpoint：

```text
checkpoint/piplus_lse_stage2_checkpoint_46200.pt
```

本地 checkpoint SHA256：
`412705dd6615a27ac15ab231546d1a0955d5b607a4649d9990ee2b32f9050fde`。
对应 metadata 为 `checkpoint/config.json`，内部迭代为 `46200`，action dimension 为 23，使用 5-direction balanced expert dataset、无 LoRA、无 action filter。

MuJoCo 无窗口快速播放：

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder checkpoint \
  --checkpoint checkpoint/piplus_lse_stage2_checkpoint_46200.pt \
  --bfm-checkpoint 'huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)/checkpoint' \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml \
  --expert-dataset dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl \
  --simulator mujoco \
  --device auto \
  --fixed-command -0.3 0.0 0.0 \
  --action-lowpass-alpha 1.0 \
  --headless --max-steps 1000 --log-every-steps 50 --no-realtime
```

该 checkpoint 训练时使用 raw BFM action，不要使用 `--action-lowpass-alpha 0.2` 复现训练行为；需要测试部署滤波闭环时，再单独显式指定滤波参数。

Isaac Sim + gamepad 播放 46200 checkpoint：

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder checkpoint \
  --checkpoint checkpoint/piplus_lse_stage2_checkpoint_46200.pt \
  --bfm-checkpoint 'huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)/checkpoint' \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H1W_LSE.yaml \
  --expert-dataset dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 1.0 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_gamepad_46200.mp4 \
  --show-viewer
```

历史固定 Isaac 速度跟踪最佳 checkpoint（推荐优先测试）：
`logs/amp_stage2_piplus_lse_1gpu_4096env_1m_backward14_resume19800_20260903/checkpoint_19900.pt`。
该版本在五指令固定 Isaac 测试中全部无终止：前向约 `0.391 m/s`、后向约 `-0.325 m/s`、横移约 `0.288 m/s`、偏航约 `0.557 rad/s`；对应训练使用负向 vx 编码倍率 `1.4` 和加权 14-motion expert 数据。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_1gpu_4096env_1m_backward14_resume19800_20260903 \
  --checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_backward14_resume19800_20260903/checkpoint_19900.pt \
  --expert-dataset dataset/pkl_cmd/piplus_lse_balanced5_plus_turn4_weighted.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 0.2 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_gamepad_19900.mp4 \
  --show-viewer
```

GPU11 最新物理平滑性候选 checkpoint（2026-09-08）：
`logs/amp_stage2_piplus_lse_1gpu_4096env_1m_physical_smooth34500_resume34500_20260907/checkpoint_40600.pt`。
该文件及同目录 `config.json` 已从 JumpServer GPU11 同步。checkpoint 大小为 50,745,334 bytes，SHA256 为
`32fa498b5aa989799f447c75becd9844de196a45ed97af628183c8fc9fe34509`，并已验证可加载且内部迭代为 40600。
该 lineage 从 `rewardtrack_upright/checkpoint_34500.pt` 恢复，并增加物理单位下的目标位置变化、目标位置二阶差分和目标力矩变化率约束。
历史速度跟踪候选 `speedtrack_forward/checkpoint_46500.pt` 仍保留；40600 优先用于验证动作平滑性，不能仅按迭代编号比较两个不同 lineage。
该 checkpoint 训练时启用了 JEPA/MPC；当前 `amp_stage2_play` 命令使用纯策略播放（不启用 MPC），部署前应单独做无 MPC 行为验证。

GPU11 最新部署滤波闭环实验 checkpoint（2026-09-09）：
`logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908/checkpoint_39900.pt`。
该 lineage 从 `checkpoint_34500.pt` 恢复，在训练中模拟部署 action low-pass `alpha=0.2`，并关闭 MPC/JEPA。当前日志显示动作/关节加速度更低，但速度跟踪、后退 wrong-way fraction 和 termination 仍未达到 34500 基线，暂作为滤波闭环研究候选，不作为已验证真机 checkpoint。

播放该 checkpoint 时必须使用与训练/部署一致的 action 低通滤波：`--action-lowpass-alpha 0.2`。播放脚本会先生成 BFM raw action，再滤波后送入环境；不加该参数会得到未滤波行为，不能与真机结果直接比较。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908 \
  --checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908/checkpoint_39900.pt \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 1.0 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_gamepad_39900.mp4 \
  --show-viewer
```

MuJoCo 快速复核：

## 无 LoRA Stage2 抬脚对照 checkpoint

最近同步的无 LoRA Stage2 checkpoint 为 `checkpoint_35700.pt`。该 lineage 冻结第一阶段 BFM，只训练 command encoder/AMP，并启用 feet-clearance/air-time 奖励与抬脚诊断；它不是 GPU2 当前的完整 BFM 训练进程。

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_1gpu_4096env_1m_feet_metrics_clearance2_airtime3_resume35400_20260909 \
  --checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_feet_metrics_clearance2_airtime3_resume35400_20260909/checkpoint_35700.pt \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 1.0 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_nolora_gamepad_35700.mp4 \
  --show-viewer
```

## LoRA BFM 输出微调 checkpoint（GPU11 双卡）

已从 `checkpoint_34500.pt` 恢复训练，并保持第一阶段 BFM 基座冻结，仅保存 actor LoRA 参数。当前已同步的首个可恢复 checkpoint 为 `checkpoint_34600.pt`；播放时需要使用与训练一致的 LoRA checkpoint metadata。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_2gpu_4096env_1m_lora_reg01_from34500_20260909 \
  --checkpoint logs/amp_stage2_piplus_lse_2gpu_4096env_1m_lora_reg01_from34500_20260909/checkpoint_34600.pt \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 1.0 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_lora_gamepad_34600.mp4 \
  --show-viewer
```

该 checkpoint 的 LoRA 参数已由 `amp_stage2_play` 自动加载；LoRA 训练未启用 action low-pass，播放命令显式使用 `--action-lowpass-alpha 1.0`，不要改成 `0.2`。

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908 \
  --checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908/checkpoint_39900.pt \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --simulator mujoco --device auto \
  --fixed-command -0.3 0.0 0.0 \
  --action-lowpass-alpha 0.2 \
  --headless --max-steps 1000 --log-every-steps 50 --no-realtime
```

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_1gpu_4096env_1m_physical_smooth34500_resume34500_20260907 \
  --checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_physical_smooth34500_resume34500_20260907/checkpoint_40600.pt \
  --expert-dataset dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --action-lowpass-alpha 0.2 \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_gamepad_40600.mp4 \
  --show-viewer
```

Isaac Sim 是训练时使用的物理后端，优先用于判断运动稳定性。显存较小时可让 Isaac Sim 使用 GPU、BFM/Stage2 网络使用 CPU；8GB 显存本机建议使用下面的命令。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2 \
  --checkpoint logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2/checkpoint_20200.pt \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --fixed-command 0.4 0.0 0.0 \
  --save-mp4 \
  --show-viewer
```

`--save-mp4` 使用 MuJoCo 离屏渲染当前 Isaac Sim 播放状态，默认写入
`<model-folder>/stage2_playback/<checkpoint>_<timestamp>.mp4`；可用 `--output path/to/file.mp4` 指定文件名。
录制 MP4 时脚本会自动跳过交互式 MuJoCo 窗口，避免同时创建窗口和离屏渲染上下文造成额外显存占用；`--headless` 仍可显式传入。
需要边录边显示时追加 `--show-viewer`（会创建第二个渲染上下文并增加显存占用）。

显存充足时可删除 `--policy-device cpu`。MuJoCo 后端用于快速调试；脚本会自动补充可碰撞平面，并复现位置目标 + PD 控制。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2 \
  --checkpoint logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2/checkpoint_20200.pt \
  --simulator mujoco \
  --device auto \
  --fixed-command 0.4 0.0 0.0
```

需要无窗口短测时增加 `--headless --max-steps 200 --log-every-steps 25 --no-realtime`。

### 手柄控制速度指令

`--gamepad` 使用与 `/home/sunteng/Project/HT_lab_hi/scripts/instinct_rl/play.py` 相同的 pygame 映射：左摇杆 Y 控制前进/后退，左摇杆 X 控制横移，右摇杆 X 控制偏航；默认轴号为 `0/1/3`，默认死区为 `0.08`。A（button 0）复位，B（button 1）退出。

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2 \
  --checkpoint logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2/checkpoint_20200.pt \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad \
  --save-mp4 \
  --output logs/instinct_rl/amp_stage2/stage2_playback_piplus_gamepad.mp4 \
  --show-viewer
```

录制期间仍可用手柄 button 1 退出；退出后脚本会关闭视频并打印保存路径。

手柄输入与 `--fixed-command` 互斥。若手柄轴号不同，可用 `--axis-lx`、`--axis-ly`、`--axis-rx` 覆盖；用 `--gamepad-debug` 检查原始轴值。pygame 未安装时执行 `python -m pip install pygame`。

## 导出 Stage2 command encoder ONNX

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python3 "/home/sunteng/Project/deployment/ROS2 Plugin/retarget/instinct_onboard/scripts/export_piplus_bfm_command_onnx.py" \
  --checkpoint logs/amp_stage2_piplus_lse_4gpu_4096env_1m_amp006_stand_velrecover_resume16200_20260803_retry2/checkpoint_20200.pt \
  --output huiying/stage2_command_encoder.onnx
```
