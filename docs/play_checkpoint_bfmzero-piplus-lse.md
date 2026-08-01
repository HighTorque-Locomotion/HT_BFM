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

Isaac Sim 是训练时使用的物理后端，优先用于判断运动稳定性。显存较小时可让 Isaac Sim 使用 GPU、BFM/Stage2 网络使用 CPU；8GB 显存本机建议使用下面的命令。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2 \
  --checkpoint logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2/checkpoint_1700.pt \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --fixed-command 0.4 0.0 0.0
```

显存充足时可删除 `--policy-device cpu`。MuJoCo 后端用于快速调试；脚本会自动补充可碰撞平面，并复现位置目标 + PD 控制。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python -m humanoidverse.amp_stage2_play \
  --model-folder logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2 \
  --checkpoint logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2/checkpoint_1700.pt \
  --simulator mujoco \
  --device auto \
  --fixed-command 0.4 0.0 0.0
```

需要无窗口短测时增加 `--headless --max-steps 200 --log-every-steps 25 --no-realtime`。

### 手柄控制速度指令

`--gamepad` 使用与 `/home/sunteng/Project/HT_lab_hi/scripts/instinct_rl/play.py` 相同的 pygame 映射：左摇杆 Y 控制前进/后退，左摇杆 X 控制横移，右摇杆 X 控制偏航；默认轴号为 `0/1/3`，默认死区为 `0.08`。A（button 0）复位，B（button 1）退出。

```bash
python -m humanoidverse.amp_stage2_play \
  --model-folder logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2 \
  --checkpoint logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2/checkpoint_1700.pt \
  --simulator isaacsim \
  --device cuda:0 \
  --policy-device cpu \
  --gamepad
```

手柄输入与 `--fixed-command` 互斥。若手柄轴号不同，可用 `--axis-lx`、`--axis-ly`、`--axis-rx` 覆盖；用 `--gamepad-debug` 检查原始轴值。pygame 未安装时执行 `python -m pip install pygame`。

## 导出 Stage2 command encoder ONNX

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

python3 "/home/sunteng/Project/deployment/ROS2 Plugin/retarget/instinct_onboard/scripts/export_piplus_bfm_command_onnx.py" \
  --checkpoint logs/instinct_rl/amp_stage2/amp_stage2_piplus_lse_4gpu_4096env_1m_20260801_loco11_resume1400_v2/checkpoint_1700.pt \
  --output huiying/stage2_command_encoder.onnx
```
