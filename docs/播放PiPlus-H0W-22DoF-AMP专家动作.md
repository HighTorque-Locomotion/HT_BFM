# 播放 PiPlus H0W 22-DoF AMP 专家动作

这组命令播放的是冻结 H0W BFM Stage2 实际使用的专家轨迹，不是策略输出。数据来自：

```text
dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl
```

该文件由 `/home/sunteng/Project/HT_LAB_AMP_CHECKPOINT/piplus_nowaist_walk_run` 转换而来，包含 15 条 22-DoF motion。转换后 AMP 使用的 joint order、四元数格式和 194 维特征合同已经通过 dry-run 验证。

## 1. 列出可播放动作

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab

uv run python - <<'PY'
import joblib

path = "dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl"
data = joblib.load(path)
for index, (key, motion) in enumerate(data.items()):
    print(f"{index:02d}  {key:60s} frames={len(motion['dof']):3d} fps={motion['fps']}")
PY
```

推荐重点检查这些 motion：

```text
forward__walk_forward_edited_retargetted
backward__walk_backward_edited_retargetted
lateral__youyi_play_retargetted
lateral__zuoyi_play_retargetted
turn__yuandiyouzhuan_modified_retargetted
turn__yuandizuozhuan_modified_retargetted
```

数据集中没有独立的 stand motion。

## 2. MuJoCo 可视化播放

先播放后退动作，确认机器人是否确实沿自身前向的反方向移动：

```bash
uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion backward__walk_backward_edited_retargetted \
  --start 0 \
  --max-frames 434 \
  --stride 1 \
  --viewer
```

分别播放前进、左侧移、右侧移、左转和右转：

```bash
uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion forward__walk_forward_edited_retargetted \
  --max-frames 434 \
  --viewer

uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion lateral__zuoyi_play_retargetted \
  --max-frames 100 \
  --viewer

uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion turn__yuandiyouzhuan_modified_retargetted \
  --max-frames 100 \
  --viewer
```

无窗口录制 MP4：

```bash
uv run python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion backward__walk_backward_edited_retargetted \
  --max-frames 434 \
  --no-viewer \
  --output logs/motion_videos/piplus_h0w_backward_expert.mp4
```

## 3. Isaac Sim 参考动作播放

这个命令使用项目自己的 Isaac Sim 参考动作脚本，加载 H0W 22-DoF Hydra robot，并把专家 pose 直接写入仿真。它不运行策略：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.visualize_motion_isaacsim \
  --robot piplus_h0w \
  --data-path dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --motion backward__walk_backward_edited_retargetted \
  --headless \
  --device cuda:0 \
  --no-realtime \
  --max-frames 434
```

将 `--motion` 替换为上面列出的 forward/lateral/turn key，即可逐条检查方向。

## 4. 方向和格式自动检查

```bash
uv run python - <<'PY'
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

path = "dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl"
data = joblib.load(path)
for key, motion in data.items():
    root = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    vel_world = np.zeros_like(root)
    vel_world[:-1] = np.diff(root, axis=0) * float(motion["fps"])
    local_vel = Rotation.from_quat(quat_xyzw).inv().apply(vel_world)
    print(
        f"{key:60s} dof={motion['dof'].shape[1]} frames={len(root):3d} "
        f"mean_local_vx={local_vel[:, 0].mean():+.3f} "
        f"mean_local_vy={local_vel[:, 1].mean():+.3f}"
    )
PY
```

预期检查：

- `backward__walk_backward_edited_retargetted` 的 `mean_local_vx` 应为负值；
- `forward__walk_forward_edited_retargetted` 的 `mean_local_vx` 应为正值；
- 左右侧移动的 `mean_local_vy` 符号应相反；
- 所有动作 `dof` 都应为 22 列，`root_rot` 为归一化的 XYZW。

## 5. 与 Stage2 冻结 BFM 合同复核

```bash
uv run python -m humanoidverse.amp_stage2 \
  --dry-run \
  --disable-lora \
  --device cpu \
  --bfm-checkpoint 'huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint' \
  --expert-dataset dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml
```

成功输出应包含：

```text
action_dim: 22
policy_joint_count: 22
expert_motion_count: 15
expert_feature_dim: 194
bfm_trainable_parameter_count: 0
robot_type: PiPlus_S_12L8A0G2H0W
stage1_robot_contract.validated: true
```

如果第 2 节或第 3 节的后退动作方向不对，先不要训练；检查 NPZ 的 `base_quat_w`、转换后的 `root_rot` 和 motion key 对应关系。
