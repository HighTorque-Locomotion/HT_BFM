# 播放 PiPlus H0W 楼梯专家动作

以下命令都从项目根目录执行。目标机器人是 `PiPlus_S_12L8A0G2H0W`，动作维度为 22 DoF。

```bash
cd /home/sunteng/Project/HT_BFM
conda activate env_isaaclab
```

## 同时播放动作和匹配楼梯

下面的命令读取原始 `*_retargetted.npz`、根据 `metadata.yaml` 自动找到对应 STL，并在 MuJoCo 中同时显示机器人和楼梯：

```bash
python scripts/play_piplus_stairs_motion.py \
  --motion-file dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/stairs_climbing_up_start_R_20260817_PiPlus_S_12L8A0G2H0W_retargetted.npz
```

播放慢跑上楼：

```bash
python scripts/play_piplus_stairs_motion.py \
  --motion-file dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/stairs_climbing_jog_up_start_R_20260817_PiPlus_S_12L8A0G2H0W_retargetted.npz
```

播放普通下楼循环：

```bash
python scripts/play_piplus_stairs_motion.py \
  --motion-file dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/stairs_climbing_down_loop_R_20260817_PiPlus_S_12L8A0G2H0W_retargetted.npz
```

程序默认使用：

```text
dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/motion_matched_terrain/metadata.yaml
```

如需检查文件映射和维度但不打开窗口：

```bash
python scripts/play_piplus_stairs_motion.py --validate-only
```

## 无窗口保存 MP4

```bash
python scripts/play_piplus_stairs_motion.py \
  --motion-file dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/stairs_climbing_up_start_R_20260817_PiPlus_S_12L8A0G2H0W_retargetted.npz \
  --output outputs/motion_videos/piplus_h0w_stairs_up.mp4
```

这条命令已完成实际渲染验证，输出为 180 帧、30 FPS、6 秒、960×540 的 H.264 视频：

```text
outputs/motion_videos/piplus_h0w_stairs_up.mp4
```

无窗口渲染需要宿主机 EGL/OpenGL；如果隔离容器看不到 GPU 驱动，请在正常宿主机终端执行命令。

可用参数：

```bash
python scripts/play_piplus_stairs_motion.py --help
```

常用选项包括 `--start`、`--max-frames`、`--stride`、`--width` 和 `--height`。

## 播放聚合 PKL 中的普通动作

仓库现有的普通动作播放器是 `humanoidverse.visualize_motion`：

```bash
python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/piplus_h0w_lafan_10s-clipped.pkl \
  --motion 0 \
  --max-frames 300
```

无窗口保存视频：

```bash
python -m humanoidverse.visualize_motion \
  --robot piplus_h0w \
  --data-path dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/piplus_h0w_lafan_10s-clipped.pkl \
  --motion 0 \
  --viewer false \
  --output outputs/motion_videos/piplus_h0w_motion_0.mp4
```

该入口只显示动作和机器人原模型，不会加载本次生成的楼梯 STL。

## 关于 `amass_visualize.py`

`/home/sunteng/Project/instinctlab/scripts/amass_visualize.py` 当前硬编码使用 G1 29 DoF、AMASS 数据和 Isaac Lab 场景，不能直接播放 PiPlus H0W 22 DoF 动作。因此检查这批楼梯动作时，应使用本文的 `scripts/play_piplus_stairs_motion.py`。

## 重新拟合楼梯

如果动作或地形被更新，先重新生成 STL 和元数据：

```bash
python data_process/generate_motion_matched_stairs.py --overwrite
```

当前自动拟合的共享楼梯规格为台阶高 `0.07 m`、深 `0.10 m`。
