# PiPlus LSE Checkpoint 播放说明

## Checkpoint

目标模型目录：

```text
/home/sunteng/Project/HT_BFM/huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)
```

该目录结构完整，包含：

- `config.json` 和 `config.yaml`
- `checkpoint/model/model.safetensors`
- `checkpoint/model/config.json`
- 已导出的 ONNX 模型

模型已在 CPU 上成功加载，类型为 `FBcprAuxModel`，动作维度为 23，latent 维度为 256。因此 checkpoint 本身有效，可以用于推理播放。

## 推荐播放方式

该 checkpoint 使用 Isaac Sim 训练，推荐在带 CUDA GPU 的 `env_isaaclab` 环境中无窗口运行并保存 MP4：

运行前先确认 NVIDIA 驱动和 PyTorch CUDA 均可用：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available()"
```

只有两个命令都成功后再启动 Isaac Sim。

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
  --disable-obs-noise
```

`--motion-list` 后可更换动作编号。播放全部动作时使用：

```bash
--motion-list motion_all
```

生成的视频为左侧专家动作、右侧策略动作的对比视频，输出到：

```text
huiying/bfmzero-piplus-lse-isaac-20260715_143758(1)/tracking_inference/tracking_0.mp4
```

更换 `--motion-list` 后，文件名中的动作编号也会相应改变。ONNX 文件保存在 checkpoint 目录下的 `exported/`。

如需打开 Isaac Sim 交互窗口而不保存视频，可将 `--headless --save-mp4` 替换为 `--no-headless --no-save-mp4`。

## 当前限制

### 动作数据路径

checkpoint 配置中记录的旧路径：

```text
humanoidverse/data/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped.pkl
```

在当前仓库中不存在，因此必须显式传入：

```text
dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped.pkl
```

### Isaac Sim URDF 路径错误

PiPlus LSE 的 URDF、XML 和网格已复制到项目内：

```text
/home/sunteng/Project/HT_BFM/humanoidverse/data/robots/piplus/PiPlus_S_12L8A0G2H1W_LSE_260611/
```

Isaac Sim 使用的 URDF 为：

```text
humanoidverse/data/robots/piplus/PiPlus_S_12L8A0G2H1W_LSE_260611/urdf/PiPlus_S_12L8A0G2H1W_LSE_260611.urdf
```

确认文件存在：

```bash
test -f humanoidverse/data/robots/piplus/PiPlus_S_12L8A0G2H1W_LSE_260611/urdf/PiPlus_S_12L8A0G2H1W_LSE_260611.urdf && echo OK
```

checkpoint 的解析配置和仓库机器人配置均已改为使用该项目内路径，不再依赖外部 `ht_urdf` 包中的 PiPlus LSE 资产。

### PhysX simulation view 创建失败

以下错误表示 CUDA/驱动后端不可用，不是 checkpoint 或 URDF 问题：

```text
Exception: Failed to create simulation view backend
AttributeError: 'NoneType' object has no attribute 'create_articulation_view'
```

先退出所有 Isaac Sim/Python 进程，再检查 `nvidia-smi`。如果 `nvidia-smi` 无法连接驱动、`torch.cuda.is_available()` 为 `False`，需要重启主机以恢复 NVIDIA 设备：

```bash
sudo reboot
```

重启后重新执行本节开头的 GPU 预检。不要在 CUDA 不可用时继续运行 `--simulator isaacsim`；后续的 `NoneType` 报错只是 PhysX backend 创建失败后的连锁错误。

### MuJoCo 暂不可直接播放

当前 PiPlus LSE MuJoCo XML 能够加载机器人关节和网格，但执行器数量为 `nu=0`。`humanoidverse.simulator.mujoco.MuJoCo` 要求 23 个训练关节均有 actuator，因此使用 `--simulator mujoco` 会报错：

```text
ValueError: Invalid MuJoCo actuator mapping; missing actuators for DOFs: ...
```

若需使用 MuJoCo，需要先为 PiPlus LSE XML 添加与 `robot.dof_names` 一一对应的 actuator，并验证控制方式、力矩范围和 PD 参数。

### 播放长度参数

当前 `humanoidverse/tracking_inference.py` 内部将 `episode_len` 强制设置为 `2000`，因此命令行传入的 `--episode-len` 暂时不会生效。

## 验证状态

- checkpoint 权重加载：通过
- PiPlus LSE 机器人配置解析：通过
- 动作数据存在：通过
- ONNX 导出：通过
- MuJoCo 环境启动：失败，缺少 actuator
- Isaac Sim 完整播放：当前验证环境无可用 CUDA GPU，需在 GPU 环境执行上述命令确认
