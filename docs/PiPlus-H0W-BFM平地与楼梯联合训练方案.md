# PiPlus H0W 22DoF BFM 平地与楼梯联合训练方案

## 1. 目标

使用同一个 PiPlus H0W 22DoF BFM policy 同时学习：

- 平地动作数据：
  `dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/piplus_h0w_lafan_10s-clipped.pkl`
- 上楼梯动作数据：
  `dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w`
- 与上楼梯动作对齐的楼梯地形：
  `dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/stairs_0w/motion_matched_terrain`

目标不是简单地把两份动作拼在一起，而是保证每次 reset 时，动作、地形、机器人初始位姿和地形观测严格匹配。

## 2. 总体训练结构

```text
平地动作 PKL ─────────┐
                     ├──> 统一 MotionLib 动作库
楼梯 NPZ -> PKL ──────┘            │
                                   ├──> 联合采样 (motion_id, terrain_id)
楼梯 metadata.yaml ────────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       │                                       │
                   平地环境池                              楼梯环境池
                   plane motion                         stair motion
                       └───────────────────┬───────────────────┘
                                           │
                         本体观测 + reference + terrain scan
                                           │
                                  同一个 22DoF BFM policy
```

推荐从已经训练好的 H0W 平地 BFM checkpoint 开始微调，不建议从零开始混合训练。

## 3. 当前代码不能直接联合训练的原因

### 3.1 当前训练只接收一个动作文件

H0W 的动作文件在 `humanoidverse/train.py` 的 `_get_robot_training_settings()` 中指定：

```python
"lafan_tail_path": (
    "humanoidverse/data/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/"
    "piplus_h0w_lafan_10s-clipped.pkl"
)
```

环境构建时，`humanoidverse/agents/envs/humanoidverse_isaac.py` 又执行：

```python
cfg.robot.motion.motion_file = self.lafan_tail_path
```

现有接口没有“平地动作库 + 楼梯动作库 + 配对地形”的概念。

### 3.2 动作和地形当前是独立分配的

`humanoidverse/envs/legged_robot_motions/legged_robot_motions.py` 当前独立随机动作：

```python
self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
```

`humanoidverse/envs/base_task/base_task.py` 则根据环境编号和 terrain level 分配地形。两者没有映射关系。

如果只把楼梯动作加入总 PKL 并启用随机地形，会出现：

- 楼梯动作在平地环境中播放；
- 平地动作在楼梯环境中播放；
- 楼梯动作与错误的楼梯相位或起点配对。

这是联合训练前必须解决的问题。

### 3.3 当前 BFM 配置实际使用平面地形

`humanoidverse/config/exp/bfm_zero_piplus/bfm_zero_piplus.yaml` 使用：

```yaml
- /terrain: terrain_locomotion_plane
```

而 `humanoidverse/config/terrain/terrain_locomotion_plane.yaml` 中为：

```yaml
terrain:
  mesh_type: plane
  measure_heights: false
```

所以当前训练不会自动加载 `stairs_0w/motion_matched_terrain` 中的 STL。

### 3.4 Actor 看不到前方楼梯

当前 actor 主要接收关节状态、IMU、上一帧动作、历史状态和 reference body state，没有前方 terrain height scan。

Isaac Sim 中虽然创建了 height scanner，但扫描范围只有约 `0.05 m × 0.05 m`，相当于脚下单点，而且当前没有被加入 BFM wrapper 输出的 `state`。

因此，即使动作和地形正确配对，policy 也很难提前感知下一阶台阶。

## 4. 数据准备

### 4.1 将楼梯 NPZ 转成 MotionLib PKL

`stairs_0w` 中的原始 NPZ 主要包含：

```text
joint_pos       [T, 22]
joint_names     [22]
base_pos_w      [T, 3]
base_quat_w     [T, 4]
framerate       30
```

平地文件则是 MotionLib 聚合 PKL。楼梯 NPZ 必须先使用 PiPlus H0W 的 canonical MJCF/URDF，转换为与平地 PKL相同的 MotionLib schema，不能直接把 NPZ 路径交给现有 BFM 训练入口。

建议生成：

```text
dataset/PiPlus_S_12L8A0G2H0W_lafan_dataset_20260629/
├── piplus_h0w_flat_stairs_10s.pkl
└── piplus_h0w_flat_stairs_terrain.yaml
```

### 4.2 使用 sidecar metadata 保存地形配对

不要把 STL 几何直接塞进 motion tensor。使用 YAML 保存动作和地形的关系：

```yaml
default:
  terrain_type: plane
  sampling_weight: 1.0

motions:
  stairs_motion_01:
    terrain_type: stairs
    terrain_id: 1
    terrain_file: stairs_0w/motion_matched_terrain/stairs_motion_01.stl
    sampling_weight: 20.0
    local_translation: [0.0, 0.0, 0.0]
    local_rotation_wxyz: [1.0, 0.0, 0.0, 0.0]
```

对未显式列出的 LAFAN 平地动作应用 `default: plane`。

只有 6 条楼梯动作，而平地 clip 数量较多，因此必须按类别控制采样比例，不能按照动作条数均匀采样。

## 5. 配置改造

建议新增文件，不直接破坏现有平地训练配置：

```text
humanoidverse/config/exp/bfm_zero_piplus/
└── bfm_zero_piplus_h0w_flat_stairs.yaml

humanoidverse/config/terrain/
└── terrain_piplus_h0w_flat_stairs.yaml

humanoidverse/config/obs/
└── bfm_zero_perceptive_obs.yaml
```

联合训练配置至少包含：

```yaml
robot:
  motion:
    motion_file: dataset/.../piplus_h0w_flat_stairs_10s.pkl
    terrain_metadata_file: dataset/.../piplus_h0w_flat_stairs_terrain.yaml

env:
  config:
    flat_env_ratio: 0.7
    stair_env_ratio: 0.3

terrain:
  mesh_type: trimesh
  terrain_types: [flat, motion_matched_stairs]
  terrain_proportions: [0.7, 0.3]
```

还需要在 `humanoidverse/train.py` 的 H0W 配置分支中增加新的 experiment/config 入口，避免构建环境时被原来的单一 `lafan_tail_path` 覆盖。

## 6. 动作与地形联合采样

### 6.1 推荐使用固定环境池

不建议在每次 reset 时动态替换 Isaac Sim 全局地形。启动训练时预先建立固定 tile：

```text
env 0～716       平地 tile
env 717～1023    楼梯 tile
```

每个环境拥有固定的 `env_terrain_type` 和 `env_terrain_id`，reset 时只从兼容动作集合中采样：

```python
flat_env_ids = env_ids[self.env_terrain_type[env_ids] == TERRAIN_FLAT]
stair_env_ids = env_ids[self.env_terrain_type[env_ids] == TERRAIN_STAIRS]

self.motion_ids[flat_env_ids] = self._motion_lib.sample_from_group(
    "flat", len(flat_env_ids)
)

self.motion_ids[stair_env_ids] = self._motion_lib.sample_matching_terrain(
    self.env_terrain_ids[stair_env_ids]
)
```

主要修改：

```text
humanoidverse/envs/legged_robot_motions/legged_robot_motions.py
```

建议新增：

```python
def _load_motion_terrain_metadata(self):
    ...

def _build_compatible_motion_ids(self):
    ...

def _sample_motion_ids_for_envs(self, env_ids):
    ...
```

然后在 `_resample_motion_time_and_ids()` 中替换无条件的 `sample_motions()`。

### 6.2 增加运行时强校验

在 debug 模式下，每次 reset 验证：

```python
assert motion_terrain_id[motion_id] == env_terrain_id[env_id]
```

平地动作对应 `plane`，楼梯动作必须对应其指定的楼梯地形或参数。发现不匹配应立即报错，不能静默继续训练。

## 7. 坐标系与 reset 对齐

当前 reference motion 使用：

```python
motion_res = self._motion_lib.get_motion_state(
    self.motion_ids,
    motion_times,
    offset=self.env_origins,
)
```

因此动作和楼梯必须共享同一个环境原点：

```text
world_stair_pose = env_origin + terrain_local_pose
world_motion_pose = env_origin + motion_local_pose
```

不能分别对动作和 STL 自动居中。需要保证：

- 楼梯起点位于 motion metadata 指定的局部位置；
- root reset 使用 motion root 加同一个 `env_origin`；
- 楼梯方向与机器人参考运动方向一致；
- 随机 motion start 时，机器人仍处于相应台阶附近，而不是悬空或穿模。

楼梯专项训练初期建议限制随机起始帧，优先从动作起点附近 reset。

## 8. 地形感知观测

### 8.1 Height scan 范围

建议将单点 height scanner 改成机器人前方局部网格：

```yaml
terrain_scan:
  forward_range: [-0.20, 1.00]
  lateral_range: [-0.45, 0.45]
  resolution: 0.10
  height_clip: [-0.5, 0.5]
```

约产生 `13 × 10 = 130` 维高度观测。实际前进轴需根据 PiPlus body frame 验证，不能默认世界坐标轴。

高度使用 root-relative 表示：

```python
terrain_height = ray_hit_z - robot_root_z
terrain_height = torch.clamp(terrain_height, -0.5, 0.5)
```

### 8.2 加入 actor 和 critic

在新 observation 配置中加入：

```yaml
obs:
  obs_dict:
    actor_obs:
      - base_ang_vel
      - projected_gravity
      - dof_pos
      - dof_vel
      - actions
      - history_actor
      - max_local_self
      - terrain_height_scan

    critic_obs:
      - max_local_self
      - terrain_height_scan

  obs_dims:
    - terrain_height_scan: 130
```

在 `humanoidverse/agents/envs/humanoidverse_isaac.py` 中将其暴露给 BFM：

```python
observation = {
    "state": g1env_state,
    "terrain": raw_obs["terrain_height_scan"],
    "privileged_state": privileged_state,
}
```

在 `humanoidverse/train.py` 中修改网络 input filter：

```python
# Actor
key=["state", "terrain", "last_action", "history_actor"]

# Forward/Critic/Aux critic
key=["state", "terrain", "privileged_state", "last_action", "history_actor"]
```

Backward 和 discriminator 可以继续使用：

```python
key=["state", "privileged_state"]
```

避免 BFM 行为表示只通过地形标签区分动作。

### 8.3 专家 buffer 必须同步增加 terrain

当前 `load_expert_trajectories_from_motion_lib()` 直接根据 MotionLib 构造专家状态，没有仿真地形观测。

如果 actor/forward 网络加入 `terrain`，专家轨迹也必须生成匹配的高度扫描：

```python
ep["observation"]["terrain"] = compute_reference_terrain_scan(
    motion_id=i,
    root_pos=motion_res["root_pos"],
    terrain_metadata=terrain_metadata,
)
```

否则会出现 expert buffer 与在线 replay buffer 观测维度或语义不一致。

## 9. 终止条件和 replay buffer

当前 `legged_motions.yaml` 中跌倒、低高度、重力异常和 motion far 终止均关闭。楼梯训练建议启用：

```yaml
termination:
  terminate_by_contact: true
  terminate_by_gravity: true
  terminate_by_low_height: true
  terminate_when_motion_far: true

termination_scales:
  termination_min_base_height: 0.35
  termination_gravity_x: 0.75
  termination_gravity_y: 0.75
  termination_motion_far_threshold: 0.8
```

启用真实 termination 后，需要修复 trajectory buffer 的 episode 边界。当前 `humanoidverse/train.py` 使用：

```python
end_key="truncated"
```

建议显式存储：

```python
done = terminated | truncated
```

并让 trajectory buffer 使用 `done` 作为 `end_key`。否则跌倒终止后的新 episode 可能与旧 episode 被当成同一条轨迹切片。

## 10. 推荐训练流程

### 阶段 A：动作与地形对齐验证

先使用一个动作、一个楼梯地形和约 32 个环境，不进行正式训练。

验收条件：

- reset 后脚底不穿台阶；
- root 与楼梯起点坐标一致；
- 每个 motion ID 对应正确 terrain ID；
- reference marker 与机器人沿相同方向上楼；
- terrain scan 能随台阶变化；
- 连续运行 500～1000 step 不出现配对变化或坐标漂移。

### 阶段 B：楼梯专项微调

从 H0W 平地 BFM checkpoint 加载模型，但 observation shape 改变后不要恢复旧 replay buffer。

建议：

```text
平地动作比例：30%
楼梯动作比例：70%
环境数：256～512
楼梯尺寸：固定 7 cm 高、10 cm 深
push DR：关闭或显著降低
motion start：优先从楼梯动作起点附近采样
```

### 阶段 C：平地与楼梯联合训练

当固定楼梯成功率稳定后：

```text
平地动作比例：70%
楼梯动作比例：30%
环境数：1024
```

保留足够的平地 rehearsal，减少平地能力的灾难性遗忘。

### 阶段 D：楼梯参数随机化

固定楼梯收敛后再逐步增加：

```text
step height: 0.06～0.09 m
step depth:  0.09～0.13 m
friction:    0.6～1.4
起点偏移:   ±0.02 m
楼梯偏航:   ±3°
```

第一轮训练不应同时启用大范围楼梯尺寸、摩擦、起点和姿态随机化。

## 11. 推荐采样比例

| 训练阶段 | 平地 | 楼梯 |
| --- | ---: | ---: |
| 楼梯专项 | 30% | 70% |
| 联合稳定 | 70% | 30% |
| 最终泛化 | 60% | 40% |

比例应按动作类别控制，而不是按 motion clip 数量自然形成。

## 12. 评估指标

必须分别报告平地和楼梯结果：

```text
flat/tracking_error
flat/fall_rate
stairs/tracking_error
stairs/success_rate
stairs/fall_rate
stairs/foot_penetration
stairs/foot_clearance
stairs/foot_slip
stairs/pelvis_height_error
motion_terrain_mismatch_count
```

其中：

```text
motion_terrain_mismatch_count == 0
```

是开始正式训练前的硬性条件。

## 13. 实现顺序

建议严格按以下顺序开发：

1. 把 6 条楼梯 NPZ 转为 PiPlus H0W MotionLib PKL。
2. 建立 `motion_key -> terrain_id/local_transform` sidecar metadata。
3. 新增平地与楼梯 terrain tiles，并验证 STL/参数化楼梯可碰撞。
4. 实现固定环境池和 motion-terrain 联合采样。
5. 验证 reset、参考动作、楼梯和机器人坐标对齐。
6. 扩大 height scanner，并将 terrain scan 加入在线 observation。
7. 为 expert buffer 合成对应的 terrain scan。
8. 修复 `terminated | truncated` 的 trajectory episode 边界。
9. 完成阶段 A 的无训练 smoke test。
10. 从平地 checkpoint 开始阶段 B～D 训练。

## 14. 最关键的三项修改

1. 实现 motion ID 与 terrain ID 的强制联合采样，禁止错误配对。
2. 给 actor 增加前方 terrain height scan，并同步修改专家 buffer 的观测构造。
3. 将楼梯 NPZ 转成统一 MotionLib PKL，使用 sidecar YAML 保存每条动作对应的地形和坐标变换。

仅完成数据拼接或仅切换到 `trimesh`，都不足以得到正确的平地与楼梯联合 BFM 训练。

## 15. 已实现的 checkpoint 兼容训练入口

当前兼容版本保持原 BFM observation 和网络输入维度不变，通过“固定地形环境池 + 严格动作前缀配对”继续训练，因此可以加载原 H0W 模型权重。

先生成联合 MotionLib 数据：

```bash
conda run -n env_isaaclab python \
  data_process/build_piplus_h0w_flat_stairs_bfm.py --overwrite
```

默认生成 76 条平地 expert，以及 6 条楼梯 expert 各重复 19 次得到的 114 条楼梯 expert，使专家 buffer 约为 40% 平地、60% 楼梯。重复条目只用于平衡采样，不修改原始动作。

然后从给定 H0W 模型权重启动联合训练：

```bash
conda run -n env_isaaclab python -m humanoidverse.train \
  --robot PiPlus_S_12L8A0G2H0W \
  --flat_stairs \
  --warm_start_model_from "/home/sunteng/Project/HT_BFM/huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)" \
  --no_resume_replay_buffer
```

该目录只包含 `checkpoint/model/model.safetensors`，没有 `train_status.json`、agent optimizer 和 replay buffer，所以不能使用完整的 `--resume_from`。`--warm_start_model_from` 会严格加载原模型权重，同时重新创建 optimizer、replay buffer 和训练计数；这是该 checkpoint 实际可执行的续训方式。

本兼容版本没有给 actor 增加 terrain scan，以保持 checkpoint 第一层权重形状完全一致。它适合训练与六份专家动作严格匹配的固定楼梯；如果后续需要泛化到未知台阶尺寸，应另做 terrain observation 扩维和权重迁移。

## 16. 迁移契约与验证记录

| 契约 | 来源 | 本次目标 | 复用方式 | 修改 |
| --- | --- | --- | --- | --- |
| 机器人 | `PiPlus_S_12L8A0G2H0W.yaml` | PiPlus H0W 22DoF | unchanged | 无形态修改 |
| 模型动作维度 | checkpoint `init_kwargs.json` | 22 | unchanged | 严格加载旧权重 |
| 控制 | P 控制、action scale 0.25、200 Hz physics/4 decimation | 保持一致 | unchanged | 无 |
| 电机参数 | `HT_lab_pipeline/source/HT_lab/HT_lab/actuators/HT_motor.py`、`HT_motor_cfg.py`、`assets/PiPlus_S_12L8A0G2H0W.py` | 与已有 H0W 配置核对 | unchanged | 未复制或猜测参数 |
| 平地数据 | 76 条 MotionLib clip | `flat__*` | adapted | 仅增加 key 前缀 |
| 楼梯数据 | 6 条 30 Hz、22DoF retarget motion | MotionLib schema | adapted | 使用现有 `convert_gmr_lafan` 转换函数 |
| 专家比例 | 楼梯仅 6/82 | 平地 40%、楼梯 60% | configured | 每条楼梯 expert 使用 19 个采样 key |
| 地形 | 六个动作匹配 STL | Isaac Lab sub-terrain | adapted | 新增 STL 薄适配器 |
| 动作地形关系 | 原先独立随机 | 固定环境列严格配对 | parameterized | 按 motion key prefix 条件采样 |
| observation/model | 原 checkpoint 输入 | 完全保持 | unchanged | 本阶段不增加 terrain scan |
| checkpoint | 只有 model 权重 | model-only warm start | adapted | 新 optimizer、buffer 和计数 |

已完成验证：

- Hydra 配置解析为 22DoF、`trimesh` 和 7 类地形；
- 联合数据共 190 条：76 平地、114 楼梯；
- 所有动作值有限，且只有一个统一的 22 关节顺序；
- checkpoint action dimension 为 22；
- 新增 Python 文件通过 Ruff；
- 修改文件通过 Python compile；
- 最终训练命令能够启动 Isaac Sim 并解析 observation dimensions。

本机 smoke test 在创建机器人 articulation 时停止，原因是 RTX 5060（CUDA capability `sm_120`）不受当前 PyTorch binary 支持，错误为 `no kernel image is available for execution on the device`。这不是数据或联合地形配置错误；正式训练需要使用支持 `sm_120` 的 PyTorch/Isaac 环境，或者在项目原来可运行的训练服务器上执行上述命令。
