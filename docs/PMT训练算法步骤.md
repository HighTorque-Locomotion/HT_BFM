# PMT 训练算法步骤(Perceptive BFM)

> 来源:arXiv:2606.08059v2《Perceptive Behavior Foundation Model: Adapting Human
> Motion Priors to Robot-Centric Terrain》,论文正文 Sec. III 与附录 A/B/D。
> 本文档只整理**训练算法**,按阶段写成步骤;架构图见论文 Fig. 7,部署协议见附录 I。

---

## 0. 总览与核心接口约定

**核心问题(operator–environment mismatch):** 人类动作参考 `m_raw` 表达行为意图
与风格,但不包含机器人在自身地形上所需的落足点、摆腿间隙、重心高度与接触时机。
PMT 的做法是:离线把 `m_raw` 合成为地形适配参考 `m_tcrs` 作为**监督信号**,而部署时
策略仍然接收**原始参考**作为命令。

**关键约定:**

- 动作是相对命令帧的**残差关节位置目标**:

  ```
  q_t^pd = q_t^cmd + a_t,    a_t = π_θ(o_t^prop, o_t^vis, m_{t:t+H}^raw)
  ```

  其中教师命令帧 `q^cmd = q^tcrs`,学生命令帧 `q^cmd = q^raw`。同一个残差语义
  在两个命令帧下含义不同,这正是第 3 阶段需要目标帧对齐的原因。
- TCRS 只在**离线训练期**使用;部署时策略只接收原始参考 + 本体感受 + 机载地形观测。
- 特权信息(真值身体位姿、全局锚点、基座线速度)只给教师、critic 和辅助损失;
  部署学生 actor 读的是**估计器头的 detached 输出**,不接触真值。

**四阶段流水线(Fig. 2 / Algorithm 1):**

| 阶段 | 名称 | 作用 |
| --- | --- | --- |
| 1 | TCRS 数据生成 | 在采样地形上合成 paired raw/TCRS 参考(离线) |
| 2 | 盲教师 PPO | 用特权观测组跟踪 TCRS 参考 |
| 3 | Raw-reference 蒸馏 | 把教师目标变换到学生 raw 命令帧后模仿 |
| 4 | 视觉 PPO 微调 | 用 raw 命令 + 高度图微调可部署视觉学生 |

```
Algorithm 1: PMT training
1: Input: 原始动作片段 M, 地形生成器 G, 机器人模型, 高度扫描器
2: for clip m ∈ M and terrain τ ∼ G do
3:     合成 m^tcrs = S_TCRS(m, τ)          # Algorithm 2
4:     导出 paired 参考 (m^raw, m^tcrs, τ)
5: end for
6: 用 PPO 训练盲教师 π_T 跟踪 m^tcrs
7: 用 Eq. (10) 从 π_T 计算目标帧标签
8: 用 raw-reference 观测蒸馏 identity-gated 视觉学生 π_S
9: 在 raw-reference 命令 + 高度图观测下用 PPO 微调 π_S
10: Output: 可部署 Perceptive BFM 策略 π_S
```

---

## 1. 训练准备:语料、地形与仿真配置

1. **动作语料:** 来自多个受试者的 locomotion + dance 类 mocap 片段
   (走、跑、转身、侧步、表达性走舞序列)。
2. **几何增强:** 每个 base clip 在采样地形上做随机平面摆放 + 随机偏航,导出
   **约 8000 条 paired 训练轨迹**,每条以 30 fps 导出并配对一份采样高度场
   `h_τ(x, y)`。
3. **地形参数(生成 TCRS 监督的地形族):**

   | 参数 | 范围 |
   | --- | --- |
   | 高度偏移 | [−0.10, 0.25] m |
   | 支撑面宽度 | [0.10, 1.00] m |
   | 台阶 riser | 5–15 cm |
   | 台阶 tread | 0.20–0.35 m |
   | 坡度 | ≤ 30° |

4. **仿真:** IsaacLab/IsaacSim,物理步长 Δt = 5 ms,control decimation 4
   → **50 Hz 策略频率**;episode 15 s = **750 策略步**;每个 GPU **6144 环境**,
   48 × A800;最终 PPO 阶段统一 **10k iterations**。
5. **领域随机化(per-episode,Table IX):**

   | 参数 | 范围 | 说明 |
   | --- | --- | --- |
   | 静摩擦 | [0.3, 1.6] | per-body 材质 |
   | 动摩擦 | [0.3, 1.2] | per-body 材质 |
   | 基座 CoM 偏移 | ±0.025 m | 每轴 |
   | 默认关节位置 | ±0.1 rad | reset 扰动,概率 0.2 |
   | 推挤速度 | ±0.2 m/s | 周期性基座直推 |
   | 高度图噪声 | ±0.06 m | 每格加性噪声 + 平面漂移(模拟 depth-to-heightmap 误差) |

---

## 2. 阶段一:TCRS 离线地形适配参考合成(Algorithm 2)

```
Algorithm 2: TCRS
1: Input: raw clip m^raw, 地形高度场 h_τ, 接触阈值, 摆动优化参数, IK 权重
2: 用脚高度、速度与迟滞阈值检测接触 mask(区分 stance/swing)
3: 把支撑足锚定到地形支撑面,构建 stance 参考
4: for 每只脚 f、每个摆动相 s = [t0, t1] do
5:     用 Eq. (4) 把 ankle 目标转换到 mid-foot 坐标系
6:     用 raw mid-foot 路径初始化轨迹 knots
7:     for 采样迭代 i = 1..N_iter do
8:         采样时间平滑的 knot 扰动
9:         用 Eq. (5) 评估 tracking/smoothness/clearance/vertical-face/endpoint 代价
10:        用 Eq. (6) softmin 加权采样更新 knots
11:    end for
12:    把优化后的 mid-foot 轨迹转回 ankle/toe/heel 目标
13: end for
14: 用 Eq. (7) 从支撑接触重建根高度
15: 修复小腿/脚碰撞;衰减台阶边缘处不可靠的 toe/heel 约束权重
16: for frame t = 1..T do
17:    用 Eq. (12) 求解多点 Jacobian IK
18:    若脚误差/穿透/关节跳变超阈值,换种子重试
19: end for
20: Output: m^tcrs, 接触 mask, 实现的脚/根轨迹, 诊断量
```

逐步说明:

1. **接触检测:** 由脚高度、脚速度与迟滞阈值估计 stance/swing 区间。支撑足
   latch 到地形支撑面,摆动端点继承 raw 的离地/落地时机 —— **不改变全局行为相位**。
2. **Mid-foot 坐标转换:** 不在 ankle 原点规划,而在虚拟 mid-foot 坐标系规划,
   以平衡脚尖/脚跟在地形不连续处附近的净空:

   ```
   r_mid = (r_toe + r_heel)/2,   p^mid_{f,t} = p^ankle_{f,t} + R_{f,t} r_mid      (4)
   ```

3. **摆动轨迹优化:** 对控制 knots `Y = {y_k}` 最小化:

   ```
   J_s(Y; τ) = λ_ref Σ_k ||y_k − y_k^raw||²
             + λ_sm  Σ_k ||Δ²y_k||²
             + λ_clr Σ_k [h_τ(x_k, y_k) + δ − z_k]₊²      # 净空余量 δ
             + λ_edge Φ_edge(Y; τ)                        # 垂直面穿透惩罚
             + λ_end ||y_1 − ȳ_1||² + λ_end ||y_K − ȳ_K||²  # 固定离地/落地端点
                                                                                (5)
   ```

   用批量采样式轨迹优化求解,softmin 加权采样更新(扰动 ϵ^(j)、温度 η):

   ```
   Y ← Y + Σ_j exp(−J_s(Y+ϵ^(j))/η) / Σ_ℓ exp(−J_s(Y+ϵ^(ℓ))/η) · ϵ^(j)          (6)
   ```

   优化完成后转回 ankle/toe/heel 目标。Φ_edge 惩罚支撑脚 footprint 内查询到
   垂直高度不连续的 toe/heel 采样点。
4. **支撑感知根高度重建:** 脚重规划后,由支撑接触重建躯干根高度:

   ```
   z*_{root,t} = Σ_f w_{f,t} [ h_τ(x^tcrs_{f,t}, y^tcrs_{f,t}) + (z^raw_{root,t} − z^raw_{f,t}) ]
                 / (Σ_f w_{f,t} + ϵ)                                              (7)
   ```

   结果按腿部可达域 clamp,跨支撑切换平滑;飞行/弱接触相回退到 raw 垂直剖面
   (限制每帧位移)。
5. **碰撞修复 + 多点腿部 IK:** 修复小腿/脚碰撞,衰减台阶边缘处不可靠的
   toe/heel 点权重,然后对 12 个腿部关节求解 damped support-aware 多点
   Jacobian IK(根平移固定为上一步结果,根朝向与非腿关节保持 raw):

   ```
   Δq^{leg,⋆} = arg min_{Δq^leg}  Σ_{f∈{L,R}} Σ_{p∈P} || W_{f,p} J_{f,p} Δq^leg − e_{f,p} ||²
                + λ_post ||q^leg − q^{raw,leg}||²     # 姿态正则(贴 raw)
                + λ_cont ||q^leg − q^{prev,leg}||²    # 连续性
                + λ_pen Ψ_pen(q^leg)                  # 脚/小腿穿透
                + λ_dls ||Δq^leg||²                   # 阻尼最小二乘
                                                                                 (12)
   ```

   其中 P = {ankle, toe, heel}。多重种子回退 + 连续性 guard 拒绝高误差/不连续解。
6. **输出:** paired 数据集 `(q^raw, q^tcrs, τ)`,供阶段二训练、阶段三蒸馏。
   产物按表 I 的指标独立评估(穿透深度 Eq. 14、浮动率 Eq. 15、净空违规
   Eq. 16、脚平滑度 Eq. 18、上身偏离 Eq. 17),**不依赖任何策略 rollout**。

---

## 3. 阶段二:盲教师 PPO 训练

1. **命令:** 教师接收 TCRS 参考 `m^tcrs` 作为命令帧(`q^cmd = q^tcrs`)。
2. **网络:** Transformer actor–critic,tokenized 本体感受历史 + 参考命令窗口;
   教师**没有视觉输入**(blind),可使用特权观测(critic 特权组)。
3. **PPO 奖励(Eq. 11,教师与微调阶段共用):**

   ```
   r_t = Σ_{k∈{a,R,f,v}} w_k exp(−||Δ_k||² / σ_k²) − c_E E_t − c_C C_t             (11)
   ```

   | 项 | 权重 | σ / 说明 |
   | --- | --- | --- |
   | 全局锚点位置跟踪 Δa | 1.0 | std 0.2 m |
   | 相对身体朝向跟踪 ΔR | 0.5 | std 0.35 |
   | 脚位置跟踪 Δf | 1.0 | ankle bodies,std 0.1 m |
   | 脚线速度跟踪 Δv | 0.5 | ankle bodies,std 1.0 m/s |
   | 能量惩罚 E_t | −2×10⁻⁵ | action/torque 能量 |
   | 脚/小腿侧向接触 C_t | −0.03 | 踝、膝阈值 5 N |

   教师奖励对 TCRS 参考评估(学生微调则对 raw 命令帧 + 地形条件残差评估)。
4. **PPO 超参(Table VIII):**

   | 项 | 值 |
   | --- | --- |
   | 学习率 | 5×10⁻⁴ |
   | Entropy coefficient | 0.005 |
   | PPO | 5 epochs,4 minibatches,KL target 0.01 |
   | 辅助损失 | 速度 / 锚点 Huber 损失 |

---

## 4. 阶段三:学生蒸馏(identity-gated + target-frame action alignment)

1. **学生架构:** 继承教师同款 command/history Transformer backbone,新增
   **地形感知分支**:17×11 高度图 + validity mask → Map CNN `f_cnn` →
   query-conditioned MapTransformer `f_mt` → 地形隐变量 `z_vis`。
2. **identity-gated 融合:** 地形特征经两条**零初始化残差通路**注入,门控
   向量与最终残差层均初始化为 0,因此**初始化的学生就是一个 raw-reference
   tracker**,地形通路不活跃:

   ```
   意图调制:  u'_t = u_t + tanh(α_u) ⊙ f_u(z_vis_t)                             (8)
   动作残差:  μ_t  = μ_t^base + tanh(α_a) ⊙ f_a([o_t^prop, z_vis_t])             (9)
   ```

3. **目标帧标签重标定:** 教师与学生围绕不同命令帧行动,学生**不能直接模仿
   教师的残差**。改为把教师的**有效 PD 目标**表达在 raw 参考帧下:

   ```
   a_t* = q_t^tcrs + μ_t^tea − q_t^raw                                          (10)
   ```

   学生最小化 `||μ_t^stu − a_t*||²`(MSE)。对齐标签与学生均值在 Eq. (2) 相同的
   残差关节位置单位下比较,并使用相同的残差限幅;教师控制 rollout 时施加的
   动作也是对齐后的目标帧动作,而非教师原生的 adapted-reference 残差。
4. **DAgger 式调度:** teacher-control 概率从 **1.0 退火到 0.0**。
5. **蒸馏超参:** lr 1×10⁻⁴,MSE action loss(表 VIII)。

---

## 5. 阶段四:视觉 PPO 微调

1. **命令切换:** 学生从蒸馏 checkpoint 继续,在 **raw-reference 命令 + 高度图
   观测**下做 PPO 微调。因为从目标帧蒸馏 warm-start,微调是在**精化已有的地形
   感知残差**,而不是从零发现它们;接触/碰撞惩罚防止退回平地跟踪。
2. **辅助估计器头:** actor 增加两个只读内部 latent 的估计头 —— 基座速度估计器
   和 motion-anchor 位置估计器,其(detached)3D 输出拼接进 actor trunk,使部署
   actor 不接触真值基座速度/锚点位置;另有 foot-trajectory 头由教师目标监督,
   **不是 actor 输入**。未来 motion-anchor 位移窗口仍是直接命令 token 输入。
3. **总损失:**

   ```
   L_total = L_PPO + L_value − entropy + L_vel + L_anchor + L_foot
   L_vel: Huber(v̂, v^gt)   L_anchor: Huber(â, a^gt)   L_foot: Huber(p̂_foot, p^gt_foot)
   ```

4. **微调超参(Table VIII):** lr 1×10⁻⁴,entropy 0.001,**backbone LR scale 0.3**
   (对继承的跟踪主干保守更新,地形分支正常更新),foot-trajectory Huber loss
   delta 0.05。

---

## 6. 部署(Fig. 2 第 4 步)

- 部署时策略只接收:**原始参考** + 本体感受 + 机载地形观测;TCRS 监督**绝不在线查询**。
- 平台:29-DoF Unitree G1;感知为躯干安装的 depth-to-height-map 管线,产出与
  训练一致的 1.6 m × 1.0 m、0.1 m 分辨率的 17×11 高度图(观测契约跨 sim-to-real 保持)。
- 安全:运行时 watchdog,检测到扭矩饱和或基座朝向越出训练包络时触发软摔倒恢复。

---

## 7. 观测契约速查(Table VI)

| 观测组 | 内容 | 形状/历史 | 部署角色 |
| --- | --- | --- | --- |
| Proprio | 投影重力、基座角速度、关节位置/速度、上一动作 | 93D | actor 输入 |
| Proprio history | 未展平的本体感受历史 | 10 步 | actor 时间输入 |
| Command window | 未来参考速度、重力、关节命令 token | 21 × 38 | actor 命令输入 |
| Anchor delta window | 局部锚点位移窗口 | 21 × 3 | actor 命令输入 |
| Vision | 高度图 + validity mask | 17×11 格 + mask | 视觉 actor 输入 |
| Critic | 特权参考/身体/基座信息 | 当前帧 | 仅训练 |
| Auxiliary targets | 基座速度、锚点、foot-trajectory 目标 | 当前帧/窗口 | 辅助损失 |
