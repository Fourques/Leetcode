<!--
使用说明：
- 将本代码框内的全部内容复制到你的 .md 文件中时，**不要包含**本消息最外层的三反引号（```md 与结尾的 ```）。
- 下面文档中的公式使用 `$$ ... $$`（块公式）与 `$ ... $`（行内公式），主流 Markdown 预览（VSCode、Typora、Obsidian、GitHub 等）在启用 KaTeX/MathJax 后可正确渲染。
-->

# CaT-PVG：语义门控与双线索驱动的时变高斯动静分离方法（项目说明）

> 在 PVG 的时变高斯框架上，引入**二维基础模型→4D 语义特征蒸馏**（上下文感知），构建由**特征差异**与**语义先验**组成的双线索动态掩码，并通过**语义门控速度**与**语义调度寿命**将 2D 知识“写入”3D/4D 表示；同时加入**时间循环一致性**与**实例原型对比**以提升时间稳定与实例一致性。在不显著增加计算开销的前提下，显著改进动静分离与渲染稳定性。

---

## 目录
- [1. 问题与直觉](#1-问题与直觉)
- [2. 方法总览（Pipeline）](#2-方法总览pipeline)
- [3. 表示与符号（PVG 回顾）](#3-表示与符号pvg-回顾)
- [4. 上下文感知：二维基础模型→4D 语义特征蒸馏](#4-上下文感知二维基础模型4d-语义特征蒸馏)
- [5. 双线索动态掩码：特征差异 + 语义先验](#5-双线索动态掩码特征差异--语义先验)
- [6. 语义门控速度（Context-Gated Motion）](#6-语义门控速度context-gated-motion)
- [7. 语义调度寿命（Semantic-Scheduled Lifespan）](#7-语义调度寿命semantic-scheduled-lifespan)
- [8. 静区零速度正则（写入 3D/4D）](#8-静区零速度正则写入-3d4d)
- [9. 时间循环一致性（Temporal Cycle Consistency）](#9-时间循环一致性temporal-cycle-consistency)
- [10. 实例原型对比（Instance-Prototype Contrast）](#10-实例原型对比instance-prototype-contrast)
- [11. 总损失](#11-总损失)
- [12. 训练流程与调度](#12-训练流程与调度)
- [13. 推理与导出](#13-推理与导出)
- [14. 实现要点（工程）](#14-实现要点工程)
- [15. 实验设置（建议）](#15-实验设置建议)
- [16. 潜在局限与扩展](#16-潜在局限与扩展)
- [17. 最小伪代码（便于移植）](#17-最小伪代码便于移植)
- [18. 术语小词典](#18-术语小词典)

---

## 1. 问题与直觉

动态街景视频的三维表达需要**区分相机运动与世界运动**，并在任意时刻/视角下重建外观与几何。  
静态 3DGS 仅建模空间；**PVG**（Periodic Vibration Gaussian）为每个高斯引入时间行为（速度/寿命/周期），能够表达动态。  
仅靠重建信号易出现：
- **背景伪动**（路/楼/天出现不必要速度）；
- **前景欠动**（车/行人边界模糊、轨迹不连贯）；

**核心直觉**：借助 2D 大模型（语义/特征）的成熟能力，将 2D 知识蒸馏为 4D 语义，引导 4D 高斯**该动的动、该静的静**。

---

## 2. 方法总览（Pipeline）

1. **时变表示（PVG）**：每粒高斯具备时间相关的中心与不透明度；
2. **上下文感知（2D→4D 语义蒸馏）**：将 LSeg / DINO / SAM 等 2D 模型的像素特征蒸馏到高斯，获得随时间搬运的 4D 语义向量；
3. **双线索动态掩码**：由**特征差异 $D$**与**语义先验**（人/车等）融合得到静/动态像素区域；
4. **语义门控速度**：用高斯的语义向量门控其速度（背景抑制、前景放开）；
5. **语义调度寿命**：对静态高斯施加寿命下界先验，鼓励背景高寿命/低速度；
6. **静区零速度正则**：在静态像素区域惩罚速度图，压制伪动；
7. **时间循环一致性**与**实例原型对比**：提升时间稳定与实例一致性；
8. **两阶段训练**：先稳静区与语义，再全量优化。

---

## 3. 表示与符号（PVG 回顾）

- 高斯参数（核心）：位置 $\mu$、旋转 $q$、尺度 $s$、不透明度 $\alpha$、颜色/SH；
- 时间参数：出现时刻 $\tau$、寿命尺度 $\beta$、速度向量 $v$、周期 $l$。

**时刻 $t$ 的中心与不透明度：**

$$
\mu(t)=\mu+\frac{l}{2\pi}\,\sin\!\left(2\pi\,\frac{t-\tau}{l}\right)\cdot v,
\qquad
\alpha(t)=\alpha\cdot \exp\!\left(-\tfrac{1}{2}\Big(\tfrac{t-\tau}{\beta}\Big)^2\right).
$$

定义“静态度/寿命比”：

$$
\rho=\frac{\beta}{l}.
$$

$\rho$ 越大，越趋于静态高寿命。

---

## 4. 上下文感知：二维基础模型→4D 语义特征蒸馏

- **老师特征（Teacher）**：对真图 $I_t$ 用冻结的 2D 模型提取像素特征图 $F_t\in\mathbb{R}^{C\times H\times W}$（如 512d）。
- **学生特征（Student）**：渲染时与 RGB 并行得到低维特征图 $F_s\in\mathbb{R}^{d\times H\times W}$（如 $d=64$），再用 $1\times1$ 卷积 $U(\cdot)$ 升到教师维度。

**特征蒸馏损失：**

$$
\mathcal{L}_f=\big\|\,U(F_s)-F_t\,\big\|_1.
$$

> 训练收敛后，**参与生成某像素的高斯**各自携带的语义向量 $f_{\text{con}}$ 学成与老师一致的“上下文语义”。由于高斯随时间形变，这些语义也随时间搬运，形成**4D 语义特征**。

---

## 5. 双线索动态掩码：特征差异 + 语义先验

- **特征差异（像素级）**：

$$
D=\frac{1-\cos\!\big(U(F_s),\,F_t\big)}{2}\in[0,1],
$$

差异越大越可能为动态。

- 由小型 MLP 产生“静态性分数” $\delta$，通过阈值 $\varepsilon$ 得到学习型静态掩码：
  
$$
M_{\text{stat}}^{\text{learn}}=\mathbf{1}\!\big(\delta>\varepsilon\big).
$$

- **语义先验**（人/车等动倾向；路/楼/天等静倾向）生成 $M_{\text{stat}}^{\text{sem}}$。
- **融合规则（保守）**：静态取交，动态取并

$$
M_{\text{stat}}=M_{\text{stat}}^{\text{learn}}\wedge M_{\text{stat}}^{\text{sem}},
\qquad
M_{\text{dyn}}=1-M_{\text{stat}}.
$$

- 对掩码做**跨时序/多视角投票**以提升稳定性。

---

## 6. 语义门控速度（Context-Gated Motion）

为每粒高斯定义：

$$
g=\sigma\!\big(W\,f_{\text{con}}\big),
\qquad
v^{\text{eff}}=g\cdot v.
$$

- 背景（语义似路/楼/天）$\Rightarrow\ g\!\downarrow$，有效速度被抑制；
- 前景（车/行人）$\Rightarrow\ g\!\uparrow$，保留/增强运动。

渲染得到**像素速度图** $V$（按可见性与 $\alpha(t)$ 加权的速度投影），用于正则与可视化。

---

## 7. 语义调度寿命（Semantic-Scheduled Lifespan）

令高斯的“静态概率” $w^{\text{stat}}$ 为其多视角投影在 $M_{\text{stat}}$ 上的覆盖比，则：

$$
\mathcal{L}_{\rho}=\sum\nolimits_i w^{\text{stat}}_i\ \max\!\big(0,\,\rho^\star-\rho_i\big),
$$

鼓励静态高斯的 $\rho$ 不低于阈值 $\rho^\star$（高寿命/低摆动）。

---

## 8. 静区零速度正则（写入 3D/4D）

$$
\mathcal{L}_v=\Big(W_{\text{sem}}\cdot M_{\text{stat}}\Big)\odot \|V\|_1,
$$

其中 $W_{\text{sem}}$ 是语义置信度权重（低置信度衰减影响）。  
该项将“**静态像素=零速度**”写入 3D/4D 参数，显著压制背景伪动。

---

## 9. 时间循环一致性（Temporal Cycle Consistency）

记 $\Phi(\cdot; t_1\!\to\!t_2)$ 为 PVG 在时间上的前向位移算子，则闭环一致性：

$$
\mathcal{L}_{\text{cycle}}=\sum\nolimits_i
\left\|
\Phi^{-1}\!\Big(\Phi(\mu_i;\,t_1\!\to\! t_2);\ t_2\!\to\! t_1\Big)-\mu_i
\right\|_1.
$$

该约束减少漂移与时间抖动，强化“相机动 $\neq$ 世界动”的解耦。

---

## 10. 实例原型对比（Instance-Prototype Contrast）

以实例跟踪或弱监督得到实例原型 $p_k$（EMA 聚合），对属于实例 $k$ 的高斯语义向量 $f_{\text{con}}$ 施对比：

$$
\mathcal{L}_{\text{ipc}}=-\sum_{i\in k}
\log\frac{\exp\big(\cos(f_{\text{con},i},p_k)/\tau\big)}
{\sum\limits_{u}\exp\big(\cos(f_{\text{con},i},p_u)/\tau\big)}.
$$

提升同一实例跨时间的一致性与边界干净度。

---

## 11. 总损失

$$
\small
\begin{aligned}
\mathcal L &=
\underbrace{(1-\alpha)\,\|I-\hat I\|_1+\alpha\,\mathrm{SSIM}(I,\hat I)}_{\mathcal L_{\text{rgb}}}
+ \lambda_f\,\underbrace{\|U(F_s)-F_t\|_1}_{\mathcal L_f}
+ \lambda_v\,\underbrace{(W_{\text{sem}}\, M_{\text{stat}})\odot \|V\|_1}_{\mathcal L_v}
\\
&\quad + \lambda_\rho\,\underbrace{\sum_i w^{\text{stat}}_i\max(0,\rho^\star-\rho_i)}_{\mathcal L_\rho}
+ \lambda_c\,\underbrace{\sum_i \|\Phi^{-1}(\Phi(\mu_i;t_1\!\to\!t_2);t_2\!\to\!t_1)-\mu_i\|_1}_{\mathcal L_{\text{cycle}}}
+ \lambda_{\text{ipc}}\,\underbrace{\mathrm{NT\textrm{-}Xent}(f_{\text{con}},\{p_k\})}_{\mathcal L_{\text{ipc}}}
\\
&\quad + \underbrace{\mathcal L_{D}+\mathcal L_{n}+\mathcal L_{s}+\mathcal L_{g}+\mathcal L_{\text{smooth}}}_{\text{深度/法线/展平/巨大高斯/时间平滑正则}}.
\end{aligned}
$$

**默认权重建议**：  
$\alpha=0.2,\ \lambda_f=1.0,\ \lambda_v:0\!\to\!0.5$ 线性热启（5k steps），$\lambda_\rho=0.1\sim0.2,\ \lambda_c=0.1,\ \lambda_{\text{ipc}}=0.05$。

---

## 12. 训练流程与调度

**Stage-1（2–5k iter）**
- 并行渲染 RGB 与低维 $F_s$；对齐老师 $F_t$ 学 $\mathcal L_f$；
- 由 $D$ + 语义先验融合 $M_{\text{stat}}$；仅在静区计 $\mathcal L_{\text{rgb}}$；
- 轻度开启 $\mathcal L_v$（权重较小），稳定静态区域。

**Stage-2（全量 30k± iter）**
- 开启**语义门控速度**与**寿命先验**；$\lambda_v$ 热启动到 0.5；
- 加入 $\mathcal L_{\text{cycle}}$、$\mathcal L_{\text{ipc}}$；
- 每 200–500 iter 以跨时序投票**重估** $M_{\text{stat}}$；
- 按既有策略做点密化/删减；远距用密度控制避免大高斯淹没细节。

**采样与数据**
- 训练以 mini-batch 抽样 $(t, \text{view})$，邻帧 $(t\pm\Delta t)$ 用于平滑/循环；
- 教师特征：建议**离线缓存 .pt**（FP16 + 下采样×2 + PCA 到 64 维），或在线计算（更慢、少占盘）。

---

## 13. 推理与导出

- **逐帧渲染**：给定 $(t^\*, \text{view})$ 输出 RGB/深度/速度/特征图；
- **新视角/插帧**：时间 $t$ 连续编码，自然支持插帧与新视角；
- **导出点云**：固定时刻 $t^\*$ 将高斯变换到该时刻后写 `.ply`（静态快照）；
- **静/动分离导出**：以阈值组合 $\|v\|<v_{\text{thr}},\ \rho>\rho_{\text{thr}},\ g<g_{\text{thr}}$ 导出静态背景；批量导出时间序列得到“动态点云视频”。

---

## 14. 实现要点（工程）

- **语义门控位置**：直接作用在高斯速度参数 $v$ 上（非像素级后乘），使梯度直达参数；
- **阈值经验**：$v_{\text{thr}}=0.3\!\sim\!0.5$（等效像素/帧），$\rho_{\text{thr}}=1.2\!\sim\!1.5,\ g_{\text{thr}}=0.4\!\sim\!0.6$；
- **鲁棒性**：语义置信度作权重；对“停着的车”采用时序 IoU 白名单（允许进入静态并不过度惩罚）；
- **效率**：学生特征 64/128 维 + $1\times1$ 升维；`.pt` 用 FP16/降采样/PCA 可将百 GB 降至十几~几十 GB；
- **可选进阶**：加入 DCN 形变补偿支路（成本更高，效果更好）。

---

## 15. 实验设置（建议）

- **数据**：KITTI-360、Waymo Open（可加 TUM-RGBD 动态子集）；
- **指标**：
  - 全局：PSNR / SSIM / LPIPS；
  - **动态区**：DPSNR / DSSIM；
  - **速度残差@静区**：$\tilde{\|V\|}$（中位数越低越好）；
  - 时间稳定：Temporal Warping Error / FVD；
  - 几何：Depth-L1 / $\delta<1.25$；
  - 效率：FPS / 训练时长；
- **对比**：PVG、DeSiRe-GS、CoDa-4DGS、Feature-3DGS（静态语义任务）；
- **消融**：去门控/去寿命先验/去双线索掩码/去循环一致/去实例对比，逐一验证增益；
- **跨域**：白天→夜晚、晴→雨雾，考察泛化与鲁棒。

---

## 16. 潜在局限与扩展

- **语义先验误差**：通过置信度加权与时序投票缓解；
- **教师域偏移**：可在目标域做轻微特征对齐/正则；
- **极端遮挡/低纹理**：引入稀疏 LiDAR 深度或多帧一致性增强；
- **多模态提示**：将文本/点/框提示引入 4D 语义检索与编辑。

---

## 17. 最小伪代码（便于移植）

```python
# 取一帧 (t, view)
I, K, Rt = load_rgb_and_pose(t, view)
F_t = load_feat_pt(t, view) if use_cache else teacher_net(I).detach()

# 前向：并行渲 RGB、低维学生特征、速度图
rgb, V, F_s_low = render_pvg_with_feat(gaussians, t, K, Rt)

# 蒸馏：升维并对齐老师特征
F_s = conv1x1(F_s_low)                 # 1x1 升维
L_f = l1(F_s, F_t)

# 动态掩码：特征差 D + 语义先验
D = (1 - cosine(F_s.detach(), F_t)) * 0.5
delta = mlp_dyn(F_s.detach(), F_t)
M_stat = fuse_masks(delta > eps, semantic_mask, rule="and_static_or_dynamic")
M_stat = temporal_vote(M_stat, poses=..., depths=..., window=3)

# 语义门控：作用在高斯速度上
g = sigmoid(W @ gaussians.f_con.T).T
gaussians.v_eff = g * gaussians.v     # 用 v_eff 渲染 V

# 寿命先验：根据投影到静态区域的覆盖率 w_stat
w_stat = estimate_gaussian_static_prob(gaussians, M_stat, cams=...)
L_rho = (w_stat * relu(rho_star - gaussians.rho)).sum()

# 静区零速度
L_v = (sem_conf * M_stat * abs(V)).mean()

# 闭环一致
L_cyc = temporal_cycle_consistency(gaussians, t1=t-Δt, t2=t+Δt)

# 重建 + 正则
L_rgb = l1_ssim(rgb, I)
L_geom = L_depth + L_normal + L_flat + L_big + L_smooth
loss = L_rgb + λf*L_f + warm(λv)*L_v + λρ*L_rho + λc*L_cyc + λipc*L_ipc + L_geom
loss.backward(); optimizer.step()