# 方法原理与文献依据总览（V3.0）

本文档系统总结本项目（`src/maize_leaf/`）当前用于玉米点云三维叶片建模、缺失补全与表型提取的全部核心算法、数学公式、文献来源及其适用边界。

---

## 一、方法总览表

| 处理阶段 | 代码模块与函数 | 采用的方法 | 主要依据与文献来源 |
|---|---|---|---|
| **1. 数据读取与法向利用** | [io.py](../src/maize_leaf/io.py) (`load_pair`) | 配对 TXT / CSV 读取坐标、RGB、语义标签与扫描单位法向量 `(Nx, Ny, Nz)` | 保留原始扫描几何法向，用于标架对齐与可微联合优化 |
| **2. 双重离群点滤波** | [geometry.py](../src/maize_leaf/geometry.py) (`filter_noise_clusters`, `sor_filter`, `clean_leaf_points`) | 欧氏半径连通分量聚类 + 统计离群点去除（SOR） | Rusu et al. (2008, *Robotics and Autonomous Systems*)；Rusu & Cousins (2011, PCL) |
| **3. 茎秆中心线重建** | [geometry.py](../src/maize_leaf/geometry.py) (`fit_stem_centerline`) | 沿 Z 轴分箱取三维中位数 + 弦长参数化有界 GCV P-spline | Eilers & Marx (1996)；Wahba (1990) |
| **4. $L_1$-Medial 骨架与中脉初始化** | [geometry.py](../src/maize_leaf/geometry.py) (`l1_medial_skeleton`, `geodesic_order`, `extract_midrib`, `project_points_to_curve`) | 最远点采样（FPS）+ 正则化 $L_1$-Medial 点云收缩骨架 + 测地排序 + 切线延长重投影主曲线（Principal Curve）+ 牛顿正交投影 | Huang et al. (2013, *ACM TOG*)；Hastie & Stuetzle (1989)；Piegl & Tiller (1997) *The NURBS Book* |
| **5. 无扭转移动标架** | [geometry.py](../src/maize_leaf/geometry.py) (`double_reflection_rmf`, `build_normal_aligned_frame`) | 双反射最小旋转标架（RMF）+ 切线正交平面点云法向 SVD + `2φ` 相位解缠与中值/GCV 样条平滑 | Bishop (1975)；Wang et al. (2008, *ACM TOG*) |
| **6. 可微几何联合优化** | [joint.py](../src/maize_leaf/joint.py) (`joint_optimize_geometry`) | 对中脉偏移 B-spline、RMF 滚转角、双变量拱度场与 Dornbusch 叶宽做端到端解析可微 `soft_l1` 联合优化（点面残差 + 扫描法向一致性 + 单侧鲁棒边界包络 + 中脉弯曲能） | 参数曲面逆向可微拟合范式（*NURBS-Diff* / FSPM 逆优化思想） |
| **7. 物种先验标定与叶宽模型** | [morphology.py](../src/maize_leaf/morphology.py) (`extract_width_profile`, `calibrate_prior_from_leaves`, `fit_dornbusch_blade`) | 左右独立分位包络 + Empirical Bayes 物种先验收缩 + MAP `soft_l1` 分段幂函数拟合 + BIC 叶尖截断模型选择 | Dornbusch et al. (2011, *Annals of Botany*)；Stewart & Dwyer (1999, *Crop Science*)；Montgomery (1911)；Schwarz (1978) & Kass & Raftery (1995) |
| **8. 截断叶尖中脉外推** | [morphology.py](../src/maize_leaf/morphology.py) (`extrapolate_midrib_and_frame`) | 垂直叶面倾角曲率阻尼外推 + 弧长积分 + RMF 延续 | Prévot, Aries & Monestiez (1991, *Agronomie*)；Fournier & Andrieu (1998, ADEL-Maize) |
| **9. 分箱中位数 TPS 与网格生成** | [surface.py](../src/maize_leaf/surface.py) (`_aggregate_tps_bins`, `ThinPlateSplineGCV`, `leaf_mesh_grid`, `fit_leaf`) | 网格分箱中位数聚合 + 二维薄板样条（TPS）+ QR 谱分解 GML/REML 自动正则化 + 叶尖单顶点扇形三角化 + Delaunay 支持区域划分 | Duchon (1977)；Wahba (1985, *Annals of Statistics*; 1990)；Wood (2011, *JRSS-B*) |

---

## 二、分步算法原理与公式

### 1. 双重离群点滤波（`filter_noise_clusters` + `sor_filter`）

- **来源**：Rusu et al. (2008) *Statistical Outlier Removal (SOR)*；Rusu & Cousins (2011) 欧氏聚类滤波思想。
- **做法**：
  1. **欧氏半径连通聚类过滤**：先按精确坐标去重；以最近邻点距中位数 $d_{\mathrm{sp}}$ 与点云包围盒直径 $D$ 确定连通半径 $r = \max(5\,d_{\mathrm{sp}},\; 0.03\,D)$，通过 `cKDTree.sparse_distance_matrix` 构建半径连通图（避免各向异性密度下固定 $k$ 近邻漏连纵向行），仅保留最大主分量以及点数满足 $N_c \ge \max(30,\; 0.05\,N)$ 的主要叶片断片，防止茎秆附近微小误标注噪点簇劫持叶根起点。
  2. **统计离群点去除（SOR）**：计算每个点的 $k=12$ 近邻平均距离 $\bar{d}_i$，基于稳健尺度 $\sigma_{\mathrm{rob}} = 1.4826\,\mathrm{MAD}(\bar{d})$ 与阈值 $\tau = \max(\mathrm{med}(\bar{d}) + 2.0\max(\sigma_{\mathrm{rob}}, 0.5\sigma),\; 2.2\,\mathrm{med}(\bar{d}))$ 剔除飘浮在主叶面周围的扫描散点毛刺，并再次执行连通分量检查。

---

### 2. $L_1$-Medial 骨架与主曲线中脉初始化

#### (1) 有界 GCV P-spline 曲线拟合（`fit_pspline_gcv`）
- **来源**：Eilers & Marx (1996) *Penalized B-splines (P-splines)*；Wahba (1990) 广义交叉验证（GCV）；B-spline 凸包性质。
- **原理**：采用带二阶差分惩罚的三次 B-spline（$n_{\mathrm{basis}} \le 10$）：
  $$\min_C \|W^{1/2}(BC - Y)\|_F^2 + \lambda \|D_2 C\|_F^2$$
  平滑权重 $\lambda$ 通过最小化 GCV 准则自动确定；对三维空间曲线拟合时（`bounded=True`），使用 `scipy.optimize.lsq_linear` 将各轴控制点约束在输入节点包围盒外扩 $2\%$ 范围内，防止样条过冲。

#### (2) $L_1$-Medial 点云收缩骨架 + 主曲线细化（`l1_medial_skeleton`, `extract_midrib`, `project_points_to_curve`）
- **来源**：Huang et al. (2013, *ACM TOG*, SIGGRAPH) *$L_1$-Medial Skeleton of Point Cloud*；Hastie & Stuetzle (1989) 主曲线；Piegl & Tiller (1997, *The NURBS Book*) 牛顿精确正交投影。
- **原理**：
  1. **最远点采样与 $L_1$-Medial 收缩**：以确定性最远点采样（FPS）在叶片上均匀布置骨架种子点 $X = \{x_i\}$，在覆盖叶片横截面宽度的渐进带宽 $h \in [\max(0.12\,D, 6\,d_{\mathrm{sp}}),\; 0.24\,D]$ 下迭代更新正则化 Weiszfeld $L_1$-中位收缩步：
     $$x_i^{(t+1)} = \frac{\sum_j \alpha_{ij}\, p_j}{\sum_j \alpha_{ij}} + \mu\,\sigma_i(X)\,\frac{\sum_{i'} \beta_{ii'}(x_i - x_{i'})}{\sum_{i'} \beta_{ii'}}$$
     其中 $\alpha_{ij} = \theta(\|x_i - p_j\|)/\|x_i - p_j\|$ 对应局部加权几何中位数（对横截面离群点与孔洞具备高鲁棒性），$\sigma_i(X) = \lambda_{\max}/\sum \lambda_k$ 为局部方向线性度（Huang et al., 2013 Eq. 5），使节点收缩到一维中心脊线同时保持纵向均匀分布。
  2. **测地排序与端点防收缩重投影**：在收缩后的骨架点上构建近邻图并从距茎秆最近点运行 Dijkstra 测地排序拟合初始中脉；两端沿切线延伸 $15\%$ 进行全景重投影，消除 $L_1$ 收缩导致的叶基/叶尖端点内缩，随后用牛顿迭代正交投影执行主曲线迭代得到弧长参数化初始中脉 $C_0(s)$。

---

### 3. 无扭转移动标架与点云法向对齐（`double_reflection_rmf`, `build_normal_aligned_frame`）

- **来源**：Bishop (1975) 平行输运标架；Wang et al. (2008, *ACM TOG*) 双反射最小旋转标架（Double Reflection RMF）。
- **原理**：沿弧长用双反射法生成无扭转参考标架 $(T_0, B_0, N_0)$；将点云自带单位法向量 `(Nx, Ny, Nz)` 投影至切线正交平面 $\perp T_0$，经倍角相位解缠 $\phi_{\mathrm{raw}} = \frac{1}{2}\mathrm{unwrap}(2\,\mathrm{atan2}(\sin\phi, \cos\phi))$、3 点中值滤波与 GCV 样条平滑滚转角后旋转得到全叶正交标架 $(t(s), b(s), n(s))$。

---

### 4. 可微几何联合优化（`joint_optimize_geometry`）

- **核心动机**：传统串联管线中，“先算中脉再算宽度”会在单侧叶缘遮挡时把中脉中位数向完好侧拉偏，且后续步骤无法反向纠正中脉。
- **联合参数化**：在观测域 $u = s / L_0 \in [0, 1]$ 上同时优化 **37 维可微几何状态向量 $\Theta$**：
  - 中脉横向与法向三次 B-spline 偏移系数 $\delta_b, \delta_n \in \mathbb{R}^8$：$C(u) = C_0(u) + \delta_b(u)\,b_0(u) + \delta_n(u)\,n_0(u)$；
  - 标架滚转角 B-spline 调整 $\delta_\phi \in \mathbb{R}^6$；
  - 二维低阶横向拱度场 $q_{\mathrm{camber}}(u, v) = a_1(u)\,v + a_2(u)\,v^2$（$v \in [-1, 1]$，强制 $q_{\mathrm{camber}}(u, 0)=0$ 使中脉锚定在叶片中心脊上）；
  - Dornbusch (2011) 叶宽参数 $(w_{\max}, x_m, k_{\mathrm{lig}}, c_1, c_2)$。
- **可微联合目标函数**（通过 `scipy.optimize.least_squares(..., loss='soft_l1')` 求解）：
  1. **点到曲面法向残差**：$(h_k - q_{\mathrm{camber}}(u_k, v_k)) / w_{\max}$；
  2. **解析法向对齐残差**（直接利用扫描单位法向量 $n_k^{\mathrm{scan}}$）：由曲面解析偏导数 $\partial q/\partial u,\;\partial q/\partial v$ 计算预测单位法向 $\tilde{N}_k$，最小化 $\|\tilde{N}_k \times n_k^{\mathrm{scan}}\|_2 = |\sin\angle(\tilde{N}_k, n_k^{\mathrm{scan}})|$（对 $\pm 180^\circ$ 法向翻折不变）；
  3. **单侧遮挡鲁棒边缘包络残差**：对各分段预提取的左右代表边界三维锚点 $(p_{L,i}, p_{R,i})$ 计算实时横向投影 $r_{L,i}(\Theta), r_{R,i}(\Theta)$，联合施加**包络越界惩罚**（$\max(0, -w_i/2 - r_{L,i}),\;\max(0, r_{R,i} - w_i/2)$）、**完好侧对齐**（匹配外伸更宽的完好边到 $\pm w_i/2$）与**饱和双侧居中项**（$\tanh$ 截断使单侧大缺口不拉偏中脉）；
  4. **中脉弯曲能量正则化**：直接惩罚更新后三维中脉的二阶差分 $\lambda_C \|D_2 C(\Theta)\|_F^2$，迫使中脉跨越单侧缺口时保持解剖学平直光滑。

---

### 5. 叶宽轮廓、先验标定与叶尖缺失反演

#### (1) Dornbusch 分段幂函数叶形模型（`DornbuschBlade`）
- **来源**：Dornbusch, Watt, Baccar, Fournier & Andrieu (2011, *Annals of Botany* 107(5): 865–873)；面积形状系数依据 Stewart & Dwyer (1999, *Crop Science* 39: 422–427) 与 Montgomery (1911)。
- **公式**：以归一化物理弧长 $u = s / L \in [0, 1]$ 表示，叶片在最宽位置 $x_m$ 处分为基部段（$0 \le u \le x_m$）和远端叶尖段（$x_m < u \le 1$），保证在 $x_m$ 处 $C^1$ 连续：
  $$w(s) = \begin{cases} w_{\max}\left[1 - (1 - k_{\mathrm{lig}})\left(\dfrac{x_m - u}{x_m}\right)^{c_2}\right], & 0 \le u \le x_m \\[10pt] w_{\max}\left[1 - \left(\dfrac{u - x_m}{1 - x_m}\right)^{c_1}\right], & x_m < u \le 1 \end{cases}$$
  其中 $k_{\mathrm{lig}} = w(0) / w_{\max}$ 为叶耳基部相对宽度，$c_1, c_2$ 控制叶尖段与基部段边缘曲率。其精确积分面积形状系数为：
  $$\alpha = \frac{1}{L\,w_{\max}}\int_0^L w(s)\,ds = x_m\left(1 - \frac{1 - k_{\mathrm{lig}}}{c_2 + 1}\right) + (1 - x_m)\frac{c_1}{c_1 + 1}$$
  玉米叶片文献典型值约为 $\alpha \approx 0.734\text{–}0.75$（Stewart & Dwyer, 1999）。

#### (2) 左右宽度提取与 Empirical Bayes 先验标定（`extract_width_profile`, `calibrate_prior_from_leaves`）
- **左右包络分离**：沿弧长分 32 箱，分别取横向坐标 $r$ 的 $2\%$ 和 $98\%$ 分位数作为左/右半宽 $(w_L, w_R)$；正常情况下总宽取 $w_L + w_R$，仅当单侧出现超过 $75\%$ 的严重缺失时，用 $2\max(w_L, w_R)$ 恢复局部总宽包络。
- **Empirical Bayes 先验收缩**：从批次中末端自然收尖（`tip_ratio < 0.22`）且点数充足的近完整叶片拟合得到的有效无量纲参数（$0.20 < x_m < 0.55,\; 1.2 < c_1 < 3.2,\; 0.65 \le \alpha \le 0.82$），按收缩权重 $w = \frac{n}{n + 10}$ 向玉米文献先验（`xm=0.36, k_lig=0.62, c1=1.85, c2=2.00`）收缩，得到当前数据集的经验贝叶斯先验。

#### (3) 可辨识叶尖长度反演与 BIC 模型选择（`fit_dornbusch_blade`）
- **来源**：MAP 鲁棒非线性最小二乘（`soft_l1`）+ 贝叶斯信息准则 BIC（Schwarz, 1978; Kass & Raftery, 1995）。
- **原理**：为避免叶尖截断时总长 $L$ 与远端指数 $c_1$ 之间的共线性失控：
  1. 先拟合固定长度的完整叶模型 $M_0$（$L = L_{\mathrm{obs}}$，5 参数）。
  2. 若末端宽度比例 `tip_ratio > 0.22`，根据 Dornbusch 远端幂律解析反演先验叶长上限：
     $$u_{\mathrm{obs}} \approx x_m + (1 - x_m)(1 - \text{tip\_ratio})^{1/c_1}, \quad L_{\mathrm{prior}} = \mathrm{clip}\!\left(\frac{L_{\mathrm{obs}}}{u_{\mathrm{obs}}},\; L_{\mathrm{obs}},\; 1.45\,L_{\mathrm{obs}}\right)$$
     并在紧约束物种先验正则化（`prior_w = 0.45`）下拟合截断模型 $M_1$（$L \ge L_{\mathrm{obs}}$，6 参数）。
  3. 按 BIC 公式 $\mathrm{BIC} = n\ln(\mathrm{RSS}/n) + k\ln n$ 对比两模型，仅当存在正贝叶斯证据 $\mathrm{BIC}_{M_1} + 4 < \mathrm{BIC}_{M_0}$（Kass & Raftery, 1995）时才认定叶尖被截断并延长 $L$。

#### (4) 截断叶尖中脉外推（`extrapolate_midrib_and_frame`）
- **来源**：Prévot, Aries & Monestiez (1991, *Agronomie*) 与 Fournier & Andrieu (1998, ADEL-Maize) 玉米中脉垂直叶面倾角几何模型。
- **做法**：当 $L > L_{\mathrm{obs}}$ 时，保持观测末端 $25\%$ 弧长段的水平方位角不变，拟合俯仰倾角 $\theta(s) = \arcsin(t_z(s))$ 的局部变化率 $\kappa \le 0$（仅允许自然下垂、禁止向上反翘），以指数阻尼曲率沿弧长积分延拓至 $s = L$（下垂角下限 $-0.48\pi$），并同步用双反射 RMF 延续正交标架。

---

### 6. 叶面起伏与孔洞补全：分箱中位数聚合 + Duchon TPS + GML/REML

- **来源**：Duchon (1977) 薄板样条；Wahba (1985, *Annals of Statistics* 13(4): 1378–1402; 1990) 广义最大似然（GML / REML）；Wood (2011, *JRSS-B*)。
- **原理**：在 $(u_{\mathrm{norm}}, v_{\mathrm{norm}}) = (s / L_{\mathrm{obs}},\; 2r / w(s))$ 网格上（经 $40\times 12$ 空间分箱并对每格内所有点取**二维坐标与法向高度 $h$ 的中位数**作为代表点，上限 350 个，彻底屏蔽离散单点毛刺），用核函数 $\phi(d) = d^2\ln d$ 的二维薄板样条拟合相对标架的法向高度 $h = (p - C(s))\cdot n(s)$，最小化薄弹性板弯曲能量：
  $$\min_q \sum_{i=1}^m \bigl(q(u_i, v_i) - h_i\bigr)^2 + \lambda \iint \bigl(q_{uu}^2 + 2q_{uv}^2 + q_{vv}^2\bigr)\,du\,dv$$
  通过线性多项式矩阵 $P=[1,u,v]$ 的完整 QR 分解 $P=[Q_1,Q_2][R;0]$ 投影到正交补空间并对 $Q_2^\top K Q_2 = U\Lambda U^\top$ 做特征分解，直接优化 **GML/REML 目标函数**自动求解最优平滑参数 $\lambda^*$：
  $$\mathrm{GML}(\lambda) = \left(\sum_{j=1}^{m-3}\frac{z_j^2}{\lambda_j + \lambda}\right) \exp\!\left(\frac{1}{m-3}\sum_{j=1}^{m-3}\ln(\lambda_j + \lambda)\right)$$
  相比有限样本 GCV，GML/REML 从理论上避免了 $\lambda \to 0$ 时的过拟合欠平滑；当离散噪点多时自动增大 $\lambda^*$、降低有效自由度 `edf`。
- **最终连续参数曲面**：
  $$S(s, v) = C(s) + \frac{v\,w(s)}{2}\,b(s) + q(s, v)\,n(s), \quad s \in [0, L],\; v \in [-1, 1]$$
  在叶尖外推段，法向残差按 $w(s) / w(L_{\mathrm{obs}})$ 平滑衰减收拢至中脉。

---

### 7. 网格构造与区域支持划分（`leaf_mesh_grid`, `fit_leaf`）

- **单顶点扇形闭合**：在最后一行横向截面与唯一叶尖顶点 $(L, 0)$ 之间采用扇形三角剖分，避免 $w(L)=0$ 处产生重复退化点与零面积三角面。
- **支持区域标签**（`0=observed`、`1=interpolated`、`2=extrapolated`）：
  - 在物理参数平面 $(s_{\mathrm{proj}}, r)$ 上构建实测点二维 `Delaunay` 凸包；
  - 在三维空间中以支持半径 $R_{\mathrm{sup}} = \max(2.5\,d_{\mathrm{sp}},\; 0.75\max(\Delta u, \Delta v))$ 判定近邻支持：凸包外或 $s > L_{\mathrm{obs}}$ 记为 `extrapolated`，凸包内且三维近邻距离 $\le R_{\mathrm{sup}}$ 记为 `observed`，凸包内其余空洞记为 `interpolated`；三角面取三顶点中的最保守标签（$2 > 1 > 0$）。

---

## 三、实测验证摘要（14 株 / 90 片叶，V3.0）

- **全量重建（`scripts/run.py`）**：
  - 14 株、90 片玉米叶全部成功重建，聚类 + SOR 共过滤 `12,778` 个离群噪点；严重中脉过冲（$L_{\mathrm{obs}} / D > 1.45$）为 `0 / 90`。
  - 去噪点到曲面中位距离 / 最大宽 $W_{\max}$ 为 `0.042`（范围 `0.021 ~ 0.104`），95% 分位残差中位数为 `0.172 * W_max`；面积形状系数 $\alpha$ 中位数为 `0.730`（与 Stewart & Dwyer 1999 文献值 `0.734` 吻合）。
- **真实玉米完整叶留出遮挡基准（`scripts/validate_holdout.py`）**：
  - 内部孔洞（`hole`）：3D 中位误差 `0.0063`（4.0% $W_{\max}$），叶长误差 `1.52%`，叶面积误差 `4.78%`；
  - 单侧叶缘缺口（`margin`）：3D 中位误差 `0.0108`（6.2% $W_{\max}$），叶长误差 `3.53%`，叶面积误差 `5.54%`；
  - 15% 叶尖截断（`tip_15pct`）：3D 中位误差 `0.0141`（8.8% $W_{\max}$），叶面积误差 `8.67%`，叶长误差 `25.72%`（极末端隐藏长度仍受远端形状可辨识性限制）。

---

## 四、主要参考文献

1. **Bishop, R. L. (1975).** There is more than one way to frame a curve. *The American Mathematical Monthly*, 82(3), 246–251.
2. **Dijkstra, E. W. (1959).** A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.
3. **Dornbusch, T., Watt, J., Baccar, R., Fournier, C., & Andrieu, B. (2011).** A comparative analysis of leaf shape of wheat, barley and maize using an empirical shape model. *Annals of Botany*, 107(5), 865–873.
4. **Duchon, J. (1977).** Splines minimizing rotation-invariant semi-norms in Sobolev spaces. In *Constructive Theory of Functions of Several Variables*, Springer, 85–100.
5. **Eilers, P. H. C., & Marx, B. D. (1996).** Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.
6. **Fournier, C., & Andrieu, B. (1998).** A 3D architectural and process-based model of maize development (ADEL-maize). *Annals of Botany*, 81(2), 233–250.
7. **Hastie, T., & Stuetzle, W. (1989).** Principal curves. *Journal of the American Statistical Association*, 84(406), 502–516.
8. **Huang, H., Wu, S., Cohen-Or, D., Gong, M., Zhang, H., Li, G., & Chen, B. (2013).** $L_1$-medial skeleton of point cloud. *ACM Transactions on Graphics*, 32(4), Article 65 (pp. 65:1–65:8).
9. **Kass, R. E., & Raftery, A. E. (1995).** Bayes factors. *Journal of the American Statistical Association*, 90(430), 773–795.
10. **Montgomery, E. G. (1911).** Correlation studies in corn. *Nebraska Agricultural Experiment Station Annual Report*, 24, 108–159.
11. **Piegl, L., & Tiller, W. (1997).** *The NURBS Book* (2nd ed.). Springer.
12. **Prévot, L., Aries, F., & Monestiez, P. (1991).** Modélisation de la structure géométrique du maïs. *Agronomie*, 11(6), 491–503.
13. **Rusu, R. B., Marton, Z. C., Blodow, N., Dolha, M., & Beetz, M. (2008).** Towards 3D point cloud based object maps for household environments. *Robotics and Autonomous Systems*, 56(11), 927–941.
14. **Rusu, R. B., & Cousins, S. (2011).** 3D is here: Point Cloud Library (PCL). *IEEE International Conference on Robotics and Automation (ICRA)*, 1–4.
15. **Schwarz, G. (1978).** Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.
16. **Stewart, D. W., & Dwyer, L. M. (1999).** Mathematical characterization of leaf shape and area of maize hybrids. *Crop Science*, 39(2), 422–427.
17. **Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000).** A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
18. **Wahba, G. (1985).** A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem. *The Annals of Statistics*, 13(4), 1378–1402.
19. **Wahba, G. (1990).** *Spline Models for Observational Data*. SIAM.
20. **Wang, W., Jüttler, B., Zheng, D., & Liu, Y. (2008).** Computation of rotation minimizing frames. *ACM Transactions on Graphics*, 27(1), Article 2 (pp. 2:1–2:18).
21. **Wood, S. N. (2011).** Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3–36.
