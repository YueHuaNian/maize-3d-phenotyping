# 玉米叶片三维重建：逐步算法、公式与实现

本文对应当前代码。原始点云只读，模型与检查报告另存于 `outputs/`。长度沿用输入坐标单位，面积为该单位的平方；程序不自动把数据换算为米或毫米。

## 1. 输入、输出与完整流程

输入是已经带有语义编号的玉米点云：`semantic=0` 为茎秆，正整数为各叶片。本项目不执行自动语义分割。每片叶最终是一个可保存、加载、连续求值的参数曲面：

$$
S(s, v) = C(s) + \frac{v\,w_{\mathrm{eff}}(s)}{2} b(s) + \widetilde{q}(s, v) n(s), \quad s \in [0, L],\ v \in [-1, 1].
$$

$C$ 是参考曲线，$(t, b, n)$ 是局部正交标架，$w_{\mathrm{eff}}$ 是横向宽度，$\widetilde{q}$ 是法向起伏。**最终中心轨迹**为 $M(s) = S(s, 0)$，不一定等于参考曲线 $C(s)$。直线参考方案通过高度场表达纵向弯曲，不能把参考直线长度直接当作叶长。

| 顺序 | 输入 | 处理 | 实现与输出 |
|---|---|---|---|
| 1 | 同名 TXT、CSV | 配对、读取、法向归一化 | `io.discover/load_pair`：点、标签、颜色、法向 |
| 2 | 单叶点与法向 | 去重、连通簇保留、逐断片 SOR | `geometry.clean_leaf_points`：保留点 |
| 3 | 茎秆点 | Z 分箱中位点、有界 P-spline | `fit_stem_centerline`：参考茎秆线 |
| 4 | 叶片点、茎秆线 | L1 骨架、测地排序、主曲线、范围回退 | `extract_midrib`：初始参考中脉 |
| 5 | 曲线、法向 | 双反射标架、测量法向滚转对齐 | `build_normal_aligned_frame`：局部标架 |
| 6 | 初始几何 | 鲁棒联合优化及多种参考曲线 | `joint.py`、`surface.fit_leaf`：几何候选 |
| 7 | 局部横向坐标 | 宽度包络、先验收缩、BIC 截断检验 | `morphology.py`：叶形和补全长度 |
| 8 | 高度与有效宽度 | 分箱 TPS、GML、叶尖闭合 | `surface.py`：连续曲面 |
| 9 | 候选曲面 | 局部翻转筛选、同点集贴合选优 | `surface.fit_leaf`：选中模型 |
| 10 | 模型与点云 | 连续性、网格、贴合复查，表型与导出 | `quality.py`、`pipeline.py` |

## 2. 配对读取和坐标约定

`discover` 递归读取首行以 `//X Y Z` 开头的 TXT，与同名 CSV 配对。CSV 提供 `X,Y,Z,semantic`，TXT 提供 RGB 和可用的 `Nx,Ny,Nz`。单位法向为：

$$
\widehat{n}_i = n_i / \max(\|n_i\|_2, 10^{-12}).
$$

两份文件必须具有相同点数、点序和坐标，因为颜色与法向依赖逐行对应。本次本地 14 对输入逐行坐标差为零；这不替后来加入的文件作保证。茎秆拟合默认 Z 轴为高度方向。

## 3. 去重、断片保留与稳健去噪

### 3.1 半径连通图

按完整 XYZ 精确去重，同步保留法向。最近非自身点距中位数记为 $d_{\mathrm{sp}}$，包围盒对角线长度记为 $D$。建图半径为：

$$
r = \max(5\,d_{\mathrm{sp}}, 0.03\,D).
$$

使用 `cKDTree.sparse_distance_matrix` 构建无向半径图，保留最大连通分量，以及点数满足 $N_c \ge \max(20, \lfloor 0.01\,N \rfloor)$ 的分量。这允许稀疏的断开末段继续参与拟合。簇阈值是工程筛选规则，不是植物学定义。

### 3.2 各断片独立 SOR

在每个保留分量内取最多 12 个非自身近邻，计算每点的近邻平均距离 $\bar{d}_i$：

$$
m = \operatorname{median}(\bar{d}_i), \qquad \sigma_{\mathrm{rob}} = 1.4826\operatorname{median}|\bar{d}_i - m|,
$$

$$
\tau = \max\{m + 2\max(\sigma_{\mathrm{rob}}, 0.5\operatorname{std}(\bar{d})),\; 2.2\,m\}.
$$

保留 $\bar{d}_i \le \tau$ 的点。少于 35 点不做 SOR；若一次筛选剩余不足 $\max(35, 0.85\,N_c)$，取消此次筛选。之后不再按全叶密度或新的簇点数阈值删除末段。

邻域统计思想参见 [PCL 官方 SOR 说明](https://pointclouds.org/documentation/classpcl_1_1_statistical_outlier_removal.html)。本实现是分断片、中位数/MAD 的稳健变体，不等同于原版全局均值/标准差 SOR，也不依赖 PCL 软件包。

## 4. 公共拟合器：有界 P-spline 与 GCV

给定节点参数 $t_i$、观测 $Y_i$ 和 B-spline 基矩阵 $B$，求系数 $A$：

$$
\min_A \|W^{1/2}(BA - Y)\|_F^2 + \lambda\|D_2 A\|_F^2.
$$

$D_2$ 是控制系数的二阶差分。默认三次样条、请求 10 个基函数，实际次数/基数受节点数限制；内节点均匀、端点夹持。默认权重为 1。无界平滑器为：

$$
H_\lambda = B_w(B_w^T B_w + \lambda D_2^T D_2 + 10^{-12}I)^{-1} B_w^T,
$$

$$
\mathrm{GCV}(\lambda) = \frac{\operatorname{mean}|Y_w - B_w A_\lambda|^2}{\max(1 - \operatorname{tr} H_\lambda / n, 10^{-4})^2}.
$$

在 $\log_{10}\lambda$ 上作有界一维优化，默认区间 $[10^{-3}, 10^4]$，各调用可另设。`bounded=True` 时再用堆叠最小二乘，将每轴控制系数限制于输入坐标范围外扩 2% 的盒内，坐标跨度下限为 $10^{-5}$。

GCV 选参使用无界线性解，最终系数可以有边界约束，因此是有界拟合的选参代理，不是对活动约束解的精确 GCV。差分惩罚背景见 [Eilers & Marx (1996)](https://doi.org/10.1214/ss/1038425655)。

## 5. 茎秆参考线

按 Z 分为 $\min(50, \max(8, \lfloor N / 40 \rfloor))$ 箱；每箱至少三点才取三维中位点 $P_j$。用弦长参数：

$$
t_0 = 0, \qquad t_j = t_{j-1} + \|P_j - P_{j-1}\|_2.
$$

调用有界 P-spline，保存样条参数并采样 160 点。它主要用于判断叶根方向，这一步不生成茎秆实体表面。

## 6. 骨架、测地排序与初始中脉

### 6.1 L1 中位收缩

随机种子默认 2026。最多取 2400 个参考点 $Q$，通过最远点采样布置最多 180 个种子 $X$，请求数量为 $\min(180, \max(24, \lfloor \lvert Q \rvert / 10 \rfloor))$，参考点不足时全取。

带宽 $h$ 从 $\max(0.12\,D, 6\,d_{\mathrm{sp}})$ 开始，每两轮扩大 1.15 倍，上限为 $\max(0.24\,D, h_{\mathrm{initial}})$，共 10 轮。定义：

$$
\theta(d) = e^{-4d^2/h^2}, \quad \alpha_{ij} = \frac{\theta(\|x_i - q_j\|)}{\max(\|x_i - q_j\|, 10^{-4}h)},
$$

$$
A_i = \frac{\sum_j \alpha_{ij} q_j}{\sum_j \alpha_{ij}}, \quad \beta_{ii'} = \frac{\theta(\|x_i - x_{i'}\|)}{\max(\|x_i - x_{i'}\|^2, (0.06h)^2)}.
$$

局部加权协方差最大特征值占总和的比例为方向线性度 $\sigma_i$，更新：

$$
x_i \leftarrow A_i + 0.28\,\sigma_i \frac{\sum_{i' \ne i} \beta_{ii'}(x_i - x_{i'})}{\sum_{i' \ne i} \beta_{ii'}}.
$$

斥力向量长度先截到 $0.25h$。该实现采用 [Huang et al. (2013)](https://doi.org/10.1145/2461912.2461913) 的局部 L1 吸引/方向斥力思想，没有实现原论文完整的分支识别和骨架网络拓扑恢复。

### 6.2 测地排序与缺口桥接

在骨架点上建立 8 近邻无向图，边权为欧氏距离；通用 `geodesic_order` 默认 12 近邻。若不连通，从最大分量开始，反复通过最近点对连接另一个分量。距茎秆最近的骨架点作为起点，运行 Dijkstra 得到排序坐标。分箱取三维中位点，再拟合曲线。桥接是几何插值假设，不是缺口中存在实测点的证据。

### 6.3 弧长与点投影

沿样条取 400 点，按折线累计长度重新参数化并拟合 P-spline，属于数值近似弧长。投影先从 300 个曲线采样点找初值，再作 5 次牛顿更新：

$$
s \leftarrow s - \frac{(C(s) - p)\cdot C'(s)}{C'(s)\cdot C'(s) + (C(s) - p)\cdot C''(s)}.
$$

单步截到 $\pm 0.05\,L$，参数截到 $[0, L]$；有限迭代不保证任意曲线上的全局最近点。主曲线步骤作两轮“投影—弧长分箱中位点—样条—弧长重参数化”。

### 6.4 端点与范围回退

初始两端暂沿切线延伸 15%，用全体点重投影的 0.5%–99.5% 参数分位数确定细化区间，减轻骨架收缩。这不是最终补全长度。

若参考曲线在点云第一主轴上的范围不足点云 1%–99% 分位跨度的 95%，或沿该轴的累计回退超过跨度的 8%，改用 32 个主轴截面中位点拟合。主轴回退适用于沿该轴没有真实折返的叶片，是工程策略。

## 7. 双反射标架与法向对齐

单位切线为 $T = C' / \Vert C' \Vert$。定义反射 $R_a(x) = x - 2a(a\cdot x)/(a\cdot a)$，先用 $a = P_{i+1} - P_i$ 反射当前参考法向和切线，得到 $n_L, t_L$；再用 $b = T_{i+1} - t_L$ 作第二次反射得到下一法向。退化反射向量采用正交投影回退。横向轴由 $B = N \times T$ 给出。方法见 [Wang et al. (2008)](https://i.cs.hku.hk/GraphicsGroup/publications/pdf/Computation%20of%20rotation%20minimizing%20frames.pdf)。

沿弧长设置 10–30 个窗口，将扫描法向投影到切线的正交平面，取散度矩阵主方向，消除正负号影响。没有扫描法向时以局部截面协方差方向回退。相对参考标架的滚转角用：

$$
\phi_{\mathrm{raw}} = \frac{1}{2}\operatorname{unwrap}\bigl(2\operatorname{atan2}(\sin\phi, \cos\phi)\bigr).
$$

三点中值滤波后，用最多 8 个基函数、$\lambda \in [10^{-2}, 10^4]$ 的 P-spline 平滑，再旋转参考标架。初始 RMF 是最小旋转参考，加入测量滚转后并非严格零滚转标架。查询时线性插值后再归一化、正交化。

## 8. 联合几何优化实际计算什么

37 维状态包含：横向位移 8、法向位移 8、滚转 6、一次/二次横向拱度各 5、叶宽形状参数 5。最多取 1200 个沿弧长分层点，以 22 箱提取叶缘锚点。

$$
C_\Theta = C_0 + \delta_b(u) b_0 + \delta_n(u) n_0, \quad q_\Theta(u, v) = a_1(u)v + a_2(u)v^2.
$$

残差包括：

1. 法向高度 $[h_i - q_\Theta(u_i, v_i)] / w_{\mathrm{scale}}$，点级权重 $1/\sqrt{N}$。
2. 预测单位法向与扫描法向叉积长度，权重 $0.35/\sqrt{N}$。预测法向使用拱度解析偏导与固定参考切线，是局部近似，未包含完整标架导数。
3. 左右越界、较完整侧对齐和 `tanh` 饱和双侧平衡项，降低单侧大缺口对中脉的牵引。
4. 中脉二阶差分、位移锚定、滚转平滑、拱度与叶形先验项。

采用 `least_squares(loss='soft_l1', f_scale=0.08, max_nfev=65)`；标量鲁棒损失为 $\rho(z)=2(\sqrt{1+z}-1)$。优化器 Jacobian 为 SciPy 默认有限差分，不是自动微分或完整解析 Jacobian。

当前最终只提取优化后的两类中脉位移，重新拟合曲线、投影并估计标架；最终叶形和 TPS 继续在后续步骤重算。优化中的滚转、拱度、叶宽是辅助状态，不应称为最终曲面直接沿用的全部联合参数。

## 9. 宽度剖面、形态先验和 BIC

### 9.1 局部物理坐标

$$
r_i = (p_i - C(s_i))\cdot b(s_i), \quad h_i = (p_i - C(s_i))\cdot n(s_i).
$$

沿参考弧长分 32 箱，至少六点才使用。取 $w_L = \max(0, -Q_{0.02}(r))$、$w_R = \max(0, Q_{0.98}(r))$。一般 $w_{\mathrm{obs}} = w_L + w_R$；两侧差超过较大侧的 75% 时，用 $2\max(w_L, w_R)$ 估计缺失侧。

### 9.2 分段幂函数

令 $u = s / L$，最宽位置为 $x_m$：

$$
w_D(s) = \begin{cases}
w_{\max}\bigl[1 - (1 - k_{\mathrm{lig}})\bigl((x_m - u)/x_m\bigr)^{c_2}\bigr], & u \le x_m, \\
w_{\max}\bigl[1 - \bigl((u - x_m)/(1 - x_m)\bigr)^{c_1}\bigr], & u > x_m.
\end{cases}
$$

$c_1, c_2 > 1$ 时最宽处一阶连续，叶尖宽度为零。该参数化借鉴 [Dornbusch et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3077976/) 的基部/尖部分段建模思路；本实现的先验和参数边界不作为对论文逐参数复现的声明。轮廓解析面积系数为：

$$
\alpha = x_m\left(1 - \frac{1 - k_{\mathrm{lig}}}{c_2 + 1}\right) + (1 - x_m)\frac{c_1}{c_1 + 1}.
$$

默认参数均值 $(x_m, k_{\mathrm{lig}}, c_1, c_2) = (0.36, 0.62, 1.85, 2.00)$，标准差 $(0.10, 0.16, 0.45, 0.55)$，属于当前工程默认值，不声称每个数都是文献直接测量值。

### 9.3 数据集先验收缩

至少 800 个原始点、末端宽度比例小于 0.22 的候选完整叶先禁止延长拟合。再保留 $0.20 < x_m < 0.55$、$1.2 < c_1 < 3.2$、$0.65 \le \alpha \le 0.82$ 的结果。少于三个有效样本用默认先验，否则以 $\omega = n / (n + 10)$ 将样本中位数收缩到默认均值；标准差类似收缩并保留下限。这是经验先验步骤，不是独立的大样本物种估计。

### 9.4 是否延长叶尖

先拟合 $M_0$：固定 $L = L_{\mathrm{obs}}$ 的五参数模型，`soft_l1`、`f_scale=0.12`、`max_nfev=400`，先验残差权重 0.08。若允许补全且末端宽度比 $t > 0.22$，由先验反演：

$$
u_{\mathrm{end}} = \mu_x + (1 - \mu_x)\max(1 - \min(t, 0.85), 0.05)^{1/\mu_{c1}},
$$

$$
L_{\mathrm{prior}} = \operatorname{clip}\left(\frac{L_{\mathrm{obs}}}{\max(u_{\mathrm{end}}, 0.65)}, L_{\mathrm{obs}}, 1.45\,L_{\mathrm{obs}}\right).
$$

$M_1$ 自由长度上限为 $\min(1.45\,L_{\mathrm{obs}}, 1.15\,L_{\mathrm{prior}})$，同时估计长度和五个形状参数，先验权重提高至 0.45。使用归一化宽度数据残差：

$$
\mathrm{BIC} = n\log(\mathrm{RSS}/n) + k\log n.
$$

只有 $\mathrm{BIC}(M_1) + 4 < \mathrm{BIC}(M_0)$ 才延长。它是模型选择指标，不是缺失末端真实长度的直接观测。

## 10. 中脉外推与实测宽度支持

### 10.1 中脉外推

拟合最后 25% 参考曲线的俯仰角 $\theta = \arcsin(t_z)$ 对弧长的斜率，取 $\kappa \leftarrow \min(\kappa, 0)$；保持末端平均水平方位：

$$
\theta(\delta) = \max\bigl[-0.48\pi,\; \theta_{\mathrm{end}} + \kappa\delta e^{-0.5\delta/L_{\mathrm{obs}}}\bigr].
$$

离散积分单位切线，连接观测与外推点重新拟合曲线，并延续参考标架。指数阻尼、向下限制是本项目工程假设，不是某篇玉米中脉论文公式的逐式复现。

### 10.2 有效宽度

为避免全局轮廓裁掉实测叶缘，每箱计算：

$$
e_j = \max\bigl[2\max(w_{L,j}, w_{R,j}) - w_D(s_j),\; 0\bigr], \quad w_{\mathrm{eff}}(s) = w_D(s) + e(s).
$$

PCHIP 连接修正值，起点延用首箱值，观测末端补零，观测区外修正为零。`axis`、`linear` 候选先用最多 8 基函数、$\lambda \in [0.1, 100]$ 平滑修正轮廓，减少锯齿。这仍不能唯一恢复两侧同时缺失的形状。

## 11. 分箱中位 TPS 与谱 GML

训练坐标是 $u = s / L_{\mathrm{obs}}$、$v = \operatorname{clip}(2r / w_{\mathrm{eff}}(s), -1.25, 1.25)$。以 `(round(40u), round(12v))` 分箱，各箱取 $u, v, h$ 中位数；默认最多 350 个代表点，超过时按种子 2026 抽样。

$$
q(x) = \sum_j a_j\phi(\|x - x_j\|) + b_0 + b_1 u + b_2 v, \quad \phi(r) = r^2\log r, \quad \phi(0) = 0.
$$

构造核矩阵 $K$、多项式矩阵 $P = [1, u, v]$。对 $P$ 完整 QR 分解，以 $Q_2$ 表示多项式空间正交补：

$$
Q_2^T K Q_2 = U\operatorname{diag}(\eta_j)U^T, \quad z = U^T Q_2^T h.
$$

在 $\lambda \in [10^{-7}, 10^2]$ 上最小化：

$$
\mathrm{GML}(\lambda) = \left(\sum_j \frac{z_j^2}{\eta_j + \lambda}\right)\exp\left[\frac{1}{m - 3}\sum_j \log(\eta_j + \lambda)\right].
$$

$$
a = Q_2 U [z / (\eta + \lambda)], \quad b = R_1^{-1} Q_1^T(h - K a - \lambda a).
$$

`ThinPlateSplineGCV` 为兼容保留的类名，实际选择准则为 GML。GML/GCV 是不同准则，均不保证永不欠平滑或过拟合，参见 [Wahba (1985)](https://doi.org/10.1214/aos/1176349743)。

### 11.1 叶尖高度闭合

仅使宽度为零不够，横向 TPS 高度也必须收敛到一点。令 $q_0(s) = q(s, 0)$：

$$
a(s) = \operatorname{clip}\left(\frac{w_{\mathrm{eff}}(s)}{0.1\,w_{\max}}, 0, 1\right), \quad g(a) = a^2(3 - 2a),
$$

$$
q_{\mathrm{tip}}(s, v) = q_0(s) + g(a(s))\bigl[q(s, v) - q_0(s)\bigr].
$$

从而 $S(L, v) = S(L, 0)$。外推段再按 $w(s)/w(L_{\mathrm{obs}})$ 衰减法向残差。NPZ 保存 `apex_taper` 标志；旧文件缺少该标志时使用旧规则，保证审计读取旧文件时不会悄悄改变其几何。

## 12. 候选曲面、折叠筛选与选优

每片叶构造五种候选，不按植株名或叶号硬编码：

| 模式 | 参考曲线 | 联合位移 |
|---|---|---|
| `joint` | 初始化后联合几何细化 | 使用 |
| `initial` | 初始参考曲线 | 不使用 |
| `smoothed` | 初始曲线取 80 点，以 12 基函数、$\lambda\in[0.1,100]$ 再平滑 | 不使用 |
| `axis` | 第一主轴 24 箱中位点，8 基函数、$\lambda\in[1,10^4]$ 强平滑 | 不使用 |
| `linear` | 保留点第一主轴范围上的直线参考；纵向弯曲由高度场表示 | 不使用 |

### 12.1 局部翻转

用 241×61 参数网格，$s/L\in[0.01,0.99]$、$v\in[-0.98,0.98]$，以二阶精度差分计算：

$$
J_n = (\partial_s S \times \partial_v S)\cdot n(s), \quad c_n = J_n / \|\partial_s S \times \partial_v S\|.
$$

$c_n < -10^{-6}$ 为反向采样。先排除有反向采样的候选，再对剩余候选使用相同保留输入点、相同 220×61 曲面采样，计算点到面的近邻距离：

$$
d_i = \min_j \|p_i - S_j\|_2.
$$

仅用这一方向会偏好无观测支撑的宽大叶面。另在 90×25 网格的观测参数范围 $s_f \le L_{\mathrm{obs}}$ 内，对每个三角形重心 $c_f$ 计算到保留点的最近距离 $e_f = \min_i \Vert c_f - p_i \Vert_2$，以三角形面积 $A_f$ 加权：

$$
D_{\mathrm{surf}} = \frac{\sum_{f:s_f \le L_{\mathrm{obs}}} A_f e_f}{\sum_{f:s_f \le L_{\mathrm{obs}}} A_f}, \qquad \mathrm{score} = Q_{0.95}(d) + 0.5\,Q_{0.50}(d) + D_{\mathrm{surf}}.
$$

分数最小者胜出。观测范围外的补尖没有点云，故不进入该支持度项。三个系数是本项目的工程选择；这仍是扫描内选优，不是独立测试误差；采样 Jacobian 也不是全局三角形自交的数学证明。

若最优候选的保留点 P95 仍大于 $3\,d_{\mathrm{sp}}$，再把代表点上限从 350 提高到 700，重复五候选比较。只有加密结果的 P95、中位距离、观测段面积加权反向距离都不增加才接受；否则保留原结果。通过 `--max-tps-points`、`--refine-tps-points` 控制两档上限；后者不大于前者时关闭加密。几何和连续性检查仍必须通过，候选全部失败或最终检查失败时终止该叶的导出。

### 12.2 断点、网格与贴合复查

`quality.py` 检查以下项目：

- 有限顶点、正面积三角形、一个网格连通分量、一个边界连通环、无非流形边。
- 41 个横向参数对应的叶尖位置必须收敛，容差 $\max(10^{-8}, 10^{-7}L)$。
- 在曲线节点、标架位置、宽度节点和观测/外推交界，计算两侧距离；步长从 $10^{-6}L$ 缩为其 1% 后，细/粗距离最大比应小于 0.1。位置连续时距离应随步长收缩，真正跳变则不会；不要求所有拼接高阶导数连续。
- 保留点 P95 超过 $\max(6\,d_{\mathrm{sp}}, 0.15\,w_{\max,\mathrm{eff}})$ 标记为贴合异常。这是工程告警线，不是统计置信区间。原始点与保留点的距离均保存，避免只展示去噪后的有利数字。
- 保存观测段面积加权反向距离，核查扩大叶面来吸收点云的假改善；该项是选优指标，不宣称已证明未观测区的真实形状。

## 13. 网格与观测支持分类

默认纵向 90、横向 25；前 89 行作规则三角化，最后接唯一叶尖顶点构成扇形，避免零宽末端重复点。对观测参数 $(s_i, r_i)$ 建 Delaunay 凸包，并使用三维近邻半径：

$$
R_{\mathrm{sup}} = \max\{2.5\,d_{\mathrm{sp}},\; 0.75\max(\Delta s, \Delta r)\}.
$$

凸包内且靠近观测点为 `observed`；凸包内但缺少近邻支持为 `interpolated`；凸包外或超出观测范围为 `extrapolated`。面取三个顶点中最保守标签。这是几何支持分类，不是真值标签。

## 14. 表型计算和文件内容

$$
A_f = \frac{1}{2}\|(P_2 - P_1)\times(P_3 - P_1)\|_2, \quad A = \sum_f A_f.
$$

各支持区比例为对应面面积除以总面积，补全比例为插值加外推比例。最大宽为有效横向宽度的最大采样值，不是拱曲截面的测地宽度。

最终中心轨迹 $M(s) = S(s, 0)$ 取 500 个参数位置并插入观测终点，累计：

$$
\ell_k = \sum_{j=1}^k \|M(s_j) - M(s_{j-1})\|_2.
$$

`observed_length`、`completed_length` 记录观测段、全段中心轨迹的数值弧长，`tip_completion_length` 为差；`parameter_observed_length`、`parameter_completed_length` 另存参考坐标跨度。叶基倾角为最终中心轨迹起始切向与 Z 轴的夹角。

`form_factor_alpha` 是第 9 节先验轮廓的解析系数，不是最终曲面面积除以长宽的系数。`fit_rmse` 是按参考曲线投影坐标回代曲面的 RMS，不等于全局最近曲面距离；审计另列采样最近距离。

NPZ 保存样条、标架、叶形、TPS、局部宽度修正及闭合版本；JSON 保存表型、候选比较；PLY/OBJ 保存网格；HTML/PNG 用于查看。`outputs/maize_v2` 是兼容目录名，不表示使用旧版算法。

## 15. 重建与复现

```bash
python scripts/run_tests.py
python scripts/run.py --output outputs/maize_checked
python scripts/validate_fit.py --baseline outputs/previous --candidate outputs/maize_checked --output outputs/maize_checked/fit_comparison.json
python scripts/visualize.py --root outputs/maize_checked
```

先归档旧结果，再生成新目录，检查通过后更新默认展示。审计应保留输入/模型/源码哈希、网格参数和逐叶记录。原始数据与生成大文件按 `.gitignore` 保留本地，不提交源代码仓库。

`scripts/validate_holdout.py` 将真实点云分别遮挡孔洞、叶缘和末端，再评估留出点。完整输入的参考面仍由算法拟合，不能作为独立测量的真叶长或真面积；旧版本的留出数字也不能当成本版验证结果。

## 16. 方法来源与工程改动的区分

| 来源 | 使用思想 | 项目自行规定的部分 |
|---|---|---|
| [PCL SOR](https://pointclouds.org/documentation/classpcl_1_1_statistical_outlier_removal.html) | 邻域统计去噪 | 分断片、MAD、保留比例和簇阈值 |
| [Eilers & Marx, 1996](https://doi.org/10.1214/ss/1038425655) | B-spline 差分惩罚 | 盒约束、基数和搜索区间 |
| [Huang et al., 2013](https://doi.org/10.1145/2461912.2461913) | L1 吸引与方向斥力 | 单叶采样、带宽、迭代上限、范围回退和桥接 |
| [Wang et al., 2008](https://doi.org/10.1145/1330511.1330513) | 双反射参考标架 | 扫描法向滚转和插值 |
| [Dornbusch et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3077976/) | 基段/尖段叶形参数化 | 当前幂函数、先验、BIC、局部宽度修正 |
| [Wahba, 1985](https://doi.org/10.1214/aos/1176349743) | GML 选参 | 中位代表点、谱实现、搜索区间 |

主轴/直线参考、候选选择、叶尖高度闭合、断点检查和指数阻尼外推均是工程实现，应与文献基础区分。程序没有基于论文名称自动获得拓扑正确或生物真实性保证。

## 17. 本次全量复查记录（2026-11-24）

在 14 株、90 片已分割叶上，使用同一原始点集比较更新前后的生成模型。逐叶输入与源码 SHA-256、参数、距离和导出一致性见 [验证记录](validation/leaf_quality_2026-11-24.json)。这些数字是扫描内重建检查；未观测区域没有独立真值。

| 检查项 | 更新前 | 更新后 |
|---|---:|---:|
| 采样局部反向的叶片数 | 31 | 0 |
| 叶尖未闭合的叶片数 | 63 | 0 |
| 本文网格、连续性与贴合联合检查通过数 | 16 | 90 |
| 保留点到曲面 P95 的逐叶中位数 | 0.012182 | 0.010582 |
| 观测段曲面到点云面积加权距离的逐叶中位数 | 0.006913 | 0.006236 |

90 片中，71 片保留点 P95 下降，67 片反向支持距离下降；没有叶片的最大宽度超过其旧模型的两倍。`1-1-2` 叶 2 的保留点 P95 为 0.009150（旧 0.009881）；`2-2-2` 叶 4 为 0.010693（旧 0.017046）。全部 90 个单叶 PLY 与对应连续模型的最大坐标差为 $5.0\times10^{-9}$，整株 PLY 与单叶拼接顶点一致。局部采样和上述工程阈值仍不能证明无全局自交，也不能验证无观测区域的真实叶形。
