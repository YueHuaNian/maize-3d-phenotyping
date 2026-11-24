# 方法原理与文献依据总览（V2.0）

本文档系统总结本项目（`src/maize_leaf/`）当前用于玉米点云三维叶片建模、缺失补全与表型提取的全部核心算法、数学公式、文献来源及其适用边界。

---

## 一、方法总览表

| 处理阶段 | 代码模块与函数 | 采用的方法 | 主要依据与文献来源 |
|---|---|---|---|
| **1. 数据读取与法向利用** | [io.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/io.py) (`load_pair`) | 配对 TXT / CSV 读取坐标、RGB、语义标签与扫描单位法向量 `(Nx, Ny, Nz)` | 保留原始扫描几何法向，用于后续标架滚转角对齐 |
| **2. 离群噪声簇过滤** | [geometry.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/geometry.py) (`filter_noise_clusters`) | 去重 + 欧氏近邻连通分量聚类滤波 | Rusu & Cousins (2011) 欧氏聚类滤波思想（PCL 标准预处理） |
| **3. 茎秆中心线重建** | [geometry.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/geometry.py) (`fit_stem_centerline`) | 沿 Z 轴分箱取三维中位数 + 弦长参数化有界 GCV P-spline | Eilers & Marx (1996)；Wahba (1990) |
| **4. 叶片几何中脉提取** | [geometry.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/geometry.py) (`geodesic_order`, `extract_midrib`, `project_points_to_curve`) | kNN 图测地距离初始化 + 有界 GCV P-spline + 弧长重参数化 + 牛顿精确正交投影 + 主曲线（Principal Curve）迭代细化 | Dijkstra (1959) / Tenenbaum et al. (2000) 测地流形思想；Hastie & Stuetzle (1989) 主曲线；Piegl & Tiller (1997) *The NURBS Book* |
| **5. 无扭转移动标架** | [geometry.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/geometry.py) (`double_reflection_rmf`, `build_normal_aligned_frame`) | 双反射最小旋转标架（RMF）+ 切线正交平面点云法向 SVD + `2φ` 相位解缠与中值/GCV 样条平滑 | Bishop (1975)；Wang et al. (2008, *ACM TOG*) |
| **6. 物种先验标定与叶宽模型** | [morphology.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/morphology.py) (`extract_width_profile`, `calibrate_prior_from_leaves`, `fit_dornbusch_blade`) | 左右独立分位包络 + Empirical Bayes 物种先验收缩 + MAP `soft_l1` 分段幂函数拟合 + BIC 叶尖截断模型选择 | Dornbusch et al. (2011, *Annals of Botany*)；Stewart & Dwyer (1999, *Crop Science*)；Montgomery (1911)；Schwarz (1978) & Kass & Raftery (1995) |
| **7. 截断叶尖中脉外推** | [morphology.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/morphology.py) (`extrapolate_midrib_and_frame`) | 垂直叶面倾角曲率阻尼外推 + 弧长积分 + RMF 延续 | Prévot, Aries & Monestiez (1991, *Agronomie*)；Fournier & Andrieu (1998, ADEL-Maize) |
| **8. 叶面起伏与孔洞补全** | [surface.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/surface.py) (`ThinPlateSplineGCV`, `MaizeLeafModel`) | 二维薄板样条（TPS）+ QR 谱分解广义最大似然（GML/REML）自动正则化 | Duchon (1977)；Wahba (1985, *Annals of Statistics*; 1990)；Wood (2011, *JRSS-B*) |
| **9. 网格生成与区域划分** | [surface.py](file:///e:/maize-3d-phenotyping/src/maize_leaf/surface.py) (`leaf_mesh_grid`, `fit_leaf`) | 叶尖单顶点扇形三角化 + 物理参数平面 Delaunay 凸包 + 三维近邻支持半径划分 | 消除退化零面积面并严格区分观测/插值/外推支持度 |

---

## 二、分步算法原理与公式

### 1. 欧氏聚类去噪（`filter_noise_clusters`）

- **来源**：点云欧氏聚类分割思想（Rusu & Cousins, 2011）。
- **做法**：先按精确坐标去重；以 $k=12$ 近邻中位点距 $d_{\mathrm{sp}}$ 与点云包围盒直径 $D$ 确定连通半径：
  $$r = \max(5\,d_{\mathrm{sp}},\; 0.03\,D)$$
  构建无向近邻图并求连通分量，仅保留最大主分量以及点数满足 $N_c \ge \max(30,\; 0.05\,N)$ 的主要叶片断片。
- **作用**：剔除扫描/语义分割中附着在茎秆附近或叶片外侧的微小误标注噪点簇，防止孤立小点簇劫持叶根起点选择。

---

### 2. 茎秆与叶片中脉提取

#### (1) 有界 GCV P-spline 曲线拟合（`fit_pspline_gcv`）
- **来源**：Eilers & Marx (1996) *Penalized B-splines (P-splines)*；Wahba (1990) 广义交叉验证（GCV）；B-spline 凸包性质。
- **原理**：采用带二阶差分惩罚的三次 B-spline（$n_{\mathrm{basis}} \le 10$）：
  $$\min_C \|W^{1/2}(BC - Y)\|_F^2 + \lambda \|D_2 C\|_F^2$$
  平滑权重 $\lambda$ 通过最小化 GCV 准则自动确定（不手工写死惩罚系数）：
  $$\mathrm{GCV}(\lambda) = \frac{\frac{1}{n}\|W^{1/2}(Y - B C(\lambda))\|_F^2}{\left(1 - \frac{1}{n}\mathrm{tr}(H(\lambda))\right)^2}$$
  对三维空间曲线拟合时（`bounded=True`），使用 `scipy.optimize.lsq_linear` 将各轴控制点约束在输入节点包围盒外扩 $2\%$ 范围内，利用 B-spline 凸包性防止局部稀疏间隔处产生样条过冲。

#### (2) 测地排序 + 主曲线 + 精确正交投影（`geodesic_order`, `extract_midrib`, `project_points_to_curve`）
- **来源**：Tenenbaum et al. (2000) / Dijkstra (1959) 测地流形排序；Hastie & Stuetzle (1989) 主曲线（Principal Curves）；Piegl & Tiller (1997, *The NURBS Book*) 最近点牛顿迭代投影。
- **原理**：
  1. **测地初始化**：抽取最多 4500 个点构造 12 近邻无向图（若存在断裂孔洞则按分量间最小距离桥接），以距茎秆中心线最近点为叶根，运行 Dijkstra 算法获得沿叶片下垂弯曲方向单调递增的测地坐标；沿测地坐标分箱取三维中位数拟合初始中脉，并按真物理弧长重参数化 $s \in [0, L_{\mathrm{obs}}]$。
  2. **精确正交投影**：对每个三维点 $p_i$ 求解最小距离正交条件 $f(s) = (C(s) - p_i)\cdot C'(s) = 0$，使用牛顿迭代更新：
     $$s_{t+1} = s_t - \frac{(C(s_t) - p_i)\cdot C'(s_t)}{C'(s_t)\cdot C'(s_t) + (C(s_t) - p_i)\cdot C''(s_t)}$$
  3. **主曲线迭代**：重复执行两轮“正交投影 $\rightarrow$ 弧长分箱取中位数 $\rightarrow$ 有界 GCV P-spline 重拟合 $\rightarrow$ 弧长重参数化”，使中脉收敛到点云的连续对称中心脊线。

---

### 3. 无扭转移动标架与点云法向对齐（`double_reflection_rmf`, `build_normal_aligned_frame`）

- **来源**：Bishop (1975) 平行输运标架；Wang et al. (2008, *ACM TOG*) 双反射最小旋转标架（Double Reflection RMF）。
- **原理**：
  1. 沿弧长采样点由双反射法计算严格无扭转、数值稳定的参考正交标架 $(T_0(s), B_0(s), N_0(s))$。
  2. 将原始点云自带的单位法向量 `(Nx, Ny, Nz)`（无自带法向量时回退到横截面协方差）在各截面处投影到切线正交平面 $\perp T_0(s)$，通过散度矩阵第一主方向估计不依赖正负朝向的局部叶面法向 $\tilde{n}(s)$。
  3. 将 $\tilde{n}(s)$ 表示为相对参考标架的滚转角 $\phi(s)$，利用倍角相位解缠 $\phi_{\mathrm{raw}} = \frac{1}{2}\mathrm{unwrap}(2\,\mathrm{atan2}(\sin\phi, \cos\phi))$ 消除 $180^\circ$ 法向反转跳变，经 3 点中值滤波和无边界 GCV P-spline 平滑得到 $\phi(s)$，旋转参考标架得到全叶光滑正交标架 $(t(s), b(s), n(s))$。

---

### 4. 叶宽轮廓、先验标定与叶尖缺失反演

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

### 5. 叶面起伏与孔洞补全：Duchon TPS + GML/REML 自动正则化

- **来源**：Duchon (1977) 薄板样条；Wahba (1985, *Annals of Statistics* 13(4): 1378–1402; 1990) 广义最大似然（GML / REML）；Wood (2011, *JRSS-B*)。
- **原理**：在 $(u_{\mathrm{norm}}, v_{\mathrm{norm}}) = (s / L_{\mathrm{obs}},\; 2r / w(s))$ 网格上（经 $40\times 12$ 空间分箱抽取最多 350 个代表点），用核函数 $\phi(d) = d^2\ln d$ 的二维薄板样条拟合相对标架的法向高度 $h = (p - C(s))\cdot n(s)$，最小化薄弹性板弯曲能量：
  $$\min_q \sum_{i=1}^m \bigl(q(u_i, v_i) - h_i\bigr)^2 + \lambda \iint \bigl(q_{uu}^2 + 2q_{uv}^2 + q_{vv}^2\bigr)\,du\,dv$$
  通过线性多项式矩阵 $P=[1,u,v]$ 的完整 QR 分解 $P=[Q_1,Q_2][R;0]$ 投影到正交补空间并对 $Q_2^\top K Q_2 = U\Lambda U^\top$ 做特征分解，直接优化 **GML/REML 目标函数**自动求解最优平滑参数 $\lambda^*$：
  $$\mathrm{GML}(\lambda) = \left(\sum_{j=1}^{m-3}\frac{z_j^2}{\lambda_j + \lambda}\right) \exp\!\left(\frac{1}{m-3}\sum_{j=1}^{m-3}\ln(\lambda_j + \lambda)\right)$$
  相比有限样本 GCV，GML/REML 从理论上避免了 $\lambda \to 0$ 时的过拟合欠平滑；当离散噪点多时自动增大 $\lambda^*$、降低有效自由度 `edf`。
- **最终连续参数曲面**：
  $$S(s, v) = C(s) + \frac{v\,w(s)}{2}\,b(s) + q(s, v)\,n(s), \quad s \in [0, L],\; v \in [-1, 1]$$
  在叶尖外推段，法向残差按 $w(s) / w(L_{\mathrm{obs}})$ 平滑衰减收拢至中脉。

---

### 6. 网格构造与区域支持划分（`leaf_mesh_grid`, `fit_leaf`）

- **单顶点扇形闭合**：在最后一行横向截面与唯一叶尖顶点 $(L, 0)$ 之间采用扇形三角剖分，避免 $w(L)=0$ 处产生重复退化点与零面积三角面。
- **支持区域标签**（`0=observed`、`1=interpolated`、`2=extrapolated`）：
  - 在物理参数平面 $(s_{\mathrm{proj}}, r)$ 上构建实测点二维 `Delaunay` 凸包；
  - 在三维空间中以支持半径 $R_{\mathrm{sup}} = \max(2.5\,d_{\mathrm{sp}},\; 0.75\max(\Delta u, \Delta v))$ 判定近邻支持：凸包外或 $s > L_{\mathrm{obs}}$ 记为 `extrapolated`，凸包内且三维近邻距离 $\le R_{\mathrm{sup}}$ 记为 `observed`，凸包内其余空洞记为 `interpolated`；三角面取三顶点中的最保守标签（$2 > 1 > 0$）。

---

## 三、主要参考文献

1. **Bishop, R. L. (1975).** There is more than one way to frame a curve. *The American Mathematical Monthly*, 82(3), 246–251.
2. **Dijkstra, E. W. (1959).** A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.
3. **Dornbusch, T., Watt, J., Baccar, R., Fournier, C., & Andrieu, B. (2011).** A comparative analysis of leaf shape of wheat, barley and maize using an empirical shape model. *Annals of Botany*, 107(5), 865–873.
4. **Duchon, J. (1977).** Splines minimizing rotation-invariant semi-norms in Sobolev spaces. In *Constructive Theory of Functions of Several Variables*, Springer, 85–100.
5. **Eilers, P. H. C., & Marx, B. D. (1996).** Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.
6. **Fournier, C., & Andrieu, B. (1998).** A 3D architectural and process-based model of maize development (ADEL-maize). *Annals of Botany*, 81(2), 233–250.
7. **Hastie, T., & Stuetzle, W. (1989).** Principal curves. *Journal of the American Statistical Association*, 84(406), 502–516.
8. **Kass, R. E., & Raftery, A. E. (1995).** Bayes factors. *Journal of the American Statistical Association*, 90(430), 773–795.
9. **Montgomery, E. G. (1911).** Correlation studies in corn. *Nebraska Agricultural Experiment Station Annual Report*, 24, 108–159.
10. **Piegl, L., & Tiller, W. (1997).** *The NURBS Book* (2nd ed.). Springer.
11. **Prévot, L., Aries, F., & Monestiez, P. (1991).** Modélisation de la structure géométrique du maïs. *Agronomie*, 11(6), 491–503.
12. **Rusu, R. B., & Cousins, S. (2011).** 3D is here: Point Cloud Library (PCL). *IEEE International Conference on Robotics and Automation (ICRA)*, 1–4.
13. **Schwarz, G. (1978).** Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.
14. **Stewart, D. W., & Dwyer, L. M. (1999).** Mathematical characterization of leaf shape and area of maize hybrids. *Crop Science*, 39(2), 422–427.
15. **Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000).** A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
16. **Wahba, G. (1985).** A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem. *The Annals of Statistics*, 13(4), 1378–1402.
17. **Wahba, G. (1990).** *Spline Models for Observational Data*. SIAM.
18. **Wang, W., Jüttler, B., Zheng, D., & Liu, Y. (2008).** Computation of rotation minimizing frames. *ACM Transactions on Graphics*, 27(1), Article 2 (pp. 2:1–2:18).
19. **Wood, S. N. (2011).** Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3–36.
