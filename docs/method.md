# 数学模型与理论依据（V2.0）

本项目的所有核心建模与补全步骤均建立在植物形态学、微分几何与数值分析的经典文献之上，不使用固定比例的硬编码叶尖延伸或手工平滑阈值。

## 1. 茎秆与中脉骨架：P-spline + GCV + Principal Curve

- **参考文献**：Eilers & Marx (1996, *Statistical Science*)；Wahba (1990, *Spline Models for Observational Data*)；Hastie & Stuetzle (1989, *JASA*)；Piegl & Tiller (1997, *The NURBS Book*)。
- **方法**：
  1. 以 kNN 无向图 + Dijkstra 最短路径从茎秆附着点建立叶片内在测地排序，获得初始中脉。
  2. 采用带二阶差分惩罚的 B-spline（P-spline），惩罚权重 $\lambda$ 由**广义交叉验证（GCV）**自动优化：
     $$\mathrm{GCV}(\lambda) = \frac{\frac{1}{n}\|(I - H(\lambda))Y\|_F^2}{\left(1 - \frac{1}{n}\mathrm{tr}(H(\lambda))\right)^2}$$
  3. 对曲线做真实弧长重参数化 $s \in [0, L_{\mathrm{obs}}]$，并用牛顿迭代做连续精确正交最近点投影（Piegl & Tiller, 1997），执行主曲线（Principal Curve）迭代细化中脉 $C(s)$。

## 2. 无扭转局部标架：Double Reflection RMF + 点云法向对齐

- **参考文献**：Bishop (1975, *Amer. Math. Monthly*)；Wang et al. (2008, *ACM TOG*, "Computation of Rotation Minimizing Frames")。
- **方法**：
  1. 沿中脉弧长采样点用**双反射法（Double Reflection）**计算严格无扭转的参考最小旋转标架 $(T_0(s), B_0(s), N_0(s))$。
  2. 直接利用点云自带单位法向量 `(Nx, Ny, Nz)` 投影到切线正交平面，求局部叶面法向并转换为相对 RMF 的连续滚转角 $\phi(s)$，用 GCV P-spline 平滑后旋转参考标架得到正交标架 $(t(s), b(s), n(s))$，避免横截面 SVD 导致的非物理翻转与扭转。

## 3. 叶宽包络与缺失叶缘/叶尖补全：Dornbusch (2011) + Prévot (1991)

- **参考文献**：
  - Dornbusch, Watt, Baccar, Fournier & Andrieu (2011, *Annals of Botany* 107(5): 865–873)。
  - Stewart & Dwyer (1999, *Crop Science* 39: 422–427)；Montgomery (1911)。
  - Prévot, Aries & Monestiez (1991, *Agronomie* 11(6): 491–503)；Fournier & Andrieu (1998, ADEL-Maize)。
- **分段幂函数叶宽模型**（归一化弧长 $u = s / L \in [0, 1]$）：
  $$w(s) = \begin{cases} w_{\max}\left[1 - (1 - k_{\mathrm{lig}})\left(\dfrac{x_m - u}{x_m}\right)^{c_2}\right], & 0 \le u \le x_m \\[10pt] w_{\max}\left[1 - \left(\dfrac{u - x_m}{1 - x_m}\right)^{c_1}\right], & x_m < u \le 1 \end{cases}$$
  其精确解析面积形状系数为：
  $$\alpha = \frac{1}{L\,w_{\max}}\int_0^L w(s)\,ds = x_m\left(1 - \frac{1 - k_{\mathrm{lig}}}{c_2 + 1}\right) + (1 - x_m)\frac{c_1}{c_1 + 1}$$
- **缺失推断**：
  1. 独立提取左右半宽包络；仅当一侧存在严重缺失（超过 75% 宽度缺损）时以完整侧包络修正宽度，避免单侧缺失拉偏。
  2. 先从当前批次数据集中近完整叶片以 **Empirical Bayes 收缩**标定玉米物种先验 $(x_m, k_{\mathrm{lig}}, c_1, c_2)$，再通过 MAP 鲁棒非线性最小二乘（soft-L1）分别拟合完整叶模型 $M_0$（$L = L_{\mathrm{obs}}$）与截断叶模型 $M_1$（$L \ge L_{\mathrm{obs}}$），按 **BIC 准则**（Schwarz, 1978；Kass & Raftery, 1995，$\Delta\mathrm{BIC} > 4$）决定是否进行叶尖延伸。
  3. 当判定叶尖截断（$L > L_{\mathrm{obs}}$）时，依据 Prévot (1991) 垂直叶面倾角演化模型拟合末端俯仰角曲率并按弧长光滑外推叶尖段中脉，同步以 RMF 延续标架。

## 4. 叶面起伏与孔洞补全：Duchon TPS + Wahba GML/REML 自动选参

- **参考文献**：Duchon (1977)；Wahba (1985, *Annals of Statistics* 13(4): 1378–1402；1990)；Wood (2011, *JRSS-B*)。
- **方法**：在归一化参数域 $(u, v)$ 上用二维薄板样条（Thin-Plate Spline，核函数 $\phi(r) = r^2 \ln r$）最小化薄弹性板弯曲能量：
  $$\min_q \sum_{i=1}^m \bigl(q(u_i, v_i) - h_i\bigr)^2 + \lambda \iint \bigl(q_{uu}^2 + 2q_{uv}^2 + q_{vv}^2\bigr)\,du\,dv$$
  通过多项式正交补空间 QR + 谱分解高效优化**广义最大似然（GML / REML）准则**自动求取最优平滑参数 $\lambda^*$（避免有限样本 GCV 过拟合）。最终连续参数曲面为：
  $$S(s, v) = C(s) + \frac{v\,w(s)}{2} b(s) + q(s, v) n(s), \quad s \in [0, L],\; v \in [-1, 1]$$
  在叶尖 $s = L$（$w(L) = 0$）处采用**单顶点扇形三角化**自然闭合，消除退化零面积三角面。
