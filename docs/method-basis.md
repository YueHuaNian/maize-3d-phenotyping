# 数学与生理依据

本文说明[技术路线](路线.md)中每种方法为什么使用、具体怎么算、证据能支持到哪里。核对日期：**2026-11-25**；适用对象为当前代码和本地 14 株、90 片叶的数据。

**文献原理、几何定义、本项目参数分开说明。** 点云去噪和曲面拟合属于数值方法；叶角、叶形等有植物形态与生理研究背景；将这些量组合成当前三类标签，则是本项目的建模方案。

## 数学依据

### 1. 欧氏聚类：用空间连接识别零散块

**原理与选择理由：** 将点作为图的顶点，两点距离不超过半径就连边，求连通分量。它直接利用局部空间连接关系，适合先剔除独立小块；KD-tree 用于加速邻域搜索。原理见 [PCL 欧氏聚类文档](https://pointclouds.org/documentation/tutorials/cluster_extraction.html)。

本项目先对单叶去重，再计算：

```math
\delta=\mathrm{median}_i\min_{j\ne i}\|p_i-p_j\|,\qquad
D=\|Q_{0.99}(p)-Q_{0.01}(p)\|,\qquad
r=\max(5\delta,0.02D).
```

其中分位数逐坐标计算。距离项都具有长度单位，整体缩放点云时，连接半径也同比缩放。用分位范围代替最大最小范围，可以减少少量远处飞点对半径的影响。

**本地改动：** 最大块始终保留，其他块点数至少达到 `max(20, floor(0.01N))` 才保留。这里的系数和点数门槛来自本地标定，不是 PCL 规定的通用数值。源码：[`geometry.py::_euclidean_clusters`](../src/maize_leaf/geometry.py)。

### 2. MAD-SOR：用稳健邻域距离识别毛刺

**原理与选择理由：** SOR 根据点到近邻的平均距离识别稀疏离群点。本项目将普通均值、标准差阈值改为中位数、MAD 阈值，并逐断片计算，减少离群距离及块间密度差异的影响。原始 SOR 见 [PCL 文档](https://pointclouds.org/documentation/tutorials/statistical_outlier.html)，稳健尺度定义见 [NIST MAD 文档](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/mad.htm)。

```math
d_i=\frac1K\sum_{j\in N_K(i)}\|p_i-p_j\|,\quad
m=\mathrm{median}(d_i),\quad
\mathrm{MAD}=\mathrm{median}|d_i-m|.
```

保留条件为：

```math
d_i\le m+3.5\times1.4826\mathrm{MAD},\qquad K=8.
```

`1.4826 ≈ 1/Φ⁻¹(0.75)` 是正态分布下把 MAD 换成标准差尺度的系数；它不说明扫描误差一定正态。`K=8`、`3.5` 和删除过多时撤销的保护规则，是本项目设置。MAD 退化时回退到普通标准差。源码：[`geometry.py::sor_filter / clean_leaf_points`](../src/maize_leaf/geometry.py)。

### 3. 中心线与中脉：在贴合点和保持平滑之间取平衡

**P-spline：** 用 B 样条表达曲线，通过下列形式的目标拟合：

```math
\min_c\ \|W^{1/2}(y-Bc)\|^2+\lambda\|D_2c\|^2.
```

第一项要求曲线贴近数据，第二项抑制相邻系数的剧烈变化。程序用广义交叉验证（GCV）选择平滑强度，并在相应调用中约束系数范围。选择理由是扫描中心点有噪声，而后续角度计算需要连续方向。方法依据为 [Eilers 与 Marx（1996）](https://doi.org/10.1214/ss/1038425655)。这是数值平滑约束，没有假定玉米茎叶服从样条生长规律。

**L1 骨架：** 局部中心可由加权距离和最小化表示：

```math
\min_x\sum_j w_j\|x-p_j\|.
```

程序借鉴 [Huang 等（2013）](https://doi.org/10.1145/2461912.2461913)的局部 L1 吸引与排斥思想，再用近邻图路径排序、平滑曲线和主轴回退形成中脉。该实现包含项目自己的采样、连接和回退规则，文献不保证本地骨架一定等于解剖学中脉。

源码：[`geometry.py::fit_pspline_gcv / fit_stem_centerline / l1_medial_skeleton / extract_midrib`](../src/maize_leaf/geometry.py)。

### 4. 局部坐标与叶宽：把长条叶片写成可计算的表面

沿参考中脉建立切向、横向和法向。用 [Wang 等（2008）](https://doi.org/10.1145/1330511.1330513)的双反射方法生成最小旋转标架，再按扫描法向调整滚转。选择它是为了减少坐标架自身引入的突然转动；调整后的坐标架仍允许表达实测叶面扭转。

在活动参数域中，叶面可概括为：

```math
S(s,v)=C(s)+r(s,v)b(s)+q(s,v)n(s).
```

这里 `s` 表示沿叶长的位置，`v` 表示横向位置，`r` 控制左右宽度，`q` 表示相对参考面的起伏；源码另有左右边界缩放和叶尖收束处理。

**叶宽选择理由：** 草本长条叶片可用“宽度随纵向位置变化”描述。包含玉米实测样本的 [Dornbusch 等（2011）](https://doi.org/10.1093/aob/mcq181)支持将叶形分成基部、远端两段建模。本项目按叶基到叶尖的方向重新参数化，使用分段幂函数并结合左右观测包络，配置的先验均值不能冒充原论文实测常数。

源码的参考宽度函数分为两段。基部到最大宽度位置：

```math
\frac{w(u)}{W}=1-(1-k_{\mathrm{lig}})\left(\frac{x_m-u}{x_m}\right)^{c_2},\qquad 0\le u\le x_m.
```

最大宽度位置到叶尖：

```math
\frac{w(u)}{W}=1-\left(\frac{u-x_m}{1-x_m}\right)^{c_1},\qquad x_m<u\le1.
```

其中 `u=s/L`，`W` 为参考最大宽度，`x_m` 为最大宽度位置，`k_lig` 为基部宽度比例，`c₁、c₂` 控制两段的形状。实际叶宽还受观测包络约束。拟合将观测宽度残差和标准化先验残差一起交给 `soft_l1` 稳健损失处理，先验惩罚强度和参数边界属于本地正则化设置。

内部叶形拟合还比较固定长度和可延长长度模型，使用 `BIC=n·ln(RSS/n)+k·ln(n)`，要求 BIC 改善超过 `4` 才采用延长参数。[Kass 与 Raftery（1995）](https://doi.org/10.1080/01621459.1995.10476572)讨论了 BIC 与 Bayes 因子的近似关系；这里的稳健带先验拟合、分箱宽度残差和阈值 `4` 构成工程判据，不能直接解读成真实缺失长度的后验概率。默认流程随后限制活动域为扫描范围，不把内部估计的延长段用于最终测量。

源码：[`geometry.py::build_normal_aligned_frame`](../src/maize_leaf/geometry.py)、[`morphology.py::DornbuschBlade / extract_width_profile`](../src/maize_leaf/morphology.py)、[`surface.py::MaizeLeafModel.evaluate`](../src/maize_leaf/surface.py)。

### 5. 薄板样条：平滑表示叶面起伏

**原理与选择理由：** 对局部高度场拟合，在贴合代表点的同时惩罚过度弯折。二维薄板样条的核为 `φ(r)=r²ln(r)`，在零处连续取零；其经典平滑目标可写为：

```math
\min_f\ \sum_i(y_i-f(u_i,v_i))^2+
\lambda\iint(f_{uu}^2+2f_{uv}^2+f_{vv}^2)\,du\,dv.
```

薄板样条的变分依据见 [Duchon（1977）](https://doi.org/10.1007/BFb0086566)。本项目对分箱中位代表点拟合，用广义最大似然（GML）选择平滑强度；GCV 与 GML 的依据见 [Wahba（1985）](https://doi.org/10.1214/aos/1176349743)。

这里的“薄板”是数学平滑模型，程序没有输入弹性模量、厚度或载荷，因此没有求解真实叶片的力学变形。GML 也不保证所有缺失区域恢复正确。源码：[`surface.py::ThinPlateSplineGCV / _aggregate_tps_bins`](../src/maize_leaf/surface.py)。

### 6. 选模、收边和质量检查：用观测约束模型

**选择理由：** 单看“点到面”的距离，可能保留一张外扩过大的叶面，所以还比较“面到点”的支撑距离。

候选排序分数为：

```math
J=Q_{0.95}(d_{p\to S})+0.5Q_{0.50}(d_{p\to S})+
\mathrm{mean}_{A}(d_{S\to p}).
```

点到面距离由密集曲面采样点的最近距离近似，反向项按三角面面积加权。它是本项目的工程评分，不是概率似然；通过方向筛选后按分数顺序接受通过质量检查的候选。

5 个候选中的联合优化方案，把点位置、扫描法向、左右边缘、中脉弯曲及先验等残差组合后，用 `soft_l1` 最小二乘调整。残差组合及权重见 [`joint.py::joint_optimize_geometry`](../src/maize_leaf/joint.py)，属于本项目模型设计；是否采用这个候选由上述观测评分和质量检查决定。

左右边界来自各段横向坐标的 2%/98% 分位及点距余量，活动长度限制在观测范围。数学理由是缺失区域缺少直接观测约束，需要显式限定估计范围；该保守策略的本地证据见[遮挡测试](validation/holdout_2026-11-25.json)和[支撑域验证](validation/classification_feedback_2026-11-25.json)。

源码：[`surface.py::fit_leaf / restrict_to_scan_support`](../src/maize_leaf/surface.py)、[`quality.py`](../src/maize_leaf/quality.py)。

### 7. 叶长、面积与茎叶角：使用几何定义

令模型表面中心轨迹 `M(s)=S(s,0)`，叶长由弧长计算：

```math
L=\int\|M'(s)\|\,ds.
```

程序对中心轨迹采样、平滑后数值求长。三角面顶点为 `a,b,c` 时，面积及叶面积为：

```math
A_f=\tfrac12\|(b-a)\times(c-a)\|,\qquad A=\sum_f A_f.
```

令 `t̂_s` 为叶基附近向上的茎秆单位切向；将叶基前 15% 范围内的叶面纵向单位切向按面积平均，再归一化为 `t̂_b`。茎叶角为：

```math
\theta=\arccos(\hat t_s\cdot\hat t_b),\qquad 0\le\theta\le180^\circ.
```

**选择理由：** 弧长跟随弯曲叶片，面积累加跟随三维叶面；基部方向平均可降低单点导数波动。它保留对基部张开方向的测量，但整片叶尖的垂落不能完全由一个基部角度代表。植物依据见下文；15% 窗口属于项目估计设置。

源码：[`traits.py::midrib_pose_and_curvature / basal_surface_stem_leaf_angle`](../src/maize_leaf/traits.py)、[`surface.py::triangle_areas`](../src/maize_leaf/surface.py)。

### 8. 投影面积与高度：保持定义和单位清楚

**水平投影：** 由平面投影的余弦关系得到：

```math
A_\mathrm{proj}=\sum_f A_f|n_f\cdot e_z|,\qquad
P_i=\frac{A_{\mathrm{proj},i}}{\max_{j\in\text{同株}}A_{\mathrm{proj},j}}.
```

`n_f` 是面片单位法向。该值把各面片投影相加，重叠部分重复计入；叶片折叠、倾斜和实际大小都会影响结果。选择同株归一化是为了保留株内相对大小，不能把 `P` 单独解释成展开比例、真实面积比例或光合效率。投影并集另由光栅化计算，二者没有混用。

**相对着生高度与远端高差：**

```math
h_i=\mathrm{clip}\left(\frac{z_{\mathrm{base},i}-z_{\mathrm{stem,min}}}{z_{\mathrm{stem,max}}-z_{\mathrm{stem,min}}},0,1\right),\qquad
\Delta z_i=z_{\mathrm{far},i}-z_{\mathrm{base},i}.
```

其中 `z_far,i` 为叶面网格上距茎秆中心线三维欧氏距离最远顶点的 `Z` 坐标。选择 `h` 是为了提供株内位置线索并消除整体长度尺度（依赖扫描茎秆范围，不等于叶序或叶龄）；选择 `Δz` 是为了用带符号的高度差直接区分叶片远端整体上扬（`Δz>0`）还是下垂（`Δz<0`），补充无方向俯仰熵的不足。`Δz` 沿用输入点云坐标单位，随叶片与植株绝对尺寸变化。

源码：[`traits.py::element_projected_area / far_end_height_diff / extract_plant_phenotypes`](../src/maize_leaf/traits.py)。

### 9. 俯仰熵：度量方向分散，而非下垂程度

沿叶长的单位切向为 `t̂(s,v)`，俯仰角为：

```math
\varphi=\arcsin(\hat t\cdot e_z),\qquad -90^\circ\le\varphi\le90^\circ.
```

把该范围等分成 `B=12` 档，按叶面面积计算每档比例 `p_b`：

```math
p_b=\frac{\int_{\varphi\in b}dA}{\int dA},\qquad
H=-\frac{\sum_{b:p_b>0}p_b\ln p_b}{\ln B}.
```

熵定义依据 [Shannon（1948）](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)；面积权重使统计对象是叶面面积，而不是某一区域扫描点的数量。归一化后 `H∈[0,1]`。

**为何选用：** 它给基部角度补充整片叶方向变化的信息。**证据边界：** 将该熵用于玉米叶姿分类是本项目指标设计，当前没有独立生理标签验证。整片笔直向上、水平或向下，方向都可能集中在一档而得到低熵；将俯仰角正负整体颠倒也不改变熵。因此它不是“下垂量”，更不是植物代谢或热力学熵。

源码：[`entropy.py::surface_pitch_entropy_from_grid / model_pitch_entropy`](../src/maize_leaf/entropy.py)。

### 10. 加权 K-means++：按综合形态寻找原型

先稳健缩放，再定义加权平方距离：

```math
\tilde x_{ij}=\frac{x_{ij}-\mathrm{median}(x_j)}{\mathrm{IQR}(x_j)},\qquad
d_{ic}^2=\sum_{j=1}^{5}w_j(\tilde x_{ij}-\mu_{cj})^2,
\qquad w=(1,1,1,1,1).
```

IQR 退化时，代码回退到标准差或单位尺度。K-means 最小化样本到所分配中心的平方距离总和；K-means++ 用距离相关概率初始化中心，依据为 [Arthur 与 Vassilvitskii（2007）](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)。

**选择理由：** 当前目标是用少量形态特征建立可检查的典型中心，距离和中心都能解释。固定 `k=3` 对应项目预先定义的三类需求，不能据此推断玉米叶片天然存在三个离散生理状态。

按中心茎叶角从小到大命名原型后，再施加[路线文档中的类别规则](路线.md#11将叶片分成紧缩舒展下垂)。最终标签属于“原型距离 + 顶部小叶条件”的组合判定。高度权重 `1.0`、远端高差权重 `1.0` 和规则中的 `0.5、0.15、2/3` 都没有由上述论文推导出来。

源码：[`classification.py::RobustScaler / KMeansModel / contextual_predict`](../src/maize_leaf/classification.py)。

### 11. 其他输出：各自测量什么

以下量用于补充表型描述，当前主分类输入为 `θ、P、H、h、Δz`。

| 输出 | 数学定义或计算方式 | 可以解释什么 |
|---|---|---|
| 中脉曲率 | `κ=‖M′×M″‖/‖M′‖³`，积分得到总转弯程度 | 几何弯曲，不直接给出组织硬度 |
| 曲面曲率 | 由第一、第二基本形式求平均曲率 `H_c`、高斯曲率 `K_G`，并计算 `∫H_c²dA` | 叶面起伏和弯折；这里 `H_c` 与俯仰熵 `H` 含义不同，没有材料参数就不能称为实际弹性能 |
| 法向熵 | 将正负等价法向归到上半球，等立体角分箱后按面积计算 Shannon 熵；每格立体角为 `2π/B` | 补充叶面朝向分散程度；不是本项目主分类使用的俯仰熵 |
| 方向可见面积、遮挡比 | 沿指定方向投影，用深度缓冲保留最近面，再累计像素面积 | 几何可见性；未计算透射、散射、太阳历程或光合反应 |
| 冠层包络 | 按高度分层，取相对茎秆水平距离的 95% 分位 | 冠层展开范围；分箱数和分位值是统计口径 |
| 质量权重 | `q=C_obs·exp(−e95/τ)`；`τ` 默认取有效正误差的中位数 | 覆盖率与贴合误差构成的工程权重，不是样本为真的概率；不作为主 K-means 的样本权重 |

上述计算见 [`traits.py`](../src/maize_leaf/traits.py) 和 [`entropy.py`](../src/maize_leaf/entropy.py)；向量夹角、叉积面积、曲率来自几何定义，派生的生理意义需要额外实验建立。

法向分箱用球面面积元 `dΩ=d(n_z)·dφ` 保证面积一致，避免高纬度网格面积较小而改变统计口径。等面积分区思想可参见 [Leopardi（2006）](https://ftp.gwdg.de/pub/EMIS/journals/ETNA/vol.25.2006/pp309-327.dir/pp309-327.pdf)；本项目的上半球环带分区是自建实现，不是直接调用该论文的完整球面递归算法。

## 植物形态与生理依据

| 研究支持的事实 | 对当前选择的支持 | 不能据此推出什么 |
|---|---|---|
| 玉米叶片—叶鞘连接区的叶舌、叶耳及组织发育参与叶角形成；[Wang 等，2022](https://doi.org/10.1093/plphys/kiac308) | 在叶基附近测量方向，而不是用整片叶质心方向代替基部叶角 | 文献没有验证本项目的前 15% 平均窗口，也没有验证局部茎秆切向与文献所用角度完全等价 |
| 玉米等禾本科叶片的宽度轮廓可由基部和远端分段模型描述；[Dornbusch 等，2011](https://doi.org/10.1093/aob/mcq181) | 用纵向叶宽函数提供可拟合的叶形参照，再接受扫描边缘约束 | 不能据完整叶的模型准确恢复所有遮挡叶尖；不能把本地先验参数称为品种通用常数 |
| `drl1/drl2` 研究将中脉发育、叶片下垂及叶角变化联系起来；[Strable 等，2017](https://doi.org/10.1105/tpc.16.00477) | 中脉、叶角和叶片整体弯曲值得分别测量 | 下垂不等于衰老；该突变体研究也没有验证俯仰熵随下垂程度单调增加 |
| 同株叶片可处于不同发育阶段，研究区分展开叶与心叶筒中发育的叶片；[Facette 等，2013](https://doi.org/10.1105/tpc.113.112227) | 用位置和大小辅助寻找顶部小叶候选，是有发育背景的建模假设 | 不能仅凭相对高度和投影面积确定叶龄、是否完全展开，更推不出 `0.15、0.5` 的阈值 |

因此，当前输出应称为**形态分类**。将某类进一步命名为幼叶、成熟叶或衰老叶，需要叶序、叶领/叶舌状态、展开程度或相应生理测量进行人工标注与独立检验。

## 参数来源与验证状态

| 参数或决定 | 当前来源 | 已有证据及适用范围 |
|---|---|---|
| 半径 `max(5δ,0.02D)`、块点数门槛 | 本地参数对照后选定 | [欧氏聚类标定](validation/euclidean_cluster_calibration_2026-11-25.json)及 2026-11-26 支撑域联合复验；14 株、90 叶内部比较 |
| `K=8`、MAD 系数 `3.5` | 本地参数对照后选定 | [SOR 标定](validation/sor_calibration_2026-11-24.json)及 2026-11-26 支撑域联合复验；同批数据内部比较 |
| 茎秆样条 `n_basis<=8, lam∈[0.1,1e5]`、L1 骨架 `D_rob` 与 `μ=0.15` | 2026-11-26 解析真值与留出标定升级 | [全流程再标定审计](validation/recalibration_upgrade_2026-11-26.json)；茎叶角最大抽样抖动降 7.6 倍，飞点半径膨胀降至 `1.0005×` |
| 先验均值、32 段宽度、2%/98% 边界、`2.0δ` 收边余量 | 2026-11-26 90 叶 Pareto 标定微调（`2.5δ→2.0δ`） | [支撑范围检查](validation/classification_feedback_2026-11-25.json)与[再标定审计](validation/recalibration_upgrade_2026-11-26.json) |
| TPS 代表点上限 `700`（完整保留 `40×12` 分箱）、4 候选池（移除默认 37 维 `joint`）、排序分数及门禁 | 2026-11-26 90 叶/28 叶 20% 空间留出标定升级 | [全流程再标定审计](validation/recalibration_upgrade_2026-11-26.json)；90 叶全通过，中位 P95 降 6.2%，去除 `joint.py` 20+ 手工权重 |
| 基部 15% 窗口、熵 12 档、46 方向 Turtle Sky（SOC 加权）受光 | 窗口复验通过；受光替换为 den Dulk (1989) 文献标准无参方法 | 绕 Z 轴最大漂移由 `11.36%` 降至 `1.39%`（`ρ(0°,45°)=0.991`），`I_A` 中 `S_light` 方差贡献由负转正 |
| 五因素无量纲化 `far_dz_norm = clip(Δz/L_obs, -1, 1)` | 2026-11-26 尺度不变性标定升级 | 坐标缩放 $2\times$ 分类翻转率由 `7.8%` 降至 `0.0%`，LOPO 折均 ARI 升至 `0.965`，蒙特卡洛扰动翻转率减半 |
| 高度权重 `1.0`、无量纲远端高差权重 `1.0`、面积阈值 `0.5`、顶部跨度 `0.15`、距离比 `2/3` | 项目预设的形态判定规则 | 有权重与距离比敏感性分析；未据独立生理标签标定这些阈值 |
| 株型符合度指数目标曲线（`I_A` / `I_L` / `I`，接入 46 向受光）、等权 `α=0.5` 与目标类别比例 `0.20/0.60/0.20` | 预注册智能冠层几何参考设置 | 完成单调性（PASS）、蒙特卡洛扰动（`ρ=0.978`）、剔除待查叶（`ρ=0.881`）、植株/叶级 Bootstrap（`ρ=0.842`, LOLO `1.000`）与 Dirichlet 敏感性（`ρ=0.978`）验证（[指数快照](validation/ideotype_index_2026-11-26.json)）；未做产量或外部生理标定 |
| `k=3`、初始化 50 次、种子 42 | 三类任务定义与计算复现设置 | 算法文献支持求原型的方法；不支持三种天然生理状态的结论 |
| 贴合告警 `max(6δ,0.15W)`、远离面积 5% 复查线 | 工程告警线 | 定位几何复查对象，不是置信区间或生物学合格线 |

内部留一株验证每次用其余植株重新拟合缩放器和中心，再预测留出植株。它检查对植株抽样的稳定性；与全量分类的一致性不是人工标签准确率。记录见[当前分类反馈](validation/classification_feedback_2026-11-25.json)，实现见 [`classification.py::lopo_validate`](../src/maize_leaf/classification.py)。

**尚待独立验证的关键点：** 四指标组合是否优于其他组合、俯仰熵能增加多少人工分类一致性、各规则阈值能否跨品种和扫描条件使用。应另取独立植株，由人工标注叶片形态及发育状态，再评价混淆矩阵、各类召回率和测量误差；不能用同批数据的视觉观感替代这些证据。

## 参考文献与原始来源

以下链接核对到论文、作者机构或官方文档；文献证明相应原理，项目改动及参数由上文单独说明。

1. Point Cloud Library. [Euclidean Cluster Extraction](https://pointclouds.org/documentation/tutorials/cluster_extraction.html)；[StatisticalOutlierRemoval](https://pointclouds.org/documentation/tutorials/statistical_outlier.html)。官方算法说明，访问日期 2026-11-25。
2. NIST. [Median Absolute Deviation](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/mad.htm)。MAD 及正态尺度换算，访问日期 2026-11-25。
3. Eilers PHC, Marx BD. 1996. *Flexible Smoothing with B-splines and Penalties*. [DOI](https://doi.org/10.1214/ss/1038425655)；[论文全文](https://sites.stat.washington.edu/courses/stat527/s13/readings/EilersMarx_StatSci_1996.pdf)。
4. Huang H, et al. 2013. *L1-Medial Skeleton of Point Cloud*. [DOI](https://doi.org/10.1145/2461912.2461913)；[作者机构提供的论文](https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf)。
5. Wang W, Jüttler B, Zheng D, Liu Y. 2008. *Computation of Rotation Minimizing Frames*. [DOI](https://doi.org/10.1145/1330511.1330513)；[香港大学论文记录](https://hub.hku.hk/handle/10722/152386)。
6. Duchon J. 1977. *Splines minimizing rotation-invariant semi-norms in Sobolev spaces*. [原始章节](https://doi.org/10.1007/BFb0086566)。
7. Wahba G. 1985. *A Comparison of GCV and GML for Choosing the Smoothing Parameter in the Generalized Spline Smoothing Problem*. [DOI](https://doi.org/10.1214/aos/1176349743)。
8. Dornbusch T, et al. 2011. *A comparative analysis of leaf shape of wheat, barley and maize using an empirical shape model*. [DOI](https://doi.org/10.1093/aob/mcq181)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3077976/)。卷期年份为 2011，在线发表为 2010。
9. Shannon CE. 1948. *A Mathematical Theory of Communication*. [论文全文](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)。
10. Arthur D, Vassilvitskii S. 2007. *k-means++: The Advantages of Careful Seeding*. [作者提供的论文](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)。
11. Wang X, et al. 2022. *Characterization of regulatory modules controlling leaf angle in maize*. [DOI](https://doi.org/10.1093/plphys/kiac308)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9434308/)。
12. Strable J, et al. 2017. *Maize YABBY Genes drooping leaf1 and drooping leaf2 Regulate Plant Architecture*. [DOI](https://doi.org/10.1105/tpc.16.00477)；[作者所在机构 USDA 的研究记录](https://www.ars.usda.gov/research/publications/publication/?seqNo115=333903)。
13. Facette MR, et al. 2013. *Parallel Proteomic and Phosphoproteomic Analyses of Successive Stages of Maize Leaf Development*. [DOI](https://doi.org/10.1105/tpc.113.112227)；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3784581/)。
14. Kass RE, Raftery AE. 1995. *Bayes Factors*. [DOI](https://doi.org/10.1080/01621459.1995.10476572)。
15. Leopardi P. 2006. *A Partition of the Unit Sphere into Regions of Equal Area and Small Diameter*. [期刊论文全文](https://ftp.gwdg.de/pub/EMIS/journals/ETNA/vol.25.2006/pp309-327.dir/pp309-327.pdf).
16. den Dulk JA. 1989. *The interpretation of remote sensing measurements over agricultural crops*. PhD Thesis, Wageningen Agricultural University（46 方向等立体角 Turtle Sky 半球划分与标准阴天 SOC 辐射加权，广泛用于 Caribu / GroIMP 玉米冠层截光模拟）.
