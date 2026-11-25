# 玉米叶片三维建模与缺失补全（V3.0）

基于几何多候选拟合、L1 骨架与植物形态先验的玉米点云建模、缺失补全、表型提取和姿态分类工具。输入需要已有叶片语义标签。

当前分类采用 **茎叶角＋同株相对投影面积＋俯仰角信息熵＋相对着生高度**。14 株、90 叶的当前结果入口为 [离线三维查看器](outputs/classification_v2/viewer/index.html)，参数、消融与高度权重对照见 [分类报告](outputs/classification_v2/classification_report.json)。

## 核心算法依据

- **双重去噪**：经全量 90 叶敏感性实验标定的单次稳健半径欧氏连通分量聚类滤波（Rusu & Cousins, 2011；$r = \max(5\,d_{\mathrm{sp}}, 0.02\,D_{\mathrm{rob}}),\ N_{\min}=\max(20, \lfloor 0.01\,N \rfloor)$）+ 分断片稳健 MAD-SOR（Rusu et al., 2008；$k=8,\ \tau = m + 3.5\,\sigma_{\mathrm{rob}}$）。
- **骨架与中脉**：$L_1$-Medial 点云收缩骨架（Huang et al., 2013, *ACM TOG*）+ 测地排序 + 弧长重参数化主曲线（Hastie & Stuetzle, 1989）+ 有界 GCV P-spline（Eilers & Marx, 1996）+ 牛顿正交投影（Piegl & Tiller, 1997）。
- **多候选几何**：比较联合优化、初始中脉、平滑中脉、主轴及直线参考坐标；先排除采样局部翻转，再综合点到叶面的 P95、中位距离及观测叶面到点云的面积加权距离选择。
- **叶尖补全与叶面拟合**：分段幂函数、经验先验、BIC 截断检验、工程阻尼外推及分箱中位 GML-TPS；零宽叶尖同时收敛横向高度，避免末端张开。
- **质量检查**：局部翻转、网格/边界连通、退化面、叶尖闭合、拼接连续性和扫描贴合。失败时报告错误，避免静默导出未通过的曲面。

详细公式见 [docs/method.md](docs/method.md)。
14 株、90 片叶的逐叶输入哈希、参数及复查指标见 [验证记录](docs/validation/leaf_quality_2026-11-24.json)。
全量 450 个遮挡案例的[逐叶误差](docs/validation/holdout_per_leaf_2026-11-25.csv)、[配对比较](docs/validation/holdout_paired_2026-11-25.csv)及[方法与结果](docs/method.md#18-逐叶遮挡留出验证2026-11-25)已记录；缺边和截尖仍有较多误差告警，扫描内检查通过不等于缺失区域已准确恢复。

## 快速运行

```bash
# 安装依赖
pip install -r requirements.txt

# 批量建模（默认读取 dataset/，输出至 outputs/maize_v2/）
python scripts/run.py

# 保留旧结果，在新目录重建后逐叶审计（距离单位与输入一致）
python scripts/run.py --output outputs/maize_checked
python scripts/validate.py fit --baseline outputs/maize_v2 --candidate outputs/maize_checked --output outputs/maize_checked/fit_comparison.json

# 指定单株或单叶
python scripts/run.py --plant 1-1-2
python scripts/run.py --plant 1-1-2 --leaf 1

# 阶段 2–4：表型提取 → 四因素分类 → 离线交互查看器
python scripts/extract_traits.py
python scripts/classify_leaf_pose.py --height-weight 0.5
python scripts/visualize.py --export-viewer outputs/classification_v2/viewer

# 独立交互可视化（打开浏览器，支持自由切换全部植株与单个叶片）
python scripts/visualize.py
python scripts/visualize.py --list
python scripts/visualize.py --plant 1-1-2 --leaf 1
python scripts/visualize.py --plant 1-1-2 --leaf 1 --backend mpl --save outputs/classification_v2/leaf_01.png
python scripts/visualize.py --model outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz

# 标定与留出验证实验（两个统一入口，详见 scripts/README.md）
python scripts/validate.py holdout --output outputs/holdout_run --workers 6
python scripts/plot_validation.py holdout --root outputs/holdout_run
python scripts/validate.py holdout --output outputs/holdout_repaired --split-root outputs/holdout_run --workers 6
python scripts/validate.py compare --before outputs/holdout_run --after outputs/holdout_repaired
python scripts/validate.py clustering
python scripts/plot_validation.py clustering
python scripts/validate.py sor
python scripts/plot_validation.py sor

# 单元测试（自动同时检查 Python 单测与 Node 查看器脚本逻辑）
python scripts/run_tests.py
```

## 连续模型调用

```python
import numpy as np
from maize_leaf.surface import MaizeLeafModel

model = MaizeLeafModel.load("outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz")
s = np.array([0.2, 0.5, 0.8]) * model.length
sv = np.column_stack([s, [-0.5, 0.0, 0.5]])  # [参考曲线参数 s, 归一化横向 v]
xyz = model.evaluate(sv)
t, b, n = model.frame.evaluate(sv[:, 0])
```

最终曲面中心轨迹为 `model.evaluate([[s, 0], ...])`。表型 CSV 同时保存中心轨迹叶长与参考参数跨度，二者不应混用。原始数据与生成大文件按 `.gitignore` 保留本地。

## 分类与查看器

同株相对投影面积为逐面元水平投影面积除以该株最大逐叶投影面积，保留叶片相对大小信息；俯仰熵统计纵向切向的面积加权角度分布（12 分箱）。相对高度使用叶基相对茎秆范围的高度 `stem_rel_height`，不使用会受叶尖下垂影响的冠层范围。四列在每个 LOPO 训练折内做 median/IQR 标准化，再按平方距离权重 `[1, 1, 1, 0.5]` 变换。高度权重是固定初值；脚本另保存 0、0.25、0.5、1、2 的敏感性对照。方法和适用范围见 [第19节](docs/method.md#19-三项形态因素与相对着生高度分类2026-11-25)。

全局K-means学习三个原型，再用经验规则解释类别：紧缩候选必须位于该株最高叶基以下0.15相对茎高以内、投影面积不超过该株最大叶的0.5，且有清楚的原型距离优势；过渡样本归舒展。距离比主初值为2/3，保留0.5—0.8敏感性对照。它们是根据用户定义形成的开发参数，不是已验证的生物学标准。每株没有类别配额，可以缺少紧缩型或其他类别。

在线、离线共用一个页面模板。选叶只更新可见性和选中标记，保留网格、相机及整株背景；勾选“仅看选中叶”可隔离。默认只显示观测支撑面，勾选“显示插值/外推面”检查完整估计域。启动在线查看器仍使用 `python scripts/visualize.py`。

LOPO 指标衡量相对全数据聚类的稳定性；没有人工标签时不称为准确率。高度已参与聚类，其分层分布属于描述结果；下垂类别不代表已经衰老。

## 观测支撑域修正

当前模型独立限制左右扫描边缘，不再把较宽一侧镜像到另一侧。未观测叶尖延长从活动域撤下；截断端保持开放横截面。model.length 表示活动域长度，model.blade.length 保留历史叶形先验参数范围；数值计算和网格使用 model.mesh_grid()。旧NPZ仍按原几何加载。

原重建和分类保存在 outputs/archive/20261125_feedback_fix/。修正保存模型的复现入口为 scripts/refine_surface_support.py，显式指定 --source 归档模型根目录和新的 --output。重新建模CLI默认也使用保守支撑域；研究历史补全模型时可加 --allow-unobserved-completion。

当前模型目录的 support_audit.json 保存90叶的前后贴合与支撑距离；仍有20叶触发远离点云面积占比超过5%的工程复查线，在查看器中标为“支撑待查”。这不等价于缺失恢复真值验证。两片指定叶的[前后对照](outputs/maize_v2/diagnostics/support_comparison.png)和[本轮验收记录](docs/validation/classification_feedback_2026-11-25.json)可供复查。

## 目录约定

- `dataset/`：原始 TXT/CSV，只读。
- `src/maize_leaf/`：算法与共用查看器模板；`scripts/`：各阶段主流程入口（见 [scripts/README.md](scripts/README.md)），验证与绘图统一由 `validate.py`、`plot_validation.py` 调用 `src/maize_leaf/experiments/` 中的实现。
- `docs/`：方法和可追溯验证记录；`tests/`：数值与功能回归测试。
- `outputs/maize_v2/`：当前重建；`outputs/phenotypes_v2/`：当前表型；`outputs/classification_v2/`：当前分类与查看器。
- `outputs/holdout_20261125/`、`outputs/*_sensitivity/`：保留的验证、标定实验。
- `outputs/archive/`：历史重建、旧分类和试验快照。本轮迁移记录见 [归档清单](outputs/archive/20261125_four_factor/moves.json)。归档材料中的路径按生成时记录保留，恢复时参考迁移清单。
- `.cache/`：运行缓存；入口生成的字节码集中到 `.cache/python/`。阶段性调试记录完成后迁入 `outputs/archive/`，避免堆放根目录与源码目录。

每次比较新算法应使用新的 `--output` 目录；确认采用后保留旧版本再更新当前结果。旧三因素多方案比较脚本已随历史结果归档，当前比较统一由 `classify_leaf_pose.py` 输出。

2026-11-25 的阶段整理将运行入口收敛为 8 个，各阶段 1–2 个；自有 Python 源码、测试与前端模板均补充中文职责和关键逻辑说明。源码快照、旧入口对应关系、缓存迁移及检查证据见 [本次整理记录](outputs/archive/20261125_stage_cleanup/README.md)。历史验证 JSON 中的原始路径与哈希仍按当时版本保存。
