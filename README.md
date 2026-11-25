# 玉米叶片三维建模与缺失补全（V3.0）

基于几何多候选拟合、L1 骨架与植物形态先验的玉米点云建模、缺失补全和表型提取工具。输入需要已有叶片语义标签。

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
python scripts/validate_fit.py --baseline outputs/maize_v2 --candidate outputs/maize_checked --output outputs/maize_checked/fit_comparison.json

# 指定单株或单叶
python scripts/run.py --plant 1-1-2
python scripts/run.py --plant 1-1-2 --leaf 1

# 独立交互可视化（打开浏览器，支持自由切换全部植株与单个叶片）
python scripts/visualize.py
python scripts/visualize.py --list
python scripts/visualize.py --plant 1-1-2 --leaf 1
python scripts/visualize.py --plant 1-1-2 --leaf 1 --backend mpl --save leaf_01.png
python scripts/visualize.py --model outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz

# 全量 90 叶留出遮挡验证（孔洞 / 叶缘 / 末端参数跨度 10%、20%、30%）
python scripts/validate_holdout.py --output outputs/holdout_run --workers 6
python scripts/plot_holdout.py --root outputs/holdout_run
# 对照修复前后的相同遮挡划分，生成逐叶配对误差表
python scripts/validate_holdout.py --output outputs/holdout_repaired --split-root outputs/holdout_run --workers 6
python scripts/compare_holdout.py --before outputs/holdout_run --after outputs/holdout_repaired

# 全量 90 叶欧氏聚类与 SOR 参数敏感性标定及诊断出图
python scripts/validate_euclidean_clustering.py
python scripts/plot_euclidean_clustering.py
python scripts/validate_sor_sensitivity.py
python scripts/plot_sor_sensitivity.py

# 单元测试
python scripts/run_tests.py
```

## 连续模型调用

```python
import numpy as np
from maize_leaf.surface import MaizeLeafModel

model = MaizeLeafModel.load("outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz")
s = np.array([0.2, 0.5, 0.8]) * model.blade.length
sv = np.column_stack([s, [-0.5, 0.0, 0.5]])  # [参考曲线参数 s, 归一化横向 v]
xyz = model.evaluate(sv)
t, b, n = model.frame.evaluate(sv[:, 0])
```

最终曲面中心轨迹为 `model.evaluate([[s, 0], ...])`。表型 CSV 同时保存中心轨迹叶长与参考参数跨度，二者不应混用。原始数据与生成大文件按 `.gitignore` 保留本地。
