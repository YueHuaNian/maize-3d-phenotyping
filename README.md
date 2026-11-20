# Maize 3D Phenotyping

V1.1 开发分支以 V1.0 全局最近邻叶基为起点，从玉米语义点云提取叶片可见几何特征、倾角 Shannon 熵和探索性形态簇。

[V1.1 数学与质量口径](docs/V1.1_METHODS.md) · [14 株分析报告](outputs/v1.1/V1.1_ANALYSIS_REPORT.md) · [V1.0 稳定发布](https://github.com/YueHuaNian/maize-3d-phenotyping/releases/tag/v1.0) · [V6 归档](archive/legacy-v6-baseline-2026-11-19/README.md)

## 14 株真实数据验证

14 株的 TXT 和语义 CSV 全部通过逐行核验，共 325,818 个点、90 片叶。90 个最近点对与 V1.0 一致，34 项测试全部通过。

| 结果 | 本批次观测 |
|---|---|
| 进入聚类 | 33 片叶的六项特征均通过质量筛查，覆盖全部 14 株 |
| K-means（K=3） | C1 10 片、C2 15 片、C3 8 片 |
| 聚类评价 | 轮廓系数 0.255，Calinski–Harabasz 12.160，Davies–Bouldin 1.256，簇内平方和 109.353 |
| 逐株留出 | 14/14 折完成。共同训练叶的 ARI 中位数 1.000、最低 0.327。留出标签一致率按叶加权为 84.85%，按株等权为 77.38% |
| 倾角 Shannon 熵 | 57 片叶获得有效值。中位数 3.101 bit，四分位区间 2.696–3.361 bit |

聚类使用六项特征齐全的叶片。全部 90 片叶均保留在逐叶表中，缺失值附有原因。熵按可见中线各倾角区间的弧长占比计算。留出一致率反映本批次复算稳定性，跨批次聚类和独立解剖叶基误差仍待验证。 [逐叶结果](outputs/v1.1/leaf_features.csv) · [逐株统计](outputs/v1.1/plant_summary.csv) · [留出折明细](outputs/v1.1/cluster_stability.csv)

![33 片完整特征叶的二维 PCA 聚类图](outputs/v1.1/cluster_2d.png)

## 数据与复现

使用 Python 3.10 或以上及 requirements.txt 中的依赖。输入为逐株同名的原始 TXT 和语义 CSV，CSV 必须含 X、Y、Z、semantic。标签 −1 为背景，0 为茎秆，全部正整数各自代表一片叶。输入格式及逐行核验沿用 [V1.0 算法说明](docs/ALGORITHM.md)。

~~~shell
python -m pip install -r requirements.txt
python run_analysis.py --sample 2-1-2 --data-dir path/to/csv --txt-dir path/to/txt
python run_analysis.py --batch --data-dir path/to/csv --txt-dir path/to/txt
python -m unittest discover -s tests -v
~~~

可用 --output-dir 指定一个新的输出目录。默认写入 results/runs 中的独立时间戳目录。CSV 路径也可使用 MAIZE_DATA_DIR 或 .maize-data-dir。TXT 路径可用 MAIZE_TXT_DIR，未配置时依照旧版数据目录约定查找。

## 输出

- leaf_features.csv 保存每片叶最近点对、源点行号、各项特征、质量状态及簇 ID。plant_summary.csv 和 feature_summary.csv 分别汇总逐株与总体特征。
- cluster_model.json、cluster_centers.csv、cluster_stability.csv 保存模型配置、质心、评价指标和逐株留出结果。
- input_manifest.csv、quality_details.json、run_provenance.json 保存输入哈希、质量诊断、软件依赖和计算参数。
- [二维聚类图](outputs/v1.1/cluster_2d.png)和[特征分布图](outputs/v1.1/feature_distributions.png)可直接查看。每次运行还在本机的 interactive/ 目录生成 cluster_3d.html（可旋转的 PCA 聚类图）和 plant_geometry_3d.html（可切换植株的拟合茎线、叶中线和最近点对）。
- 单株分析输出叶片特征和空间核验，跨植株批次额外拟合聚类。V1.0 最近邻独立入口仍为 python -m src.nearest_attachment，V6 运行方式见历史归档。

坐标的物理单位待确认。质量标记和独立解剖叶基验证范围见批次报告。
