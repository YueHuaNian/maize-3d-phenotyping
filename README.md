# Maize 3D Phenotyping

V1.1 开发分支以 V1.0 全局最近邻叶基为起点，从玉米语义点云提取叶片可见几何特征、倾角 Shannon 熵和探索性形态簇。

[V1.1 数学与质量口径](docs/V1.1_METHODS.md) · [14 株分析报告](outputs/v1.1/V1.1_ANALYSIS_REPORT.md) · [V1.0 稳定发布](https://github.com/YueHuaNian/maize-3d-phenotyping/releases/tag/v1.0) · [V6 归档](archive/legacy-v6-baseline-2026-11-19/README.md)

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
- cluster_2d.png 和 feature_distributions.png 为静态图。interactive/ 内的 cluster_3d.html 和 plant_geometry_3d.html 可在本地浏览器中离线旋转、缩放并查看标签或切换植株。
- 单株分析输出叶片特征和空间核验，跨植株批次额外拟合聚类。V1.0 最近邻独立入口仍为 python -m src.nearest_attachment，V6 运行方式见历史归档。

坐标的物理单位待确认。质量标记和独立解剖叶基验证范围见批次报告。
