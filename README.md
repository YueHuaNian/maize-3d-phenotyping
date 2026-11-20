# Maize 3D Phenotyping V1.0

基于全局欧氏最近邻的玉米叶基定位算法。给定带有语义标签的茎秆和叶片点云，逐叶寻找三维欧氏距离最短的叶点与茎点，作为叶基的初始几何估计。

[真实数据实验报告](outputs/v1.0/V1.0_EXPERIMENT_REPORT.md) · [算法说明](docs/ALGORITHM.md) · [版本和研究计划](docs/PROJECT_STATUS.md) · [原 v1.0 / V6 历史归档](archive/legacy-v6-baseline-2026-11-19/README.md)

## 准备输入

- 每株提供一份语义 CSV 和一份同名的原始 TXT，分别放入对应的输入目录。同名文件的去扩展名部分即植株编号。
- CSV 需要 X、Y、Z、semantic 列。TXT 第一行为 CloudCompare 字段名，第二行为点数，点数据中含 X、Y、Z、Constant，可含 Constant_#2。后者存在且有值时优先读取，否则使用 Constant。
- 标签 −1 表示背景或噪声，0 表示茎秆，所有正整数各自表示一个叶实例。程序保留源叶号，逐行核对两个文件的点数、XYZ 和语义标签。
- 运行需 Python 3.10 或以上，并安装依赖。实验使用的具体版本见实验报告。

## 运行

~~~shell
python -m pip install -r requirements.txt
python -m src.nearest_attachment --data-dir path/to/csv --txt-dir path/to/txt
python -m unittest discover -s tests -v
~~~

已存在 outputs/v1.0 时，请用 --output-dir 指定新目录，例如 outputs/reproduction。输入核验或点对计算失败时，程序写出对应记录并返回非零退出状态。逐株输入核验详见 input_manifest.csv。

## 算法和结果

程序以全部有效茎点建立 SciPy cKDTree，再以每片叶的全部有效点查询精确欧氏最近邻。取查询距离的全局最小值。并列时先取原始行号最小的叶点，再取原始行号最小的茎点。三维图显示有效的原坐标叶片与茎秆点云、各叶点对及最短连线。

| 默认输出 | 内容 |
|---|---|
| [nearest_attachment_results.csv](outputs/v1.0/nearest_attachment_results.csv) | 逐叶点数、两点 XYZ、源行号、距离及状态 |
| [input_manifest.csv](outputs/v1.0/input_manifest.csv) | 逐株配对核验、行数及输入 SHA-256 |
| [plant_summary.csv](outputs/v1.0/plant_summary.csv) | 逐株叶数、成功数和距离统计 |
| [distance_summary.json](outputs/v1.0/distance_summary.json) | 批次距离统计和异常围栏 |
| [distance_outliers.csv](outputs/v1.0/distance_outliers.csv) | 超出合并四分位距围栏的叶片 |
| [nearest_distance_overview.png](outputs/v1.0/nearest_distance_overview.png) | 直方图、箱线图和逐株比较 |
| [visualizations/](outputs/v1.0/visualizations/) | 逐株等比例三维核验图 |

源点行号从 0 开始，以首条点数据为第 0 行，排除 CSV 表头和 TXT 的两行头部。计数列分别提供原始标签点数和有效 XYZ 点数。距离单位沿用原始坐标，物理单位待确认。统计异常反映本批次的距离分布，所有异常叶均保留。

最短点对只是初始几何估计。遮挡、贴茎叶段、点云疏密及语义边界可能影响点对位置。实际叶基定位误差需要独立解剖标注验证。原 V6 的运行方式见历史归档。
