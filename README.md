# Maize 3D Phenotyping

从带语义标签的玉米三维点云定位茎叶最近点对，测量可见叶片的几何特征，并按形态或叶基高度做逐株分类。结果保留原始植株编号、叶号、点数据行号、输入哈希和质量状态，可以从报告追溯到原始点云。

[分析流程](#分析流程) · [PAI 与 LMI](#pai-与-lmi) · [不足与误差](#不足与误差) · [复现步骤](#从原始点云复现) · [交互查看器](#交互查看器) · [V1.0 发布](https://github.com/YueHuaNian/maize-3d-phenotyping/releases/tag/v1.0) · [V6 历史归档](archive/legacy-v6-baseline-2026-11-19/README.md)

## 分析流程

分析从已有器官标签的玉米点云开始。同株同名 TXT 与 CSV 应使用相同坐标和点顺序。流程先核对输入并定位茎叶最近点对，再分别计算通过质量筛查的拟合特征和基于原始叶点的观测剖面特征。

~~~mermaid
flowchart TD
    Input["逐株 TXT 和语义 CSV"] --> Check["逐行核验并记录哈希"]
    Check --> Pair["V1.0 茎叶最近点对"]
    Pair --> Fit["V1.1 茎线与叶中线拟合"]
    Pair --> Profile["V1.1 观测叶点剖面"]
    Fit --> Batch["六特征整批聚类与逐株留出"]
    Fit --> Angle["夹角质量排查与近端对照"]
    Profile --> Shape["逐株熵形态聚类"]
    Profile --> Layer["V2 叶基高度分层"]
    Profile --> Index["逐株 PAI 与逐叶 LMI"]
    Layer --> Compare["层间形态对照"]
    Fit --> Compare
    Layer --> Viewer["V2 查看器"]
    Index --> Viewer
~~~

| 步骤 | 输入、计算和产物 |
|---|---|
| **1. 输入核验** | 按植株 ID 配对，逐行核对点数、XYZ 和标签。核验失败的植株记录在输入清单。保留原始文件名和 SHA-256，含非有限坐标的点不参与几何计算。[输入清单](outputs/v1.1/input_manifest.csv) |
| **2. V1.0 叶基估计** | 为有效茎点建立 cKDTree。对每片叶，在全部有效叶点和茎点中搜索欧氏距离最小的点对。两点的坐标、距离及原始行号写入逐叶表。V1.1 从原始点云调用同一定位逻辑，可独立运行 V1.0 生成核验报告。[算法](docs/ALGORITHM.md) · [逐叶最近点对](outputs/v1.0/nearest_attachment_results.csv) |
| **3. V1.1 拟合特征与整批聚类** | 拟合稳健茎线和可见叶中线，按各自质量规则计算夹角、着生高度、下垂、曲率、水平延展及拟合中线倾角熵。正式特征未通过筛查时留空。六项齐全的叶片进入 StandardScaler + K-means（K=3），按植株留出复算。另可用质量明细做夹角诊断和近端 PCA 对照。[逐叶特征](outputs/v1.1/leaf_features.csv) · [计算定义](docs/V1.1_METHODS.md) |
| **4. V1.1 观测剖面与逐株聚类** | 从最近叶点出发，按离叶基的径向距离分段，连接各段有效叶点的坐标中位数，得到观测剖面。各株独立用 K-means（K=3）按相对形态分组。观测方向熵权重 55%，水平伸展、末端落差和下行段占比各 15%。[逐叶分组与熵](outputs/v1.1/plant_morphotypes/leaf_clusters.csv) |
| **5. V2 高度优先分层** | 读取最近茎点的 Z 高度和观测剖面特征。株内标准化后，以加权簇内平方和最小选两个高度切点。高度权重 65%、观测熵 20%、其余三项各 5%。同高叶片留在同一层。默认要求中层超过半数，达不到时保留三层并使中层尽可能大。自由切分可用于层间形态对照。[默认方案](outputs/v2-middle-majority/HEIGHT_ORDERED_REPORT.md) · [自由切分对照](outputs/entropy-layers/REPORT.md) |
| **6. PAI、LMI 与可视化** | `run_analysis.py` 同时以最近点对和观测剖面特征计算 PAI 与 LMI。V2 查看器将这两张指标表和分层结果按植株 ID 与原始叶号关联。导出的 CSV 保留各分量、质量状态和输入溯源。[指标报告](outputs/pai-lmi-baseline/PAI_LMI_REPORT.md) |

读表时要区分两套高度和熵。V1.1 整批模型将最近茎点投影到拟合茎线上，以累计弧长占比表示高度，熵来自可见中线相对茎主轴的倾角。V2 和 PAI 用最近茎点的 Z 坐标及茎点高度分位数，逐株形态聚类、V2 和 LMI 用观测剖面相对 +Z 的方向熵。整批模型的 C1/C2/C3 依簇中心高度由低到高编号。逐株形态模型的编号来自形态特征，V2 按高度从上到下编号。查看或比较分组时应先确认所用模式。[V1.1 方法](docs/V1.1_METHODS.md) · [V2 分层规则](outputs/v2-middle-majority/HEIGHT_ORDERED_REPORT.md)

## PAI 与 LMI

**PAI（Plant Architecture Index，株型指数）** 每株一个值，描述叶片横向伸展和叶基在茎上的分布宽度。横向伸展分量 `Rₚ` 取本株各叶伸展分量 `R` 的中位数。高度分量 `Vₚ` 取本株相对叶基高度的第 90 与第 10 百分位数之差，再用下文的 `f` 换算。**PAI = 100 × (Rₚ + Vₚ) / 2**。分数越高，表示横向伸展较大、叶基高度分布较宽，或两者兼有。

**LMI（Leaf Morphology Index，叶形指数）** 每片叶一个值。**LMI = 100 × (E + R + D) / 3**，三个分量等权。

| 分量 | 如何计算 |
|---|---|
| 方向熵 `E` | 把观测叶片剖面弧段与 +Z 的夹角按 10° 分为 18 箱，按各箱的弧段长度占比计算 Shannon 熵，再除以 log₂(18)。方向分布越分散，`E` 越大。 |
| 水平伸展 `R` | 取叶点到叶基最近叶点的水平距离第 95 百分位数，除以本株茎点高度跨度 `H`，再用 `f` 换算。 |
| 下垂 `D` | 将剖面最高点到远端的高度落差除以 `H`，用 `f` 换算后，与下行弧段的长度占比取平均。下行弧段与 +Z 的夹角至少为 100°。 |

`H` 为茎点 Z 坐标的第 99 与第 1 百分位数之差。相对叶基高度为 `(最近茎点的 Z − 茎点 Z 的第 1 百分位数) / H`。非负比值统一用 `f(x) = x / (1 + x)` 换算。

例如，实际样本 `2-1-2` 的 **PAI 为 28.13**，第 1 片叶的 **LMI 为 26.22**，全株 10 片叶的 LMI 中位数为 **35.99**。两个分数都用 0–100 尺度描述观测几何，采用项目定义的基线权重，独立于 C1/C2/C3 分类。读数时可查看各分量和有效叶数，数据不满足计算条件时分数留空。[逐株结果](outputs/pai-lmi-baseline/plant_indices.csv) · [逐叶结果](outputs/pai-lmi-baseline/leaf_indices.csv) · [完整公式与质量规则](docs/PAI_LMI_METHODS.md)

## 已验证的数据与结果

发布批次有 **14 株、325,818 个点、90 片叶**。14 对 TXT 和 CSV 全部通过逐行核验，90 片叶均获得最近点对。

- **V1.1 整批模型**纳入 33 片六特征完整的叶，轮廓系数 0.255。14 折逐株留出相对于整批参照模型的标签一致率按叶数加权为 84.85%，按株等权为 77.38%。[整批报告](outputs/v1.1/V1.1_ANALYSIS_REPORT.md) · [逐株留出](outputs/v1.1/cluster_stability.csv)
- **V1.1 逐株熵模型**覆盖 90 片叶，观测方向熵中位数 2.826 bit。聚拢型 25 片、伸展型 33 片、耷拉型 32 片。[逐株形态报告](outputs/v1.1/plant_morphotypes/PLANT_CLUSTER_REPORT.md)
- **V2 默认分层**上层 19 片、中层 56 片、下层 15 片，13 株中层超过半数。样本 `2-4-1` 共 4 片叶，在三层均非空时中层最多 2 片。**V2 自由切分**上层 28 片、中层 29 片、下层 33 片。[默认逐株统计](outputs/v2-middle-majority/plant_summary.csv) · [自由切分逐株统计](outputs/v2/plant_summary.csv)
- **PAI** 覆盖 14/14 株，中位数 30.61，范围 25.35–51.93。**LMI** 覆盖 90/90 片叶，逐叶中位数 36.49，范围 17.62–60.22。[综合评价报告](outputs/pai-lmi-baseline/PAI_LMI_REPORT.md)

静态分层图采用自由切分结果，红、橙、蓝依次表示上、中、下层，灰色为茎。V2 交互查看器使用本页下方列出的配色。[14 株静态图](outputs/layer-figures/) · [分层总览](outputs/layer-figures/分层总览.jpg)

![V1.1 整批模型的二维 PCA 投影](outputs/v1.1/cluster_2d.png)

## 不足与误差

目前可量化的是**茎叶点间距、不同角度算法的读数差、特征覆盖率和同批次的分组稳定性**。解剖叶基位置和真实叶茎夹角缺少独立人工标注，本批次暂无这两项的绝对误差或均方根误差（RMSE）。植株是独立观测单位，汇总的叶片统计描述本批次的可见几何。

| 环节 | 现有证据及适用范围 |
|---|---|
| **输入与方向** | TXT/CSV 逐行一致说明两份输入可配对。遮挡、漏扫、语义标注偏差和贴茎叶点会传到后续特征。坐标的物理单位待确认，距离保留原始坐标单位。V2、PAI 与 LMI 依赖输入的 +Z 竖直约定，其中 PAI/LMI 的部分分量相对同株茎点跨度归一化，茎点覆盖过短会改变这些比值。[输入核验与坐标口径](docs/ALGORITHM.md) · [指标归一化](docs/PAI_LMI_METHODS.md#输入与归一化) |
| **最近点对与叶基估计** | 90 片叶均有最近点对。点间距中位数为 0.00501，范围为 0.000690–0.18116。14 片超过本批次按 `Q3 + 1.5 × IQR` 计算的上界 0.01790。这些数值衡量所选两点的间隔，可用于优先核查异常叶。即使间隔很小，贴茎叶段也可能被选为叶基。[距离统计](outputs/v1.0/distance_summary.json) · [异常叶清单](outputs/v1.0/distance_outliers.csv) |
| **拟合特征覆盖率** | 在 90 片叶中，正式夹角有效 33 片，着生高度 61 片，归一化曲率 52 片，下垂、水平延展和拟合熵各 57 片。57 片夹角缺失中有 26 片涉及最近叶点与拟合中线起点距离超限。多个原因可重叠，早期阻断也可能使后续检查未执行。整批聚类的结论对应通过全部六项筛查的 33 片叶。[特征覆盖](outputs/v1.1/feature_summary.csv) · [夹角缺失诊断](outputs/angle-diagnostics/REPORT.md) |
| **近端 PCA 角度差** | 与正式夹角可比较的 27 片叶，绝对差中位数 5.1°，第 90 百分位数 30.3°。两种估计共用最近叶点和茎线，数值反映两种方向估计的差异。26 片叶基与中线起点距离超限的叶中，14 片在三个窗口都可计算，0 片通过探索性筛选。相应正式字段继续留空。[近端对照报告](outputs/proximal-angle-experiment/REPORT.md) |
| **观测剖面与分层** | 径向分段可能把回折或遮挡后的不同部位合到同一区间。熵衡量方向分散程度，判读下垂方向需同时看落差与下行段。自由切分中，14 株有 10 株中层水平伸展中位数大于上层，9 株下层下行段占比中位数大于中层，8 株下层末端落差中位数大于中层。上层夹角较小的只有 3 株有支持，且均为部分覆盖，另 11 株数据不足。V2 默认方案的中层多数来自预设约束。层间对照复用部分分层输入，跨批次表现仍需独立验证。[层间对照](outputs/entropy-layers/REPORT.md#层间形态对照) |
| **整批聚类与综合指数** | 逐株留出的一致率衡量与整批模型的标签复算稳定性。整批参照模型含被留出的植株，实际类别准确率仍需独立标签。PAI/LMI 的权重尚未外部标定。指数适合描述观测到的几何特征，叶面积、产量和田间表现需另行测量。比较不同植株的 LMI 时，应同时看逐株中位数和有效叶覆盖率。[留出方法](docs/V1.1_METHODS.md#聚类与分组复算) · [指标适用范围](docs/PAI_LMI_METHODS.md#适用范围与复现) |

核查单片叶时，先用 `plant_id` 和 `leaf_id` 在逐叶表定位，查看对应的 `*_status` 与质量明细。回看原始点时，`leaf_point_row` 和 `stem_point_row` 从第一条点数据的 0 开始计数。换算为文本行号时，CSV 加 2，TXT 加 3。诊断值与正式字段分开记录，正式字段留空时应同时查看失败阶段和缺失原因。[逐叶表](outputs/v1.1/leaf_features.csv) · [质量明细](outputs/v1.1/quality_details.json)

## 从原始点云复现

### 1. 准备环境和数据

从仓库根目录执行以下示例。使用 Python 3.10 或以上，建议在独立的 Python 环境中安装依赖。发布批次使用 Python 3.14.10，具体依赖版本记录于 [V1.1 运行溯源](outputs/v1.1/run_provenance.json) 和 [逐株熵模型溯源](outputs/v1.1/plant_morphotypes/provenance.json)。

~~~sh
python -m pip install -r requirements.txt
~~~

原始点云存放在仓库外。复现发布批次需准备 [输入清单](outputs/v1.1/input_manifest.csv) 所列的 14 对文件，逐文件核对 SHA-256。可使用相同格式的自有数据运行完整流程。目录示例：

~~~text
maize-data/
├── csv/
│   └── 2-1-2.csv
└── txt/
    └── 2-1-2.txt
~~~

- 每株提供同名 CSV 和 TXT，文件名主体为植株编号，两种文件的点顺序一致。
- 两种文件采用 UTF-8 编码。CSV 首行为表头，包含大小写一致的 X、Y、Z 和 semantic 列，每个点占一行。
- TXT 第一行为以空格分隔的列名，可带 CloudCompare 常见的 // 前缀。第二行为点数，后续每行一个点。列名含 X、Y、Z、Constant，可另含 Constant_#2。若后者存在且该行取值为有效标签，则该行优先使用后者。以 nan 表示缺失时回退到 Constant。
- 语义值 −1 为背景或噪声，0 为茎秆，所有正整数各自代表一片叶。程序逐行核对点数、XYZ 和语义。点数据行号从 0 起算，两种文件的头部均排除。

### 2. 运行定位、批次分析和 V2 分层

将示例中的 path/to/csv 和 path/to/txt 换成自己的配对数据目录。每条命令使用一个新输出目录。

~~~sh
python -m src.nearest_attachment --data-dir "path/to/csv" --txt-dir "path/to/txt" --output-dir results/reproduction/v1.0

python run_analysis.py --sample 2-1-2 --data-dir "path/to/csv" --txt-dir "path/to/txt" --output-dir results/reproduction/sample-2-1-2

python run_analysis.py --batch --data-dir "path/to/csv" --txt-dir "path/to/txt" --output-dir results/reproduction/v1.1

python view_phenotypes_v2.py --results-dir results/reproduction/v1.1 --export-dir results/reproduction/v2-majority --export-only

python view_phenotypes_v2.py --results-dir results/reproduction/v1.1 --export-dir results/reproduction/v2-original --middle-size free --export-only
~~~

第一条命令独立生成 V1.0 最近点对报告。单株分析用于检查输入、最近点对、叶片特征和逐株聚类。批次分析用同样的原始点云重新计算最近点对，并生成整批模型和逐株留出结果。V2 的两个导出命令从本次批次目录读取叶基和观测剖面数据，可在没有图形窗口的环境中运行。

单株和整批分析都会生成 `indices/`。用 `plant_indices.csv` 查看 PAI、逐株 LMI 中位数及覆盖率，用 `leaf_indices.csv` 查看逐叶 LMI 与各分量，并用 `PAI_LMI_REPORT.md` 汇总样本结果。

V1.1 的 CSV 目录也可通过 `MAIZE_DATA_DIR` 或项目根目录的 `.maize-data-dir` 指定，后者只写一行本机目录。TXT 目录可通过 `MAIZE_TXT_DIR` 指定。显式路径参数便于迁移和复现，V1.0 独立命令要求提供两种目录。

### 3. 复现层间对照和夹角诊断

这三条命令复用上一步的批次结果，分别生成自由切分的层间统计、正式夹角缺失诊断和近端 PCA 对照。最后一条还要读取同一批原始 CSV，并按输入哈希核对。诊断和对照产物各自保存在指定目录。

~~~sh
python -m src.entropy_layers --results-dir results/reproduction/v1.1 --output-dir results/reproduction/entropy-layers

python -m src.angle_diagnostics --results-dir results/reproduction/v1.1 --output-dir results/reproduction/angle-diagnostics

python -m src.proximal_angle_experiment --results-dir results/reproduction/v1.1 --data-dir "path/to/csv" --output-dir results/reproduction/proximal-angle-experiment
~~~

### 4. 检查结果和运行测试

~~~sh
python -m unittest discover -s tests -v
~~~

复现发布批次时，先比较 `input_manifest.csv` 的植株编号、点数和输入哈希，再对照行数、质量状态及分组。预期 V1.0 输出 90 个最近点对，V1.1 整批模型纳入 33 片完整特征叶，V1.1 逐株模型和 V2 均覆盖 14 株、90 片叶，综合评价输出 14 个有效 PAI 和 90 个有效 LMI。真实样本测试会使用已配置的数据目录，其中配对测试沿用旧版约定，从 CSV 目录的同级 `0709单天多个材料原始数据` 查找 `1-1-2.txt`。

## 交互查看器

在 VS Code 选择项目的 Python 解释器，打开所需脚本，点击 **运行 Python 文件**。桌面查看器使用 Matplotlib 的图形后端，例如 TkAgg。

| 入口 | 用途 |
|---|---|
| [view_phenotypes.py](view_phenotypes.py) | V1.1 点云和最近点对。默认显示逐株熵聚类，有批次结果时也可通过 --cluster-scope batch 查看整批模型 |
| [view_phenotypes_v2.py](view_phenotypes_v2.py) | V2 高度优先方案。默认启用中层多数规则，可通过 --middle-size free 查看原版自由切分 |
| [view_pointcloud.py](view_pointcloud.py) | 历史 V6 查看器，用于原始点云及旧版几何叠加 |

前两个查看器默认使用 outputs/v1.1 和本机数据配置，也可通过 --results-dir、--data-dir、--txt-dir 指向复算结果和输入。左侧切换植株，右侧可单独显示一类。C1 蓝、C2 橙、C3 绿，叶号与点云同色。

V2 查看器切换植株时，右下角显示该株的 **PAI**。左键单击点云中的叶片或同色叶号，可查看该叶的 **LMI**，包括方向熵 E、水平伸展 R、下垂 D 及原始 Shannon 熵（bit）。叶号边框会加粗，隐藏该类或切换植株会清空叶片读数。

默认读取发布批次的 PAI/LMI。查看自己复算的结果时，V2 自动读取 `--results-dir` 下的 `indices/`，也可通过 `--indices-dir` 指定与当前分析匹配的指标目录。旧结果没有指标表时，界面仍可查看点云和分类。

运行 V1.1 后生成的 interactive/ 包含 plant_morphotypes_3d.html（逐株熵形态）、plant_geometry_3d.html（整批模型的逐株几何）和 cluster_3d.html（整批 PCA）。可在浏览器中离线查看，HTML 与配套的 plotly.min.js 需放在同一目录。

## 输出与代码位置

| 路径 | 内容 |
|---|---|
| [outputs/v1.0/](outputs/v1.0/) | 最近点对、距离统计、14 株三维图和实验报告 |
| [outputs/v1.1/](outputs/v1.1/) | 逐叶特征、逐株统计、输入与运行溯源、质量明细、整批模型及留出结果 |
| [outputs/v1.1/plant_morphotypes/](outputs/v1.1/plant_morphotypes/) | 逐叶熵形态分组、特征、簇中心和逐株计数 |
| [outputs/v2-middle-majority/](outputs/v2-middle-majority/) | 默认 V2 的 14 株复算报告、逐叶分层、逐株统计和运行溯源 |
| [outputs/v2/](outputs/v2/) | 原版自由切分的逐叶高度分层、逐株切点、复算参数和来源哈希 |
| [outputs/pai-lmi-baseline/](outputs/pai-lmi-baseline/) | 14 株的 PAI 与 LMI 复算报告、逐株和逐叶指标、分量、质量状态及运行溯源 |
| [outputs/entropy-layers/](outputs/entropy-layers/) · [outputs/layer-figures/](outputs/layer-figures/) | 自由切分逐层特征、层间对照与静态分层图 |
| [outputs/angle-diagnostics/](outputs/angle-diagnostics/) · [outputs/proximal-angle-experiment/](outputs/proximal-angle-experiment/) | 夹角缺失诊断、近端 PCA 候选读数及问题叶清单 |
| [run_analysis.py](run_analysis.py)、[src/](src/)、[tests/](tests/) | V1.1 分析入口、实现与测试 |

## 开发分支

日常开发、提交和测试在 `dev` 进行。`main` 用于发布，经项目维护者明确要求后才合并 `dev`。
