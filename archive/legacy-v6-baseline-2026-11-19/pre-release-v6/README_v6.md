# V6 全叶实例几何分析

在代码目录运行 `python run_analysis_v6.py --batch`。使用 `--sample 1-1-2` 可单独分析一株。输入目录可用 `--data-dir` 与 `--txt-dir` 指定。每次新建 `results/v6/run_时间戳/`，保留 V5 历史和原始数据。

语义按研究者确认的定义解析。`-1` 为背景或噪声，`0` 为茎秆，每个出现的正整数是独立的 `LEAF_INSTANCE`。程序只取当株实际出现的叶号，保持原号和原坐标。每株按行校验 CSV 与 TXT，并检查输入文件前后的 SHA-256。

V6 复用 V5 的叶基候选和 V4 的茎线、中线与正式质量门槛。每片实际叶都执行正式检查。`leaf_angle_deg`、`curvature_rms_norm` 等正式列来自原规则重算，通过不了的保留 NaN。`height_ratio_raw`、`diagnostic_*` 与 `visible_midline_length` 单独保留。`attachment_confidence` 是 V5 诊断等级，`v4_base_confidence` 是原正式门槛下的叶基等级。`LEAF_INSTANCE` 表示语义身份，几何可信度仍由原评分控制。

叶序按实际叶号两两比较，原几何高度不被修正。后编号估计高度大于前编号为反序，以归一化高度差标记 SMALL、MODERATE 或 SEVERE。严格降序要求下等高单独标为 TIE。缺候选高度的叶对保留并标记 MISSING_HEIGHT，不能判断冲突。逐叶列记录最严重的可比结果。

- `all_leaf_features.csv` 保存所有叶号的正式和诊断列。逐株目录另有 `v4_formal_reference.csv`。
- `leaf_position_summary.csv` 以该叶号实际出现的植株数为分母，提供点数分布、各等级计数和通过比例。
- `batch_plant_summary.csv`、`attachment_candidate_comparison.csv`、`midline_method_comparison.csv`、`height_order_diagnostics.csv` 提供逐株及逐方法核验。
- `batch_quality_report.json` 和 `batch_failure_analysis.md` 保存验收计数与历史对照。旧版来源按逐株输入哈希核实，可用 `--v5-baseline` 指定 V5 批次目录。

植株是独立样本单位。单叶指标用于描述批次中的几何覆盖与质量，正式值仍是点云几何估计。本轮不计算 PAI、LMI 或聚类。
