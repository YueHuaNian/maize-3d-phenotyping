# 研究验证模块

本目录是实现模块，不是另一套独立脚本。所有任务从项目根目录通过两个统一入口运行。

| 旧脚本名 | 当前命令前缀 | 用途 |
|---|---|---|
| `validate_fit.py` | `python scripts/validate.py fit` | 基线与候选重建的逐叶审计 |
| `validate_holdout.py` | `python scripts/validate.py holdout` | 冻结遮挡留出验证 |
| `compare_holdout.py` | `python scripts/validate.py compare` | 核验冻结划分并做配对比较 |
| `validate_euclidean_clustering.py` | `python scripts/validate.py clustering` | 欧氏聚类半径与次簇阈值标定 |
| `validate_sor_sensitivity.py` | `python scripts/validate.py sor` | 分断片 SOR 参数标定 |
| `calibrate_trait_windows.py` | `python scripts/validate.py trait-windows` | 茎叶角与曲率窗口标定及配套图 |
| `experiment_tip_policy.py` | `python scripts/validate.py tip-ablation` | 复现已撤回的强制截尖策略 |
| `plot_holdout.py` | `python scripts/plot_validation.py holdout` | 留出汇总与案例诊断图 |
| `plot_euclidean_clustering.py` | `python scripts/plot_validation.py clustering` | 欧氏聚类敏感性图 |
| `plot_sor_sensitivity.py` | `python scripts/plot_validation.py sor` | SOR 敏感性图 |
| `plot_support_comparison.py` | `python scripts/plot_validation.py support` | 支撑域修正前后对照图 |
| `plot_ideotype_index.py` | `python scripts/plot_validation.py index` | 理想株型结构符合度指数四联诊断图（PNG/PDF/SVG） |

原有参数名继续有效。两个去噪绘图任务新增 `--root`，默认仍读取原标定结果目录；表型窗口标定新增 `--models-root`、`--cases-root`、`--output` 与 `--workers`，默认计算设置不变。

叶尖消融只在工作进程初始化时安装策略，正常导入不会覆盖默认拟合器；该任务要求尚不存在的 `--output` 目录。各轮留出验证仍检查输入、源码和配置指纹，不能将改版程序的产物混入已有冻结缓存。历史 JSON 中的原路径和哈希保持原样，迁移前源码与本次清单放在 `outputs/archive/20261125_stage_cleanup/`。

绘图应使用已配置 Matplotlib 的科研环境。生成图不会自动重跑验证；`trait-windows` 保留原有计算后同时输出配套图的行为。
