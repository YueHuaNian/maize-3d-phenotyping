# 脚本目录说明

`scripts/` 只保留以下 8 个执行入口，每阶段 1–2 个。算法放在 `src/maize_leaf/`，验证与绘图实现放在其 `experiments/` 子包；不需要逐个运行库模块。

| 阶段 | 脚本 | 用途 | 典型命令 |
|---|---|---|---|
| **阶段 1：三维建模与支撑域修正** | `run.py` | 点云去噪、茎秆拟合、单叶三维曲面重建主入口 | `python scripts/run.py` |
| | `refine_surface_support.py` | 基于已有模型导出保守观测支撑域并审计（不重跑 TPS） | `python scripts/refine_surface_support.py --source <旧目录> --output <新目录>` |
| **阶段 2：表型提取** | `extract_traits.py` | 提取单叶/整株几何表型、方向信息熵、光遮挡与冠层包络 | `python scripts/extract_traits.py` |
| **阶段 3：姿态三分类** | `classify_leaf_pose.py` | 四因素三分类（紧缩/舒展/下垂）、LOPO 稳定性与诊断页 | `python scripts/classify_leaf_pose.py --height-weight 0.5` |
| **阶段 4：三维交互可视化** | `visualize.py` | 启动在线三维查看器、导出离线查看器或输出静态图 | `python scripts/visualize.py --export-viewer outputs/classification_v2/viewer` |
| **研究验证与诊断** | `validate.py` | 统一执行贴合、留出、配对比较、去噪标定、表型窗口与历史消融 | `python scripts/validate.py --help` |
| | `plot_validation.py` | 统一绘制已有验证结果与支撑域对照 | `python scripts/plot_validation.py --help` |
| **工程测试** | `run_tests.py` | 运行全部 Python 单元测试及查看器前端逻辑测试 | `python scripts/run_tests.py` |

验证入口的任务名为 `fit`、`holdout`、`compare`、`clustering`、`sor`、`trait-windows`、`tip-ablation`；绘图入口的任务名为 `holdout`、`clustering`、`sor`、`support`。任务名后加 `--help` 查看该任务参数：

```bash
python scripts/validate.py holdout --help
python scripts/plot_validation.py holdout --help
```

`run.py` 与 `refine_surface_support.py` 是两条建模路径：前者从原始点云重建，后者从旧模型另存支撑域修正结果。通常只需选择一条，再依次运行表型、分类与查看器入口。分类学习全局三个形态原型，各株可以缺少某些类别。

新实验显式指定新的输出目录。冻结验证结果中的旧路径与源码哈希保留原始记录；旧入口到新命令的对应关系见 [实验模块说明](../src/maize_leaf/experiments/README.md)。
