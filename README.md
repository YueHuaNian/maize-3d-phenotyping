# 玉米点云三维表型分析

从带叶片标签的玉米点云重建三维叶面，提取表型并进行叶姿分类，支持整株与单叶交互查看。当前本地数据为 **14 株、90 片叶**。

## 快速开始

在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

已有本地重建结果时，直接打开查看器：

```bash
python scripts/visualize.py
```

从原始数据完整运行，依次执行四个阶段：

```bash
python scripts/run.py                              # 1. 点云去噪与三维叶面重建
python scripts/extract_traits.py                   # 2. 提取几何表型与信息熵
python scripts/classify_leaf_pose.py               # 3. 四因素叶姿分类
python scripts/visualize.py                        # 4. 交互查看
```

输入放在 `dataset/`：同名 TXT 点云与 CSV 标签文件配对，CSV 的 `semantic` 列中 0 表示茎秆，正整数表示叶片编号。原始数据只读；新实验请通过各入口的路径参数指定新的输出目录，参数用 `--help` 查看。

## 分类与显示

分类采用 **茎叶角、同株相对投影面积、俯仰角信息熵、相对着生高度** 四个因素，学习紧缩、舒展、下垂三种全局形态原型。紧缩候选限于顶部细小、尚未充分展开的叶片；**每株可以缺少某些类别，不强制凑齐三类**。

查看器支持点选叶片和类别筛选，默认保留整株背景并显示观测支撑面；可切换“仅看选中叶”和“显示插值/外推面”。当前重建分别约束左右观测边缘，截断叶尖保持开放。

导出可直接打开的离线查看器：

```bash
python scripts/visualize.py --export-viewer outputs/classification_v2/viewer
```

## 目录与结果

| 路径 | 用途 |
|---|---|
| `dataset/` | 原始点云与标签 |
| `scripts/` | 各阶段运行入口，每阶段 1–2 个 |
| `src/maize_leaf/` | 算法、验证模块与查看器模板 |
| `outputs/maize_v2/` | 当前三维模型、网格与质量审计 |
| `outputs/phenotypes_v2/` | 逐叶、逐株表型表与元数据 |
| `outputs/classification_v2/` | 分类表、报告与离线查看器 |
| `docs/`、`tests/` | 方法、验证记录与回归测试 |
| `outputs/archive/`、`.cache/` | 历史归档与运行缓存 |

原始数据、生成结果和环境依赖按 `.gitignore` 保留在本地，不随 Git 仓库分发。

## 测试与验证

```bash
python scripts/run_tests.py                 # Python 测试；安装 Node.js 时同时检查前端逻辑
python scripts/validate.py --help           # 贴合审计、遮挡留出、参数标定等
python scripts/plot_validation.py --help    # 已有验证结果的诊断绘图
```

- [各阶段脚本与参数](scripts/README.md)
- [完整技术路线：每个环节的具体步骤](docs/路线.md)
- [算法、分类规则与适用范围](docs/method.md)
- [近期分类与曲面修正验证记录](docs/validation/classification_feedback_2026-11-25.json)
