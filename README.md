# 玉米叶片三维建模与缺失补全 V1.0

读取成对的 TXT / CSV 点云，以原始叶片编号建立连续三维参数曲面，输出单叶与整株的 PLY、OBJ、PNG、离线交互 HTML、连续模型文件及基础指标。

## 环境与运行

Python 3.11+，依赖见 [requirements.txt](requirements.txt)：

```bash
python -m venv .venv
pip install -r requirements.txt
```

直接使用 Python 脚本运行：

```bash
# 全部植株
python scripts/run_leaf_modeling.py --input-root <数据目录>

# 指定单株 / 单叶
python scripts/run_leaf_modeling.py --input-root <数据目录> --plant 1-1-2
python scripts/run_leaf_modeling.py --input-root <数据目录> --plant 1-1-2 --leaf 1 --output outputs/single_leaf

# 运行测试与合成补全示例
python scripts/test_v1.py
python scripts/completion_demo.py

# 独立可视化（默认 Plotly 交互，支持 --backend mpl 与 --save 导出）
python scripts/visualize.py --plant 1-1-2
python scripts/visualize.py --plant 1-1-2 --leaf 1 --save outputs/leaf_01.html
python scripts/visualize.py --plant 1-1-2 --backend mpl --save outputs/plant_1-1-2.png
python scripts/visualize.py --model outputs/leaf_modeling_v1_0/plant_1-1-2/leaves/leaf_01_model.npz
```

## 可调参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--input-root` | 必填 | 包含成对 TXT 和 CSV 的目录 |
| `--output` | `outputs/leaf_modeling_v1_0` | 输出目录 |
| `--plant` | 全部 | TXT/CSV 文件名主干 |
| `--leaf` | 该株全部 | 原始正整数叶片编号 |
| `--nu` / `--nv` | 90 / 25 | 纵向 / 横向采样网格节点数 |
| `--tip-extension` | 0.04 | 端部较宽时的额外参数延伸比例（0 关闭） |
| `--edge-expansion` | 0.06 | 宽度包络相对扩张比例 |
| `--support-radius-factor` | 1.8 | 支持区判定距离系数 |

## 输出结构

```text
<output>/
  index.html
  assets/plotly.min.js
  metrics.csv
  plant_<主干>/
    whole_plant.html / .png
    whole_plant_mesh.ply / .obj / .mtl
    stem_centerline.json / .ply
    leaves/
      leaf_<编号>_points.ply
      leaf_<编号>_surface.ply / .obj / .mtl
      leaf_<编号>_visualization.html / .png
      leaf_<编号>_midrib.json / _width.json / _frames.json
      leaf_<编号>_model.json / .npz
      leaf_<编号>_regions.json / _metrics.json
```

区域标签：`0` = `observed`（绿），`1` = `interpolated`（黄），`2` = `extrapolated`（粉）。

## 连续模型调用

```python
import numpy as np
from maize_leaf_v1.model import LeafModel

model = LeafModel.load("outputs/leaf_modeling_v1_0/plant_1-1-2/leaves/leaf_01_model.npz")
uv = np.array([[0.25, -0.5], [0.50, 0.0], [0.75, 0.5]])
xyz = model.evaluate(uv)
t, b, n = model.frame(uv[:, 0])
```

核心算法说明见 [docs/method.md](docs/method.md)。
