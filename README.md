# 玉米叶片三维建模与缺失补全可视化 V1.0

读取成对的 TXT / CSV，以原始叶片编号建立连续三维参数曲面，输出单叶、整株交互 HTML、PNG、PLY、OBJ、样条及逐叶质量指标。输入文件只读。

本批次入口：**[打开14株结果](outputs/leaf_modeling_v1_0/index.html)**。在文件管理器中双击该 HTML，用支持 WebGL 的本机浏览器打开。HTML 不依赖网络，移动结果时须保留整个输出目录，尤其是 `assets/plotly.min.js`。根目录页面进入植株，再点击原编号叶片；图例可显示/隐藏点云、中脉、曲面各区域，鼠标可旋转、缩放。PNG 可直接查看。

上述结果入口及验收文档中的 `outputs/` 链接指向本地生成产物。Git仓库仅包含代码、文档和测试，不包含原始点云、生成结果或本地Python环境；从GitHub克隆后，需提供自己的配对输入并运行程序生成结果。

已完成本地数据的单叶、单株、14株分级运行：14组匹配，90片叶成功，0片失败。按当前保守支持阈值，22片为 `usable_with_inference`，68片为 `review`。这表示模型输出成功与算法支持程度，不等于全部形态已被真实测量验证。验收记录见 [validation.md](docs/validation.md)。

## 环境与运行

Python 3.11+，依赖见 `requirements.txt`。推荐在自己的虚拟环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MAIZE_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
$env:MAIZE_INPUT_ROOT = '填写包含14组TXT和CSV的父目录'
```

本机本次使用已经核验的 Nature Figure 独立 Python 科学计算环境，Plotly 7.1.0 安装在项目 `.vendor/` 内；没有修改该外部科学计算环境。`run_v1.ps1` 未指定 Python 时先读 `MAIZE_PYTHON`，否则检查本机2026-11-18记录的 Nature Figure 环境路径。换机器后设置 `-PythonPath` 或环境变量，不依赖历史路径。本次曾尝试创建项目 `.venv`，其 ensurepip 未完成，因此本批次不使用该目录。

依次运行：

```powershell
# 1株1叶
./run_v1.ps1 -InputRoot $env:MAIZE_INPUT_ROOT --plant 1-1-2 --leaf 1 --output outputs/leaf_modeling_v1_0/validation/my_single_leaf

# 1株全部叶
./run_v1.ps1 -InputRoot $env:MAIZE_INPUT_ROOT --plant 1-1-2 --output outputs/leaf_modeling_v1_0/validation/my_single_plant

# 14株
./run_v1.ps1 -InputRoot $env:MAIZE_INPUT_ROOT

# 只检查输入匹配
./run_v1.ps1 -InputRoot $env:MAIZE_INPUT_ROOT --validate-only --output outputs/leaf_modeling_v1_0/validation/my_input_check
```

也可直接运行：

```powershell
& $env:MAIZE_PYTHON scripts/run_leaf_modeling.py --input-root $env:MAIZE_INPUT_ROOT
& $env:MAIZE_PYTHON scripts/test_v1.py
& $env:MAIZE_PYTHON scripts/audit_v1.py outputs/leaf_modeling_v1_0
& $env:MAIZE_PYTHON scripts/completion_demo.py
```

输出目录已有文件时，新运行自动写入其 `runs/<时间戳>/`，不会覆盖已有结果。新的完整批次默认入口以命令末尾 `RESULT` 打印的实际目录为准。初次分级验证的 `validation/` 可以与顶层完整批次并存。失败返回非零退出码，记录在 `errors.jsonl`，具体叶片另有 `status=failed` 的指标 JSON。

## 可调参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--plant` | 全部 | TXT/CSV文件名主干 |
| `--leaf` | 该株全部 | 原始正整数编号，须配合 `--plant` |
| `--nu` / `--nv` | 90 / 25 | 规则网格的纵向/横向节点数 |
| `--tip-extension` | 0.04 | 端部仍较宽时，额外参数长度比例；允许0–0.10，0关闭 |
| `--edge-expansion` | 0.06 | 平滑宽度包络的相对扩张；允许0–0.15 |
| `--support-radius-factor` | 1.8 | 附近测量支持的距离系数；改变区域标签与补全比，不代表真实缺失量 |
| `--expect-plants` | 14 | 期望成对文件数；缺文件立即记录错误 |

## 输出

```text
outputs/leaf_modeling_v1_0/
  index.html                     # 14株总入口
  assets/plotly.min.js            # 离线共享脚本
  summary.json                   # 版本、参数、运行状态、依赖
  input_validation.json          # 匹配、点数、标签、SHA256
  metrics.csv                    # 每叶质量、补全比例、拟合诊断
  errors.jsonl                    # 0错误时为空文件
  plant_<原文件主干>/
    whole_plant.html / .png
    whole_plant_mesh.ply / .obj / .mtl
    stem_centerline.json / .ply
    leaves/
      leaf_<原编号>_points.ply
      leaf_<原编号>_surface.ply / .obj / .mtl
      leaf_<原编号>_visualization.html / .png
      leaf_<原编号>_midrib.json
      leaf_<原编号>_width.json
      leaf_<原编号>_frames.json
      leaf_<原编号>_model.json / .npz
      leaf_<原编号>_regions.json
      leaf_<原编号>_metrics.json
  validation/                    # 分级运行、合成缺失测试、产物验收
```

PLY 顶点带 `region`、`leaf_id`，面带 `region`；OBJ通过同名MTL区分区域。整株网格包含全部成功叶片，茎秆以中心线单独导出。原始点云点数与编号保留；内部拟合可排除离群点，但原始点云 PLY 保留全部该叶点。

为控制页面与图片大小，点云显示有确定性抽样：单叶HTML最多9000点、整株HTML最多16000点、PNG最多7000点。抽样点坐标直接来自输入；叶片点云PLY仍保留全部原始点，显示抽样不影响已导出的曲面和逐叶指标。

## 区域与质量解释

| 区域 | 颜色 | 判定 |
|---|---|---|
| `observed` / 0 | 绿色 | 位于观测参数凸包内，附近有点云支持的**拟合曲面**，不是直接观测的三角面 |
| `interpolated` / 1 | 黄色 | 凸包内缺乏近邻支持的区域，包括内部孔洞 |
| `extrapolated` / 2 | 粉色 | 凸包外的叶缘或超出观测终端的推断 |

一个三角面采用其三个顶点中最保守的标签：外推优先于插值，插值优先于观测。补全比 =（插值面积 + 外推面积）/ 总曲面面积。该比例依赖支持阈值、网格和建模假设，**不是有完整真值确认的生物学缺失面积比例**。参数凸包也不能识别每一处真实叶缘损伤。

| 质量或标记 | 含义 |
|---|---|
| `usable_with_inference` | 没有触发其他复核项，仍有推断 |
| `review` | 至少一项稀疏、连通性、覆盖或残差诊断需复核 |
| `sparse_point_cloud` | 拟合点少于300 |
| `isolated_points_excluded` | 排除的孤立点超过去重点数2% |
| `disconnected_cloud_bridged` | 近邻图不连通，显式连接分量，桥长另存 |
| `high_extrapolation` | 外推面积超过25% |
| `low_observation_coverage` | 按保守面标签，补全面积超过50% |
| `high_fit_residual` | 拟合RMSE超过最大宽度18% |
| `tip_length_assumed` | 使用了固定比例叶尖延伸假设 |
| `rapid_centerline_turn` | 检测到局部中脉方向急变 |

TXT/CSV的物理坐标单位未给出，不标为米或厘米；长度使用源坐标单位，面积使用其平方。`centerline_length` 是中脉分箱节点折线的近似长度，也是纵向参数尺度；`fit_rmse` 是拟合数据内的误差，`point_to_grid_*` 是点到采样网格顶点的距离，均不能替代独立的补全真值精度。

## 连续模型调用

```python
# 将项目 src 加到 PYTHONPATH，或运行时添加到 sys.path
import numpy as np
from maize_leaf_v1.model import LeafModel

model = LeafModel.load('outputs/leaf_modeling_v1_0/plant_1-1-2/leaves/leaf_01_model.npz')
uv = np.array([[0.25, -0.5], [0.50, 0.0], [0.75, 0.5]])
xyz = model.evaluate(uv)   # 任意参数位置求值，不限于导出的网格节点
t, b, n = model.frame(uv[:, 0])
```

参数范围见每叶 `model.json`。数学方法、假设与来源见 [method.md](docs/method.md)。
