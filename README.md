# 玉米叶片三维建模与缺失补全（V3.0）

基于现代可微几何联合优化、$L_1$-Medial 骨架提取与植物形态学先验的玉米三维点云建模与缺失补全工具。

## 核心算法依据

- **双重去噪**：欧氏近邻连通分量聚类滤波（Rusu & Cousins, 2011）+ 统计离群点去除 SOR（Rusu et al., 2008）。
- **骨架与中脉**：$L_1$-Medial 点云收缩骨架（Huang et al., 2013, *ACM TOG*）+ 测地排序 + 弧长重参数化主曲线（Hastie & Stuetzle, 1989）+ 有界 GCV P-spline（Eilers & Marx, 1996）+ 牛顿正交投影（Piegl & Tiller, 1997）。
- **可微几何联合优化**：在观测域上对中脉偏移 B-spline、双反射 RMF 滚转角（Wang et al., 2008）与法向一致性、叶面横向拱度场及 Dornbusch 叶宽包络进行 `soft_l1` 联合反演优化，克服单侧缺失对中脉中心线的拉偏。
- **叶尖补全与叶面拟合**：Dornbusch et al. (2011, *Annals of Botany*) 分段幂函数 + Empirical Bayes 先验 + BIC 截断检验（$\Delta\mathrm{BIC} > 4$）+ Prévot et al. (1991) 倾角中脉外推 + 分箱中位数 GML-TPS（Duchon, 1977; Wahba, 1985）。

详细公式见 [docs/method.md](docs/method.md)。

## 快速运行

```bash
# 安装依赖
pip install -r requirements.txt

# 批量建模（默认读取 dataset/，输出至 outputs/maize_v2/）
python scripts/run.py

# 指定单株或单叶
python scripts/run.py --plant 1-1-2
python scripts/run.py --plant 1-1-2 --leaf 1

# 独立交互可视化（打开浏览器，支持自由切换全部植株与单个叶片）
python scripts/visualize.py
python scripts/visualize.py --list
python scripts/visualize.py --plant 1-1-2 --leaf 1
python scripts/visualize.py --plant 1-1-2 --leaf 1 --backend mpl --save leaf_01.png
python scripts/visualize.py --model outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz

# 真实玉米完整叶留出遮挡验证（孔洞 / 叶缘 / 15%叶尖截断）
python scripts/validate_holdout.py

# 单元测试
python scripts/run_tests.py
```

## 连续模型调用

```python
import numpy as np
from maize_leaf.surface import MaizeLeafModel

model = MaizeLeafModel.load("outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz")
sv = np.array([[0.2, -0.5], [0.5, 0.0], [0.8, 0.5]])  # [物理弧长 s, 归一化横向 v]
xyz = model.evaluate(sv)
t, b, n = model.frame.evaluate(sv[:, 0])
```
