# 玉米叶片三维建模与缺失补全（V2.0）

基于植物形态学解析先验与微分几何/薄板力学的玉米三维点云建模与缺失补全工具。

## 核心算法依据

- **中脉与茎秆骨架**：测地排序初始化 + 弧长重参数化主曲线（Hastie & Stuetzle, 1989）+ GCV 自动选参 P-spline（Eilers & Marx, 1996; Wahba, 1990）+ 牛顿精确正交投影（Piegl & Tiller, 1997）。
- **无扭转移动标架**：双反射最小旋转标架 RMF（Bishop, 1975; Wang et al., 2008）+ 点云单位法向量 `(Nx, Ny, Nz)` 滚转角对齐。
- **叶宽轮廓与叶缘/叶尖补全**：Dornbusch et al. (2011, *Annals of Botany*) 分段幂函数叶形模型 + 批次完整叶先验标定 + Prévot et al. (1991) / ADEL-Maize 弧长倾角中脉外推。
- **叶面起伏与孔洞补全**：Duchon (1977) 薄板样条 TPS + Wahba (1990) QR 谱分解精确 GCV 自动求解最优平滑系数 $\lambda^*$。

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

# 独立交互/静态可视化
python scripts/visualize.py --plant 1-1-2
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
