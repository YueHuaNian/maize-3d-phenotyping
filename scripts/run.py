"""阶段 1（主入口）：玉米点云三维建模与叶片曲面重建。

功能说明：
    从带语义标签的玉米点云目录（默认 dataset/）批量或按指定植株/叶片执行：
    1. 欧氏连通聚类 + 分断片 MAD-SOR 双重去噪；
    2. 茎秆中心线 B-spline 拟合；
    3. L1 骨架、中脉提取、法向标架、多候选几何拟合与 GML-TPS 连续曲面重建；
    4. 默认启用保守观测支撑域（独立限制左右边缘，不外推未观测叶尖）；
    5. 质量门禁检查，并导出单叶/整株网格（PLY/OBJ）、连续模型（NPZ）与汇总 CSV。

常用命令：
    python scripts/run.py                         # 全量重建至 outputs/maize_v2/
    python scripts/run.py --output outputs/demo   # 输出到新目录（避免覆盖旧结果）
    python scripts/run.py --plant 1-1-2 --leaf 1  # 仅重建指定植株与叶片
"""
import os
from pathlib import Path
import sys

# 定位项目根目录，并将本地依赖与源码目录加入模块搜索路径
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT / "src")]
# 将 Matplotlib 字体缓存固定在项目 .cache/ 目录，避免污染系统目录
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from maize_leaf.cli import main

if __name__ == "__main__":
    # 交由 maize_leaf.cli.main 解析命令行参数并执行重建流水线
    raise SystemExit(main())
