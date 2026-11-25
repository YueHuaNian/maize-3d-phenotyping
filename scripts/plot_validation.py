"""研究诊断绘图统一入口：读取既有验证结果，不重跑重建。

示例：python scripts/plot_validation.py holdout --help
科研绘图请使用已配置 Matplotlib 的科研 Python 环境。
"""
from pathlib import Path
import sys

# 统一设置源码搜索路径；各图的字体缓存由实现模块配置。
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
from maize_leaf.experiments import PLOT_TASKS, dispatch

if __name__ == "__main__":
    raise SystemExit(dispatch(PLOT_TASKS))
