"""研究验证统一入口：贴合、留出、去噪标定、表型窗口与历史消融。

示例：python scripts/validate.py holdout --help
实现位于 src/maize_leaf/experiments/，不在入口中复制算法。
"""
from pathlib import Path
import sys

# 允许从任意工作目录调用，并优先使用项目内源码及依赖。
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
from maize_leaf.experiments import VALIDATION_TASKS, dispatch

if __name__ == "__main__":
    raise SystemExit(dispatch(VALIDATION_TASKS))
