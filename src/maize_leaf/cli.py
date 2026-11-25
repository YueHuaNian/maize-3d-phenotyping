"""阶段 1 命令行入口：解析参数并调用三维建模流水线。"""
import argparse
from pathlib import Path
from .pipeline import run
from .surface import ModelConfig


def main():
    """解析命令行参数，组装 ModelConfig 并启动全量/单株/单叶重建。"""
    p = argparse.ArgumentParser(description="玉米叶片三维建模与缺失补全命令行入口")
    p.add_argument("--input-root", type=Path, default=Path("dataset"), help="原始点云根目录（只读）")
    p.add_argument("--output", type=Path, default=Path("outputs/maize_v2"), help="重建结果输出目录")
    p.add_argument("--plant", help="仅重建指定植株名称，例如 1-1-2")
    p.add_argument("--leaf", type=int, help="仅重建指定语义编号叶片")
    p.add_argument("--nu", type=int, default=90, help="纵向网格采样数")
    p.add_argument("--nv", type=int, default=25, help="横向网格采样数")
    p.add_argument("--max-tps-points", type=int, default=350, help="TPS 初始代表点上限")
    p.add_argument("--refine-tps-points", type=int, default=700, help="贴合超阈时 TPS 加密点数上限（<=初始值则关闭）")
    p.add_argument("--no-tip-completion", action="store_true", help="关闭叶尖长度先验补全")
    p.add_argument("--no-prior-calibration", action="store_true", help="跳过数据集完整叶形先验标定")
    p.add_argument("--allow-unobserved-completion", action="store_true", help="启用历史全补全参数域（默认使用保守观测支撑域）")
    a = p.parse_args()
    cfg = ModelConfig(nu=a.nu, nv=a.nv, allow_tip_completion=not a.no_tip_completion,
                      max_tps_points=a.max_tps_points, refine_tps_points=a.refine_tps_points,
                      conservative_support=not a.allow_unobserved_completion)
    run(a.input_root, a.output, a.plant, a.leaf, cfg, calibrate_prior=not a.no_prior_calibration)
    return 0
