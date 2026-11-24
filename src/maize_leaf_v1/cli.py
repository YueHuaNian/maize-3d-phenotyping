import argparse
from pathlib import Path
from .model import Config
from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="Maize leaf 3D modeling")
    p.add_argument("--input-root", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/leaf_modeling_v1_0"))
    p.add_argument("--plant", help="File stem, e.g. 1-1-2")
    p.add_argument("--leaf", type=int, help="Leaf semantic ID")
    p.add_argument("--nu", type=int, default=90)
    p.add_argument("--nv", type=int, default=25)
    p.add_argument("--tip-extension", type=float, default=0.04)
    p.add_argument("--edge-expansion", type=float, default=0.06)
    p.add_argument("--support-radius-factor", type=float, default=1.8)
    a = p.parse_args()
    cfg = Config(
        nu=a.nu,
        nv=a.nv,
        tip_extension=a.tip_extension,
        edge_expansion=a.edge_expansion,
        support_radius_factor=a.support_radius_factor,
    )
    run(a.input_root, a.output, a.plant, a.leaf, cfg)
    return 0
