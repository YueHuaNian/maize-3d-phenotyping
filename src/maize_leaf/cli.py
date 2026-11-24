"""Command-line entry point for maize_leaf."""
import argparse
from pathlib import Path
from .pipeline import run
from .surface import ModelConfig


def main():
    p = argparse.ArgumentParser(description="Literature-backed 3D maize leaf modeling and completion")
    p.add_argument("--input-root", type=Path, default=Path("dataset"))
    p.add_argument("--output", type=Path, default=Path("outputs/maize_v2"))
    p.add_argument("--plant", help="Plant stem name, e.g. 1-1-2")
    p.add_argument("--leaf", type=int, help="Leaf semantic ID")
    p.add_argument("--nu", type=int, default=90)
    p.add_argument("--nv", type=int, default=25)
    p.add_argument("--no-tip-completion", action="store_true")
    p.add_argument("--no-prior-calibration", action="store_true")
    a = p.parse_args()
    cfg = ModelConfig(nu=a.nu, nv=a.nv, allow_tip_completion=not a.no_tip_completion)
    run(a.input_root, a.output, a.plant, a.leaf, cfg, calibrate_prior=not a.no_prior_calibration)
    return 0
