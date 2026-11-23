import argparse
from pathlib import Path
from .model import Config
from .pipeline import run


def main():
    p=argparse.ArgumentParser(description="Maize leaf modeling V1.0; read-only paired inputs")
    p.add_argument("--input-root",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("outputs/leaf_modeling_v1_0"))
    p.add_argument("--plant",help="Original file stem, e.g. 1-1-2")
    p.add_argument("--leaf",type=int,help="Original positive semantic ID")
    p.add_argument("--expect-plants",type=int,default=14)
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--nu",type=int,default=90)
    p.add_argument("--nv",type=int,default=25)
    p.add_argument("--tip-extension",type=float,default=.04)
    p.add_argument("--edge-expansion",type=float,default=.06)
    p.add_argument("--support-radius-factor",type=float,default=1.8)
    a=p.parse_args()
    if not (15<=a.nu<=400 and 9<=a.nv<=100):
        p.error("grid must satisfy 15<=nu<=400, 9<=nv<=100")
    if not 0<=a.tip_extension<=.10 or not 0<=a.edge_expansion<=.15:
        p.error("extension must be 0..0.10, expansion 0..0.15")
    if not .5<=a.support_radius_factor<=5:
        p.error("support-radius-factor must be 0.5..5; affects support labels, not truth")
    cfg=Config(nu=a.nu,nv=a.nv,tip_extension=a.tip_extension,edge_expansion=a.edge_expansion,
               support_radius_factor=a.support_radius_factor)
    return run(a.input_root,a.output,a.plant,a.leaf,cfg,a.expect_plants,a.validate_only)[0]
