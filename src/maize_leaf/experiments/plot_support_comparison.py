"""绘制叶片支撑域收紧前后的网格和扫描对照。

Diagnostic before/after plots of the two reported leaf support problems (Python)."""
import argparse
import os
from pathlib import Path

# 诊断绘图（重建阶段）：指定叶片观测支撑域修正前后三维对比图
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from maize_leaf.io import read_ply
from maize_leaf.surface import MaizeLeafModel


def main(argv=None):
    """读取修正前后模型，绘制指定叶片的观测支撑域对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "outputs/archive/20261125_feedback_fix/reconstruction_before")
    parser.add_argument("--after", type=Path, default=ROOT / "outputs/maize_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/maize_v2/diagnostics/support_comparison.png")
    args = parser.parse_args(argv)
    fig = plt.figure(figsize=(13, 10), facecolor="white")
    for row, (plant, leaf) in enumerate((("2-2-1", 3), ("2-1-1", 3))):
        key = Path(f"plant_{plant}/leaves/leaf_{leaf:02d}")
        points = read_ply(args.before / f"{key}_points.ply")[0][:, :3]
        models = [MaizeLeafModel.load(base / f"{key}_model.npz") for base in (args.before, args.after)]
        meshes = [(m.evaluate(m.mesh_grid()[0]), m.mesh_grid()[1]) for m in models]
        extent = np.vstack([points] + [v for v, _ in meshes])
        lo, hi = extent.min(axis=0), extent.max(axis=0)
        for col, ((vertices, faces), model) in enumerate(zip(meshes, models)):
            ax = fig.add_subplot(2, 2, row*2+col+1, projection="3d")
            ax.scatter(*points.T, color="#263c34", s=.8, alpha=.35, rasterized=True)
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor=("#e7a45b" if col == 0 else "#3b9a78"),
                                                  edgecolor="none", alpha=.72))
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel="X", ylabel="Y", zlabel="Z")
            ax.set_box_aspect(np.maximum(hi-lo, .01))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=8, pad=0)
            ax.view_init(elev=22, azim=42)
            ax.set_title(f"{plant} / L{leaf} — {'Before' if col == 0 else 'Supported domain' + (' (open end)' if model.open_tip else '')}", fontsize=11)
    fig.subplots_adjust(left=.02, right=.96, bottom=.04, top=.94, wspace=.05, hspace=.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)
