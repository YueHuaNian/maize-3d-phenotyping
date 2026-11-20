"""Interactive viewer for raw maize points and published v1.0 geometry."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.io_utils import load_cloud, locate_data, sha256
from src.visualization import color, configure, setup_3d


def read_geometry(directory, cloud_path):
    details = directory / "attachment_details.json"
    if not details.is_file():
        return None
    report = json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
    hashes = report["audit"]["input_sha256_before"]
    result = json.loads(details.read_text(encoding="utf-8"))
    if result["plant_id"] != cloud_path.stem or sha256(cloud_path) not in hashes.values():
        raise ValueError(f"{cloud_path.name} 与所选几何结果的输入文件不一致")
    with (directory / "leaf_features.csv").open(encoding="utf-8-sig", newline="") as stream:
        features = {int(row["leaf_id"]): row for row in csv.DictReader(stream)}
    stem = np.loadtxt(directory / "stem_centerline.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1:4]
    return stem, result["leaves"], features


def draw_cloud(ax, xyz, labels, geometry, leaf_id, background, point_size):
    visible = (labels != -1) if not background else np.ones(len(labels), dtype=bool)
    if leaf_id is not None:
        visible &= (labels == leaf_id) | (labels == 0)
    for label in np.unique(labels[visible]):
        points = xyz[visible & (labels == label)]
        ax.scatter(*points.T, s=point_size, color=color(int(label)), linewidths=0,
                   depthshade=False, alpha=.28 if geometry else .8, zorder=1)
    setup_3d(ax, xyz[visible])
    if geometry is None:
        return ""
    stem, leaves, features = geometry
    ax.plot(*stem.T, color="#202020", linewidth=2, zorder=5)
    for leaf in leaves:
        label = leaf["leaf_id"]
        if leaf_id is not None and label != leaf_id:
            continue
        curve = leaf["visible_curve"]
        if curve:
            ax.plot(*np.asarray(curve).T, color=color(label), linewidth=1.8, zorder=4)
            ax.text(*curve[-1], str(label), fontsize=9, zorder=6)
        candidate = leaf["attachment"]["selected"]
        if candidate:
            ax.scatter(*candidate["xyz"], color="#202020", marker="x", s=32, zorder=6)
    caption = "黑线为茎秆中心线，彩线为可见中线，黑叉为叶基候选。"
    if leaf_id is not None:
        item = features[leaf_id]
        formal = float(item["leaf_angle_deg"])
        diagnostic = float(item["diagnostic_angle_deg"])
        formal_text = f"{formal:.2f}°" if np.isfinite(formal) else "无"
        diag_text = f"{diagnostic:.2f}°" if np.isfinite(diagnostic) else "无"
        caption += f"\n中线 {item['midline_confidence']} · 正式夹角 {formal_text} · 诊断夹角 {diag_text}"
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="打开一个 CSV 或原始 TXT")
    parser.add_argument("--sample", help="样本编号，例如 2-5-1")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "v1.0",
                        help="含逐株结果目录的批次目录")
    parser.add_argument("--raw", action="store_true", help="只看原始点云")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--leaf", type=int, help="聚焦指定原始叶号")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--save", type=Path, help="保存静态预览")
    args = parser.parse_args()
    if args.point_size <= 0:
        parser.error("--point-size 必须大于 0")
    path = args.file
    if path is None:
        data = locate_data(args.data_dir)
        path = data / f"{args.sample}.csv" if args.sample else sorted(data.glob("*.csv"))[0]
    xyz, labels = load_cloud(path)
    if args.leaf is not None and (args.leaf <= 0 or args.leaf not in labels):
        parser.error(f"{path.stem} 无叶号 {args.leaf}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = None if args.raw else read_geometry(args.results_dir / path.stem, path)
    configure()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    caption = draw_cloud(ax, xyz, labels, geometry, args.leaf, not args.no_background, args.point_size)
    leaf_text = f" · 叶 {args.leaf}" if args.leaf is not None else ""
    mode = "v1.0 几何诊断" if geometry else "原始点云"
    ax.set_title(f"{path.stem}{leaf_text} · {mode}")
    if caption:
        fig.text(.05, .035, caption, fontsize=9)
    if args.save:
        fig.savefig(args.save, dpi=180, facecolor="white")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
