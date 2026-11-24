"""Interactive (Plotly) and static (Matplotlib) 3D visualization script."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from maize_leaf_v1.data import discover, load_pair
from maize_leaf_v1.model import COLORS, REGIONS, LeafModel, faces_grid


def read_ply(path):
    with open(path, encoding="ascii") as f:
        nv, nf = 0, 0
        for line in f:
            s = line.strip()
            if s.startswith("element vertex"):
                nv = int(s.split()[-1])
            elif s.startswith("element face"):
                nf = int(s.split()[-1])
            elif s == "end_header":
                break
        verts = np.loadtxt(f, max_rows=nv) if nv else np.empty((0, 6))
        faces = np.loadtxt(f, max_rows=nf, dtype=int) if nf else np.empty((0, 4), dtype=int)
    return np.atleast_2d(verts), np.atleast_2d(faces) if nf else np.empty((0, 4), dtype=int)


def load_leaf_output(leaves_dir, leaf_id):
    key = f"leaf_{leaf_id:02d}"
    pts, _ = read_ply(leaves_dir / f"{key}_points.ply")
    verts, faces = read_ply(leaves_dir / f"{key}_surface.ply")
    midrib = np.asarray(json.loads((leaves_dir / f"{key}_midrib.json").read_text(encoding="utf-8"))["xyz"])
    return dict(
        leaf_id=leaf_id,
        points=pts[:, :3],
        rgb=pts[:, 3:6].astype(np.uint8),
        vertices=verts[:, :3],
        faces=faces[:, 1:4],
        face_regions=faces[:, 4] if faces.shape[1] > 4 else np.zeros(len(faces), dtype=int),
        midrib=midrib,
    )


def load_from_outputs(root, plant=None, leaf=None):
    root = Path(root)
    if plant is None:
        plant_dir = sorted(root.glob("plant_*"))[0]
        plant = plant_dir.name.removeprefix("plant_")
    else:
        plant_dir = root / f"plant_{plant}"
    leaves_dir = plant_dir / "leaves"
    if leaf is not None:
        return dict(title=f"Plant {plant} / Leaf {leaf}", stem=None, leaves=[load_leaf_output(leaves_dir, leaf)])
    stem = np.asarray(json.loads((plant_dir / "stem_centerline.json").read_text(encoding="utf-8"))["xyz"])
    leaf_ids = sorted(int(p.name.split("_")[1]) for p in leaves_dir.glob("leaf_*_surface.ply"))
    leaves = [load_leaf_output(leaves_dir, lid) for lid in leaf_ids]
    return dict(title=f"Plant {plant} ({len(leaves)} leaves)", stem=stem, leaves=leaves)


def load_from_model(model_path, nu, nv):
    m = LeafModel.load(model_path)
    us, vs = np.linspace(0, 1 + m.extension, nu), np.linspace(-1, 1, nv)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    verts = m.evaluate(uv)
    faces = faces_grid(nu, nv)
    freg = np.where((uv[:, 0] > 1 + 1e-10)[faces].any(axis=1), 2, 0)
    return dict(
        title=Path(model_path).stem,
        stem=None,
        leaves=[dict(leaf_id=0, points=None, rgb=None, vertices=verts, faces=faces, face_regions=freg, midrib=m.center(us))],
    )


def load_from_raw(input_root, plant=None, leaf=None):
    pairs = discover(input_root)
    name, txt, csv_path = next(p for p in pairs if plant is None or p[0] == plant)
    data = load_pair(name, txt, csv_path)
    mask = (data.labels == leaf) if leaf is not None else (data.labels >= 0)
    return dict(
        title=f"Raw {name}" + (f" / Leaf {leaf}" if leaf is not None else ""),
        stem=None,
        leaves=[dict(leaf_id=leaf or 0, points=data.xyz[mask], rgb=data.rgb[mask], vertices=None, faces=None, face_regions=None, midrib=None)],
    )


def sample_points(pts, rgb, limit):
    if len(pts) <= limit:
        return pts, rgb
    idx = np.linspace(0, len(pts) - 1, limit).astype(int)
    return pts[idx], (rgb[idx] if rgb is not None else None)


def show_plotly(scene, args):
    import plotly.graph_objects as go

    fig = go.Figure()
    limit = 9000 if len(scene["leaves"]) == 1 else max(2000, 16000 // max(1, len(scene["leaves"])))

    if scene["stem"] is not None:
        s = scene["stem"]
        fig.add_trace(go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="lines", name="Stem", line=dict(color="#5134a1", width=7)))

    for item in scene["leaves"]:
        prefix = f"Leaf {item['leaf_id']} · " if len(scene["leaves"]) > 1 else ""
        if not args.no_points and item["points"] is not None:
            pts, rgb = sample_points(item["points"], item["rgb"], limit)
            colors = [f"rgb({r},{g},{b})" for r, g, b in rgb] if rgb is not None else "#334c42"
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=prefix + "Points",
                    marker=dict(size=2, color=colors, opacity=0.7),
                )
            )
        if not args.no_midrib and item["midrib"] is not None:
            m = item["midrib"]
            fig.add_trace(go.Scatter3d(x=m[:, 0], y=m[:, 1], z=m[:, 2], mode="lines", name=prefix + "Midrib", line=dict(color="#175eb4", width=4)))
        if not args.no_surface and item["vertices"] is not None:
            v, f, fr = item["vertices"], item["faces"], item["face_regions"]
            for k, name in REGIONS.items():
                fk = f[fr == k]
                if not len(fk):
                    continue
                color = "rgb(" + ",".join(str(x) for x in COLORS[k]) + ")"
                fig.add_trace(
                    go.Mesh3d(
                        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=fk[:, 0], j=fk[:, 1], k=fk[:, 2],
                        color=color, name=prefix + name, showlegend=True, opacity=0.88,
                    )
                )

    fig.update_layout(
        title=scene["title"],
        height=760,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out, include_plotlyjs=True)
        print(f"Saved {out.resolve()}")
    else:
        fig.show()


def show_mpl(scene, args):
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10.5, 8), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7faf8")
    all_xyz = []

    if scene["stem"] is not None:
        ax.plot(*scene["stem"].T, color="#5134a1", linewidth=3)
        all_xyz.append(scene["stem"])

    for item in scene["leaves"]:
        if not args.no_points and item["points"] is not None:
            pts, rgb = sample_points(item["points"], item["rgb"], 7000)
            c = rgb / 255.0 if rgb is not None else "#293e36"
            ax.scatter(*pts.T, s=1.0, c=c, alpha=0.4)
            all_xyz.append(pts)
        if not args.no_surface and item["vertices"] is not None:
            v = item["vertices"]
            coll = Poly3DCollection(v[item["faces"]], facecolors=COLORS[item["face_regions"]] / 255.0, edgecolor="none", alpha=0.85, zsort="average")
            ax.add_collection3d(coll)
            all_xyz.append(v)
        if not args.no_midrib and item["midrib"] is not None:
            ax.plot(*item["midrib"].T, color="#175eb4", linewidth=1.6)
            all_xyz.append(item["midrib"])

    xyz = np.vstack(all_xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel="X", ylabel="Y", zlabel="Z", title=scene["title"])
    ax.set_box_aspect(np.maximum(hi - lo, 0.01))
    ax.view_init(elev=args.elev, azim=args.azim)
    if not args.no_surface and any(item["vertices"] is not None for item in scene["leaves"]):
        ax.legend(handles=[Patch(color=COLORS[k] / 255.0, label=REGIONS[k]) for k in REGIONS], loc="upper left")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Visualize maize 3D leaf models and point clouds")
    p.add_argument("--root", type=Path, default=ROOT / "outputs/leaf_modeling_v1_0", help="Modeling output directory")
    p.add_argument("--input-root", type=Path, help="Raw TXT/CSV input directory")
    p.add_argument("--model", type=Path, help="Single leaf *_model.npz file")
    p.add_argument("--plant", help="Plant name, e.g. 1-1-2")
    p.add_argument("--leaf", type=int, help="Leaf semantic ID")
    p.add_argument("--backend", choices=("plotly", "mpl"), default="plotly")
    p.add_argument("--save", type=Path, help="Save to HTML (plotly) or PNG (mpl) instead of opening window")
    p.add_argument("--nu", type=int, default=90)
    p.add_argument("--nv", type=int, default=25)
    p.add_argument("--elev", type=float, default=22)
    p.add_argument("--azim", type=float, default=42)
    p.add_argument("--no-points", action="store_true")
    p.add_argument("--no-midrib", action="store_true")
    p.add_argument("--no-surface", action="store_true")
    args = p.parse_args()

    if args.model:
        scene = load_from_model(args.model, args.nu, args.nv)
    elif args.input_root:
        scene = load_from_raw(args.input_root, args.plant, args.leaf)
    else:
        scene = load_from_outputs(args.root, args.plant, args.leaf)

    if args.backend == "mpl":
        show_mpl(scene, args)
    else:
        show_plotly(scene, args)


if __name__ == "__main__":
    main()
