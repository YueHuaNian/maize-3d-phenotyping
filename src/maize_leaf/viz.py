"""Offline Plotly HTML and Matplotlib PNG visualization helpers."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .surface import COLORS, REGIONS


STYLE = """
body{margin:0;background:#f3f6f5;color:#16372e;font:15px system-ui,'Microsoft YaHei',sans-serif}
header{padding:20px 28px;background:#113e33;color:white}h1{font-size:22px;margin:0 0 6px}
main{max-width:1450px;margin:auto;padding:18px 24px}a{color:#16715b}header a{color:#d6f5ec}
.card{background:white;border-radius:12px;padding:18px;margin-bottom:16px;border:1px solid #dce7e2}
.chips{display:flex;gap:12px;flex-wrap:wrap}.chip{background:#ecf3f0;padding:8px 14px;border-radius:8px}
table{border-collapse:collapse;width:100%;background:white}
th,td{text-align:left;padding:10px;border-bottom:1px solid #dce7e2}th{background:#e8f1ec}
img{max-width:100%;border-radius:8px}
"""


def _write_page(title, body, target, script=None):
    extra = f'<script src="{script}"></script>' if script else ""
    Path(target).write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        f"<style>{STYLE}</style>{extra}<body>{body}</body></html>",
        encoding="utf-8",
    )


def ensure_plotly_js(output):
    p = Path(output) / "assets" / "plotly.min.js"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(get_plotlyjs(), encoding="utf-8")


def scatter_trace(p, name, color, size=2, limit=9000, mode="markers", width=5):
    if len(p) > limit:
        p = p[np.linspace(0, len(p) - 1, limit).astype(int)]
    return go.Scatter3d(
        x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
        mode=mode, name=name, marker=dict(size=size, color=color, opacity=0.65),
        line=dict(color=color, width=width),
    )


def mesh_traces(result, prefix=""):
    v, faces = result["vertices"], result["faces"]
    traces = []
    for k, name in REGIONS.items():
        f = faces[result["face_regions"] == k]
        if not len(f):
            continue
        color = "rgb(" + ",".join(str(x) for x in COLORS[k]) + ")"
        traces.append(
            go.Mesh3d(
                x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
                color=color, name=prefix + name, showlegend=True, opacity=0.88,
            )
        )
    return traces


def apply_layout(fig):
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=15, b=0), paper_bgcolor="white",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   camera=dict(eye=dict(x=1.5, y=1.6, z=0.9))),
        legend=dict(orientation="h", y=1.05, x=0), uirevision="keep-camera",
    )


def save_static_plot(path, points, results, stem=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10.5, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7faf8")
    p = points[np.linspace(0, len(points) - 1, min(len(points), 7000)).astype(int)]
    ax.scatter(*p.T, s=0.8, color="#293e36", alpha=0.25)
    all_xyz = [p]
    for r in results:
        v = r["vertices"]
        coll = Poly3DCollection(v[r["faces"]], facecolors=COLORS[r["face_regions"]] / 255.0,
                                edgecolor="none", alpha=0.85, zsort="average")
        ax.add_collection3d(coll)
        ax.plot(*r["midrib"].T, color="#175eb4", linewidth=1.5)
        all_xyz.append(v)
    if stem is not None:
        ax.plot(*stem.T, color="#5134a1", linewidth=3)
        all_xyz.append(stem)
    ax.legend(handles=[Patch(color=COLORS[k] / 255.0, label=REGIONS[k]) for k in REGIONS], loc="upper left", fontsize=9)
    xyz = np.vstack(all_xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel="X", ylabel="Y", zlabel="Z", title=title)
    ax.set_box_aspect(np.maximum(hi - lo, 0.01))
    ax.view_init(elev=22, azim=42)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def export_leaf_html(path, plant_name, leaf_id, result, png_name):
    fig = go.Figure(
        [scatter_trace(result["points"], "Points", "#334c42"),
         scatter_trace(result["midrib"], "Midrib", "#175eb4", mode="lines")]
        + mesh_traces(result)
    )
    apply_layout(fig)
    tr = result["traits"]
    key = f"leaf_{leaf_id:02d}"
    chips = (
        f'<span class="chip">L_obs={tr["observed_length"]:.3f}</span>'
        f'<span class="chip">L_full={tr["completed_length"]:.3f}</span>'
        f'<span class="chip">W_max={tr["max_width"]:.3f}</span>'
        f'<span class="chip">alpha={tr["form_factor_alpha"]:.3f}</span>'
        f'<span class="chip">RMSE={tr["fit_rmse"]:.4f}</span>'
        f'<span class="chip">GCV λ={tr["tps_gcv_lambda"]:.2e}</span>'
    )
    body = (
        f'<header><h1>植株 {plant_name} · 叶片 {leaf_id}</h1>'
        f'<a href="../whole_plant.html">整株视图</a> · <a href="../../index.html">全部植株</a></header><main>'
        f'<div class="card"><div class="chips">{chips}</div></div>'
        f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})}</div>'
        f'<details class="card"><summary>静态图与导出</summary><img src="{png_name}">'
        f'<p><a href="{key}_surface.ply">PLY</a> · <a href="{key}_surface.obj">OBJ</a> · '
        f'<a href="{key}_traits.json">表型 JSON</a> · <a href="{key}_model.npz">模型 NPZ</a></p></details></main>'
    )
    _write_page(f"{plant_name} leaf {leaf_id}", body, path, "../../assets/plotly.min.js")


def export_plant_html(path, plant, stem, results):
    fig = go.Figure(
        [scatter_trace(plant.xyz, "Plant points", "#566c62", limit=16000),
         scatter_trace(stem, "Stem centerline", "#5134a1", mode="lines", width=7)]
    )
    rows = []
    for lid, r in results.items():
        fig.add_traces(mesh_traces(r, f"Leaf {lid} · "))
        fig.add_trace(scatter_trace(r["midrib"], f"Leaf {lid} midrib", "#175eb4", mode="lines", width=3))
        tr = r["traits"]
        rows.append(
            f'<tr><td><a href="leaves/leaf_{lid:02d}_visualization.html">叶片 {lid}</a></td>'
            f'<td>{tr["observed_length"]:.3f}</td><td>{tr["completed_length"]:.3f}</td>'
            f'<td>{tr["max_width"]:.3f}</td><td>{tr["form_factor_alpha"]:.3f}</td>'
            f'<td>{tr["surface_area"]:.4f}</td><td>{tr["fit_rmse"]:.4f}</td></tr>'
        )
    apply_layout(fig)
    body = (
        f'<header><h1>植株 {plant.name}</h1><a href="../index.html">全部植株</a></header><main>'
        f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})}</div>'
        f'<div class="card"><table><tr><th>叶片</th><th>观测叶长</th><th>补全叶长</th><th>最大宽</th><th>形状系数 α</th><th>叶面积</th><th>拟合 RMSE</th></tr>'
        f'{"".join(rows)}</table></div>'
        f'<details class="card"><summary>静态图与导出</summary><img src="whole_plant.png">'
        f'<p><a href="whole_plant_mesh.ply">PLY</a> · <a href="whole_plant_mesh.obj">OBJ</a> · '
        f'<a href="stem_centerline.json">茎秆中心线</a></p></details></main>'
    )
    _write_page(f"Plant {plant.name}", body, path, "../assets/plotly.min.js")


def export_index_html(path, plants, traits):
    rows = [
        f'<tr><td><a href="plant_{p["plant"]}/whole_plant.html">{p["plant"]}</a></td><td>{p["leaf_count"]}</td></tr>'
        for p in plants
    ]
    body = (
        "<header><h1>玉米叶片三维建模与缺失补全 V2.0</h1></header><main>"
        f'<div class="card"><div class="chips"><span class="chip">植株 {len(plants)}</span>'
        f'<span class="chip">叶片 {len(traits)}</span></div></div>'
        f'<div class="card"><table><tr><th>植株</th><th>叶片数</th></tr>{"".join(rows)}</table></div>'
        '<div class="card"><a href="traits.csv">每叶表型指标 CSV</a> · <a href="species_prior.json">标定叶形先验</a></div></main>'
    )
    _write_page("玉米叶片三维建模 V2.0", body, path)
