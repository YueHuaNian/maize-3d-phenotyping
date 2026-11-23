"""Offline HTML, static PNG, annotated meshes and reproducible model export."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .model import COLORS, REGIONS, spline_dict


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export_model_metadata(folder, key, model, u):
    """Human-readable continuous model and sampled orthonormal frame."""
    folder=Path(folder)
    t,b,n=model.frame(np.asarray(u))
    json_write(folder/f"{key}_frames.json",dict(u=np.asarray(u).tolist(),t=t.tolist(),b=b.tolist(),n=n.tolist(),
        definition="t=normalize(C'); b=normalize(B-(B dot t)t); n=t cross b"))
    json_write(folder/f"{key}_model.json",dict(format_version="1.0",equation="S(u,v)=C(u)+v*w(u)/2*b(u)+q(u,v)*n(u)",
        domain=dict(u=[0,1+model.extension],v=[-1,1]),
        centerline=spline_dict(model.center),frame_guide=spline_dict(model.lateral),width=spline_dict(model.width),
        q=dict(kernel="thin_plate_spline",degree=1,smoothing=model.cfg.rbf_smoothing,
            training_uv=model.rbf_xy.tolist(),training_normalized_displacement=model.rbf_d.tolist(),
            scale=model.q_scale,cap=model.q_cap,
            evaluation="q=cap*tanh(RBF(min(u,1),v)*scale/cap)*min(1,effective_width/tip_width)"),
        width_floor=model.width_floor,tip_width=model.tip_width,tip_extension=model.extension,
        tip_rule="For u>1, width(1)*[1-(1-width_floor/tip_width)*(3e^2-2e^3)], e=(u-1)/tip_extension",
        inference="geometric midrib estimate; assumed edge expansion and tip length; no intact-leaf ground truth",
        npz_file=f"{key}_model.npz"))


def ply(path, points, faces=None, colors=None, regions=None, face_regions=None, leaf_ids=None):
    points = np.asarray(points)
    if colors is None:
        colors = np.tile([140, 155, 148], (len(points), 1))
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\ncomment region 0 observed 1 interpolated 2 extrapolated\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if regions is not None:
            f.write("property uchar region\n")
        if leaf_ids is not None:
            f.write("property int leaf_id\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
            if face_regions is not None:
                f.write("property uchar region\n")
        f.write("end_header\n")
        for i, (p, c) in enumerate(zip(points, colors)):
            tail = (f" {int(regions[i])}" if regions is not None else "")
            tail += f" {int(leaf_ids[i])}" if leaf_ids is not None else ""
            f.write(" ".join(f"{x:.9g}" for x in p)+" "+" ".join(str(int(x)) for x in c)+tail+"\n")
        if faces is not None:
            for i, face in enumerate(faces):
                tail = f" {int(face_regions[i])}" if face_regions is not None else ""
                f.write("3 "+" ".join(str(int(x)) for x in face)+tail+"\n")


def obj(path, vertices, faces, face_regions):
    mtl = Path(path).with_suffix(".mtl")
    with mtl.open("w", encoding="ascii") as f:
        for k, name in REGIONS.items():
            f.write(f"newmtl {name}\nKd "+" ".join(f"{x/255:.5f}" for x in COLORS[k])+"\nd 1\n\n")
    with open(path, "w", encoding="ascii") as f:
        f.write(f"mtllib {mtl.name}\n")
        for p in vertices:
            f.write("v "+" ".join(f"{x:.9g}" for x in p)+"\n")
        for k, name in REGIONS.items():
            f.write(f"usemtl {name}\n")
            for face in faces[face_regions==k]+1:
                f.write("f "+" ".join(str(x) for x in face)+"\n")


STYLE = """
body{margin:0;background:#f3f6f5;color:#16372e;font:15px system-ui,'Microsoft YaHei',sans-serif}
header{padding:22px 30px;background:#113e33;color:white}h1{font-size:24px;margin:0 0 8px}
main{max-width:1450px;margin:auto;padding:18px 24px}a{color:#16715b}header a{color:#d6f5ec}
.card{background:white;border-radius:12px;padding:18px;margin-bottom:16px;border:1px solid #dce7e2}
.chips{display:flex;gap:12px;flex-wrap:wrap}.chip{background:#ecf3f0;padding:10px 16px;border-radius:8px}
.note{color:#53665f;line-height:1.65}table{border-collapse:collapse;width:100%;background:white}
th,td{text-align:left;padding:11px;border-bottom:1px solid #dce7e2}th{background:#e8f1ec}
.ok{color:#16715b}.warn{color:#a24430}details{margin:12px 0}img{max-width:100%;border-radius:8px}
"""


def page(title, body, target, script=None):
    extra = f'<script src="{script}"></script>' if script else ""
    Path(target).write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
        f'<style>{STYLE}</style>{extra}<body>{body}</body></html>', encoding="utf-8")


def ensure_js(output):
    p = Path(output)/"assets"/"plotly.min.js"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(get_plotlyjs(), encoding="utf-8")


def scatter(p, name, color, size=2, limit=9000, mode="markers", width=5):
    if len(p) > limit:
        p = p[np.linspace(0, len(p)-1, limit).astype(int)]
    return go.Scatter3d(x=p[:, 0].tolist(), y=p[:, 1].tolist(), z=p[:, 2].tolist(),
        mode=mode, name=name, marker=dict(size=size, color=color, opacity=.65),
        line=dict(color=color, width=width), hovertemplate=name+"<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>")


def mesh_traces(result, prefix=""):
    v, faces = result["vertices"], result["faces"]
    traces = []
    for k, name in REGIONS.items():
        f = faces[result["face_regions"]==k]
        if not len(f):
            continue
        color = "rgb("+",".join(str(x) for x in COLORS[k])+")"
        traces.append(go.Mesh3d(x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
            i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(), color=color,
            name=prefix+name, showlegend=True, opacity=.88, flatshading=False,
            hovertemplate=prefix+name+"<extra></extra>"))
    return traces


def layout(fig):
    fig.update_layout(height=720, margin=dict(l=0,r=0,t=15,b=0), paper_bgcolor="white",
        scene=dict(aspectmode="data", xaxis_title="X (source units)", yaxis_title="Y (source units)",
                   zaxis_title="Z (source units)", camera=dict(eye=dict(x=1.5,y=1.6,z=.9))),
        legend=dict(orientation="h", y=1.05, x=0), uirevision="keep-camera")


def static_plot(path, points, results, stem=None, title=""):
    fig = plt.figure(figsize=(10.5, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7faf8")
    p = points[np.linspace(0,len(points)-1,min(len(points),7000)).astype(int)]
    ax.scatter(*p.T, s=.8, color="#293e36", alpha=.25, label="Raw points")
    all_vertices = [p]
    for r in results:
        verts = r["vertices"]
        coll = Poly3DCollection(verts[r["faces"]], facecolors=COLORS[r["face_regions"]]/255,
                                edgecolor="none", alpha=.85, zsort="average")
        ax.add_collection3d(coll)
        ax.plot(*r["midrib"].T, color="#175eb4", linewidth=1.4)
        all_vertices.append(verts)
    if stem is not None:
        ax.plot(*stem.T, color="#5134a1", linewidth=3, label="Stem centerline")
        all_vertices.append(stem)
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[k]/255, label=REGIONS[k]) for k in REGIONS]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    xyz = np.vstack(all_vertices)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2],hi[2]),
           xlabel="X", ylabel="Y", zlabel="Z", title=title)
    ax.set_box_aspect(np.maximum(hi-lo, .01))
    ax.view_init(elev=22, azim=42)
    fig.text(.06,.025,"Source coordinate units; colored surfaces include model inference.", fontsize=9, color="#52645c")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def export_leaf(folder, plant, leaf_id, result):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = f"leaf_{leaf_id:02d}"
    m, v, f, reg = result["model"], result["vertices"], result["faces"], result["regions"]
    result["metrics"]["plant"] = plant.name
    ply(folder/f"{key}_points.ply", result["points"], colors=plant.rgb[plant.labels==leaf_id])
    ply(folder/f"{key}_surface.ply", v, f, COLORS[reg], reg, result["face_regions"], np.full(len(v),leaf_id))
    obj(folder/f"{key}_surface.obj", v, f, result["face_regions"])
    m.save(folder/f"{key}_model.npz")
    export_model_metadata(folder,key,m,result["u"])
    json_write(folder/f"{key}_midrib.json", dict(u=result["u"].tolist(), xyz=result["midrib"].tolist(),
        spline=spline_dict(m.center), frame_guide_spline=spline_dict(m.lateral),
        frame_rule="t=normalize(C'); b=normalize(B-(B dot t)t); n=t cross b"))
    # Effective width includes the explicitly assumed tip taper.
    u = result["u"]
    w = result["width"].copy()
    if m.extension:
        e = np.clip((u-1)/m.extension,0,1)
        w *= 1-(1-m.width_floor/m.tip_width)*(3*e**2-2*e**3)
    json_write(folder/f"{key}_width.json", dict(u=u.tolist(), width=w.tolist(),
        spline_observed_domain=spline_dict(m.width), tip_extension=m.extension,
        tip_taper="1-(1-width_floor/tip_width)*(3e^2-2e^3), e=clip((u-1)/extension,0,1)"))
    json_write(folder/f"{key}_regions.json", dict(labels=REGIONS, uv=result["uv"].tolist(),
        vertex_regions=reg.tolist(), face_regions=result["face_regions"].tolist(),
        grid_shape=[m.cfg.nu,m.cfg.nv], support_radius=result["metrics"]["support_radius"],
        area_fractions=result["metrics"]["area_fractions"],
        rules={"observed":"Inside measured parameter convex hull and within support radius of a measured sample; fitted, not raw geometry",
               "interpolated":"Inside measured convex hull, farther than support radius (including holes)",
               "extrapolated":"Outside measured convex hull or beyond observed tip; assumed geometry"},
        face_rule="maximum vertex region: extrapolated overrides interpolated overrides observed",
        caveat="Convex hull cannot resolve every true biological edge; labels describe algorithmic support"))
    json_write(folder/f"{key}_metrics.json", result["metrics"])
    png = folder/f"{key}_visualization.png"
    static_plot(png, result["points"], [result], title=f"Plant {plant.name} / leaf {leaf_id}")
    fig = go.Figure([scatter(result["points"],"Original points", "#334c42"),
                     scatter(result["midrib"],"B-spline midrib", "#175eb4",mode="lines")]+mesh_traces(result))
    layout(fig)
    metrics = result["metrics"]
    frac = metrics["area_fractions"]
    chips = "".join(f'<span class="chip">{k}: {v:.1%}</span>' for k,v in frac.items())
    body = f'<header><h1>植株 {plant.name} · 原编号 {leaf_id} 叶</h1><a href="../whole_plant.html">整株视图</a> · <a href="../../index.html">全部植株</a></header><main>'
    body += f'<div class="card"><div class="chips">{chips}<span class="chip">补全面积 {metrics["completion_ratio"]:.1%}</span><span class="chip">质量：{metrics["quality"]}</span></div>'
    body += '<p class="note">绿色 observed：附近有测量支持的拟合面；黄色 interpolated：观测范围内插值；粉色 extrapolated：超出测量范围的推断。拖动旋转、滚轮缩放，点击图例显示/隐藏图层，双击图例单独显示。点云未删改。</p>'
    body += '<p class="note">质量标记：'+html.escape(", ".join(metrics["flags"]) or "none")+'</p></div>'
    body += '<div class="card">'+fig.to_html(full_html=False,include_plotlyjs=False,config={"displaylogo":False,"responsive":True})+'</div>'
    body += f'<details class="card"><summary>静态预览及导出</summary><img src="{png.name}"><p><a href="{key}_surface.ply">PLY（含区域标签）</a> · <a href="{key}_surface.obj">OBJ</a> · <a href="{key}_metrics.json">指标 JSON</a> · <a href="{key}_regions.json">区域 JSON</a> · <a href="{key}_model.npz">连续模型</a></p></details></main>'
    page(f"{plant.name} leaf {leaf_id}", body, folder/f"{key}_visualization.html", "../../assets/plotly.min.js")


def export_plant(folder, plant, stem, results, failures):
    folder = Path(folder)
    fig = go.Figure([scatter(plant.xyz,"Original plant points", "#566c62",limit=16000),
                     scatter(stem,"Stem centerline", "#5134a1",mode="lines",width=7)])
    vertices, faces, regions, freg, ids = [], [], [], [], []
    offset = 0
    rows = []
    for leaf_id, r in results.items():
        fig.add_traces(mesh_traces(r, f"Leaf {leaf_id} · "))
        fig.add_trace(scatter(r["midrib"],f"Leaf {leaf_id} midrib", "#175eb4",mode="lines",width=3))
        vertices.append(r["vertices"])
        faces.append(r["faces"]+offset)
        regions.append(r["regions"])
        freg.append(r["face_regions"])
        ids.append(np.full(len(r["vertices"]),leaf_id))
        offset += len(r["vertices"])
        mt = r["metrics"]
        rows.append(f'<tr><td><a href="leaves/leaf_{leaf_id:02d}_visualization.html">原编号 {leaf_id} 叶</a></td><td>{mt["raw_points"]}</td><td>{mt["completion_ratio"]:.1%}</td><td>{mt["area_fractions"]["extrapolated"]:.1%}</td><td>{mt["quality"]}</td><td>{html.escape(", ".join(mt["flags"]))}</td></tr>')
    for fail in failures:
        rows.append(f'<tr class="warn"><td>原编号 {fail["leaf_id"]} 叶</td><td colspan="5">失败：{html.escape(fail["error"])}</td></tr>')
    if vertices:
        verts, fs, reg, fr = np.vstack(vertices), np.vstack(faces), np.concatenate(regions), np.concatenate(freg)
        ply(folder/"whole_plant_mesh.ply", verts, fs, COLORS[reg], reg, fr, np.concatenate(ids))
        obj(folder/"whole_plant_mesh.obj", verts, fs, fr)
    ply(folder/"stem_centerline.ply", stem)
    layout(fig)
    static_plot(folder/"whole_plant.png",plant.xyz,list(results.values()),stem,title=f"Plant {plant.name} / all modeled leaves")
    body = f'<header><h1>植株 {plant.name} · 三维叶片与茎秆</h1><a href="../index.html">全部植株</a></header><main>'
    body += f'<div class="card"><b>{len(results)} 片叶建模成功 / {len(failures)} 片失败</b><p class="note">坐标使用原始单位，物理单位暂无资料。粉色区域为外推假设。点击图例可选看单叶和观测点；茎秆中心线为紫色。整株网格由各叶片组成，茎秆中心线另存。</p></div>'
    body += '<div class="card">'+fig.to_html(full_html=False,include_plotlyjs=False,config={"displaylogo":False,"responsive":True})+'</div>'
    body += '<div class="card"><table><tr><th>叶片</th><th>原始点数</th><th>补全面积比</th><th>外推面积比</th><th>质量</th><th>标记</th></tr>'+"".join(rows)+'</table></div>'
    body += '<details class="card"><summary>整株静态图与导出</summary><img src="whole_plant.png"><p><a href="whole_plant_mesh.ply">全部叶片 PLY</a> · <a href="whole_plant_mesh.obj">全部叶片 OBJ</a> · <a href="stem_centerline.json">茎秆中心线</a></p></details></main>'
    page(f"Plant {plant.name}", body, folder/"whole_plant.html", "../assets/plotly.min.js")


def export_index(root, summary, metrics):
    rows=[]
    for p in summary["plants"]:
        name=p["plant"]
        link = f'<a href="plant_{name}/whole_plant.html">{name}</a>' if p.get("stem_success") else name
        rows.append(f'<tr><td>{link}</td><td>{p.get("leaf_count",0)}</td><td>{p.get("success",0)}</td><td>{p.get("failed",0)}</td><td>{p["status"]}</td></tr>')
    body='<header><h1>玉米叶片三维建模与缺失补全 · V1.0</h1>离线可视化 / 原始叶片编号 / 连续参数曲面</header><main>'
    body+=f'<div class="card"><div class="chips"><span class="chip">植株 {len(summary["plants"])}</span><span class="chip">成功叶片 {len(metrics)}</span><span class="chip">失败 {summary["failed_leaves"]}</span></div><p class="note">绿色：有测量支持的拟合面；黄色：孔洞等内部插值；粉色：叶缘、叶尖外推。所有曲面均为模型，外推不代表真实测量。补全比例以三角面面积加权，质量标记为算法诊断。</p></div>'
    body+='<div class="card"><table><tr><th>打开植株</th><th>原始叶片数</th><th>成功</th><th>失败</th><th>状态</th></tr>'+"".join(rows)+'</table></div>'
    usable=sum(m["quality"]=="usable_with_inference" for m in metrics)
    review=sum(m["quality"]=="review" for m in metrics)
    body+=f'<div class="card"><b>质量统计：{usable} 片可用但含推断 · {review} 片需复核</b><p class="note">建模成功与质量等级分别统计；补全比例受采样密度、支持阈值和网格影响，不是已知真值缺失比例。</p>'
    if (Path(root)/"validation/synthetic_demo/index.html").is_file():
        body+='<p><a href="validation/synthetic_demo/index.html">孔洞、叶缘和叶尖缺失补全对比（合成验证）</a></p>'
    body+='</div>'
    body+='<div class="card"><p><a href="metrics.csv">每叶指标 CSV</a> · <a href="summary.json">运行汇总</a> · <a href="input_validation.json">输入匹配报告</a> · <a href="errors.jsonl">错误日志</a></p><p class="note">usable_with_inference：未触发需复核项，仍包含模型推断；review：建议检查稀疏采样、拟合残差或较大补全范围。fit_rmse 是训练数据内的拟合诊断，不能解释为真实补全精度。实际缺失面积和隐藏叶尖长度暂无完整真值。</p></div></main>'
    page("玉米叶片建模 V1.0",body,Path(root)/"index.html")
