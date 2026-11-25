"""Interactive 3D visualizer with plant selector, single-leaf view, and pose classification mode.

Usage:
    python scripts/visualize.py                                  # Launch interactive browser UI
    python scripts/visualize.py --color-by kmeans                # Launch directly in K-means 3D pose classification mode
    python scripts/visualize.py --color-by fcm                   # Launch in Fuzzy C-Means 3D pose mode
    python scripts/visualize.py --export-viewer outputs/classification_v2/viewer
                                                                 # Export offline interactive 3D HTML viewer (all plants)
    python scripts/visualize.py --list                           # List available plants and leaf IDs
    python scripts/visualize.py --plant 1-1-2                    # Open UI pre-selected on plant 1-1-2
    python scripts/visualize.py --plant 1-1-2 --leaf 1           # Open UI pre-selected on plant 1-1-2, leaf 1
    python scripts/visualize.py --plant 1-1-2 --save out.html
    python scripts/visualize.py --plant 1-1-2 --leaf 1 --backend mpl --save leaf_01.png
    python scripts/visualize.py --model outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz
"""
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from maize_leaf.io import discover, load_pair, read_ply
from maize_leaf.surface import COLORS, REGIONS, MaizeLeafModel, leaf_mesh_grid
from maize_leaf.viz import ensure_plotly_js


DEFAULT_CLASSIFICATION_CSV = ROOT / "outputs" / "classification_v2" / "leaf_classifications.csv"

CLASS_META = {
    "convergent": dict(zh="收敛", en="convergent", color="#1f77b4", rgb=(31, 119, 180)),
    "expanded": dict(zh="开展", en="expanded", color="#1b9e77", rgb=(27, 158, 119)),
    "drooping": dict(zh="下垂", en="drooping", color="#d95f02", rgb=(217, 95, 2)),
}

COLOR_MODE_COLUMNS = {
    "kmeans": ("cluster_kmeans", "姿态分类 (K-means)"),
    "kmeans_qweighted": ("cluster_kmeans_qweighted", "姿态分类 (K-means 质量加权)"),
    "fcm": ("cluster_fcm", "姿态分类 (FCM 软聚类)"),
    "gmm": ("cluster_gmm", "姿态分类 (GMM 高斯混合)"),
}


def _parse_bool(val):
    return str(val).strip().lower() in ("1", "true", "yes")


def _parse_float(val):
    if val is None or val == "":
        return None
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except ValueError:
        return None


def load_classifications(csv_path):
    """Load Stage E leaf_classifications.csv keyed by (plant_id, leaf_id)."""
    if csv_path is None:
        return {}
    path = Path(csv_path)
    if not path.is_file():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row["plant_id"]).strip()
            lid = int(row["leaf_id"])
            out[(pid, lid)] = dict(
                plant_id=pid,
                leaf_id=lid,
                norm_base_height=_parse_float(row.get("norm_base_height")),
                quality_weight=_parse_float(row.get("quality_weight")),
                stem_leaf_angle_deg=_parse_float(row.get("stem_leaf_angle_deg")),
                tip_gain=_parse_float(row.get("tip_gain")),
                droop_ratio=_parse_float(row.get("droop_ratio")),
                midrib_curvature_total=_parse_float(row.get("midrib_curvature_total")),
                projection_ratio=_parse_float(row.get("projection_ratio")),
                normal_entropy_b24=_parse_float(row.get("normal_entropy_b24")),
                cluster_kmeans=row.get("cluster_kmeans", "").strip(),
                cluster_kmeans_lopo_oof=row.get("cluster_kmeans_lopo_oof", "").strip(),
                lopo_switched=_parse_bool(row.get("lopo_switched")),
                distance_ratio_r=_parse_float(row.get("distance_ratio_r")),
                ambiguous_r080=_parse_bool(row.get("ambiguous_r080")),
                cluster_kmeans_qweighted=row.get("cluster_kmeans_qweighted", "").strip(),
                cluster_fcm=row.get("cluster_fcm", "").strip(),
                fcm_prob_convergent=_parse_float(row.get("fcm_prob_convergent")),
                fcm_prob_expanded=_parse_float(row.get("fcm_prob_expanded")),
                fcm_prob_drooping=_parse_float(row.get("fcm_prob_drooping")),
                cluster_gmm=row.get("cluster_gmm", "").strip(),
                gmm_prob_convergent=_parse_float(row.get("gmm_prob_convergent")),
                gmm_prob_expanded=_parse_float(row.get("gmm_prob_expanded")),
                gmm_prob_drooping=_parse_float(row.get("gmm_prob_drooping")),
            )
    return out


def _class_info(cls_name):
    return CLASS_META.get(cls_name, dict(zh="未分类", en="unknown", color="#8c8c8c", rgb=(140, 140, 140)))


def _leaf_hover_text(leaf_id, cls_rec, active_mode):
    if not cls_rec:
        return f"叶片 {leaf_id}"
    col, mode_label = COLOR_MODE_COLUMNS.get(active_mode, ("cluster_kmeans", "姿态分类"))
    cls_key = cls_rec.get(col, "")
    cm = _class_info(cls_key)
    km = _class_info(cls_rec.get("cluster_kmeans", ""))["zh"]
    fcm = _class_info(cls_rec.get("cluster_fcm", ""))["zh"]
    gmm = _class_info(cls_rec.get("cluster_gmm", ""))["zh"]
    flags = []
    if cls_rec.get("ambiguous_r080"):
        flags.append("边界模糊 r>0.80")
    if cls_rec.get("cluster_kmeans") and cls_rec.get("cluster_fcm") and cls_rec["cluster_kmeans"] != cls_rec["cluster_fcm"]:
        flags.append("K-means/FCM分歧")
    if cls_rec.get("lopo_switched"):
        flags.append("LOPO切换")
    flag_line = f"<br><b>标记:</b> {' · '.join(flags)}" if flags else ""
    return (
        f"<b>叶片 {leaf_id} · {cm['zh']} ({cm['en']})</b> [{mode_label}]"
        f"<br>茎叶角 θ={cls_rec['stem_leaf_angle_deg']:.1f}° | 叶尖抬升 g={cls_rec['tip_gain']:+.2f} | 下垂比 d={cls_rec['droop_ratio']:.2f}"
        f"<br>中脉曲率 K_mid={cls_rec['midrib_curvature_total']:.2f} | 投影比 ρ={cls_rec['projection_ratio']:.2f} | 相对高度 h={cls_rec['norm_base_height']:.2f}"
        f"<br>边界比 r={cls_rec['distance_ratio_r']:.2f} | K-means={km} / FCM={fcm} / GMM={gmm}"
        f"<br>FCM隶属度: 收{cls_rec['fcm_prob_convergent']*100:.0f}% · 展{cls_rec['fcm_prob_expanded']*100:.0f}% · 垂{cls_rec['fcm_prob_drooping']*100:.0f}%"
        f"{flag_line}"
    )


def scan_catalog(root):
    """Discover all modeled plants and their available leaf IDs under output root."""
    root = Path(root)
    catalog = {}
    for plant_dir in sorted(root.glob("plant_*")):
        if not plant_dir.is_dir():
            continue
        name = plant_dir.name.removeprefix("plant_")
        leaves_dir = plant_dir / "leaves"
        lids = sorted(int(p.name.split("_")[1]) for p in leaves_dir.glob("leaf_*_surface.ply"))
        if lids:
            catalog[name] = lids
    return catalog


def sample_points(pts, rgb, limit):
    if len(pts) <= limit:
        return pts, rgb
    idx = np.linspace(0, len(pts) - 1, limit).astype(int)
    return pts[idx], (rgb[idx] if rgb is not None else None)


def load_leaf_output(leaves_dir, leaf_id):
    key = f"leaf_{leaf_id:02d}"
    pts, _ = read_ply(leaves_dir / f"{key}_points.ply")
    verts, faces = read_ply(leaves_dir / f"{key}_surface.ply")
    midrib = np.asarray(json.loads((leaves_dir / f"{key}_midrib.json").read_text(encoding="utf-8"))["xyz"])
    traits_path = leaves_dir / f"{key}_traits.json"
    traits = json.loads(traits_path.read_text(encoding="utf-8")) if traits_path.is_file() else {}
    return dict(
        leaf_id=leaf_id,
        points=pts[:, :3],
        rgb=pts[:, 3:6].astype(np.uint8) if pts.shape[1] >= 6 else None,
        vertices=verts[:, :3],
        faces=faces[:, 1:4],
        face_regions=faces[:, 4].astype(int) if faces.shape[1] > 4 else np.zeros(len(faces), dtype=int),
        midrib=midrib,
        traits=traits,
    )


def load_scene_from_outputs(root, plant=None, leaf=None):
    root = Path(root)
    catalog = scan_catalog(root)
    if plant is None:
        plant = next(iter(catalog))
    plant_dir = root / f"plant_{plant}"
    leaves_dir = plant_dir / "leaves"
    if leaf is not None:
        item = load_leaf_output(leaves_dir, leaf)
        return dict(title=f"Plant {plant} / Leaf {leaf}", plant=plant, leaf=leaf, stem=None, leaves=[item])
    stem_path = plant_dir / "stem_centerline.json"
    stem = np.asarray(json.loads(stem_path.read_text(encoding="utf-8"))["xyz"]) if stem_path.is_file() else None
    leaves = [load_leaf_output(leaves_dir, lid) for lid in catalog[plant]]
    return dict(title=f"Plant {plant} ({len(leaves)} leaves)", plant=plant, leaf=None, stem=stem, leaves=leaves)


def load_from_model(model_path, nu, nv):
    m = MaizeLeafModel.load(model_path)
    sv, faces = leaf_mesh_grid(m.blade.length, nu, nv)
    verts = m.evaluate(sv)
    freg = np.where((sv[:, 0] > m.length_obs + 1e-8)[faces].any(axis=1), 2, 0)
    return dict(
        title=Path(model_path).stem,
        plant=None,
        leaf=0,
        stem=None,
        leaves=[dict(leaf_id=0, points=None, rgb=None, vertices=verts, faces=faces, face_regions=freg,
                     midrib=m.evaluate(np.column_stack([np.linspace(0.0, m.blade.length, nu), np.zeros(nu)])),
                     traits=m.blade.to_dict())],
    )


def load_from_raw(input_root, plant=None, leaf=None):
    pairs = discover(input_root)
    name, txt, csv_path = next(p for p in pairs if plant is None or p[0] == plant)
    data = load_pair(name, txt, csv_path)
    mask = (data.labels == leaf) if leaf is not None else (data.labels >= 0)
    return dict(
        title=f"Raw {name}" + (f" / Leaf {leaf}" if leaf is not None else ""),
        plant=name,
        leaf=leaf,
        stem=None,
        leaves=[dict(leaf_id=leaf or 0, points=data.xyz[mask], rgb=data.rgb[mask], vertices=None,
                     faces=None, face_regions=None, midrib=None, traits={})],
    )


def scene_to_plotly_payload(scene, classifications=None, color_by="region"):
    """Build lightweight Plotly traces + trait/classification summary JSON for a plant or single leaf."""
    classifications = classifications or {}
    use_class = color_by in COLOR_MODE_COLUMNS and bool(classifications)
    cls_col = COLOR_MODE_COLUMNS[color_by][0] if use_class else None
    mode_label = COLOR_MODE_COLUMNS[color_by][1] if use_class else "重建区域"

    traces = []
    n_leaves = len(scene["leaves"])
    pt_limit = 9000 if n_leaves == 1 else max(1800, 15000 // max(1, n_leaves))

    if scene["stem"] is not None:
        s = np.round(scene["stem"], 5)
        traces.append(dict(
            type="scatter3d", mode="lines", name="茎秆中心线", group="stem",
            x=s[:, 0].tolist(), y=s[:, 1].tolist(), z=s[:, 2].tolist(),
            line=dict(color="#4b5563" if use_class else "#5134a1", width=7),
            hoverinfo="name",
        ))

    label_x, label_y, label_z, label_text, label_colors, label_hover = [], [], [], [], [], []
    class_counts = {"convergent": 0, "expanded": 0, "drooping": 0}
    n_ambiguous = 0
    n_disagree_km_fcm = 0
    leaf_summaries = []

    for item in scene["leaves"]:
        lid = item["leaf_id"]
        cls_rec = classifications.get((scene["plant"], lid))
        cls_key = cls_rec.get(cls_col, "") if (use_class and cls_rec) else (cls_rec.get("cluster_kmeans", "") if cls_rec else "")
        cm = _class_info(cls_key)
        is_ambig = bool(cls_rec and cls_rec.get("ambiguous_r080"))
        is_disagree = bool(
            cls_rec and cls_rec.get("cluster_kmeans") and cls_rec.get("cluster_fcm")
            and cls_rec["cluster_kmeans"] != cls_rec["cluster_fcm"]
        )
        if cls_key in class_counts:
            class_counts[cls_key] += 1
        if is_ambig:
            n_ambiguous += 1
        if is_disagree:
            n_disagree_km_fcm += 1

        prefix = f"L{lid} · " if n_leaves > 1 else ""
        if item["points"] is not None and len(item["points"]):
            pts, rgb = sample_points(item["points"], item["rgb"], pt_limit)
            pts = np.round(pts, 5)
            colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb] if rgb is not None else "#334c42"
            traces.append(dict(
                type="scatter3d", mode="markers", name=prefix + "点云", group="points", leaf_id=lid,
                x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=pts[:, 2].tolist(),
                marker=dict(size=2, color=colors, opacity=0.45 if use_class else 0.68),
                showlegend=(n_leaves == 1),
            ))

        if item["midrib"] is not None and len(item["midrib"]):
            m = np.round(item["midrib"], 5)
            midrib_color = "#f59e0b" if (use_class and (is_ambig or is_disagree)) else ("#1f2937" if use_class else "#175eb4")
            midrib_width = 6 if (use_class and (is_ambig or is_disagree)) else 4
            traces.append(dict(
                type="scatter3d", mode="lines", name=prefix + "中脉", group="midrib", leaf_id=lid,
                x=m[:, 0].tolist(), y=m[:, 1].tolist(), z=m[:, 2].tolist(),
                line=dict(color=midrib_color, width=midrib_width),
                showlegend=False,
                hovertext=_leaf_hover_text(lid, cls_rec, color_by) if cls_rec else f"叶片 {lid} 中脉",
                hoverinfo="text",
            ))
            mid_idx = int(len(m) * 0.58)
            pt_lbl = m[min(mid_idx, len(m) - 1)]
            label_x.append(float(pt_lbl[0]))
            label_y.append(float(pt_lbl[1]))
            label_z.append(float(pt_lbl[2]))
            mark = ("*" if is_ambig else "") + ("!" if is_disagree else "")
            lbl = f"L{lid} {cm['zh']}{mark}" if cls_rec else f"L{lid}"
            label_text.append(lbl)
            label_colors.append(cm["color"] if cls_rec else "#16372e")
            label_hover.append(_leaf_hover_text(lid, cls_rec, color_by) if cls_rec else f"叶片 {lid}")

        if item["vertices"] is not None and len(item["faces"]):
            v = np.round(item["vertices"], 5)
            f, fr = item["faces"], item["face_regions"]
            if use_class:
                hover_txt = _leaf_hover_text(lid, cls_rec, color_by)
                r_c, g_c, b_c = cm["rgb"]
                traces.append(dict(
                    type="mesh3d",
                    name=f"L{lid} · {cm['zh']}" + ("*" if is_ambig else ""),
                    group="surface",
                    leaf_id=lid,
                    pose_class=cls_key,
                    x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                    i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
                    color=f"rgb({r_c},{g_c},{b_c})",
                    opacity=0.80 if is_ambig else 0.90,
                    flatshading=False,
                    lighting=dict(ambient=0.58, diffuse=0.82, specular=0.18, roughness=0.55),
                    hovertext=hover_txt,
                    hoverinfo="text",
                    showlegend=True,
                ))
            else:
                for k, rname in REGIONS.items():
                    fk = f[fr == k]
                    if not len(fk):
                        continue
                    color = "rgb(" + ",".join(str(int(c)) for c in COLORS[k]) + ")"
                    traces.append(dict(
                        type="mesh3d", name=prefix + rname, group="surface", region=rname, leaf_id=lid,
                        x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                        i=fk[:, 0].tolist(), j=fk[:, 1].tolist(), k=fk[:, 2].tolist(),
                        color=color, opacity=0.88, showlegend=True,
                    ))

        leaf_summaries.append(dict(
            leaf_id=lid,
            observed_length=item["traits"].get("observed_length"),
            completed_length=item["traits"].get("completed_length"),
            max_width=item["traits"].get("max_width"),
            form_factor_alpha=item["traits"].get("form_factor_alpha"),
            surface_area=item["traits"].get("surface_area"),
            fit_rmse=item["traits"].get("fit_rmse"),
            classification=cls_rec,
            active_class=cls_key,
        ))

    if label_x:
        traces.append(dict(
            type="scatter3d",
            mode="markers+text",
            name="叶片类别标签",
            group="labels",
            x=label_x, y=label_y, z=label_z,
            text=label_text,
            textposition="top center",
            textfont=dict(size=12, color=label_colors, family="Microsoft YaHei, sans-serif"),
            marker=dict(size=4, color=label_colors, line=dict(color="#ffffff", width=1)),
            hovertext=label_hover,
            hoverinfo="text",
            showlegend=False,
        ))

    if n_leaves == 1:
        info = dict(scene["leaves"][0].get("traits", {}))
        info["classification"] = leaf_summaries[0]["classification"]
        info["active_class"] = leaf_summaries[0]["active_class"]
        info["color_by"] = color_by if use_class else "region"
    else:
        trs = [x.get("traits", {}) for x in scene["leaves"] if x.get("traits")]
        info = dict(
            plant=scene["plant"],
            leaf_count=n_leaves,
            total_area=float(sum(t.get("surface_area", 0.0) for t in trs)),
            mean_alpha=float(np.mean([t["form_factor_alpha"] for t in trs])) if trs else 0.0,
            mean_fit_rmse=float(np.mean([t["fit_rmse"] for t in trs])) if trs else 0.0,
            color_by=color_by if use_class else "region",
            has_classification=bool(classifications),
            class_counts=class_counts,
            n_ambiguous=n_ambiguous,
            n_disagree_km_fcm=n_disagree_km_fcm,
            leaves=leaf_summaries,
        )

    title_suffix = f" · {mode_label}" if use_class else ""
    return dict(
        title=scene["title"] + title_suffix,
        plant=scene["plant"],
        leaf=scene["leaf"],
        color_by=color_by if use_class else "region",
        traces=traces,
        info=info,
    )


APP_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>玉米三维叶片与姿态分类可视化浏览器</title>
<script src="/assets/plotly.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#f3f6f5;color:#16372e;font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;height:100vh}
  header{background:#113e33;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;margin-right:6px;white-space:nowrap}
  .ctrl{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
  select,button{height:31px;border-radius:6px;border:1px solid #2b6b59;padding:0 9px;font-size:13px;background:#fff;color:#16372e;cursor:pointer}
  button{background:#1c6b57;color:#fff;border-color:#2c8c74}
  button:hover{background:#23826a}
  button.active{background:#f7b93f;color:#16372e;border-color:#f7b93f;font-weight:600}
  .toggles{display:flex;align-items:center;gap:10px;margin-left:auto;font-size:13px;flex-wrap:wrap}
  .toggles label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
  .main{display:flex;flex:1;min-height:0}
  #plot{flex:1;min-width:0;background:#fff}
  aside{width:355px;background:#fff;border-left:1px solid #dce7e2;padding:14px;overflow-y:auto}
  .status{font-size:12px;color:#526a61;margin-bottom:10px}
  .kv{display:grid;grid-template-columns:1fr auto;gap:5px 10px;padding:10px;background:#f6faf8;border-radius:8px;border:1px solid #e2ece7;margin-bottom:11px;font-size:13px}
  .kv .k{color:#536a62}.kv .v{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
  .sec-title{font-weight:700;font-size:13px;margin:10px 0 6px;color:#113e33}
  .leaf-list{display:flex;flex-direction:column;gap:6px}
  .leaf-btn{display:flex;justify-content:space-between;align-items:center;width:100%;height:auto;padding:8px 10px;background:#f6faf8;color:#16372e;border:1px solid #dce7e2;border-left:5px solid #9ca3af;text-align:left}
  .leaf-btn:hover{background:#e7f3ee}
  .leaf-btn.selected{background:#113e33;color:#fff;border-color:#113e33}
  .legend{display:flex;gap:10px;margin:6px 0 10px;font-size:12px;flex-wrap:wrap}
  .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;color:#fff;margin-left:5px}
  .tag{display:inline-block;padding:0 5px;border-radius:4px;font-size:10px;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;margin-left:4px}
  .bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:6px 0 10px;background:#e5e7eb}
  .algo-tbl{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px}
  .algo-tbl th,.algo-tbl td{padding:5px 6px;border-bottom:1px solid #e2ece7;text-align:left}
  .algo-tbl th{background:#edf5f1;color:#16372e}
</style>
</head>
<body>
<header>
  <h1>玉米叶片三维可视化</h1>
  <div class="ctrl">
    <label>植株:
      <select id="plantSel"></select>
    </label>
    <button id="prevPlant" title="上一株">&lsaquo;</button>
    <button id="nextPlant" title="下一株">&rsaquo;</button>
    <button id="wholeBtn">整株视图</button>
    <label>单叶:
      <select id="leafSel"></select>
    </label>
    <button id="prevLeaf" title="上一叶">&lsaquo;</button>
    <button id="nextLeaf" title="下一叶">&rsaquo;</button>
    <label>着色:
      <select id="colorBySel">
        <option value="kmeans">姿态分类 (K-means)</option>
        <option value="fcm">姿态分类 (FCM 软聚类)</option>
        <option value="gmm">姿态分类 (GMM 高斯混合)</option>
        <option value="kmeans_qweighted">姿态分类 (K-means 加权)</option>
        <option value="region">重建区域 (观测/插值/外推)</option>
      </select>
    </label>
  </div>
  <div class="toggles">
    <label><input type="checkbox" id="chkPoints" checked>点云</label>
    <label><input type="checkbox" id="chkSurface" checked>曲面</label>
    <label><input type="checkbox" id="chkMidrib" checked>中脉</label>
    <label><input type="checkbox" id="chkStem" checked>茎秆</label>
    <label><input type="checkbox" id="chkLabels" checked>标签</label>
  </div>
</header>
<div class="main">
  <div id="plot"></div>
  <aside>
    <div class="status" id="status">正在加载目录...</div>
    <div class="legend" id="legendBox"></div>
    <div id="metrics"></div>
    <div id="leafPanel"></div>
  </aside>
</div>
<script>
const CLS = {
  convergent: { zh: '收敛', color: '#1f77b4' },
  expanded:   { zh: '开展', color: '#1b9e77' },
  drooping:   { zh: '下垂', color: '#d95f02' },
};
let catalog = {};
let plantNames = [];
let hasClassification = false;
let state = { plant: null, leaf: null, colorBy: 'kmeans', payload: null };

const plantSel = document.getElementById('plantSel');
const leafSel = document.getElementById('leafSel');
const colorBySel = document.getElementById('colorBySel');
const wholeBtn = document.getElementById('wholeBtn');
const statusEl = document.getElementById('status');
const legendBox = document.getElementById('legendBox');
const metricsEl = document.getElementById('metrics');
const leafPanelEl = document.getElementById('leafPanel');

function fmt(v, d=3) { return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }
function clsInfo(k) { return CLS[k] || { zh: '未分类', color: '#8c8c8c' }; }
function pill(k) {
  const c = clsInfo(k);
  return `<span class="pill" style="background:${c.color}">${c.zh}</span>`;
}

function populateLeaves(plant) {
  const lids = catalog[plant] || [];
  leafSel.innerHTML = '<option value="">-- 整株全部叶片 --</option>' +
    lids.map(id => `<option value="${id}">叶片 ${id}</option>`).join('');
  leafSel.value = state.leaf !== null ? String(state.leaf) : '';
  wholeBtn.classList.toggle('active', state.leaf === null);
}

function renderLegend() {
  if (state.payload && state.payload.color_by !== 'region') {
    legendBox.innerHTML =
      `<span><i class="dot" style="background:#1f77b4"></i>收敛 (convergent)</span>` +
      `<span><i class="dot" style="background:#1b9e77"></i>开展 (expanded)</span>` +
      `<span><i class="dot" style="background:#d95f02"></i>下垂 (drooping)</span>` +
      `<span style="color:#92400e"><i class="dot" style="background:#f59e0b"></i>金色中脉 = 边界模糊(*)或算法分歧(!)</span>`;
  } else {
    legendBox.innerHTML =
      `<span><i class="dot" style="background:rgb(40,158,123)"></i>observed</span>` +
      `<span><i class="dot" style="background:rgb(247,185,63)"></i>interpolated</span>` +
      `<span><i class="dot" style="background:rgb(218,79,116)"></i>extrapolated</span>`;
  }
}

function renderPlot() {
  if (!state.payload) return;
  const show = {
    points: document.getElementById('chkPoints').checked,
    surface: document.getElementById('chkSurface').checked,
    midrib: document.getElementById('chkMidrib').checked,
    stem: document.getElementById('chkStem').checked,
    labels: document.getElementById('chkLabels').checked,
  };
  const traces = state.payload.traces.filter(t => show[t.group] !== false);
  const layout = {
    title: { text: state.payload.title, x: 0.02, font: { size: 15 } },
    margin: { l: 0, r: 0, t: 36, b: 0 },
    paper_bgcolor: '#ffffff',
    scene: {
      aspectmode: 'data',
      xaxis: { title: 'X' }, yaxis: { title: 'Y' }, zaxis: { title: 'Z' },
      camera: { eye: { x: 1.45, y: 1.55, z: 0.9 } }
    },
    legend: { orientation: 'h', y: 1.0, x: 0.0 },
    uirevision: `${state.plant}_${state.leaf}`
  };
  Plotly.react('plot', traces, layout, { responsive: true, displaylogo: false });
}

function renderSidebar() {
  renderLegend();
  const info = state.payload.info || {};
  const lids = catalog[state.plant] || [];
  if (state.leaf === null) {
    const cc = info.class_counts || {};
    const n = info.leaf_count || lids.length || 1;
    const pC = ((cc.convergent || 0) / n * 100).toFixed(0);
    const pE = ((cc.expanded || 0) / n * 100).toFixed(0);
    const pD = ((cc.drooping || 0) / n * 100).toFixed(0);
    const clsBlock = info.has_classification ? `
      <div class="bar" title="收敛 ${pC}% / 开展 ${pE}% / 下垂 ${pD}%">
        <div style="width:${pC}%;background:#1f77b4"></div>
        <div style="width:${pE}%;background:#1b9e77"></div>
        <div style="width:${pD}%;background:#d95f02"></div>
      </div>
      <div class="kv">
        <span class="k">收敛 / 开展 / 下垂</span>
        <span class="v">${cc.convergent||0} (${pC}%) / ${cc.expanded||0} (${pE}%) / ${cc.drooping||0} (${pD}%)</span>
        <span class="k">边界模糊叶 (r&gt;0.80)</span><span class="v">${info.n_ambiguous ?? 0} / ${n}</span>
        <span class="k">K-means 与 FCM 分歧</span><span class="v">${info.n_disagree_km_fcm ?? 0} / ${n}</span>
      </div>` : '';
    metricsEl.innerHTML = `<div class="kv">
      <span class="k">当前植株</span><span class="v">${state.plant}</span>
      <span class="k">叶片数量</span><span class="v">${info.leaf_count ?? lids.length}</span>
      <span class="k">总叶面积</span><span class="v">${fmt(info.total_area, 4)}</span>
      <span class="k">平均拟合 RMSE</span><span class="v">${fmt(info.mean_fit_rmse, 4)}</span>
    </div>` + clsBlock;

    const rows = (info.leaves || []).map(l => {
      const c = l.classification;
      const ac = clsInfo(l.active_class);
      const border = c ? ac.color : '#9ca3af';
      const tags = (c && c.ambiguous_r080 ? '<span class="tag">模糊</span>' : '') +
        (c && c.cluster_kmeans && c.cluster_fcm && c.cluster_kmeans !== c.cluster_fcm ? '<span class="tag">分歧</span>' : '');
      const rightTxt = c
        ? `θ=${fmt(c.stem_leaf_angle_deg,1)}° · h=${fmt(c.norm_base_height,2)}`
        : `L=${fmt(l.completed_length,2)} · W=${fmt(l.max_width,3)}`;
      return `<button class="leaf-btn" style="border-left-color:${border}" onclick="selectView('${state.plant}', ${l.leaf_id})">
        <span><b>L${l.leaf_id}</b>${c ? pill(l.active_class) : ''}${tags}</span>
        <span style="font-size:12px;color:#4b635a">${rightTxt}</span>
      </button>`;
    }).join('');
    leafPanelEl.innerHTML = '<div class="sec-title">点击叶片进入单叶检查：</div><div class="leaf-list">' + rows + '</div>';
  } else {
    const c = info.classification;
    const frac = info.area_fractions || {};
    const clsHtml = c ? `
      <div class="sec-title">姿态分类与多算法对照 ${pill(info.active_class)}</div>
      <table class="algo-tbl">
        <tr><th>算法</th><th>类别</th><th>说明 / 概率</th></tr>
        <tr><td>K-means (主)</td><td>${pill(c.cluster_kmeans)}</td><td>r=${fmt(c.distance_ratio_r,2)}${c.ambiguous_r080?' · <b>模糊</b>':''}${c.lopo_switched?' · LOPO切换':''}</td></tr>
        <tr><td>K-means 加权</td><td>${pill(c.cluster_kmeans_qweighted)}</td><td>质量权重 q=${fmt(c.quality_weight,2)}</td></tr>
        <tr><td>FCM (m=2)</td><td>${pill(c.cluster_fcm)}</td><td>收${(c.fcm_prob_convergent*100).toFixed(0)}% 展${(c.fcm_prob_expanded*100).toFixed(0)}% 垂${(c.fcm_prob_drooping*100).toFixed(0)}%</td></tr>
        <tr><td>GMM</td><td>${pill(c.cluster_gmm)}</td><td>收${(c.gmm_prob_convergent*100).toFixed(0)}% 展${(c.gmm_prob_expanded*100).toFixed(0)}% 垂${(c.gmm_prob_drooping*100).toFixed(0)}%</td></tr>
      </table>
      <div class="kv">
        <span class="k">茎叶角 θ (15%基部叶面)</span><span class="v">${fmt(c.stem_leaf_angle_deg, 1)}°</span>
        <span class="k">叶尖抬升 g</span><span class="v">${fmt(c.tip_gain, 3)}</span>
        <span class="k">下垂比 d</span><span class="v">${fmt(c.droop_ratio, 3)}</span>
        <span class="k">中脉总曲率 K_mid</span><span class="v">${fmt(c.midrib_curvature_total, 2)}</span>
        <span class="k">投影比 ρ</span><span class="v">${fmt(c.projection_ratio, 3)}</span>
        <span class="k">相对叶位高度 h (未入聚类)</span><span class="v">${fmt(c.norm_base_height, 3)}</span>
      </div>` : '';
    metricsEl.innerHTML = clsHtml + `
      <div class="sec-title">三维重建指标</div>
      <div class="kv">
        <span class="k">植株 / 叶片</span><span class="v">${state.plant} · 叶片 ${state.leaf}</span>
        <span class="k">观测 / 补全叶长</span><span class="v">${fmt(info.observed_length, 3)} / ${fmt(info.completed_length, 3)}</span>
        <span class="k">最大叶宽 W_max</span><span class="v">${fmt(info.max_width, 4)}</span>
        <span class="k">总叶面积</span><span class="v">${fmt(info.surface_area, 4)}</span>
        <span class="k">拟合 RMSE</span><span class="v">${fmt(info.fit_rmse, 4)}</span>
        <span class="k">obs / interp / extrap</span>
        <span class="v">${((frac.observed||0)*100).toFixed(1)}% / ${((frac.interpolated||0)*100).toFixed(1)}% / ${((frac.extrapolated||0)*100).toFixed(1)}%</span>
      </div>`;
    const btns = lids.map(id =>
      `<button class="leaf-btn ${id === state.leaf ? 'selected' : ''}" onclick="selectView('${state.plant}', ${id})">
        <span>叶片 ${id}</span><span>${id === state.leaf ? '当前' : '切换'}</span>
      </button>`
    ).join('');
    leafPanelEl.innerHTML = `<div style="margin-bottom:8px"><button style="width:100%" onclick="selectView('${state.plant}', null)">返回整株视图</button></div>
      <div class="leaf-list">${btns}</div>`;
  }
}

async function selectView(plant, leaf, colorBy=state.colorBy) {
  state.plant = plant;
  state.leaf = leaf;
  state.colorBy = colorBy;
  plantSel.value = plant;
  colorBySel.value = colorBy;
  populateLeaves(plant);
  statusEl.textContent = leaf === null ? `正在加载植株 ${plant} 整株数据...` : `正在加载植株 ${plant} 叶片 ${leaf}...`;
  const params = new URLSearchParams({ plant, color_by: colorBy });
  if (leaf !== null) params.set('leaf', String(leaf));
  const resp = await fetch(`/api/scene?${params.toString()}`);
  state.payload = await resp.json();
  statusEl.textContent = `已加载：${state.payload.title}（拖动旋转，滚轮缩放，悬停看单叶特征）`;
  renderPlot();
  renderSidebar();
}

plantSel.onchange = () => selectView(plantSel.value, null);
leafSel.onchange = () => selectView(state.plant, leafSel.value ? Number(leafSel.value) : null);
colorBySel.onchange = () => selectView(state.plant, state.leaf, colorBySel.value);
wholeBtn.onclick = () => selectView(state.plant, null);
document.getElementById('prevPlant').onclick = () => {
  const i = (plantNames.indexOf(state.plant) - 1 + plantNames.length) % plantNames.length;
  selectView(plantNames[i], null);
};
document.getElementById('nextPlant').onclick = () => {
  const i = (plantNames.indexOf(state.plant) + 1) % plantNames.length;
  selectView(plantNames[i], null);
};
document.getElementById('prevLeaf').onclick = () => {
  const lids = catalog[state.plant] || [];
  if (!lids.length) return;
  const idx = state.leaf === null ? lids.length - 1 : (lids.indexOf(state.leaf) - 1 + lids.length) % lids.length;
  selectView(state.plant, lids[idx]);
};
document.getElementById('nextLeaf').onclick = () => {
  const lids = catalog[state.plant] || [];
  if (!lids.length) return;
  const idx = state.leaf === null ? 0 : (lids.indexOf(state.leaf) + 1) % lids.length;
  selectView(state.plant, lids[idx]);
};
['chkPoints','chkSurface','chkMidrib','chkStem','chkLabels'].forEach(id => {
  document.getElementById(id).onchange = renderPlot;
});

async function boot() {
  const resp = await fetch('/api/catalog');
  const data = await resp.json();
  catalog = data.catalog;
  hasClassification = !!data.has_classification;
  plantNames = Object.keys(catalog);
  plantSel.innerHTML = plantNames.map(p => `<option value="${p}">${p} (${catalog[p].length}叶)</option>`).join('');
  if (!hasClassification) {
    colorBySel.value = 'region';
    Array.from(colorBySel.options).forEach(opt => { if (opt.value !== 'region') opt.disabled = true; });
  }
  const initPlant = data.initial_plant && catalog[data.initial_plant] ? data.initial_plant : plantNames[0];
  const initLeaf = data.initial_leaf !== null && (catalog[initPlant] || []).includes(data.initial_leaf) ? data.initial_leaf : null;
  const initColor = hasClassification ? (data.initial_color_by || 'kmeans') : 'region';
  await selectView(initPlant, initLeaf, initColor);
}
boot();
</script>
</body>
</html>
"""


def make_handler(root, catalog, init_plant, init_leaf, classifications, init_color_by):
    plotly_js = Path(root) / "assets" / "plotly.min.js"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, code, content_type, body):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", APP_HTML)
            elif parsed.path == "/assets/plotly.min.js":
                self._send(200, "application/javascript; charset=utf-8", plotly_js.read_bytes())
            elif parsed.path == "/api/catalog":
                body = json.dumps(
                    dict(
                        catalog=catalog,
                        initial_plant=init_plant,
                        initial_leaf=init_leaf,
                        initial_color_by=init_color_by,
                        has_classification=bool(classifications),
                    ),
                    ensure_ascii=False,
                )
                self._send(200, "application/json; charset=utf-8", body)
            elif parsed.path == "/api/scene":
                qs = parse_qs(parsed.query)
                plant = qs.get("plant", [next(iter(catalog))])[0]
                leaf = int(qs["leaf"][0]) if "leaf" in qs else None
                color_by = qs.get("color_by", [init_color_by])[0]
                scene = load_scene_from_outputs(root, plant, leaf)
                payload = scene_to_plotly_payload(scene, classifications=classifications, color_by=color_by)
                self._send(200, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False))
            else:
                self._send(404, "text/plain; charset=utf-8", "Not found")

    return Handler


def run_server(
    root,
    init_plant=None,
    init_leaf=None,
    host="127.0.0.1",
    port=8050,
    open_browser=True,
    classifications=None,
    init_color_by="region",
):
    ensure_plotly_js(root)
    catalog = scan_catalog(root)
    server = ThreadingHTTPServer(
        (host, port),
        make_handler(root, catalog, init_plant, init_leaf, classifications or {}, init_color_by),
    )
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(
        f"Visualizer running at {url} ({len(catalog)} plants, {sum(len(v) for v in catalog.values())} leaves, "
        f"mode={init_color_by}). Press Ctrl+C to stop.",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


OFFLINE_PLANT_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>植株 __PLANT__ · 叶片姿态三维分类交互视图</title>
<script src="assets/plotly.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#f3f6f5;color:#16372e;font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;height:100vh}
  header{background:#113e33;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;white-space:nowrap}
  header a{color:#d6f5ec;text-decoration:none;font-size:13px;padding:4px 9px;border:1px solid #2b6b59;border-radius:6px}
  header a:hover{background:#1c6b57}
  select,button{height:30px;border-radius:6px;border:1px solid #2b6b59;padding:0 9px;font-size:13px;background:#fff;color:#16372e;cursor:pointer}
  .toggles{display:flex;align-items:center;gap:10px;margin-left:auto;font-size:13px;flex-wrap:wrap}
  .toggles label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
  .main{display:flex;flex:1;min-height:0}
  #plot{flex:1;min-width:0;background:#fff}
  aside{width:355px;background:#fff;border-left:1px solid #dce7e2;padding:14px;overflow-y:auto}
  .legend{display:flex;gap:10px;margin:0 0 10px;font-size:12px;flex-wrap:wrap}
  .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  .kv{display:grid;grid-template-columns:1fr auto;gap:5px 10px;padding:10px;background:#f6faf8;border-radius:8px;border:1px solid #e2ece7;margin-bottom:10px;font-size:13px}
  .kv .k{color:#536a62}.kv .v{font-weight:600;text-align:right}
  .bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:6px 0 10px;background:#e5e7eb}
  .leaf-list{display:flex;flex-direction:column;gap:6px}
  .leaf-btn{display:flex;justify-content:space-between;align-items:center;width:100%;height:auto;padding:8px 10px;background:#f6faf8;color:#16372e;border:1px solid #dce7e2;border-left:5px solid #9ca3af;text-align:left}
  .leaf-btn:hover{background:#e7f3ee}
  .leaf-btn.selected{background:#113e33;color:#fff;border-color:#113e33}
  .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;color:#fff;margin-left:5px}
  .tag{display:inline-block;padding:0 5px;border-radius:4px;font-size:10px;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;margin-left:4px}
</style>
</head>
<body>
<header>
  <h1>植株 __PLANT__ · 姿态分类 3D</h1>
  <a href="index.html">全部 14 株总览</a>
  <a href="__PREV_HREF__">&lsaquo; 上一株</a>
  <a href="__NEXT_HREF__">下一株 &rsaquo;</a>
  <label>算法配色:
    <select id="algoSel">
      <option value="kmeans">K-means (主模型 M4)</option>
      <option value="fcm">FCM 软聚类 (m=2.0)</option>
      <option value="gmm">GMM 高斯混合</option>
      <option value="kmeans_qweighted">K-means 质量加权</option>
    </select>
  </label>
  <button id="resetLeafBtn">显示全部叶片</button>
  <div class="toggles">
    <label><input type="checkbox" id="chkPoints" checked>点云</label>
    <label><input type="checkbox" id="chkSurface" checked>曲面</label>
    <label><input type="checkbox" id="chkMidrib" checked>中脉</label>
    <label><input type="checkbox" id="chkStem" checked>茎秆</label>
    <label><input type="checkbox" id="chkLabels" checked>标签</label>
  </div>
</header>
<div class="main">
  <div id="plot"></div>
  <aside>
    <div class="legend">
      <span><i class="dot" style="background:#1f77b4"></i>收敛</span>
      <span><i class="dot" style="background:#1b9e77"></i>开展</span>
      <span><i class="dot" style="background:#d95f02"></i>下垂</span>
      <span style="color:#92400e"><i class="dot" style="background:#f59e0b"></i>金色中脉 = 模糊(*) / 分歧(!)</span>
    </div>
    <div id="summary"></div>
    <div id="leafList" class="leaf-list"></div>
  </aside>
</div>
<script>
const DATA = __PAYLOAD_JSON__;
const CLS = {
  convergent: { zh: '收敛', rgb: 'rgb(31,119,180)', hex: '#1f77b4' },
  expanded:   { zh: '开展', rgb: 'rgb(27,158,119)', hex: '#1b9e77' },
  drooping:   { zh: '下垂', rgb: 'rgb(217,95,2)',   hex: '#d95f02' },
};
const ALGO_COL = {
  kmeans: 'cluster_kmeans',
  fcm: 'cluster_fcm',
  gmm: 'cluster_gmm',
  kmeans_qweighted: 'cluster_kmeans_qweighted',
};
let activeAlgo = '__DEFAULT_ALGO__';
let focusLeaf = null;
const leafMap = {};
(DATA.info.leaves || []).forEach(l => { leafMap[l.leaf_id] = l; });

function fmt(v, d=2) { return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }
function clsInfo(k) { return CLS[k] || { zh: '未分类', rgb: 'rgb(140,140,140)', hex: '#8c8c8c' }; }

function render() {
  const col = ALGO_COL[activeAlgo];
  const show = {
    points: document.getElementById('chkPoints').checked,
    surface: document.getElementById('chkSurface').checked,
    midrib: document.getElementById('chkMidrib').checked,
    stem: document.getElementById('chkStem').checked,
    labels: document.getElementById('chkLabels').checked,
  };
  const traces = [];
  for (const raw of DATA.traces) {
    if (show[raw.group] === false) continue;
    if (focusLeaf !== null && raw.leaf_id !== undefined && raw.leaf_id !== focusLeaf && raw.group !== 'stem') continue;
    const t = Object.assign({}, raw);
    if (t.group === 'surface' && t.leaf_id !== undefined) {
      const rec = (leafMap[t.leaf_id] || {}).classification;
      if (rec) {
        const ck = rec[col];
        const ci = clsInfo(ck);
        t.color = ci.rgb;
        t.name = `L${t.leaf_id} · ${ci.zh}` + (rec.ambiguous_r080 ? '*' : '');
      }
    } else if (t.group === 'labels') {
      const xs = [], ys = [], zs = [], txts = [], cols = [], hov = [];
      (DATA.info.leaves || []).forEach((l, idx) => {
        if (focusLeaf !== null && l.leaf_id !== focusLeaf) return;
        const rec = l.classification;
        const ck = rec ? rec[col] : '';
        const ci = clsInfo(ck);
        const mk = (rec && rec.ambiguous_r080 ? '*' : '') +
          (rec && rec.cluster_kmeans !== rec.cluster_fcm ? '!' : '');
        xs.push(raw.x[idx]); ys.push(raw.y[idx]); zs.push(raw.z[idx]);
        txts.push(`L${l.leaf_id} ${ci.zh}${mk}`);
        cols.push(ci.hex);
        hov.push(raw.hovertext[idx]);
      });
      t.x = xs; t.y = ys; t.z = zs; t.text = txts;
      t.textfont = Object.assign({}, raw.textfont, { color: cols });
      t.marker = Object.assign({}, raw.marker, { color: cols });
      t.hovertext = hov;
    }
    traces.push(t);
  }
  const titleTxt = `植株 __PLANT__ · ${document.getElementById('algoSel').selectedOptions[0].text}` +
    (focusLeaf !== null ? ` · 叶片 ${focusLeaf}` : '');
  Plotly.react('plot', traces, {
    title: { text: titleTxt, x: 0.02, font: { size: 15 } },
    margin: { l: 0, r: 0, t: 36, b: 0 },
    paper_bgcolor: '#ffffff',
    scene: { aspectmode: 'data', xaxis: { title: 'X' }, yaxis: { title: 'Y' }, zaxis: { title: 'Z' },
             camera: { eye: { x: 1.45, y: 1.55, z: 0.9 } } },
    legend: { orientation: 'h', y: 1.0, x: 0.0 },
    uirevision: 'keep-camera',
  }, { responsive: true, displaylogo: false });

  const leaves = DATA.info.leaves || [];
  const cnt = { convergent: 0, expanded: 0, drooping: 0 };
  let amb = 0, dis = 0;
  leaves.forEach(l => {
    const r = l.classification;
    if (!r) return;
    if (cnt[r[col]] !== undefined) cnt[r[col]] += 1;
    if (r.ambiguous_r080) amb += 1;
    if (r.cluster_kmeans !== r.cluster_fcm) dis += 1;
  });
  const n = Math.max(1, leaves.length);
  const pC = (cnt.convergent / n * 100).toFixed(0);
  const pE = (cnt.expanded / n * 100).toFixed(0);
  const pD = (cnt.drooping / n * 100).toFixed(0);
  document.getElementById('summary').innerHTML = `
    <div class="bar">
      <div style="width:${pC}%;background:#1f77b4"></div>
      <div style="width:${pE}%;background:#1b9e77"></div>
      <div style="width:${pD}%;background:#d95f02"></div>
    </div>
    <div class="kv">
      <span class="k">收敛 / 开展 / 下垂</span><span class="v">${cnt.convergent} / ${cnt.expanded} / ${cnt.drooping}</span>
      <span class="k">边界模糊叶 (r&gt;0.80)</span><span class="v">${amb} / ${leaves.length}</span>
      <span class="k">K-means / FCM 分歧</span><span class="v">${dis} / ${leaves.length}</span>
    </div>`;

  document.getElementById('leafList').innerHTML = leaves.map(l => {
    const r = l.classification || {};
    const ci = clsInfo(r[col]);
    const sel = focusLeaf === l.leaf_id ? ' selected' : '';
    const tags = (r.ambiguous_r080 ? '<span class="tag">模糊</span>' : '') +
      (r.cluster_kmeans && r.cluster_fcm && r.cluster_kmeans !== r.cluster_fcm ? '<span class="tag">分歧</span>' : '');
    return `<button class="leaf-btn${sel}" style="border-left-color:${ci.hex}" onclick="toggleLeaf(${l.leaf_id})">
      <span><b>L${l.leaf_id}</b><span class="pill" style="background:${ci.hex}">${ci.zh}</span>${tags}</span>
      <span style="font-size:12px">θ=${fmt(r.stem_leaf_angle_deg,1)}° · g=${fmt(r.tip_gain,2)} · h=${fmt(r.norm_base_height,2)}</span>
    </button>`;
  }).join('');
}

function toggleLeaf(lid) {
  focusLeaf = (focusLeaf === lid) ? null : lid;
  render();
}

document.getElementById('algoSel').value = activeAlgo;
document.getElementById('algoSel').onchange = e => { activeAlgo = e.target.value; render(); };
document.getElementById('resetLeafBtn').onclick = () => { focusLeaf = null; render(); };
['chkPoints','chkSurface','chkMidrib','chkStem','chkLabels'].forEach(id => {
  document.getElementById(id).onchange = render;
});
render();
</script>
</body>
</html>
"""


def export_classification_viewer(root, classification_csv, out_dir, default_color_by="kmeans"):
    """Export standalone offline interactive 3D classification HTML viewer (index + 14 plant pages)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(out_dir)
    classifications = load_classifications(classification_csv)
    if not classifications:
        raise FileNotFoundError(f"Classification CSV not found or empty: {classification_csv}")
    catalog = scan_catalog(root)
    plants = list(catalog.keys())
    if default_color_by not in COLOR_MODE_COLUMNS:
        default_color_by = "kmeans"

    index_rows = []
    tot_counts = {"convergent": 0, "expanded": 0, "drooping": 0}
    tot_leaves = 0
    tot_ambig = 0
    tot_disagree = 0

    for idx, plant in enumerate(plants):
        scene = load_scene_from_outputs(root, plant, leaf=None)
        payload = scene_to_plotly_payload(scene, classifications=classifications, color_by=default_color_by)
        prev_p = plants[(idx - 1) % len(plants)]
        next_p = plants[(idx + 1) % len(plants)]
        html = (
            OFFLINE_PLANT_TEMPLATE
            .replace("__PLANT__", plant)
            .replace("__PREV_HREF__", f"plant_{prev_p}.html")
            .replace("__NEXT_HREF__", f"plant_{next_p}.html")
            .replace("__DEFAULT_ALGO__", default_color_by)
            .replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
        )
        (out_dir / f"plant_{plant}.html").write_text(html, encoding="utf-8")

        info = payload["info"]
        cc = info["class_counts"]
        n = info["leaf_count"]
        for k in tot_counts:
            tot_counts[k] += cc.get(k, 0)
        tot_leaves += n
        tot_ambig += info["n_ambiguous"]
        tot_disagree += info["n_disagree_km_fcm"]
        index_rows.append(
            f'<tr><td><a href="plant_{plant}.html"><b>{plant}</b></a></td>'
            f'<td>{n}</td>'
            f'<td style="color:#1f77b4;font-weight:600">{cc["convergent"]} ({cc["convergent"]/n*100:.0f}%)</td>'
            f'<td style="color:#1b9e77;font-weight:600">{cc["expanded"]} ({cc["expanded"]/n*100:.0f}%)</td>'
            f'<td style="color:#d95f02;font-weight:600">{cc["drooping"]} ({cc["drooping"]/n*100:.0f}%)</td>'
            f'<td>{info["n_ambiguous"]}</td><td>{info["n_disagree_km_fcm"]}</td></tr>'
        )

    index_html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>玉米叶片姿态三维分类交互总览</title>
<style>
  body{{margin:0;background:#f3f6f5;color:#16372e;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif}}
  header{{padding:20px 28px;background:#113e33;color:#fff}}h1{{font-size:21px;margin:0 0 6px}}
  main{{max-width:1280px;margin:auto;padding:18px 24px}}a{{color:#16715b;text-decoration:none}}
  a:hover{{text-decoration:underline}}.card{{background:#fff;border-radius:12px;padding:18px;margin-bottom:16px;border:1px solid #dce7e2}}
  .chips{{display:flex;gap:12px;flex-wrap:wrap}}.chip{{background:#ecf3f0;padding:8px 14px;border-radius:8px;font-weight:600}}
  table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce7e2}}
  th{{background:#e8f1ec}}img{{max-width:100%;border-radius:8px}}
</style>
<body>
<header>
  <h1>玉米叶片姿态三分类 · 三维可交互总览（Stage E）</h1>
  <div>基于纯形态特征 M4 = (θ, g, d, K_mid, ρ)，每株 3D 页面支持实时切换 K-means / FCM / GMM / 质量加权着色、单叶聚焦与悬停指标查看</div>
</header>
<main>
  <div class="card"><div class="chips">
    <span class="chip">植株：{len(plants)}</span>
    <span class="chip">叶片：{tot_leaves}</span>
    <span class="chip" style="color:#1f77b4">收敛：{tot_counts['convergent']} ({tot_counts['convergent']/max(1,tot_leaves)*100:.1f}%)</span>
    <span class="chip" style="color:#1b9e77">开展：{tot_counts['expanded']} ({tot_counts['expanded']/max(1,tot_leaves)*100:.1f}%)</span>
    <span class="chip" style="color:#d95f02">下垂：{tot_counts['drooping']} ({tot_counts['drooping']/max(1,tot_leaves)*100:.1f}%)</span>
    <span class="chip">边界模糊 (r&gt;0.80)：{tot_ambig}</span>
    <span class="chip">K-means/FCM 分歧：{tot_disagree}</span>
  </div></div>
  <div class="card">
    <table>
      <tr><th>植株（点击打开 3D 交互）</th><th>叶片数</th><th>收敛 (蓝)</th><th>开展 (绿)</th><th>下垂 (橙)</th><th>边界模糊叶</th><th>K-means/FCM 分歧</th></tr>
      {"".join(index_rows)}
    </table>
  </div>
  <details class="card" open><summary><b>Stage E 统计分析四联图</b></summary>
    <p><img src="../classification_analysis.png" alt="Stage E Classification Analysis"></p>
  </details>
</main>
</body></html>"""
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    print(f"Exported offline 3D classification viewer ({len(plants)} plants) -> {index_path.resolve()}", flush=True)
    return index_path


def show_plotly_static(scene, args, classifications=None):
    import plotly.graph_objects as go

    payload = scene_to_plotly_payload(scene, classifications=classifications, color_by=args.color_by)
    traces = []
    for t in payload["traces"]:
        t = dict(t)
        grp = t.pop("group", None)
        t.pop("region", None)
        t.pop("leaf_id", None)
        t.pop("pose_class", None)
        if (grp == "points" and args.no_points) or (grp == "midrib" and args.no_midrib) or (grp == "surface" and args.no_surface):
            continue
        typ = t.pop("type")
        traces.append(go.Mesh3d(**t) if typ == "mesh3d" else go.Scatter3d(**t))
    fig = go.Figure(traces)
    fig.update_layout(
        title=payload["title"], height=760, margin=dict(l=0, r=0, t=40, b=0),
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


def show_mpl(scene, args, classifications=None):
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    classifications = classifications or {}
    use_class = args.color_by in COLOR_MODE_COLUMNS and bool(classifications)
    cls_col = COLOR_MODE_COLUMNS[args.color_by][0] if use_class else None

    fig = plt.figure(figsize=(10.5, 8), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7faf8")
    all_xyz = []

    if scene["stem"] is not None:
        ax.plot(*scene["stem"].T, color="#4b5563" if use_class else "#5134a1", linewidth=3)
        all_xyz.append(scene["stem"])

    for item in scene["leaves"]:
        cls_rec = classifications.get((scene["plant"], item["leaf_id"]))
        cls_key = cls_rec.get(cls_col, "") if (use_class and cls_rec) else ""
        cm = _class_info(cls_key)
        if not args.no_points and item["points"] is not None:
            pts, rgb = sample_points(item["points"], item["rgb"], 7000)
            c = rgb / 255.0 if rgb is not None else "#293e36"
            ax.scatter(*pts.T, s=1.0, c=c, alpha=0.25 if use_class else 0.4)
            all_xyz.append(pts)
        if not args.no_surface and item["vertices"] is not None:
            v = item["vertices"]
            fc = (np.array(cm["rgb"]) / 255.0) if use_class else (COLORS[item["face_regions"]] / 255.0)
            coll = Poly3DCollection(v[item["faces"]], facecolors=fc, edgecolor="none", alpha=0.88, zsort="average")
            ax.add_collection3d(coll)
            all_xyz.append(v)
        if not args.no_midrib and item["midrib"] is not None:
            ax.plot(*item["midrib"].T, color="#1f2937" if use_class else "#175eb4", linewidth=1.6)
            all_xyz.append(item["midrib"])

    xyz = np.vstack(all_xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), xlabel="X", ylabel="Y", zlabel="Z", title=scene["title"])
    ax.set_box_aspect(np.maximum(hi - lo, 0.01))
    ax.view_init(elev=args.elev, azim=args.azim)
    if not args.no_surface and any(item["vertices"] is not None for item in scene["leaves"]):
        if use_class:
            handles = [Patch(color=np.array(m["rgb"]) / 255.0, label=f"{m['zh']} ({k})") for k, m in CLASS_META.items()]
        else:
            handles = [Patch(color=COLORS[k] / 255.0, label=REGIONS[k]) for k in REGIONS]
        ax.legend(handles=handles, loc="upper left")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Interactive 3D visualizer with plant, single-leaf, and pose classification views")
    p.add_argument("--root", type=Path, default=ROOT / "outputs/maize_v2", help="Modeling output directory")
    p.add_argument(
        "--classification-csv",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CSV,
        help="Stage E leaf_classifications.csv path (auto-loaded if present)",
    )
    p.add_argument(
        "--color-by",
        choices=("region", "kmeans", "kmeans_qweighted", "fcm", "gmm"),
        default="kmeans",
        help="Initial 3D leaf coloring mode (falls back to region if classification CSV is absent)",
    )
    p.add_argument(
        "--export-viewer",
        type=Path,
        help="Export standalone offline interactive 3D HTML viewer directory (index.html + per-plant HTMLs)",
    )
    p.add_argument("--list", action="store_true", help="List all available plants and leaf IDs")
    p.add_argument("--input-root", type=Path, help="Raw TXT/CSV input directory")
    p.add_argument("--model", type=Path, help="Single leaf *_model.npz file")
    p.add_argument("--plant", help="Initial or target plant name, e.g. 1-1-2")
    p.add_argument("--leaf", type=int, help="Initial or target leaf ID")
    p.add_argument("--backend", choices=("web", "plotly", "mpl"), default="web")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--save", type=Path, help="Save single view to HTML (plotly) or PNG (mpl)")
    p.add_argument("--nu", type=int, default=90)
    p.add_argument("--nv", type=int, default=25)
    p.add_argument("--elev", type=float, default=22)
    p.add_argument("--azim", type=float, default=42)
    p.add_argument("--no-points", action="store_true")
    p.add_argument("--no-midrib", action="store_true")
    p.add_argument("--no-surface", action="store_true")
    args = p.parse_args()

    classifications = load_classifications(args.classification_csv)
    if args.color_by != "region" and not classifications:
        args.color_by = "region"

    if args.list:
        catalog = scan_catalog(args.root)
        for plant, lids in catalog.items():
            print(f"{plant}: leaves {lids}")
        print(f"Total: {len(catalog)} plants, {sum(len(v) for v in catalog.values())} leaves")
        return

    if args.export_viewer:
        export_classification_viewer(args.root, args.classification_csv, args.export_viewer, default_color_by=args.color_by)
        return

    if args.model:
        scene = load_from_model(args.model, args.nu, args.nv)
        show_mpl(scene, args) if args.backend == "mpl" else show_plotly_static(scene, args)
        return

    if args.input_root:
        scene = load_from_raw(args.input_root, args.plant, args.leaf)
        show_mpl(scene, args) if args.backend == "mpl" else show_plotly_static(scene, args)
        return

    if args.backend == "web" and not args.save:
        run_server(
            args.root,
            args.plant,
            args.leaf,
            port=args.port,
            open_browser=not args.no_browser,
            classifications=classifications,
            init_color_by=args.color_by,
        )
        return

    scene = load_scene_from_outputs(args.root, args.plant, args.leaf)
    if args.backend == "mpl" or (args.save and args.save.suffix.lower() in (".png", ".jpg", ".svg", ".pdf")):
        show_mpl(scene, args, classifications=classifications)
    else:
        show_plotly_static(scene, args, classifications=classifications)


if __name__ == "__main__":
    main()
