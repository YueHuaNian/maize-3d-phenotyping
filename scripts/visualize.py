"""阶段 4（主入口）：玉米植株与单叶三维交互查看器（支持在线服务、离线导出与静态出图）。

功能说明：
    在线模式与离线模式共用 src/maize_leaf/viewer.html 模板，支持：
    - 自由切换全部植株、整株/单叶视角及“紧缩/舒展/下垂”四因素姿态着色；
    - 默认仅展示观测支撑面，可按需勾选查看插值/外推面；
    - 点选叶片时保留三维网格缓冲与当前视角，避免重绘卡顿。

常用命令：
    python scripts/visualize.py                                  # 启动本地浏览器交互查看器
    python scripts/visualize.py --color-by kmeans                # 默认按姿态分类着色启动
    python scripts/visualize.py --export-viewer outputs/classification_v2/viewer
                                                                 # 导出可脱离服务器打开的离线 HTML 查看器
    python scripts/visualize.py --list                           # 列出全部已重建植株与叶片编号
    python scripts/visualize.py --plant 1-1-2 --leaf 1           # 启动并预选指定植株与叶片
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
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from maize_leaf.io import discover, load_pair, read_ply
from maize_leaf.surface import COLORS, REGIONS, MaizeLeafModel, leaf_mesh_grid
from maize_leaf.viz import ensure_plotly_js


DEFAULT_CLASSIFICATION_CSV = ROOT / "outputs" / "classification_v2" / "leaf_classifications.csv"

CLASS_META = {
    "convergent": dict(zh="紧缩", en="convergent", color="#1f77b4", rgb=(31, 119, 180)),
    "expanded": dict(zh="舒展", en="expanded", color="#1b9e77", rgb=(27, 158, 119)),
    "drooping": dict(zh="下垂", en="drooping", color="#d95f02", rgb=(217, 95, 2)),
}

COLOR_MODE_COLUMNS = {
    "kmeans": ("cluster_kmeans", "四因素姿态分类 (K-means)"),
}


def _parse_bool(val):
    """将分类表中的布尔文本转换为布尔值。"""
    return str(val).strip().lower() in ("1", "true", "yes")


def _parse_float(val):
    """解析可选数值字段，缺失或非法值返回空值。"""
    if val is None or val == "":
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def load_classifications(csv_path):
    """读取阶段 3 逐叶分类结果表 leaf_classifications.csv，按 (plant_id, leaf_id) 建立索引。"""
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
            if (pid, lid) in out:
                raise ValueError(f"Duplicate classification key: {(pid, lid)}")
            out[(pid, lid)] = dict(
                plant_id=pid,
                leaf_id=lid,
                norm_base_height=_parse_float(row.get("norm_base_height")),
                relative_height=_parse_float(row.get("relative_height", row.get("stem_rel_height", row.get("norm_base_height")))),
                stem_rel_height=_parse_float(row.get("stem_rel_height")),
                projected_area_element=_parse_float(row.get("projected_area_element")),
                coverage_observed=_parse_float(row.get("coverage_observed")),
                coverage_extrapolated=_parse_float(row.get("coverage_extrapolated")),
                cluster_without_height=row.get("cluster_without_height", "").strip(),
                changed_by_height=_parse_bool(row.get("changed_by_height")),
                semantic_reassigned=_parse_bool(row.get("semantic_reassigned")),
                quality_weight=_parse_float(row.get("quality_weight")),
                stem_leaf_angle_deg=_parse_float(row.get("stem_leaf_angle_deg")),
                projection_ratio=_parse_float(row.get("projection_ratio")),
                rel_projected_area=_parse_float(row.get("rel_projected_area")),
                pitch_entropy_b12=_parse_float(row.get("pitch_entropy_b12")),
                mean_pitch_deg=_parse_float(row.get("mean_pitch_deg")),
                tip_gain=_parse_float(row.get("tip_gain")),
                droop_ratio=_parse_float(row.get("droop_ratio")),
                cluster_kmeans=row.get("cluster_kmeans", "").strip(),
                cluster_kmeans_lopo_oof=row.get("cluster_kmeans_lopo_oof", "").strip(),
                lopo_switched=_parse_bool(row.get("lopo_switched")),
                distance_ratio_r=_parse_float(row.get("distance_ratio_r")),
                ambiguous_r080=_parse_bool(row.get("ambiguous_r080")),
            )
    return out


def _class_info(cls_name):
    """查询叶片分类记录并提供缺失分类时的默认显示信息。"""
    return CLASS_META.get(cls_name, dict(zh="未分类", en="unknown", color="#8c8c8c", rgb=(140, 140, 140)))


def _leaf_hover_text(leaf_id, cls_rec, active_mode):
    """组合叶片编号、分类、四因素和质量提示文字。"""
    if not cls_rec:
        return f"叶片 {leaf_id}"
    col, mode_label = COLOR_MODE_COLUMNS.get(active_mode, ("cluster_kmeans", "四因素姿态分类 (K-means)"))
    cls_key = cls_rec.get(col, "")
    cm = _class_info(cls_key)
    flags = []
    if cls_rec.get("ambiguous_r080"):
        flags.append("边界模糊 r>0.80")
    flag_line = f"<br><b>标记:</b> {' · '.join(flags)}" if flags else ""
    def fmt(name, digits=3):
        """将可选表型值格式化为悬浮提示文本。"""
        value = cls_rec.get(name)
        return f"{value:.{digits}f}" if value is not None else "—"
    return (
        f"<b>叶片 {leaf_id} · {cm['zh']} ({cm['en']})</b> [{mode_label}]"
        f"<br>茎叶角 θ={fmt('stem_leaf_angle_deg', 1)}° | 同株相对投影面积 P={fmt('rel_projected_area')} | 俯仰熵 H={fmt('pitch_entropy_b12')}"
        f"<br>相对着生高度 h={fmt('relative_height')} | 观测覆盖率={fmt('coverage_observed')}"
        f"<br>边界比 r={fmt('distance_ratio_r')}"
        f"{flag_line}"
    )


def scan_catalog(root):
    """扫描重建输出目录，返回每株已生成曲面网格的叶片编号列表。"""
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
    """确定性抽稀点云，控制前端传输和渲染开销。"""
    if len(pts) <= limit:
        return pts, rgb
    idx = np.linspace(0, len(pts) - 1, limit).astype(int)
    return pts[idx], (rgb[idx] if rgb is not None else None)


def load_leaf_output(leaves_dir, leaf_id):
    """读取单叶导出网格、点云、中脉及质量信息。"""
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
    """从已有重建结果组装单株或单叶查看场景。"""
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
    """从连续模型文件采样并组装查看场景。"""
    m = MaizeLeafModel.load(model_path)
    sv, faces = m.mesh_grid(nu, nv)
    verts = m.evaluate(sv)
    freg = np.where((sv[:, 0] > m.length_obs + 1e-8)[faces].any(axis=1), 2, 0)
    return dict(
        title=Path(model_path).stem,
        plant=None,
        leaf=0,
        stem=None,
        leaves=[dict(leaf_id=0, points=None, rgb=None, vertices=verts, faces=faces, face_regions=freg,
                     midrib=m.evaluate(np.column_stack([np.linspace(0.0, m.length, nu), np.zeros(nu)])),
                     traits=m.blade.to_dict())],
    )


def load_from_raw(input_root, plant=None, leaf=None):
    """从原始配对点云拟合后组装查看场景。"""
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
    """将整株或单叶几何与分类信息打包为前端查看器所需的轻量 Plotly JSON 载荷。"""
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
    n_lopo_switched = 0
    leaf_summaries = []

    for item in scene["leaves"]:
        lid = item["leaf_id"]
        cls_rec = classifications.get((scene["plant"], lid))
        cls_key = cls_rec.get(cls_col, "") if (use_class and cls_rec) else (cls_rec.get("cluster_kmeans", "") if cls_rec else "")
        cm = _class_info(cls_key)
        is_ambig = bool(cls_rec and cls_rec.get("ambiguous_r080"))
        is_sw = bool(cls_rec and cls_rec.get("lopo_switched"))
        if cls_key in class_counts:
            class_counts[cls_key] += 1
        if is_ambig:
            n_ambiguous += 1
        if is_sw:
            n_lopo_switched += 1

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
            midrib_color = "#f59e0b" if (use_class and (is_ambig or is_sw)) else ("#1f2937" if use_class else "#175eb4")
            midrib_width = 6 if (use_class and (is_ambig or is_sw)) else 4
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
            mark = ("*" if is_ambig else "") + ("!" if is_sw else "")
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
                    region_facecolors=["rgb(" + ",".join(str(int(c)) for c in COLORS[int(k)]) + ")" for k in fr],
                    face_regions=fr.tolist(),
                    x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                    i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
                    color=f"rgb({r_c},{g_c},{b_c})",
                    opacity=1.0,
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
            support_review_required=bool(item["traits"].get("support_review_required", False)),
            support_far_area_fraction=item["traits"].get("support_far_area_fraction"),
        ))

    if label_x:
        traces.append(dict(
            type="scatter3d",
            mode="markers+text",
            name="叶片类别标签",
            group="labels",
            x=label_x, y=label_y, z=label_z,
            text=label_text,
            leaf_ids=[x["leaf_id"] for x in scene["leaves"] if x["midrib"] is not None and len(x["midrib"])],
            textposition="top center",
            textfont=dict(size=12, color=label_colors, family="Microsoft YaHei, sans-serif"),
            marker=dict(size=4, color=label_colors, line=dict(color="#ffffff", width=1)),
            hovertext=label_hover,
            hoverinfo="text",
            showlegend=False,
        ))

    trs = [x.get("traits", {}) for x in scene["leaves"] if x.get("traits")]
    info = dict(scene["leaves"][0].get("traits", {})) if n_leaves == 1 else {}
    info.update(
        plant=scene["plant"], leaf_count=n_leaves,
        total_area=float(sum(t.get("surface_area", 0.0) for t in trs)),
        mean_fit_rmse=float(np.mean([t.get("fit_rmse", 0.0) for t in trs])) if trs else 0.0,
        color_by=color_by if use_class else "region", has_classification=bool(classifications),
        class_counts=class_counts, n_ambiguous=n_ambiguous, n_lopo_switched=n_lopo_switched,
        leaves=leaf_summaries,
    )
    if n_leaves == 1:
        info["classification"] = leaf_summaries[0]["classification"]
        info["active_class"] = leaf_summaries[0]["active_class"]

    title_suffix = f" · {mode_label}" if use_class else ""
    return dict(
        title=scene["title"] + title_suffix,
        plant=scene["plant"],
        leaf=scene["leaf"],
        color_by=color_by if use_class else "region",
        traces=traces,
        info=info,
    )


APP_HTML = (ROOT / "src/maize_leaf/viewer.html").read_text(encoding="utf-8")


def load_classification_metadata(csv_path):
    """读取当前分类模型的特征、权重及规则元数据。"""
    path = Path(csv_path).parent / "classification_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def viewer_page(bootstrap=None, offline=False):
    """将初始化数据安全嵌入共用在线与离线查看器模板。"""
    data = json.dumps(bootstrap, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return (APP_HTML.replace("__PLOTLY_SRC__", "assets/plotly.min.js" if offline else "/assets/plotly.min.js")
            .replace("__BOOTSTRAP_JSON__", data))


def make_handler(root, catalog, init_plant, init_leaf, classifications, init_color_by,
                 classification_metadata=None):
    """创建提供页面、目录和场景接口的 HTTP 处理器。"""
    from functools import lru_cache
    plotly_js = Path(root) / "assets" / "plotly.min.js"

    @lru_cache(maxsize=32)
    def scene_json(plant, leaf, mode):
        """缓存并序列化请求场景，避免重复读取大网格。"""
        scene = load_scene_from_outputs(root, plant, leaf)
        payload = scene_to_plotly_payload(scene, classifications=classifications, color_by=mode)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)

    class Handler(BaseHTTPRequestHandler):
        """处理共用查看器页面与只读数据接口。"""
        def log_message(self, format, *args):
            """按查看器约定输出 HTTP 请求日志。"""
            pass

        def _send(self, code, content_type, body):
            """发送指定状态、内容类型及响应体。"""
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            """校验查询参数并返回页面、资源或场景 JSON。"""
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", viewer_page())
            elif parsed.path == "/assets/plotly.min.js":
                self._send(200, "application/javascript; charset=utf-8", plotly_js.read_bytes())
            elif parsed.path == "/api/catalog":
                body = dict(catalog=catalog, initial_plant=init_plant, initial_leaf=init_leaf,
                            initial_color_by=init_color_by, has_classification=bool(classifications),
                            classification_metadata=classification_metadata or {})
                self._send(200, "application/json; charset=utf-8", json.dumps(body, ensure_ascii=False))
            elif parsed.path == "/api/scene":
                qs = parse_qs(parsed.query)
                plant = qs.get("plant", [next(iter(catalog))])[0]
                mode = qs.get("color_by", [init_color_by])[0]
                try:
                    leaf = int(qs["leaf"][0]) if "leaf" in qs else None
                    if plant not in catalog or leaf is not None and leaf not in catalog[plant]:
                        raise ValueError("Unknown plant or leaf")
                    if mode not in ("kmeans", "region"):
                        raise ValueError("Unknown coloring mode")
                    body = scene_json(plant, leaf, mode)
                except ValueError as exc:
                    self._send(400, "application/json", json.dumps({"error": str(exc)}))
                    return
                self._send(200, "application/json; charset=utf-8", body)
            else:
                self._send(404, "text/plain; charset=utf-8", "Not found")
    return Handler


def run_server(root, init_plant=None, init_leaf=None, port=8050, open_browser=True,
               classifications=None, init_color_by="kmeans", classification_metadata=None):
    """启动本地查看器服务并按参数打开浏览器。"""
    catalog = scan_catalog(root)
    if not catalog:
        raise FileNotFoundError(f"No modeled plants under {root}")
    if init_plant is not None and init_plant not in catalog:
        raise ValueError(f"Unknown plant: {init_plant}")
    init_plant = init_plant or next(iter(catalog))
    if init_leaf is not None and init_leaf not in catalog[init_plant]:
        raise ValueError(f"Unknown leaf: {init_leaf}")
    ensure_plotly_js(root)
    handler = make_handler(root, catalog, init_plant, init_leaf, classifications or {},
                           init_color_by, classification_metadata)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Viewer: {url} ({len(catalog)} plants)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export_classification_viewer(root, classification_csv, out_dir, default_color_by="kmeans"):
    """导出与在线查看器一致的离线 HTML 目录（index.html + 逐株页面，无需联网或本地服务）。"""
    from html import escape
    out_dir = Path(out_dir)
    classifications = load_classifications(classification_csv)
    if not classifications:
        raise FileNotFoundError(f"Classification CSV not found or empty: {classification_csv}")
    catalog = scan_catalog(root)
    if not catalog:
        raise FileNotFoundError(f"No modeled plants under {root}")
    metadata = load_classification_metadata(classification_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(out_dir)
    counts = {c: 0 for c in CLASS_META}
    total = ambiguous = 0
    index_rows = []
    for plant in catalog:
        scene = load_scene_from_outputs(root, plant)
        # Class mesh traces also carry per-face region colors for offline mode switching.
        payload = scene_to_plotly_payload(scene, classifications, "kmeans")
        bootstrap = dict(offline=True, catalog=catalog, initial_plant=plant, initial_leaf=None,
                         initial_color_by=default_color_by, payload=payload, has_classification=True,
                         classification_metadata=metadata)
        (out_dir / f"plant_{plant}.html").write_text(viewer_page(bootstrap, offline=True), encoding="utf-8")
        info = payload["info"]
        cc = info["class_counts"]
        n = info["leaf_count"]
        total += n
        ambiguous += info["n_ambiguous"]
        for c in counts:
            counts[c] += cc[c]
        cells = "".join(f'<td style="color:{CLASS_META[c]["color"]}">{cc[c]}</td>' for c in counts)
        index_rows.append(f'<tr><td><a href="plant_{escape(plant, quote=True)}.html">{escape(plant)}</a></td>'
                          f'<td>{n}</td>{cells}<td>{info["n_ambiguous"]}</td></tr>')
    weight = metadata.get("primary_model", {}).get("height_weight", "未记录")
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>玉米叶片 · 四因素分类总览</title>
<style>body{{margin:0;background:#f2f5f3;color:#183e32;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif}}
header{{background:#123f33;color:#fff;padding:24px}}h1{{font-size:24px;margin:0}}
main{{max-width:1100px;margin:auto;padding:24px}}.card{{background:#fff;padding:20px;border:1px solid #dce5df;border-radius:12px;margin-bottom:18px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #e5ece7}}
a{{color:#197452}}.chips{{display:flex;gap:20px;flex-wrap:wrap}}.table-scroll{{overflow:auto}}
</style><header><h1>玉米叶片 · 四因素姿态分类</h1>
<div>茎叶角 θ · 同株相对投影面积 P · 俯仰熵 H · 相对茎秆着生高度 h</div></header><main>
<div class="card chips"><b>{len(catalog)} 株 / {total} 片叶</b><span>紧缩 {counts['convergent']}</span>
<span>舒展 {counts['expanded']}</span><span>下垂 {counts['drooping']}</span>
<span>边界模糊 {ambiguous}</span></div>
<div class="card"><b>点击植株打开交互视图</b><div class="table-scroll"><table>
<tr><th>植株</th><th>叶片数</th><th>紧缩</th><th>舒展</th><th>下垂</th><th>边界</th></tr>
{''.join(index_rows)}</table></div></div>
<div class="card">高度距离权重：{weight}（标准化后）。<br>
紧缩候选为顶部小型未充分展开叶；没有每株类别配额。高度分布仅用于描述；下垂不等于衰老。<br>
<a href="../classification_analysis.html">查看交互式消融与高度权重对比</a> ·
<a href="../leaf_classifications.csv">逐叶分类表</a> · <a href="../classification_report.json">参数与验证记录</a>
</div></main></html>"""
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    print(f"Exported {len(catalog)} plants / {total} leaves -> {index}", flush=True)
    return index


def show_plotly_static(scene, args, classifications=None):
    """用 Plotly 显示或保存指定场景。"""
    import plotly.graph_objects as go

    payload = scene_to_plotly_payload(scene, classifications=classifications, color_by=args.color_by)
    traces = []
    for t in payload["traces"]:
        t = dict(t)
        grp = t.pop("group", None)
        t.pop("region", None)
        t.pop("leaf_id", None)
        t.pop("pose_class", None)
        t.pop("region_facecolors", None)
        t.pop("face_regions", None)
        t.pop("leaf_ids", None)
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
    """用 Matplotlib 显示或保存指定场景的静态检查图。"""
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
    """解析场景和后端选项，启动在线查看器或导出离线与静态结果。"""
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
        choices=("region", "kmeans"),
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
    metadata = load_classification_metadata(args.classification_csv)
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
            classification_metadata=metadata,
        )
        return

    scene = load_scene_from_outputs(args.root, args.plant, args.leaf)
    if args.backend == "mpl" or (args.save and args.save.suffix.lower() in (".png", ".jpg", ".svg", ".pdf")):
        show_mpl(scene, args, classifications=classifications)
    else:
        show_plotly_static(scene, args, classifications=classifications)


if __name__ == "__main__":
    main()
