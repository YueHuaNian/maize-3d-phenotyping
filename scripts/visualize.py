"""Interactive 3D visualizer with plant selector and single-leaf view.

Usage:
    python scripts/visualize.py                         # Launch interactive browser UI (all plants & leaves)
    python scripts/visualize.py --list                  # List available plants and leaf IDs
    python scripts/visualize.py --plant 1-1-2           # Open UI pre-selected on plant 1-1-2
    python scripts/visualize.py --plant 1-1-2 --leaf 1  # Open UI pre-selected on plant 1-1-2, leaf 1
    python scripts/visualize.py --plant 1-1-2 --save out.html
    python scripts/visualize.py --plant 1-1-2 --leaf 1 --backend mpl --save leaf_01.png
    python scripts/visualize.py --model outputs/maize_v2/plant_1-1-2/leaves/leaf_01_model.npz
"""
import argparse
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


def scene_to_plotly_payload(scene):
    """Build lightweight Plotly traces + trait summary JSON for a plant or single leaf."""
    traces = []
    n_leaves = len(scene["leaves"])
    pt_limit = 9000 if n_leaves == 1 else max(1800, 15000 // max(1, n_leaves))

    if scene["stem"] is not None:
        s = np.round(scene["stem"], 5)
        traces.append(dict(
            type="scatter3d", mode="lines", name="Stem", group="stem",
            x=s[:, 0].tolist(), y=s[:, 1].tolist(), z=s[:, 2].tolist(),
            line=dict(color="#5134a1", width=7),
        ))

    for item in scene["leaves"]:
        prefix = f"Leaf {item['leaf_id']} · " if n_leaves > 1 else ""
        if item["points"] is not None and len(item["points"]):
            pts, rgb = sample_points(item["points"], item["rgb"], pt_limit)
            pts = np.round(pts, 5)
            colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb] if rgb is not None else "#334c42"
            traces.append(dict(
                type="scatter3d", mode="markers", name=prefix + "Points", group="points",
                x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=pts[:, 2].tolist(),
                marker=dict(size=2, color=colors, opacity=0.68),
            ))
        if item["midrib"] is not None and len(item["midrib"]):
            m = np.round(item["midrib"], 5)
            traces.append(dict(
                type="scatter3d", mode="lines", name=prefix + "Midrib", group="midrib",
                x=m[:, 0].tolist(), y=m[:, 1].tolist(), z=m[:, 2].tolist(),
                line=dict(color="#175eb4", width=4),
            ))
        if item["vertices"] is not None and len(item["faces"]):
            v = np.round(item["vertices"], 5)
            f, fr = item["faces"], item["face_regions"]
            for k, rname in REGIONS.items():
                fk = f[fr == k]
                if not len(fk):
                    continue
                color = "rgb(" + ",".join(str(int(c)) for c in COLORS[k]) + ")"
                traces.append(dict(
                    type="mesh3d", name=prefix + rname, group="surface", region=rname,
                    x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                    i=fk[:, 0].tolist(), j=fk[:, 1].tolist(), k=fk[:, 2].tolist(),
                    color=color, opacity=0.88, showlegend=True,
                ))

    if n_leaves == 1:
        info = scene["leaves"][0].get("traits", {})
    else:
        trs = [x.get("traits", {}) for x in scene["leaves"] if x.get("traits")]
        info = dict(
            plant=scene["plant"],
            leaf_count=n_leaves,
            total_area=float(sum(t.get("surface_area", 0.0) for t in trs)),
            mean_alpha=float(np.mean([t["form_factor_alpha"] for t in trs])) if trs else 0.0,
            mean_fit_rmse=float(np.mean([t["fit_rmse"] for t in trs])) if trs else 0.0,
            leaves=[
                dict(
                    leaf_id=x["leaf_id"],
                    observed_length=x["traits"].get("observed_length"),
                    completed_length=x["traits"].get("completed_length"),
                    max_width=x["traits"].get("max_width"),
                    form_factor_alpha=x["traits"].get("form_factor_alpha"),
                    surface_area=x["traits"].get("surface_area"),
                    fit_rmse=x["traits"].get("fit_rmse"),
                )
                for x in scene["leaves"]
            ],
        )

    return dict(title=scene["title"], plant=scene["plant"], leaf=scene["leaf"], traces=traces, info=info)


APP_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>玉米三维叶片可视化浏览器</title>
<script src="/assets/plotly.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#f3f6f5;color:#16372e;font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;height:100vh}
  header{background:#113e33;color:#fff;padding:12px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:17px;margin:0;margin-right:8px;white-space:nowrap}
  .ctrl{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  select,button{height:32px;border-radius:6px;border:1px solid #2b6b59;padding:0 10px;font-size:13px;background:#fff;color:#16372e;cursor:pointer}
  button{background:#1c6b57;color:#fff;border-color:#2c8c74}
  button:hover{background:#23826a}
  button.active{background:#f7b93f;color:#16372e;border-color:#f7b93f;font-weight:600}
  .toggles{display:flex;align-items:center;gap:12px;margin-left:auto;font-size:13px}
  .toggles label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
  .main{display:flex;flex:1;min-height:0}
  #plot{flex:1;min-width:0;background:#fff}
  aside{width:320px;background:#fff;border-left:1px solid #dce7e2;padding:14px;overflow-y:auto}
  .status{font-size:12px;color:#526a61;margin-bottom:10px}
  .kv{display:grid;grid-template-columns:1fr auto;gap:6px 10px;padding:10px;background:#f6faf8;border-radius:8px;border:1px solid #e2ece7;margin-bottom:12px}
  .kv .k{color:#536a62}.kv .v{font-weight:600;font-variant-numeric:tabular-nums}
  .leaf-list{display:flex;flex-direction:column;gap:6px}
  .leaf-btn{display:flex;justify-content:space-between;align-items:center;width:100%;height:auto;padding:8px 10px;background:#f6faf8;color:#16372e;border:1px solid #dce7e2;text-align:left}
  .leaf-btn:hover{background:#e7f3ee}
  .leaf-btn.selected{background:#113e33;color:#fff;border-color:#113e33}
  .legend{display:flex;gap:10px;margin:8px 0 12px;font-size:12px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
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
  </div>
  <div class="toggles">
    <label><input type="checkbox" id="chkPoints" checked>点云</label>
    <label><input type="checkbox" id="chkSurface" checked>曲面</label>
    <label><input type="checkbox" id="chkMidrib" checked>中脉</label>
    <label><input type="checkbox" id="chkStem" checked>茎秆</label>
  </div>
</header>
<div class="main">
  <div id="plot"></div>
  <aside>
    <div class="status" id="status">正在加载目录...</div>
    <div class="legend">
      <span><i class="dot" style="background:rgb(40,158,123)"></i>observed</span>
      <span><i class="dot" style="background:rgb(247,185,63)"></i>interpolated</span>
      <span><i class="dot" style="background:rgb(218,79,116)"></i>extrapolated</span>
    </div>
    <div id="metrics"></div>
    <div id="leafPanel"></div>
  </aside>
</div>
<script>
let catalog = {};
let plantNames = [];
let state = { plant: null, leaf: null, payload: null };

const plantSel = document.getElementById('plantSel');
const leafSel = document.getElementById('leafSel');
const wholeBtn = document.getElementById('wholeBtn');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const leafPanelEl = document.getElementById('leafPanel');

function fmt(v, d=4) { return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }

function populateLeaves(plant) {
  const lids = catalog[plant] || [];
  leafSel.innerHTML = '<option value="">-- 整株全部叶片 --</option>' +
    lids.map(id => `<option value="${id}">叶片 ${id}</option>`).join('');
  leafSel.value = state.leaf !== null ? String(state.leaf) : '';
  wholeBtn.classList.toggle('active', state.leaf === null);
}

function renderPlot() {
  if (!state.payload) return;
  const show = {
    points: document.getElementById('chkPoints').checked,
    surface: document.getElementById('chkSurface').checked,
    midrib: document.getElementById('chkMidrib').checked,
    stem: document.getElementById('chkStem').checked,
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
  const info = state.payload.info || {};
  const lids = catalog[state.plant] || [];
  if (state.leaf === null) {
    metricsEl.innerHTML = `<div class="kv">
      <span class="k">当前植株</span><span class="v">${state.plant}</span>
      <span class="k">叶片数量</span><span class="v">${info.leaf_count ?? lids.length}</span>
      <span class="k">总叶面积</span><span class="v">${fmt(info.total_area, 4)}</span>
      <span class="k">平均形态系数 α</span><span class="v">${fmt(info.mean_alpha, 3)}</span>
      <span class="k">平均拟合 RMSE</span><span class="v">${fmt(info.mean_fit_rmse, 4)}</span>
    </div>`;
    const rows = (info.leaves || []).map(l =>
      `<button class="leaf-btn" onclick="selectView('${state.plant}', ${l.leaf_id})">
        <span><b>叶片 ${l.leaf_id}</b></span>
        <span>L=${fmt(l.completed_length,2)} · W=${fmt(l.max_width,3)}</span>
      </button>`
    ).join('');
    leafPanelEl.innerHTML = '<div style="font-weight:600;margin-bottom:6px">点击查看单个叶片：</div><div class="leaf-list">' + rows + '</div>';
  } else {
    const frac = info.area_fractions || {};
    metricsEl.innerHTML = `<div class="kv">
      <span class="k">植株 / 叶片</span><span class="v">${state.plant} · 叶片 ${state.leaf}</span>
      <span class="k">原始点数</span><span class="v">${info.raw_points ?? '-'}</span>
      <span class="k">观测叶长 L_obs</span><span class="v">${fmt(info.observed_length, 3)}</span>
      <span class="k">补全叶长 L_full</span><span class="v">${fmt(info.completed_length, 3)}</span>
      <span class="k">叶尖补全长</span><span class="v">${fmt(info.tip_completion_length, 3)}</span>
      <span class="k">最大叶宽 W_max</span><span class="v">${fmt(info.max_width, 4)}</span>
      <span class="k">形态系数 α</span><span class="v">${fmt(info.form_factor_alpha, 3)}</span>
      <span class="k">总叶面积</span><span class="v">${fmt(info.surface_area, 4)}</span>
      <span class="k">叶基倾角 (°)</span><span class="v">${fmt(info.base_inclination_deg, 1)}</span>
      <span class="k">拟合 RMSE</span><span class="v">${fmt(info.fit_rmse, 4)}</span>
      <span class="k">observed / interp / extrap</span>
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

async function selectView(plant, leaf) {
  state.plant = plant;
  state.leaf = leaf;
  plantSel.value = plant;
  populateLeaves(plant);
  statusEl.textContent = leaf === null ? `正在加载植株 ${plant} 整株数据...` : `正在加载植株 ${plant} 叶片 ${leaf}...`;
  const url = leaf === null
    ? `/api/scene?plant=${encodeURIComponent(plant)}`
    : `/api/scene?plant=${encodeURIComponent(plant)}&leaf=${leaf}`;
  const resp = await fetch(url);
  state.payload = await resp.json();
  statusEl.textContent = `已加载：${state.payload.title}（拖动旋转，滚轮缩放，点击图例隐藏图层）`;
  renderPlot();
  renderSidebar();
}

plantSel.onchange = () => selectView(plantSel.value, null);
leafSel.onchange = () => selectView(state.plant, leafSel.value ? Number(leafSel.value) : null);
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
['chkPoints','chkSurface','chkMidrib','chkStem'].forEach(id => {
  document.getElementById(id).onchange = renderPlot;
});

async function boot() {
  const resp = await fetch('/api/catalog');
  const data = await resp.json();
  catalog = data.catalog;
  plantNames = Object.keys(catalog);
  plantSel.innerHTML = plantNames.map(p => `<option value="${p}">${p} (${catalog[p].length}叶)</option>`).join('');
  const initPlant = data.initial_plant && catalog[data.initial_plant] ? data.initial_plant : plantNames[0];
  const initLeaf = data.initial_leaf !== null && (catalog[initPlant] || []).includes(data.initial_leaf) ? data.initial_leaf : null;
  await selectView(initPlant, initLeaf);
}
boot();
</script>
</body>
</html>
"""


def make_handler(root, catalog, init_plant, init_leaf):
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
                body = json.dumps(dict(catalog=catalog, initial_plant=init_plant, initial_leaf=init_leaf), ensure_ascii=False)
                self._send(200, "application/json; charset=utf-8", body)
            elif parsed.path == "/api/scene":
                qs = parse_qs(parsed.query)
                plant = qs.get("plant", [next(iter(catalog))])[0]
                leaf = int(qs["leaf"][0]) if "leaf" in qs else None
                scene = load_scene_from_outputs(root, plant, leaf)
                payload = scene_to_plotly_payload(scene)
                self._send(200, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False))
            else:
                self._send(404, "text/plain; charset=utf-8", "Not found")

    return Handler


def run_server(root, init_plant=None, init_leaf=None, host="127.0.0.1", port=8050, open_browser=True):
    ensure_plotly_js(root)
    catalog = scan_catalog(root)
    server = ThreadingHTTPServer((host, port), make_handler(root, catalog, init_plant, init_leaf))
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"Visualizer running at {url} ({len(catalog)} plants, {sum(len(v) for v in catalog.values())} leaves). Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def show_plotly_static(scene, args):
    import plotly.graph_objects as go

    payload = scene_to_plotly_payload(scene)
    traces = []
    for t in payload["traces"]:
        t = dict(t)
        grp = t.pop("group", None)
        t.pop("region", None)
        if (grp == "points" and args.no_points) or (grp == "midrib" and args.no_midrib) or (grp == "surface" and args.no_surface):
            continue
        typ = t.pop("type")
        traces.append(go.Mesh3d(**t) if typ == "mesh3d" else go.Scatter3d(**t))
    fig = go.Figure(traces)
    fig.update_layout(
        title=scene["title"], height=760, margin=dict(l=0, r=0, t=40, b=0),
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
            coll = Poly3DCollection(v[item["faces"]], facecolors=COLORS[item["face_regions"]] / 255.0,
                                    edgecolor="none", alpha=0.85, zsort="average")
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
    p = argparse.ArgumentParser(description="Interactive 3D visualizer with plant and single-leaf selector")
    p.add_argument("--root", type=Path, default=ROOT / "outputs/maize_v2", help="Modeling output directory")
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

    if args.list:
        catalog = scan_catalog(args.root)
        for plant, lids in catalog.items():
            print(f"{plant}: leaves {lids}")
        print(f"Total: {len(catalog)} plants, {sum(len(v) for v in catalog.values())} leaves")
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
        run_server(args.root, args.plant, args.leaf, port=args.port, open_browser=not args.no_browser)
        return

    scene = load_scene_from_outputs(args.root, args.plant, args.leaf)
    if args.backend == "mpl" or (args.save and args.save.suffix.lower() in (".png", ".jpg", ".svg", ".pdf")):
        show_mpl(scene, args)
    else:
        show_plotly_static(scene, args)


if __name__ == "__main__":
    main()
