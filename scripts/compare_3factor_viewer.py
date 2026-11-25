"""Generate side-by-side 3D interactive HTML comparison for 3-factor pose classification schemes.

Compares 4 mathematically grounded (stem-leaf angle + projected area + entropy) schemes
against the previous M4 baseline, all strictly excluding leaf height from clustering:
  - S1: [theta, rho, H_pit]             (K-means++, pure Shannon pitch entropy)
  - S2: [theta, Ah_rel, H_pit]          (K-means++, within-plant relative projected area + pure Shannon pitch entropy)
  - S3: [theta, rho, delta_JSD]         (K-means++, signed Jensen-Shannon relative entropy of pitch distribution)
  - S4: [theta, Avis_rel, delta_JSD]    (FCM m=2.0, canopy-visible nadir projected area + signed relative entropy)
  - M4: [theta, g, d, K_mid, rho]       (previous Stage E K-means baseline for before/after visual comparison)

Outputs:
  - outputs/classification_v2/compare_3factor_leaves.csv
  - outputs/classification_v2/compare_3factor_report.json
  - outputs/classification_v2/viewer_3factor/index.html + 14 plant_<name>.html
"""
import argparse
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from maize_leaf.classification import (
    CLASS_NAMES,
    FuzzyCMeansModel,
    KMeansModel,
    RobustScaler,
    adjusted_rand_index,
    silhouette_score,
)
from maize_leaf.surface import MaizeLeafModel
from maize_leaf.viz import ensure_plotly_js
import visualize as viz_mod


SCHEMES = [
    dict(
        key="f1_4factor_km",
        short="F1(4因素KM)",
        label="四因素 F1：[θ, ρ, Ah_rel, H_pit, g, d] (K-means, 零模板)",
        cols=["theta", "rho", "ah_rel", "H_pit", "g", "d"],
        algo="kmeans",
        anchor_sign=(-1.0, 0.0, 0.0, -0.35, +1.0, -1.0),
        note="四大因素完整版（零模板、纯单叶）：茎叶角 + 投影比&相对投影面积 + 俯仰熵 + 叶尖抬升&下垂比",
    ),
    dict(
        key="f1_4factor_fcm",
        short="F1(4因素FCM)",
        label="四因素 F1：[θ, ρ, Ah_rel, H_pit, g, d] (FCM 软聚类)",
        cols=["theta", "rho", "ah_rel", "H_pit", "g", "d"],
        algo="fcm",
        anchor_sign=(-1.0, 0.0, 0.0, -0.35, +1.0, -1.0),
        note="同 F1 特征的 FCM 软聚类版（LOPO OOF ARI=0.900，可输出软隶属概率）",
    ),
    dict(
        key="c2_s123_fusion",
        short="C2(3因素融合)",
        label="三因素融合 C2：[θ, ρ, Ah_rel, H_pit, ΔJSD] (K-means)",
        cols=["theta", "rho", "ah_rel", "H_pit", "js_bal"],
        algo="kmeans",
        anchor_sign=(-1.0, 0.0, 0.0, -0.35, -0.60),
        note="三因素完整融合：茎叶角 + 投影比&相对投影面积 + 俯仰熵&相对熵",
    ),
    dict(
        key="s1_rho_hpit",
        short="S1",
        label="方案 1：茎叶角 + 投影比 ρ + 俯仰熵 H_pit (K-means)",
        cols=["theta", "rho", "H_pit"],
        algo="kmeans",
        anchor_sign=(-1.0, 0.0, -0.35),
        note="三因素精简版（投影比 + 无模板俯仰熵）",
    ),
    dict(
        key="s2_ahrel_hpit",
        short="S2",
        label="方案 2：茎叶角 + 相对投影面积 Ah_rel + 俯仰熵 H_pit (K-means)",
        cols=["theta", "ah_rel", "H_pit"],
        algo="kmeans",
        anchor_sign=(-1.0, 0.0, -0.35),
        note="三因素精简版（相对投影面积 + 无模板俯仰熵）",
    ),
    dict(
        key="m4_old_kmeans",
        short="旧M4",
        label="对照：旧版 M4 [θ, g, d, K_mid, ρ] (K-means)",
        cols=None,
        algo="existing",
        anchor_sign=None,
        note="之前包含无符号三维曲率 K_mid 的 5 维基线",
    ),
]


def _shannon_norm(hist):
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    p = hist[hist > 0.0] / total
    return float(-np.sum(p * np.log(p)) / np.log(len(hist)))


def _pitch_entropy_and_jsd(model, ns=60, nv=17, n_bins=12):
    """Area-weighted longitudinal pitch-angle Shannon entropy H_pit and signed JS divergence."""
    s_vals = np.linspace(0.0, model.blade.length, ns)
    v_vals = np.linspace(-0.9, 0.9, nv)
    ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
    xyz = model.evaluate(np.column_stack([ss.ravel(), vv.ravel()])).reshape(ns, nv, 3)

    ds = np.gradient(xyz, s_vals, axis=0, edge_order=2)
    dv = np.gradient(xyz, v_vals, axis=1, edge_order=2)
    jac = np.linalg.norm(np.cross(ds, dv), axis=2)
    ds_norm = np.maximum(np.linalg.norm(ds, axis=2), 1e-12)
    pitch_deg = np.degrees(np.arcsin(np.clip(ds[:, :, 2] / ds_norm, -1.0, 1.0)))

    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    hist, _ = np.histogram(pitch_deg.ravel(), bins=edges, weights=jac.ravel())
    h_pit = _shannon_norm(hist)

    centers = 0.5 * (edges[:-1] + edges[1:])
    p = (hist + 1e-6) / np.sum(hist + 1e-6)

    def _gauss_tpl(mu, sig):
        q = np.exp(-0.5 * ((centers - mu) / sig) ** 2) + 1e-6
        return q / np.sum(q)

    def _jsd(a, b):
        m = 0.5 * (a + b)
        return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))

    js_erect = _jsd(p, _gauss_tpl(60.0, 20.0))
    js_droop = _jsd(p, _gauss_tpl(-35.0, 22.0))
    js_bal = js_erect - js_droop
    mean_pitch = float(np.sum(pitch_deg * jac) / max(float(np.sum(jac)), 1e-12))
    return h_pit, js_bal, js_erect, js_droop, mean_pitch


def _fit_and_eval_scheme(records, scheme, seed=42):
    X = np.column_stack([[r[c] for r in records] for c in scheme["cols"]])
    plants = np.array([r["plant_id"] for r in records])
    uniq_p = sorted(set(plants))
    w_sign = np.asarray(scheme["anchor_sign"], dtype=float)

    def _fit_predict(Xtr, Xte):
        sc = RobustScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if scheme["algo"] == "fcm":
            mdl = FuzzyCMeansModel(n_clusters=3, m=2.0, n_init=30, seed=seed).fit(Ztr)
        else:
            mdl = KMeansModel(n_clusters=3, n_init=50, seed=seed).fit(Ztr)
        C = mdl.centers_
        c_std = np.std(C, axis=0)
        Zc = (C - np.mean(C, axis=0)) / np.where(c_std > 1e-12, c_std, 1.0)
        scores = Zc @ w_sign
        order = np.argsort(scores)[::-1]  # 0=convergent, 1=expanded, 2=drooping
        remap = {int(old): new for new, old in enumerate(order)}
        pred = np.array([remap[int(c)] for c in mdl.predict(Zte)], dtype=int)
        r_ratio = mdl.distance_ratio(Zte) if hasattr(mdl, "distance_ratio") else np.zeros(len(Zte))
        return pred, r_ratio, Ztr

    full_ids, full_r, Zfull = _fit_predict(X, X)
    oof_ids = np.empty(len(X), dtype=int)
    fold_aris = []
    for p in uniq_p:
        tr = plants != p
        te = ~tr
        oof_ids[te], _, _ = _fit_predict(X[tr], X[te])
        all_p, _, _ = _fit_predict(X[tr], X)
        fold_aris.append(adjusted_rand_index(full_ids, all_p))

    oof_ari = adjusted_rand_index(full_ids, oof_ids)
    fold_ari = float(np.mean(fold_aris))
    sil = silhouette_score(Zfull, full_ids)
    labels = [CLASS_NAMES[int(i)] for i in full_ids]
    cnts = {k: int(sum(1 for lb in labels if lb == k)) for k in CLASS_NAMES}
    top_c0 = float(np.mean([labels[i] == "convergent" for i, r in enumerate(records) if r["rank_top"] <= 2]))
    mid_c1 = float(np.mean([labels[i] == "expanded" for i, r in enumerate(records) if r["rank_top"] >= 3 and r["rank_bot"] >= 2]))
    bot_c2 = float(np.mean([labels[i] == "drooping" for i, r in enumerate(records) if r["rank_bot"] == 1]))
    err_up = int(sum(1 for i, r in enumerate(records) if r["g"] > 0.25 and r["d"] < 0.05 and labels[i] == "drooping"))
    err_dr = int(sum(1 for i, r in enumerate(records) if r["g"] < -0.20 and r["d"] > 0.30 and labels[i] == "convergent"))
    return dict(
        labels=labels,
        distance_ratio=full_r.tolist(),
        counts=cnts,
        oof_ari=oof_ari,
        fold_ari=fold_ari,
        silhouette=sil,
        top12_convergent=top_c0,
        mid_expanded=mid_c1,
        bot1_drooping=bot_c2,
        err_upward_as_drooping=err_up,
        err_drooping_as_convergent=err_dr,
    )


PLANT_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>植株 __PLANT__ · 三因素分类方案对比 3D</title>
<script src="assets/plotly.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#f3f6f5;color:#16372e;font:14px/1.5 system-ui,'Microsoft YaHei',sans-serif;display:flex;flex-direction:column;height:100vh}
  header{background:#113e33;color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;white-space:nowrap}
  header a{color:#d6f5ec;text-decoration:none;font-size:13px;padding:4px 9px;border:1px solid #2b6b59;border-radius:6px}
  header a:hover{background:#1c6b57}
  select,button{height:31px;border-radius:6px;border:1px solid #2b6b59;padding:0 9px;font-size:13px;background:#fff;color:#16372e;cursor:pointer}
  .scheme-bar{background:#e7f1ec;border-bottom:1px solid #cfe0d8;padding:8px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .sch-btn{height:30px;padding:0 11px;border-radius:6px;border:1px solid #9ab9ad;background:#fff;color:#16372e;font-size:12.5px;font-weight:600}
  .sch-btn:hover{background:#dcf0e7}
  .sch-btn.active{background:#113e33;color:#fff;border-color:#113e33}
  .toggles{display:flex;align-items:center;gap:10px;margin-left:auto;font-size:13px;flex-wrap:wrap}
  .toggles label{display:flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
  .main{display:flex;flex:1;min-height:0}
  #plot{flex:1;min-width:0;background:#fff}
  aside{width:385px;background:#fff;border-left:1px solid #dce7e2;padding:14px;overflow-y:auto}
  .legend{display:flex;gap:10px;margin:0 0 8px;font-size:12px;flex-wrap:wrap}
  .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
  .note{font-size:12px;color:#4a635a;background:#f4f9f7;padding:7px 10px;border-radius:6px;border:1px solid #dfeae5;margin-bottom:10px}
  .kv{display:grid;grid-template-columns:1fr auto;gap:4px 10px;padding:9px 10px;background:#f6faf8;border-radius:8px;border:1px solid #e2ece7;margin-bottom:10px;font-size:12.5px}
  .kv .k{color:#536a62}.kv .v{font-weight:600;text-align:right}
  .bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:4px 0 8px;background:#e5e7eb}
  .leaf-list{display:flex;flex-direction:column;gap:6px}
  .leaf-btn{display:flex;flex-direction:column;gap:4px;width:100%;height:auto;padding:8px 10px;background:#f6faf8;color:#16372e;border:1px solid #dce7e2;border-left:5px solid #9ca3af;text-align:left}
  .leaf-btn:hover{background:#e7f3ee}
  .leaf-btn.selected{background:#113e33;color:#fff;border-color:#113e33}
  .leaf-top{display:flex;justify-content:space-between;align-items:center;width:100%}
  .leaf-sub{display:flex;gap:5px;flex-wrap:wrap;font-size:11px;opacity:0.92}
  .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;color:#fff;margin-left:5px}
  .mini{display:inline-block;padding:0 5px;border-radius:4px;font-size:10.5px;color:#fff;font-weight:600}
  .diff-tag{display:inline-block;padding:0 5px;border-radius:4px;font-size:10px;background:#fef3c7;color:#92400e;border:1px solid #fcd34d;margin-left:4px}
</style>
</head>
<body>
<header>
  <h1>植株 __PLANT__ · 三因素对比 3D</h1>
  <a href="index.html">全部 14 株总览</a>
  <a href="__PREV_HREF__">&lsaquo; 上一株</a>
  <a href="__NEXT_HREF__">下一株 &rsaquo;</a>
  <button id="resetLeafBtn">显示全部叶片</button>
  <div class="toggles">
    <label><input type="checkbox" id="chkPoints" checked>点云</label>
    <label><input type="checkbox" id="chkSurface" checked>曲面</label>
    <label><input type="checkbox" id="chkMidrib" checked>中脉</label>
    <label><input type="checkbox" id="chkStem" checked>茎秆</label>
    <label><input type="checkbox" id="chkLabels" checked>标签</label>
  </div>
</header>
<div class="scheme-bar" id="schemeBar"></div>
<div class="main">
  <div id="plot"></div>
  <aside>
    <div class="legend">
      <span><i class="dot" style="background:#1f77b4"></i>紧缩 (蓝)</span>
      <span><i class="dot" style="background:#1b9e77"></i>舒展 (绿)</span>
      <span><i class="dot" style="background:#d95f02"></i>下垂 (橙)</span>
      <span style="color:#92400e"><i class="dot" style="background:#f59e0b"></i>金脉 = S1/S2/S3 间存在分歧</span>
    </div>
    <div id="schemeNote" class="note"></div>
    <div id="summary"></div>
    <div id="leafList" class="leaf-list"></div>
  </aside>
</div>
<script>
const DATA = __PAYLOAD_JSON__;
const SCHEMES = __SCHEMES_JSON__;
const CLS = {
  convergent: { zh: '紧缩', rgb: 'rgb(31,119,180)', hex: '#1f77b4' },
  expanded:   { zh: '舒展', rgb: 'rgb(27,158,119)', hex: '#1b9e77' },
  drooping:   { zh: '下垂', rgb: 'rgb(217,95,2)',   hex: '#d95f02' },
};
let activeKey = SCHEMES[0].key;
let focusLeaf = null;
const leafMap = {};
DATA.leaves_meta.forEach(l => { leafMap[l.leaf_id] = l; });

function fmt(v, d=2) { return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(d); }
function ci(k) { return CLS[k] || { zh: '未知', rgb: 'rgb(140,140,140)', hex: '#8c8c8c' }; }

function buildHover(l) {
  const c = ci(l.labels[activeKey]);
  const sRow = SCHEMES.map(s => `${s.short}=${ci(l.labels[s.key]).zh}`).join(' | ');
  return `<b>叶片 L${l.leaf_id} · ${c.zh}</b> (株内从顶第 #${l.rank_top} 叶, h=${fmt(l.h,2)})` +
    `<br>茎叶角 θ=${fmt(l.theta,1)}° | 投影比 ρ=${fmt(l.rho,2)} | 相对投影 Ah_rel=${fmt(l.ah_rel,2)} | 可见投影 Avis_rel=${fmt(l.av_rel,2)}` +
    `<br>俯仰熵 H_pit=${fmt(l.H_pit,2)} | 相对熵 ΔJSD=${fmt(l.js_bal,2)} | 平均仰角=${fmt(l.mean_pitch,1)}° | g=${fmt(l.g,2)}, d=${fmt(l.d,2)}` +
    `<br><b>各方案对照:</b> ${sRow}`;
}

function renderButtons() {
  document.getElementById('schemeBar').innerHTML = SCHEMES.map(s =>
    `<button class="sch-btn ${s.key===activeKey?'active':''}" onclick="setScheme('${s.key}')">${s.short} · ${s.label.split('：')[1]}</button>`
  ).join('');
}

function setScheme(k) {
  activeKey = k;
  renderButtons();
  render();
}

function render() {
  const sch = SCHEMES.find(s => s.key === activeKey);
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
      const l = leafMap[t.leaf_id];
      const c = ci(l.labels[activeKey]);
      t.color = c.rgb;
      t.name = `L${l.leaf_id} · ${c.zh}`;
      t.hovertext = buildHover(l);
      t.hoverinfo = 'text';
    } else if (t.group === 'midrib' && t.leaf_id !== undefined) {
      const l = leafMap[t.leaf_id];
      t.hovertext = buildHover(l);
      t.hoverinfo = 'text';
    } else if (t.group === 'labels') {
      const xs = [], ys = [], zs = [], txts = [], cols = [], hov = [];
      DATA.leaves_meta.forEach((l, idx) => {
        if (focusLeaf !== null && l.leaf_id !== focusLeaf) return;
        const c = ci(l.labels[activeKey]);
        xs.push(raw.x[idx]); ys.push(raw.y[idx]); zs.push(raw.z[idx]);
        txts.push(`L${l.leaf_id} ${c.zh}${l.s123_diff?'*':''}`);
        cols.push(c.hex);
        hov.push(buildHover(l));
      });
      t.x = xs; t.y = ys; t.z = zs; t.text = txts;
      t.textfont = Object.assign({}, raw.textfont, { color: cols });
      t.marker = Object.assign({}, raw.marker, { color: cols });
      t.hovertext = hov;
    }
    traces.push(t);
  }
  const titleTxt = `植株 __PLANT__ · ${sch.label}` + (focusLeaf !== null ? ` · 叶片 ${focusLeaf}` : '');
  Plotly.react('plot', traces, {
    title: { text: titleTxt, x: 0.02, font: { size: 14.5 } },
    margin: { l: 0, r: 0, t: 36, b: 0 },
    paper_bgcolor: '#ffffff',
    scene: { aspectmode: 'data', xaxis: { title: 'X' }, yaxis: { title: 'Y' }, zaxis: { title: 'Z' },
             camera: { eye: { x: 1.45, y: 1.55, z: 0.9 } } },
    legend: { orientation: 'h', y: 1.0, x: 0.0 },
    uirevision: 'keep-camera',
  }, { responsive: true, displaylogo: false });

  document.getElementById('schemeNote').innerHTML =
    `<b>${sch.label}</b><br>${sch.note}` +
    (sch.oof_ari !== null ? `<br>全集 90 叶：紧缩 ${sch.counts.convergent} / 舒展 ${sch.counts.expanded} / 下垂 ${sch.counts.drooping} · <b>LOPO ARI = ${sch.oof_ari.toFixed(3)}</b>` : '');

  const cnt = { convergent: 0, expanded: 0, drooping: 0 };
  DATA.leaves_meta.forEach(l => { const k = l.labels[activeKey]; if (cnt[k] !== undefined) cnt[k] += 1; });
  const n = Math.max(1, DATA.leaves_meta.length);
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
      <span class="k">本株分布（紧缩 / 舒展 / 下垂）</span>
      <span class="v">${cnt.convergent} / ${cnt.expanded} / ${cnt.drooping}</span>
      <span class="k">S1/S2/S3 存在分歧的叶片</span>
      <span class="v">${DATA.leaves_meta.filter(l=>l.s123_diff).length} / ${DATA.leaves_meta.length}</span>
    </div>`;

  // Sort sidebar top-to-bottom by height so user can inspect canopy order naturally!
  const sorted = [...DATA.leaves_meta].sort((a, b) => a.rank_top - b.rank_top);
  document.getElementById('leafList').innerHTML = sorted.map(l => {
    const c = ci(l.labels[activeKey]);
    const sel = focusLeaf === l.leaf_id ? ' selected' : '';
    const badges = SCHEMES.slice(0, 5).map(s => {
      const sc = ci(l.labels[s.key]);
      return `<span class="mini" style="background:${sc.hex}">${s.short}:${sc.zh}</span>`;
    }).join('');
    return `<button class="leaf-btn${sel}" style="border-left-color:${c.hex}" onclick="toggleLeaf(${l.leaf_id})">
      <div class="leaf-top">
        <span><b>#${l.rank_top} L${l.leaf_id}</b><span class="pill" style="background:${c.hex}">${c.zh}</span>${l.s123_diff?'<span class="diff-tag">分歧</span>':''}</span>
        <span style="font-size:11.5px">θ=${fmt(l.theta,0)}° ρ=${fmt(l.rho,2)} H=${fmt(l.H_pit,2)} h=${fmt(l.h,2)}</span>
      </div>
      <div class="leaf-sub">${badges}</div>
    </button>`;
  }).join('');
}

function toggleLeaf(lid) { focusLeaf = (focusLeaf === lid) ? null : lid; render(); }
document.getElementById('resetLeafBtn').onclick = () => { focusLeaf = null; render(); };
['chkPoints','chkSurface','chkMidrib','chkStem','chkLabels'].forEach(id => {
  document.getElementById(id).onchange = render;
});
renderButtons();
render();
</script>
</body>
</html>
"""


def main():
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description="Build 3-factor classification 3D comparison viewer")
    p.add_argument("--root", type=Path, default=ROOT / "outputs/maize_v2")
    p.add_argument("--traits-csv", type=Path, default=ROOT / "outputs/phenotypes_v2/leaf_traits.csv")
    p.add_argument("--old-cls-csv", type=Path, default=ROOT / "outputs/classification_v2/leaf_classifications.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/classification_v2/viewer_3factor")
    args = p.parse_args()

    traits = list(csv.DictReader(args.traits_csv.open("r", encoding="utf-8-sig")))
    old_cls = viz_mod.load_classifications(args.old_cls_csv)
    by_plant = defaultdict(list)
    for r in traits:
        by_plant[r["plant_id"]].append(r)

    records = []
    for pid, prows in sorted(by_plant.items()):
        prows_sorted = sorted(prows, key=lambda x: float(x["norm_base_height"]), reverse=True)
        n = len(prows_sorted)
        max_ah = max(float(x["projected_area_element"]) for x in prows_sorted)
        max_av = max(float(x["canopy_visible_nadir_area"]) for x in prows_sorted)
        for rank_top, r in enumerate(prows_sorted, start=1):
            lid = int(r["leaf_id"])
            m = MaizeLeafModel.load(args.root / f"plant_{pid}" / "leaves" / f"leaf_{lid:02d}_model.npz")
            h_pit, js_bal, js_erect, js_droop, mean_pitch = _pitch_entropy_and_jsd(m)
            old_rec = old_cls.get((pid, lid), {})
            records.append(dict(
                plant_id=pid,
                leaf_id=lid,
                rank_top=rank_top,
                rank_bot=n - rank_top + 1,
                h=float(r["norm_base_height"]),
                theta=float(r["stem_leaf_angle_deg"]),
                ah=float(r["projected_area_element"]),
                ah_rel=float(r["projected_area_element"]) / max(max_ah, 1e-12),
                av=float(r["canopy_visible_nadir_area"]),
                av_rel=float(r["canopy_visible_nadir_area"]) / max(max_av, 1e-12),
                rho=float(r["projection_ratio"]),
                H_N=float(r["normal_entropy_b24"]),
                H_pit=h_pit,
                js_bal=js_bal,
                js_erect=js_erect,
                js_droop=js_droop,
                mean_pitch=mean_pitch,
                g=float(r["tip_gain"]),
                d=float(r["droop_ratio"]),
                m4_old_kmeans=old_rec.get("cluster_kmeans", ""),
            ))

    # Run all 4 new schemes
    scheme_reports = {}
    schemes_for_js = []
    for sch in SCHEMES:
        if sch["algo"] == "existing":
            labels = [r["m4_old_kmeans"] for r in records]
            cnts = {k: int(sum(1 for lb in labels if lb == k)) for k in CLASS_NAMES}
            top_c0 = float(np.mean([labels[i] == "convergent" for i, r in enumerate(records) if r["rank_top"] <= 2]))
            mid_c1 = float(np.mean([labels[i] == "expanded" for i, r in enumerate(records) if r["rank_top"] >= 3 and r["rank_bot"] >= 2]))
            bot_c2 = float(np.mean([labels[i] == "drooping" for i, r in enumerate(records) if r["rank_bot"] == 1]))
            err_up = int(sum(1 for i, r in enumerate(records) if r["g"] > 0.25 and r["d"] < 0.05 and labels[i] == "drooping"))
            err_dr = int(sum(1 for i, r in enumerate(records) if r["g"] < -0.20 and r["d"] > 0.30 and labels[i] == "convergent"))
            rep = dict(
                counts=cnts, oof_ari=0.4631, fold_ari=0.6916, silhouette=0.2641,
                top12_convergent=top_c0, mid_expanded=mid_c1, bot1_drooping=bot_c2,
                err_upward_as_drooping=err_up, err_drooping_as_convergent=err_dr,
            )
            scheme_reports[sch["key"]] = rep
        else:
            res = _fit_and_eval_scheme(records, sch, seed=42)
            for i, r in enumerate(records):
                r[sch["key"]] = res["labels"][i]
            rep = {k: v for k, v in res.items() if k not in ("labels", "distance_ratio")}
            scheme_reports[sch["key"]] = rep
        schemes_for_js.append(dict(
            key=sch["key"],
            short=sch["short"],
            label=sch["label"],
            note=sch["note"],
            counts=rep["counts"],
            oof_ari=rep["oof_ari"],
            fold_ari=rep["fold_ari"],
            silhouette=rep["silhouette"],
        ))

    # Save comparison CSV + report JSON
    csv_out = args.out_dir.parent / "compare_3factor_leaves.csv"
    csv_fields = [
        "plant_id", "leaf_id", "rank_top", "h", "theta", "ah", "ah_rel", "av", "av_rel",
        "rho", "H_pit", "js_bal", "mean_pitch", "g", "d", "H_N",
        "f1_4factor_km", "f1_4factor_fcm", "c2_s123_fusion", "s1_rho_hpit", "s2_ahrel_hpit", "m4_old_kmeans",
    ]
    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in csv_fields})

    rep_out = args.out_dir.parent / "compare_3factor_report.json"
    rep_out.write_text(json.dumps(scheme_reports, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export offline 3D comparison viewer
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(args.out_dir)
    rec_by_key = {(r["plant_id"], r["leaf_id"]): r for r in records}
    catalog = viz_mod.scan_catalog(args.root)
    plants = list(catalog.keys())

    plant_rows_html = []
    for idx, pid in enumerate(plants):
        scene = viz_mod.load_scene_from_outputs(args.root, pid, leaf=None)
        traces = []
        if scene["stem"] is not None:
            s = np.round(scene["stem"], 5)
            traces.append(dict(
                type="scatter3d", mode="lines", name="茎秆中心线", group="stem",
                x=s[:, 0].tolist(), y=s[:, 1].tolist(), z=s[:, 2].tolist(),
                line=dict(color="#4b5563", width=7), hoverinfo="name",
            ))

        leaves_meta = []
        lx, ly, lz = [], [], []
        pt_limit = max(1800, 15000 // max(1, len(scene["leaves"])))

        for item in scene["leaves"]:
            lid = item["leaf_id"]
            rec = rec_by_key[(pid, lid)]
            s123 = {rec["f1_4factor_km"], rec["c2_s123_fusion"], rec["s1_rho_hpit"], rec["s2_ahrel_hpit"]}
            s123_diff = len(s123) > 1
            leaves_meta.append(dict(
                leaf_id=lid,
                rank_top=rec["rank_top"],
                h=rec["h"],
                theta=rec["theta"],
                rho=rec["rho"],
                ah_rel=rec["ah_rel"],
                av_rel=rec["av_rel"],
                H_pit=rec["H_pit"],
                js_bal=rec["js_bal"],
                mean_pitch=rec["mean_pitch"],
                g=rec["g"],
                d=rec["d"],
                s123_diff=s123_diff,
                labels={sch["key"]: rec[sch["key"]] for sch in SCHEMES},
            ))
            if item["points"] is not None and len(item["points"]):
                pts, rgb = viz_mod.sample_points(item["points"], item["rgb"], pt_limit)
                pts = np.round(pts, 5)
                colors = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb] if rgb is not None else "#334c42"
                traces.append(dict(
                    type="scatter3d", mode="markers", name=f"L{lid} 点云", group="points", leaf_id=lid,
                    x=pts[:, 0].tolist(), y=pts[:, 1].tolist(), z=pts[:, 2].tolist(),
                    marker=dict(size=2, color=colors, opacity=0.42), showlegend=False,
                ))
            if item["midrib"] is not None and len(item["midrib"]):
                m = np.round(item["midrib"], 5)
                traces.append(dict(
                    type="scatter3d", mode="lines", name=f"L{lid} 中脉", group="midrib", leaf_id=lid,
                    x=m[:, 0].tolist(), y=m[:, 1].tolist(), z=m[:, 2].tolist(),
                    line=dict(color="#f59e0b" if s123_diff else "#1f2937", width=6 if s123_diff else 4),
                    showlegend=False,
                ))
                pt_lbl = m[min(int(len(m) * 0.58), len(m) - 1)]
                lx.append(float(pt_lbl[0]))
                ly.append(float(pt_lbl[1]))
                lz.append(float(pt_lbl[2]))
            if item["vertices"] is not None and len(item["faces"]):
                v = np.round(item["vertices"], 5)
                f = item["faces"]
                traces.append(dict(
                    type="mesh3d", name=f"L{lid}", group="surface", leaf_id=lid,
                    x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
                    i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
                    color="rgb(27,158,119)", opacity=0.90, flatshading=False,
                    lighting=dict(ambient=0.58, diffuse=0.82, specular=0.18, roughness=0.55),
                    showlegend=True,
                ))

        if lx:
            traces.append(dict(
                type="scatter3d", mode="markers+text", name="叶片标签", group="labels",
                x=lx, y=ly, z=lz, text=[], textposition="top center",
                textfont=dict(size=12, color=[], family="Microsoft YaHei, sans-serif"),
                marker=dict(size=4, color=[], line=dict(color="#ffffff", width=1)),
                hovertext=[], hoverinfo="text", showlegend=False,
            ))

        payload = dict(plant=pid, traces=traces, leaves_meta=leaves_meta)
        prev_p = plants[(idx - 1) % len(plants)]
        next_p = plants[(idx + 1) % len(plants)]
        html = (
            PLANT_HTML_TEMPLATE
            .replace("__PLANT__", pid)
            .replace("__PREV_HREF__", f"plant_{prev_p}.html")
            .replace("__NEXT_HREF__", f"plant_{next_p}.html")
            .replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
            .replace("__SCHEMES_JSON__", json.dumps(schemes_for_js, ensure_ascii=False))
        )
        (args.out_dir / f"plant_{pid}.html").write_text(html, encoding="utf-8")

        def _fmt_cnt(k):
            c = {nm: sum(1 for lm in leaves_meta if lm["labels"][k] == nm) for nm in CLASS_NAMES}
            return f"{c['convergent']} / {c['expanded']} / {c['drooping']}"

        n_diff = sum(1 for lm in leaves_meta if lm["s123_diff"])
        plant_rows_html.append(
            f'<tr><td><a href="plant_{pid}.html"><b>{pid}</b></a></td><td>{len(leaves_meta)}</td>'
            f'<td><b>{_fmt_cnt("f1_4factor_km")}</b></td><td>{_fmt_cnt("f1_4factor_fcm")}</td>'
            f'<td>{_fmt_cnt("c2_s123_fusion")}</td><td>{_fmt_cnt("s1_rho_hpit")}</td>'
            f'<td>{_fmt_cnt("s2_ahrel_hpit")}</td><td>{_fmt_cnt("m4_old_kmeans")}</td><td>{n_diff}</td></tr>'
        )

    sch_rows_html = []
    for sch in SCHEMES:
        r = scheme_reports[sch["key"]]
        c = r["counts"]
        sch_rows_html.append(
            f'<tr><td><b>{sch["short"]}</b></td><td>{sch["label"].split("：")[1]}</td>'
            f'<td><span style="color:#1f77b4;font-weight:600">{c["convergent"]}</span> / '
            f'<span style="color:#1b9e77;font-weight:600">{c["expanded"]}</span> / '
            f'<span style="color:#d95f02;font-weight:600">{c["drooping"]}</span></td>'
            f'<td><b>{r["oof_ari"]:.3f}</b></td><td>{r["silhouette"]:.2f}</td>'
            f'<td>{r["top12_convergent"]*100:.0f}% / {r["mid_expanded"]*100:.0f}% / {r["bot1_drooping"]*100:.0f}%</td>'
            f'<td>{r["err_upward_as_drooping"]} / {r["err_drooping_as_convergent"]}</td></tr>'
        )

    index_html = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三因素叶片姿态分类 · 四方案三维交互对比</title>
<style>
  body{{margin:0;background:#f3f6f5;color:#16372e;font:14.5px/1.6 system-ui,'Microsoft YaHei',sans-serif}}
  header{{padding:18px 26px;background:#113e33;color:#fff}}h1{{font-size:20px;margin:0 0 6px}}
  main{{max-width:1360px;margin:auto;padding:18px 24px}}a{{color:#16715b;text-decoration:none}}
  a:hover{{text-decoration:underline}}.card{{background:#fff;border-radius:12px;padding:16px 18px;margin-bottom:16px;border:1px solid #dce7e2}}
  table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #dce7e2}}
  th{{background:#e8f1ec}}
</style>
<body>
<header>
  <h1>茎叶角 + 垂直投影面积 + 信息熵 · 三维交互方案对比（S1–S4 vs 旧版 M4）</h1>
  <div>点击下方任意植株进入 3D 页面，顶部一键实时切换 S1 / S2 / S3 / S4 / 旧 M4 配色（保持三维视角不动），肉眼对比每片叶归类</div>
</header>
<main>
  <div class="card">
    <h2 style="margin:0 0 10px;font-size:16px">各方案全局指标对比（90 片叶，14 株 LOPO）</h2>
    <table>
      <tr><th>代号</th><th>特征与算法</th><th>紧缩 / 舒展 / 下垂</th><th>LOPO ARI</th><th>Silhouette</th><th>顶1-2→紧 / 中→舒 / 底1→垂</th><th>严重反例(上→垂/垂→紧)</th></tr>
      {"".join(sch_rows_html)}
    </table>
  </div>
  <div class="card">
    <h2 style="margin:0 0 10px;font-size:16px">逐株进入 3D 交互对比（表格数字均为 紧缩 / 舒展 / 下垂 叶数）</h2>
    <table>
      <tr><th>植株（点击打开 3D）</th><th>叶数</th><th>F1 (4因素KM)</th><th>F1 (4因素FCM)</th><th>C2 (3因素融合)</th><th>S1 (θ,ρ,H_pit)</th><th>S2 (θ,Ah_rel,H_pit)</th><th>旧版 M4</th><th>方案分歧叶</th></tr>
      {"".join(plant_rows_html)}
    </table>
  </div>
</main>
</body></html>"""
    (args.out_dir / "index.html").write_text(index_html, encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"DONE in {elapsed:.2f}s -> {(args.out_dir / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
