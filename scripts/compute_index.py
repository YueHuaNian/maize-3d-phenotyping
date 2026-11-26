"""阶段 F（主入口）：计算五因素联动理想株型结构符合度指数（I_A / I_L / I）与四项方法学检验。

功能说明：
    1. 读取只读表型目录 outputs/phenotypes_v2/（leaf_traits.csv、plant_traits.csv、plant_envelopes.csv）
       与阶段 3 分类结果 outputs/classification_v2/leaf_classifications.csv；
    2. 计算 90 片单叶的五因素理想符合度 I_L（0-100，基于茎叶角 θ、相对投影面积 P、俯仰熵 H、
       相对着生高度 h 条件目标与无量纲远端高差 Δz/L_obs）及 14 株整株构型符合度 I_A（25 层包络
       S_env + 5 方向受光 S_light + 五因素三类比例结构 S_comp）与综合指数 I = 0.5 I_A + 0.5 I_L；
    3. 执行四项方法学验证：单调性检验、蒙特卡洛测量/遮挡扰动与待查叶剔除对照、植株级 Bootstrap
       与株内叶片重抽样/LOLO、权重/Dirichlet/目标组成/留一分项敏感性；
    4. 输出逐叶/逐株 CSV、完整溯源报告 index_report.json 与独立网页报告 index_analysis.html。

常用命令：
    python scripts/compute_index.py
    python scripts/compute_index.py --output outputs/index_v2 --n-trials 200 --n-boot 500
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import scipy

from maize_leaf.ideotype import (
    DEFAULT_ALPHA,
    DEFAULT_ARCH_TARGETS,
    DEFAULT_ARCH_WEIGHTS,
    DEFAULT_LEAF_TARGETS,
    DEFAULT_LEAF_WEIGHTS,
    INDEX_SCHEMA_VERSION,
    compute_leaf_ideotype_scores,
    compute_plant_ideotype_scores,
    run_bootstrap_validation,
    run_monotonicity_validation,
    run_perturbation_validation,
    run_sensitivity_validation,
)


def _sha256(path: Path) -> str:
    """计算单个文件的 SHA-256 摘要用于溯源记录。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    """按 UTF-8（兼容 BOM）读取 CSV 为字典列表。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """按固定字段顺序写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def load_joined_inputs(phenotypes_dir: Path, classification_csv: Path):
    """读取并按 (plant_id, leaf_id) 严格内连接表型表与五因素分类表。"""
    leaf_traits_path = phenotypes_dir / "leaf_traits.csv"
    plant_traits_path = phenotypes_dir / "plant_traits.csv"
    envelopes_path = phenotypes_dir / "plant_envelopes.csv"
    for p in (leaf_traits_path, plant_traits_path, envelopes_path, classification_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Required input file not found: {p}")

    leaf_traits = _read_csv(leaf_traits_path)
    plant_traits = _read_csv(plant_traits_path)
    envelopes = _read_csv(envelopes_path)
    cls_rows = _read_csv(classification_csv)

    cls_map = {}
    for r in cls_rows:
        key = (str(r["plant_id"]).strip(), int(r["leaf_id"]))
        if key in cls_map:
            raise ValueError(f"Duplicate classification key: {key}")
        cls_map[key] = r

    joined_leaves = []
    for r in leaf_traits:
        key = (str(r["plant_id"]).strip(), int(r["leaf_id"]))
        if key not in cls_map:
            raise ValueError(f"Missing classification record for leaf {key}")
        merged = dict(r)
        crec = cls_map[key]
        merged["cluster_kmeans"] = crec.get("cluster_kmeans", "").strip()
        merged["relative_height"] = crec.get("relative_height", r.get("stem_rel_height", r.get("norm_base_height")))
        joined_leaves.append(merged)

    hashes = {
        "leaf_traits.csv": _sha256(leaf_traits_path),
        "plant_traits.csv": _sha256(plant_traits_path),
        "plant_envelopes.csv": _sha256(envelopes_path),
        "leaf_classifications.csv": _sha256(classification_csv),
    }
    return plant_traits, envelopes, joined_leaves, hashes


def write_analysis_html(out_path: Path, plant_scores: list[dict], report: dict) -> None:
    """生成可直接离线打开的阶段 F 指数与四项验证汇总网页。"""
    sorted_plants = sorted(plant_scores, key=lambda r: r["rank"])
    rows_html = []
    for r in sorted_plants:
        rows_html.append(
            f"<tr><td>#{r['rank']}</td><td><b>{escape(r['plant_id'])}</b></td><td>{r['leaf_count']}</td>"
            f"<td><b>{r['composite_index_i']:.2f}</b></td>"
            f"<td>{r['arch_index_ia']:.2f}</td><td>{r['leaf_index_il']:.2f}</td>"
            f"<td>{100*r['s_envelope']:.1f}</td><td>{100*r['s_light']:.1f}</td><td>{100*r['s_composition']:.1f}</td>"
            f"<td>{100*r['mean_s_angle']:.1f} / {100*r['mean_s_proj']:.1f} / {100*r['mean_s_entropy']:.1f} / {100*r['mean_s_far_dz']:.1f}</td>"
            f"<td>{r['support_review_leaf_count']}</td></tr>"
        )

    mono = report["validations"]["monotonicity"]
    pert = report["validations"]["perturbation"]
    boot = report["validations"]["bootstrap"]
    sens = report["validations"]["sensitivity"]

    alpha_rows = "".join(
        f"<tr><td>{a['alpha']:.2f}</td><td>{a['spearman_vs_primary']:.3f}</td>"
        f"<td>{a['kendall_vs_primary']:.3f}</td><td>{a['top3_overlap']:.2f}</td>"
        f"<td>{a['bottom3_overlap']:.2f}</td><td>{a['mean_i']:.2f} ± {a['std_i']:.2f}</td></tr>"
        for a in sens["alpha_sweep"]
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>阶段 F · 玉米理想株型结构符合度指数报告</title>
<style>
body{{margin:0;background:#f4f7f5;color:#17352b;font:14px/1.6 system-ui,'Microsoft YaHei',sans-serif}}
header{{background:#123f33;color:#fff;padding:20px 26px}}
h1{{margin:0;font-size:22px}}h2{{font-size:16px;margin:0 0 10px}}
main{{max-width:1180px;margin:auto;padding:22px}}
.card{{background:#fff;border:1px solid #d9e4de;border-radius:12px;padding:18px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:14px}}
.metric{{background:#f7faf8;border:1px solid #e1ebe5;border-radius:10px;padding:12px}}
.metric b{{display:block;font-size:20px;color:#14523d}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 10px;border-bottom:1px solid #e5ece8;text-align:left}}
th{{background:#f6faf7;font-size:13px}}
.note{{font-size:12px;color:#556b60}}
</style></head><body>
<header>
<h1>玉米理想株型结构符合度指数（I_A / I_L / I）与四项方法学检验</h1>
<div>五因素联动方案 · 14 株 / 90 片叶 · 固定等权主方案（α = {report['parameters']['alpha']:.2f}）</div>
</header>
<main>
<div class="grid">
<div class="metric"><span>综合指数 I 中位数 (范围)</span><b>{report['summary']['composite_i_median']:.2f} ({report['summary']['composite_i_min']:.1f}–{report['summary']['composite_i_max']:.1f})</b></div>
<div class="metric"><span>1. 单调性检验</span><b>{'全部通过' if mono['passed'] else '未通过'} ({mono['n_steps']} 步)</b></div>
<div class="metric"><span>2. 扰动鲁棒性 (200 次)</span><b>ρ = {pert['monte_carlo']['spearman_i_median']:.3f} (min {pert['monte_carlo']['spearman_i_min']:.3f})</b></div>
<div class="metric"><span>3. 株内叶片 Bootstrap</span><b>ρ = {boot['within_plant_leaf_bootstrap']['spearman_i_median']:.3f} (LOLO min {boot['leave_one_leaf_out']['spearman_i_min']:.3f})</b></div>
<div class="metric"><span>4. Dirichlet 权重扰动</span><b>ρ = {sens['dirichlet_random_weights']['spearman_i_median']:.3f} (min {sens['dirichlet_random_weights']['spearman_i_min']:.3f})</b></div>
</div>
<div class="card">
<h2>14 株排名与分项符合度（0–100）</h2>
<table>
<tr><th>排名</th><th>植株</th><th>叶数</th><th>综合 I</th><th>构型 I_A</th><th>叶形 I_L</th><th>包络</th><th>受光</th><th>类别比例</th><th>单叶四分项 (θ/P/H/Δz)</th><th>支撑待查叶</th></tr>
{''.join(rows_html)}
</table>
<p class="note">注：I = 0.5 I_A + 0.5 I_L；单叶四分项基于相对着生高度 h 条件下的理想目标计算。本指数衡量相对预设智能冠层几何参考的符合程度，不代表产量、光合速率或发育期标定结论。</p>
</div>
<div class="card">
<h2>主权重 α 敏感性扫描</h2>
<table>
<tr><th>α (I_A 权重)</th><th>Spearman ρ</th><th>Kendall τ</th><th>Top-3 重合</th><th>Bottom-3 重合</th><th>I 均值 ± 标准差</th></tr>
{alpha_rows}
</table>
</div>
<div class="card">
<h2>稳健性对照与方法边界</h2>
<ul>
<li>面积加权 I_L 对照：Spearman ρ(I) = {pert['structural_comparisons']['aggregation_area']['spearman_i']:.3f}，最大分值差 {pert['structural_comparisons']['aggregation_area']['max_abs_diff_i']:.2f}；</li>
<li>质量加权 I_L 对照：Spearman ρ(I) = {pert['structural_comparisons']['aggregation_quality']['spearman_i']:.3f}，最大分值差 {pert['structural_comparisons']['aggregation_quality']['max_abs_diff_i']:.2f}；</li>
<li>剔除支撑复查叶（{pert['structural_comparisons']['exclude_support_review_leaves']['excluded_leaves']} 片）对照：Spearman ρ(I) = {pert['structural_comparisons']['exclude_support_review_leaves']['spearman_i']:.3f}，最大分值差 {pert['structural_comparisons']['exclude_support_review_leaves']['max_abs_diff_i']:.2f}；</li>
<li>植株级 Bootstrap（独立单位 n=14，{boot['n_boot']} 次）：群体 I 均值 95% CI [{boot['plant_level_bootstrap']['mean_i_ci95'][0]:.2f}, {boot['plant_level_bootstrap']['mean_i_ci95'][1]:.2f}]，中位数 95% CI [{boot['plant_level_bootstrap']['median_i_ci95'][0]:.2f}, {boot['plant_level_bootstrap']['median_i_ci95'][1]:.2f}]。</li>
</ul>
</div>
</main></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    """命令行入口：执行指数计算、四项检验并落盘结果。"""
    t0 = time.perf_counter()
    p = argparse.ArgumentParser(description="Compute five-factor linked ideotype conformity indices (I_A, I_L, I) and 4 validations")
    p.add_argument("--phenotypes-dir", type=Path, default=ROOT / "outputs" / "phenotypes_v2")
    p.add_argument("--classification-csv", type=Path, default=ROOT / "outputs" / "classification_v2" / "leaf_classifications.csv")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "index_v2")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--n-trials", type=int, default=200, help="Monte Carlo perturbation trials")
    p.add_argument("--n-boot", type=int, default=500, help="Bootstrap resamples")
    p.add_argument("--n-dirichlet", type=int, default=100, help="Random Dirichlet weight draws")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save-snapshot", type=Path, help="Optional validation snapshot JSON path (e.g. docs/validation/ideotype_index_2026-11-26.json)")
    args = p.parse_args()

    plant_rows, env_rows, leaf_rows, input_hashes = load_joined_inputs(args.phenotypes_dir, args.classification_csv)

    leaf_scores = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=DEFAULT_LEAF_TARGETS, leaf_weights=DEFAULT_LEAF_WEIGHTS)
    plant_scores = compute_plant_ideotype_scores(
        plant_rows, env_rows, leaf_scores,
        arch_targets=DEFAULT_ARCH_TARGETS, arch_weights=DEFAULT_ARCH_WEIGHTS,
        alpha=args.alpha, aggregation="equal",
    )

    mono = run_monotonicity_validation(
        leaf_targets=DEFAULT_LEAF_TARGETS, arch_targets=DEFAULT_ARCH_TARGETS,
        leaf_weights=DEFAULT_LEAF_WEIGHTS, arch_weights=DEFAULT_ARCH_WEIGHTS, alpha=args.alpha,
    )
    pert = run_perturbation_validation(
        plant_rows, env_rows, leaf_rows,
        leaf_targets=DEFAULT_LEAF_TARGETS, arch_targets=DEFAULT_ARCH_TARGETS,
        leaf_weights=DEFAULT_LEAF_WEIGHTS, arch_weights=DEFAULT_ARCH_WEIGHTS,
        alpha=args.alpha, n_trials=args.n_trials, seed=args.seed,
    )
    boot = run_bootstrap_validation(
        plant_rows, env_rows, leaf_rows,
        leaf_targets=DEFAULT_LEAF_TARGETS, arch_targets=DEFAULT_ARCH_TARGETS,
        leaf_weights=DEFAULT_LEAF_WEIGHTS, arch_weights=DEFAULT_ARCH_WEIGHTS,
        alpha=args.alpha, n_boot=args.n_boot, seed=args.seed,
    )
    sens = run_sensitivity_validation(
        plant_rows, env_rows, leaf_rows,
        leaf_targets=DEFAULT_LEAF_TARGETS, arch_targets=DEFAULT_ARCH_TARGETS,
        leaf_weights=DEFAULT_LEAF_WEIGHTS, arch_weights=DEFAULT_ARCH_WEIGHTS,
        alpha=args.alpha, n_dirichlet=args.n_dirichlet, seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    leaf_fields = [
        "plant_id", "leaf_id", "cluster_kmeans", "relative_height",
        "stem_leaf_angle_deg", "rel_projected_area", "pitch_entropy_b12",
        "far_end_height_diff", "observed_length", "far_dz_norm",
        "theta_target_deg", "proj_target", "entropy_target", "far_dz_norm_target",
        "s_angle", "s_proj", "s_entropy", "s_far_dz", "leaf_index_il",
        "surface_area", "quality_weight", "support_review_required",
    ]
    plant_fields = [
        "rank", "plant_id", "leaf_count", "support_review_leaf_count",
        "composite_index_i", "arch_index_ia", "leaf_index_il",
        "leaf_index_il_equal", "leaf_index_il_area", "leaf_index_il_quality",
        "s_envelope", "s_light", "s_composition",
        "envelope_l1_error", "envelope_valid_bins",
        "light_non_overlap", "light_slant_retention",
        "composition_tv_distance", "frac_convergent", "frac_expanded", "frac_drooping",
        "mean_s_angle", "mean_s_proj", "mean_s_entropy", "mean_s_far_dz",
    ]
    _write_csv(args.output / "leaf_ideotype_index.csv", leaf_scores, leaf_fields)
    _write_csv(args.output / "plant_ideotype_index.csv", sorted(plant_scores, key=lambda r: r["rank"]), plant_fields)

    i_arr = np.array([r["composite_index_i"] for r in plant_scores], dtype=float)
    ia_arr = np.array([r["arch_index_ia"] for r in plant_scores], dtype=float)
    il_arr = np.array([r["leaf_index_il"] for r in plant_scores], dtype=float)
    leaf_il_arr = np.array([r["leaf_index_il"] for r in leaf_scores], dtype=float)

    report = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "n_plants": len(plant_scores),
        "n_leaves": len(leaf_scores),
        "seed": int(args.seed),
        "input_paths": {
            "phenotypes_dir": str(args.phenotypes_dir),
            "classification_csv": str(args.classification_csv),
        },
        "input_sha256": input_hashes,
        "source_sha256": {
            "src/maize_leaf/ideotype.py": _sha256(ROOT / "src" / "maize_leaf" / "ideotype.py"),
            "scripts/compute_index.py": _sha256(ROOT / "scripts" / "compute_index.py"),
        },
        "parameters": {
            "alpha": float(args.alpha),
            "leaf_aggregation": "equal",
            "leaf_targets": DEFAULT_LEAF_TARGETS,
            "leaf_weights": DEFAULT_LEAF_WEIGHTS,
            "arch_targets": DEFAULT_ARCH_TARGETS,
            "arch_weights": DEFAULT_ARCH_WEIGHTS,
        },
        "summary": {
            "composite_i_mean": float(np.mean(i_arr)),
            "composite_i_std": float(np.std(i_arr)),
            "composite_i_median": float(np.median(i_arr)),
            "composite_i_min": float(np.min(i_arr)),
            "composite_i_max": float(np.max(i_arr)),
            "arch_ia_median": float(np.median(ia_arr)),
            "arch_ia_min": float(np.min(ia_arr)),
            "arch_ia_max": float(np.max(ia_arr)),
            "plant_il_median": float(np.median(il_arr)),
            "plant_il_min": float(np.min(il_arr)),
            "plant_il_max": float(np.max(il_arr)),
            "leaf_il_median": float(np.median(leaf_il_arr)),
            "leaf_il_q25": float(np.quantile(leaf_il_arr, 0.25)),
            "leaf_il_q75": float(np.quantile(leaf_il_arr, 0.75)),
            "top3_plants": [r["plant_id"] for r in sorted(plant_scores, key=lambda r: r["rank"])[:3]],
            "bottom3_plants": [r["plant_id"] for r in sorted(plant_scores, key=lambda r: r["rank"])[-3:]],
        },
        "plant_rankings": sorted(plant_scores, key=lambda r: r["rank"]),
        "validations": {
            "monotonicity": mono,
            "perturbation": pert,
            "bootstrap": boot,
            "sensitivity": sens,
        },
        "interpretation_boundaries": [
            "Indices quantify geometric conformity to a pre-specified smart-canopy ideotype target, not calibrated yield or photosynthetic capacity.",
            "All target profiles, tolerances, and equal weights are fixed development constants, not empirically fitted to the 14 plants.",
            "far_end_height_diff is normalized by observed_length (far_dz / L_obs) so I_L is invariant to global coordinate scaling.",
            "14 plants are the independent sampling units; 90 leaves are nested within plants.",
            "External ground truth (manual morphology/developmental labels and physical unit calibration) remains required for biological validation.",
        ],
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    (args.output / "index_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_analysis_html(args.output / "index_analysis.html", plant_scores, report)

    if args.save_snapshot:
        # 精简快照（去掉大数组 trajectory/samples）写入验证快照路径
        snap = json.loads(json.dumps(report))
        snap["validations"]["monotonicity"].pop("trajectory", None)
        snap["validations"]["perturbation"]["monte_carlo"].pop("spearman_i_samples", None)
        snap["validations"]["sensitivity"]["dirichlet_random_weights"].pop("spearman_i_samples", None)
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.save_snapshot.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"DONE {args.output} | plants={len(plant_scores)}, leaves={len(leaf_scores)} | "
        f"I med={report['summary']['composite_i_median']:.2f} ({report['summary']['composite_i_min']:.1f}-{report['summary']['composite_i_max']:.1f}) | "
        f"mono={'PASS' if mono['passed'] else 'FAIL'} | "
        f"pert_rho={pert['monte_carlo']['spearman_i_median']:.3f} | "
        f"leaf_boot_rho={boot['within_plant_leaf_bootstrap']['spearman_i_median']:.3f} | "
        f"dirichlet_rho={sens['dirichlet_random_weights']['spearman_i_median']:.3f} | "
        f"elapsed={report['elapsed_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
