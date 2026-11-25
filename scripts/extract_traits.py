"""阶段 2（主入口）：提取单叶与整株三维几何表型、方向信息熵及冠层径向包络。

功能说明：
    只读加载阶段 1 产出的连续叶片模型（outputs/maize_v2/plant_*/leaves/leaf_*_model.npz）
    与茎秆样条中心线，计算并导出：
    - leaf_traits.csv       ：逐叶表型表（叶长、叶宽、表面积、投影面积与同株相对投影面积、
                              基部茎叶角、叶尖抬升/下垂比、中脉与曲面曲率、俯仰/法向熵、
                              相对着生高度、遮挡比及质量权重）；
    - plant_traits.csv      ：逐株汇总表（株高、茎高、总叶面积、冠层并集投影、重叠指数、
                              五方向 Z-buffer 可见面积与质量加权平均茎叶角）；
    - plant_envelopes.csv   ：逐株按高度分箱的 95% 分位冠层径向包络 R(h)；
    - traits_metadata.json  ：参数口径、源码哈希、质量权重尺度 tau 与分箱/窗口稳定性统计。

常用命令：
    python scripts/extract_traits.py
    python scripts/extract_traits.py --skip-sensitivity  # 跳过耗时的边距/基部窗口敏感性复算
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]

import numpy as np
from scipy.stats import spearmanr

from maize_leaf.io import write_json
from maize_leaf.surface import MaizeLeafModel
from maize_leaf.traits import (
    LIGHT_DIRECTIONS,
    attach_quality_weights,
    basal_surface_stem_leaf_angle,
    extract_plant_phenotypes,
    load_stem_spline,
    surface_curvature_fundamental_forms,
)


def _sha256(path):
    """分块读取文件并计算 SHA-256，记录输入来源。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_margin_trim_sensitivity(model_root, trims=(0.025, 0.05, 0.10)):
    """检验曲面曲率（平均曲率与 Willmore 弯曲泛函）在不同侧缘裁切比例下的秩稳定性。"""
    model_paths = sorted(Path(model_root).glob("plant_*/leaves/leaf_*_model.npz"))
    results = {t: {"h_abs": [], "willmore": []} for t in trims}
    for mp in model_paths:
        m = MaizeLeafModel.load(mp)
        for t in trims:
            c = surface_curvature_fundamental_forms(m, margin_trim_frac=t)
            results[t]["h_abs"].append(c["mean_curvature_abs"])
            results[t]["willmore"].append(c["willmore_bending_functional"])

    ref = 0.05
    summary = {}
    for t in trims:
        r_h = float(spearmanr(results[ref]["h_abs"], results[t]["h_abs"], nan_policy="omit").statistic)
        r_w = float(spearmanr(results[ref]["willmore"], results[t]["willmore"], nan_policy="omit").statistic)
        summary[f"trim_{t:g}"] = {
            "med_mean_curvature_abs": float(np.nanmedian(results[t]["h_abs"])),
            "med_willmore": float(np.nanmedian(results[t]["willmore"])),
            "spearman_vs_5pct_h_abs": r_h,
            "spearman_vs_5pct_willmore": r_w,
        }
    return summary


def run_basal_span_sensitivity(model_root, spans=(0.05, 0.10, 0.15)):
    """检验基部叶面茎叶角在不同纵向基部跨度（5%/10%/15%）下的中位偏移与秩稳定性。"""
    results = {sp: [] for sp in spans}
    for pdir in sorted(p for p in Path(model_root).glob("plant_*") if p.is_dir()):
        stem_c, stem_b, stem_z, _ = load_stem_spline(pdir / "stem_centerline.json")
        for mp in sorted((pdir / "leaves").glob("leaf_*_model.npz")):
            m = MaizeLeafModel.load(mp)
            for sp in spans:
                res = basal_surface_stem_leaf_angle(m, stem_c, stem_b, stem_z, basal_span_frac=sp)
                results[sp].append(res["stem_leaf_angle_deg"])
    ref = 0.10
    ref_arr = np.asarray(results[ref], dtype=float)
    summary = {}
    for sp in spans:
        arr = np.asarray(results[sp], dtype=float)
        summary[f"span_{int(round(sp * 100))}pct"] = {
            "med_angle_deg": float(np.median(arr)),
            "med_abs_diff_vs_10pct_deg": float(np.median(np.abs(arr - ref_arr))),
            "spearman_vs_10pct": float(spearmanr(ref_arr, arr).statistic),
        }
    return summary


def main():
    # 解析命令行参数（默认只读 outputs/maize_v2，写入 outputs/phenotypes_v2）
    """读取冻结模型并导出逐叶表型、整株表型和窗口敏感性结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "outputs/maize_v2", help="阶段 1 重建模型根目录")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phenotypes_v2", help="表型 CSV 与元数据输出目录")
    parser.add_argument("--no-entropy", action="store_true", help="关闭俯仰角与法向信息熵计算")
    parser.add_argument("--margin-trim", type=float, default=0.05, help="曲率积分时单侧剔除的横向边缘比例")
    parser.add_argument("--basal-span-frac", type=float, default=0.15, help="计算基部茎叶角使用的叶基纵向跨度比例（已标定最优值 0.15）")
    parser.add_argument("--zbuf-resolution", type=int, default=360, help="五方向光遮挡 Z-buffer 光栅化主轴分辨率")
    parser.add_argument("--envelope-bins", type=int, default=25, help="冠层径向包络 R(h) 的高度分箱数")
    parser.add_argument("--skip-sensitivity", action="store_true", help="跳过侧缘裁切与基部跨度敏感性诊断以加速运行")
    args = parser.parse_args()

    t0 = time.time()
    plant_dirs = sorted(p for p in args.models_root.glob("plant_*") if p.is_dir())
    if not plant_dirs:
        raise FileNotFoundError(f"No plant_* folders found under {args.models_root}")

    all_leaves = []
    all_plants = []
    all_envelopes = []
    include_entropy = not args.no_entropy

    # 1. 逐株提取单叶表型、整株光遮挡指标与高度分箱径向包络
    for pdir in plant_dirs:
        t_p = time.time()
        leaves, plant_rec, env_rows = extract_plant_phenotypes(
            pdir,
            include_entropy=include_entropy,
            margin_trim_frac=args.margin_trim,
            basal_span_frac=args.basal_span_frac,
            zbuf_resolution=args.zbuf_resolution,
            envelope_bins=args.envelope_bins,
        )
        all_leaves.extend(leaves)
        all_plants.append(plant_rec)
        all_envelopes.extend(env_rows)
        print(
            f"  {plant_rec['plant_id']:8s} | leaves={plant_rec['leaf_count']:2d} "
            f"| H={plant_rec['plant_height']:.3f} | A_tot={plant_rec['total_leaf_area']:.4f} "
            f"| A_union={plant_rec['canopy_union_area']:.4f} | overlap={plant_rec['canopy_overlap_index']:.2f} "
            f"| theta={plant_rec['mean_stem_leaf_angle_deg']:.1f}deg [{time.time() - t_p:.1f}s]",
            flush=True,
        )

    tau = attach_quality_weights(all_leaves)

    # Plant-level quality-weighted averages
    by_plant = {}
    for r in all_leaves:
        by_plant.setdefault(r["plant_id"], []).append(r)
    for pr in all_plants:
        lrs = by_plant[pr["plant_id"]]
        qw = np.array([max(r["quality_weight"], 0.0) for r in lrs], dtype=float)
        wsum = float(np.sum(qw))
        pr["mean_quality_weight"] = float(np.mean(qw))
        pr["qw_stem_leaf_angle_deg"] = (
            float(np.sum(qw * np.array([r["stem_leaf_angle_deg"] for r in lrs])) / wsum)
            if wsum > 0 else pr["mean_stem_leaf_angle_deg"]
        )

    args.output.mkdir(parents=True, exist_ok=True)

    def _write_csv(path, rows):
        """按统一列顺序保存表型或敏感性结果表。"""
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(args.output / "leaf_traits.csv", all_leaves)
    _write_csv(args.output / "plant_traits.csv", all_plants)
    _write_csv(args.output / "plant_envelopes.csv", all_envelopes)

    # Entropy bin-resolution Spearman stability (pitch B=6/12/24 & normal B=12/24/48)
    entropy_stability = {}
    if include_entropy:
        hp6 = [r["pitch_entropy_b6"] for r in all_leaves]
        hp12 = [r["pitch_entropy_b12"] for r in all_leaves]
        hp24 = [r["pitch_entropy_b24"] for r in all_leaves]
        h12 = [r["normal_entropy_b12"] for r in all_leaves]
        h24 = [r["normal_entropy_b24"] for r in all_leaves]
        h48 = [r["normal_entropy_b48"] for r in all_leaves]
        entropy_stability = {
            "pitch_entropy_med_b6": float(np.nanmedian(hp6)),
            "pitch_entropy_med_b12": float(np.nanmedian(hp12)),
            "pitch_entropy_med_b24": float(np.nanmedian(hp24)),
            "pitch_spearman_b6_vs_b12": float(spearmanr(hp6, hp12, nan_policy="omit").statistic),
            "pitch_spearman_b12_vs_b24": float(spearmanr(hp12, hp24, nan_policy="omit").statistic),
            "pitch_spearman_b6_vs_b24": float(spearmanr(hp6, hp24, nan_policy="omit").statistic),
            "med_b12": float(np.nanmedian(h12)),
            "med_b24": float(np.nanmedian(h24)),
            "med_b48": float(np.nanmedian(h48)),
            "spearman_b12_vs_b24": float(spearmanr(h12, h24, nan_policy="omit").statistic),
            "spearman_b24_vs_b48": float(spearmanr(h24, h48, nan_policy="omit").statistic),
            "spearman_b12_vs_b48": float(spearmanr(h12, h48, nan_policy="omit").statistic),
        }

    trim_stability = {}
    basal_stability = {}
    if not args.skip_sensitivity:
        print("Running lateral margin-trim (2.5%/5%/10%) & basal-span (5%/10%/15%) sensitivity...", flush=True)
        trim_stability = run_margin_trim_sensitivity(args.models_root, trims=(0.025, 0.05, 0.10))
        basal_stability = run_basal_span_sensitivity(args.models_root, spans=(0.05, 0.10, 0.15))

    meta = {
        "models_root": str(args.models_root),
        "n_plants": len(all_plants),
        "n_leaves": len(all_leaves),
        "distance_units": "input coordinate units (dimensionless ratios/angles/K_mid/Willmore/entropy are scale-invariant)",
        "parameters": {
            "include_entropy": include_entropy,
            "margin_trim_frac": args.margin_trim,
            "basal_span_frac": args.basal_span_frac,
            "stem_leaf_angle_definition": "arccos(clip(t_s . normalize(int_basal hat(S_s) dA), -1, 1))",
            "zbuf_resolution": args.zbuf_resolution,
            "envelope_bins": args.envelope_bins,
            "light_directions": [
                {"name": n, "zenith_deg": z, "azimuth_deg": a} for n, z, a in LIGHT_DIRECTIONS
            ],
            "quality_weight_formula": "q_i = coverage_observed * exp(-scan_distance_p95 / tau)",
            "quality_weight_tau_median_p95": tau,
        },
        "source_sha256": {
            "src/maize_leaf/traits.py": _sha256(ROOT / "src/maize_leaf/traits.py"),
            "src/maize_leaf/entropy.py": _sha256(ROOT / "src/maize_leaf/entropy.py"),
            "scripts/extract_traits.py": _sha256(ROOT / "scripts/extract_traits.py"),
        },
        "entropy_bin_stability": entropy_stability,
        "curvature_margin_trim_stability": trim_stability,
        "stem_leaf_angle_basal_span_stability": basal_stability,
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(args.output / "traits_metadata.json", meta)
    print(f"\nDONE {args.output.resolve()} | plants={len(all_plants)}, leaves={len(all_leaves)}, tau={tau:.6f} [{meta['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
