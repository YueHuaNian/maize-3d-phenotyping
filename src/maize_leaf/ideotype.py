"""阶段 F：衔接五因素分类体系的玉米理想株型结构符合度指数与四项检验。

Five-factor linked ideotype conformity indices (I_A, I_L, I) and four methodological validations:
  1. Single-leaf five-factor conformity I_L (0-100):
     Evaluates (theta, P, H_pit, far_dz_norm) against height-conditioned smart-canopy targets
     indexed by relative insertion height h = stem_rel_height in [0, 1], with dimensionless
     far_dz_norm = clip(far_end_height_diff / max(observed_length, 1e-6), -1, 1).
     Primary leaf weights are fixed equal (0.25, 0.25, 0.25, 0.25).
  2. Plant architecture conformity I_A (0-100):
     Equal-weighted combination (1/3, 1/3, 1/3) of three dimensionless canopy sub-indices:
       - S_env: 25-layer normalized radial envelope R(h)/R_max conformity to a compact-top
         spindle/tower profile R*(h).
       - S_light: 5-direction Z-buffer light interception efficiency (0.5 * non-overlap +
         0.5 * oblique-to-nadir visible area retention).
       - S_comp: 1 - TV(f_plant, f*) conformity of five-factor pose class fractions
         (convergent, expanded, drooping) to the pre-specified smart-canopy target f*.
  3. Composite ideotype conformity index I = alpha * I_A + (1 - alpha) * I_L,
     with pre-registered equal weight alpha = 0.5.
  4. Four methodological validations:
     (1) Monotonicity along synthetic non-ideal -> ideal trajectories;
     (2) Measurement/occlusion perturbation robustness & quality/review-exclusion comparison;
     (3) Plant-level bootstrap (independent unit = plant) + within-plant leaf bootstrap & LOLO;
     (4) Weight, Dirichlet, target-composition, and leave-one-component-out sensitivity.

Scientific boundary:
  All targets, tolerances, and weights are pre-specified geometric ideotype constants rather than
  yield-calibrated biological optima. Scores measure conformity to the defined smart-canopy
  reference geometry and must not be interpreted as yield, photosynthetic rate, or leaf age.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from scipy.stats import kendalltau, spearmanr

from .classification import CLASS_NAMES


INDEX_SCHEMA_VERSION = "five_factor_ideotype_index_v1"

DEFAULT_ALPHA = 0.5

DEFAULT_LEAF_TARGETS = {
    # 茎叶角目标：上紧下展，theta*(h) = 60 - 40*h (h=1 -> 20 deg, h=0 -> 60 deg)
    "angle_base_deg": 60.0,
    "angle_slope_deg": -40.0,
    "angle_sigma_deg": 25.0,
    # 同株相对水平投影面积目标：顶部小叶紧凑、中层主叶舒展、底层适中
    "proj_floor": 0.30,
    "proj_amp": 0.55,
    "proj_peak_h": 0.55,
    "proj_peak_width": 0.25,
    "proj_sigma": 0.35,
    # 俯仰角信息熵目标：上层方向一致低熵、中下层允许适度拱曲
    "entropy_base": 0.75,
    "entropy_slope": -0.30,
    "entropy_sigma": 0.20,
    # 无量纲远端高差目标：far_dz / observed_length，上层显著上举、下层不剧烈下垂
    "far_dz_base": 0.05,
    "far_dz_slope": 0.65,
    "far_dz_sigma": 0.35,
}

DEFAULT_LEAF_WEIGHTS = {
    "s_angle": 0.25,
    "s_proj": 0.25,
    "s_entropy": 0.25,
    "s_far_dz": 0.25,
}

DEFAULT_ARCH_TARGETS = {
    # 25 层归一化径向包络目标 R*(h) = floor + (1 - floor) * exp(-(h - peak_h)^2 / (2 * width^2))
    # peak_h = 0.46 精确对齐 25 层离散中点网格（第 11 层 0.44-0.48），保证理想离散包络可达精确 1.0
    "env_floor": 0.25,
    "env_peak_h": 0.46,
    "env_peak_width": 0.26,
    "env_l1_scale": 0.35,
    # 5 方向受光：顶视非互遮率与斜射光保持率等权
    "light_non_overlap_weight": 0.5,
    "light_slant_weight": 0.5,
    # 五因素三类比例目标（紧缩 / 舒展 / 下垂）：上紧、中展为主、下垂受控
    "target_class_fractions": {
        "convergent": 0.20,
        "expanded": 0.60,
        "drooping": 0.20,
    },
}

DEFAULT_ARCH_WEIGHTS = {
    "s_envelope": 1.0 / 3.0,
    "s_light": 1.0 / 3.0,
    "s_composition": 1.0 / 3.0,
}

# 基于 450 组留出遮挡与窗宽标定的实测抖动量级（固定参考扰动标准差）
DEFAULT_PERTURBATION_SCALES = {
    "stem_leaf_angle_deg": 3.5,
    "rel_projected_area_rel": 0.05,
    "pitch_entropy_b12": 0.03,
    "relative_height": 0.02,
    "far_dz_norm": 0.05,
    "envelope_radius_rel": 0.05,
    "light_area_rel": 0.05,
}


def _safe_float(value, default=0.0):
    """将输入值安全转换为有限浮点数。"""
    if value is None or value == "":
        return float(default)
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value):
    """将表格中的布尔表示解析为 Python bool。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def _normalize_weight_dict(weights, keys):
    """校验非负权重字典并归一化为和为 1。"""
    raw = {k: float(weights[k]) for k in keys}
    if any((not math.isfinite(v)) or v < 0.0 for v in raw.values()):
        raise ValueError(f"Weights must be finite and nonnegative: {raw}")
    total = sum(raw.values())
    if total <= 1e-12:
        raise ValueError(f"Weight sum must be positive: {raw}")
    return {k: v / total for k, v in raw.items()}


def _gaussian_conformity(val, target, sigma):
    """高斯核符合度映射，取值 (0, 1]，在 val == target 处取唯一最大值 1。"""
    z = (float(val) - float(target)) / float(sigma)
    return float(math.exp(-0.5 * z * z))


def ideal_leaf_targets_at_height(h, leaf_targets=None):
    """给定相对着生高度 h in [0, 1]，返回四项形态特征的理想目标值。"""
    cfg = {**DEFAULT_LEAF_TARGETS, **(leaf_targets or {})}
    hc = float(np.clip(h, 0.0, 1.0))
    theta_star = cfg["angle_base_deg"] + cfg["angle_slope_deg"] * hc
    z_p = (hc - cfg["proj_peak_h"]) / cfg["proj_peak_width"]
    proj_star = cfg["proj_floor"] + cfg["proj_amp"] * math.exp(-0.5 * z_p * z_p)
    entropy_star = cfg["entropy_base"] + cfg["entropy_slope"] * hc
    far_dz_star = cfg["far_dz_base"] + cfg["far_dz_slope"] * hc
    return {
        "h": hc,
        "theta_target_deg": float(theta_star),
        "proj_target": float(proj_star),
        "entropy_target": float(entropy_star),
        "far_dz_norm_target": float(far_dz_star),
    }


def ideal_envelope_radius(h, arch_targets=None):
    """计算理想塔型冠层在归一化高度 h in [0, 1] 处的无量纲包络半径 R*(h)。"""
    cfg = {**DEFAULT_ARCH_TARGETS, **(arch_targets or {})}
    arr = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    z = (arr - cfg["env_peak_h"]) / cfg["env_peak_width"]
    res = cfg["env_floor"] + (1.0 - cfg["env_floor"]) * np.exp(-0.5 * z * z)
    return float(res) if np.ndim(h) == 0 else res


def normalize_far_height_diff(far_end_height_diff, observed_length):
    """将远端高差按观测叶长无量纲化，确保坐标整体缩放不变。"""
    length = max(_safe_float(observed_length, 0.0), 1e-6)
    dz = _safe_float(far_end_height_diff, 0.0)
    return float(np.clip(dz / length, -1.0, 1.0))


def score_single_leaf(row, leaf_targets=None, leaf_weights=None):
    """计算单叶五因素理想符合度 I_L（0-100）及四项分值。"""
    cfg = {**DEFAULT_LEAF_TARGETS, **(leaf_targets or {})}
    w = _normalize_weight_dict(leaf_weights or DEFAULT_LEAF_WEIGHTS, ("s_angle", "s_proj", "s_entropy", "s_far_dz"))

    h = _safe_float(row.get("relative_height", row.get("stem_rel_height", row.get("norm_base_height", 0.5))), 0.5)
    h = float(np.clip(h, 0.0, 1.0))
    theta = _safe_float(row.get("stem_leaf_angle_deg"))
    proj = float(np.clip(_safe_float(row.get("rel_projected_area")), 0.0, 1.0))
    ent = float(np.clip(_safe_float(row.get("pitch_entropy_b12")), 0.0, 1.0))
    if "far_dz_norm" in row and row["far_dz_norm"] not in (None, ""):
        far_dz_norm = float(np.clip(_safe_float(row["far_dz_norm"]), -1.0, 1.0))
    else:
        far_dz_norm = normalize_far_height_diff(row.get("far_end_height_diff"), row.get("observed_length", 1.0))

    targets = ideal_leaf_targets_at_height(h, cfg)
    s_angle = _gaussian_conformity(theta, targets["theta_target_deg"], cfg["angle_sigma_deg"])
    s_proj = _gaussian_conformity(proj, targets["proj_target"], cfg["proj_sigma"])
    s_entropy = _gaussian_conformity(ent, targets["entropy_target"], cfg["entropy_sigma"])
    s_far_dz = _gaussian_conformity(far_dz_norm, targets["far_dz_norm_target"], cfg["far_dz_sigma"])

    il = 100.0 * (
        w["s_angle"] * s_angle
        + w["s_proj"] * s_proj
        + w["s_entropy"] * s_entropy
        + w["s_far_dz"] * s_far_dz
    )
    return {
        "plant_id": str(row.get("plant_id", "")),
        "leaf_id": int(row.get("leaf_id", 0)),
        "cluster_kmeans": str(row.get("cluster_kmeans", "")),
        "relative_height": h,
        "stem_leaf_angle_deg": theta,
        "rel_projected_area": proj,
        "pitch_entropy_b12": ent,
        "far_end_height_diff": _safe_float(row.get("far_end_height_diff")),
        "observed_length": _safe_float(row.get("observed_length")),
        "far_dz_norm": far_dz_norm,
        "surface_area": _safe_float(row.get("surface_area")),
        "quality_weight": _safe_float(row.get("quality_weight"), 1.0),
        "support_review_required": _safe_bool(row.get("support_review_required", False)),
        "theta_target_deg": targets["theta_target_deg"],
        "proj_target": targets["proj_target"],
        "entropy_target": targets["entropy_target"],
        "far_dz_norm_target": targets["far_dz_norm_target"],
        "s_angle": s_angle,
        "s_proj": s_proj,
        "s_entropy": s_entropy,
        "s_far_dz": s_far_dz,
        "leaf_index_il": float(il),
    }


def compute_leaf_ideotype_scores(leaf_rows, leaf_targets=None, leaf_weights=None):
    """批量计算全部叶片的五因素理想符合度记录。"""
    return [score_single_leaf(r, leaf_targets=leaf_targets, leaf_weights=leaf_weights) for r in leaf_rows]


def score_plant_envelope(env_rows, arch_targets=None):
    """基于单株 25 层径向包络计算塔型轮廓符合度 S_env in (0, 1] 与 L1 偏差。"""
    cfg = {**DEFAULT_ARCH_TARGETS, **(arch_targets or {})}
    hs, rs = [], []
    for row in env_rows:
        raw_r = row.get("radius_q95")
        if raw_r is None or raw_r == "" or str(raw_r).strip().lower() == "nan":
            continue
        rv = float(raw_r)
        if not math.isfinite(rv) or rv < 0.0:
            continue
        hs.append(_safe_float(row.get("h_mid")))
        rs.append(rv)
    if not rs:
        return {"s_envelope": 0.0, "envelope_l1_error": 1.0, "envelope_valid_bins": 0}
    h_arr = np.asarray(hs, dtype=float)
    r_arr = np.asarray(rs, dtype=float)
    rmax = float(np.max(r_arr))
    if rmax <= 1e-12:
        return {"s_envelope": 0.0, "envelope_l1_error": 1.0, "envelope_valid_bins": len(rs)}
    r_norm = r_arr / rmax
    r_target = ideal_envelope_radius(h_arr, cfg)
    l1_err = float(np.mean(np.abs(r_norm - r_target)))
    s_env = float(math.exp(-l1_err / cfg["env_l1_scale"]))
    return {"s_envelope": s_env, "envelope_l1_error": l1_err, "envelope_valid_bins": len(rs)}


def score_plant_light(plant_row, arch_targets=None):
    """计算冠层受光效率 S_light in (0, 1]：优先使用 46 方向 Turtle Sky（SOC 加权，绕 Z 轴旋转不变），缺省回退 5 方向。"""
    cfg = {**DEFAULT_ARCH_TARGETS, **(arch_targets or {})}
    w_no = float(cfg["light_non_overlap_weight"])
    w_sl = float(cfg["light_slant_weight"])
    w_sum = w_no + w_sl
    w_no, w_sl = w_no / w_sum, w_sl / w_sum

    t46_no_raw = plant_row.get("turtle46_non_overlap")
    t46_sl_raw = plant_row.get("turtle46_slant_retention")
    if (
        t46_no_raw not in (None, "")
        and t46_sl_raw not in (None, "")
        and str(t46_no_raw).strip().lower() != "nan"
        and str(t46_sl_raw).strip().lower() != "nan"
    ):
        non_overlap = float(np.clip(_safe_float(t46_no_raw), 0.0, 1.0))
        slant_retention = float(np.clip(_safe_float(t46_sl_raw), 0.0, 1.0))
        s_light = float(w_no * non_overlap + w_sl * slant_retention)
        return {
            "s_light": s_light,
            "light_non_overlap": non_overlap,
            "light_slant_retention": slant_retention,
            "light_source": "turtle_sky_46_soc",
        }

    sum_proj = _safe_float(plant_row.get("sum_projected_area_element"))
    union_area = _safe_float(plant_row.get("canopy_union_area"))
    nadir = _safe_float(plant_row.get("visible_area_nadir"), union_area)
    slant_vals = [
        _safe_float(plant_row.get("visible_area_zen30_az0")),
        _safe_float(plant_row.get("visible_area_zen30_az90")),
        _safe_float(plant_row.get("visible_area_zen60_az0")),
        _safe_float(plant_row.get("visible_area_zen60_az90")),
    ]
    non_overlap = float(np.clip(union_area / max(sum_proj, 1e-12), 0.0, 1.0)) if sum_proj > 1e-12 else 0.0
    slant_mean = float(np.mean(slant_vals))
    slant_retention = float(np.clip(slant_mean / max(nadir, 1e-12), 0.0, 1.0)) if nadir > 1e-12 else 0.0
    s_light = float(w_no * non_overlap + w_sl * slant_retention)
    return {
        "s_light": s_light,
        "light_non_overlap": non_overlap,
        "light_slant_retention": slant_retention,
        "light_source": "zbuffer_5dir",
    }


def score_plant_composition(class_labels, arch_targets=None):
    """基于五因素三类叶姿比例计算与智能冠层目标比例的符合度 S_comp = 1 - TV(f, f*)。"""
    cfg = {**DEFAULT_ARCH_TARGETS, **(arch_targets or {})}
    target_map = _normalize_weight_dict(cfg["target_class_fractions"], CLASS_NAMES)
    n = len(class_labels)
    if n == 0:
        return {
            "s_composition": 0.0,
            "composition_tv_distance": 1.0,
            "frac_convergent": 0.0,
            "frac_expanded": 0.0,
            "frac_drooping": 0.0,
        }
    counts = {c: 0 for c in CLASS_NAMES}
    for lbl in class_labels:
        if lbl in counts:
            counts[lbl] += 1
    fracs = {c: counts[c] / n for c in CLASS_NAMES}
    tv = 0.5 * sum(abs(fracs[c] - target_map[c]) for c in CLASS_NAMES)
    return {
        "s_composition": float(np.clip(1.0 - tv, 0.0, 1.0)),
        "composition_tv_distance": float(tv),
        "frac_convergent": fracs["convergent"],
        "frac_expanded": fracs["expanded"],
        "frac_drooping": fracs["drooping"],
    }


def _weighted_mean(values, weights):
    """计算非负加权均值，总权退化时回退等权均值。"""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not len(v):
        return 0.0
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    s = float(np.sum(w))
    return float(np.sum(v * w) / s) if s > 1e-12 else float(np.mean(v))


def compute_plant_ideotype_scores(
    plant_rows,
    envelope_rows,
    leaf_scored_rows,
    arch_targets=None,
    arch_weights=None,
    alpha=DEFAULT_ALPHA,
    aggregation="equal",
):
    """汇总全部植株的 I_A、I_L 与综合指数 I，并生成由高到低的排名。"""
    if aggregation not in ("equal", "area", "quality"):
        raise ValueError(f"Unsupported leaf aggregation: {aggregation}")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    w_arch = _normalize_weight_dict(arch_weights or DEFAULT_ARCH_WEIGHTS, ("s_envelope", "s_light", "s_composition"))

    env_by_plant = defaultdict(list)
    for row in envelope_rows:
        env_by_plant[str(row["plant_id"])].append(row)
    leaves_by_plant = defaultdict(list)
    for row in leaf_scored_rows:
        leaves_by_plant[str(row["plant_id"])].append(row)

    results = []
    for prow in plant_rows:
        pid = str(prow["plant_id"])
        pleaves = leaves_by_plant.get(pid, [])
        env_res = score_plant_envelope(env_by_plant.get(pid, []), arch_targets=arch_targets)
        light_res = score_plant_light(prow, arch_targets=arch_targets)
        comp_res = score_plant_composition([l["cluster_kmeans"] for l in pleaves], arch_targets=arch_targets)

        ia = 100.0 * (
            w_arch["s_envelope"] * env_res["s_envelope"]
            + w_arch["s_light"] * light_res["s_light"]
            + w_arch["s_composition"] * comp_res["s_composition"]
        )

        il_vals = [l["leaf_index_il"] for l in pleaves]
        area_w = [max(l["surface_area"], 0.0) for l in pleaves]
        qual_w = [max(l["quality_weight"], 0.0) for l in pleaves]
        il_equal = float(np.mean(il_vals)) if il_vals else 0.0
        il_area = _weighted_mean(il_vals, area_w)
        il_quality = _weighted_mean(il_vals, qual_w)
        il_active = {"equal": il_equal, "area": il_area, "quality": il_quality}[aggregation]
        comp_i = alpha * ia + (1.0 - alpha) * il_active

        results.append({
            "plant_id": pid,
            "leaf_count": len(pleaves),
            "support_review_leaf_count": sum(1 for l in pleaves if l["support_review_required"]),
            "composite_index_i": float(comp_i),
            "arch_index_ia": float(ia),
            "leaf_index_il": float(il_active),
            "leaf_index_il_equal": float(il_equal),
            "leaf_index_il_area": float(il_area),
            "leaf_index_il_quality": float(il_quality),
            "s_envelope": env_res["s_envelope"],
            "s_light": light_res["s_light"],
            "s_composition": comp_res["s_composition"],
            "envelope_l1_error": env_res["envelope_l1_error"],
            "envelope_valid_bins": env_res["envelope_valid_bins"],
            "light_non_overlap": light_res["light_non_overlap"],
            "light_slant_retention": light_res["light_slant_retention"],
            "composition_tv_distance": comp_res["composition_tv_distance"],
            "frac_convergent": comp_res["frac_convergent"],
            "frac_expanded": comp_res["frac_expanded"],
            "frac_drooping": comp_res["frac_drooping"],
            "mean_s_angle": float(np.mean([l["s_angle"] for l in pleaves])) if pleaves else 0.0,
            "mean_s_proj": float(np.mean([l["s_proj"] for l in pleaves])) if pleaves else 0.0,
            "mean_s_entropy": float(np.mean([l["s_entropy"] for l in pleaves])) if pleaves else 0.0,
            "mean_s_far_dz": float(np.mean([l["s_far_dz"] for l in pleaves])) if pleaves else 0.0,
        })

    # 按综合指数降序编排排名（并列时按 plant_id 确定性排序）
    order = sorted(range(len(results)), key=lambda idx: (-results[idx]["composite_index_i"], results[idx]["plant_id"]))
    for rank_num, idx in enumerate(order, start=1):
        results[idx]["rank"] = rank_num
    return results


def safe_spearman(a, b):
    """安全计算 Spearman 秩相关，常数序列返回 1.0（相同）或 0.0。"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2:
        return 1.0
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return 1.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    val = float(spearmanr(x, y).statistic)
    return val if math.isfinite(val) else 0.0


def safe_kendall(a, b):
    """安全计算 Kendall tau-b，常数序列做退化保护。"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2:
        return 1.0
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return 1.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    val = float(kendalltau(x, y).statistic)
    return val if math.isfinite(val) else 0.0


def _topk_overlap(scores_a, scores_b, plant_ids, k=3, largest=True):
    """计算两组排序的 Top-k 或 Bottom-k 植株集合重合度。"""
    order_a = sorted(range(len(plant_ids)), key=lambda i: (-scores_a[i] if largest else scores_a[i], plant_ids[i]))
    order_b = sorted(range(len(plant_ids)), key=lambda i: (-scores_b[i] if largest else scores_b[i], plant_ids[i]))
    set_a = {plant_ids[i] for i in order_a[:k]}
    set_b = {plant_ids[i] for i in order_b[:k]}
    return len(set_a & set_b) / max(1, min(k, len(plant_ids)))


def run_monotonicity_validation(
    leaf_targets=None,
    arch_targets=None,
    leaf_weights=None,
    arch_weights=None,
    alpha=DEFAULT_ALPHA,
    n_steps=21,
):
    """检验一（单调性）：沿极端非理想 -> 理想目标的线性轨迹，验证 I_A、I_L、I 及全部 7 分项严格单调递增。"""
    lambdas = np.linspace(0.0, 1.0, int(n_steps))
    h_bins = np.linspace(0.02, 0.98, 25)
    r_ideal = ideal_envelope_radius(h_bins, arch_targets)
    # 极端非理想包络：倒置塔型（顶部极宽 1.0，中下部极窄 0.15）
    r_bad = np.clip(0.15 + 0.85 * h_bins, 0.05, 1.0)
    r_bad = r_bad / np.max(r_bad)

    # 10 片代表不同高度的合成叶（2 紧缩 + 6 舒展 + 2 下垂 = 精确目标比例 0.2/0.6/0.2）
    leaf_hs = np.linspace(0.95, 0.05, 10)
    ideal_classes = ["convergent"] * 2 + ["expanded"] * 6 + ["drooping"] * 2
    bad_classes = ["drooping"] * 10
    # 按 lambda 逐步将非理想类别替换为理想类别（确保 TV 距离单调下降）
    # bad 0/0/10 vs ideal 2/6/2 -> TV=0.8，每步切换 1 片可形成 8 次递增
    replace_order = [0, 1, 2, 3, 4, 5, 6, 7]

    series = {k: [] for k in ("lambda", "I", "I_A", "I_L", "s_envelope", "s_light", "s_composition",
                              "s_angle", "s_proj", "s_entropy", "s_far_dz")}
    for lam in lambdas:
        lam = float(lam)
        r_cur = (1.0 - lam) * r_bad + lam * r_ideal
        env_rows = [{"plant_id": "syn", "h_mid": float(hb), "radius_q95": float(rv)} for hb, rv in zip(h_bins, r_cur)]

        # 受光：非互遮率 0.45 -> 1.0，斜射保持率 0.40 -> 1.0
        non_ov = 0.45 + 0.55 * lam
        slant_ret = 0.40 + 0.60 * lam
        nadir = 1.0
        plant_row = {
            "plant_id": "syn",
            "sum_projected_area_element": nadir / non_ov,
            "canopy_union_area": nadir,
            "visible_area_nadir": nadir,
            "visible_area_zen30_az0": nadir * slant_ret,
            "visible_area_zen30_az90": nadir * slant_ret,
            "visible_area_zen60_az0": nadir * slant_ret,
            "visible_area_zen60_az90": nadir * slant_ret,
        }

        n_swapped = int(round(lam * len(replace_order)))
        cur_classes = list(bad_classes)
        for idx in replace_order[:n_swapped]:
            cur_classes[idx] = ideal_classes[idx]

        leaf_rows = []
        for idx, (lh, cls_name) in enumerate(zip(leaf_hs, cur_classes), start=1):
            t = ideal_leaf_targets_at_height(lh, leaf_targets)
            # 非理想端点在每个特征上偏离理想值固定方向
            theta = (1.0 - lam) * (t["theta_target_deg"] + 40.0) + lam * t["theta_target_deg"]
            proj = float(np.clip((1.0 - lam) * (t["proj_target"] - 0.45 if t["proj_target"] > 0.5 else t["proj_target"] + 0.45) + lam * t["proj_target"], 0.0, 1.0))
            ent = float(np.clip((1.0 - lam) * min(0.98, t["entropy_target"] + 0.35) + lam * t["entropy_target"], 0.0, 1.0))
            far_dz_norm = float(np.clip((1.0 - lam) * max(-0.95, t["far_dz_norm_target"] - 0.70) + lam * t["far_dz_norm_target"], -1.0, 1.0))
            leaf_rows.append({
                "plant_id": "syn",
                "leaf_id": idx,
                "cluster_kmeans": cls_name,
                "relative_height": float(lh),
                "stem_leaf_angle_deg": float(theta),
                "rel_projected_area": proj,
                "pitch_entropy_b12": ent,
                "far_end_height_diff": far_dz_norm,
                "observed_length": 1.0,
                "far_dz_norm": far_dz_norm,
                "surface_area": 1.0,
                "quality_weight": 1.0,
                "support_review_required": False,
            })

        scored_l = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=leaf_weights)
        scored_p = compute_plant_ideotype_scores(
            [plant_row], env_rows, scored_l, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
        )[0]
        series["lambda"].append(lam)
        series["I"].append(scored_p["composite_index_i"])
        series["I_A"].append(scored_p["arch_index_ia"])
        series["I_L"].append(scored_p["leaf_index_il"])
        for key in ("s_envelope", "s_light", "s_composition"):
            series[key].append(scored_p[key])
        series["s_angle"].append(scored_p["mean_s_angle"])
        series["s_proj"].append(scored_p["mean_s_proj"])
        series["s_entropy"].append(scored_p["mean_s_entropy"])
        series["s_far_dz"].append(scored_p["mean_s_far_dz"])

    checks = {}
    all_passed = True
    for name in ("I", "I_A", "I_L", "s_envelope", "s_light", "s_composition",
                 "s_angle", "s_proj", "s_entropy", "s_far_dz"):
        arr = np.asarray(series[name], dtype=float)
        diffs = np.diff(arr)
        min_delta = float(np.min(diffs))
        total_gain = float(arr[-1] - arr[0])
        rho = safe_spearman(lambdas, arr)
        ok = (min_delta >= -1e-12) and (total_gain > 0.0) and (rho > 0.99)
        if not ok:
            all_passed = False
        checks[name] = {
            "start": float(arr[0]),
            "end": float(arr[-1]),
            "total_gain": total_gain,
            "min_step_delta": min_delta,
            "spearman_rho": rho,
            "passed": bool(ok),
        }

    return {
        "passed": bool(all_passed),
        "n_steps": int(n_steps),
        "endpoint_ideal_i": float(series["I"][-1]),
        "checks": checks,
        "trajectory": series,
    }


def run_perturbation_validation(
    plant_rows,
    envelope_rows,
    leaf_rows,
    leaf_targets=None,
    arch_targets=None,
    leaf_weights=None,
    arch_weights=None,
    alpha=DEFAULT_ALPHA,
    perturbation_scales=None,
    n_trials=200,
    seed=2026,
):
    """检验二（扰动鲁棒性）：模拟测量/遮挡误差蒙特卡洛扰动，并对比质量加权、面积加权及剔除 20 片待查叶的排序一致性。"""
    scales = {**DEFAULT_PERTURBATION_SCALES, **(perturbation_scales or {})}
    base_leaves = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=leaf_weights)
    base_plants = compute_plant_ideotype_scores(
        plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
    )
    pids = [p["plant_id"] for p in base_plants]
    base_i = np.array([p["composite_index_i"] for p in base_plants], dtype=float)
    base_ia = np.array([p["arch_index_ia"] for p in base_plants], dtype=float)
    base_il = np.array([p["leaf_index_il"] for p in base_plants], dtype=float)

    rng = np.random.default_rng(int(seed))
    rhos_i, taus_i, rhos_ia, rhos_il = [], [], [], []
    max_abs_diffs_i, top3_overlaps = [], []

    for _ in range(int(n_trials)):
        pert_leaves = []
        for row in base_leaves:
            r = dict(row)
            r["stem_leaf_angle_deg"] = float(np.clip(r["stem_leaf_angle_deg"] + rng.normal(0.0, scales["stem_leaf_angle_deg"]), 0.0, 180.0))
            r["rel_projected_area"] = float(np.clip(r["rel_projected_area"] * (1.0 + rng.normal(0.0, scales["rel_projected_area_rel"])), 0.0, 1.0))
            r["pitch_entropy_b12"] = float(np.clip(r["pitch_entropy_b12"] + rng.normal(0.0, scales["pitch_entropy_b12"]), 0.0, 1.0))
            r["relative_height"] = float(np.clip(r["relative_height"] + rng.normal(0.0, scales["relative_height"]), 0.0, 1.0))
            r["far_dz_norm"] = float(np.clip(r["far_dz_norm"] + rng.normal(0.0, scales["far_dz_norm"]), -1.0, 1.0))
            pert_leaves.append(r)

        pert_envs = []
        for row in envelope_rows:
            r = dict(row)
            if r.get("radius_q95") not in (None, "") and str(r["radius_q95"]).strip().lower() != "nan":
                rv = float(r["radius_q95"])
                if math.isfinite(rv):
                    r["radius_q95"] = max(1e-6, rv * (1.0 + float(rng.normal(0.0, scales["envelope_radius_rel"]))))
            pert_envs.append(r)

        pert_plants_in = []
        for row in plant_rows:
            r = dict(row)
            for col in ("sum_projected_area_element", "canopy_union_area", "visible_area_nadir",
                        "visible_area_zen30_az0", "visible_area_zen30_az90",
                        "visible_area_zen60_az0", "visible_area_zen60_az90"):
                v = _safe_float(r.get(col))
                r[col] = max(1e-8, v * (1.0 + float(rng.normal(0.0, scales["light_area_rel"]))))
            for col in ("turtle46_non_overlap", "turtle46_slant_retention"):
                if r.get(col) not in (None, "") and str(r.get(col)).strip().lower() != "nan":
                    v = _safe_float(r.get(col))
                    r[col] = float(np.clip(v * (1.0 + float(rng.normal(0.0, scales["light_area_rel"]))), 0.0, 1.0))
            pert_plants_in.append(r)

        scored_l = compute_leaf_ideotype_scores(pert_leaves, leaf_targets=leaf_targets, leaf_weights=leaf_weights)
        scored_p = compute_plant_ideotype_scores(
            pert_plants_in, pert_envs, scored_l, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in scored_p], dtype=float)
        cur_ia = np.array([p["arch_index_ia"] for p in scored_p], dtype=float)
        cur_il = np.array([p["leaf_index_il"] for p in scored_p], dtype=float)

        rhos_i.append(safe_spearman(base_i, cur_i))
        taus_i.append(safe_kendall(base_i, cur_i))
        rhos_ia.append(safe_spearman(base_ia, cur_ia))
        rhos_il.append(safe_spearman(base_il, cur_il))
        max_abs_diffs_i.append(float(np.max(np.abs(cur_i - base_i))))
        top3_overlaps.append(_topk_overlap(base_i, cur_i, pids, k=3, largest=True))

    # 聚合与质量复查稳健性对照：面积加权、质量加权、剔除 >5% 支撑待查叶
    structural_comparisons = {}
    for agg in ("area", "quality"):
        alt_p = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=arch_weights,
            alpha=alpha, aggregation=agg
        )
        alt_i = np.array([p["composite_index_i"] for p in alt_p], dtype=float)
        alt_il = np.array([p["leaf_index_il"] for p in alt_p], dtype=float)
        structural_comparisons[f"aggregation_{agg}"] = {
            "spearman_i": safe_spearman(base_i, alt_i),
            "kendall_i": safe_kendall(base_i, alt_i),
            "spearman_il": safe_spearman(base_il, alt_il),
            "max_abs_diff_i": float(np.max(np.abs(alt_i - base_i))),
            "top3_overlap": _topk_overlap(base_i, alt_i, pids, k=3, largest=True),
        }

    # 剔除 support_review_required=True 的待查叶（若某株全被剔除则保留原叶以避免空株）
    by_plant = defaultdict(list)
    for row in base_leaves:
        by_plant[row["plant_id"]].append(row)
    filtered_leaves = []
    excluded_count = 0
    for pid, rows in by_plant.items():
        kept = [r for r in rows if not r["support_review_required"]]
        if kept:
            filtered_leaves.extend(kept)
            excluded_count += len(rows) - len(kept)
        else:
            filtered_leaves.extend(rows)
    filt_p = compute_plant_ideotype_scores(
        plant_rows, envelope_rows, filtered_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
    )
    filt_i = np.array([p["composite_index_i"] for p in filt_p], dtype=float)
    filt_il = np.array([p["leaf_index_il"] for p in filt_p], dtype=float)
    structural_comparisons["exclude_support_review_leaves"] = {
        "excluded_leaves": int(excluded_count),
        "retained_leaves": len(filtered_leaves),
        "spearman_i": safe_spearman(base_i, filt_i),
        "kendall_i": safe_kendall(base_i, filt_i),
        "spearman_il": safe_spearman(base_il, filt_il),
        "max_abs_diff_i": float(np.max(np.abs(filt_i - base_i))),
        "top3_overlap": _topk_overlap(base_i, filt_i, pids, k=3, largest=True),
    }

    return {
        "n_trials": int(n_trials),
        "seed": int(seed),
        "perturbation_scales": scales,
        "monte_carlo": {
            "spearman_i_median": float(np.median(rhos_i)),
            "spearman_i_q25": float(np.quantile(rhos_i, 0.25)),
            "spearman_i_q75": float(np.quantile(rhos_i, 0.75)),
            "spearman_i_min": float(np.min(rhos_i)),
            "kendall_i_median": float(np.median(taus_i)),
            "spearman_ia_median": float(np.median(rhos_ia)),
            "spearman_il_median": float(np.median(rhos_il)),
            "max_abs_diff_i_median": float(np.median(max_abs_diffs_i)),
            "max_abs_diff_i_p95": float(np.quantile(max_abs_diffs_i, 0.95)),
            "top3_overlap_mean": float(np.mean(top3_overlaps)),
            "spearman_i_samples": [float(v) for v in rhos_i],
        },
        "structural_comparisons": structural_comparisons,
    }


def run_bootstrap_validation(
    plant_rows,
    envelope_rows,
    leaf_rows,
    leaf_targets=None,
    arch_targets=None,
    leaf_weights=None,
    arch_weights=None,
    alpha=DEFAULT_ALPHA,
    n_boot=500,
    seed=2026,
):
    """检验三（植株级 Bootstrap 与株内叶片重抽样/LOLO 秩稳定性）。

    - 植株级（独立观测单位 = 14 株）：有放回重抽样估算群体均值/中位数/IQR 95% CI 及每株进入 Top-3/Bottom-3 频次。
    - 株内叶片级：在每株内部有放回重抽样叶片评估单株 I_L 与 I 排序稳定性，并执行全量 Leave-One-Leaf-Out。
    """
    base_leaves = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=leaf_weights)
    base_plants = compute_plant_ideotype_scores(
        plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
    )
    n_plants = len(base_plants)
    pids = [p["plant_id"] for p in base_plants]
    base_i = np.array([p["composite_index_i"] for p in base_plants], dtype=float)
    base_ia = np.array([p["arch_index_ia"] for p in base_plants], dtype=float)
    base_il = np.array([p["leaf_index_il"] for p in base_plants], dtype=float)

    rng = np.random.default_rng(int(seed))

    # 1. 植株级 Bootstrap（以植株为独立抽样单位）
    boot_mean_i, boot_med_i, boot_iqr_i = [], [], []
    boot_mean_ia, boot_mean_il = [], []
    top3_counts = {pid: 0 for pid in pids}
    bot3_counts = {pid: 0 for pid in pids}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_plants, size=n_plants)
        si = base_i[idx]
        boot_mean_i.append(float(np.mean(si)))
        boot_med_i.append(float(np.median(si)))
        boot_iqr_i.append(float(np.quantile(si, 0.75) - np.quantile(si, 0.25)))
        boot_mean_ia.append(float(np.mean(base_ia[idx])))
        boot_mean_il.append(float(np.mean(base_il[idx])))

        # 在本次重抽样的不重复植株集合中统计 Top-3 / Bottom-3 入选频次
        uniq = sorted(set(int(j) for j in idx))
        if len(uniq) >= 3:
            order_desc = sorted(uniq, key=lambda j: (-base_i[j], pids[j]))
            for j in order_desc[:3]:
                top3_counts[pids[j]] += 1
            for j in order_desc[-3:]:
                bot3_counts[pids[j]] += 1

    # 2. 株内叶片重抽样（同时重算 S_comp 与 I_L）
    leaves_by_plant = defaultdict(list)
    for row in base_leaves:
        leaves_by_plant[row["plant_id"]].append(row)

    within_rhos_i, within_taus_i, within_rhos_il = [], [], []
    plant_rank_samples = {pid: [] for pid in pids}
    for _ in range(int(n_boot)):
        resampled_leaves = []
        for pid in pids:
            pleaves = leaves_by_plant[pid]
            if not pleaves:
                continue
            pick = rng.integers(0, len(pleaves), size=len(pleaves))
            for k in pick:
                resampled_leaves.append(pleaves[int(k)])
        bp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, resampled_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in bp], dtype=float)
        cur_il = np.array([p["leaf_index_il"] for p in bp], dtype=float)
        within_rhos_i.append(safe_spearman(base_i, cur_i))
        within_taus_i.append(safe_kendall(base_i, cur_i))
        within_rhos_il.append(safe_spearman(base_il, cur_il))
        for p in bp:
            plant_rank_samples[p["plant_id"]].append(int(p["rank"]))

    # 3. 逐叶留一（Leave-One-Leaf-Out）确定性敏感性
    lolo_rhos_i, lolo_max_shifts = [], []
    for drop_idx in range(len(base_leaves)):
        target_pid = base_leaves[drop_idx]["plant_id"]
        if len(leaves_by_plant[target_pid]) <= 1:
            continue
        sub_leaves = base_leaves[:drop_idx] + base_leaves[drop_idx + 1:]
        lp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, sub_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in lp], dtype=float)
        lolo_rhos_i.append(safe_spearman(base_i, cur_i))
        lolo_max_shifts.append(float(np.max(np.abs(cur_i - base_i))))

    def ci95(arr):
        """返回 2.5% 与 97.5% 分位区间。"""
        return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "independent_unit": "plant (n=14); leaves are nested within plants",
        "plant_level_bootstrap": {
            "mean_i_ci95": ci95(boot_mean_i),
            "median_i_ci95": ci95(boot_med_i),
            "iqr_i_ci95": ci95(boot_iqr_i),
            "mean_ia_ci95": ci95(boot_mean_ia),
            "mean_il_ci95": ci95(boot_mean_il),
            "top3_inclusion_rate": {pid: round(top3_counts[pid] / n_boot, 4) for pid in pids},
            "bottom3_inclusion_rate": {pid: round(bot3_counts[pid] / n_boot, 4) for pid in pids},
        },
        "within_plant_leaf_bootstrap": {
            "spearman_i_median": float(np.median(within_rhos_i)),
            "spearman_i_ci95": ci95(within_rhos_i),
            "kendall_i_median": float(np.median(within_taus_i)),
            "spearman_il_median": float(np.median(within_rhos_il)),
            "spearman_il_ci95": ci95(within_rhos_il),
            "per_plant_rank_ci95": {pid: ci95(plant_rank_samples[pid]) for pid in pids},
        },
        "leave_one_leaf_out": {
            "n_cases": len(lolo_rhos_i),
            "spearman_i_min": float(np.min(lolo_rhos_i)) if lolo_rhos_i else 1.0,
            "spearman_i_median": float(np.median(lolo_rhos_i)) if lolo_rhos_i else 1.0,
            "max_abs_shift_i_max": float(np.max(lolo_max_shifts)) if lolo_max_shifts else 0.0,
            "max_abs_shift_i_median": float(np.median(lolo_max_shifts)) if lolo_max_shifts else 0.0,
        },
    }


def run_sensitivity_validation(
    plant_rows,
    envelope_rows,
    leaf_rows,
    leaf_targets=None,
    arch_targets=None,
    leaf_weights=None,
    arch_weights=None,
    alpha=DEFAULT_ALPHA,
    n_dirichlet=100,
    seed=2026,
):
    """检验四（权重、目标组成与分项消融敏感性）。"""
    base_leaves = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=leaf_weights)
    base_plants = compute_plant_ideotype_scores(
        plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
    )
    pids = [p["plant_id"] for p in base_plants]
    base_i = np.array([p["composite_index_i"] for p in base_plants], dtype=float)
    base_ia = np.array([p["arch_index_ia"] for p in base_plants], dtype=float)
    base_il = np.array([p["leaf_index_il"] for p in base_plants], dtype=float)

    # 1. 主权重 alpha 扫描
    alpha_sweep = []
    for a in (0.0, 0.25, 0.35, 0.50, 0.65, 0.75, 1.0):
        cur_p = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=arch_weights, alpha=a
        )
        cur_i = np.array([p["composite_index_i"] for p in cur_p], dtype=float)
        alpha_sweep.append({
            "alpha": float(a),
            "spearman_vs_primary": safe_spearman(base_i, cur_i),
            "kendall_vs_primary": safe_kendall(base_i, cur_i),
            "top3_overlap": _topk_overlap(base_i, cur_i, pids, k=3, largest=True),
            "bottom3_overlap": _topk_overlap(base_i, cur_i, pids, k=3, largest=False),
            "mean_i": float(np.mean(cur_i)),
            "std_i": float(np.std(cur_i)),
        })

    # 2. 预设子项偏重组合
    weight_presets = {
        "arch_env_heavy": ({"s_envelope": 0.6, "s_light": 0.2, "s_composition": 0.2}, DEFAULT_LEAF_WEIGHTS),
        "arch_light_heavy": ({"s_envelope": 0.2, "s_light": 0.6, "s_composition": 0.2}, DEFAULT_LEAF_WEIGHTS),
        "arch_comp_heavy": ({"s_envelope": 0.2, "s_light": 0.2, "s_composition": 0.6}, DEFAULT_LEAF_WEIGHTS),
        "leaf_angle_heavy": (DEFAULT_ARCH_WEIGHTS, {"s_angle": 0.55, "s_proj": 0.15, "s_entropy": 0.15, "s_far_dz": 0.15}),
        "leaf_proj_heavy": (DEFAULT_ARCH_WEIGHTS, {"s_angle": 0.15, "s_proj": 0.55, "s_entropy": 0.15, "s_far_dz": 0.15}),
        "leaf_entropy_heavy": (DEFAULT_ARCH_WEIGHTS, {"s_angle": 0.15, "s_proj": 0.15, "s_entropy": 0.55, "s_far_dz": 0.15}),
        "leaf_far_dz_heavy": (DEFAULT_ARCH_WEIGHTS, {"s_angle": 0.15, "s_proj": 0.15, "s_entropy": 0.15, "s_far_dz": 0.55}),
    }
    preset_results = {}
    for name, (aw, lw) in weight_presets.items():
        sl = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=lw)
        sp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, sl, arch_targets=arch_targets, arch_weights=aw, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in sp], dtype=float)
        preset_results[name] = {
            "spearman_vs_primary": safe_spearman(base_i, cur_i),
            "kendall_vs_primary": safe_kendall(base_i, cur_i),
            "top3_overlap": _topk_overlap(base_i, cur_i, pids, k=3, largest=True),
        }

    # 3. 100 组随机 Dirichlet 权重抽样（围绕等权浓度 c=6）
    rng = np.random.default_rng(int(seed))
    dir_rhos, dir_taus = [], []
    for _ in range(int(n_dirichlet)):
        aw_vec = rng.dirichlet([6.0, 6.0, 6.0])
        lw_vec = rng.dirichlet([6.0, 6.0, 6.0, 6.0])
        aw = dict(zip(("s_envelope", "s_light", "s_composition"), aw_vec))
        lw = dict(zip(("s_angle", "s_proj", "s_entropy", "s_far_dz"), lw_vec))
        a_cur = float(rng.uniform(0.35, 0.65))
        sl = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=lw)
        sp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, sl, arch_targets=arch_targets, arch_weights=aw, alpha=a_cur
        )
        cur_i = np.array([p["composite_index_i"] for p in sp], dtype=float)
        dir_rhos.append(safe_spearman(base_i, cur_i))
        dir_taus.append(safe_kendall(base_i, cur_i))

    # 4. 理想三类比例目标 f* 敏感性
    comp_targets = {
        "primary_20_60_20": {"convergent": 0.20, "expanded": 0.60, "drooping": 0.20},
        "compact_25_55_20": {"convergent": 0.25, "expanded": 0.55, "drooping": 0.20},
        "moderate_15_60_25": {"convergent": 0.15, "expanded": 0.60, "drooping": 0.25},
        "upright_30_60_10": {"convergent": 0.30, "expanded": 0.60, "drooping": 0.10},
    }
    comp_target_results = {}
    for name, f_star in comp_targets.items():
        at = {**DEFAULT_ARCH_TARGETS, **(arch_targets or {}), "target_class_fractions": f_star}
        sp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, base_leaves, arch_targets=at, arch_weights=arch_weights, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in sp], dtype=float)
        cur_ia = np.array([p["arch_index_ia"] for p in sp], dtype=float)
        comp_target_results[name] = {
            "target_class_fractions": f_star,
            "spearman_i_vs_primary": safe_spearman(base_i, cur_i),
            "spearman_ia_vs_primary": safe_spearman(base_ia, cur_ia),
            "top3_overlap": _topk_overlap(base_i, cur_i, pids, k=3, largest=True),
        }

    # 5. 分项留一消融（7 个分项逐一置零）
    component_ablation = {}
    for key in ("s_envelope", "s_light", "s_composition"):
        aw = {k: (0.0 if k == key else 0.5) for k in ("s_envelope", "s_light", "s_composition")}
        sp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, base_leaves, arch_targets=arch_targets, arch_weights=aw, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in sp], dtype=float)
        component_ablation[f"drop_{key}"] = {
            "spearman_i_vs_primary": safe_spearman(base_i, cur_i),
            "kendall_i_vs_primary": safe_kendall(base_i, cur_i),
            "max_abs_diff_i": float(np.max(np.abs(cur_i - base_i))),
        }
    for key in ("s_angle", "s_proj", "s_entropy", "s_far_dz"):
        lw = {k: (0.0 if k == key else 1.0 / 3.0) for k in ("s_angle", "s_proj", "s_entropy", "s_far_dz")}
        sl = compute_leaf_ideotype_scores(leaf_rows, leaf_targets=leaf_targets, leaf_weights=lw)
        sp = compute_plant_ideotype_scores(
            plant_rows, envelope_rows, sl, arch_targets=arch_targets, arch_weights=arch_weights, alpha=alpha
        )
        cur_i = np.array([p["composite_index_i"] for p in sp], dtype=float)
        cur_il = np.array([p["leaf_index_il"] for p in sp], dtype=float)
        component_ablation[f"drop_{key}"] = {
            "spearman_i_vs_primary": safe_spearman(base_i, cur_i),
            "spearman_il_vs_primary": safe_spearman(base_il, cur_il),
            "kendall_i_vs_primary": safe_kendall(base_i, cur_i),
            "max_abs_diff_i": float(np.max(np.abs(cur_i - base_i))),
        }

    return {
        "ia_il_spearman": safe_spearman(base_ia, base_il),
        "alpha_sweep": alpha_sweep,
        "weight_presets": preset_results,
        "dirichlet_random_weights": {
            "n_draws": int(n_dirichlet),
            "concentration": 6.0,
            "alpha_range": [0.35, 0.65],
            "spearman_i_median": float(np.median(dir_rhos)),
            "spearman_i_q25": float(np.quantile(dir_rhos, 0.25)),
            "spearman_i_min": float(np.min(dir_rhos)),
            "kendall_i_median": float(np.median(dir_taus)),
            "spearman_i_samples": [float(v) for v in dir_rhos],
        },
        "composition_target_sensitivity": comp_target_results,
        "component_ablation": component_ablation,
    }
