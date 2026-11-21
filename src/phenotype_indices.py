"""Compute baseline plant architecture and leaf morphology indices."""

from collections import defaultdict
from hashlib import sha256
from math import isfinite, log2
from statistics import median

import numpy as np

from .io_utils import dump_json
from .phenotype_features import ENTROPY_EDGES
from .plant_clustering import FIELDS, MIN_PROFILE_SECTIONS
from .tables import save_csv


METHOD = "pai-lmi-baseline-v1"
SCORE_SCALE = 100
ENTROPY_MAX = log2(len(ENTROPY_EDGES) - 1)
MIN_PAI_LEAVES = 3
HEIGHT_QUANTILES = (0.10, 0.90)
LMI_WEIGHTS = {"entropy_component": 1 / 3, "reach_component": 1 / 3,
               "droop_component": 1 / 3}
PAI_WEIGHTS = {"median_reach_component": 0.5, "height_band_component": 0.5}
LEAF_FIELDS = (
    "plant_id", "leaf_id", "nearest_pair_status", "nearest_stem_z",
    "attachment_height_ratio", "height_status", "profile_sections", *FIELDS,
    *LMI_WEIGHTS, "lmi", "lmi_status", "lmi_issue",
)
INPUTS = ("input_manifest.csv", "leaf_features.csv", "plant_morphotypes/leaf_clusters.csv",
          "plant_morphotypes/plant_summary.csv")


def _finite(value):
    return value is not None and isfinite(value)


def _nonnegative(value):
    return _finite(value) and value >= 0


def _bounded_ratio(value):
    return value / (1 + value)


def _score(components, weights):
    return SCORE_SCALE * sum(components[name] * weight for name, weight in weights.items())


def _leaf_index(leaf, profile, stem_lower, stem_span):
    row = dict.fromkeys(LEAF_FIELDS)
    row.update(plant_id=leaf["plant_id"], leaf_id=leaf["leaf_id"],
               nearest_pair_status=leaf["status"], nearest_stem_z=leaf["stem_z"],
               profile_sections=profile.get("profile_sections"),
               **{field: profile.get(field) for field in FIELDS})
    if leaf["status"] != "ok" or not _finite(leaf["stem_z"]):
        row["height_status"] = "NO_NEAREST_STEM_POINT"
    elif stem_span is None:
        row["height_status"] = "INVALID_STEM_SPAN"
    else:
        row.update(attachment_height_ratio=(leaf["stem_z"] - stem_lower) / stem_span,
                   height_status="ok")

    entropy, reach, drop, downward = (row[field] for field in FIELDS)
    entropy_ok = _nonnegative(entropy) and entropy <= ENTROPY_MAX + 1e-12
    downward_ok = _nonnegative(downward) and downward <= 1
    if _nonnegative(reach):
        row["reach_component"] = _bounded_ratio(reach)
    profile_supported = (row["profile_sections"] or 0) >= MIN_PROFILE_SECTIONS
    if profile_supported and entropy_ok:
        row["entropy_component"] = min(entropy / ENTROPY_MAX, 1.0)
    if profile_supported and _nonnegative(drop) and downward_ok:
        row["droop_component"] = (_bounded_ratio(drop) + downward) / 2

    missing = [field for field, valid in zip(FIELDS, (entropy_ok, _nonnegative(reach),
                                                      _nonnegative(drop), downward_ok))
               if not valid]
    if leaf["status"] != "ok":
        row.update(lmi_status="NO_NEAREST_LEAF_POINT", lmi_issue=leaf["status"])
    elif stem_span is None:
        row["lmi_status"] = "INVALID_STEM_SPAN"
    elif not profile_supported:
        row["lmi_status"] = "INSUFFICIENT_PROFILE_SECTIONS"
    elif missing:
        row.update(lmi_status="INVALID_PROFILE_FEATURES", lmi_issue="|".join(missing))
    else:
        row.update(lmi_status="ok", lmi=_score(row, LMI_WEIGHTS))
    return row


def evaluate_plant(record, leaves, profiles, stem):
    """Return the leaf scores and the plant summary for one manifest record."""
    lower, upper = stem["stem_z_q01"], stem["stem_z_q99"]
    stem_span = upper - lower if _finite(lower) and _finite(upper) and upper > lower else None
    if not _finite(stem_span):
        stem_span = None
    rows = [_leaf_index(leaf, profiles.get(leaf["leaf_id"], {}), lower, stem_span) for leaf in leaves]
    heights = [row["attachment_height_ratio"] for row in rows if row["height_status"] == "ok"]
    reaches = [row["reach_component"] for row in rows if row["reach_component"] is not None]
    lmis = [row["lmi"] for row in rows if row["lmi_status"] == "ok"]
    n = len(rows)
    summary = {
        "plant_id": record["plant_id"], "input_status": record["input_status"],
        "leaf_count": n if record["input_status"] == "ok" else None, "stem_z_span": stem_span,
        "height_valid_leaves": len(heights), "reach_valid_leaves": len(reaches),
        "median_reach_component": median(reaches) if reaches else None,
        "attachment_height_q10": None, "attachment_height_q90": None,
        "height_band_component": None, "pai": None, "pai_status": "ok",
        "lmi_valid_leaves": len(lmis), "lmi_coverage": len(lmis) / n if n else None,
        "lmi_median": median(lmis) if lmis else None,
        "lmi_summary_status": ("unavailable" if not lmis else "complete" if len(lmis) == n else "partial"),
    }
    if len(heights) >= MIN_PAI_LEAVES:
        low, high = np.quantile(heights, HEIGHT_QUANTILES)
        summary.update(attachment_height_q10=float(low), attachment_height_q90=float(high),
                       height_band_component=_bounded_ratio(float(high - low)))
    if record["input_status"] != "ok":
        summary["pai_status"] = "INPUT_FAILED"
    elif n < MIN_PAI_LEAVES:
        summary["pai_status"] = "TOO_FEW_LEAVES"
    elif stem_span is None:
        summary["pai_status"] = "INVALID_STEM_SPAN"
    elif len(heights) != n or len(reaches) != n:
        summary["pai_status"] = "INCOMPLETE_HEIGHT_OR_REACH"
    else:
        summary["pai"] = _score(summary, PAI_WEIGHTS)
    return rows, summary


def _report(directory, leaves, plants):
    pai = [row["pai"] for row in plants if row["pai_status"] == "ok"]
    lmi = [row["lmi"] for row in leaves if row["lmi_status"] == "ok"]

    def describe(values):
        return (f"中位数 {median(values):.2f}，范围 {min(values):.2f}–{max(values):.2f}。"
                if values else "暂无有效值。")

    def display(value):
        return f"{value:.2f}" if value is not None else "—"

    lines = [
        "# PAI 株型指数与 LMI 叶形指数", "",
        f"方法版本 {METHOD}。PAI 有效 {len(pai)}/{len(plants)} 株。{describe(pai)}",
        f"LMI 有效 {len(lmi)}/{len(leaves)} 片叶。{describe(lmi)}",
        "LMI 的批次描述按有效叶汇总，植株是独立观测单位。", "",
        "PAI 综合横向伸展和叶基高度分布。LMI 综合观测方向熵、单叶伸展和下垂。"
        "两个分数均为项目定义的 0–100 描述性指数。数值大小描述几何特征表达程度。", "",
        "| 植株 | 叶片数 | PAI | PAI 状态 | LMI 中位数 | 有效 LMI 叶片数 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in plants:
        n = row["leaf_count"] if row["leaf_count"] is not None else "—"
        lines.append(f"| {row['plant_id']} | {n} | {display(row['pai'])} | {row['pai_status']} | "
                     f"{display(row['lmi_median'])} | {row['lmi_valid_leaves']} |")
    lines.extend(["", "[逐株分量与覆盖率](plant_indices.csv) · [逐叶分量与质量状态](leaf_indices.csv) · "
                  "[参数和输入哈希](provenance.json)", ""])
    (directory / "PAI_LMI_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_indices(out, manifest, leaves, profiles, stems, provenance):
    leaves_by_plant = defaultdict(list)
    profiles_by_plant = defaultdict(dict)
    for leaf in leaves:
        leaves_by_plant[leaf["plant_id"]].append(leaf)
    for profile in profiles:
        profiles_by_plant[profile["plant_id"]][profile["leaf_id"]] = profile
    stems_by_plant = {row["plant_id"]: row for row in stems}

    leaf_rows, plant_rows = [], []
    for record in manifest:
        plant = record["plant_id"]
        rows, summary = evaluate_plant(record, leaves_by_plant[plant],
                                       profiles_by_plant[plant], stems_by_plant[plant])
        leaf_rows.extend(rows)
        plant_rows.append(summary)
    directory = out / "indices"
    directory.mkdir()
    save_csv(directory / "leaf_indices.csv", leaf_rows, LEAF_FIELDS)
    save_csv(directory / "plant_indices.csv", plant_rows)
    dump_json(directory / "provenance.json", {
        "method": METHOD, "scale": SCORE_SCALE, "lmi_weights": LMI_WEIGHTS,
        "pai_weights": PAI_WEIGHTS, "droop_subcomponent_weights": [0.5, 0.5],
        "entropy_max_bits": ENTROPY_MAX, "height_quantiles": HEIGHT_QUANTILES,
        "min_pai_leaves": MIN_PAI_LEAVES, "min_profile_sections": MIN_PROFILE_SECTIONS,
        "run_at": provenance["run_at"], "git_head": provenance["git_head"],
        "python": provenance["python"], "packages": provenance["packages"],
        "sha256_text_source_lf": provenance["sha256_text_source_lf"],
        "input_table_sha256_lf": {
            name: sha256((out / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in INPUTS
        },
        "input_point_cloud_sha256": {
            row["plant_id"]: {name: row[name] for name in ("csv_sha256", "txt_sha256")}
            for row in manifest
        },
    })
    _report(directory, leaf_rows, plant_rows)
