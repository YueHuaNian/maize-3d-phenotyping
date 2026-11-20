"""Publishable tables, figures, and recorded provenance for one V1.0 batch."""

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import platform
import subprocess

import matplotlib
import numpy as np
import scipy

from .io_utils import dump_json
from .nearest_visualization import overview_figure, save_figure


def _describe(values):
    if not values:
        return dict.fromkeys(("min", "q1", "median", "q3", "max", "mean", "sample_sd"))
    array = np.asarray(values)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {
        "min": float(array.min()), "q1": float(q1), "median": float(median), "q3": float(q3),
        "max": float(array.max()), "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
    }


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_batch_outputs(output, results, audits):
    _write_csv(output / "nearest_attachment_results.csv", list(results[0]), results)
    _write_csv(output / "input_manifest.csv", list(audits[0]), audits)
    successful = [r for r in results if r["status"] == "ok"]
    stats = _describe([r["nearest_distance"] for r in successful])
    bounds = (None, None)
    outliers = []
    if successful:
        iqr = stats["q3"] - stats["q1"]
        bounds = (stats["q1"] - 1.5 * iqr, stats["q3"] + 1.5 * iqr)
        for row in successful:
            distance = row["nearest_distance"]
            if distance < bounds[0] or distance > bounds[1]:
                outliers.append({
                    "plant_id": row["plant_id"], "leaf_id": row["leaf_id"],
                    "nearest_distance": distance, "leaf_point_row": row["leaf_point_row"],
                    "stem_point_row": row["stem_point_row"],
                    "fence": "lower" if distance < bounds[0] else "upper",
                })
    _write_csv(output / "distance_outliers.csv",
               ("plant_id", "leaf_id", "nearest_distance", "leaf_point_row", "stem_point_row", "fence"),
               outliers)

    plants = []
    for audit in audits:
        plant_id = audit["plant_id"]
        plant_rows = [r for r in results if r["plant_id"] == plant_id]
        leaves = [r for r in plant_rows if r["leaf_id"] is not None]
        distances = [r["nearest_distance"] for r in plant_rows if r["status"] == "ok"]
        plants.append({
            "plant_id": plant_id, "point_rows": audit["point_rows"],
            "invalid_xyz_rows": audit["invalid_xyz_rows"], "input_status": audit["status"],
            "leaf_count": len(leaves) if audit["status"] == "ok" else None,
            "localized_leaves": len(distances), "failed_leaves": len(leaves) - len(distances),
            **_describe(distances),
        })
    _write_csv(output / "plant_summary.csv", list(plants[0]), plants)

    metadata = {
        "plants": len(plants), "validated_pairs": sum(r["status"] == "ok" for r in audits),
        "csv_files_seen": sum(r["csv_file"] is not None for r in audits),
        "txt_files_seen": sum(r["txt_file"] is not None for r in audits),
        "point_rows": sum(row["point_rows"] or 0 for row in audits),
        "invalid_xyz_rows": sum(row["invalid_xyz_rows"] or 0 for row in audits),
        "leaves": sum(row["leaf_count"] or 0 for row in plants),
        "successful": len(successful), "failed_leaves": sum(row["failed_leaves"] for row in plants),
        "failed_plant_records": sum(row["leaf_id"] is None for row in results),
        "distance_unit": "original coordinate units (unconfirmed physical unit)",
        "distance_summary": stats,
        "outlier_fences": {"lower": bounds[0], "upper": bounds[1]},
        "outliers": len(outliers),
    }
    dump_json(output / "distance_summary.json", metadata)
    save_figure(overview_figure(results, [row["plant_id"] for row in plants], bounds),
                output / "nearest_distance_overview.png")
    _write_report(output, metadata, plants, outliers, [r for r in results if r["status"] != "ok"])


def _write_report(output, counts, plants, outliers, failures):
    root = Path(__file__).resolve().parents[1]
    files = ["src/nearest_attachment.py", "src/nearest_report.py", "src/nearest_visualization.py",
             "src/io_utils.py", "requirements.txt"]
    hashes = {
        path: sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in files
    }
    commit = "不可用"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    date = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    stats = counts["distance_summary"]
    fences = counts["outlier_fences"]

    def fmt(value):
        return f"{value:.8g}" if value is not None else "—"

    plant_lines = [
        f"| {r['plant_id']} | {r['leaf_count'] if r['leaf_count'] is not None else '—'} | "
        f"{r['localized_leaves']} | {r['failed_leaves']} | {fmt(r['median'])} | {fmt(r['max'])} | "
        + (f"[三维图](visualizations/{r['plant_id']}.png) |" if r["input_status"] == "ok" else "— |")
        for r in plants
    ]
    abnormal_lines = [
        f"| {r['plant_id']} | {r['leaf_id']} | {fmt(r['nearest_distance'])} | "
        f"{r['leaf_point_row']} | {r['stem_point_row']} |"
        for r in outliers
    ]
    failure_lines = [f"- {r['plant_id']} / {r['leaf_id'] or '—'} · {r['status']}" for r in failures]
    source_lines = [f"- {path} · `{digest}`" for path, digest in hashes.items()]
    report = [
        "# V1.0 全局最近邻叶基定位实验",
        "",
        f"运行时间（北京时间） {date}。运行时 Git HEAD 为 `{commit}`。下方 SHA-256 记录实际执行的代码及依赖清单，文本统一按 LF 换行计算。",
        f"Python {platform.python_version()}，NumPy {np.__version__}，SciPy {scipy.__version__}，Matplotlib {matplotlib.__version__}。",
        "",
        f"读取到 {counts['txt_files_seen']} 个原始 TXT 和 {counts['csv_files_seen']} 个语义 CSV，"
        f"对齐通过 {counts['validated_pairs']}/{counts['plants']} 株，共 {counts['point_rows']:,} 行点数据，"
        f"其中 {counts['invalid_xyz_rows']} 行无有效 XYZ。",
        "输入仅以文件名标识。[输入核验与逐文件 SHA-256](input_manifest.csv)",
        "",
        f"识别 {counts['leaves']} 片叶，定位成功 {counts['successful']} 片，失败 {counts['failed_leaves']} 片。"
        f"另有 {counts['failed_plant_records']} 条植株级失败记录。",
        "行号以 0 为起点，以原始文件首条点数据为第 0 行，CSV 表头和 TXT 的两行头部均排除。",
        "距离单位沿用原始坐标，物理单位待确认。[逐叶结果](nearest_attachment_results.csv)",
        "",
        "## 最近距离分布",
        "",
        "| 最小值 | Q1 | 中位数 | Q3 | 最大值 | 均值 | 样本标准差 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(fmt(stats[key]) for key in ("min","q1","median","q3","max","mean","sample_sd")) + " |",
        "",
        "四分位数采用线性插值。叶片距离用于批次描述，独立植株为统计样本单位。",
        "",
        "![最近邻距离直方图、合并箱线图与逐株比较](nearest_distance_overview.png)",
        "",
        "## 逐株核验",
        "",
        "| 植株 | 识别叶数 | 成功 | 失败叶数 | 距离中位数 | 最大距离 | 原坐标三维图 |",
        "|---|---:|---:|---:|---:|---:|---|",
        *plant_lines,
        "",
        "[逐株统计表](plant_summary.csv) · [机器可读距离统计](distance_summary.json)",
        "",
        "## 距离异常和计算失败",
        "",
        f"以全部成功叶片的 Q1−1.5×IQR 和 Q3+1.5×IQR 为界，"
        f"区间为 [{fmt(fences['lower'])}, {fmt(fences['upper'])}]。共标记 {len(outliers)} 片。",
        "异常叶片全部参与统计。异常表示距离超出本批次分布的围栏，定位准确性须通过独立证据验证。",
        "[异常清单](distance_outliers.csv)",
        "",
        *(["| 植株 | 叶号 | 距离 | 叶点行号 | 茎点行号 |", "|---|---:|---:|---:|---:|",
           *abnormal_lines, ""] if abnormal_lines else []),
        *(["计算失败记录", "", *failure_lines, ""] if failure_lines else []),
        "## 结果解释",
        "",
        "最短点对是初始几何估计。叶片贴茎、遮挡、采样疏密和语义边界均会影响点对位置。"
        "当前批次缺少独立解剖叶基标注，因此定位成功计数表示计算获得有效点对，实际定位误差待后续验证。",
        "",
        "## 运行代码 SHA-256",
        "",
        *source_lines,
        "",
    ]
    (output / "V1.0_EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
