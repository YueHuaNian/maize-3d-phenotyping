"""Check frozen splits and write paired before/after occlusion-error tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ("held_median", "held_p95", "held_rmse", "held_p95_over_width",
           "length_change_pct", "area_change_pct")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(before_root, after_root):
    roots = [Path(before_root), Path(after_root)]
    reports = [json.loads((root / "holdout_benchmark.json").read_text(encoding="utf-8")) for root in roots]
    keys = lambda r: (r["plant"], r["leaf_id"], r["scenario"])
    tables = [{keys(r): r for r in report["records"]} for report in reports]
    if set(tables[0]) != set(tables[1]):
        raise ValueError("Before/after case lists differ")
    rows, checked = [], 0
    for key in sorted(tables[0]):
        pair = [table[key] for table in tables]
        if all("artifact" in row for row in pair):
            paths = [root / (row["artifact"] + "_split.npz") for root, row in zip(roots, pair)]
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                if np.any(a["observed"] & a["held"]) or np.any(b["observed"] & b["held"]):
                    raise ValueError(f"Hidden points leaked into observed input for {key}")
                for name in ("points", "observed", "held", "reference_vertices", "reference_faces"):
                    if not np.array_equal(a[name], b[name]):
                        raise ValueError(f"Changed {name} for {key}")
            checked += 1
        row = dict(plant=key[0], leaf_id=key[1], scenario=key[2],
                   status_before=pair[0]["status"], status_after=pair[1]["status"],
                   withheld_points=pair[1].get("withheld_points"),
                   tip_completion_status=pair[1].get("tip_completion_status"))
        for metric in METRICS:
            for stage, record in zip(("before", "after"), pair):
                row[f"{metric}_{stage}"] = record.get(metric)
            if all(metric in record for record in pair):
                row[f"{metric}_difference"] = pair[1][metric] - pair[0][metric]
        rows.append(row)
    out = roots[1]
    write_csv(out / "paired_leaf_errors.csv", rows)
    write_csv(out / "failure_cases.csv", [r for r in rows if r["status_after"] in ("alert", "failed", "reference_failed")])
    plants = sorted({r["plant"] for r in rows})
    plant_rows = []
    for plant in plants:
        for scenario in reports[1]["summary"]:
            records = [r for r in rows if r["plant"] == plant and r["scenario"] == scenario]
            row = dict(plant=plant, scenario=scenario, total=len(records))
            for stage in ("before", "after"):
                row[f"alerts_{stage}"] = sum(r[f"status_{stage}"] == "alert" for r in records)
                for metric in METRICS:
                    vals = [r[f"{metric}_{stage}"] for r in records if r[f"{metric}_{stage}"] is not None]
                    row[f"median_{metric}_{stage}"] = float(np.median(vals)) if vals else None
            plant_rows.append(row)
    write_csv(out / "per_plant_errors.csv", plant_rows)
    result = {"paired_cases": len(rows), "identical_frozen_split_cases": checked,
              "split_disjointness_checked_cases": checked,
              "plants": len(plants), "leaves": reports[1]["leaves"],
              "summary": {stage: report["summary"] for stage, report in zip(("before", "after"), reports)},
              "paired_p95_decreased": sum(r.get("held_p95_difference", 0) < -1e-12 for r in rows),
              "paired_p95_increased": sum(r.get("held_p95_difference", 0) > 1e-12 for r in rows),
              "paired_p95_equal": sum(abs(r.get("held_p95_difference", float("inf"))) <= 1e-12 for r in rows),
              "table_sha256": hashlib.sha256((out / "paired_leaf_errors.csv").read_bytes()).hexdigest()}
    result["paired_complete_summary"] = {}
    for scenario in reports[1]["summary"]:
        common = [r for r in rows if r["scenario"] == scenario
                  and r["held_p95_before"] is not None and r["held_p95_after"] is not None]
        entry = {"n": len(common)}
        for stage in ("before", "after"):
            entry[stage] = {metric: float(np.median([r[f"{metric}_{stage}"] for r in common]))
                            if common else None for metric in METRICS}
        result["paired_complete_summary"][scenario] = entry
    (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    compare(args.before, args.after)
