import csv
from datetime import datetime, timezone
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import time
import traceback
from .data import discover, load_pair, sha256
from .model import Config, stem_model, fit_leaf
from .export import json_write, ensure_js, export_leaf, export_plant, export_index


def fresh_output(requested):
    p = Path(requested)
    # A staged validation folder alone does not occupy final batch filenames.
    if p.exists() and any(x.name != "validation" for x in p.iterdir()):
        p = p/"runs"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p.mkdir(parents=True, exist_ok=False if not p.exists() else True)
    return p


def run(input_root, output, plant_name=None, leaf_id=None, cfg=None, expect_plants=14, validate_only=False):
    cfg = cfg or Config()
    out = fresh_output(output)
    start = time.monotonic()
    summary = dict(version="1.0.0", started_utc=datetime.now(timezone.utc).isoformat(),
        output=str(out.resolve()), input_root=str(Path(input_root).resolve()), config=asdict(cfg),
        selection=dict(plant=plant_name,leaf=leaf_id), plants=[], failed_leaves=0,
        dependencies={p:importlib.metadata.version(p) for p in ("numpy","scipy","matplotlib","plotly")})
    metrics, reports, errors = [], [], []
    error_file = out/"errors.jsonl"
    error_file.touch(exist_ok=False)

    def error(plant, leaf, exc):
        import json
        record=dict(plant=plant,leaf_id=leaf,error=str(exc),traceback=traceback.format_exc(),
                    time_utc=datetime.now(timezone.utc).isoformat())
        errors.append(record)
        with error_file.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
        print(f"ERROR {plant} leaf={leaf}: {exc}",flush=True)
        return record

    try:
        pairs=discover(input_root)
        if expect_plants is not None and len(pairs)!=expect_plants:
            raise ValueError(f"Expected {expect_plants} pairs, found {len(pairs)}")
        if plant_name and plant_name not in [p[0] for p in pairs]:
            raise ValueError(f"Plant {plant_name} not found")
        if leaf_id is not None and not plant_name:
            raise ValueError("--leaf requires --plant")
        selected=[p for p in pairs if not plant_name or p[0]==plant_name]
        summary["discovered_pairs"]=len(pairs)
        if not validate_only:
            ensure_js(out)
        for name, txt, csv_path in selected:
            ps=dict(plant=name,status="started",success=0,failed=0,stem_success=False)
            summary["plants"].append(ps)
            leaf_ids=[]
            handled=set()
            try:
                plant=load_pair(name,txt,csv_path)
                reports.append(plant.report)
                leaf_ids=sorted(int(x) for x in set(plant.labels) if x>0)
                ps["leaf_count"]=len(leaf_ids)
                ps["original_leaf_ids"]=leaf_ids
                if leaf_id is not None:
                    if leaf_id not in leaf_ids:
                        raise ValueError(f"Requested leaf {leaf_id} missing; available {leaf_ids}")
                    leaf_ids=[leaf_id]
                if validate_only:
                    ps["status"]="validated"
                    continue
                folder=out/f"plant_{name}"
                folder.mkdir()
                stem, stem_json=stem_model(plant.xyz[plant.labels==0])
                json_write(folder/"stem_centerline.json",stem_json)
                ps["stem_success"]=True
                results, failures={},[]
                for lid in leaf_ids:
                    print(f"Fitting {name} leaf {lid} ({sum(plant.labels==lid)} points)",flush=True)
                    try:
                        result=fit_leaf(plant.xyz[plant.labels==lid],stem,cfg,lid)
                        export_leaf(folder/"leaves",plant,lid,result)
                        results[lid]=result
                        metrics.append(result["metrics"])
                        ps["success"]+=1
                    except Exception as exc:
                        record=error(name,lid,exc)
                        failures.append(record)
                        ps["failed"]+=1
                        summary["failed_leaves"]+=1
                        (folder/"leaves").mkdir(exist_ok=True)
                        json_write(folder/"leaves"/f"leaf_{lid:02d}_metrics.json",
                                   dict(plant=name,leaf_id=lid,status="failed",quality="failed",error=str(exc)))
                    handled.add(lid)
                export_plant(folder,plant,stem,results,failures)
                ps["status"]="success" if not failures else "partial_failure"
                # Confirm read-only source protection by hashing again after each plant.
                if sha256(txt)!=plant.report["txt_sha256"] or sha256(csv_path)!=plant.report["csv_sha256"]:
                    raise RuntimeError("Source changed during processing")
                ps["input_hashes_unchanged"]=True
            except Exception as exc:
                error(name,None,exc)
                ps["status"]="failed"
                # A failed stem or plant export must not silently lose leaf IDs.
                for lid in leaf_ids:
                    if lid in handled:
                        continue
                    dependent=RuntimeError(f"Plant dependency failed: {exc}")
                    error(name,lid,dependent)
                    leaf_folder=out/f"plant_{name}"/"leaves"
                    leaf_folder.mkdir(parents=True,exist_ok=True)
                    json_write(leaf_folder/f"leaf_{lid:02d}_metrics.json",dict(
                        plant=name,leaf_id=lid,status="failed",quality="failed",error=str(dependent)))
                    ps["failed"]+=1
                    summary["failed_leaves"]+=1
            json_write(out/"summary.json",summary)
    except Exception as exc:
        error(None,None,exc)
    summary.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-start,successful_leaves=len(metrics),
                   errors=len(errors),status="success" if not errors else "failed_or_partial")
    json_write(out/"input_validation.json",dict(pairs=reports,checked=len(reports),errors=errors))
    json_write(out/"summary.json",summary)
    fields=["plant","leaf_id","status","quality","raw_points","completion_ratio",
            "observed","interpolated","extrapolated","fit_rmse","fit_rmse_over_max_width",
            "surface_area","centerline_length","max_width","flags"]
    with (out/"metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for m in metrics:
            row={k:m.get(k) for k in fields}
            row.update(m["area_fractions"])
            row["flags"]=";".join(m["flags"])
            w.writerow(row)
    if not validate_only:
        export_index(out,summary,metrics)
    print(f"RESULT {out.resolve()} | leaves={len(metrics)}, errors={len(errors)}",flush=True)
    return (0 if not errors else 1),out
