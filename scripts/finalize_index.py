"""Refresh only the current batch landing page, retaining its preceding draft."""
import os
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".vendor")]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
from maize_leaf_v1.export import export_index
out=ROOT/"outputs/leaf_modeling_v1_0"
summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
metrics=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("plant_*/leaves/*_metrics.json"))]
backup=out/"validation"/"index_before_acceptance.html.txt"
if backup.exists():raise FileExistsError("Landing page draft already archived; refusing to overwrite")
backup.write_bytes((out/"index.html").read_bytes())
export_index(out,summary,metrics)
print("Updated current batch landing page; archived preceding draft")
