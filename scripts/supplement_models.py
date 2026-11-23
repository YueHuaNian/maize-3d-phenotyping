"""Add readable model/frame JSON to an existing run without changing meshes."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".vendor")]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import json
from maize_leaf_v1.model import LeafModel
from maize_leaf_v1.export import export_model_metadata

base=ROOT/"outputs/leaf_modeling_v1_0"
count=0
for path in sorted(base.glob("plant_*/leaves/*_model.npz")):
    key=path.name.removesuffix("_model.npz")
    if (path.parent/f"{key}_model.json").exists() or (path.parent/f"{key}_frames.json").exists():
        raise FileExistsError(f"Supplement already exists for {path}; existing artifacts protected")
    midrib=json.loads((path.parent/f"{key}_midrib.json").read_text())
    export_model_metadata(path.parent,key,LeafModel.load(path),midrib["u"])
    count+=1
print(f"Added readable model and frame JSON for {count} leaves; meshes unchanged")
