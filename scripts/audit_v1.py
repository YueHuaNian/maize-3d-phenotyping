"""Read-only validation of batch artifacts; write a new acceptance report."""
import argparse
import json
import os
from pathlib import Path
import sys
from collections import Counter
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".vendor")]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import numpy as np
from PIL import Image,ImageDraw
from maize_leaf_v1.model import LeafModel, triangle_areas
from maize_leaf_v1.data import sha256
from maize_leaf_v1.export import json_write
from maize_leaf_v1.pipeline import fresh_output


def read_ply(path):
    with open(path,encoding="ascii") as f:
        header=[]
        for line in f:
            header.append(line.strip())
            if line.strip()=="end_header":break
        nv=int(next(x.split()[-1] for x in header if x.startswith("element vertex")))
        nf=next((int(x.split()[-1]) for x in header if x.startswith("element face")),0)
        v=np.array([[float(x) for x in f.readline().split()] for _ in range(nv)])
        faces=np.array([[int(x) for x in f.readline().split()] for _ in range(nf)],dtype=int)
        assert not f.read().strip(),f"Unexpected trailing PLY rows: {path}"
    return v,faces


class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        self.paths.extend(v for k,v in attrs if k in ("href","src") and v and not v.startswith(("#","http:","https:","data:")))


def audit(root,report_dir=None):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    validation=json.loads((root/"input_validation.json").read_text(encoding="utf-8"))
    output=fresh_output(report_dir or root/"validation"/"artifact_audit")
    records=[]; issues=[]
    for pair in validation["pairs"]:
        for ext in ("txt","csv"):
            if sha256(pair[ext])!=pair[ext+"_sha256"]:
                issues.append(f"Input hash changed: {pair[ext]}")
        plant=pair["plant"]
        folder=root/f"plant_{plant}"
        stem=json.loads((folder/"stem_centerline.json").read_text())
        assert np.isfinite(stem["xyz"]).all()
        expected=[int(x) for x in pair["labels"] if int(x)>0]
        whole,whole_f=read_ply(folder/"whole_plant_mesh.ply")
        assert set(whole[:,-1].astype(int))==set(expected)
        expected_faces=2*(summary["config"]["nu"]-1)*(summary["config"]["nv"]-1)
        assert len(whole_f)==len(expected)*expected_faces
        for lid in expected:
            key=f"leaf_{lid:02d}"
            leaf=folder/"leaves"
            try:
                for suffix in ("_points.ply","_surface.ply","_surface.obj","_surface.mtl","_visualization.html","_visualization.png","_midrib.json","_width.json","_frames.json","_regions.json","_metrics.json","_model.npz","_model.json"):
                    assert (leaf/(key+suffix)).stat().st_size>0, suffix
                metric=json.loads((leaf/f"{key}_metrics.json").read_text())
                regions=json.loads((leaf/f"{key}_regions.json").read_text())
                model=LeafModel.load(leaf/f"{key}_model.npz")
                spec=json.loads((leaf/f"{key}_model.json").read_text())
                np.testing.assert_allclose(spec["q"]["training_uv"],model.rbf_xy,atol=1e-12)
                np.testing.assert_allclose(spec["q"]["training_normalized_displacement"],model.rbf_d,atol=1e-12)
                frame=json.loads((leaf/f"{key}_frames.json").read_text())
                for symbol,arr in zip(("t","b","n"),model.frame(np.asarray(frame["u"]))):
                    np.testing.assert_allclose(frame[symbol],arr,atol=1e-10)
                verts,face=read_ply(leaf/f"{key}_surface.ply")
                raw,_=read_ply(leaf/f"{key}_points.ply")
                assert len(raw)==pair["labels"][str(lid)]
                assert metric["leaf_id"]==lid and metric["plant"]==plant
                assert np.isfinite(verts).all()
                assert np.all(face[:,0]==3)
                assert np.all(verts[:,-1]==lid)
                assert face[:,1:4].min()>=0 and face[:,1:4].max()<len(verts)
                area=triangle_areas(verts[:,:3],face[:,1:4])
                assert np.all(area>1e-16),"degenerate mesh face"
                reg=np.asarray(regions["vertex_regions"])
                np.testing.assert_array_equal(reg,verts[:,6].astype(int))
                np.testing.assert_array_equal(reg[face[:,1:4]].max(axis=1),face[:,4])
                uv=np.asarray(regions["uv"])
                assert np.all(reg[uv[:,0]>1+1e-10]==2)
                np.testing.assert_allclose(model.evaluate(uv),verts[:,:3],atol=1e-7,rtol=1e-7)
                t,b,n=model.frame(np.linspace(0,1+model.extension,300))
                for x in (t,b,n):
                    np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-8)
                np.testing.assert_allclose(np.sum(t*b,axis=1),0,atol=1e-8)
                for k,name in {0:"observed",1:"interpolated",2:"extrapolated"}.items():
                    np.testing.assert_allclose(area[face[:,4]==k].sum()/area.sum(),metric["area_fractions"][name],atol=1e-6)
                extent_ratio=float(np.linalg.norm(np.ptp(verts[:,:3],axis=0))/np.linalg.norm(np.ptp(raw[:,:3],axis=0)))
                assert extent_ratio<2,"surface extent exceeds twice the observed extent"
                obj=(leaf/f"{key}_surface.obj").read_text()
                assert sum(l.startswith("v ") for l in obj.splitlines())==len(verts)
                assert sum(l.startswith("f ") for l in obj.splitlines())==len(face)
                records.append(dict(plant=plant,leaf_id=lid,passed=True,extent_ratio=extent_ratio,
                    min_triangle_area=float(area.min()),quality=metric["quality"],flags=metric["flags"],
                    completion_ratio=metric["completion_ratio"],area_fractions=metric["area_fractions"],
                    fit_rmse_over_max_width=metric["fit_rmse_over_max_width"]))
            except Exception as exc:
                issues.append(f"{plant}/{key}: {exc}")
    pages=[root/"index.html"]+list(root.glob("plant_*/*.html"))+list(root.glob("plant_*/leaves/*.html"))
    for p in pages:
        parser=Links();parser.feed(p.read_text(encoding="utf-8"))
        for rel in parser.paths:
            if not (p.parent/rel).is_file():issues.append(f"Broken link {p}: {rel}")
        if p.name!="index.html" and ('"mesh3d"' not in p.read_text(encoding="utf-8") or 'Plotly.newPlot' not in p.read_text(encoding="utf-8")):
            issues.append(f"Missing Plotly mesh in {p}")
    if len(records)!=summary["successful_leaves"] or len(records)!=90 or len(validation["pairs"])!=14:
        issues.append("Batch coverage is not 14 plants / 90 leaves")
    quality=Counter(r["quality"] for r in records)
    flags=Counter(f for r in records for f in r["flags"])
    report=dict(status="passed" if not issues else "failed",plants=len(validation["pairs"]),
        leaves=len(records),html_pages_checked=len(pages),quality_counts=dict(quality),flag_counts=dict(flags),
        checks="input SHA256; complete files; PLY/OBJ topology; finite positive face area; original IDs; orthonormal frames; model round trip; region rules; area fractions; extent bounds; local HTML links and Plotly payload",
        browser_interaction="not verified: in-app browser blocks local file URLs",
        software_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted(
            list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py"))+
            list((ROOT/"tests").glob("*.py"))+[ROOT/"requirements.txt",ROOT/"run_v1.ps1"])},
        issues=issues,records=records)
    json_write(output/"acceptance.json",report)
    # Contact sheet is a visual QA artifact, not a replacement for individual PNGs.
    sheet=Image.new("RGB",(1600,4*335),(243,246,245));draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(sorted(root.glob("plant_*/whole_plant.png"))):
        im=Image.open(p).convert("RGB");im.thumbnail((390,305))
        x=(i%4)*400; y=(i//4)*335
        sheet.paste(im,(x+(400-im.width)//2,y+25))
        draw.text((x+12,y+6),p.parent.name,fill=(20,55,42))
    sheet.save(output/"whole_plants_contact_sheet.png")
    print(json.dumps({k:v for k,v in report.items() if k not in ("records","software_sha256")},ensure_ascii=False,indent=2))
    print("AUDIT",output)
    return not issues


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("output",type=Path,nargs="?",default=ROOT/"outputs/leaf_modeling_v1_0")
    ap.add_argument("--report-dir",type=Path)
    args=ap.parse_args()
    raise SystemExit(0 if audit(args.output,args.report_dir) else 1)
