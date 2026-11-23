"""Known synthetic ground truth: explicitly withhold a hole, edge and tip."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/".vendor"),str(ROOT/"tests")]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from test_modeling import synthetic
from maize_leaf_v1.model import fit_leaf,Config,COLORS,REGIONS
from maize_leaf_v1.export import scatter,mesh_traces,page,ply,json_write,export_model_metadata
from maize_leaf_v1.pipeline import fresh_output


def main():
    base=ROOT/"outputs/leaf_modeling_v1_0"
    out=fresh_output(base/"validation"/"synthetic_demo")
    truth,missing,hole,stem=synthetic()
    observed=truth[~missing]
    result=fit_leaf(observed,stem,Config(nu=100,nv=31),7)
    tree=cKDTree(result["vertices"])
    metrics={"source":"deterministic synthetic fixture in tests/test_modeling.py",
        "truth_points":len(truth),"input_points":len(observed),"withheld_points":int(missing.sum()),
        "withheld_hole_points":int(hole.sum()),
        "hole_distance_median":float(np.median(tree.query(truth[hole])[0])),
        "hole_distance_p95":float(np.quantile(tree.query(truth[hole])[0],.95)),
        "all_withheld_distance_p95":float(np.quantile(tree.query(truth[missing])[0],.95)),
        "scope":"synthetic unit-scale geometry; nearest sampled-grid distance, not real-maize accuracy",
        "fit_metrics":result["metrics"]}
    json_write(out/"metrics.json",metrics)
    ply(out/"synthetic_truth.ply",truth)
    ply(out/"synthetic_observed.ply",observed)
    ply(out/"synthetic_surface.ply",result["vertices"],result["faces"],COLORS[result["regions"]],result["regions"],result["face_regions"])
    result["model"].save(out/"synthetic_model.npz")
    export_model_metadata(out,"synthetic",result["model"],result["u"])
    fig=make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"scene"}]],subplot_titles=["Input: hole / edge / tip withheld","Completed continuous surface"])
    fig.add_trace(scatter(observed,"Synthetic input", "#334c42"),row=1,col=1)
    for trace in mesh_traces(result):fig.add_trace(trace,row=1,col=2)
    fig.add_trace(scatter(observed,"Input overlay", "#334c42",size=1),row=1,col=2)
    fig.update_scenes(aspectmode="data",camera=dict(eye=dict(x=.5,y=-1.5,z=2)))
    fig.update_layout(height=650,margin=dict(l=0,r=0,t=40,b=0),legend=dict(orientation="h",y=-.05))
    body='<header><h1>合成曲面补全验证 · 明确删除孔洞、叶缘和叶尖</h1><a href="../../index.html">返回真实玉米数据</a></header><main><div class="card"><p>这是已知真值的人工曲面测试，不是实测玉米叶。左侧仅使用删除后的输入，右侧重新拟合连续曲面；黄色为内部插值，粉色为范围外推。隐藏真值未参与拟合。</p>'
    body+=f'<p>删除 {missing.sum()} / {len(truth)} 个点，其中孔洞 {hole.sum()} 个点。留出孔洞点到曲面网格的距离 P95 = {metrics["hole_distance_p95"]:.5f}（合成单位）。此结果不能当作真实玉米补全精度。</p></div>'
    body+='<div class="card">'+fig.to_html(full_html=False,include_plotlyjs=False,config={"displaylogo":False})+'</div><div class="card"><img src="completion_comparison.png"><p><a href="metrics.json">完整数值与参数</a></p></div></main>'
    page("合成缺失补全验证",body,out/"index.html","../../assets/plotly.min.js")
    figure=plt.figure(figsize=(13,6),dpi=140)
    for i in range(2):
        ax=figure.add_subplot(1,2,i+1,projection="3d")
        ax.scatter(*observed.T,s=2,color="#334c42",alpha=.7)
        if i:
            ax.add_collection3d(Poly3DCollection(result["vertices"][result["faces"]],facecolors=COLORS[result["face_regions"]]/255,edgecolor="none",alpha=.90))
        ax.set(xlim=(-.03,1.03),ylim=(-.10,.10),zlim=(-.02,.25),xlabel="X",ylabel="Y",zlabel="Z",
               title="Input: withheld hole, edge and tip" if i==0 else "Reconstructed continuous surface")
        ax.set_box_aspect([1,.32,.32]);ax.view_init(60,-70)
    figure.legend(handles=[Patch(color=COLORS[k]/255,label=REGIONS[k]) for k in REGIONS],loc="lower center",ncol=3)
    figure.suptitle("Synthetic test only: withheld ground truth is not used for fitting",fontsize=12)
    figure.savefig(out/"completion_comparison.png",bbox_inches="tight")
    plt.close(figure)
    print(out)
    print(metrics)


if __name__=="__main__":main()
