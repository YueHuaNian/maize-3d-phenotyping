"""Plant-level 3D diagnostics and local candidate-region panels."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .attachment import METHODS
from .stem import tangent_at
from .visualization import configure, color, setup_3d, save_figure


MARKER = dict(zip((*METHODS,'surface_connected_region'),('o','s','^','D','x')))


def ring(center: np.ndarray, tangent: np.ndarray, radius: float) -> np.ndarray:
    reference = np.eye(3)[np.argmin(abs(tangent))]
    u = np.cross(tangent,reference)
    u /= np.linalg.norm(u)
    v = np.cross(tangent,u)
    theta = np.linspace(0,2*np.pi,38)
    return center+radius*(np.cos(theta[:,None])*u+np.sin(theta[:,None])*v)


def plant_plots(out: Path, sample: str, xyz: np.ndarray, labels: np.ndarray,
                stem: dict, surface: dict, leaves: list[dict]) -> None:
    configure()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(projection='3d',computed_zorder=False)
    stem_points = xyz[labels==0]
    ax.scatter(*stem_points.T,c='#888888',s=3,alpha=.35,depthshade=False,zorder=1)
    ax.plot(*stem['curve'].T,c='#222222',lw=2,zorder=4)
    for i in np.flatnonzero(surface['supported'])[::14]:
        cross = ring(surface['center'][i],tangent_at(stem,surface['arc'][i]),surface['radius'][i])
        ax.plot(*cross.T,c='#0072b2',lw=.9,alpha=.85,zorder=3)
    for leaf in leaves:
        label = leaf['leaf_id']
        tag = str(label)
        points = xyz[labels==label]
        ax.scatter(*points.T,c=color(label),s=2,alpha=.21,zorder=1,depthshade=False)
        if 'curve' in leaf['geometry']:
            line = leaf['geometry']['curve']
            ax.plot(*line.T,c=color(label),lw=1.5,zorder=4)
            ax.text(*line[-1],tag,fontsize=8,zorder=7)
        candidate = leaf['attachment_v5']['selected']
        if candidate is not None:
            ax.scatter(*candidate['xyz'],c='#111111',marker='x',s=25,zorder=5)
    setup_3d(ax,xyz[labels>=0])
    ax.set_title(f'{sample} · v1.0 诊断几何')
    fig.text(.07,.04,'原始坐标单位未确认。蓝圈为拟合茎表面，彩线为可见中线，黑叉为首选叶基候选。',fontsize=9)
    save_figure(fig,out/'plant_geometry_3d.png')
    for page,start in enumerate(range(0,len(leaves),6),1):
        fig,axes = plt.subplots(2,3,figsize=(12,7.5),squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax,leaf in zip(axes.ravel(),leaves[start:start+6]):
            ax.set_visible(True)
            label = leaf['leaf_id']
            record = leaf['attachment_v5']
            evidence = leaf['gap_evidence']
            choices = [item for item in record['candidates'] if 'nominal_rank' in item]
            center = record['selected']['xyz'] if record['selected'] else evidence['raw_stem_xyz']
            anchors = [center,evidence['raw_leaf_xyz'],evidence['raw_stem_xyz']]
            if 'proximal_xyz' in leaf['geometry']:
                anchors.append(leaf['geometry']['proximal_xyz'])
            anchors.extend(item['xyz'] for item in choices)
            anchor = np.asarray(anchors)
            pad = max(record['native_spacing']*6,.015)
            low,high = anchor.min(0)-pad,anchor.max(0)+pad
            visible = np.all((xyz>=low)&(xyz<=high),axis=1)
            span = np.ptp(anchor[:,:2],axis=0)
            horizontal = int(np.argmax(span))
            for tag,shade in [(0,'#888888'),(label,color(label))]:
                pts = xyz[visible & (labels==tag)]
                ax.scatter(pts[:,horizontal],pts[:,2],s=4,c=shade,alpha=.32)
            line = stem['curve']
            select = (line[:,2]>=low[2])&(line[:,2]<=high[2])
            ax.plot(line[select,horizontal],line[select,2],color='#222222',lw=1)
            if 'curve' in leaf['geometry']:
                line = leaf['geometry']['curve']
                select = np.all((line>=low)&(line<=high),axis=1)
                display = np.where(select[:,None],line,np.nan)
                ax.plot(display[:,horizontal],display[:,2],c=color(label),lw=1.4)
            for candidate in choices:
                point = candidate['xyz']
                ax.scatter(point[horizontal],point[2],c='#a52423',s=26,marker=MARKER[candidate['method']],zorder=4)
            if record['selected']:
                ax.scatter(center[horizontal],center[2],facecolors='none',edgecolors='black',s=85,zorder=5)
            a,b = evidence['raw_leaf_xyz'],evidence['raw_stem_xyz']
            ax.plot([a[horizontal],b[horizontal]],[a[2],b[2]],c='#087f8c',ls=':',lw=2)
            ax.set(title=f'标签 {label} · {record["attachment_confidence_diagnostic"]}',
                   xlabel='XY'[horizontal],ylabel='Z')
            ax.tick_params(labelsize=8)
        fig.suptitle(f'{sample} · 叶基候选与原始叶茎近邻')
        legend = [Line2D([],[],c='#a52423',marker=MARKER[method],ls='',label=title) for method,title in
                  zip((*METHODS,'surface_connected_region'),('表面最近','连通近端','曲线反延','切线反延','表面连通区'))]
        legend.append(Line2D([],[],c='#087f8c',ls=':',label='原始叶茎最近点对'))
        fig.legend(handles=legend,ncol=3,loc='lower center',bbox_to_anchor=(.5,.026),frameon=False,fontsize=9)
        fig.text(.04,.014,'黑圈为首选候选。局部投影使用原始坐标单位，点云仍以独立三维图为准。',fontsize=8)
        fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.88,wspace=.44,hspace=.48)
        suffix = '' if page==1 else f'_{page}'
        save_figure(fig,out/f'attachment_candidate_distribution{suffix}.png')
