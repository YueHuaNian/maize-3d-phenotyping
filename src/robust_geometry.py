"""V2: robust axial stem spline, surface-graph leaf ordering, explicit diagnostics.

No leaf ID is used to locate a base or fit a line. IDs are used only for the
final independent ordering check. Observed geometry is not anatomical truth.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from .geometry import unit, arc_lengths, spline_fit, project_to_curve, tangent_angle, Parameters


@dataclass
class ParametersV2:
    seed: int = 20261119
    stem_bins: int = 26
    stem_bin_candidates: tuple[int,...] = (18,26,34)
    stem_lambda_candidates: tuple[float,...] = (0.0,0.000001,0.0001,0.01,1.0)
    stem_gate_span_fraction: float = 0.015
    graph_neighbors: int = 10
    graph_gap_factor: float = 6.0
    graph_nodes_max: int = 2200
    leaf_sections: int = 36
    smoothing_rel: float = 0.004
    base_fraction: float = 0.12
    min_fit_points: int = 200
    order_tolerance: float = 0.04


def spline_design(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    knots = np.r_[np.zeros(4),np.linspace(0,1,6)[1:-1],np.ones(4)]
    identity = np.eye(len(knots)-4)
    basis = BSpline(knots,identity,3)
    return basis(u),basis(u,1),basis(u,2),basis(np.linspace(0,1,150),2)


def penalized_fit(u: np.ndarray, values: np.ndarray, weights: np.ndarray, penalty: float) -> tuple[np.ndarray,np.ndarray]:
    design,_,_,rough = spline_design(u)
    normal = design.T @ (weights[:,None]*design)
    matrix = normal+penalty*(rough.T @ rough/len(rough))+np.eye(normal.shape[0])*1e-10
    coefficients = np.linalg.solve(matrix,design.T @ (weights[:,None]*values))
    return coefficients,design @ coefficients


def robust_stem(points: np.ndarray, params: ParametersV2) -> dict:
    if len(points) < 80:
        raise ValueError('V2 茎点不足 80')
    origin = np.median(points,axis=0)
    _,_,axes = np.linalg.svd(points-points.mean(0),full_matrices=False)
    axis = axes[0]
    if axis[2] < 0:
        axis = -axis
    side1 = unit(np.cross(axis,[1.,0,0]) if abs(axis[0]) < .8 else np.cross(axis,[0.,1,0]))
    side2 = unit(np.cross(axis,side1))
    transverse = np.column_stack([(points-origin) @ side1,(points-origin) @ side2])
    axial = (points-origin) @ axis
    low,high = np.quantile(axial,[.005,.995])
    span = float(high-low)
    spacing = float(np.median(cKDTree(points).query(points,k=2)[0][:,1]))
    gate = max(span*params.stem_gate_span_fraction,3*spacing)
    # Equal axial-bin support prevents the dense upper cloud dominating the line.
    bins = np.clip(((axial-low)/span*18).astype(int),0,17)
    rng = np.random.default_rng(params.seed)
    best = (-np.inf,None)
    for _ in range(650):
        i,j = rng.integers(0,len(points),2)
        if abs(axial[i]-axial[j]) < .45*span:
            continue
        slope = (transverse[j]-transverse[i])/(axial[j]-axial[i])
        if np.linalg.norm(slope) > .5:
            continue
        intercept = transverse[i]-axial[i]*slope
        residual = np.linalg.norm(transverse-(intercept+axial[:,None]*slope),axis=1)
        score = np.mean([np.mean(np.maximum(1-(residual[bins==b]/gate)**2,0)) for b in np.unique(bins)])
        if score > best[0]:
            best = (float(score),(intercept,slope))
    if best[1] is None:
        raise ValueError('茎稳健轴初始化失败')
    intercept,slope = best[1]
    prediction = intercept+axial[:,None]*slope
    initial_radial = np.linalg.norm(transverse-prediction,axis=1)
    core = initial_radial <= gate
    all_variants, sections_by_bins = [], {}
    for count in params.stem_bin_candidates:
        edges = np.linspace(low,high,count+1)
        records = []
        for section_id,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            indices = np.flatnonzero((axial >= a)&(axial <= b))
            chosen = indices[core[indices]]
            raw_center = np.median(points[indices],axis=0) if len(indices) else np.full(3,np.nan)
            valid = len(chosen) >= 5
            center = np.median(points[chosen],axis=0) if valid else np.full(3,np.nan)
            records.append({'section':section_id,'axial_lo':a,'axial_hi':b,'point_count':len(indices),
                            'core_count':len(chosen),'valid':valid,'raw_center':raw_center,'center':center,
                            'excluded_fraction':float(1-len(chosen)/len(indices)) if len(indices) else np.nan,
                            'raw_to_core_center_shift':float(np.linalg.norm(raw_center-center)) if valid else np.nan,
                            'abnormal_raw_center':bool(valid and np.linalg.norm(raw_center-center)>gate*.5),
                            'excluded_point_indices':indices[~core[indices]].tolist()})
        valid = [r for r in records if r['valid']]
        if len(valid) < 8:
            continue
        centers = np.asarray([r['center'] for r in valid])
        u = ((centers-origin) @ axis-low)/span
        values = np.column_stack([(centers-origin) @ side1,(centers-origin) @ side2])
        # Huber weights flag displaced centers; do not interpolate them blindly.
        anchor = intercept+(low+span*u)[:,None]*slope
        deviation = np.linalg.norm(values-anchor,axis=1)
        med = np.median(deviation)
        cutoff = max(med+3*1.4826*np.median(abs(deviation-med)),gate*.45)
        weights = np.minimum(1,cutoff/np.maximum(deviation,1e-12))
        for row,w,dev in zip(valid,weights,deviation):
            row.update(center_weight=float(w),center_axis_deviation=float(dev),abnormal_center=bool(w < .8))
        sections_by_bins[count] = (u,values,weights,records,centers)
        for penalty in params.stem_lambda_candidates:
            errors = np.zeros(len(u))
            # Interleaved 4-fold section prediction; same held-out sections for each lambda.
            for fold in range(4):
                held = np.arange(len(u))%4 == fold
                coef,_ = penalized_fit(u[~held],values[~held],weights[~held],penalty)
                design,_,_,_ = spline_design(u[held])
                errors[held] = np.linalg.norm(design @ coef-values[held],axis=1)
            coef,pred = penalized_fit(u,values,weights,penalty)
            grid = np.linspace(u.min(),u.max(),240)
            design,d1,d2,_ = spline_design(grid)
            derivative = axis[None,:]*span+(d1 @ coef)[:,0,None]*side1+(d1 @ coef)[:,1,None]*side2
            tangents = derivative/np.linalg.norm(derivative,axis=1)[:,None]
            angle = np.degrees(np.arccos(np.clip(tangents @ axis,-1,1)))
            all_variants.append({'bins':count,'lambda':penalty,'valid_sections':len(valid),
                                 'cv_mean_distance':float(errors.mean()),'cv_sem':float(errors.std(ddof=1)/np.sqrt(len(errors))),
                                 'center_rmse':float(np.sqrt(np.mean(np.sum((pred-values)**2,axis=1)))),
                                 'tangent_max_deg':float(angle.max()),'core_points':int(core.sum())})
    if not all_variants:
        raise ValueError('没有足够的核心截面拟合茎中心线')
    # Compare smoothing within fixed bin count, then select bin count by prediction.
    selected_by_bins = []
    for count in params.stem_bin_candidates:
        variants = [v for v in all_variants if v['bins']==count]
        if not variants:
            continue
        optimum = min(variants,key=lambda v:v['cv_mean_distance'])
        eligible = [v for v in variants if v['cv_mean_distance'] <= optimum['cv_mean_distance']+optimum['cv_sem']]
        selected_by_bins.append(max(eligible,key=lambda v:v['lambda']))
    selected = min(selected_by_bins,key=lambda v:v['cv_mean_distance'])
    u,values,weights,records,centers = sections_by_bins[selected['bins']]
    coefficients,_ = penalized_fit(u,values,weights,selected['lambda'])
    grid = np.linspace(u.min(),u.max(),240)
    design,d1,d2,_ = spline_design(grid)
    side = design @ coefficients
    curve = origin+(low+span*grid)[:,None]*axis+side[:,0,None]*side1+side[:,1,None]*side2
    derivative = span*axis+(d1 @ coefficients)[:,0,None]*side1+(d1 @ coefficients)[:,1,None]*side2
    second = (d2 @ coefficients)[:,0,None]*side1+(d2 @ coefficients)[:,1,None]*side2
    radial,_,_ = project_to_curve(points,curve)
    radius = float(np.median(radial[core]))
    deviations = np.degrees(np.arccos(np.clip(derivative @ axis/np.linalg.norm(derivative,axis=1),-1,1)))
    supported = np.min(abs(grid[:,None]-u[None,:]),axis=1) <= 1.5/selected['bins']
    return {'curve':curve,'arc':arc_lengths(curve),'d1':derivative,'d2':second,
            'centers':centers,'axis':axis,'radius':radius,'point_count':len(points),
            'pca_z_angle_deg':float(np.degrees(np.arccos(np.clip(axis[2],-1,1)))),
            'local_axis_deviation_max_deg':float(deviations.max()),'fit_rmse':selected['center_rmse'],
            'method':'axial_robust_core_penalized_cubic_spline', 'selected':selected,
            'parameter_comparison':all_variants,'section_records':records,
            'all_section_records':{str(k):v[3] for k,v in sections_by_bins.items()},
            'core_mask':core,'gate_distance':gate,'initial_axis_radial':initial_radial,
            'curve_supported_mask':supported,'unsupported_curve_fraction':float(np.mean(~supported)),
            'occupied_bins':len(centers),'quality_flag':'ESTIMATED_NEEDS_REVIEW',
            'warnings':['离轴茎标签点仅在拟合时降权/排除，原始标签与点保留；几何无法确认其是否叶鞘',
                        '轴向参数单调，不代表真实茎不能弯曲；核心点选择和方向仍需人工核验',
                        '平滑按留出截面预测误差的一标准误规则选择，未以切线偏差最小为目标']}


def voxel_points(points: np.ndarray, resolution: float) -> tuple[np.ndarray,np.ndarray]:
    keys = np.floor((points-points.min(0))/resolution).astype(int)
    _,inverse,counts = np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    nodes = np.column_stack([np.bincount(inverse,weights=points[:,j])/counts for j in range(3)])
    return nodes,counts


def ordered_midline_fit(centers: np.ndarray) -> dict:
    """Low-complexity spline with blocked section prediction; no noisy chord-length parameter."""
    u = np.linspace(0,1,len(centers))
    origin = centers.mean(0)
    values = centers-origin
    weights = np.ones(len(centers))
    variants = []
    for penalty in (1e-6,1e-5,1e-4,.001,.01):
        errors = np.zeros(len(u))
        for fold in range(4):
            held = np.arange(len(u))%4==fold
            coef,_ = penalized_fit(u[~held],values[~held],weights[~held],penalty)
            design,_,_,_ = spline_design(u[held])
            errors[held] = np.linalg.norm(design @ coef-values[held],axis=1)
        variants.append({'lambda':penalty,'cv_mean_distance':float(errors.mean()),
                         'cv_sem':float(errors.std(ddof=1)/np.sqrt(len(errors)))})
    best = min(variants,key=lambda v:v['cv_mean_distance'])
    selected = max((v for v in variants if v['cv_mean_distance'] <= best['cv_mean_distance']+best['cv_sem']),key=lambda v:v['lambda'])
    coef,fitted = penalized_fit(u,values,weights,selected['lambda'])
    design,d1,d2,_ = spline_design(np.linspace(0,1,240))
    curve = origin+design @ coef
    derivative,second = d1 @ coef,d2 @ coef
    speed = np.linalg.norm(derivative,axis=1)
    if np.any(speed < 1e-9):
        raise ValueError('正则中线出现退化导数')
    curvature = np.linalg.norm(np.cross(derivative,second),axis=1)/speed**3
    return {'curve':curve,'d1':derivative,'d2':second,'curvature':curvature,
            'arc':arc_lengths(curve),'centers':centers,
            'fit_rmse':float(np.sqrt(np.mean(np.sum((fitted-values)**2,axis=1)))),
            'smoothing_comparison':variants,'selected_smoothing':selected,'degree':3,
            'parameterization':'ordered geodesic section index; 8 cubic basis coefficients, second-derivative penalty'}


def leaf_graph(points: np.ndarray, params: ParametersV2) -> dict:
    size = float(np.linalg.norm(np.ptp(points,axis=0)))/120
    for _ in range(20):
        nodes,counts = voxel_points(points,max(size,1e-9))
        if len(nodes) <= params.graph_nodes_max:
            break
        size *= 1.2
    distances,neighbors = cKDTree(nodes).query(nodes,k=min(params.graph_neighbors+1,len(nodes)))
    spacing = float(np.median(distances[:,1]))
    start = np.repeat(np.arange(len(nodes)),neighbors.shape[1]-1)
    end,weights = neighbors[:,1:].ravel(),distances[:,1:].ravel()
    valid = weights <= spacing*params.graph_gap_factor
    graph = coo_matrix((weights[valid],(start[valid],end[valid])),shape=(len(nodes),len(nodes))).tocsr()
    graph = graph.maximum(graph.T)
    number,groups = connected_components(graph,directed=False)
    main = int(np.argmax(np.bincount(groups,weights=counts)))
    keep = groups == main
    component = np.flatnonzero(keep)
    if len(component) < 10:
        raise ValueError('叶片主连通分量不足 10 个体素')
    return {'nodes':nodes[component],'weights':counts[component],
            'graph':graph[component][:,component], 'retained_fraction':float(counts[component].sum()/len(points)),
            'spacing':spacing,'voxel_size':size,'component_count':number,
            'other_nodes':nodes[~keep], 'component_sizes_source_points':sorted(np.bincount(groups,weights=counts).astype(int).tolist(),reverse=True)}


def geodesic_midline(points: np.ndarray, stem: dict, params: ParametersV2, *, anchor=None) -> tuple[dict,dict]:
    cloud = leaf_graph(points,params)
    nodes,graph = cloud['nodes'],cloud['graph']
    if anchor is None:
        distance,_,_ = project_to_curve(nodes,stem['curve'])
    else:
        distance = np.linalg.norm(nodes-anchor,axis=1)
    seed = int(np.argmin(distance))
    first = dijkstra(graph,directed=False,indices=seed)
    end1 = int(np.argmax(first))
    from_end1 = dijkstra(graph,directed=False,indices=end1)
    end2 = int(np.argmax(from_end1))
    # Distance is compared at endpoint REGIONS rather than individual noisy extrema.
    extent = float(from_end1[end2])
    band = max(extent*.035,cloud['spacing']*2)
    regions = [np.flatnonzero(from_end1 <= band),np.flatnonzero(from_end1 >= extent-band)]
    endpoint_centers = np.array([np.median(nodes[r],axis=0) for r in regions])
    if anchor is None:
        end_dist,_,_ = project_to_curve(endpoint_centers,stem['curve'])
    else:
        end_dist = np.linalg.norm(endpoint_centers-anchor,axis=1)
    root_end = int(np.argmin(end_dist))
    root_region = regions[root_end]
    root_center = endpoint_centers[root_end]
    root = int(root_region[np.argmin(np.linalg.norm(nodes[root_region]-root_center,axis=1))])
    geo_from_root = dijkstra(graph,directed=False,indices=root)
    tip_node = int(np.argmax(geo_from_root))
    geo_from_tip = dijkstra(graph,directed=False,indices=tip_node)
    # Two-ended coordinate cancels detours common to both paths around holes.
    geo = .5*(geo_from_root-geo_from_tip+geo_from_root[tip_node])
    max_geo = float(np.quantile(geo,.995))
    # Group by intrinsic surface distance rather than one global PCA projection.
    centers,sections = [],[]
    edges = np.linspace(0,max_geo,params.leaf_sections+1)
    for a,b in zip(edges[:-1],edges[1:]):
        subset = np.flatnonzero((geo >= a)&(geo < b))
        if len(subset) < 2:
            continue
        center = np.average(nodes[subset],axis=0,weights=cloud['weights'][subset])
        centers.append(center)
        sections.append({'geodesic_lo':a,'geodesic_hi':b,'nodes':len(subset),
                         'source_points':int(cloud['weights'][subset].sum()),
                         'transverse_rms':float(np.sqrt(np.average(np.sum((nodes[subset]-center)**2,axis=1),weights=cloud['weights'][subset])))})
    centers = np.asarray(centers)
    if len(centers) < 8:
        raise ValueError('测地截面中心不足')
    fit = ordered_midline_fit(centers)
    fit.update(raw_centers=centers,
               method='two_ended_surface_geodesic_penalized_bspline',sections=sections,
               retained_fraction=cloud['retained_fraction'],voxel_size=cloud['voxel_size'],
               graph_spacing=cloud['spacing'],graph_nodes=len(nodes),component_count=cloud['component_count'],
               endpoints=endpoint_centers,
               other_component_nodes=cloud['other_nodes'],
               component_sizes_source_points=cloud['component_sizes_source_points'],
               reliable=cloud['retained_fraction'] >= .98,warnings=[])
    if anchor is not None:
        fit['endpoint_anchor_distances'] = end_dist
        return fit, {}
    fit['endpoint_stem_distances'] = end_dist
    # Observed proximal support, with geometric contact uncertainty kept separate.
    base = fit['curve'][0]
    gap,at,foot = project_to_curve(base[None,:],stem['curve'])
    candidate_distance = max(.06*fit['arc'][-1],cloud['spacing']*3)
    proximal_nodes = nodes[geo <= candidate_distance]
    source_distance = cKDTree(proximal_nodes).query(points)[0]
    candidates = points[source_distance <= cloud['spacing']*2]
    ambiguous = abs(end_dist[0]-end_dist[1]) < max(stem['radius']*2,cloud['spacing']*2)
    reliable = len(candidates) >= 8 and gap[0] <= max(3*stem['radius'],cloud['spacing']*4) and not ambiguous
    warnings = []
    if ambiguous:
        warnings.append('两个端部与茎的距离接近，基尖方向有歧义')
    if gap[0] > max(3*stem['radius'],cloud['spacing']*4):
        warnings.append('可见近端离茎过远，连接区可能缺失；不向茎外推补点')
    if len(candidates) < 8:
        warnings.append('近端支持点少于 8')
    if cloud['retained_fraction'] < .95:
        fit['warnings'].append(f'主连通分量保留 {cloud["retained_fraction"]:.1%} 原始点；其余点仍在图中显示')
    support = cKDTree(points).query(fit['curve'])[0]
    fit['curve_support_p95'] = float(np.quantile(support,.95))
    fit['curve_support_p95_in_spacings'] = fit['curve_support_p95']/cloud['spacing']
    if fit['curve_support_p95_in_spacings'] > 4:
        fit['reliable'] = False
        fit['warnings'].append('中线部分位置离观测点超过 4 倍体素间距，存在跨空洞/截面混合风险')
    base_info = {'base':base,'attachment':foot[0],'attachment_s':float(at[0]),
                 'height_ratio':float(at[0]/stem['arc'][-1]),'base_gap':float(gap[0]),
                 'candidate_points':candidates,'candidate_count':len(candidates),'reliable':reliable,
                 'warnings':warnings,'root_selection':'compare_two_geodesic_endpoint_regions_to_stem',
                 'endpoint_stem_distances':end_dist,'geodesic_proximal_fraction':.06}
    return fit,base_info
