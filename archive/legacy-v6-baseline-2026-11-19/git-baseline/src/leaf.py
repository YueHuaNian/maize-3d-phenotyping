"""Automatic leaf candidates, continuous midlines, and separate feature gates."""
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components, dijkstra
from .geometry import arc_lengths, project_to_curve, pca_midline, Parameters
from .robust_geometry import geodesic_midline, leaf_graph, ordered_midline_fit
from .stem import surface_query, tangent_at


LEVELS = ('FAILED','LOW','MEDIUM','HIGH')
PASS = {'MEDIUM','HIGH'}
POINT_MINIMUM = 200


def unit(v: np.ndarray) -> np.ndarray:
    return v/max(float(np.linalg.norm(v)),1e-12)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b),-1,1))))


def proximal(curve: np.ndarray, fraction: float = .12) -> np.ndarray:
    arc = arc_lengths(curve)
    nodes = curve[arc<=fraction*arc[-1]]
    if len(nodes)<4:
        nodes = curve[:4]
    _,_,axes = np.linalg.svd(nodes-nodes.mean(0),full_matrices=False)
    return axes[0] if axes[0] @ (nodes[-1]-nodes[0]) >= 0 else -axes[0]


def orient(fit: dict, stem: dict, surface: dict) -> dict:
    distances = surface_query(fit['curve'][[0,-1]],stem,surface)['distance']
    if distances[0]>distances[1]:
        fit = dict(fit)
        for field in ('curve','curvature','centers'):
            fit[field] = fit[field][::-1]
        fit['d1'],fit['d2'] = -fit['d1'][::-1],fit['d2'][::-1]
        fit['arc'] = arc_lengths(fit['curve'])
    return fit


def graph_path(cloud: dict, reference: dict) -> dict:
    nodes = cloud['nodes']
    endpoints = cKDTree(nodes).query(reference['curve'][[0,-1]])[1]
    _,previous = dijkstra(cloud['graph'],directed=False,indices=int(endpoints[0]),return_predecessors=True)
    path = [int(endpoints[1])]
    while path[-1]!=endpoints[0] and previous[path[-1]]>=0:
        path.append(int(previous[path[-1]]))
    if len(path)<8 or path[-1]!=endpoints[0]:
        raise ValueError('图最短路径没有足够的连通点')
    return ordered_midline_fit(nodes[path[::-1]])


def local_sections(points: np.ndarray, reference: dict, spacing: float, iterations: int) -> dict:
    fit = reference
    for _ in range(iterations):
        distance,arc,_ = project_to_curve(points,fit['curve'])
        total = fit['arc'][-1]
        stations = np.linspace(.015,.985,26)*total
        centers = []
        for position in stations:
            neighborhood = points[(abs(arc-position)<=.027*total)&(distance<=max(7*spacing,.14*total))]
            if len(neighborhood)<5:
                continue
            center = np.median(neighborhood,axis=0)
            # Geodesic reference fixes ordering. Local transverse PCA removes sampling-density bias.
            if iterations==1:
                _,_,axes = np.linalg.svd(neighborhood-center,full_matrices=False)
                idx = np.argmin(abs(fit['arc']-position))
                tangent = unit(fit['d1'][idx])
                across = max(axes,key=lambda v: np.var((neighborhood-center) @ v)*(1-abs(v@tangent)))
                across = unit(across-(across@tangent)*tangent)
                center += across*np.quantile((neighborhood-center) @ across,[.15,.85]).mean()
            centers.append(center)
        if len(centers)<13:
            raise ValueError('有效局部截面不足')
        fit = ordered_midline_fit(np.asarray(centers))
    return fit


def inspect_curve(fit: dict, points: np.ndarray, spacing: float, retained: float) -> dict:
    curve, arc = fit['curve'],fit['arc']
    length = float(arc[-1])
    dist = cKDTree(points).query(curve)[0]
    near = dist <= 3.5*spacing
    # Arc occupancy checks both missing endpoints and collapsed curves.
    _, positions, _ = project_to_curve(points,curve)
    histogram = np.histogram(positions,bins=np.linspace(0,length,17))[0]
    window = [proximal(curve,f) for f in (.08,.12,.20)]
    k = fit['curvature']
    return {'length':length,'fit_rmse':float(fit['fit_rmse']),'curve_support_fraction':float(near.mean()),
            'support_p95_in_spacings':float(np.quantile(dist,.95)/spacing),
            'support_max_in_spacings':float(dist.max()/spacing),
            'leaf_arc_occupancy':float((histogram>=3).mean()),'retained_fraction':retained,
            'curvature_rms_norm':float(np.sqrt(length*np.trapezoid(k*k,arc))),
            'proximal_window_span_deg':max(angle(a,b) for a in window for b in window)}


def midline_candidates(points: np.ndarray, stem: dict, surface: dict, params) -> tuple[dict,dict,list, list]:
    reference,_ = geodesic_midline(points,stem,params)
    cloud = leaf_graph(points,params)
    # Local methods use points supported by the same dominant component.
    nodes = cloud['nodes']
    d = cKDTree(nodes).query(points)[0]
    other = cloud['other_nodes']
    main_points = points[d <= (cKDTree(other).query(points)[0] if len(other) else np.inf)]
    spacing = max(cloud['spacing'],1e-10)
    retained = cloud['retained_fraction']
    fitted, errors = {}, []
    methods = {'geodesic_sections':lambda: reference, 'knn_shortest_path':lambda:graph_path(cloud,reference),
               'local_pca_sections':lambda:local_sections(main_points,reference,spacing,1),
               'principal_curve_refinement':lambda:local_sections(main_points,reference,spacing,3),
               'global_pca_baseline':lambda:pca_midline(points,reference['curve'][0],Parameters())}
    for name,run in methods.items():
        try:
            fit = orient(run(),stem,surface)
            quality = inspect_curve(fit,main_points,spacing,retained)
            fitted[name] = (fit,quality)
        except (ValueError,np.linalg.LinAlgError) as exc:
            errors.append({'method':name,'error':str(exc)})
    primary = {name:(fit,q) for name,(fit,q) in fitted.items() if name!='global_pca_baseline'}
    if not primary:
        raise ValueError('没有可用的局部中线候选')
    for name,(fit,q) in fitted.items():
        curve = fit['curve']
        deltas = []
        for peer,(other_fit,other_q) in primary.items():
            if peer==name or other_q['curve_support_fraction']<.75:
                continue
            other_curve = other_fit['curve']
            delta = .5*(cKDTree(curve).query(other_curve)[0].mean()+cKDTree(other_curve).query(curve)[0].mean())
            deltas.append(delta/spacing)
        q['method_disagreement_in_spacings'] = float(np.median(deltas)) if deltas else 10.
        q['selection_score'] = (2*q['curve_support_fraction']+q['leaf_arc_occupancy']
                                -.15*q['method_disagreement_in_spacings']
                                -.015*q['proximal_window_span_deg']
                                -.03*q['support_p95_in_spacings']-.08*min(q['fit_rmse']/spacing,5))
    name,(fit,quality) = max(primary.items(),key=lambda pair:pair[1][1]['selection_score'])
    alternatives = []
    for k,gap in ((8,4.),(12,8.),(10,6.)):
        try:
            variant,_ = geodesic_midline(points,stem,replace(params,graph_neighbors=k,graph_gap_factor=gap))
            variant = orient(variant,stem,surface)
            delta = np.linalg.norm(variant['curve'][0]-fit['curve'][0])
            q = inspect_curve(variant,main_points,spacing,variant['retained_fraction'])
            alternatives.append(dict(k=k,gap_factor=gap,start_shift=float(delta),
                                     proximal_direction_delta_deg=angle(proximal(fit['curve']),proximal(variant['curve'])),**q))
        except (ValueError,np.linalg.LinAlgError) as exc:
            alternatives.append({'k':k,'gap_factor':gap,'error':str(exc)})
    valid = [q for q in alternatives if 'length' in q]
    if valid:
        quality['length_range_fraction'] = float(np.ptp([q['length'] for q in valid]+[quality['length']])/quality['length'])
        quality['direction_delta_max_deg'] = max(q['proximal_direction_delta_deg'] for q in valid)
        quality['proximal_shift_max'] = max(q['start_shift'] for q in valid)
        quality['curvature_relative_range'] = float(np.ptp([q['curvature_rms_norm'] for q in valid]+[quality['curvature_rms_norm']])/
                                                     max(quality['curvature_rms_norm'],.25))
    else:
        quality.update(length_range_fraction=np.inf,direction_delta_max_deg=np.inf,
                       proximal_shift_max=np.inf,curvature_relative_range=np.inf)
    quality.update(method=name,spacing=spacing,valid_perturbations=len(valid))
    methods_report = [dict(q,method=n) for n,(_,q) in fitted.items()]+errors
    return fit,quality,methods_report,alternatives


def contact_candidates(points: np.ndarray, rows: np.ndarray, fit: dict, stem: dict,
                       surface: dict, spacing: float, near_scale: float = 3.) -> list[dict]:
    queried = surface_query(points,stem,surface)
    tree = cKDTree(points)
    native = float(np.median(tree.query(points,k=2)[0][:,1]))
    narrow = max(near_scale*native,float(np.median(surface['radius']))*.65)
    supported_distances = queried['distance'][queried['supported']]
    # Keep plausible near-stem groups for inspection even when outside the strict contact band.
    discovery = min(max(2*narrow,float(supported_distances.min())+near_scale*native), 4*narrow) if len(supported_distances) else narrow
    mask = queried['supported'] & (queried['distance']<=discovery)
    indices = np.flatnonzero(mask)
    curve = fit['curve']
    length = float(fit['arc'][-1])
    candidates = []
    if len(indices):
        network = cKDTree(points[indices]).sparse_distance_matrix(cKDTree(points[indices]),max(4*native,1e-10),output_type='coo_matrix')
        _,groups = connected_components(network,directed=False)
        for group in np.unique(groups):
            local = indices[groups==group]
            if len(local)<3:
                continue
            median = np.median(points[local],axis=0)
            representative = local[np.argmin(np.linalg.norm(points[local]-median,axis=1))]
            estimate = queried['surface_point'][representative]
            offset = float(np.linalg.norm(estimate-curve[0]))
            gap = float(queried['distance'][representative])
            direction_alignment = float(unit(curve[0]-estimate) @ proximal(curve))
            score = (1.1*np.log1p(len(local))-.9*gap/narrow
                     -3*offset/max(.20*length,8*spacing)+.7*direction_alignment)
            source = 'surface_projection_of_observed_group' if gap<=narrow else 'outside_contact_band_projection_diagnostic'
            candidates.append({'source':source,'xyz':estimate,'observed_xyz':points[representative],
                               'observed_row_0based':int(rows[representative]),'support_rows_0based':rows[local],
                               'support_count':len(local),'s':float(queried['s'][representative]),'surface_index':int(queried['index'][representative]),
                               'score':float(score),'offset_to_midline':offset,'gap':gap,'extrapolation_length':0.,
                               'direction_alignment':direction_alignment,'strict_contact_limit':narrow,
                               'discovery_band':discovery})
    # Follow the reversed proximal direction only for a short, recorded range.
    limit = min(.12*length,max(8*spacing,2*float(np.median(surface['radius']))))
    t = np.linspace(0,limit,48)
    ray = curve[0]-t[:,None]*proximal(curve)
    hits = surface_query(ray,stem,surface)
    for idx in range(1,len(t)):
        if not (hits['supported'][idx] and hits['supported'][idx-1]
                and hits['signed_distance'][idx-1]>=0>hits['signed_distance'][idx]):
            continue
        # Axis-at-end artifacts are excluded by surface_query support.
        before,after = hits['signed_distance'][idx-1:idx+1]
        step = before/(before-after) if before!=after else .5
        touch = ray[idx-1]+step*(ray[idx]-ray[idx-1])
        hit = surface_query(touch[None],stem,surface)
        if not hit['supported'][0]:
            continue
        extension = float(np.linalg.norm(touch-curve[0]))
        supporting = tree.query_ball_point(touch,max(narrow,3*native))
        candidates.append({'source':'bounded_backward_extrapolation','xyz':hit['surface_point'][0],
                           'observed_xyz':None,'observed_row_0based':None,
                           'support_rows_0based':rows[supporting], 'support_count':len(supporting),
                           's':float(hit['s'][0]),'surface_index':int(hit['index'][0]),
                           'score':1.1*np.log1p(len(supporting))-1.5*extension/max(limit,1e-12),
                           'offset_to_midline':extension,'gap':float(hit['distance'][0]),'extrapolation_length':extension,
                           'direction_alignment':1.0,'strict_contact_limit':narrow,'discovery_band':discovery})
        break
    return sorted(candidates,key=lambda c:(c['source']!='outside_contact_band_projection_diagnostic',c['score']),reverse=True)


def envelope(points: np.ndarray) -> dict:
    center = points.mean(0)
    _,_,axes = np.linalg.svd(points-center,full_matrices=len(points)<3)
    if np.linalg.det(axes)<0:
        axes[-1] *= -1
    rotated = (points-center) @ axes.T
    lo,hi = rotated.min(0),rotated.max(0)
    qlo,qhi = np.quantile(rotated,[.01,.99],axis=0)
    corners = np.array([[i,j,k] for i in (lo[0],hi[0]) for j in (lo[1],hi[1]) for k in (lo[2],hi[2])])
    return {'kind':'observed_leaf_point_envelope','origin':center,'axes':axes,
            'obb_min':lo,'obb_max':hi,'robust_local_min':qlo,'robust_local_max':qhi,
            'obb_corners_xyz':center+corners @ axes,'robust_lengths':qhi-qlo,
            'xyz_min':points.min(0),'xyz_max':points.max(0)}


def full_supported_curve(fit: dict, points: np.ndarray, base: np.ndarray, spacing: float) -> dict | None:
    """Anchor the continuous curve only when the adjusted endpoints and path are observed."""
    original = fit['curve']
    tangent = -proximal(original[::-1])
    offset = points-original[-1]
    along = offset @ tangent
    end_band = offset[np.linalg.norm(offset-along[:,None]*tangent,axis=1)<=4*spacing]
    if len(end_band)<5:
        return None
    tip_projection = end_band @ tangent
    distal = original[-1]+np.median(end_band[tip_projection>=np.quantile(tip_projection,.95)],axis=0)
    endpoint_limit = min(.10*fit['arc'][-1],12*spacing)
    if max(np.linalg.norm(base-original[0]),np.linalg.norm(distal-original[-1]))>endpoint_limit:
        return None
    u = np.linspace(0,1,len(original))
    span = .25
    curve = original.copy()
    derivative,second = fit['d1'].copy(),fit['d2'].copy()
    for x,sign,delta in ((u,1,base-original[0]),(1-u,-1,distal-original[-1])):
        z = np.clip(x/span,0,1)
        # C² taper keeps the source spline outside each endpoint window.
        curve += (1-10*z**3+15*z**4-6*z**5)[:,None]*delta
        derivative += (sign*(-30*z**2+60*z**3-30*z**4)/span)[:,None]*delta
        second += ((-60*z+180*z**2-120*z**3)/span**2)[:,None]*delta
    support = cKDTree(points).query(curve)[0]
    nearby = support <= 3.5*spacing
    speed = np.linalg.norm(derivative,axis=1)
    if not nearby[[0,-1]].all() or not nearby.all() or np.any(speed<1e-9):
        return None
    # Reject endpoint corrections that turn back along their original local direction.
    if (proximal(curve) @ proximal(original)<=.5 or
            proximal(curve[::-1]) @ proximal(original[::-1])<=.5):
        return None
    curvature = np.linalg.norm(np.cross(derivative,second),axis=1)/speed**3
    return {'curve':curve,'arc':arc_lengths(curve),'curvature':curvature,
            'distal_source':'robust_cap_of_observed_leaf_points','max_support_distance':float(support.max())}


def extract_leaf(points: np.ndarray, rows: np.ndarray, label: int, stem: dict, surface: dict, params) -> dict:
    result = {'leaf_id':label,'point_count':len(points),'warnings':[],'envelope':envelope(points)}
    row = {'leaf_id':label,'point_count':len(points),'height_ratio_raw':np.nan,'height_ratio':np.nan,
           'attachment_arc_length':np.nan,'leaf_angle_deg':np.nan,'leaf_length':np.nan,
           'max_downward_displacement':np.nan,'droop_index':np.nan,'curvature_rms_norm':np.nan,
           'spread_index':np.nan,'envelope_obb_length':np.nan,'envelope_obb_width':np.nan,'envelope_obb_depth':np.nan,
           'diagnostic_angle_deg':np.nan,'visible_midline_length':np.nan,'diagnostic_curvature_rms_norm':np.nan,
           'base_confidence':'FAILED','midline_confidence':'FAILED','angle_confidence':'FAILED',
           'feature_confidence':'FAILED','geometry_quality':'FAILED'}
    result['feature'] = row
    result['geometry'] = {}
    low_count = len(points)<POINT_MINIMUM
    if low_count:
        result['warnings'].append(f'原始叶点 {len(points)} < {POINT_MINIMUM}，正式字段保持 NaN')
    try:
        fit,q,methods,variants = midline_candidates(points,stem,surface,params)
    except (ValueError,np.linalg.LinAlgError) as exc:
        result['warnings'].append(str(exc))
        return result
    curve,arc = fit['curve'],fit['arc']
    length,spacing = q['length'],q['spacing']
    result['geometry'].update(curve=curve,arc=arc,curvature=fit['curvature'],proximal_xyz=curve[0],
                              distal_xyz=curve[-1],endpoint_source='geometric_midline_endpoints')
    stable = q['valid_perturbations']>=2 and q['length_range_fraction']<=.20 and q['method_disagreement_in_spacings']<=4
    supported = q['retained_fraction']>=.94 and q['curve_support_fraction']>=.90 and q['leaf_arc_occupancy']>=.85
    good_mid = stable and supported and not low_count
    high_mid = good_mid and q['retained_fraction']>=.98 and q['length_range_fraction']<=.10 and q['curve_support_fraction']>=.97
    row['midline_confidence'] = 'HIGH' if high_mid else 'MEDIUM' if good_mid else 'LOW'
    if not stable:
        result['warnings'].append('不同中线方法或邻域扰动的弧长/路径不一致')
    if not supported:
        result['warnings'].append('中线覆盖不足或叶点存在断连')
    candidates = contact_candidates(points,rows,fit,stem,surface,spacing)
    perturbed = [contact_candidates(points,rows,fit,stem,surface,spacing,scale) for scale in (2.,4.)]
    result.update(midline_quality=q,method_comparison=methods,neighborhood_sensitivity=variants,
                  base_candidates=candidates, near_band_sensitivity=[
                      {'near_scale':scale,'candidate_count':len(items),'best_s':items[0]['s'] if items else None}
                      for scale,items in zip((2.,4.),perturbed)])
    relative = curve-curve[0]
    rise = relative @ stem['axis']
    down = max(0.,-float(rise.min()))
    horizontal = relative[-1]-rise[-1]*stem['axis']
    row.update(visible_midline_length=length,diagnostic_droop_index=down/length,
               diagnostic_max_downward_displacement=down,diagnostic_spread_index=float(np.linalg.norm(horizontal)/length),
               diagnostic_curvature_rms_norm=q['curvature_rms_norm'],midline_method=q['method'],
               diagnostic_fit_rmse=q['fit_rmse'])
    if candidates:
        best = candidates[0]
        index = best['surface_index']
        other = [c for c in candidates[1:] if abs(c['s']-best['s'])>.035*stem['arc'][-1]]
        competing = bool(any(c['source']!='outside_contact_band_projection_diagnostic' and
                             best['score']-c['score']<.85 for c in other))
        shifts = [abs(items[0]['s']-best['s']) for items in perturbed if items]
        base_stable = len(shifts)==2 and max(shifts)<=max(5*spacing,.035*stem['arc'][-1])
        local_stem_stable = (surface['variant_count'][index]>=3 and
                             surface['stem_center_delta_max'][index]<=max(1.5*surface['radius'][index],3*spacing))
        close = best['offset_to_midline']<=max(.12*length,4*spacing)
        nearby = (best['support_count']>=8 and close and best['direction_alignment']>=-.2
                  and best['source']!='outside_contact_band_projection_diagnostic')
        good_base = bool(nearby and base_stable and not competing and local_stem_stable and not low_count)
        high_base = good_base and best['source']=='surface_projection_of_observed_group' and best['support_count']>=20 and surface['azimuth_sectors'][index]>=5
        row['base_confidence'] = 'HIGH' if high_base else 'MEDIUM' if good_base else 'LOW'
        t = tangent_at(stem,np.asarray(best['s']))
        diagnostic_angle = angle(t,proximal(curve))
        row.update(height_ratio_raw=best['s']/float(stem['arc'][-1]),diagnostic_attachment_arc_length=best['s'],
                   diagnostic_angle_deg=diagnostic_angle,base_source=best['source'],diagnostic_base_X=best['xyz'][0],
                   diagnostic_base_Y=best['xyz'][1],diagnostic_base_Z=best['xyz'][2],base_extrapolation_length=best['extrapolation_length'])
        result.update(base_candidate_selected=best,base_uniqueness_conflict=competing,base_band_stable=base_stable)
        if good_base:
            row.update(height_ratio=row['height_ratio_raw'],attachment_arc_length=best['s'])
        else:
            result['warnings'].append('叶基接近度、点支持、唯一性或扰动稳定性未达正式门槛')
        dt = surface['stem_tangent_delta_max_deg'][index]
        good_angle = good_base and good_mid and dt<=14 and q['direction_delta_max_deg']<=20 and q['proximal_window_span_deg']<=15
        row['angle_confidence'] = ('HIGH' if high_mid and high_base and dt<=8 and q['direction_delta_max_deg']<=10
                                   else 'MEDIUM') if good_angle else 'LOW'
        if good_angle:
            row['leaf_angle_deg'] = diagnostic_angle
        result['angle_sensitivity'] = {'stem_tangent_delta_max_deg':float(dt),
                                       'leaf_neighbor_direction_delta_max_deg':q['direction_delta_max_deg'],
                                       'leaf_proximal_window_span_deg':q['proximal_window_span_deg'],
                                       'bounded_total_direction_variation_deg':float(min(180.,dt+q['direction_delta_max_deg']+q['proximal_window_span_deg'])) if np.isfinite(dt) else np.nan}
        # Full shape requires a supported proximal endpoint and a well-observed distal endpoint.
        full = (full_supported_curve(fit,points,best['xyz'],spacing)
                if good_base and good_mid and best['source']=='surface_projection_of_observed_group'
                and q['retained_fraction']>=.98 else None)
        if full is not None:
            result['geometry']['supported_full_curve'] = full
            full_length = float(full['arc'][-1])
            full_relative = full['curve']-full['curve'][0]
            full_vertical = full_relative @ stem['axis']
            full_down = max(0.,-float(full_vertical.min()))
            full_horizontal = full_relative[-1]-full_vertical[-1]*stem['axis']
            row.update(leaf_length=full_length,max_downward_displacement=full_down,droop_index=full_down/full_length,
                       spread_index=float(np.linalg.norm(full_horizontal)/full_length))
            curvature = float(np.sqrt(full_length*np.trapezoid(full['curvature']**2,full['arc'])))
            if q['curvature_relative_range']<=.6 and abs(curvature-q['curvature_rms_norm'])<=.6*max(.25,curvature):
                row['curvature_rms_norm'] = curvature
            else:
                result['warnings'].append('连续样条曲率对邻域参数敏感')
        else:
            result['warnings'].append('基端至远端的完整覆盖未达门槛，保留可见片段特征')
    else:
        result['warnings'].append('无受支持茎表面上的连接候选')
    row['envelope_confidence'] = 'MEDIUM' if supported and not low_count else 'LOW'
    widths = result['envelope']['robust_lengths']
    row.update(diagnostic_envelope_obb_length=widths[0],diagnostic_envelope_obb_width=widths[1],diagnostic_envelope_obb_depth=widths[2])
    if row['envelope_confidence'] in PASS:
        row.update(envelope_obb_length=widths[0],envelope_obb_width=widths[1],envelope_obb_depth=widths[2])
    row['feature_confidence'] = ('HIGH' if high_mid and row['angle_confidence']=='HIGH' else 'MEDIUM') if (
        np.isfinite(row['curvature_rms_norm']) and np.isfinite(row['leaf_angle_deg'])) else 'LOW'
    row['geometry_quality'] = min((row[k] for k in ('base_confidence','midline_confidence','angle_confidence','feature_confidence')),key=LEVELS.index)
    return result
