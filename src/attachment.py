"""Leaf attachment candidates and raw label gap diagnostics."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .geometry import arc_lengths
from .robust_geometry import leaf_graph
from .leaf import proximal, unit
from .stem import surface_query


METHODS = ('surface_nearest', 'component_proximal', 'quadratic_midline_backward', 'local_tangent_backward')
SCALES = (2.,3.,4.)


def groups(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points)<2:
        return np.zeros(len(points),int)
    distances, indices = cKDTree(points).query(points,k=min(len(points),13))
    start = np.repeat(np.arange(len(points)),indices.shape[1]-1)
    mask = distances[:,1:].ravel()<=radius
    end = indices[:,1:].ravel()
    network = coo_matrix((np.ones(mask.sum()),(start[mask],end[mask])),shape=(len(points),len(points)))
    return connected_components(network,directed=False)[1]


def contact_root(path: np.ndarray, distances: np.ndarray, stem: dict, surface: dict) -> tuple[np.ndarray,float] | None:
    queried = surface_query(path,stem,surface)
    valid = (queried['supported'][:-1]&queried['supported'][1:]
             &(queried['signed_distance'][:-1]>=0)&(queried['signed_distance'][1:]<0))
    for found in np.flatnonzero(valid):
        a,b = queried['signed_distance'][found:found+2]
        fraction = a/(a-b)
        root = path[found]+fraction*(path[found+1]-path[found])
        if surface_query(root[None],stem,surface)['supported'][0]:
            return root,float(distances[found]+fraction*(distances[found+1]-distances[found]))
    return None


def gap_evidence(points: np.ndarray, rows: np.ndarray, xyz: np.ndarray, labels: np.ndarray,
                 stem_rows: np.ndarray, stem_tree, all_tree, spacing: float, proximal: np.ndarray | None) -> dict:
    raw_distance, nearest = stem_tree.query(points)
    i = int(np.argmin(raw_distance))
    a,b = points[i],xyz[stem_rows[nearest[i]]]
    delta = b-a
    gap = float(raw_distance[i])
    threshold = 4*spacing
    counts = {}
    if gap>0:
        selection = all_tree.query_ball_point((a+b)/2,gap/2+threshold/2)
        other = xyz[selection]-a
        fraction = (other @ delta)/(gap*gap)
        interior = (fraction>.15)&(fraction<.85)&(np.linalg.norm(other-fraction[:,None]*delta,axis=1)<=threshold/2)
        for tag in labels[np.asarray(selection,dtype=int)[interior]]:
            counts[str(int(tag))] = counts.get(str(int(tag)),0)+1
    prox_gap = float(stem_tree.query(proximal)[0]) if proximal is not None else np.nan
    return {'raw_leaf_to_stem_min':gap,'raw_leaf_row_0based':int(rows[i]),
            'raw_stem_row_0based':int(stem_rows[nearest[i]]),'raw_leaf_xyz':a,'raw_stem_xyz':b,
            'midline_proximal_to_raw_stem':prox_gap,'raw_gap_threshold':threshold,
            'raw_label_gap_status':'SEPARATED_AT_LOCAL_SCALE' if gap>threshold else 'CLOSE_RAW_LABEL_POINTS',
            'corridor_label_counts':counts,'gap_cause':'UNKNOWN',
            'cause_note':'TXT 和 CSV 对应同一批采样点。邻近或间隙证据不能可靠区分采集缺失、分割断开和叶近端缺损。'}


def attachment_diagnostics(points: np.ndarray, rows: np.ndarray, leaf: dict, stem: dict,
                           surface: dict, params) -> dict:
    observed = surface_query(points,stem,surface)
    tree = cKDTree(points)
    native = float(np.median(tree.query(points,k=2)[0][:,1])) if len(points)>1 else 0.
    spacing = max(native,1e-9)
    radius = float(np.median(surface['radius']))
    scale_distance = max(3*spacing,.65*radius)
    curve = leaf['geometry'].get('curve')
    arc = arc_lengths(curve) if curve is not None else None
    origin = curve[0] if curve is not None else None
    length = float(arc[-1]) if arc is not None else 0.
    direction = proximal(curve) if curve is not None else None
    all_candidates, bands, failures = [],[],[]
    local_counts = tree.query_ball_point(points,4*spacing,return_length=True)
    support_indices = np.flatnonzero(observed['supported'])
    closest = int(support_indices[np.argmin(observed['distance'][support_indices])]) if len(support_indices) else None
    closest_any = int(np.argmin(observed['distance']))
    selected_by_band = {}

    def add(method: str, variant: str, point: np.ndarray, source: str, picked: int | None,
            support: np.ndarray, extension: float=0.) -> None:
        hit = surface_query(point[None],stem,surface)
        if not hit['supported'][0]:
            return
        projected = hit['surface_point'][0]
        gap = float(hit['distance'][0])
        if picked is None:
            gap = float(tree.query(projected)[0])
        offset = float(np.linalg.norm(projected-origin)) if origin is not None else np.nan
        alignment = float(unit(origin-projected) @ direction) if origin is not None and offset>1e-10 else np.nan
        density = float(np.median(local_counts[support])) if len(support) else 0.
        local_limit = max(.12*length,6*spacing)
        score = (np.log1p(min(len(support),40))+.35*np.log1p(density)
                 -1.35*gap/scale_distance
                 -2*(offset/local_limit if np.isfinite(offset) else 2.)
                 +.55*(alignment if np.isfinite(alignment) else 0.)-.7*extension/max(local_limit,1e-9))
        all_candidates.append({'method':method,'variant':variant,'coordinate_source':source,'xyz':projected,
            'observed_xyz':None if picked is None else points[picked],
            'observed_row_0based':None if picked is None else int(rows[picked]),'support_rows_0based':rows[support],
            'distance_to_surface_or_nearest_observation':gap,'s':float(hit['s'][0]),
            'height_ratio_raw':float(hit['s'][0]/stem['arc'][-1]), 'proximal_offset':offset,
            'direction_alignment_cosine':alignment,'support_count':len(support),'local_density_median':density,
            'extrapolation_length':extension,'score':float(score)})

    for mult in SCALES:
        band = max(mult*spacing,.65*radius)
        core = support_indices[observed['distance'][support_indices]<=band]
        # Extended band retains near-contact evidence but does not change the strict band.
        search = min(max(2*band,observed['distance'][closest]+mult*spacing),4*band) if closest is not None else band
        region = support_indices[observed['distance'][support_indices]<=search]
        memberships = groups(points[region],4*spacing)
        sizes = np.bincount(memberships)
        bands.append({'spacing_multiplier':mult,'strict_distance':band,'search_distance':search,
                      'strict_count':len(core),'search_count':len(region),'connected_components':len(sizes),
                      'component_sizes':sorted(sizes.tolist(),reverse=True),
                      'median_density_neighbors_in_4_spacings':float(np.median(local_counts[region])) if len(region) else np.nan})
        start = len(all_candidates)
        for component in np.flatnonzero(sizes>=3):
            indices = region[memberships==component]
            # Choose from the closest half of a connected region, then rank using proximal consistency.
            near = indices[observed['distance'][indices]<=np.median(observed['distance'][indices])]
            center = np.median(points[indices],axis=0)
            chosen = int(near[np.argmin(np.linalg.norm(points[near]-center,axis=1))])
            add('surface_connected_region',str(int(mult)),points[chosen],'surface_projection',chosen,indices)
        selected_by_band[mult] = all_candidates[start:]

    if closest is not None:
        near = np.asarray(tree.query_ball_point(points[closest],4*spacing),int)
        add('surface_nearest','nominal',points[closest],'surface_projection',closest,near)
    else:
        failures.append(('surface_nearest','NO_SUPPORTED_SURFACE_PROJECTION'))
    if curve is not None:
        try:
            body = leaf_graph(points,params)
            near_body = cKDTree(body['nodes']).query(points)[0]
            other = body['other_nodes']
            belongs = near_body <= (cKDTree(other).query(points)[0] if len(other) else np.inf)
            near_start = np.linalg.norm(points-origin,axis=1)
            cap = np.flatnonzero(belongs & (near_start<=max(.10*length,6*spacing)) & observed['supported'])
            if len(cap)>=3:
                score = near_start[cap]+.5*observed['distance'][cap]
                chosen = int(cap[np.argmin(score)])
                add('component_proximal','nominal',points[chosen],'surface_projection',chosen,cap)
            else:
                failures.append(('component_proximal','DOMINANT_PROXIMAL_COMPONENT_HAS_NO_SUPPORTED_CAP'))
        except (ValueError,np.linalg.LinAlgError) as exc:
            failures.append(('component_proximal',str(exc)))
        limit = min(.12*length,max(8*leaf['midline_quality']['spacing'],2*radius))
        steps = np.linspace(0,limit,48)
        for fraction in (.08,.12,.20):
            t = proximal(curve,fraction)
            path = origin-steps[:,None]*t
            for method in METHODS[2:]:
                extended = path
                if method=='quadratic_midline_backward':
                    window = arc<=fraction*length
                    if window.sum()<5:
                        continue
                    x = arc[window]/max(limit,1e-10)
                    coefficient = np.linalg.lstsq(np.column_stack([x,x*x]),curve[window]-origin,rcond=None)[0]
                    if np.linalg.norm(coefficient[1])>.35*limit or not (.5<=np.linalg.norm(coefficient[0])/limit<=1.6):
                        continue
                    u = -steps/limit
                    extended = origin+u[:,None]*coefficient[0]+u[:,None]**2*coefficient[1]
                hit = contact_root(extended,steps,stem,surface)
                if hit is None:
                    continue
                point,_ = hit
                near = np.asarray(tree.query_ball_point(point,max(4*spacing,scale_distance)),int)
                add(method,str(fraction),point,'bounded_extrapolation',None,near,float(np.linalg.norm(point-origin)))
        for method in METHODS[2:]:
            if not any(item['method']==method for item in all_candidates):
                failures.append((method,'NO_SUPPORTED_INTERSECTION_WITHIN_LIMIT'))
    else:
        failures += [(method,'NO_VISIBLE_MIDLINE') for method in METHODS[1:]]

    nominal = [item for item in all_candidates if item['method']!='surface_connected_region' or item['variant']=='3']
    nominal.sort(key=lambda item:item['score'],reverse=True)
    for rank,item in enumerate(nominal,1):
        item['nominal_rank'] = rank
    picked = nominal[0] if nominal else None
    band_best = {}
    evidence = 'FAILED'
    span = None
    agreement = []
    if picked is not None:
        shared = [item for item in nominal if item['method']!='surface_connected_region']
        for m in SCALES:
            comparables = shared+selected_by_band[m]
            if comparables:
                band_best[str(int(m))] = max(comparables,key=lambda item:item['score'])['height_ratio_raw']
        span = float(np.ptp(list(band_best.values()))) if len(band_best)==3 else None
        nearby = max(.035*stem['arc'][-1],5*spacing)
        alternative_best = {method:next((item for item in nominal if item['method']==method),None)
                            for method in (*METHODS,'surface_connected_region')}
        agreement = [method for method,item in alternative_best.items() if item and abs(item['s']-picked['s'])<=nearby]
        conflict = any(item and abs(item['s']-picked['s'])>nearby and
                       picked['score']-item['score']<.85 for item in alternative_best.values())
        sufficient_support = (len(points)>=200 and picked['support_count']>=8 and
                              picked['distance_to_surface_or_nearest_observation']<=scale_distance and
                              np.isfinite(picked['proximal_offset']) and picked['proximal_offset']<=max(.12*length,4*spacing)
                              and np.isfinite(picked['direction_alignment_cosine']) and picked['direction_alignment_cosine']>=-.2)
        if sufficient_support and span is not None and span<=.035 and len(agreement)>=2 and not conflict:
            evidence = 'HIGH' if len(agreement)>=3 and picked['support_count']>=20 and picked['coordinate_source']=='surface_projection' else 'MEDIUM'
        else:
            evidence = 'LOW'
    return {'candidates':all_candidates,'selected':picked,'bands':bands,
            'method_failures':[{'method':name,'reason':reason} for name,reason in failures],
            'native_spacing':native,'attachment_confidence_diagnostic':evidence,
            'best_height_by_search_multiplier':band_best,'band_height_span':span,'agreement_methods':agreement,
            'closest_supported_surface_distance':float(observed['distance'][closest]) if closest is not None else np.nan,
            'closest_supported_surface_xyz':None if closest is None else points[closest],
            'closest_supported_surface_row_0based':None if closest is None else int(rows[closest]),
            'closest_model_distance_any_section':float(observed['distance'][closest_any]),
            'closest_model_row_0based':int(rows[closest_any]),'closest_model_xyz':points[closest_any],
            'closest_model_has_supported_section':bool(observed['supported'][closest_any])}
