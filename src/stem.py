"""V2 stem centerline with an independently checked local circular surface."""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from .geometry import project_to_curve
from .robust_geometry import robust_stem


def tangent_at(stem: dict, position: np.ndarray) -> np.ndarray:
    tangent = np.stack([np.interp(position, stem['arc'], col) for col in stem['d1'].T],axis=-1)
    return tangent/np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)


def sections(points: np.ndarray, stem: dict, count: int) -> list[dict]:
    radius, arc, foot = project_to_curve(points,stem['curve'])
    tangents = tangent_at(stem,arc)
    first = np.cross(tangents,np.eye(3)[np.argmin(abs(tangents),axis=1)])
    first /= np.maximum(np.linalg.norm(first,axis=1,keepdims=True),1e-12)
    second = np.cross(tangents,first)
    offset = points-foot
    azimuth = np.arctan2(np.sum(offset*second,axis=1),np.sum(offset*first,axis=1))
    sector = np.floor((azimuth+np.pi)/(2*np.pi)*8).astype(int)%8
    edges = np.linspace(0,stem['arc'][-1],count+1)
    records = []
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        inside = (arc >= lo)&(arc < hi if i+1<count else arc<=hi)
        # Points projected to a clipped end do not constrain a stem cross-section.
        usable = inside & (arc>0)&(arc<stem['arc'][-1])
        cap = max(1.6*stem['gate_distance'],2.5*stem['radius'])
        core = usable & (radius <= cap)
        radii = [np.median(radius[core & (sector==j)]) for j in range(8) if np.any(core & (sector==j))]
        estimate = float(np.median(radii)) if radii else np.nan
        station = (lo+hi)/2
        shift = np.zeros(3)
        sectors = len(radii)
        count_fit = int(core.sum())
        method = 'fixed_v2_axis_sector_median'
        # Fit a free circular cross-section only when observations constrain its circumference.
        # The V2 stem line is kept for arc and orientation. The ring may have an offset.
        local = usable & (radius<=2.6*cap)
        if local.sum()>=18:
            normal = tangent_at(stem,np.asarray(station))
            axis_u = np.cross(normal,np.eye(3)[np.argmin(abs(normal))])
            axis_u /= np.linalg.norm(axis_u)
            axis_v = np.cross(normal,axis_u)
            origin = np.array([np.interp(station,stem['arc'],coord) for coord in stem['curve'].T])
            local_uv = (points[local]-origin) @ np.vstack([axis_u,axis_v]).T
            scale = max(cap*.12,1e-9)
            def residual(coefficients):
                return np.linalg.norm(local_uv-coefficients[:2],axis=1)-coefficients[2]
            fitted = least_squares(residual,[0.,0.,max(np.median(radius[local]),scale)],
                                   bounds=([-2*cap,-2*cap,scale/4],[2*cap,2*cap,4*cap]),
                                   loss='soft_l1',f_scale=scale)
            delta,trial_radius = fitted.x[:2],float(fitted.x[2])
            err = abs(residual(fitted.x))
            good = err<=max(2*scale,.14*trial_radius)
            offsets = local_uv[good]-delta
            phi = np.arctan2(offsets[:,1],offsets[:,0])
            occupied = len(np.unique(np.floor((phi+np.pi)/(2*np.pi)*8).astype(int)%8))
            fixed = np.linalg.norm(local_uv,axis=1)
            fixed_dispersion = np.median(abs(fixed-np.median(fixed)))
            if (fitted.success and occupied>=6 and good.sum()>=14 and good.mean()>=.65 and
                    np.linalg.norm(delta)<=2*cap and np.median(err)<.75*fixed_dispersion):
                shift = delta[0]*axis_u+delta[1]*axis_v
                estimate,sectors,count_fit = trial_radius,occupied,int(good.sum())
                method = 'robust_free_circle_with_azimuth_support'
        valid = bool(count_fit>=7 and sectors>=3 and np.isfinite(estimate) and estimate>0)
        records.append({'s':(lo+hi)/2,'lo':lo,'hi':hi,'point_count':int(inside.sum()),
                        'support_count':count_fit,'azimuth_sectors':sectors,'radius':estimate,'usable':valid,
                        'center_offset_xyz':shift,'radius_method':method,
                        'radius_sector_iqr':float(np.subtract(*np.quantile(radii,[.75,.25]))) if radii else np.nan,
                        'outer_point_count':int(np.sum(usable & ~core))})
    return records


def fit_stem_surface(points: np.ndarray, params) -> tuple[dict,dict]:
    stem = robust_stem(points,params)
    measurements = sections(points,stem,24)
    valid = [r for r in measurements if r['usable']]
    arc = stem['arc']
    if valid:
        station = np.array([r['s'] for r in valid])
        radius = np.interp(arc,station,[r['radius'] for r in valid])
        offsets = np.asarray([r['center_offset_xyz'] for r in valid])
        shift = np.column_stack([np.interp(arc,station,coord) for coord in offsets.T])
        nearest = np.argmin(abs(arc[:,None]-station),axis=1)
        support = abs(arc-station[nearest]) <= 1.25*arc[-1]/24
        angular = np.array([r['azimuth_sectors'] for r in valid])[nearest]
    else:
        radius = np.full_like(arc,stem['radius'])
        shift = np.zeros_like(stem['curve'])
        support = np.zeros(len(arc),bool)
        angular = np.zeros(len(arc),int)
    displacement = np.full(len(arc),np.nan)
    angle_spread = np.full(len(arc),np.nan)
    count = np.zeros(len(arc),int)
    alternatives = []
    axis = stem['axis']
    axial = stem['curve'] @ axis
    for name,variant in [
        ('sections_18', replace(params,stem_bin_candidates=(18,))),
        ('sections_26', replace(params,stem_bin_candidates=(26,))),
        ('narrow_core', replace(params,stem_gate_span_fraction=.012)),
        ('wide_core', replace(params,stem_gate_span_fraction=.020)),
        ('weaker_penalty', replace(params,stem_lambda_candidates=(1e-6,1e-5,1e-4))),
        ('stronger_penalty', replace(params,stem_lambda_candidates=(.001,.01))),
    ]:
        try:
            alt = robust_stem(points,variant)
            position = alt['curve'] @ axis
            if np.any(np.diff(position)<=0):
                raise ValueError('扰动模型在共同轴向坐标上不单调')
            common = (axial >= position[0]) & (axial <= position[-1])
            curve = np.column_stack([np.interp(axial[common],position,c) for c in alt['curve'].T])
            t = np.column_stack([np.interp(axial[common],position,c) for c in alt['d1'].T])
            t /= np.linalg.norm(t,axis=1,keepdims=True)
            dist = np.linalg.norm(curve-stem['curve'][common],axis=1)
            deg = np.degrees(np.arccos(np.clip(np.sum(t*tangent_at(stem,arc[common]),axis=1),-1,1)))
            displacement[common] = np.fmax(displacement[common],dist)
            angle_spread[common] = np.fmax(angle_spread[common],deg)
            count += common
            rad_sections = [r['radius'] for r in sections(points,alt,24) if r['usable']]
            alternatives.append({'variant':name,'selected':alt['selected'],'common_fraction':float(common.mean()),
                                 'center_delta_p95':float(np.quantile(dist,.95)) if len(dist) else np.nan,
                                 'tangent_delta_p95_deg':float(np.quantile(deg,.95)) if len(deg) else np.nan,
                                 'median_section_radius':float(np.median(rad_sections)) if rad_sections else np.nan,
                                 'core_points':int(alt['core_mask'].sum())})
        except (ValueError,np.linalg.LinAlgError) as exc:
            alternatives.append({'variant':name,'error':str(exc)})
    surface = {'arc':arc,'center':stem['curve']+shift,'center_offset_xyz':shift,'radius':radius,'supported':support,'azimuth_sectors':angular,
               'stem_center_delta_max':displacement,'stem_tangent_delta_max_deg':angle_spread,'variant_count':count}
    evidence = {'method':'local robust circular cross-sections with a fixed V2 axis fallback',
                'sections':measurements,'usable_surface_sections':len(valid),'surface_supported_fraction':float(support.mean()),
                'core_points':int(stem['core_mask'].sum()),'stem_points':len(points),
                'selected_centerline':stem['selected'], 'v2_parameter_comparison':stem['parameter_comparison'],
                'free_circle_section_count':sum(r['radius_method']=='robust_free_circle_with_azimuth_support' for r in valid),
                'v2_center_sections_without_core':sum(not item['valid'] for item in stem['section_records']),
                'v2_center_sections_without_support_fraction':stem['unsupported_curve_fraction'],
                'max_global_pca_tangent_deviation_deg':stem['local_axis_deviation_max_deg'],
                'variants':alternatives,
                'interpretation':'宽窄核心和去/增平滑用于检查被排除的弯曲与平滑敏感性。超出局部带的点保留，无法由标签确认其组织身份。'}
    return stem,surface,evidence


def surface_query(points: np.ndarray, stem: dict, surface: dict) -> dict:
    dist,arc,foot = project_to_curve(points,stem['curve'])
    foot += np.column_stack([np.interp(arc,surface['arc'],offset) for offset in surface['center_offset_xyz'].T])
    tangent = tangent_at(stem,arc)
    radial = points-foot
    axial = np.sum(radial*tangent,axis=1)
    normal = radial-axial[:,None]*tangent
    radial_length = np.linalg.norm(normal,axis=1)
    direction = normal/np.maximum(radial_length[:,None],1e-12)
    radius = np.interp(arc,surface['arc'],surface['radius'])
    # Nearest sampled cross-section supplies support flags. End caps are excluded.
    index = np.round(np.interp(arc,surface['arc'],np.arange(len(surface['arc'])))).astype(int)
    supported = surface['supported'][index]&(arc>0)&(arc<surface['arc'][-1])&(radial_length>1e-12)
    return {'signed_distance':radial_length-radius,'distance':np.hypot(radial_length-radius,axial),
            'surface_point':foot+direction*radius[:,None],'s':arc,'supported':supported,'index':index}
