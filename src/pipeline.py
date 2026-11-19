"""Evaluate every positive leaf instance with unchanged V4 formal rules."""
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .attachment import attachment_diagnostics, gap_evidence
from .robust_geometry import ParametersV2
from .io_utils import audit_inputs,dump_json,sha256
from .leaf import PASS, POINT_MINIMUM, angle, extract_leaf, proximal
from .tables import MIDLINE_METHODS, save_csv
from .plots import plant_plots
from .stem import fit_stem_surface,tangent_at


LEAF_INSTANCE = 'LEAF_INSTANCE'
FORMAL_NAMES = ('height_ratio','attachment_arc_length','leaf_angle_deg','leaf_length',
                'curvature_rms_norm','max_downward_displacement','droop_index','spread_index',
                'envelope_obb_length','envelope_obb_width','envelope_obb_depth')


def leaf_instance_ids(labels: np.ndarray) -> list[int]:
    return [int(label) for label in np.unique(labels[labels>0])]


def failure_codes(v4: dict, record: dict) -> dict:
    row = v4['feature']
    q = v4.get('midline_quality')
    issues = {key:[] for key in ('midline','attachment','v4_formal_angle','v4_formal_curvature')}
    if row['point_count']<POINT_MINIMUM:
        for key in issues:
            issues[key].append('LOW_RAW_POINT_COUNT')
    if not q:
        issues['midline'].append('NO_VISIBLE_MIDLINE')
    else:
        if q['retained_fraction']<.94:
            issues['midline'].append('FRAGMENTED_DOMINANT_COMPONENT')
        if q['curve_support_fraction']<.90 or q['leaf_arc_occupancy']<.85:
            issues['midline'].append('INSUFFICIENT_CURVE_SUPPORT')
        if q['valid_perturbations']<2 or q['length_range_fraction']>.20 or q['method_disagreement_in_spacings']>4:
            issues['midline'].append('MIDLINE_PARAMETER_OR_METHOD_INSTABILITY')
    selected = record['selected']
    if not selected:
        issues['attachment'].append('NO_SUPPORTED_SURFACE_CANDIDATE')
    elif record['attachment_confidence_diagnostic'] not in PASS:
        if selected['support_count']<8:
            issues['attachment'].append('TOO_FEW_PROXIMAL_SUPPORT_POINTS')
        limit = record['bands'][1]['strict_distance']
        if selected['distance_to_surface_or_nearest_observation']>limit:
            issues['attachment'].append('CONTACT_OUTSIDE_STRICT_BAND')
        if not q or selected['proximal_offset']>max(.12*q['length'],4*max(record['native_spacing'],1e-9)):
            issues['attachment'].append('DISAGREEMENT_WITH_VISIBLE_PROXIMAL_END')
        if len(record['agreement_methods'])<2 or record['band_height_span'] is None or record['band_height_span']>.035:
            issues['attachment'].append('LOW_CANDIDATE_AGREEMENT_OR_BAND_STABILITY')
        if not issues['attachment'] or issues['attachment']==['LOW_RAW_POINT_COUNT']:
            issues['attachment'].append('CANDIDATE_DIRECTION_OR_COMPETITION_GATE')
    if not np.isfinite(row['leaf_angle_deg']):
        if row['base_confidence'] not in PASS:
            issues['v4_formal_angle'].append('V4_BASE_GATE_NOT_MET')
        if row['midline_confidence'] not in PASS:
            issues['v4_formal_angle'].append('V4_MIDLINE_GATE_NOT_MET')
        if row['base_confidence'] in PASS and row['midline_confidence'] in PASS:
            issues['v4_formal_angle'].append('V4_LOCAL_DIRECTION_STABILITY_GATE_NOT_MET')
    if not np.isfinite(row['curvature_rms_norm']):
        if not np.isfinite(row['leaf_length']):
            issues['v4_formal_curvature'].append('V4_NO_SUPPORTED_FULL_LENGTH')
        else:
            issues['v4_formal_curvature'].append('V4_CURVATURE_SENSITIVITY_GATE_NOT_MET')
    return issues


def order_diagnostics(rows: list[dict], sample: str) -> list[dict]:
    diagnostics = []
    ranks = {'NOT_COMPARABLE':0,'NO_INVERSION':1,'TIE':2,'SMALL':3,'MODERATE':4,'SEVERE':5}
    for row in rows:
        row.update(height_order_conflict=False,height_order_comparable=False,
                   height_order_inversion=np.nan,height_order_severity='NOT_COMPARABLE')
    for upper,lower in combinations(sorted(rows,key=lambda row:row['leaf_id']),2):
        a,b = upper['height_ratio_raw'],lower['height_ratio_raw']
        comparable = np.isfinite(a) and np.isfinite(b)
        if not comparable:
            inversion,severity,conflict = np.nan,'MISSING_HEIGHT',None
        else:
            delta = float(b-a)
            inversion,conflict = max(delta,0.),delta>=0
            severity = ('SEVERE' if delta>.10 else 'MODERATE' if delta>.04 else 'SMALL' if delta>0
                        else 'TIE' if delta==0 else 'NO_INVERSION')
        diagnostics.append({'plant_id':sample,'upper_leaf_id':upper['leaf_id'],'lower_leaf_id':lower['leaf_id'],
                            'upper_height_raw':a,'lower_height_raw':b,'comparable':bool(comparable),
                            'height_order_conflict':conflict,'height_order_inversion':inversion,
                            'height_order_severity':severity})
        for row in (upper,lower):
            if comparable:
                row['height_order_comparable'] = True
                row['height_order_conflict'] |= conflict
                row['height_order_inversion'] = float(np.fmax(row['height_order_inversion'],inversion))
                if ranks[severity]>ranks[row['height_order_severity']]:
                    row['height_order_severity'] = severity
    for row in rows:
        if not row['height_order_comparable']:
            row['height_order_conflict'] = None
    return diagnostics


def diagnose_plant(csv_path: Path, txt_path: Path, out: Path) -> dict:
    sample = csv_path.stem
    if not txt_path.is_file():
        raise FileNotFoundError(txt_path)
    xyz,labels,audit = audit_inputs(csv_path,txt_path)
    leaf_ids = leaf_instance_ids(labels)
    audit['leaf_instance_ids'] = leaf_ids
    params = ParametersV2()
    stem_rows = np.flatnonzero(labels==0)
    stem_points = xyz[stem_rows]
    stem,surface,stem_quality = fit_stem_surface(stem_points,params)
    stem_tree,all_tree = cKDTree(stem_points),cKDTree(xyz)
    stem_native = float(np.median(stem_tree.query(stem_points,k=2)[0][:,1]))
    leaves, rows, candidate_rows, method_rows, official = [],[],[],[],[]
    for label in leaf_ids:
        ids = np.flatnonzero(labels==label)
        points = xyz[ids]
        leaf = extract_leaf(points,ids,label,stem,surface,params)
        v4 = leaf['feature']
        scope = LEAF_INSTANCE
        record = attachment_diagnostics(points,ids,leaf,stem,surface,params)
        best = record['selected']
        gap = gap_evidence(points,ids,xyz,labels,stem_rows,stem_tree,all_tree,
                           max(stem_native,record['native_spacing'],1e-9),leaf['geometry'].get('proximal_xyz'))
        issues = failure_codes(leaf,record)
        leaf.update(attachment_v5=record,gap_evidence=gap,failure_codes=issues,label_status=scope)
        diag_angle = np.nan
        if best and 'curve' in leaf['geometry']:
            diag_angle = angle(tangent_at(stem,np.asarray(best['s'])),proximal(leaf['geometry']['curve']))
        old = leaf.get('base_candidate_selected')
        row = {'plant_id':sample,'leaf_id':label,'label_status':scope,'point_count':len(points),
               'attachment_method':best['method'] if best else 'NONE',
               'attachment_confidence':record['attachment_confidence_diagnostic'],'attachment_grade_scope':'DIAGNOSTIC_ONLY',
               'height_ratio_raw':best['height_ratio_raw'] if best else np.nan,
               'diagnostic_attachment_arc_length':best['s'] if best else np.nan,
               'v4_height_ratio_raw':v4['height_ratio_raw'],'v4_attachment_method':old['source'] if old else 'NONE',
               'v4_base_confidence':v4['base_confidence'],
               'visible_midline_computed':bool('curve' in leaf['geometry']),
               'midline_confidence':v4['midline_confidence'],'midline_method':v4.get('midline_method','NONE'),
               'geometry_quality':v4['geometry_quality'],'v4_angle_confidence':v4['angle_confidence'],
               'diagnostic_angle_deg':diag_angle,'v4_diagnostic_angle_deg':v4['diagnostic_angle_deg'],
               'surface_gap_min':record['closest_supported_surface_distance'],
               'raw_leaf_stem_point_gap':gap['raw_leaf_to_stem_min'],'raw_label_gap_status':gap['raw_label_gap_status'],
               'proximal_to_raw_stem_distance':gap['midline_proximal_to_raw_stem'],'gap_cause':gap['gap_cause'],
               'attached_candidate_support_points':best['support_count'] if best else 0,
               'attachment_agreement_method_count':len(record['agreement_methods']),
               'attachment_band_height_span':record['band_height_span']}
        for key in ('visible_midline_length','diagnostic_curvature_rms_norm','diagnostic_droop_index',
                    'diagnostic_spread_index','diagnostic_max_downward_displacement','diagnostic_fit_rmse',
                    'diagnostic_envelope_obb_length','diagnostic_envelope_obb_width','diagnostic_envelope_obb_depth'):
            row[key] = v4.get(key,np.nan)
        for key in ('length_range_fraction','retained_fraction','curve_support_fraction','leaf_arc_occupancy',
                    'curvature_relative_range','direction_delta_max_deg'):
            row[f'midline_{key}'] = leaf.get('midline_quality',{}).get(key,np.nan)
        row.update({f'{key}_failure_codes':';'.join(codes) for key,codes in issues.items()})
        formal = {key:v4[key] for key in FORMAL_NAMES}
        row.update(formal_scope='V4_FORMAL_GATES_UNCHANGED',**formal)
        rows.append(row)
        official.append({'plant_id':sample,'leaf_id':label,'label_status':scope,
                         'scope':'V4_FORMAL_GATES_UNCHANGED',**formal})
        for item in record['candidates']:
            flat = {k:v for k,v in item.items() if k not in ('xyz','observed_xyz','support_rows_0based')}
            flat['is_selected'] = item is best
            flat.update(zip(('candidate_x','candidate_y','candidate_z'),item['xyz']))
            if item['observed_xyz'] is not None:
                flat.update(zip(('observed_x','observed_y','observed_z'),item['observed_xyz']))
            candidate_rows.append(dict(plant_id=sample,leaf_id=label,label_status=scope,outcome='CANDIDATE',**flat))
        candidate_rows.extend(dict(plant_id=sample,leaf_id=label,label_status=scope,outcome='NO_CANDIDATE',**failure)
                              for failure in record['method_failures'])
        reported_methods = {item['method']:item for item in leaf.get('method_comparison',[])}
        for name in MIDLINE_METHODS:
            item = reported_methods.get(name)
            computed = bool(item and 'error' not in item)
            method_rows.append({'plant_id':sample,'leaf_id':label,'label_status':scope,'method':name,
                                'outcome':'COMPUTED' if computed else 'FAILED' if item else 'BLOCKED_BY_REFERENCE',
                                'selected':v4.get('midline_method')==name,
                                'meets_single_fit_support':bool(computed and item['curve_support_fraction']>=.90
                                     and item['leaf_arc_occupancy']>=.85 and item['retained_fraction']>=.94
                                     and len(points)>=POINT_MINIMUM),
                                **({k:v for k,v in item.items() if k!='method'} if item else
                                   {'error':'No usable reference geometry or method output'})})
        leaves.append(leaf)
    order = order_diagnostics(rows,sample)
    summary = {'plant_id':sample,'status':'OK','source_points':len(xyz),'stem_points':len(stem_points),
               'surface_support_fraction':stem_quality['surface_supported_fraction'],'surface_valid_sections':stem_quality['usable_surface_sections'],
               'stem_perturbations_ok':sum('error' not in fit for fit in stem_quality['variants']),
               'stem_center_shift_p95_worst':max([v['center_delta_p95'] for v in stem_quality['variants'] if 'error' not in v],default=np.nan),
               'stem_tangent_shift_p95_worst_deg':max([v['tangent_delta_p95_deg'] for v in stem_quality['variants'] if 'error' not in v],default=np.nan),
               'leaf_instance_count':len(rows),'formal_checks_run':len(official),
               'visible_midline_count':sum(row['visible_midline_computed'] for row in rows),
               'midline_medium_high_count':sum(row['midline_confidence'] in PASS for row in rows),
               'v4_attachment_candidate_count':sum(np.isfinite(row['v4_height_ratio_raw']) for row in rows),
               'attachment_candidate_count':sum(np.isfinite(row['height_ratio_raw']) for row in rows),
               'attachment_medium_high_diagnostic_count':sum(row['attachment_confidence'] in PASS for row in rows),
               'v4_formal_base_medium_high_count':sum(row['v4_base_confidence'] in PASS for row in rows),
               'v4_formal_angle_count':sum(np.isfinite(row['leaf_angle_deg']) for row in official),
               'v4_formal_curvature_count':sum(np.isfinite(row['curvature_rms_norm']) for row in official),
               'height_order_pairs_comparable':sum(item['comparable'] for item in order),
               'height_order_pairs_conflicting':sum(item['height_order_conflict'] is True for item in order),
               'height_order_pairs_total':len(order)}
    for grade in ('HIGH','MEDIUM','LOW','FAILED'):
        summary[f'geometry_quality_{grade.lower()}_count'] = sum(row['geometry_quality']==grade for row in rows)
    output = out/sample
    output.mkdir()
    save_csv(output/'leaf_features.csv',rows)
    save_csv(output/'v4_formal_reference.csv',official)
    dump_json(output/'attachment_details.json',{'plant_id':sample,
             'leaves':[{'leaf_id':leaf['leaf_id'],'label_status':leaf['label_status'],
                        'attachment':leaf['attachment_v5'],'raw_gap':leaf['gap_evidence'],
                        'failure_codes':leaf['failure_codes'],'v4_warnings':leaf['warnings'],
                        'midline_quality':leaf.get('midline_quality'),
                        'neighborhood_sensitivity':leaf.get('neighborhood_sensitivity',[]),
                        'observed_envelope':leaf['envelope'],
                        'visible_curve':leaf['geometry'].get('curve'),
                        'visible_curve_curvature':leaf['geometry'].get('curvature')} for leaf in leaves]})
    dump_json(output/'stem_geometry_quality.json',stem_quality)
    dump_json(output/'stem_surface_model.json',surface)
    np.savetxt(output/'stem_centerline.csv',np.column_stack([stem['arc'],stem['curve']]),delimiter=',',
               header='arc_length,X,Y,Z',comments='')
    plant_plots(output,sample,xyz,labels,stem,surface,leaves)
    after = {path:sha256(Path(path)) for path in audit['input_sha256_before']}
    unchanged = after==audit['input_sha256_before']
    if not unchanged:
        raise ValueError('源文件哈希在诊断中发生变化')
    quality = {'audit':audit,'hashes_after':after,'inputs_unchanged':unchanged,'summary':summary,
               'semantic_definition':{'-1':'BACKGROUND','0':'STEM','positive_integer':LEAF_INSTANCE},
               'formal_feature_policy':'All positive leaf IDs use unchanged V4 formal rules. Diagnostic fields remain distinct.',
               'analysis_parameters':vars(params),'raw_label_gap_attribution':'UNKNOWN unless external independent evidence'}
    dump_json(output/'quality_report.json',quality)
    return dict(summary=summary,rows=rows,attachments=candidate_rows,midlines=method_rows,order=order,quality=quality)
