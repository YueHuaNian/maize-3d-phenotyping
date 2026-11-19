"""Summarize diagnostic counts with plants as the comparative unit."""
from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import spearmanr
from .attachment import METHODS
from .io_utils import dump_json,sha256
from .leaf import LEVELS,PASS
from .pipeline import FORMAL_NAMES
from .tables import MIDLINE_METHODS, finite_quantile, save_csv


def distribution(values: list) -> dict:
    return {label:finite_quantile(values,q) for label,q in [('min',0),('q25',.25),('median',.5),('q75',.75),('max',1)]}


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def same_number(left, right) -> bool:
    a,b = (float('nan') if value is None or value=='' else float(value) for value in (left,right))
    return bool((np.isnan(a) and np.isnan(b)) or a==b)


def compare_v5(baseline: Path | None, plants: list[dict]) -> dict:
    """Keep comparisons on source-identical scans and original leaf IDs."""
    if baseline is None or not (baseline/'batch_quality_report.json').is_file():
        return {'status':'V5_BASELINE_UNAVAILABLE'}
    report = json.loads((baseline/'batch_quality_report.json').read_text(encoding='utf-8'))
    old_diag = {(r['plant_id'],int(r['leaf_id'])):r for r in csv_rows(baseline/'all_leaf_diagnostic_features.csv')}
    old_formal_count = new_formal_count = compared = newly_formal = 0
    diag_changes, formal_changes, mismatch = [],[],[]
    matched = []
    new_formal_angle = new_formal_curvature = 0
    for plant in plants:
        sample = plant['summary']['plant_id']
        prior_path = baseline/sample/'quality_report.json'
        if not prior_path.is_file():
            mismatch.append(sample)
            continue
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        if set(prior['audit']['input_sha256_before'].values()) != set(plant['quality']['audit']['input_sha256_before'].values()):
            mismatch.append(sample)
            continue
        matched.append(sample)
        old_formal = {int(r['leaf_id']):r for r in csv_rows(baseline/sample/'v4_formal_reference.csv')}
        old_formal_count += len(old_formal)
        for current in plant['rows']:
            leaf = current['leaf_id']
            new_formal_count += 1
            previous = old_diag.get((sample,leaf))
            if previous is not None:
                compared += 1
                for name in ('height_ratio_raw','visible_midline_length','diagnostic_angle_deg','diagnostic_curvature_rms_norm'):
                    if not same_number(previous[name],current[name]):
                        diag_changes.append({'plant_id':sample,'leaf_id':leaf,'field':name})
                for name in ('midline_confidence','attachment_confidence','midline_method','attachment_method'):
                    if previous[name]!=current[name]:
                        diag_changes.append({'plant_id':sample,'leaf_id':leaf,'field':name})
            if leaf not in old_formal:
                newly_formal += 1
                new_formal_angle += np.isfinite(current['leaf_angle_deg'])
                new_formal_curvature += np.isfinite(current['curvature_rms_norm'])
            else:
                for name in FORMAL_NAMES:
                    if name in old_formal[leaf] and not same_number(old_formal[leaf][name],current[name]):
                        formal_changes.append({'plant_id':sample,'leaf_id':leaf,'field':name})
    return {'status':'COMPARED','baseline':str(baseline.resolve()),'matched_plants':matched,
            'unmatched_plants':mismatch,'paired_diagnostic_leaf_count':compared,
            'diagnostic_changes':diag_changes,'common_formal_value_changes':formal_changes,
            'v5_formal_scope_count':old_formal_count,'v6_formal_scope_count':new_formal_count,
            'newly_formally_evaluated_leaves':newly_formal,
            'formal_angles_among_newly_formally_evaluated':int(new_formal_angle),
            'formal_curvatures_among_newly_formally_evaluated':int(new_formal_curvature),
            'v5_order_reference':report['height_order']}


def summarize(run: Path, requested: list[str], plants: list[dict], errors: list[dict],
              baseline: Path | None = None) -> None:
    summaries = [item['summary'] for item in plants]
    errors_by_id = {item['plant_id']:item for item in errors}
    failed_rows = [dict(plant_id=sample,status='FAILED',error=errors_by_id[sample]['error'])
                   for sample in requested if sample in errors_by_id]
    leaves = [r for item in plants for r in item['rows']]
    base = [r for item in plants for r in item['attachments']]
    methods = [r for item in plants for r in item['midlines']]
    order = [r for item in plants for r in item['order']]
    save_csv(run/'batch_plant_summary.csv',summaries+failed_rows,('plant_id','status'))
    save_csv(run/'all_leaf_features.csv',leaves,('plant_id','leaf_id'))
    save_csv(run/'attachment_candidate_comparison.csv',base,('plant_id','leaf_id','method'))
    save_csv(run/'midline_method_comparison.csv',methods,('plant_id','leaf_id','method'))
    save_csv(run/'height_order_diagnostics.csv',order,('plant_id','upper_leaf_id','lower_leaf_id'))
    positions = []
    for label in sorted({row['leaf_id'] for row in leaves}):
        scope = [r for r in leaves if r['leaf_id']==label]
        row = {'leaf_id':label,'plants_with_label':len(scope),'point_count_distribution':distribution([r['point_count'] for r in scope]),
               'visible_midlines':sum(r['visible_midline_computed'] for r in scope),
               'medium_high_midlines':sum(r['midline_confidence'] in PASS for r in scope),
               'v5_candidates':sum(np.isfinite(r['height_ratio_raw']) for r in scope),
               'v4_candidates':sum(np.isfinite(r['v4_height_ratio_raw']) for r in scope),
               'attachment_medium_high_diagnostic':sum(r['attachment_confidence'] in PASS for r in scope),
               'formal_base_medium_high':sum(r['v4_base_confidence'] in PASS for r in scope),
               'formal_angles':sum(np.isfinite(r['leaf_angle_deg']) for r in scope),
               'formal_curvatures':sum(np.isfinite(r['curvature_rms_norm']) for r in scope),
               'midline_grades':dict(Counter(r['midline_confidence'] for r in scope)),
               'attachment_grades':dict(Counter(r['attachment_confidence'] for r in scope)),
               'geometry_grades':dict(Counter(r['geometry_quality'] for r in scope))}
        for key in ('visible_midlines','medium_high_midlines','v5_candidates','attachment_medium_high_diagnostic',
                    'formal_base_medium_high','formal_angles','formal_curvatures'):
            row[f'{key}_rate'] = row[key]/len(scope)
        positions.append(row)
    save_csv(run/'leaf_position_summary.csv',[
        {**{key:value for key,value in row.items() if not isinstance(value,dict)},
         **{f'points_{k}':v for k,v in row['point_count_distribution'].items()},
         **{f'{group}_{grade}':row[f'{group}_grades'].get(grade,0)
            for group in ('midline','attachment','geometry') for grade in LEVELS}}
        for row in positions],('leaf_id','plants_with_label'))
    algorithm_stats = {}
    for method in MIDLINE_METHODS:
        scope = [r for r in methods if r['method']==method]
        algorithm_stats[method] = {'eligible_rows':len(scope),'computed':sum(r['outcome']=='COMPUTED' for r in scope),
                                   'single_fit_supported':sum(r['meets_single_fit_support'] for r in scope),
                                   'selected':sum(r['selected'] for r in scope)}
    attachment_stats = {method:len({(r['plant_id'],r['leaf_id']) for r in base if r['method']==method
                                   and r['outcome']=='CANDIDATE'})
                        for method in METHODS}
    failure = {}
    for kind in ('midline','attachment','v4_formal_angle','v4_formal_curvature'):
        failure[kind] = dict(Counter(c for r in leaves for c in r[f'{kind}_failure_codes'].split(';') if c))
    associations = {}
    for sample in summaries:
        available = [r for r in leaves if r['plant_id']==sample['plant_id']]
        counts = [r['point_count'] for r in available]
        grades = [LEVELS.index(r['midline_confidence']) for r in available]
        if len(available)>=3 and len(set(counts))>1 and len(set(grades))>1:
            associations[sample['plant_id']] = float(spearmanr(counts,grades).statistic)
    point_by_grade = {grade:distribution([r['point_count'] for r in leaves if r['midline_confidence']==grade]) for grade in LEVELS}
    v4_found = sum(np.isfinite(r['v4_height_ratio_raw']) for r in leaves)
    v5_found = sum(np.isfinite(r['height_ratio_raw']) for r in leaves)
    both = [r for r in leaves if np.isfinite(r['v4_height_ratio_raw']) and np.isfinite(r['height_ratio_raw'])]
    comparable = sum(r['comparable'] for r in order)
    inversions = sum(r['height_order_conflict'] is True for r in order)
    plants_with_issue = sum(r['attachment_medium_high_diagnostic_count']<r['leaf_instance_count'] for r in summaries)
    code = Path(__file__).resolve().parent.parent
    modules = sorted((code/'src').glob('*.py')) + sorted(code.glob('*.py'))
    comparisons = {'v4_candidate_count':v4_found,'v5_candidate_count':v5_found,'both_candidate_count':len(both),
                   'v5_new_candidate_count':sum(np.isfinite(r['height_ratio_raw']) and not np.isfinite(r['v4_height_ratio_raw']) for r in leaves),
                   'v5_lost_candidate_count':sum(not np.isfinite(r['height_ratio_raw']) and np.isfinite(r['v4_height_ratio_raw']) for r in leaves),
                   'absolute_candidate_height_change':distribution([abs(r['height_ratio_raw']-r['v4_height_ratio_raw']) for r in both]),
                   'v4_formal_base_count':sum(row['v4_formal_base_medium_high_count'] for row in summaries),
                   'v5_medium_high_diagnostic_base_count':sum(row['attachment_medium_high_diagnostic_count'] for row in summaries)}
    v5_comparison = compare_v5(baseline,plants)
    quality = {'requested_count':len(requested),'processed_count':len(plants),'failed_plants':errors,'leaf_instance_count':len(leaves),
               'formal_checks_run':sum(row['formal_checks_run'] for row in summaries),
               'all_processed_sources_unchanged':bool(plants) and all(p['quality']['inputs_unchanged'] for p in plants),
               'positions':positions,'midline_method_statistics':algorithm_stats,'attachment_method_candidate_counts':attachment_stats,
               'failure_counts':failure,'comparisons':comparisons,'v5_history_comparison':v5_comparison,
               'height_order':{'total_pairs':len(order),'comparable':comparable,
                   'conflicts':inversions,'strict_inversions':sum(r['height_order_severity'] in ('SMALL','MODERATE','SEVERE') for r in order),
                   'ties':sum(r['height_order_severity']=='TIE' for r in order),
                   'severity':dict(Counter(r['height_order_severity'] for r in order if r['height_order_conflict']))},
               'raw_label_gap_counts':dict(Counter(r['raw_label_gap_status'] for r in leaves)),
               'gap_origin':'UNKNOWN from delivered data','plant_spearman_point_count_vs_midline_grade':associations,
               'descriptive_point_count_by_midline_grade':point_by_grade,'plants_with_any_attachment_issue':plants_with_issue,
               'formal_gate_policy':'Every positive instance runs unmodified V4 geometry and formal gates. V5 diagnostics stay in diagnostic fields.',
               'release_version':'v1.0',
               'code_sha256':{p.relative_to(code).as_posix():sha256(p) for p in modules}}
    dump_json(run/'batch_quality_report.json',quality)
    n = len(leaves)
    have_mid = sum(r['visible_midline_computed'] for r in leaves)
    mid_pass = sum(r['midline_confidence'] in PASS for r in leaves)
    formal = sum(row['v4_formal_angle_count'] for row in summaries)
    formal_curve = sum(row['v4_formal_curvature_count'] for row in summaries)
    angle_plants = sum(row['v4_formal_angle_count']>0 for row in summaries)
    inverting_plants = sum(row['height_order_pairs_conflicting']>0 for row in summaries)
    v4_some_mid_plants = sum(row['midline_medium_high_count']>0 for row in summaries)
    support = distribution([r['surface_support_fraction'] for r in summaries])
    tangent = distribution([r['stem_tangent_shift_p95_worst_deg'] for r in summaries])
    gap_counts = quality['raw_label_gap_counts']
    def ratio(a,b) -> str:
        return f'{a}/{b}（{a/b:.1%}）' if b else '无分母'
    text = ['# v1.0 玉米点云全叶实例批次报告','',
            f'完成 {len(plants)}/{len(requested)} 株。对 {n} 个实际出现的正整数叶片实例逐片执行几何和正式质量检查。植株是独立样本单位。',
            f'共 {have_mid} 条可见中线。中高等级中线为 {ratio(mid_pass,n)}。茎和中线沿用原算法。',
            f'按 V4 原门槛重新计算，正式夹角为 {formal} 片，分布于 {angle_plants}/{len(summaries)} 株，正式曲率为 {formal_curve} 片。正式列与候选和诊断列分别保存。',
            '', '## V5 到 v1.0 的统计范围变化','']
    if v5_comparison['status']=='COMPARED':
        text += [f'对照 {len(v5_comparison["matched_plants"])} 株输入哈希一致的历史记录。正式几何审查分母由 {v5_comparison["v5_formal_scope_count"]} 增至 {v5_comparison["v6_formal_scope_count"]} 片。'
                 f'新纳入正式审查 {v5_comparison["newly_formally_evaluated_leaves"]} 片，其中正式夹角通过 {v5_comparison["formal_angles_among_newly_formally_evaluated"]} 片，正式曲率通过 {v5_comparison["formal_curvatures_among_newly_formally_evaluated"]} 片。',
                 f'共同诊断记录 {v5_comparison["paired_diagnostic_leaf_count"]} 片。比较的诊断字段变化 {len(v5_comparison["diagnostic_changes"])} 项，共同正式字段变化 {len(v5_comparison["common_formal_value_changes"])} 项。'
                 f'旧叶序表可比 {v5_comparison["v5_order_reference"]["comparable"]} 对，反序 {v5_comparison["v5_order_reference"]["inversions"]} 对。']
    else:
        text.append('本次没有提供可对照的 V5 历史批次。')
    text += [
            '', '## V4 与 V5 的叶基比较','',
            f'V4 原规则得到几何候选 {ratio(v4_found,n)}，V5 多方法得到 {ratio(v5_found,n)}。两版都有候选 {len(both)} 片，新增 {comparisons["v5_new_candidate_count"]} 片，减少 {comparisons["v5_lost_candidate_count"]} 片。',
            f'其中 V5 诊断中高等级 {comparisons["v5_medium_high_diagnostic_base_count"]} 片。V4 按原正式规则通过叶基 {comparisons["v4_formal_base_count"]} 片。两种计数采用各自规则，候选增加不能用于证明定位更准确。',
            f'两版均有候选时，归一化高度绝对变化中位数为 {comparisons["absolute_candidate_height_change"]["median"]:.3f}。'
            f' 本轮全部实际叶号形成 {len(order)} 个叶对，其中 {comparable} 对可比较，{inverting_plants} 株出现冲突。冲突共 {inversions} 对，包括 {quality["height_order"]["ties"]} 对相等高度。保留原坐标和原叶号。',
            '', '|定位方法|至少产生一个候选的已知叶|','|---|---:|']
    text += [f'|{key}|{ratio(value,n)}|' for key,value in attachment_stats.items()]
    text += ['', '## 叶位与点数','',
             '|叶号|出现株数|点数中位数（四分位区间）|可见中线|中高等级中线|叶基候选|诊断中高叶基|正式中高叶基|正式夹角|正式曲率|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in positions:
        d = row['point_count_distribution']
        count = row['plants_with_label']
        text.append(f'|{row["leaf_id"]}|{count}|{d["median"]:.0f}（{d["q25"]:.0f}–{d["q75"]:.0f}）|'
                    f'{ratio(row["visible_midlines"],count)}|{ratio(row["medium_high_midlines"],count)}|'
                    f'{ratio(row["v5_candidates"],count)}|{ratio(row["attachment_medium_high_diagnostic"],count)}|'
                    f'{ratio(row["formal_base_medium_high"],count)}|{ratio(row["formal_angles"],count)}|{ratio(row["formal_curvatures"],count)}|')
    if n:
        available = [r for r in positions if r['plants_with_label']]
        visible_lowest = min(r['visible_midlines']/r['plants_with_label'] for r in available)
        stable_lowest = min(r['medium_high_midlines']/r['plants_with_label'] for r in available)
        visible_ids = '、'.join(str(r['leaf_id']) for r in available if r['visible_midlines']/r['plants_with_label']==visible_lowest)
        stable_ids = '、'.join(str(r['leaf_id']) for r in available if r['medium_high_midlines']/r['plants_with_label']==stable_lowest)
        sentence = '各已知叶位在出现时均能输出可见曲线。' if visible_lowest==1 else f'可见中线输出率最低的是叶 {visible_ids}。'
        text += ['', sentence+f'中高稳定等级比例最低的是叶 {stable_ids}。']
    assoc = distribution(list(associations.values()))
    text += ['', f'点数与中线等级的 Spearman 相关在各株内计算，再汇总到植株。{len(associations)} 株有等级变异，相关系数中位数 {assoc["median"]:.3f}，四分位区间 {assoc["q25"]:.3f} 到 {assoc["q75"]:.3f}。'
             '仅作描述，点数本身已进入质量筛选，不足以识别因果关系。',
             '', '## 株间差异与方法稳定性','',
             f'茎线附近的表面模型支持比例范围为 {support["min"]:.1%}–{support["max"]:.1%}，中位数 {support["median"]:.1%}。'
             f'各株扰动切线差的最差 P95 的中位数为 {tangent["median"]:.2f}°，株间范围 {tangent["min"]:.2f}°–{tangent["max"]:.2f}°。',
             f'{v4_some_mid_plants}/{len(summaries)} 株至少有一条 V4 中高等级中线。{plants_with_issue}/{len(summaries)} 株仍有至少一片叶的 V5 叶基诊断证据不足。各株计数见批次汇总表。',
             '', '|中线方法|计算完成|单次曲线支持达门槛|被选择|','|---|---:|---:|---:|']
    for method,row in algorithm_stats.items():
        size = row['eligible_rows']
        text.append(f'|{method}|{ratio(row["computed"],size)}|{ratio(row["single_fit_supported"],size)}|{row["selected"]}|')
    text += ['', '单次曲线支持只检验点数、主分量、曲线支持与弧长占据。中高等级还需比较多方法和邻域扰动。各方法共享初始几何参考。',
             '', '## 失败与未判明问题','',
             f'按原始叶标签与茎标签的最近点距离，{gap_counts.get("SEPARATED_AT_LOCAL_SCALE",0)}/{n} 片超过局部点间距门槛。'
             '该距离取全叶最邻近处，可能并非着生处。单株 JSON 同时记录中线近端到茎点的距离、最近点源行号和间隙走廊内的标签计数。'
             '现有配对 TXT/CSV 使用同一批采样点，间隙成因标为未知。',
             '', '|环节|原因代码|叶片记录数|','|---|---|---:|']
    for kind,codes in failure.items():
        text.extend(f'|{kind}|{code}|{count}|' for code,count in sorted(codes.items(),key=lambda pair:-pair[1]))
    text += ['', '同一叶片可有多个失败原因。所有正标签使用同一套几何质量规则。']
    if errors:
        text.extend(['','未完成样本。']+[f'- {r["plant_id"]}。{r["error"]}' for r in errors])
    (run/'batch_failure_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
