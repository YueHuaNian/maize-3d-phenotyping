"""Run the v1.0 maize point-cloud release on all positive leaf instances."""
import argparse
from datetime import datetime
from pathlib import Path
import traceback
import matplotlib
matplotlib.use('Agg')
from src.io_utils import locate_data
from src.pipeline import diagnose_plant
from src.report import summarize
from src.batch_overview import save_overview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='v1.0')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--batch',action='store_true',help='evaluate all available pairs (default)')
    selection.add_argument('--sample',help='evaluate one plant')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--txt-dir',type=Path)
    parser.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--v5-baseline',type=Path,help='historic V5 run directory for source-matched comparison')
    args = parser.parse_args()
    data = locate_data(args.data_dir)
    original = args.txt_dir or data.parent/'0709单天多个材料原始数据'
    paths = [data/f'{args.sample}.csv'] if args.sample else sorted(data.glob('*.csv'))
    baseline = args.v5_baseline
    run = args.results_dir/'runs'/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True,exist_ok=False)
    plants,errors = [],[]
    for path in paths:
        try:
            result = diagnose_plant(path,original/f'{path.stem}.txt',run)
            plants.append(result)
            r = result['summary']
            print(f'{path.stem}: 叶实例 {r["leaf_instance_count"]}，已运行正式质量检查 {r["formal_checks_run"]}，'
                  f'正式夹角 {r["v4_formal_angle_count"]}，正式曲率 {r["v4_formal_curvature_count"]}',flush=True)
        except Exception as exc:
            errors.append({'plant_id':path.stem,'error':f'{type(exc).__name__}: {exc}'})
            print(f'{path.stem}: {exc}',flush=True)
            (run/f'{path.stem}_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
    summarize(run,[path.stem for path in paths],plants,errors,baseline)
    if plants:
        save_overview(run)
    print(f'v1.0 完成 {len(plants)}/{len(paths)} 株。结果 {run}',flush=True)
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main())
