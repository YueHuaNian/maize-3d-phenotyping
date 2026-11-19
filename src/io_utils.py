"""Read-only source loading and full CSV/TXT correspondence audit."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLORS = ['#999999', '#8c510a', '#e41a1c', '#377eb8', '#4daf4a',
          '#984ea3', '#ff7f00', '#d6b600', '#a65628', '#f781bf',
          '#17becf', '#4959a6']  # -1, 0, ..., 10


def load_cloud(path):
    """Return finite Nx3 coordinates and integer semantic labels."""
    if path.suffix.lower() == '.csv':
        data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig')
        data = np.atleast_1d(data)
        if not {'X', 'Y', 'Z', 'semantic'}.issubset(data.dtype.names or ()):
            raise ValueError('CSV must contain X,Y,Z,semantic columns')
        xyz = np.column_stack([data[c] for c in ('X', 'Y', 'Z')])
        labels = data['semantic']
    else:
        with path.open(encoding='utf-8-sig') as f:
            columns = f.readline().strip().removeprefix('//').split()
            declared = int(f.readline())
            data = np.loadtxt(f, ndmin=2)
        if data.shape != (declared, len(columns)):
            raise ValueError(f'TXT shape {data.shape} differs from header ({declared}, {len(columns)})')
        xyz = data[:, [columns.index(c) for c in ('X', 'Y', 'Z')]]
        labels = data[:, columns.index('Constant')].copy()
        if 'Constant_#2' in columns:
            secondary = data[:, columns.index('Constant_#2')]
            labels = np.where(np.isfinite(secondary), secondary, labels)
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(labels).all():
        raise ValueError('Empty cloud or nonfinite coordinates/semantic labels')
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Semantic labels must be integers')
    return xyz, labels.astype(int)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_data(explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.is_dir() or not any(explicit.glob('*.csv')):
            raise FileNotFoundError(f'指定目录不存在或不含 CSV: {explicit}')
        return explicit.resolve()
    env_path = os.environ.get('MAIZE_DATA_DIR')
    local_config = PROJECT_ROOT / '.maize-data-dir'
    local_path = local_config.read_text(encoding='utf-8-sig').strip() if local_config.is_file() else ''
    candidates = [Path(env_path) if env_path else None,
                  PROJECT_ROOT / local_path if local_path else None,
                  Path.cwd() / 'data', Path.cwd() / 'csvdata']
    for path in candidates:
        if path is not None and path.is_dir() and any(path.glob('*.csv')):
            return path.resolve()
    raise FileNotFoundError('未找到数据目录。请用 --data-dir 或 MAIZE_DATA_DIR 指定；未生成模拟数据。')


def audit_inputs(csv_path: Path, txt_path: Path | None) -> tuple[np.ndarray, np.ndarray, dict]:
    xyz, labels = load_cloud(csv_path)
    before = {str(csv_path): sha256(csv_path)}
    report = {'csv_file': str(csv_path.resolve()), 'csv_rows': len(xyz),
              'labels': {str(i): int(np.sum(labels == i)) for i in np.unique(labels)},
              'coordinate_min': xyz.min(0).tolist(), 'coordinate_max': xyz.max(0).tolist(),
              'units': '原始坐标单位，尚未确认', 'xyz_finite': bool(np.isfinite(xyz).all()),
              'duplicate_xyz_extra_rows': int(len(xyz)-len(np.unique(xyz, axis=0))),
              'txt_present': txt_path is not None and txt_path.is_file()}
    if report['txt_present']:
        before[str(txt_path)] = sha256(txt_path)
        other_xyz, other_labels = load_cloud(txt_path)
        with txt_path.open(encoding='utf-8-sig') as f:
            columns = f.readline().strip().removeprefix('//').split()
            declared = int(f.readline())
            raw = np.loadtxt(f, ndmin=2)
        report.update(txt_file=str(txt_path.resolve()), txt_declared_points=declared,
                      txt_actual_rows=len(raw), txt_fields=columns,
                      txt_count_matches=bool(declared == len(raw)),
                      xyz_rowwise_exact=bool(np.array_equal(xyz, other_xyz)),
                      merged_labels_rowwise_exact=bool(np.array_equal(labels, other_labels)),
                      txt_nonfinite_by_field={col:int(np.sum(~np.isfinite(raw[:, j]))) for j,col in enumerate(columns)
                                              if not np.isfinite(raw[:, j]).all()})
        if not all(report[k] for k in ('txt_count_matches','xyz_rowwise_exact','merged_labels_rowwise_exact')):
            raise ValueError('CSV/TXT 点数、坐标或合并标签不一致，停止几何提取')
    else:
        report['warning'] = '缺少 TXT，无法验证源文件对应关系'
    report['input_sha256_before'] = before
    return xyz, labels, report


def dump_json(path: Path, data: object) -> None:
    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
