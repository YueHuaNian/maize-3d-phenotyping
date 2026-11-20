"""Tabular fields and output shared by batch analysis and reports."""
import csv
from pathlib import Path
import numpy as np


MIDLINE_METHODS = ('geodesic_sections','knn_shortest_path','local_pca_sections',
                   'principal_curve_refinement','global_pca_baseline')


def finite_quantile(values, q: float) -> float:
    array = np.asarray(values,dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array,q)) if len(array) else np.nan


def save_csv(path: Path, rows: list[dict], empty_fields=()) -> None:
    # Preserve a stable header even for a batch with zero events.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(','.join(empty_fields)+'\n',encoding='utf-8-sig')
