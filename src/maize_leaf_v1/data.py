"""Strict paired input validation. Input files are always opened read-only."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np


@dataclass
class Plant:
    name: str
    xyz: np.ndarray
    labels: np.ndarray
    rgb: np.ndarray
    report: dict


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover(root):
    root = Path(root)
    files = {}
    for ext in ("txt", "csv"):
        groups = {}
        for p in sorted(root.rglob("*." + ext)):
            # Accept point files, not README / explanatory documents.
            if ext == "txt":
                with p.open(encoding="utf-8-sig") as f:
                    if not f.readline().strip().startswith("//X Y Z"):
                        continue
            if p.stem in groups:
                raise ValueError(f"Duplicate {ext} stem: {p.stem}")
            groups[p.stem] = p
        files[ext] = groups
    a, b = set(files["txt"]), set(files["csv"])
    if a != b:
        raise ValueError(f"TXT/CSV unmatched: txt_only={sorted(a-b)}, csv_only={sorted(b-a)}")
    if not a:
        raise ValueError(f"No paired point clouds under {root}")
    return [(n, files["txt"][n], files["csv"][n]) for n in sorted(a)]


def load_pair(name, txt, csv, tolerance=1e-7):
    with open(txt, encoding="utf-8-sig") as f:
        header = f.readline().strip()
        declared = int(f.readline().strip())
    raw = np.loadtxt(txt, skiprows=2)
    table = np.genfromtxt(csv, delimiter=",", names=True, encoding="utf-8-sig")
    if table.dtype.names != ("X", "Y", "Z", "semantic"):
        raise ValueError(f"{name}: invalid CSV header: {table.dtype.names}")
    xyz = np.column_stack([table[x] for x in ("X", "Y", "Z")])
    semantic = np.asarray(table["semantic"])
    if raw.ndim != 2 or raw.shape[1] < 7 or len(raw) != declared or len(xyz) != declared:
        raise ValueError(f"{name}: row count or TXT columns mismatch")
    if not np.isfinite(xyz).all() or not np.isfinite(raw[:, :7]).all():
        raise ValueError(f"{name}: nonfinite coordinates, color or label")
    if not np.isfinite(semantic).all() or not np.all(semantic == np.round(semantic)):
        raise ValueError(f"{name}: non-integer labels")
    err = float(np.max(np.abs(raw[:, :3] - xyz)))
    if err > tolerance:
        raise ValueError(f"{name}: TXT/CSV coordinate/order mismatch {err:g}")
    # Finite Constant_#2 overrides Constant; NaN denotes no override.
    columns = header.removeprefix("//").split()
    merged = raw[:, columns.index("Constant")].copy()
    if "Constant_#2" in columns:
        extra = raw[:, columns.index("Constant_#2")]
        override = np.isfinite(extra)
        merged[override] = extra[override]
    if not np.array_equal(merged, semantic):
        raise ValueError(f"{name}: merged TXT labels and CSV semantic mismatch")
    labels = semantic.astype(int)
    if np.any(labels < -1) or not np.any(labels == 0) or not np.any(labels > 0):
        raise ValueError(f"{name}: invalid labels or missing stem/leaves")
    ids, counts = np.unique(labels, return_counts=True)
    report = dict(plant=name, points=len(xyz), declared_txt_points=declared,
                  coordinate_max_absolute_difference=err, coordinate_tolerance=tolerance,
                  labels={str(i): int(c) for i, c in zip(ids, counts)},
                  txt=str(txt), csv=str(csv), txt_sha256=sha256(txt), csv_sha256=sha256(csv),
                  matched=True, txt_header=header,
                  txt_label_rule="finite Constant_#2 overrides Constant; otherwise Constant",
                  units="source coordinate units; physical unit not documented")
    return Plant(name, xyz, labels, np.clip(raw[:, 3:6], 0, 255).astype(np.uint8), report)
