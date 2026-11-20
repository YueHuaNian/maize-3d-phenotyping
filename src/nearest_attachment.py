"""Exact global leaf-to-stem nearest pairs and the V1.0 batch entry point."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Attachment:
    plant_id: str
    leaf_id: int | None
    leaf_point_count: int
    stem_point_count: int
    status: str
    leaf_point_row: int | None = None
    stem_point_row: int | None = None
    leaf_x: float | None = None
    leaf_y: float | None = None
    leaf_z: float | None = None
    stem_x: float | None = None
    stem_y: float | None = None
    stem_z: float | None = None
    nearest_distance: float | None = None
    leaf_raw_point_count: int = 0
    stem_raw_point_count: int = 0


class InputError(ValueError):
    """Input failure with a diagnostic safe to include in a public report."""


def _integer_label(token: str) -> int:
    try:
        value = Decimal(token)
    except InvalidOperation:
        raise InputError("invalid semantic label") from None
    if not value.is_finite() or value < -1 or value != value.to_integral_value():
        raise InputError("semantic label must be -1, 0 or a positive integer")
    return int(value)


def _parse_rows(rows, columns, label_column, secondary=None):
    try:
        coord_cols = [columns.index(axis) for axis in ("X", "Y", "Z")]
        label_col = columns.index(label_column)
        secondary_col = columns.index(secondary) if secondary in columns else None
    except ValueError:
        raise InputError("required XYZ or semantic columns are missing") from None
    if len(set(columns)) != len(columns):
        raise InputError("duplicate column names")

    coordinates = []
    labels = []
    for row_index, row in enumerate(rows):
        if len(row) != len(columns):
            raise InputError(f"wrong number of fields at point row {row_index}")
        try:
            coordinates.append([float(row[col]) for col in coord_cols])
        except ValueError:
            raise InputError(f"invalid XYZ at point row {row_index}") from None
        token = row[label_col]
        if secondary_col is not None and row[secondary_col].lower() != "nan":
            token = row[secondary_col]
        try:
            labels.append(_integer_label(token))
        except InputError as exc:
            raise InputError(f"{exc} at point row {row_index}") from None

    return np.asarray(coordinates, dtype=float).reshape(-1, 3), np.asarray(labels, dtype=object)


def _read_text(contents: bytes, suffix: str):
    try:
        return contents.decode("utf-8-sig").splitlines()
    except UnicodeError:
        raise InputError(f"invalid UTF-8 in {suffix} input") from None


def load_pair(csv_content: bytes, txt_content: bytes):
    """Validate exact point-row correspondence before returning original CSV values."""
    csv_lines = _read_text(csv_content, "CSV")
    txt_lines = _read_text(txt_content, "TXT")
    if not csv_lines or len(txt_lines) < 2:
        raise InputError("missing CSV or TXT header")
    csv_rows = csv.reader(csv_lines, strict=True)
    try:
        csv_columns = next(csv_rows)
        xyz, labels = _parse_rows(csv_rows, csv_columns, "semantic")
    except csv.Error:
        raise InputError("malformed CSV fields") from None
    if csv_rows.line_num != len(xyz) + 1:
        raise InputError("CSV point records must occupy one line each")

    txt_columns = txt_lines[0].strip().removeprefix("//").split()
    try:
        declared_count = int(txt_lines[1])
    except ValueError:
        raise InputError("invalid TXT point count header") from None
    if len(txt_lines) - 2 != declared_count or len(xyz) != declared_count:
        raise InputError(
            f"point counts differ (CSV={len(xyz)}, TXT={len(txt_lines)-2}, "
            f"TXT declared={declared_count})"
        )
    other_xyz, other_labels = _parse_rows(
        (line.split() for line in txt_lines[2:]), txt_columns, "Constant", "Constant_#2"
    )
    match = (xyz == other_xyz) | (np.isnan(xyz) & np.isnan(other_xyz))
    different = np.flatnonzero(~np.all(match, axis=1))
    if len(different):
        raise InputError(f"TXT/CSV XYZ differ at point row {different[0]}")
    different = np.flatnonzero(labels != other_labels)
    if len(different):
        raise InputError(f"TXT/CSV semantic labels differ at point row {different[0]}")
    return xyz, labels


def _first_tied_stem(tree: cKDTree, point: np.ndarray) -> int:
    """Find the first source row among stem points at the exact queried distance."""
    count = min(2, tree.n)
    distances, indices = tree.query(point, k=count, eps=0, p=2)
    distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
    while count < tree.n and distances[-1] == distances[0]:
        count = min(2 * count, tree.n)
        distances, indices = tree.query(point, k=count, eps=0, p=2)
    return int(np.min(indices[distances == distances[0]]))


def locate_attachments(plant_id: str, xyz: np.ndarray, labels: np.ndarray) -> list[Attachment]:
    """Return one row per positive label, including leaves without finite XYZ."""
    valid = np.isfinite(xyz).all(axis=1)
    stem = labels == 0
    stem_rows = np.flatnonzero(stem & valid)
    stem_raw_count = int(np.count_nonzero(stem))
    leaf_ids = sorted(set(labels[labels > 0]))
    if not leaf_ids:
        return [Attachment(plant_id, None, 0, len(stem_rows), "no_leaf_points",
                           stem_raw_point_count=stem_raw_count)]

    tree = cKDTree(xyz[stem_rows]) if len(stem_rows) else None
    attachments = []
    for leaf_id in leaf_ids:
        leaf = labels == leaf_id
        leaf_rows = np.flatnonzero(leaf & valid)
        counts = dict(
            plant_id=plant_id, leaf_id=int(leaf_id), leaf_point_count=len(leaf_rows),
            stem_point_count=len(stem_rows), leaf_raw_point_count=int(np.count_nonzero(leaf)),
            stem_raw_point_count=stem_raw_count,
        )
        if not len(leaf_rows):
            attachments.append(Attachment(**counts, status="no_valid_leaf_points"))
            continue
        if tree is None:
            attachments.append(Attachment(**counts, status="no_valid_stem_points"))
            continue
        distances, _ = tree.query(xyz[leaf_rows], k=1, eps=0, p=2)
        best = int(np.argmin(distances))
        if not np.isfinite(distances[best]):
            attachments.append(Attachment(**counts, status="distance_out_of_range"))
            continue
        leaf_row = int(leaf_rows[best])
        leaf_point = xyz[leaf_row]
        stem_row = int(stem_rows[_first_tied_stem(tree, leaf_point)])
        stem_point = xyz[stem_row]
        attachments.append(Attachment(
            **counts, status="ok", leaf_point_row=leaf_row, stem_point_row=stem_row,
            leaf_x=float(leaf_point[0]), leaf_y=float(leaf_point[1]), leaf_z=float(leaf_point[2]),
            stem_x=float(stem_point[0]), stem_y=float(stem_point[1]), stem_z=float(stem_point[2]),
            nearest_distance=float(distances[best]),
        ))
    return attachments


def _run_batch(csv_dir: Path, txt_dir: Path, output: Path) -> int:
    from .nearest_report import write_batch_outputs
    from .nearest_visualization import plot_plant

    csv_inputs = {path.stem: path for path in csv_dir.glob("*.csv") if path.is_file()}
    txt_inputs = {path.stem: path for path in txt_dir.glob("*.txt") if path.is_file()}
    plant_ids = sorted(csv_inputs.keys() | txt_inputs.keys())
    if not plant_ids:
        raise ValueError("No CSV or TXT input files found")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output-dir.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".nearest-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        (staged / "visualizations").mkdir(parents=True)
        results = []
        audits = []
        for plant_id in plant_ids:
            audit = {"plant_id": plant_id, "csv_file": None, "csv_sha256": None,
                     "txt_file": None, "txt_sha256": None, "point_rows": None,
                     "invalid_xyz_rows": None, "status": "ok", "detail": ""}
            inputs = {}
            try:
                for suffix, paths in (("csv", csv_inputs), ("txt", txt_inputs)):
                    if plant_id in paths:
                        path = paths[plant_id]
                        audit[f"{suffix}_file"] = path.name
                        inputs[suffix] = path.read_bytes()
                        audit[f"{suffix}_sha256"] = sha256(inputs[suffix]).hexdigest()
                if len(inputs) != 2:
                    missing = "CSV" if "csv" not in inputs else "TXT"
                    raise InputError(f"missing matching {missing}")
                xyz, labels = load_pair(inputs["csv"], inputs["txt"])
            except (InputError, OSError) as exc:
                audit.update(status="input_failed",
                             detail=str(exc) if isinstance(exc, InputError) else "input read failed")
                results.append(Attachment(plant_id, None, 0, 0, "input_failed"))
            else:
                attachments = locate_attachments(plant_id, xyz, labels)
                audit.update(point_rows=len(xyz),
                             invalid_xyz_rows=int(np.count_nonzero(~np.isfinite(xyz).all(axis=1))))
                results.extend(attachments)
                plot_plant(staged / "visualizations" / f"{plant_id}.png", plant_id, xyz, labels,
                           attachments)
            audits.append(audit)

        write_batch_outputs(staged, [asdict(result) for result in results], audits)
        staged.rename(output)

    failures = sum(record.status != "ok" for record in results)
    print(f"{len(plant_ids)} plants, {sum(record.leaf_id is not None for record in results)} "
          f"leaves, {failures} failed records. Outputs: {output}")
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory of semantic CSVs")
    parser.add_argument("--txt-dir", type=Path, required=True, help="directory of matching original TXTs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v1.0"))
    args = parser.parse_args()
    for directory in (args.data_dir, args.txt_dir):
        if not directory.is_dir():
            parser.error(f"Input directory does not exist: {directory}")
    try:
        return _run_batch(args.data_dir, args.txt_dir, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
