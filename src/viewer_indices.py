"""Load verified PAI/LMI result tables and display their values in V2."""

import csv
from hashlib import sha256
import json
from math import isfinite
from textwrap import fill

from .phenotype_indices import INPUTS


MISSING_STATUS = {
    "INPUT_FAILED": "输入核验失败",
    "TOO_FEW_LEAVES": "有效叶片数量不足",
    "INVALID_STEM_SPAN": "茎点高度跨度无效",
    "INCOMPLETE_HEIGHT_OR_REACH": "叶基高度或伸展分量不完整",
    "NO_NEAREST_LEAF_POINT": "缺少最近叶点",
    "INSUFFICIENT_PROFILE_SECTIONS": "观测剖面区间不足",
    "INVALID_PROFILE_FEATURES": "观测剖面分量无效",
}


def _read_table(path, leaf=False):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            plant = row["plant_id"]
            key = (plant, int(row["leaf_id"])) if leaf else plant
            if key in rows:
                raise ValueError(f"{path.name}: duplicate row for {key}")
            rows[key] = row
    return rows


def read_index_results(directory, source, leaves):
    """Match the index tables to the exact V1.1 inputs used by the viewer."""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    expected = provenance.get("input_table_sha256_lf", {})
    for name in INPUTS:
        content = (source / name).read_bytes().replace(b"\r\n", b"\n")
        if sha256(content).hexdigest() != expected.get(name):
            raise ValueError(f"PAI/LMI results were computed from a different {name}: {directory}")

    plants = _read_table(directory / "plant_indices.csv")
    metrics = _read_table(directory / "leaf_indices.csv", leaf=True)
    if not plants.keys() >= leaves.keys() or metrics.keys() != {
        (plant, leaf) for plant, rows in leaves.items() for leaf in rows
    }:
        raise ValueError(f"PAI/LMI plant or leaf IDs do not match the viewer: {directory}")
    return plants, metrics


def _format(row, field, digits=2):
    raw = row.get(field)
    if raw in (None, ""):
        return "—"
    value = float(raw)
    return f"{value:.{digits}f}" if isfinite(value) else "—"


def _status_note(status):
    return MISSING_STATUS.get(status) or fill(status.replace("_", " "), width=29)


class IndexPanel:
    def __init__(self, fig, results):
        self.plants, self.leaves = results
        ax = fig.add_axes((0.777, 0.082, 0.207, 0.40), facecolor="#f5f7fa", label="PAI / LMI")
        ax.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        for spine in ax.spines.values():
            spine.set_edgecolor("#d6dfe9")
        ax.axhline(0.64, xmin=0.055, xmax=0.945, color="#d6dfe9", linewidth=0.9)
        self.pai = ax.text(0.055, 0.945, "", va="top", fontsize=19, weight="bold", color="#192535")
        self.plant_detail = ax.text(0.055, 0.81, "", va="top", fontsize=9.5, color="#475767")
        self.lmi = ax.text(0.055, 0.587, "", va="top", fontsize=18, weight="bold", color="#192535")
        self.leaf_detail = ax.text(0.055, 0.455, "", va="top", fontsize=9.5, color="#475767",
                                   linespacing=1.5)

    def show_plant(self, plant):
        row = self.plants.get(plant, {})
        self.pai.set_text(f"PAI  {_format(row, 'pai')}")
        status = row.get("pai_status")
        detail = f"株型指数 · {plant}" if plant is not None else "株型指数"
        if status and status != "ok":
            detail += "\n" + _status_note(status)
        elif plant is not None and not row:
            detail += "\n暂无 PAI/LMI 结果"
        self.plant_detail.set_text(detail)

    def show_leaf(self, plant, leaf=None, cluster=""):
        if leaf is None:
            self.lmi.set_text("LMI")
            self.leaf_detail.set_text("点击叶片或叶号查看叶形指数" if plant else "")
            return
        row = self.leaves.get((plant, leaf), {})
        self.lmi.set_text(f"LMI  {_format(row, 'lmi')}")
        detail = f"第 {leaf} 片叶" + (f" · {cluster}" if cluster else "")
        status = row.get("lmi_status")
        if status and status != "ok":
            detail += "\n" + _status_note(status)
        elif not row:
            detail += "\n暂无该叶的 LMI 结果"
        if row:
            entropy = _format(row, "observed_shannon_entropy_bits", 3)
            raw_entropy = f"  ({entropy} bit)" if entropy != "—" else ""
            detail += (f"\n方向熵 E    {_format(row, 'entropy_component', 3)}{raw_entropy}"
                       f"\n水平伸展 R    {_format(row, 'reach_component', 3)}"
                       f"\n下垂 D    {_format(row, 'droop_component', 3)}")
        self.leaf_detail.set_text(detail)
