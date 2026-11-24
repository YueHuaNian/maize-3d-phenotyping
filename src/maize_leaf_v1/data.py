"""Paired TXT / CSV point cloud loader."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Plant:
    name: str
    xyz: np.ndarray
    labels: np.ndarray
    rgb: np.ndarray


def discover(root):
    root = Path(root)
    txts = {}
    for p in sorted(root.rglob("*.txt")):
        with p.open(encoding="utf-8-sig") as f:
            if f.readline().strip().startswith("//X Y Z"):
                txts[p.stem] = p
    csvs = {p.stem: p for p in sorted(root.rglob("*.csv"))}
    names = sorted(set(txts) & set(csvs))
    return [(n, txts[n], csvs[n]) for n in names]


def load_pair(name, txt, csv):
    raw = np.loadtxt(txt, skiprows=2)
    table = np.genfromtxt(csv, delimiter=",", names=True, encoding="utf-8-sig")
    xyz = np.column_stack([table["X"], table["Y"], table["Z"]])
    labels = np.asarray(table["semantic"], dtype=int)
    rgb = np.clip(raw[:, 3:6], 0, 255).astype(np.uint8)
    return Plant(name, xyz, labels, rgb)
