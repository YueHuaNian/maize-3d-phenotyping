"""Point cloud (with normals) and 3D mesh I/O."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class Plant:
    name: str
    xyz: np.ndarray
    labels: np.ndarray
    rgb: np.ndarray
    normals: np.ndarray | None = None


def discover(root):
    root = Path(root)
    txts = {}
    for p in sorted(root.rglob("*.txt")):
        with p.open(encoding="utf-8-sig") as f:
            if f.readline().strip().startswith("//X Y Z"):
                txts[p.stem] = p
    csvs = {p.stem: p for p in sorted(root.rglob("*.csv"))}
    return [(n, txts[n], csvs[n]) for n in sorted(set(txts) & set(csvs))]


def load_pair(name, txt, csv):
    with open(txt, encoding="utf-8-sig") as f:
        cols = f.readline().strip().removeprefix("//").split()
    raw = np.loadtxt(txt, skiprows=2)
    tab = np.genfromtxt(csv, delimiter=",", names=True, encoding="utf-8-sig")
    xyz = np.column_stack([tab["X"], tab["Y"], tab["Z"]])
    labels = np.asarray(tab["semantic"], dtype=int)
    rgb = np.clip(raw[:, 3:6], 0, 255).astype(np.uint8)
    normals = None
    if all(c in cols for c in ("Nx", "Ny", "Nz")):
        normals = raw[:, [cols.index("Nx"), cols.index("Ny"), cols.index("Nz")]]
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nrm, 1e-12)
    return Plant(name, xyz, labels, rgb, normals)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ply(path, points, faces=None, colors=None, regions=None, face_regions=None, leaf_ids=None):
    points = np.asarray(points)
    if colors is None:
        colors = np.tile([140, 155, 148], (len(points), 1))
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if regions is not None:
            f.write("property uchar region\n")
        if leaf_ids is not None:
            f.write("property int leaf_id\n")
        if faces is not None:
            f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
            if face_regions is not None:
                f.write("property uchar region\n")
        f.write("end_header\n")
        for i, (p, c) in enumerate(zip(points, colors)):
            tail = f" {int(regions[i])}" if regions is not None else ""
            tail += f" {int(leaf_ids[i])}" if leaf_ids is not None else ""
            f.write(" ".join(f"{x:.9g}" for x in p) + " " + " ".join(str(int(x)) for x in c) + tail + "\n")
        if faces is not None:
            for i, face in enumerate(faces):
                tail = f" {int(face_regions[i])}" if face_regions is not None else ""
                f.write("3 " + " ".join(str(int(x)) for x in face) + tail + "\n")


def read_ply(path):
    with open(path, encoding="ascii") as f:
        nv, nf = 0, 0
        for line in f:
            s = line.strip()
            if s.startswith("element vertex"):
                nv = int(s.split()[-1])
            elif s.startswith("element face"):
                nf = int(s.split()[-1])
            elif s == "end_header":
                break
        verts = np.loadtxt(f, max_rows=nv) if nv else np.empty((0, 6))
        faces = np.loadtxt(f, max_rows=nf, dtype=int) if nf else np.empty((0, 4), dtype=int)
    return np.atleast_2d(verts), (np.atleast_2d(faces) if nf else np.empty((0, 4), dtype=int))


def write_obj(path, vertices, faces, face_regions, regions, colors):
    mtl = Path(path).with_suffix(".mtl")
    with mtl.open("w", encoding="ascii") as f:
        for k, name in regions.items():
            f.write(f"newmtl {name}\nKd " + " ".join(f"{x / 255:.5f}" for x in colors[k]) + "\nd 1\n\n")
    with open(path, "w", encoding="ascii") as f:
        f.write(f"mtllib {mtl.name}\n")
        for p in vertices:
            f.write("v " + " ".join(f"{x:.9g}" for x in p) + "\n")
        for k, name in regions.items():
            f.write(f"usemtl {name}\n")
            for face in faces[face_regions == k] + 1:
                f.write("f " + " ".join(str(x) for x in face) + "\n")
