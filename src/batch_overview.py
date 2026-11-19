"""Render a batch summary and a local index of per-plant diagnostic plots."""
import csv
from html import escape
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from .visualization import configure


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_overview(run: Path):
    plants = read_rows(run / "batch_plant_summary.csv")
    positions = read_rows(run / "leaf_position_summary.csv")
    total = sum(int(row["plants_with_label"]) for row in positions)
    measures = [
        ("visible_midlines", "可见中线"),
        ("medium_high_midlines", "中高等级中线"),
        ("v5_candidates", "至少一个叶基候选"),
        ("attachment_medium_high_diagnostic", "中高等级诊断叶基"),
        ("formal_base_medium_high", "通过正式规则的叶基"),
        ("formal_angles", "正式叶茎夹角"),
        ("formal_curvatures", "正式中线曲率"),
    ]
    configure()
    ink = "#142d37"
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=.245, right=.958, top=.832, bottom=.135,
                           width_ratios=(1, .82), height_ratios=(1, 1.16), wspace=.38, hspace=.47)
    fig.text(.055, .951, "玉米点云几何诊断", fontsize=25, weight="bold", color=ink)
    fig.text(.055, .901, f"v1.0   /   {len(plants)} 株植株   /   {total} 片叶实例",
             fontsize=13, color="#4c626c")

    ax = fig.add_subplot(grid[0, 0])
    counts = np.asarray([sum(int(row[key]) for row in positions) for key, _ in measures])
    y = np.arange(len(measures))
    ax.barh(y, counts, height=.57, color=["#00837d"] * 4 + ["#966344"] * 3)
    for idx, count in enumerate(counts):
        ax.text(count+2, idx, f"{count}/{total}", va="center", color=ink, fontsize=9)
    ax.set(yticks=y, yticklabels=[label for _, label in measures], xlim=(0, max(total*1.25, 1)))
    ax.invert_yaxis()
    ax.set_title("批次几何覆盖", loc="left", color=ink, fontsize=12, pad=10)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    supports = np.array([float(row["surface_support_fraction"]) for row in plants])
    ax.bar(np.arange(len(plants)), supports, width=.68, color="#32758e")
    ax.set(ylim=(0, 1.05), xticks=np.arange(len(plants)),
           xticklabels=[row["plant_id"] for row in plants], yticks=[0,.25,.5,.75,1])
    ax.tick_params(axis="x", rotation=65, labelsize=7, length=0)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("逐株茎表面支持率", loc="left", color=ink, fontsize=12, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e3eaec")
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(grid[1, :])
    denom = np.array([int(row["plants_with_label"]) for row in positions])
    values = np.array([[int(row[key]) for row in positions] for key, _ in measures[1:]])
    ratios = values / denom
    cmap = LinearSegmentedColormap.from_list("coverage", ["#f1f5f4", "#a6d2ca", "#09675e"])
    ax.imshow(ratios, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set(xticks=np.arange(len(positions)), xticklabels=[row["leaf_id"] for row in positions],
           yticks=np.arange(len(values)), yticklabels=[label for _, label in measures[1:]],
           xlabel="原始叶号")
    ax.set_title("各叶位质量与覆盖", loc="left", color=ink, fontsize=12, pad=10)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.spines[:].set_visible(False)
    for row, col in np.ndindex(values.shape):
        ax.text(col, row, f"{values[row,col]}/{denom[col]}", ha="center", va="center",
                fontsize=8, color="white" if ratios[row,col] > .67 else ink)
    fig.text(.055, .053, "各叶位的分母为实际出现该叶号的植株数。候选和诊断等级反映点云几何证据。",
             fontsize=9, color="#4c626c")
    fig.savefig(run / "batch_overview.png", dpi=180, facecolor="white")
    plt.close(fig)
    write_gallery(run, plants)


def write_gallery(run, plants):
    sections = []
    for row in plants:
        plant = row["plant_id"]
        images = []
        for index, path in enumerate(sorted((run / plant).glob("attachment_candidate_distribution*.png")), 1):
            url = quote(path.relative_to(run).as_posix())
            images.append(f'<a href="{url}">叶基候选 {index}</a>')
        geometry = quote(f"{plant}/plant_geometry_3d.png")
        sections.append(
            f'<article><h2>{escape(plant)}</h2><a href="{geometry}">'
            f'<img src="{geometry}" loading="lazy" alt="{escape(plant)} 三维几何"></a>'
            f'<p>{row["leaf_instance_count"]} 片叶 · 中高中线 {row["midline_medium_high_count"]}'
            f' · 正式夹角 {row["v4_formal_angle_count"]}</p>'
            f'<nav>{" ".join(images)}</nav></article>'
        )
    page = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>玉米点云几何诊断 v1.0</title>
<style>
body {margin: 0 auto; padding: 1.5rem; max-width: 78rem; font-family: system-ui, sans-serif;
      color: #142d37; background: #f4f7f7}
img {display: block; width: 100%; height: auto}
.overview {max-width: 61rem; margin-bottom: 1.5rem}
section {display: grid; grid-template-columns: repeat(auto-fit, minmax(min(16rem,100%),1fr)); gap: 1rem}
article {padding: 1rem; background: #fff; border: 1px solid #dce4e6}
h2 {margin: 0; font-size: 1.05rem}
a {color: #075a68}
nav a {margin-right: 1rem}
</style>
<main><h1>玉米点云几何诊断 v1.0</h1>
<a class="overview" href="batch_overview.png"><img class="overview" src="batch_overview.png" alt="批次几何覆盖总览"></a>
<h2>逐株三维核验</h2>
<p>打开三维图查看大图。叶基候选图提供局部投影。</p>
<section>""" + "\n".join(sections) + "</section></main>\n</html>\n"
    (run / "index.html").write_text(page, encoding="utf-8")
