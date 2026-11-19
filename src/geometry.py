"""Geometric estimates, not anatomical ground truth. Requires NumPy/SciPy."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


@dataclass
class Parameters:
    stem_bins: int = 26
    leaf_bins: int = 30
    smoothing_rel: float = 0.006
    base_fraction: float = 0.12
    min_fit_points: int = 200
    min_centers: int = 8
    angle_sensitivity_deg: float = 12.0
    order_tolerance: float = 0.04


def unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        raise ValueError('零长度方向向量')
    return vector/length


def arc_lengths(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def spline_fit(centers: np.ndarray, smoothing_rel: float, samples: int = 240) -> dict:
    centers = centers[np.r_[True, np.linalg.norm(np.diff(centers, axis=0), axis=1) > 1e-10]]
    if len(centers) < 4:
        raise ValueError('有效截面中心不足 4 个')
    parameter = arc_lengths(centers)
    scale = float(parameter[-1])
    parameter /= scale
    smooth = len(centers)*(smoothing_rel*scale)**2
    tck, _ = splprep(centers.T, u=parameter, s=smooth, k=3)
    u = np.linspace(0, 1, samples)
    curve = np.asarray(splev(u, tck)).T
    d1 = np.asarray(splev(u, tck, der=1)).T
    d2 = np.asarray(splev(u, tck, der=2)).T
    speed = np.linalg.norm(d1, axis=1)
    if np.any(speed < 1e-9):
        raise ValueError('拟合曲线存在近零速度，曲率不可用')
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1)/speed**3
    fitted = np.asarray(splev(parameter, tck)).T
    return {'curve':curve, 'centers':centers, 'd1':d1, 'd2':d2, 'curvature':curvature,
            'arc':arc_lengths(curve), 'fit_rmse':float(np.sqrt(np.mean(np.sum((fitted-centers)**2, axis=1)))),
            'spline_s':smooth, 'smoothing_rel':smoothing_rel, 'degree':3}


def project_to_curve(points: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = np.full(len(points), np.inf)
    at = np.zeros(len(points))
    foot = np.zeros_like(points, dtype=float)
    arc = arc_lengths(curve)
    for i,(a,b) in enumerate(zip(curve[:-1], curve[1:])):
        delta = b-a
        squared = float(delta @ delta)
        if squared <= 1e-24:
            continue
        t = np.clip((points-a) @ delta / squared, 0, 1)
        candidate = a+t[:,None]*delta
        d = np.linalg.norm(points-candidate, axis=1)
        mask = d < distance
        distance[mask], foot[mask], at[mask] = d[mask], candidate[mask], arc[i]+t[mask]*np.sqrt(squared)
    return distance, at, foot


def pca_midline(points: np.ndarray, base: np.ndarray, params: Parameters) -> dict:
    origin = points.mean(0)
    _,singular,axes = np.linalg.svd(points-origin,full_matrices=False)
    axis = axes[0]
    projected = (points-origin) @ axis
    lo,hi = np.quantile(projected,[.003,.997])
    endpoint_lo = np.median(points[projected <= np.quantile(projected,.035)],axis=0)
    endpoint_hi = np.median(points[projected >= np.quantile(projected,.965)],axis=0)
    if np.linalg.norm(endpoint_hi-base) < np.linalg.norm(endpoint_lo-base):
        axis = -axis
        projected = -projected
        lo,hi = -hi,-lo
    edges = np.linspace(lo,hi,params.leaf_bins+1)
    centers, counts = [], []
    # Robust geometric cross-section centers; not directly fitting all leaf-surface points to a line.
    for a,b in zip(edges[:-1],edges[1:]):
        section = points[(projected >= a) & (projected <= b)]
        if len(section) >= 5:
            centers.append(np.median(section,axis=0))
            counts.append(len(section))
    if len(centers) < params.min_centers:
        raise ValueError('PCA 截面有效中心不足')
    centers = np.asarray(centers)
    raw_centers = centers.copy()
    # Remove isolated cross-section jumps only; record every excluded center.
    residual = np.linalg.norm(centers[1:-1]-(centers[:-2]+centers[2:])/2,axis=1)
    median = np.median(residual)
    cutoff = max(median+5*1.4826*np.median(abs(residual-median)), (hi-lo)*.07)
    keep = np.ones(len(centers),bool)
    keep[1:-1] = residual <= cutoff
    centers = centers[keep]
    fit = spline_fit(centers,params.smoothing_rel)
    base_distance = float(np.linalg.norm(fit['curve'][0]-base))
    warnings = []
    if base_distance > .12*fit['arc'][-1]:
        warnings.append('PCA 曲线起点与连接候选区不一致；弯折/遮挡需人工校正')
    # Two lobes within a longitudinal slice suggest a folded/non-single-valued leaf.
    wide_split = 0
    for a,b in zip(edges[:-1],edges[1:]):
        section = points[(projected >= a)&(projected <= b)]
        if len(section) < 12:
            continue
        # In PCA's short directions, gaps larger than leaf's sampled width can signal separate sheets.
        transverse = (section-origin) @ axes[1:].T
        span = np.ptp(transverse,axis=0)
        long_secondary = int(np.argmax(span))
        sequence = np.sort(transverse[:,long_secondary])
        if span[long_secondary] > 0 and np.diff(sequence).max() > .55*span[long_secondary]:
            wide_split += 1
    if wide_split >= 3:
        warnings.append('多截面出现分裂，PCA 单值中心线假设可能不成立')
    fit.update(raw_centers=raw_centers, removed_center_count=int((~keep).sum()),
               section_counts=counts, pca_variance_ratio=singular**2/np.sum(singular**2),
               base_curve_gap=base_distance, warnings=warnings,
               reliable=base_distance <= .12*fit['arc'][-1] and wide_split < 3)
    return fit


def tangent_angle(leaf_fit: dict, stem: dict, attach_s: float, fraction: float) -> dict:
    stem_at = int(np.argmin(abs(stem['arc']-attach_s)))
    stem_direction = unit(stem['d1'][stem_at])
    curve, arc = leaf_fit['curve'],leaf_fit['arc']
    section = curve[arc <= arc[-1]*fraction]
    if len(section) < 4:
        raise ValueError('基部切线估计窗口点不足')
    _,_,axes = np.linalg.svd(section-section.mean(0),full_matrices=False)
    leaf_direction = axes[0]
    if leaf_direction @ (section[-1]-section[0]) < 0:
        leaf_direction = -leaf_direction
    leaf_direction = unit(leaf_direction)
    angle = float(np.degrees(np.arccos(np.clip(stem_direction @ leaf_direction,-1,1))))
    return {'angle_deg':angle, 'stem_direction':stem_direction, 'leaf_direction':leaf_direction,
            'base_fraction':fraction, 'base_window_length':float(arc[-1]*fraction)}
