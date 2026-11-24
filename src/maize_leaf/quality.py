"""Reproducible topology, seam-continuity and scan-fit checks for leaf models."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .surface import leaf_mesh_grid, surface_orientation_diagnostics, triangle_areas


def surface_support_diagnostics(model, points, nu=90, nv=25):
    """Area-weighted distance from the observed part of the blade to its scan."""
    sv, faces = leaf_mesh_grid(model.blade.length, nu, nv)
    vertices = model.evaluate(sv)
    areas = triangle_areas(vertices, faces)
    face_s = sv[faces, 0].mean(axis=1)
    observed = face_s <= model.length_obs
    centers = vertices[faces].mean(axis=1)
    distances = cKDTree(points).query(centers[observed])[0]
    weights = areas[observed]
    return {"mean": float(np.average(distances, weights=weights)),
            "area": float(weights.sum()), "nu": nu, "nv": nv}


def mesh_diagnostics(model, nu=90, nv=25):
    sv, faces = leaf_mesh_grid(model.blade.length, nu, nv)
    vertices = model.evaluate(sv)
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
    components, _ = connected_components(graph, directed=False)
    boundary = edges[counts == 1]
    ids, inverse = np.unique(boundary, return_inverse=True)
    be = inverse.reshape(-1, 2)
    bg = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(len(ids), len(ids)))
    boundary_components, _ = connected_components(bg, directed=False)
    v = np.linspace(-1.0, 1.0, 41)
    tip = model.evaluate(np.column_stack([np.full(len(v), model.blade.length), v]))
    apex_spread = float(np.max(np.linalg.norm(tip - tip[len(v) // 2], axis=1)))

    # A jump stays finite as epsilon shrinks; a continuous seam's displacement
    # shrinks proportionally. Include curve, frame, width and extrapolation joins.
    breaks = np.unique(np.r_[model.midrib.t, model.frame.s_grid, model.width_s, model.length_obs])
    eps = model.blade.length * 1e-6
    breaks = breaks[(breaks > eps) & (breaks < model.blade.length - eps)]
    ss, vv = np.meshgrid(breaks, np.linspace(-1.0, 1.0, 5), indexing="ij")
    jumps = []
    for step in (eps, eps * 0.01):
        left = model.evaluate(np.column_stack([(ss - step).ravel(), vv.ravel()]))
        right = model.evaluate(np.column_stack([(ss + step).ravel(), vv.ravel()]))
        jumps.append(np.linalg.norm(right - left, axis=1))
    seam_ratio = float(np.max(jumps[1] / np.maximum(jumps[0], 1e-12))) if len(breaks) else 0.0
    areas = triangle_areas(vertices, faces)
    result = {"finite_vertices": bool(np.isfinite(vertices).all()), "mesh_components": int(components),
              "boundary_components": int(boundary_components), "nonmanifold_edges": int(np.sum(counts > 2)),
              "degenerate_faces": int(np.sum(areas <= 0.0)), "apex_spread": apex_spread,
              "seam_shrink_ratio": seam_ratio, "nu": nu, "nv": nv}
    result["passed"] = bool(result["finite_vertices"] and components == 1 and boundary_components == 1
                            and result["nonmanifold_edges"] == 0 and result["degenerate_faces"] == 0
                            and apex_spread < max(1e-8, model.blade.length * 1e-7) and seam_ratio < 0.1)
    return result


def scan_fit_diagnostics(model, points, nu=220, nv=61):
    sv, _ = leaf_mesh_grid(model.blade.length, nu, nv)
    distances = cKDTree(model.evaluate(sv)).query(points)[0]
    spacing = float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))
    width = float(np.max(model.width(np.linspace(0.0, model.blade.length, 500))))
    # Engineering alert threshold, in the scan's own length units. This is not a
    # statistical confidence interval or a proof of missing-region accuracy.
    threshold = max(6.0 * spacing, 0.15 * width)
    p95 = float(np.quantile(distances, 0.95))
    return {"median": float(np.median(distances)), "p95": p95,
            "rmse": float(np.sqrt(np.mean(distances ** 2))), "spacing": spacing,
            "coverage_3_spacing": float(np.mean(distances <= 3.0 * spacing)),
            "p95_alert_threshold": threshold, "passed": bool(p95 <= threshold), "nu": nu, "nv": nv}


def audit_model(model, retained_points):
    orientation = surface_orientation_diagnostics(model, 241, 61)
    mesh = mesh_diagnostics(model)
    fit = scan_fit_diagnostics(model, retained_points)
    support = surface_support_diagnostics(model, retained_points)
    return {"orientation": orientation, "mesh": mesh, "fit": fit, "support": support,
            "passed": bool(orientation["reversed_fraction"] == 0.0 and mesh["passed"] and fit["passed"])}
