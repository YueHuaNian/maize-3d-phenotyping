"""Nearest-neighbor anchored, quality-gated leaf geometry for V1.1."""

from dataclasses import asdict, replace

import numpy as np
from scipy.spatial import cKDTree

from .geometry import arc_lengths, project_to_curve, unit
from .leaf import POINT_MINIMUM, angle, inspect_curve, proximal
from .nearest_attachment import locate_attachments
from .robust_geometry import ParametersV2, geodesic_midline, robust_stem
from .stem import tangent_at

FEATURES = {
    "leaf_angle_deg": "Leaf–stem angle (deg)",
    "height_ratio": "Attachment height ratio",
    "droop_index": "Droop index",
    "curvature_rms_norm": "Normalized RMS curvature",
    "spread_index": "Horizontal spread index",
    "shannon_entropy_bits": "Inclination entropy (bits)",
}
ENTROPY_EDGES = np.linspace(0, 180, 19)
PARAMS = ParametersV2()
VARIANTS = ((10, 6.0), (8, 4.0), (12, 8.0))
RULES = {
    "min_leaf_points": POINT_MINIMUM,
    "min_retained_fraction": 0.94,
    "min_curve_support": 0.90,
    "min_arc_occupancy": 0.85,
    "min_supported_variants": 2,
    "max_relative_length_range": 0.20,
    "max_relative_curvature_range": 0.60,
    "max_variant_angle_deg": 20,
    "max_window_angle_deg": 15,
    "anchor_gap_length_fraction": 0.12,
    "anchor_gap_spacing_factor": 4,
    "endpoint_separation_spacing_factor": 2,
    "pair_gap_spacing_factor": 4,
    "pair_gap_stem_radius_factor": 3,
}


def inclination_entropy(curve: np.ndarray, axis: np.ndarray) -> tuple[float, np.ndarray]:
    steps = np.diff(curve, axis=0)
    length = np.linalg.norm(steps, axis=1)
    valid = length > 0
    if not valid.any():
        raise ValueError("zero-length midline")
    cosine = np.clip((steps[valid] @ unit(axis)) / length[valid], -1, 1)
    inclination = np.degrees(np.arccos(cosine))
    weights = np.histogram(inclination, ENTROPY_EDGES, weights=length[valid])[0]
    probability = weights / weights.sum()
    positive = probability[probability > 0]
    return max(0.0, -float(np.sum(positive * np.log2(positive)))), probability


def curve_features(fit, axis):
    curve, arc = fit["curve"], fit["arc"]
    length = float(arc[-1])
    if length <= 0:
        raise ValueError("zero-length midline")
    axis = unit(axis)
    relative = curve - curve[0]
    rise = relative @ axis
    down = max(0.0, -float(rise.min()))
    spread = float(np.linalg.norm(relative[-1] - rise[-1] * axis))
    entropy, probability = inclination_entropy(curve, axis)
    return {
        "visible_midline_length": length, "max_downward_displacement": down,
        "horizontal_displacement": spread, "droop_index": down / length,
        "spread_index": spread / length,
        "curvature_rms_norm": float(np.sqrt(length * np.trapezoid(fit["curvature"] ** 2, arc))),
        "shannon_entropy_bits": entropy,
    }, probability


def _spacing(points):
    if len(points) < 2:
        return 0.0
    return float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))


def _project_stem(q, stem):
    _, stations, _ = project_to_curve(q[None], stem["curve"])
    station = float(stations[0])
    index = int(np.argmin(np.abs(stem["arc"] - station)))
    reason = "ok"
    if not stem["curve_supported_mask"][index]:
        reason = "STEM_PROJECTION_UNSUPPORTED"
    elif not 0 < station < stem["arc"][-1]:
        reason = "STEM_PROJECTION_AT_BOUNDARY"
    return station, reason


def _variant(points, stem, anchor, k, factor):
    config = replace(PARAMS, graph_neighbors=k, graph_gap_factor=factor)
    fit, _ = geodesic_midline(points, stem, config, anchor=anchor)
    spacing = fit["graph_spacing"]
    quality = inspect_curve(fit, points, spacing, fit["retained_fraction"])
    root_gap = float(np.linalg.norm(fit["curve"][0] - anchor))
    end_delta = float(np.ptp(fit["endpoint_anchor_distances"]))
    quality.update(root_gap=root_gap, graph_spacing=spacing, endpoint_gap_difference=end_delta,
                   voxel_size=fit["voxel_size"], graph_nodes=fit["graph_nodes"],
                   smoothing_lambda=fit["selected_smoothing"]["lambda"])
    flags = []
    if quality["retained_fraction"] < RULES["min_retained_fraction"]:
        flags.append("LEAF_COMPONENT_FRAGMENTED")
    if (quality["curve_support_fraction"] < RULES["min_curve_support"]
            or quality["leaf_arc_occupancy"] < RULES["min_arc_occupancy"]):
        flags.append("CURVE_SUPPORT_LOW")
    if root_gap > max(RULES["anchor_gap_length_fraction"] * quality["length"],
                      RULES["anchor_gap_spacing_factor"] * spacing):
        flags.append("ANCHOR_CURVE_GAP")
    if end_delta <= RULES["endpoint_separation_spacing_factor"] * spacing:
        flags.append("LEAF_DIRECTION_AMBIGUOUS")
    quality["flags"] = flags
    return fit, quality


def _statuses(row, names, reason):
    for name in names:
        row[f"{name}_status"] = reason


def _accept(row, names, estimates):
    for name in names:
        row[name] = estimates[name]
    _statuses(row, names, "ok")


def _analyse_leaf(attachment, points, stem, stem_spacing):
    row = asdict(attachment)
    row.update(dict.fromkeys(FEATURES))
    row.update(attachment_arc_length=None, visible_midline_length=None,
               max_downward_displacement=None, horizontal_displacement=None)
    audit = {"plant_id": attachment.plant_id, "leaf_id": attachment.leaf_id,
             "variants": [], "diagnostic_values": {}, "quality_scope": "visible_geometry"}
    if attachment.status != "ok" or stem is None:
        reason = attachment.status if attachment.status != "ok" else "STEM_FIT_UNAVAILABLE"
        _statuses(row, FEATURES, reason)
        return row, audit, None

    p = np.array((attachment.leaf_x, attachment.leaf_y, attachment.leaf_z))
    q = np.array((attachment.stem_x, attachment.stem_y, attachment.stem_z))
    station, height_reason = _project_stem(q, stem)
    gate = max(RULES["pair_gap_stem_radius_factor"] * stem["radius"],
               RULES["pair_gap_spacing_factor"] * max(_spacing(points), stem_spacing))
    audit.update(pair_gap_gate=gate, stem_projection_status=height_reason)
    if attachment.nearest_distance > gate:
        height_reason = "LEAF_STEM_GAP_LARGE"
    height = station / float(stem["arc"][-1])
    audit["diagnostic_values"].update(height_ratio=height, attachment_arc_length=station)
    _statuses(row, ("height_ratio",), height_reason)
    if height_reason == "ok":
        row.update(height_ratio=height, attachment_arc_length=station)

    curve_names = tuple(name for name in FEATURES if name != "height_ratio")
    if len(points) < RULES["min_leaf_points"]:
        _statuses(row, curve_names, "LEAF_POINTS_TOO_FEW")
        return row, audit, None

    fits = []
    for k, factor in VARIANTS:
        entry = {"neighbors": k, "gap_factor": factor}
        try:
            fit, quality = _variant(points, stem, p, k, factor)
        except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
            entry.update(flags=["MIDLINE_FIT_FAILED"], detail=str(exc))
        else:
            entry.update(quality)
            fits.append((k, fit, quality))
        audit["variants"].append(entry)
    if not fits or fits[0][0] != VARIANTS[0][0]:
        _statuses(row, curve_names, "MIDLINE_FIT_FAILED")
        return row, audit, None

    _, fit, quality = fits[0]
    curve = fit["curve"]
    estimates, probability = curve_features(fit, stem["axis"])
    estimates["leaf_angle_deg"] = angle(proximal(curve), tangent_at(stem, np.asarray(station)))
    audit["diagnostic_values"].update(estimates)
    audit["entropy_bin_probabilities"] = probability
    if quality["flags"]:
        _statuses(row, curve_names, "|".join(quality["flags"]))
        return row, audit, curve

    supported = [(candidate, evidence) for _, candidate, evidence in fits if not evidence["flags"]]
    length_range = float(np.ptp([e["length"] for _, e in supported])) / quality["length"]
    curvature_range = float(np.ptp([e["curvature_rms_norm"] for _, e in supported])) / max(
        0.25, estimates["curvature_rms_norm"]
    )
    direction_delta = max(angle(proximal(curve), proximal(c["curve"])) for c, _ in supported)
    audit.update(supported_variant_count=len(supported), relative_length_range=length_range,
                 relative_curvature_range=curvature_range, variant_angle_max_deg=direction_delta)
    if len(supported) < RULES["min_supported_variants"] or length_range > RULES["max_relative_length_range"]:
        _statuses(row, curve_names, "MIDLINE_VARIANTS_UNSTABLE")
        return row, audit, curve

    _accept(row, ("droop_index", "spread_index", "shannon_entropy_bits"), estimates)
    for name in ("visible_midline_length", "max_downward_displacement", "horizontal_displacement"):
        row[name] = estimates[name]
    if curvature_range <= RULES["max_relative_curvature_range"]:
        _accept(row, ("curvature_rms_norm",), estimates)
    else:
        _statuses(row, ("curvature_rms_norm",), "CURVATURE_VARIANTS_UNSTABLE")
    if height_reason != "ok":
        _statuses(row, ("leaf_angle_deg",), height_reason)
    elif direction_delta > RULES["max_variant_angle_deg"] or (
        quality["proximal_window_span_deg"] > RULES["max_window_angle_deg"]
    ):
        _statuses(row, ("leaf_angle_deg",), "PROXIMAL_DIRECTION_UNSTABLE")
    else:
        _accept(row, ("leaf_angle_deg",), estimates)
    return row, audit, curve


def analyse_plant(plant_id, xyz, labels):
    valid = np.isfinite(xyz).all(axis=1)
    stem_points = xyz[(labels == 0) & valid]
    stem_spacing = _spacing(stem_points)
    stem = None
    stem_info = {"valid_points": len(stem_points), "median_neighbor_spacing": stem_spacing}
    try:
        if len(stem_points) < 80 or not np.ptp(stem_points, axis=0).any():
            raise ValueError("fewer than 80 valid stem points or degenerate span")
        stem = robust_stem(stem_points, PARAMS)
    except (ValueError, np.linalg.LinAlgError) as exc:
        stem_info.update(status="STEM_FIT_UNAVAILABLE", detail=str(exc))
    else:
        stem_info.update(status="ok", method=stem["method"], selected=stem["selected"],
                         axis=stem["axis"], radius=stem["radius"], length=stem["arc"][-1],
                         unsupported_fraction=stem["unsupported_curve_fraction"])

    rows, audits, curves = [], [], {}
    for attachment in locate_attachments(plant_id, xyz, labels):
        if attachment.leaf_id is None:
            continue
        points = xyz[valid & (labels == attachment.leaf_id)]
        row, audit, curve = _analyse_leaf(attachment, points, stem, stem_spacing)
        rows.append(row)
        audits.append(audit)
        if curve is not None:
            curves[attachment.leaf_id] = curve
    geometry = {"stem": stem["curve"] if stem else None, "leaves": curves}
    return rows, stem_info, audits, geometry
