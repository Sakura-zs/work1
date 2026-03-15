#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import open3d as o3d


def load_pcd(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    return pcd


def preprocess_pcd(pcd: o3d.geometry.PointCloud, voxel_size=0.0, sor_k=0, sor_std=2.0):
    q = pcd
    if voxel_size and voxel_size > 0:
        q = q.voxel_down_sample(voxel_size=float(voxel_size))
    if sor_k and sor_k > 0:
        q, _ = q.remove_statistical_outlier(nb_neighbors=int(sor_k), std_ratio=float(sor_std))
    if len(q.points) == 0:
        raise ValueError("All points removed by preprocessing")
    return q


def make_pcd(points: np.ndarray, color=None):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if color is not None:
        p = p.paint_uniform_color(color)
    return p


def orthonormal_basis_from_axis(axis: np.ndarray):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    if abs(axis[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(axis, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v, axis


# ----------------------------- Box -----------------------------

def infer_is_box(ply_path: Path) -> bool:
    parent = ply_path.parent.name.lower()
    return ("box" in parent) or ("fangti" in parent) or ("cuboid" in parent) or ("rect" in parent)


def obb_to_quat_wxyz(R: np.ndarray):
    tr = float(np.trace(R))
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def fit_box_obb(pcd: o3d.geometry.PointCloud):
    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center, dtype=np.float64)
    dims = np.asarray(obb.extent, dtype=np.float64)
    R = np.asarray(obb.R, dtype=np.float64)
    quat = obb_to_quat_wxyz(R)
    return center, dims, quat


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def keep_largest_cluster(pcd: o3d.geometry.PointCloud, eps=0.03, min_points=20):
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return pcd
    counts = np.bincount(labels[labels >= 0])
    largest = int(np.argmax(counts))
    idx = np.where(labels == largest)[0].tolist()
    return pcd.select_by_index(idx)


def fit_box_from_planes(pcd: o3d.geometry.PointCloud, plane_dist=0.01, plane_iters=2000, min_inliers=150, max_planes=3):
    work = pcd
    normals = []

    for _ in range(max_planes):
        if len(work.points) < min_inliers:
            break
        model, inliers = work.segment_plane(distance_threshold=float(plane_dist), ransac_n=3, num_iterations=int(plane_iters))
        if len(inliers) < min_inliers:
            break
        a, b, c, _d = model
        normals.append(_normalize([a, b, c]))
        work = work.select_by_index(inliers, invert=True)

    if len(normals) == 0:
        return fit_box_obb(pcd)

    n1 = normals[0]
    n2 = None
    best_absdot = 1e9
    for n in normals[1:]:
        absdot = abs(float(np.dot(n1, n)))
        if absdot < best_absdot:
            best_absdot = absdot
            n2 = n

    pts = np.asarray(pcd.points, dtype=np.float64)
    mu = pts.mean(axis=0)

    if n2 is None:
        X = pts - mu
        cov = (X.T @ X) / max(1, len(pts))
        w, v = np.linalg.eigh(cov)
        t = _normalize(v[:, np.argmax(w)])
        n2 = _normalize(t - np.dot(t, n1) * n1)

    x = _normalize(n1)
    y = _normalize(n2 - np.dot(n2, x) * x)
    z = _normalize(np.cross(x, y))

    R = np.stack([x, y, z], axis=1)

    local = (pts - mu) @ R
    mn = local.min(axis=0)
    mx = local.max(axis=0)
    dims = (mx - mn)

    center_local = (mn + mx) * 0.5
    center_world = mu + (R @ center_local)

    quat = obb_to_quat_wxyz(R)
    return center_world, dims, quat


def allocate_points_by_area(areas, total_points):
    areas = np.array(areas, dtype=np.float64)
    areas = np.maximum(areas, 1e-12)
    weights = areas / areas.sum()
    counts = np.floor(weights * total_points).astype(int)
    rem = total_points - counts.sum()
    if rem > 0:
        order = np.argsort(-(weights * total_points - counts))
        for i in range(rem):
            counts[order[i % len(order)]] += 1
    return counts.tolist()


def generate_box_surface_model_fixedN(center, dims, quat_wxyz, num_points: int, seed=0):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    dims = np.asarray(dims, dtype=np.float64).reshape(3)
    R = quat_wxyz_to_R(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))

    dx, dy, dz = dims.tolist()
    hx, hy, hz = dx/2.0, dy/2.0, dz/2.0

    areas = [dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy]
    counts = allocate_points_by_area(areas, num_points)

    rng = np.random.default_rng(seed)

    def sample_face(const_axis, const_val, a_lim, b_lim, k):
        a = rng.uniform(-a_lim, a_lim, size=(k,))
        b = rng.uniform(-b_lim, b_lim, size=(k,))
        P = np.zeros((k, 3), dtype=np.float64)
        if const_axis == 0:
            P[:, 0] = const_val; P[:, 1] = a; P[:, 2] = b
        elif const_axis == 1:
            P[:, 0] = a; P[:, 1] = const_val; P[:, 2] = b
        else:
            P[:, 0] = a; P[:, 1] = b; P[:, 2] = const_val
        return P

    local_pts = []
    local_pts.append(sample_face(0, +hx, hy, hz, counts[0]))
    local_pts.append(sample_face(0, -hx, hy, hz, counts[1]))
    local_pts.append(sample_face(1, +hy, hx, hz, counts[2]))
    local_pts.append(sample_face(1, -hy, hx, hz, counts[3]))
    local_pts.append(sample_face(2, +hz, hx, hy, counts[4]))
    local_pts.append(sample_face(2, -hz, hx, hy, counts[5]))

    local_pts = np.vstack(local_pts)
    world_pts = (local_pts @ R.T) + center.reshape(1, 3)
    return make_pcd(world_pts, color=[0.0, 0.4, 1.0])


def make_box_id(index: int) -> str:
    return f"{index:04d}_box"


def run_box(args):
    root = Path(args.root).resolve()
    models_root = (root / args.models_dir)
    models_root.mkdir(parents=True, exist_ok=True)

    if getattr(args, "input", None):
        ply_files = [Path(args.input).resolve()]
    else:
        ply_files = sorted(root.rglob("*.ply"))
    results = []
    idx = 0
    n_models = 0

    for ply_path in ply_files:
        if not infer_is_box(ply_path):
            continue
        rel = ply_path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")

        try:
            pcd = preprocess_pcd(load_pcd(ply_path), voxel_size=args.voxel, sor_k=args.sor_k, sor_std=args.sor_std)
            if args.box_cluster_eps and args.box_cluster_eps > 0:
                pcd = keep_largest_cluster(pcd, eps=args.box_cluster_eps, min_points=args.box_cluster_min_points)

            if args.box_method == "pca_obb":
                center, dims, quat = fit_box_obb(pcd)
            else:
                center, dims, quat = fit_box_from_planes(
                    pcd,
                    plane_dist=args.box_plane_dist,
                    plane_iters=args.box_plane_iters,
                    min_inliers=args.box_min_inliers,
                )

            entry = {
                "id": make_box_id(idx),
                "file_path": rel_str,
                "shape_type": "box",
                "shape_idx": 2,
                "gt_params": {"center": center.tolist(), "dims": dims.tolist(), "quat": quat.tolist()},
            }
            results.append(entry)
            idx += 1

            model_out_path = (models_root / rel).with_name(rel.stem + "_model.ply")
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            model_pcd = generate_box_surface_model_fixedN(center, dims, quat, num_points=int(args.model_points), seed=0)
            o3d.io.write_point_cloud(str(model_out_path), model_pcd)
            n_models += 1

        except Exception as e:
            print(f"[WARN] Failed on {rel}: {e}")

    out_path = Path(args.out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(results)} box entries to {out_path}")
    print(f"Wrote {n_models} box model ply files under {models_root}")


# ----------------------------- Plane -----------------------------

def infer_is_plane(ply_path: Path) -> bool:
    parent = ply_path.parent.name.lower()
    return ("pingmian" in parent) or ("plane" in parent)


def fit_plane_ransac(pcd: o3d.geometry.PointCloud, dist=0.01, iters=3000):
    model, inliers = pcd.segment_plane(distance_threshold=float(dist), ransac_n=3, num_iterations=int(iters))
    a, b, c, d = model
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm
    d = float(d / n_norm)
    return n, d, inliers


def classify_shape_geom(pcd: o3d.geometry.PointCloud, args) -> str:
    total = len(pcd.points)
    if total == 0:
        return "box"

    try:
        n, d, inliers = fit_plane_ransac(pcd, dist=args.plane_dist, iters=args.plane_iters)
        plane_ratio = float(len(inliers)) / float(max(1, total))
        if plane_ratio >= float(args.geom_plane_ratio):
            return "plane"
        if plane_ratio >= float(args.geom_plane_min_ratio):
            pts = np.asarray(pcd.points, dtype=np.float64)
            dist_all = np.abs(pts @ n + float(d))
            thickness = float(np.percentile(dist_all, 95) - np.percentile(dist_all, 5))
            if thickness <= float(args.geom_plane_thickness_max):
                return "plane"
    except Exception:
        pass

    try:
        center, axis, h = fit_cylinder_axis_height(pcd)
        pts = np.asarray(pcd.points, dtype=np.float64)
        r = point_to_axis_distance(pts, np.asarray(center, dtype=np.float64), np.asarray(axis, dtype=np.float64))
        r_mean = float(np.mean(r)) if r.size > 0 else 0.0
        r_std = float(np.std(r)) if r.size > 0 else 1e9
        if r_mean > 1e-9:
            r_std_ratio = r_std / r_mean
            height_ratio = float(h) / float(max(1e-9, 2.0 * r_mean))
            if r_std_ratio <= float(args.geom_cyl_rstd_max) and height_ratio >= float(args.geom_cyl_min_height_ratio):
                return "cylinder"
            r0 = float(np.median(r))
            mad = float(np.median(np.abs(r - r0)) + 1e-12)
            inlier_ratio = float(np.mean(np.abs(r - r0) < float(args.geom_cyl_mad_k) * mad))
            if inlier_ratio >= float(args.geom_cyl_inlier_ratio_min) and height_ratio >= float(args.geom_cyl_min_height_ratio):
                return "cylinder"
    except Exception:
        pass

    return "box"


def generate_plane_model_fixedN(pcd: o3d.geometry.PointCloud, inliers, n: np.ndarray, d: float, num_points: int, seed=0):
    inlier_pcd = pcd.select_by_index(inliers)
    pts = np.asarray(inlier_pcd.points, dtype=np.float64)
    if pts.shape[0] < 10:
        raise ValueError("Too few inlier points for plane model")

    u, v, _ = orthonormal_basis_from_axis(n)
    p0 = -float(d) * n

    rel = pts - p0
    uu = rel @ u
    vv = rel @ v
    u_min, u_max = float(np.min(uu)), float(np.max(uu))
    v_min, v_max = float(np.min(vv)), float(np.max(vv))

    rng = np.random.default_rng(seed)
    u_s = rng.uniform(u_min, u_max, size=(num_points,))
    v_s = rng.uniform(v_min, v_max, size=(num_points,))
    grid = p0.reshape(1, 3) + np.outer(u_s, u) + np.outer(v_s, v)
    return make_pcd(grid, color=[1.0, 0.0, 0.0])


def make_plane_id(index: int) -> str:
    return f"{index:04d}_plane"


def run_plane(args):
    root = Path(args.root).resolve()
    models_root = (root / args.models_dir)
    models_root.mkdir(parents=True, exist_ok=True)

    if getattr(args, "input", None):
        ply_files = [Path(args.input).resolve()]
    else:
        ply_files = sorted(root.rglob("*.ply"))
    results = []
    idx = 0
    n_models = 0

    for ply_path in ply_files:
        if not infer_is_plane(ply_path):
            continue

        rel = ply_path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")
        try:
            pcd = preprocess_pcd(load_pcd(ply_path), voxel_size=args.voxel, sor_k=args.sor_k, sor_std=args.sor_std)
            n, d, inliers = fit_plane_ransac(pcd, dist=args.plane_dist, iters=args.plane_iters)

            entry = {
                "id": make_plane_id(idx),
                "file_path": rel_str,
                "shape_type": "plane",
                "shape_idx": 0,
                "gt_params": {"normal": n.tolist(), "d": float(d)},
            }
            results.append(entry)
            idx += 1

            model_out_path = (models_root / rel).with_name(rel.stem + "_model.ply")
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            model_pcd = generate_plane_model_fixedN(pcd, inliers, n, d, num_points=int(args.model_points), seed=0)
            o3d.io.write_point_cloud(str(model_out_path), model_pcd)
            n_models += 1

        except Exception as e:
            print(f"[WARN] Failed on {rel}: {e}")

    out_path = Path(args.out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(results)} plane entries to {out_path}")
    print(f"Wrote {n_models} plane model ply files under {models_root}")


# ----------------------------- Cylinder -----------------------------

def infer_is_cylinder(ply_path: Path) -> bool:
    parent = ply_path.parent.name.lower()
    return ("yuanzhu" in parent) or ("cyl" in parent) or ("cylinder" in parent)


def pca_axis(points: np.ndarray):
    mu = points.mean(axis=0)
    X = points - mu
    cov = (X.T @ X) / max(1, len(points))
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return mu, axis


def fit_cylinder_axis_height(pcd: o3d.geometry.PointCloud, robust_percentile=(2, 98)):
    pts = np.asarray(pcd.points, dtype=np.float64)
    mu, axis = pca_axis(pts)

    t = (pts - mu) @ axis
    t_min, t_max = np.percentile(t, list(robust_percentile))
    h = float(t_max - t_min)

    center = mu + axis * float((t_min + t_max) * 0.5)
    return center, axis, h


def point_to_axis_distance(points: np.ndarray, center: np.ndarray, axis: np.ndarray):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    X = points - center.reshape(1, 3)
    t = X @ axis
    radial = X - np.outer(t, axis)
    return np.linalg.norm(radial, axis=1)


def fit_cylinder_radius_robust(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    mad_k: float = 3.0,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    d = point_to_axis_distance(points, center, axis)

    if r_min is not None:
        d = d[d >= float(r_min)]
    if r_max is not None:
        d = d[d <= float(r_max)]

    if d.size < 20:
        raise ValueError("Too few points to estimate radius (after optional r_min/r_max filtering)")

    r0 = float(np.median(d))
    mad = float(np.median(np.abs(d - r0)) + 1e-12)
    inliers = np.abs(d - r0) < float(mad_k) * mad

    if int(inliers.sum()) >= 20:
        r = float(np.median(d[inliers]))
    else:
        r = r0

    stats = {
        "r0": r0,
        "mad": mad,
        "inlier_ratio": float(inliers.mean()),
        "num_used": int(d.size),
    }
    return r, inliers, stats


def generate_cylinder_model_fixedN(center, axis, radius, height, num_points: int, seed=0):
    rng = np.random.default_rng(seed)
    u, v, w = orthonormal_basis_from_axis(axis)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(num_points,))
    t = rng.uniform(-height / 2.0, height / 2.0, size=(num_points,))
    pts = (
        center.reshape(1, 3)
        + np.outer(t, w)
        + radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    )
    return make_pcd(pts, color=[0.1, 1.0, 0.1])


def load_radius_from_existing_gt(gt_json_path: Path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for item in data:
        fp = item.get("file_path")
        if not fp:
            continue
        if item.get("shape_type") == "cylinder":
            r = item.get("gt_params", {}).get("r", None)
            m[fp] = r
    return m


def make_cylinder_id(index: int) -> str:
    return f"{index:04d}_cylinder"


def run_cylinder(args):
    root = Path(args.root).resolve()
    models_root = root / args.models_dir
    models_root.mkdir(parents=True, exist_ok=True)

    radius_map = {}
    if args.use_radius_from:
        radius_map = load_radius_from_existing_gt(Path(args.use_radius_from).resolve())

    if getattr(args, "input", None):
        ply_files = [Path(args.input).resolve()]
    else:
        ply_files = sorted(root.rglob("*.ply"))

    results = []
    idx = 0
    n_models = 0
    skipped_no_r = 0
    n_total_cyl = 0

    for ply_path in ply_files:
        if not infer_is_cylinder(ply_path):
            continue

        n_total_cyl += 1
        rel = ply_path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")

        try:
            pcd = preprocess_pcd(load_pcd(ply_path), voxel_size=args.voxel, sor_k=args.sor_k, sor_std=args.sor_std)

            center, axis, h = fit_cylinder_axis_height(pcd)

            r = radius_map.get(rel_str, None)

            radius_stats = None
            if r is None and args.fit_radius:
                pts = np.asarray(pcd.points, dtype=np.float64)
                r_fit, _inliers, stats = fit_cylinder_radius_robust(
                    pts,
                    center=np.asarray(center, dtype=np.float64),
                    axis=np.asarray(axis, dtype=np.float64),
                    mad_k=float(args.radius_mad_k),
                    r_min=args.radius_min,
                    r_max=args.radius_max,
                )
                if stats["inlier_ratio"] < float(args.min_radius_inlier_ratio):
                    r = None
                else:
                    r = float(r_fit)
                radius_stats = stats

            entry = {
                "id": make_cylinder_id(idx),
                "file_path": rel_str,
                "shape_type": "cylinder",
                "shape_idx": 1,
                "gt_params": {
                    "center": np.asarray(center, dtype=np.float64).tolist(),
                    "h": float(h),
                    "r": r,
                    "axis": np.asarray(axis, dtype=np.float64).tolist(),
                },
            }
            if radius_stats is not None:
                entry["radius_fit_stats"] = radius_stats

            results.append(entry)
            idx += 1

            if r is None:
                skipped_no_r += 1
                continue

            model_out_path = (models_root / rel).with_name(rel.stem + "_model.ply")
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            model_pcd = generate_cylinder_model_fixedN(
                np.asarray(center, dtype=np.float64),
                np.asarray(axis, dtype=np.float64),
                float(r),
                float(h),
                num_points=int(args.model_points),
                seed=0,
            )
            o3d.io.write_point_cloud(str(model_out_path), model_pcd)
            n_models += 1

        except Exception as e:
            print(f"[WARN] Failed on {rel}: {e}")

    out_path = Path(args.out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Found {n_total_cyl} cylinder ply files under {root}")
    print(f"Wrote {len(results)} cylinder entries to {out_path}")
    print(f"Wrote {n_models} cylinder model ply files under {models_root}")
    if skipped_no_r:
        print(f"Skipped {skipped_no_r} cylinder model ply files because r is None.")
        print("Tip: provide r via --use_radius_from <json> or enable --fit_radius")


def run_auto_geom(args):
    root = Path(args.root).resolve()

    models_root_box = root / args.models_dir_box
    models_root_plane = root / args.models_dir_plane
    models_root_cyl = root / args.models_dir_cylinder
    models_root_box.mkdir(parents=True, exist_ok=True)
    models_root_plane.mkdir(parents=True, exist_ok=True)
    models_root_cyl.mkdir(parents=True, exist_ok=True)

    radius_map = {}
    if args.use_radius_from:
        radius_map = load_radius_from_existing_gt(Path(args.use_radius_from).resolve())

    if getattr(args, "input", None):
        ply_files = [Path(args.input).resolve()]
    else:
        ply_files = sorted(root.rglob("*.ply"))

    results_box = []
    results_plane = []
    results_cyl = []
    idx_box = 0
    idx_plane = 0
    idx_cyl = 0
    n_models_box = 0
    n_models_plane = 0
    n_models_cyl = 0
    skipped_no_r = 0

    for ply_path in ply_files:
        rel = ply_path.relative_to(root)
        rel_str = str(rel).replace("\\", "/")

        try:
            pcd = preprocess_pcd(load_pcd(ply_path), voxel_size=args.voxel, sor_k=args.sor_k, sor_std=args.sor_std)
            if args.debug_stats:
                try:
                    n_dbg, d_dbg, inliers_dbg = fit_plane_ransac(pcd, dist=args.plane_dist, iters=args.plane_iters)
                    plane_ratio_dbg = float(len(inliers_dbg)) / float(max(1, len(pcd.points)))
                    pts_dbg = np.asarray(pcd.points, dtype=np.float64)
                    dist_dbg = np.abs(pts_dbg @ n_dbg + float(d_dbg))
                    thickness_dbg = float(np.percentile(dist_dbg, 95) - np.percentile(dist_dbg, 5))
                    print(
                        f"[DEBUG] {rel_str} plane_ratio={plane_ratio_dbg:.4f} thickness={thickness_dbg:.6f}"
                    )
                except Exception as e:
                    print(f"[DEBUG] {rel_str} plane_stats_failed: {e}")
            shape = classify_shape_geom(pcd, args)

            if shape == "plane":
                n, d, inliers = fit_plane_ransac(pcd, dist=args.plane_dist, iters=args.plane_iters)
                entry = {
                    "id": make_plane_id(idx_plane),
                    "file_path": rel_str,
                    "shape_type": "plane",
                    "shape_idx": 0,
                    "gt_params": {"normal": n.tolist(), "d": float(d)},
                }
                results_plane.append(entry)
                idx_plane += 1

                model_out_path = (models_root_plane / rel).with_name(rel.stem + "_model.ply")
                model_out_path.parent.mkdir(parents=True, exist_ok=True)
                model_pcd = generate_plane_model_fixedN(pcd, inliers, n, d, num_points=int(args.model_points), seed=0)
                o3d.io.write_point_cloud(str(model_out_path), model_pcd)
                n_models_plane += 1

            elif shape == "cylinder":
                center, axis, h = fit_cylinder_axis_height(pcd)
                r = radius_map.get(rel_str, None)

                radius_stats = None
                if r is None and args.fit_radius:
                    pts = np.asarray(pcd.points, dtype=np.float64)
                    r_fit, _inliers, stats = fit_cylinder_radius_robust(
                        pts,
                        center=np.asarray(center, dtype=np.float64),
                        axis=np.asarray(axis, dtype=np.float64),
                        mad_k=float(args.radius_mad_k),
                        r_min=args.radius_min,
                        r_max=args.radius_max,
                    )
                    if stats["inlier_ratio"] < float(args.min_radius_inlier_ratio):
                        r = None
                    else:
                        r = float(r_fit)
                    radius_stats = stats

                entry = {
                    "id": make_cylinder_id(idx_cyl),
                    "file_path": rel_str,
                    "shape_type": "cylinder",
                    "shape_idx": 1,
                    "gt_params": {
                        "center": np.asarray(center, dtype=np.float64).tolist(),
                        "h": float(h),
                        "r": r,
                        "axis": np.asarray(axis, dtype=np.float64).tolist(),
                    },
                }
                if radius_stats is not None:
                    entry["radius_fit_stats"] = radius_stats

                results_cyl.append(entry)
                idx_cyl += 1

                if r is None:
                    skipped_no_r += 1
                    continue

                model_out_path = (models_root_cyl / rel).with_name(rel.stem + "_model.ply")
                model_out_path.parent.mkdir(parents=True, exist_ok=True)
                model_pcd = generate_cylinder_model_fixedN(
                    np.asarray(center, dtype=np.float64),
                    np.asarray(axis, dtype=np.float64),
                    float(r),
                    float(h),
                    num_points=int(args.model_points),
                    seed=0,
                )
                o3d.io.write_point_cloud(str(model_out_path), model_pcd)
                n_models_cyl += 1

            else:
                if args.box_cluster_eps and args.box_cluster_eps > 0:
                    pcd = keep_largest_cluster(pcd, eps=args.box_cluster_eps, min_points=args.box_cluster_min_points)

                if args.box_method == "pca_obb":
                    center, dims, quat = fit_box_obb(pcd)
                else:
                    center, dims, quat = fit_box_from_planes(
                        pcd,
                        plane_dist=args.box_plane_dist,
                        plane_iters=args.box_plane_iters,
                        min_inliers=args.box_min_inliers,
                    )

                entry = {
                    "id": make_box_id(idx_box),
                    "file_path": rel_str,
                    "shape_type": "box",
                    "shape_idx": 2,
                    "gt_params": {"center": center.tolist(), "dims": dims.tolist(), "quat": quat.tolist()},
                }
                results_box.append(entry)
                idx_box += 1

                model_out_path = (models_root_box / rel).with_name(rel.stem + "_model.ply")
                model_out_path.parent.mkdir(parents=True, exist_ok=True)
                model_pcd = generate_box_surface_model_fixedN(center, dims, quat, num_points=int(args.model_points), seed=0)
                o3d.io.write_point_cloud(str(model_out_path), model_pcd)
                n_models_box += 1

        except Exception as e:
            print(f"[WARN] Failed on {rel}: {e}")

    out_box = Path(args.out_box).resolve()
    out_plane = Path(args.out_plane).resolve()
    out_cyl = Path(args.out_cylinder).resolve()

    with open(out_box, "w", encoding="utf-8") as f:
        json.dump(results_box, f, ensure_ascii=False, indent=4)
    with open(out_plane, "w", encoding="utf-8") as f:
        json.dump(results_plane, f, ensure_ascii=False, indent=4)
    with open(out_cyl, "w", encoding="utf-8") as f:
        json.dump(results_cyl, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(results_box)} box entries to {out_box}")
    print(f"Wrote {len(results_plane)} plane entries to {out_plane}")
    print(f"Wrote {len(results_cyl)} cylinder entries to {out_cyl}")
    print(f"Wrote {n_models_box} box model ply files under {models_root_box}")
    print(f"Wrote {n_models_plane} plane model ply files under {models_root_plane}")
    print(f"Wrote {n_models_cyl} cylinder model ply files under {models_root_cyl}")
    if skipped_no_r:
        print(f"Skipped {skipped_no_r} cylinder model ply files because r is None.")


# ----------------------------- CLI -----------------------------

def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=False)

    def add_common(p):
        p.add_argument("--root", default=".")
        p.add_argument("--input", default="", help="optional single .ply file path to process")
        p.add_argument("--debug_stats", action="store_true", help="print plane inlier ratio/thickness for each file")
        p.add_argument("--model_points", type=int, default=8000)
        p.add_argument("--voxel", type=float, default=0.0)
        p.add_argument("--sor_k", type=int, default=0)
        p.add_argument("--sor_std", type=float, default=2.0)

    p_box = sub.add_parser("box")
    add_common(p_box)
    p_box.add_argument("--out", default="gt_box.json")
    p_box.add_argument("--models_dir", default="_gt_models_box")
    p_box.add_argument("--box_method", choices=["pca_obb", "planes"], default="planes")
    p_box.add_argument("--box_plane_dist", type=float, default=0.01)
    p_box.add_argument("--box_plane_iters", type=int, default=2000)
    p_box.add_argument("--box_min_inliers", type=int, default=150)
    p_box.add_argument("--box_cluster_eps", type=float, default=0.0)
    p_box.add_argument("--box_cluster_min_points", type=int, default=20)

    p_plane = sub.add_parser("plane")
    add_common(p_plane)
    p_plane.add_argument("--out", default="gt_plane.json")
    p_plane.add_argument("--models_dir", default="_gt_models_plane")
    p_plane.add_argument("--plane_dist", type=float, default=0.01)
    p_plane.add_argument("--plane_iters", type=int, default=3000)

    p_cyl = sub.add_parser("cylinder")
    add_common(p_cyl)
    p_cyl.add_argument("--out", default="gt_cylinder.json")
    p_cyl.add_argument("--models_dir", default="_gt_models_cylinder")
    p_cyl.add_argument("--use_radius_from", default="")
    p_cyl.add_argument("--fit_radius", action="store_true", default=True)
    p_cyl.add_argument("--radius_mad_k", type=float, default=3.0)
    p_cyl.add_argument("--radius_min", type=float, default=None)
    p_cyl.add_argument("--radius_max", type=float, default=None)
    p_cyl.add_argument("--min_radius_inlier_ratio", type=float, default=0.2)

    p_all = sub.add_parser("all")
    add_common(p_all)
    p_all.add_argument("--out_box", default="gt_box.json")
    p_all.add_argument("--out_plane", default="gt_plane.json")
    p_all.add_argument("--out_cylinder", default="gt_cylinder.json")
    p_all.add_argument("--models_dir_box", default="_gt_models_box")
    p_all.add_argument("--models_dir_plane", default="_gt_models_plane")
    p_all.add_argument("--models_dir_cylinder", default="_gt_models_cylinder")
    p_all.add_argument("--box_method", choices=["pca_obb", "planes"], default="planes")
    p_all.add_argument("--box_plane_dist", type=float, default=0.01)
    p_all.add_argument("--box_plane_iters", type=int, default=2000)
    p_all.add_argument("--box_min_inliers", type=int, default=150)
    p_all.add_argument("--box_cluster_eps", type=float, default=0.0)
    p_all.add_argument("--box_cluster_min_points", type=int, default=20)
    p_all.add_argument("--plane_dist", type=float, default=0.01)
    p_all.add_argument("--plane_iters", type=int, default=3000)
    p_all.add_argument("--use_radius_from", default="")
    p_all.add_argument("--fit_radius", action="store_true", default=True)
    p_all.add_argument("--radius_mad_k", type=float, default=3.0)
    p_all.add_argument("--radius_min", type=float, default=None)
    p_all.add_argument("--radius_max", type=float, default=None)
    p_all.add_argument("--min_radius_inlier_ratio", type=float, default=0.2)

    p_auto = sub.add_parser("auto")
    add_common(p_auto)
    p_auto.add_argument("--out_box", default="gt_box.json")
    p_auto.add_argument("--out_plane", default="gt_plane.json")
    p_auto.add_argument("--out_cylinder", default="gt_cylinder.json")
    p_auto.add_argument("--models_dir_box", default="_gt_models_box")
    p_auto.add_argument("--models_dir_plane", default="_gt_models_plane")
    p_auto.add_argument("--models_dir_cylinder", default="_gt_models_cylinder")
    p_auto.add_argument("--box_method", choices=["pca_obb", "planes"], default="planes")
    p_auto.add_argument("--box_plane_dist", type=float, default=0.01)
    p_auto.add_argument("--box_plane_iters", type=int, default=2000)
    p_auto.add_argument("--box_min_inliers", type=int, default=150)
    p_auto.add_argument("--box_cluster_eps", type=float, default=0.0)
    p_auto.add_argument("--box_cluster_min_points", type=int, default=20)
    p_auto.add_argument("--plane_dist", type=float, default=0.01)
    p_auto.add_argument("--plane_iters", type=int, default=3000)
    p_auto.add_argument("--use_radius_from", default="")
    p_auto.add_argument("--fit_radius", action="store_true", default=True)
    p_auto.add_argument("--radius_mad_k", type=float, default=3.0)
    p_auto.add_argument("--radius_min", type=float, default=None)
    p_auto.add_argument("--radius_max", type=float, default=None)
    p_auto.add_argument("--min_radius_inlier_ratio", type=float, default=0.2)
    p_auto.add_argument("--geom_plane_ratio", type=float, default=0.6)
    p_auto.add_argument("--geom_plane_min_ratio", type=float, default=0.3)
    p_auto.add_argument("--geom_plane_thickness_max", type=float, default=0.1)
    p_auto.add_argument("--geom_cyl_rstd_max", type=float, default=0.12)
    p_auto.add_argument("--geom_cyl_mad_k", type=float, default=3.0)
    p_auto.add_argument("--geom_cyl_inlier_ratio_min", type=float, default=0.4)
    p_auto.add_argument("--geom_cyl_min_height_ratio", type=float, default=1.0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd is None:
        args.cmd = "auto"
        auto_defaults = {
            "out_box": "gt_box.json",
            "out_plane": "gt_plane.json",
            "out_cylinder": "gt_cylinder.json",
            "models_dir_box": "_gt_models_box",
            "models_dir_plane": "_gt_models_plane",
            "models_dir_cylinder": "_gt_models_cylinder",
            "box_method": "planes",
            "box_plane_dist": 0.01,
            "box_plane_iters": 2000,
            "box_min_inliers": 150,
            "box_cluster_eps": 0.0,
            "box_cluster_min_points": 20,
            "plane_dist": 0.01,
            "plane_iters": 3000,
            "use_radius_from": "",
            "fit_radius": True,
            "radius_mad_k": 3.0,
            "radius_min": None,
            "radius_max": None,
            "min_radius_inlier_ratio": 0.2,
            "geom_plane_ratio": 0.6,
            "geom_plane_min_ratio": 0.3,
            "geom_plane_thickness_max": 0.1,
            "geom_cyl_rstd_max": 0.12,
            "geom_cyl_mad_k": 3.0,
            "geom_cyl_inlier_ratio_min": 0.4,
            "geom_cyl_min_height_ratio": 1.0,
        }
        for k, v in auto_defaults.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    if args.cmd == "box":
        run_box(args)
    elif args.cmd == "plane":
        run_plane(args)
    elif args.cmd == "cylinder":
        run_cylinder(args)
    elif args.cmd == "all":
        box_args = argparse.Namespace(**vars(args))
        box_args.out = args.out_box
        box_args.models_dir = args.models_dir_box
        run_box(box_args)

        plane_args = argparse.Namespace(**vars(args))
        plane_args.out = args.out_plane
        plane_args.models_dir = args.models_dir_plane
        run_plane(plane_args)

        cyl_args = argparse.Namespace(**vars(args))
        cyl_args.out = args.out_cylinder
        cyl_args.models_dir = args.models_dir_cylinder
        run_cylinder(cyl_args)
    elif args.cmd == "auto":
        run_auto_geom(args)


if __name__ == "__main__":
    main()