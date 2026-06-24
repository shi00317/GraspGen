#!/usr/bin/env python3
"""Convert a PLY point cloud to the JSON format expected by demo_object_pc.py."""

import argparse
import json
import numpy as np
import trimesh


def voxel_downsample(pc: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Downsample by keeping one point per voxel cell."""
    voxel_indices = np.floor(pc / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pc[unique_idx], colors[unique_idx]


def adaptive_voxel_downsample(pc: np.ndarray, colors: np.ndarray, target: int):
    """Voxel-downsample to approximately `target` points using binary search.

    Preserves spatial density uniformity (unlike random subsampling), which is
    critical for the kNN-based outlier removal used in demo_object_pc.py.
    """
    bbox_diag = np.linalg.norm(pc.max(axis=0) - pc.min(axis=0))
    lo, hi = bbox_diag / len(pc), bbox_diag

    for _ in range(40):
        mid = (lo + hi) / 2
        voxel_indices = np.floor(pc / mid).astype(np.int64)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        n = len(unique_idx)
        if n == target:
            break
        elif n > target:
            lo = mid
        else:
            hi = mid

    # Final downsample at the converged voxel size
    voxel_indices = np.floor(pc / mid).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    # If still over target, randomly trim the excess (tiny adjustment)
    if len(unique_idx) > target:
        unique_idx = np.random.choice(unique_idx, size=target, replace=False)
    return pc[unique_idx], colors[unique_idx]


def main():
    parser = argparse.ArgumentParser(description="Convert PLY to demo_object_pc.py JSON")
    parser.add_argument("ply_file", help="Path to input .ply file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: same as input with .json extension)",
    )

    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument(
        "--max-points",
        type=int,
        default=None,
        metavar="N",
        help="Voxel-downsample to approximately N uniformly spaced points",
    )
    simplify_group.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        metavar="S",
        help="Voxel grid leaf size in meters (e.g. 0.005). Keeps one point per cell.",
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        default=None,
        metavar=("CX", "CY", "CZ", "R"),
        help=(
            "Crop a sphere from the scene before downsampling. "
            "Specify the center (cx cy cz) and radius r in the same units as the PLY. "
            "Example: --crop 0.1 0.0 0.9 0.2"
        ),
    )

    args = parser.parse_args()

    cloud = trimesh.load(args.ply_file)
    if not hasattr(cloud, "vertices"):
        raise ValueError(f"Loaded file has no vertices: {args.ply_file}")

    pc = np.asarray(cloud.vertices, dtype=np.float64)
    if hasattr(cloud, "colors") and cloud.colors is not None:
        colors = np.asarray(cloud.colors)
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        pc_color = colors
    else:
        pc_color = np.full((len(pc), 3), 128, dtype=np.uint8)

    print(f"Loaded {len(pc)} points from {args.ply_file}")

    bbox = pc.max(axis=0) - pc.min(axis=0)
    print(f"Bounding box: {bbox.round(3)}  (max extent: {bbox.max():.3f})")
    if bbox.max() > 0.5 and args.crop is None:
        print(
            "Warning: point cloud is large (>0.5 m). "
            "The model expects a single object (~0.1–0.3 m). "
            "Use --crop CX CY CZ R to extract the object of interest."
        )

    if args.crop is not None:
        cx, cy, cz, r = args.crop
        center = np.array([cx, cy, cz])
        dist = np.linalg.norm(pc - center, axis=1)
        mask = dist <= r
        pc, pc_color = pc[mask], pc_color[mask]
        print(f"After crop (center={center}, r={r}): {len(pc)} points")
        if len(pc) == 0:
            raise ValueError("Crop removed all points. Check --crop parameters.")

    if args.voxel_size is not None:
        pc, pc_color = voxel_downsample(pc, pc_color, args.voxel_size)
        print(f"After voxel downsampling (size={args.voxel_size}): {len(pc)} points")

    if args.max_points is not None and len(pc) > args.max_points:
        pc, pc_color = adaptive_voxel_downsample(pc, pc_color, args.max_points)
        print(f"After voxel downsampling to ~{args.max_points}: {len(pc)} points")

    # demo_object_pc.py expects grasp_poses and grasp_conf; use one dummy so process_point_cloud doesn't break
    identity_4x4 = np.eye(4).tolist()
    data = {
        "pc": pc.tolist(),
        "pc_color": pc_color.tolist(),
        "grasp_poses": [identity_4x4],
        "grasp_conf": [0.0],
    }

    out_path = args.output or args.ply_file.rsplit(".", 1)[0] + ".json"
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Wrote {len(pc)} points to {out_path}")


if __name__ == "__main__":
    main()
