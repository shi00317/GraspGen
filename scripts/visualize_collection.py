#!/usr/bin/env python3
"""Visualize workspace capture point clouds and manual grasp contact samples.

Loads a collection JSON produced by ``capture_and_collect.py`` and opens an
interactive viser viewer with:
  - merged workspace scene point cloud (gray)
  - segmented object point cloud (RGB from capture)
  - per-sample fingertip positions and nearest object contact points
  - lines from each fingertip to its matched object point
  - gripper end-effector frame and wireframe at the recorded pose

Example:
    python scripts/visualize_collection.py \\
        data/workspace_20260701_125843/collection_bottle_inst01_20260701_130005.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections

    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    setattr(collections, "MutableSequence", collections.abc.MutableSequence)


import numpy as np
import open3d as o3d
import trimesh.transformations as tra

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kinova_gen3.camera.object_points import (  # noqa: E402
    load_capture_metadata,
    load_object_points_csv,
    merge_object_point_clouds,
)
from kinova_gen3.robot.execute import load_robot_base_transform  # noqa: E402
from grasp_gen.utils.viser_utils import (  # noqa: E402
    create_visualizer,
    make_frame,
    visualize_grasp,
    visualize_pointcloud,
)

VIZ_GRIPPER_ALIASES = {
    "robotiq_2f85": "robotiq_2f_140",
    "robotiq_2f_85": "robotiq_2f_140",
    "robotiq_2f140": "robotiq_2f_140",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize capture_and_collect contact samples in viser"
    )
    parser.add_argument(
        "collection_json",
        type=str,
        help="Path to collection_*.json from capture_and_collect.py",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="1-based sample index to highlight (default: show all samples)",
    )
    parser.add_argument(
        "--no-workspace-pcd",
        action="store_true",
        help="Skip the full workspace scene point cloud",
    )
    parser.add_argument(
        "--workspace-downsample",
        type=float,
        default=0.008,
        help="Voxel size for workspace PCD downsampling (0 to disable)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port",
    )
    return parser.parse_args()


def load_collection(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def resolve_viz_gripper_name(gripper_name: str) -> str:
    key = gripper_name.lower().replace("-", "_")
    return VIZ_GRIPPER_ALIASES.get(key, gripper_name)


def load_object_cloud_from_collection(collection: dict):
    workspace_dir = Path(collection["workspace_dir"]).expanduser().resolve()
    source_csv = Path(collection["source_csv"]).expanduser().resolve()
    camera_serial = collection.get("camera_serial", "")

    if "+" in camera_serial:
        metadata = load_capture_metadata(workspace_dir)
        from kinova_gen3.camera.object_points import iter_segmented_objects

        objects = list(
            iter_segmented_objects(
                workspace_dir,
                metadata,
                object_name=collection["object"],
                camera_serial=None,
                instance_index=int(collection["instance_index"]),
            )
        )
        return merge_object_point_clouds(objects)

    points, colors = load_object_points_csv(source_csv)
    from kinova_gen3.camera.object_points import ObjectPointCloud

    return ObjectPointCloud(
        camera_serial=camera_serial,
        prompt=collection["object"],
        instance_index=int(collection["instance_index"]),
        points=points,
        colors=colors,
        source_csv=source_csv,
    )


def load_workspace_point_cloud(
    workspace_dir: Path,
    metadata: dict,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    pcd_name = metadata.get("workspace_pcd", "workspace.pcd")
    pcd_path = workspace_dir / pcd_name
    if not pcd_path.is_file():
        raise FileNotFoundError(f"Workspace point cloud not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    points = np.asarray(pcd.points, dtype=np.float64)
    if len(pcd.colors) == len(points):
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    return points, colors


def visualize_line_segment(
    vis,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    color: list[int],
    line_width: float = 2.0,
) -> None:
    segments = np.array([[start, end]], dtype=np.float32)
    vis.scene.add_line_segments(
        name,
        points=segments,
        colors=tuple(int(c) for c in color[:3]),
        line_width=line_width,
    )


def visualize_collection(
    collection_path: Path,
    *,
    sample_index: Optional[int] = None,
    show_workspace_pcd: bool = True,
    workspace_downsample: float = 0.008,
    port: int = 8080,
) -> None:
    collection = load_collection(collection_path)
    workspace_dir = Path(collection["workspace_dir"]).expanduser().resolve()
    metadata = load_capture_metadata(workspace_dir)
    object_cloud = load_object_cloud_from_collection(collection)

    samples = collection.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {collection_path}")

    if sample_index is not None:
        if sample_index < 1 or sample_index > len(samples):
            raise ValueError(
                f"--sample {sample_index} out of range for {len(samples)} sample(s)"
            )
        samples = [samples[sample_index - 1]]

    T_w_r = None
    robot_T_w_r = collection.get("robot_T_w_r")
    if robot_T_w_r:
        T_w_r = load_robot_base_transform(robot_T_w_r)

    center = object_cloud.points.mean(axis=0)
    T_center = tra.translation_matrix(-center)
    viz_gripper = resolve_viz_gripper_name(collection.get("gripper_name", "robotiq_2f_140"))

    vis = create_visualizer(port=port)

    if show_workspace_pcd:
        ws_points, ws_colors = load_workspace_point_cloud(
            workspace_dir,
            metadata,
            voxel_size=workspace_downsample,
        )
        visualize_pointcloud(
            vis,
            "scene/workspace_pcd",
            ws_points - center,
            ws_colors,
            size=0.002,
        )

    visualize_pointcloud(
        vis,
        "scene/object_pcd",
        object_cloud.points - center,
        object_cloud.colors,
        size=0.003,
    )

    make_frame(vis, "frames/world", h=0.20, radius=0.006, T=T_center)
    if T_w_r is not None:
        make_frame(vis, "frames/robot_base", h=0.15, radius=0.006, T=T_center @ T_w_r)

    for sample in samples:
        idx = int(sample.get("sample_index", 0))
        prefix = f"samples/{idx:03d}"

        left_tip = np.asarray(sample["left_fingertip_world"], dtype=np.float64)
        right_tip = np.asarray(sample["right_fingertip_world"], dtype=np.float64)
        left_contact = np.asarray(sample["left_contact_on_object"], dtype=np.float64)
        right_contact = np.asarray(sample["right_contact_on_object"], dtype=np.float64)

        tip_points = np.vstack([left_tip, right_tip])
        tip_colors = np.array([[255, 80, 80], [80, 80, 255]], dtype=np.uint8)
        visualize_pointcloud(
            vis,
            f"{prefix}/fingertips",
            tip_points - center,
            color=tip_colors,
            size=0.012,
        )

        contact_points = np.vstack([left_contact, right_contact])
        contact_colors = np.array([[255, 200, 0], [255, 140, 0]], dtype=np.uint8)
        visualize_pointcloud(
            vis,
            f"{prefix}/contacts_on_object",
            contact_points - center,
            color=contact_colors,
            size=0.014,
        )

        for side, tip, contact, color in (
            ("left", left_tip, left_contact, [255, 80, 80]),
            ("right", right_tip, right_contact, [80, 80, 255]),
        ):
            visualize_line_segment(
                vis,
                f"{prefix}/{side}_contact_line",
                tip - center,
                contact - center,
                color=color,
                line_width=2.5,
            )

        for side, index, color in (
            ("left", sample.get("left_contact_object_index"), [255, 220, 0]),
            ("right", sample.get("right_contact_object_index"), [255, 160, 0]),
        ):
            if index is None:
                continue
            pt = object_cloud.points[int(index)]
            visualize_pointcloud(
                vis,
                f"{prefix}/{side}_object_index",
                pt.reshape(1, 3) - center,
                color=[color],
                size=0.018,
            )

        T_w_e = np.asarray(sample["T_w_e"], dtype=np.float64)
        make_frame(
            vis,
            f"{prefix}/gripper_frame",
            h=0.12,
            radius=0.005,
            T=T_center @ T_w_e,
        )
        visualize_grasp(
            vis,
            f"{prefix}/gripper",
            T_center @ T_w_e,
            color=[0, 220, 120],
            gripper_name=viz_gripper,
            linewidth=1.5,
        )

        print(
            f"Sample {idx}: left contact {sample.get('left_contact_distance_m', 0):.4f} m, "
            f"right contact {sample.get('right_contact_distance_m', 0):.4f} m"
        )

    print(f"\nViser viewer: http://localhost:{port}")
    print("Legend:")
    print("  gray workspace PCD + colored object PCD")
    print("  red/blue spheres = left/right fingertips")
    print("  yellow/orange spheres = nearest object contact points")
    print("  lines = fingertip-to-contact association from capture_and_collect")
    input("Press Enter to close visualization...")


def main() -> None:
    args = parse_args()
    visualize_collection(
        Path(args.collection_json).expanduser().resolve(),
        sample_index=args.sample,
        show_workspace_pcd=not args.no_workspace_pcd,
        workspace_downsample=args.workspace_downsample,
        port=args.port,
    )


if __name__ == "__main__":
    main()
