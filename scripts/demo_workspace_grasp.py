#!/usr/bin/env python3
"""Generate grasp poses from workspace capture object point clouds.

Reads segmented object points saved by capture_workspace.py (object_points_*.csv
under segments/<camera_serial>/<prompt>/) and runs GraspGen inference.

Example:
    python scripts/demo_workspace_grasp.py \\
        --workspace_dir data/workspace_20260623_172424 \\
        --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \\
        --object bottle \\
        --merge_cameras \\
        --return_topk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh.transformations as tra

# Ensure the repo root is importable so ``kinova_gen3`` resolves when this
# script is run directly (e.g. ``python scripts/demo_workspace_grasp.py``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
DEFAULT_ROBOT_T_W_R_FILE = _REPO_ROOT / "config" / "robot_T_w_r.json"

from kinova_gen3.camera.object_points import (
    ObjectPointCloud,
    iter_segmented_objects,
    load_capture_metadata,
    merge_object_point_clouds,
    voxel_downsample,
)

from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    make_frame,
    visualize_grasp,
    visualize_pointcloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraspGen on segmented object point clouds from a workspace capture"
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        required=True,
        help="Path to a workspace_* capture folder containing capture.json",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Segmentation prompt / object name (default: all segmented objects)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Use only this camera serial (default: all cameras in capture.json)",
    )
    parser.add_argument(
        "--merge_cameras",
        action="store_true",
        help="Merge object points from all selected cameras before inference",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=1,
        help="1-based object instance index from segmentation (default: 1)",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1.0,
        help="Confidence threshold. Use -1.0 to return top-k ranked grasps",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate per inference call",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Return only the top-k grasps (k=100 by default)",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps when --return_topk is set",
    )
    parser.add_argument(
        "--no_outlier_removal",
        action="store_true",
        help="Skip kNN-based outlier removal before inference",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.002,
        help="Optional voxel size (meters) to downsample merged object points",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for grasp outputs (default: <workspace_dir>/grasps)",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable viser visualization",
    )
    parser.add_argument(
        "--robot_T_w_r",
        type=str,
        default=str(DEFAULT_ROBOT_T_W_R_FILE),
        help=(
            "JSON file with the 4x4 T_w_r transform (robot base -> world frame). "
            "Used to draw the world and robot-base frames when the capture "
            "metadata does not already include one."
        ),
    )
    parser.add_argument(
        "--query_gripper",
        action="store_true",
        help=(
            "Connect to the live Kinova robot and draw the current gripper frame "
            "plus left/right fingertip locations in the world frame"
        ),
    )
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Robot IP address")
    parser.add_argument(
        "-u", "--username", type=str, default="admin", help="Robot login username"
    )
    parser.add_argument(
        "-p", "--password", type=str, default="admin", help="Robot login password"
    )
    return parser.parse_args()


def output_stem(object_cloud: ObjectPointCloud, merged: bool) -> str:
    if merged:
        return f"{object_cloud.prompt}_merged_inst{object_cloud.instance_index:02d}"
    return (
        f"{object_cloud.prompt}_{object_cloud.camera_serial}_"
        f"inst{object_cloud.instance_index:02d}"
    )


def save_grasp_results(
    output_dir: Path,
    stem: str,
    grasps: np.ndarray,
    confidences: np.ndarray,
    object_cloud: ObjectPointCloud,
    metadata: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / f"{stem}.yml"
    save_to_isaac_grasp_format(grasps, confidences, str(yaml_path))

    json_path = output_dir / f"{stem}.json"
    payload = {
        "workspace_dir": str(metadata["workspace_dir"]),
        "calibration_file": metadata.get("calibration_file"),
        "object": object_cloud.prompt,
        "camera_serial": object_cloud.camera_serial,
        "instance_index": object_cloud.instance_index,
        "source_csv": str(object_cloud.source_csv),
        "num_points": int(len(object_cloud.points)),
        "grasp_poses": grasps.tolist(),
        "grasp_conf": confidences.tolist(),
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(grasps)} grasps to:")
    print(f"  {yaml_path}")
    print(f"  {json_path}")
    return yaml_path, json_path


def run_inference_for_object(
    object_cloud: ObjectPointCloud,
    grasp_sampler: GraspGenSampler,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pc = object_cloud.points
    bbox = pc.max(axis=0) - pc.min(axis=0)
    print(
        f"Object '{object_cloud.prompt}' "
        f"(camera={object_cloud.camera_serial}, instance={object_cloud.instance_index})"
    )
    print(f"  source: {object_cloud.source_csv}")
    print(f"  points: {len(pc)}")
    print(f"  bbox (m): {bbox.round(4)}  max extent: {bbox.max():.4f}")

    if bbox.max() > 0.5:
        print(
            "Warning: object bounding box is large (>0.5 m). "
            "GraspGen expects a single object (~0.1–0.3 m)."
        )

    grasps, grasp_conf = GraspGenSampler.run_inference(
        pc,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=not args.no_outlier_removal,
    )
    if len(grasps) == 0:
        return np.empty((0, 4, 4)), np.empty((0,))

    grasps_np = grasps.cpu().numpy()
    conf_np = grasp_conf.cpu().numpy()
    grasps_np[:, 3, 3] = 1.0
    return grasps_np, conf_np


def load_world_robot_transform(
    metadata: dict,
    fallback_file: Optional[str | Path] = None,
) -> Optional[np.ndarray]:
    """Resolve T_w_r (robot base -> world frame) for the capture.

    Prefers the transform embedded in the capture metadata (saved by
    ``capture_workspace``) and falls back to a standalone JSON file.
    """
    robot_meta = metadata.get("robot_base_transform")
    if isinstance(robot_meta, dict) and "T_w_r" in robot_meta:
        return np.asarray(robot_meta["T_w_r"], dtype=np.float64)

    if fallback_file is not None:
        fallback_path = Path(fallback_file)
        if fallback_path.is_file():
            with fallback_path.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("T_w_r", data.get("transform", data))
            return np.asarray(data, dtype=np.float64)

    return None


def query_world_gripper_pose(
    T_w_r: np.ndarray,
    ip: str,
    username: str,
    password: str,
) -> np.ndarray:
    """Query the live robot end-effector pose and express it in the world frame.

    Returns ``T_w_e = T_w_r @ T_r_e`` where ``T_r_e`` is the current gripper
    pose in the robot base frame.
    """
    from kinova_gen3.robot.execute import get_current_pose
    from kinova_gen3.robot.utilities import DeviceConnection
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

    conn_args = argparse.Namespace(ip=ip, username=username, password=password)
    with DeviceConnection.createUdpConnection(conn_args) as router:
        base_cyclic = BaseCyclicClient(router)
        T_r_e = get_current_pose(base_cyclic)
    return np.asarray(T_w_r, dtype=np.float64) @ np.asarray(T_r_e, dtype=np.float64)


def query_world_fingertip_positions(
    T_w_r: np.ndarray,
    ip: str,
    username: str,
    password: str,
    *,
    gripper_name: str = "robotiq_2f85",
):
    """Query live left/right fingertip positions in the calibration world frame."""
    from kinova_gen3.robot.execute import query_world_fingertip_positions as _query_tips

    return _query_tips(
        T_w_r,
        ip=ip,
        username=username,
        password=password,
        gripper_name=gripper_name,
    )


def visualize_results(
    object_cloud: ObjectPointCloud,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
    highlight_index: Optional[int] = None,
    T_w_r: Optional[np.ndarray] = None,
    T_w_e: Optional[np.ndarray] = None,
    fingertip_positions: Optional[object] = None,
) -> None:
    if highlight_index is not None and (
        highlight_index < 0 or highlight_index >= len(grasps)
    ):
        raise ValueError(
            f"highlight_index {highlight_index} out of range for {len(grasps)} grasp(s)"
        )

    vis = create_visualizer()
    center = object_cloud.points.mean(axis=0)
    T_center = tra.translation_matrix(-center)
    pc_centered = object_cloud.points - center
    grasps_centered = np.array([T_center @ grasp for grasp in grasps])
    visualize_pointcloud(
        vis,
        "object_pc",
        pc_centered,
        object_cloud.colors,
        size=0.0025,
    )

    # Draw reference frames in the same centered coordinates as the point
    # cloud. The world frame is the ChArUco/dual-camera-extrinsic origin
    # (identity), the robot base comes from T_w_r, and the gripper frame is
    # T_w_e = T_w_r @ T_r_e from the live robot.
    make_frame(vis, "frames/world", h=0.20, radius=0.006, T=T_center)
    print("World frame (camera extrinsics): RGB triad at 'frames/world'")
    if T_w_r is not None:
        make_frame(vis, "frames/robot_base", h=0.15, radius=0.006, T=T_center @ T_w_r)
        print("Robot base frame: RGB triad at 'frames/robot_base'")
    else:
        print(
            "Robot base frame skipped: no T_w_r found in capture metadata or "
            "--robot_T_w_r file."
        )
    if T_w_e is not None:
        make_frame(vis, "frames/robot_gripper", h=0.12, radius=0.006, T=T_center @ T_w_e)
        print("Robot gripper frame: RGB triad at 'frames/robot_gripper'")
    if fingertip_positions is not None:
        tip_points = np.vstack(
            [fingertip_positions.left_world, fingertip_positions.right_world]
        )
        tip_colors = np.array([[255, 80, 80], [80, 80, 255]], dtype=np.uint8)
        visualize_pointcloud(
            vis,
            "frames/robot_fingertips",
            tip_points - center,
            color=tip_colors,
            size=0.012,
        )
        print(
            "Robot fingertips (world): left=red, right=blue at "
            "'frames/robot_fingertips'"
        )
    scores = get_color_from_score(confidences, use_255_scale=True)
    for idx, grasp in enumerate(grasps_centered):
        is_highlight = highlight_index is not None and idx == highlight_index
        visualize_grasp(
            vis,
            f"grasps/{idx:03d}/grasp",
            grasp,
            color=[0, 200, 255] if is_highlight else scores[idx],
            gripper_name=gripper_name,
            linewidth=2.5 if is_highlight else 0.6,
        )
    if highlight_index is not None:
        print(
            f"Highlighted grasp index {highlight_index} "
            f"(conf={confidences[highlight_index]:.4f})"
        )
    input("Press Enter to close visualization...")


def absolute_path_without_resolving(path: str | Path) -> Path:
    """Make a path absolute while preserving its final symlink.

    Hugging Face snapshot files are symlinks into the cache's ``blobs``
    directory. GraspGen resolves checkpoint filenames relative to the config's
    snapshot directory, so resolving the config symlink would break that
    relationship.
    """
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def validate_gripper_config(gripper_config: str | Path):
    """Load a gripper config and verify its referenced checkpoints exist."""
    config_path = absolute_path_without_resolving(gripper_config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Gripper config not found: {config_path}")

    grasp_cfg = load_grasp_cfg(str(config_path))
    checkpoint_paths = [Path(grasp_cfg.eval.checkpoint)]
    if grasp_cfg.eval.model_name == "diffusion-discriminator":
        checkpoint_paths.append(Path(grasp_cfg.discriminator.checkpoint))

    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Gripper checkpoint file(s) not found:\n  {formatted}\n"
            f"Config: {config_path}"
        )
    return grasp_cfg


def generate_workspace_grasps(
    workspace_dir: str | Path,
    gripper_config: str | Path,
    *,
    object_name: Optional[str] = None,
    camera_serial: Optional[str] = None,
    merge_cameras: bool = False,
    instance_index: int = 1,
    grasp_threshold: float = -1.0,
    num_grasps: int = 200,
    return_topk: bool = False,
    topk_num_grasps: int = -1,
    remove_outliers: bool = True,
    voxel_size: float = 0.002,
    output_dir: Optional[str | Path] = None,
    visualize: bool = True,
    highlight_index: Optional[int] = None,
    highlight_set: int = 0,
    robot_T_w_r_file: Optional[str | Path] = None,
    query_gripper: bool = False,
    robot_ip: str = "192.168.1.10",
    robot_username: str = "admin",
    robot_password: str = "admin",
) -> list[tuple[Path, Path]]:
    """Generate and save grasps for segmented clouds in a workspace capture.

    This is the programmatic entry point used by the capture-to-grasp pipeline.
    Paths in the returned list are ``(Isaac YAML, JSON)`` output pairs.
    """
    workspace_dir = Path(workspace_dir).resolve()
    if not workspace_dir.is_dir():
        raise FileNotFoundError(f"Workspace directory not found: {workspace_dir}")

    gripper_config = absolute_path_without_resolving(gripper_config)

    if return_topk and topk_num_grasps == -1:
        topk_num_grasps = 100

    # Keep the low-level inference helper CLI-compatible while exposing a
    # stable keyword API to other Python scripts.
    inference_args = argparse.Namespace(
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        no_outlier_removal=not remove_outliers,
    )

    metadata = load_capture_metadata(workspace_dir)
    metadata["workspace_dir"] = str(workspace_dir)

    segmented_objects = list(
        iter_segmented_objects(
            workspace_dir,
            metadata,
            object_name=object_name,
            camera_serial=camera_serial,
            instance_index=instance_index,
        )
    )
    if not segmented_objects:
        raise ValueError(
            "No segmented object point clouds matched the requested filters. "
            "Check --object / --camera / --instance."
        )

    grasp_output_dir = (
        Path(output_dir).resolve() if output_dir else workspace_dir / "grasps"
    )

    grasp_cfg = validate_gripper_config(gripper_config)
    grasp_sampler = GraspGenSampler(grasp_cfg)
    gripper_name = grasp_cfg.data.gripper_name
    saved_outputs: list[tuple[Path, Path]] = []

    # Resolve reference frames for visualization.
    T_w_r: Optional[np.ndarray] = None
    T_w_e: Optional[np.ndarray] = None
    fingertip_positions = None
    if visualize:
        T_w_r = load_world_robot_transform(metadata, robot_T_w_r_file)
        if query_gripper:
            if T_w_r is None:
                print(
                    "Cannot draw gripper frame: no T_w_r available "
                    "(provide --robot_T_w_r or capture with a robot transform)."
                )
            else:
                try:
                    fingertip_positions = query_world_fingertip_positions(
                        T_w_r,
                        robot_ip,
                        robot_username,
                        robot_password,
                        gripper_name="robotiq_2f85",
                    )
                    T_w_e = fingertip_positions.T_w_e
                except Exception as exc:  # noqa: BLE001 - visualization is best-effort
                    print(f"Warning: could not query live gripper pose: {exc}")

    if merge_cameras:
        prompts = sorted({obj.prompt for obj in segmented_objects})
        if len(prompts) != 1:
            raise ValueError(
                f"--merge_cameras requires a single object prompt, got: {prompts}"
            )
        object_cloud = merge_object_point_clouds(segmented_objects)
        if voxel_size > 0:
            points, colors = voxel_downsample(
                object_cloud.points, object_cloud.colors, voxel_size
            )
            object_cloud = ObjectPointCloud(
                camera_serial=object_cloud.camera_serial,
                prompt=object_cloud.prompt,
                instance_index=object_cloud.instance_index,
                points=points,
                colors=colors,
                source_csv=object_cloud.source_csv,
            )
            print(
                f"After merge + voxel downsample ({voxel_size} m): "
                f"{len(points)} points"
            )

        grasps, confidences = run_inference_for_object(
            object_cloud, grasp_sampler, inference_args
        )
        if len(grasps) == 0:
            print("No grasps found.")
            return saved_outputs

        stem = output_stem(object_cloud, merged=True)
        saved_outputs.append(
            save_grasp_results(
                grasp_output_dir,
                stem,
                grasps,
                confidences,
                object_cloud,
                metadata,
            )
        )
        if visualize:
            visualize_results(
                object_cloud,
                grasps,
                confidences,
                gripper_name,
                highlight_index=highlight_index if highlight_set == 0 else None,
                T_w_r=T_w_r,
                T_w_e=T_w_e,
                fingertip_positions=fingertip_positions,
            )
        return saved_outputs

    for object_cloud in segmented_objects:
        if voxel_size > 0:
            points, colors = voxel_downsample(
                object_cloud.points, object_cloud.colors, voxel_size
            )
            object_cloud = ObjectPointCloud(
                camera_serial=object_cloud.camera_serial,
                prompt=object_cloud.prompt,
                instance_index=object_cloud.instance_index,
                points=points,
                colors=colors,
                source_csv=object_cloud.source_csv,
            )
            print(
                f"After voxel downsample ({voxel_size} m): {len(points)} points"
            )

        grasps, confidences = run_inference_for_object(
            object_cloud, grasp_sampler, inference_args
        )
        if len(grasps) == 0:
            print(
                f"No grasps found for camera={object_cloud.camera_serial}, "
                f"object={object_cloud.prompt}"
            )
            continue

        stem = output_stem(object_cloud, merged=False)
        saved_outputs.append(
            save_grasp_results(
                grasp_output_dir,
                stem,
                grasps,
                confidences,
                object_cloud,
                metadata,
            )
        )
        if visualize:
            output_idx = len(saved_outputs) - 1
            visualize_results(
                object_cloud,
                grasps,
                confidences,
                gripper_name,
                highlight_index=(
                    highlight_index if output_idx == highlight_set else None
                ),
                T_w_r=T_w_r,
                T_w_e=T_w_e,
                fingertip_positions=fingertip_positions,
            )

    return saved_outputs


def main() -> None:
    args = parse_args()
    generate_workspace_grasps(
        workspace_dir=args.workspace_dir,
        gripper_config=args.gripper_config,
        object_name=args.object,
        camera_serial=args.camera,
        merge_cameras=args.merge_cameras,
        instance_index=args.instance,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        return_topk=args.return_topk,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=not args.no_outlier_removal,
        voxel_size=args.voxel_size,
        output_dir=args.output_dir,
        visualize=not args.no_visualization,
        robot_T_w_r_file=args.robot_T_w_r,
        query_gripper=args.query_gripper,
        robot_ip=args.ip,
        robot_username=args.username,
        robot_password=args.password,
    )


if __name__ == "__main__":
    main()
