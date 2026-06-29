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
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import trimesh.transformations as tra

from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)


@dataclass(frozen=True)
class ObjectPointCloud:
    """Segmented object point cloud in the workspace world frame."""

    camera_serial: str
    prompt: str
    instance_index: int
    points: np.ndarray
    colors: np.ndarray
    source_csv: Path


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
    return parser.parse_args()


def load_capture_metadata(workspace_dir: Path) -> dict:
    capture_path = workspace_dir / "capture.json"
    if not capture_path.is_file():
        raise FileNotFoundError(f"Missing capture metadata: {capture_path}")
    with capture_path.open() as f:
        return json.load(f)


def resolve_object_points_csv(
    workspace_dir: Path,
    camera_serial: str,
    prompt: str,
    instance_index: int,
    recorded_path: str,
) -> Path:
    """Resolve object_points CSV path from capture metadata."""
    candidates = [
        Path(recorded_path),
        workspace_dir
        / "segments"
        / camera_serial
        / prompt
        / f"object_points_{instance_index:02d}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find object points CSV for "
        f"camera={camera_serial}, object={prompt}, instance={instance_index}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def load_object_points_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load xyz and rgb from an object_points_*.csv file."""
    points: list[list[float]] = []
    colors: list[list[int]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
            colors.append([int(row["r"]), int(row["g"]), int(row["b"])])
    if not points:
        raise ValueError(f"No points found in {csv_path}")
    return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def iter_segmented_objects(
    workspace_dir: Path,
    metadata: dict,
    object_name: Optional[str],
    camera_serial: Optional[str],
    instance_index: int,
) -> Iterator[ObjectPointCloud]:
    segmentation = metadata.get("segmentation")
    if not segmentation or not segmentation.get("enabled"):
        raise ValueError(f"No segmentation data found in {workspace_dir / 'capture.json'}")

    cameras = segmentation.get("cameras", {})
    for serial, objects in cameras.items():
        if camera_serial is not None and serial != camera_serial:
            continue
        for prompt, object_meta in objects.items():
            if object_name is not None and prompt != object_name:
                continue

            object_points_meta = object_meta.get("object_points", [])
            if instance_index < 1 or instance_index > len(object_points_meta):
                raise IndexError(
                    f"Instance {instance_index} not found for camera={serial}, object={prompt}. "
                    f"Available instances: {len(object_points_meta)}"
                )
            entry = object_points_meta[instance_index - 1]
            csv_path = resolve_object_points_csv(
                workspace_dir,
                serial,
                prompt,
                instance_index,
                entry["path"],
            )
            points, colors = load_object_points_csv(csv_path)
            yield ObjectPointCloud(
                camera_serial=serial,
                prompt=prompt,
                instance_index=instance_index,
                points=points,
                colors=colors,
                source_csv=csv_path,
            )


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return points[unique_idx], colors[unique_idx]


def merge_object_point_clouds(objects: Iterable[ObjectPointCloud]) -> ObjectPointCloud:
    clouds = list(objects)
    if not clouds:
        raise ValueError("No segmented object point clouds matched the selection")
    if len(clouds) == 1:
        return clouds[0]

    points = np.concatenate([cloud.points for cloud in clouds], axis=0)
    colors = np.concatenate([cloud.colors for cloud in clouds], axis=0)
    serials = "+".join(cloud.camera_serial for cloud in clouds)
    return ObjectPointCloud(
        camera_serial=serials,
        prompt=clouds[0].prompt,
        instance_index=clouds[0].instance_index,
        points=points,
        colors=colors,
        source_csv=clouds[0].source_csv,
    )


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


def visualize_results(
    object_cloud: ObjectPointCloud,
    grasps: np.ndarray,
    confidences: np.ndarray,
    gripper_name: str,
) -> None:
    vis = create_visualizer()
    pc_centered = object_cloud.points - object_cloud.points.mean(axis=0)
    grasps_centered = np.array(
        [
            tra.translation_matrix(-object_cloud.points.mean(axis=0)) @ grasp
            for grasp in grasps
        ]
    )
    visualize_pointcloud(
        vis,
        "object_pc",
        pc_centered,
        object_cloud.colors,
        size=0.0025,
    )
    scores = get_color_from_score(confidences, use_255_scale=True)
    for idx, grasp in enumerate(grasps_centered):
        visualize_grasp(
            vis,
            f"grasps/{idx:03d}/grasp",
            grasp,
            color=scores[idx],
            gripper_name=gripper_name,
            linewidth=0.6,
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
            visualize_results(object_cloud, grasps, confidences, gripper_name)
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
            visualize_results(object_cloud, grasps, confidences, gripper_name)

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
    )


if __name__ == "__main__":
    main()
