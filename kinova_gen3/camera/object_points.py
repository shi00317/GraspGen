"""Load segmented object point clouds from workspace capture outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class ObjectPointCloud:
    """Segmented object point cloud in the workspace world frame."""

    camera_serial: str
    prompt: str
    instance_index: int
    points: np.ndarray
    colors: np.ndarray
    source_csv: Path


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
