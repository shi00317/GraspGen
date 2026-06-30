#!/usr/bin/env python3
"""Convert Robotiq 2F-140 GraspGen poses for use with a Robotiq 2F-85.

The pretrained model in this repo predicts poses using the Robotiq 2F-140
gripper convention. The 2F-85 shares the same base/chassis convention but has
shorter fingers and a smaller maximum opening. This script post-processes saved
grasp poses by moving the gripper base forward along each grasp's local +Z
approach axis so that the 2F-85 fingertip/TCP location stays aligned with the
location predicted for the 2F-140.

It supports the JSON files written by my_data/demo_workspace_grasp.py and the
Isaac grasp YAML files written beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# GraspGen-convention base_frame -> fingerpad/contact depths, taken from the
# gripper configs (config/grippers/robotiq_2f_140.yaml and robotiq_2f_85.yaml).
# These are the distances that define where each gripper's TCP sits along +Z,
# NOT the datasheet outer "open height" of the gripper body.
ROBOTIQ_2F140_DEPTH_M = 0.1950  # robotiq_2f_140.yaml: depth / contact z
ROBOTIQ_2F85_DEPTH_M = 0.052324 #0.130324  # robotiq_2f_85.yaml: contact location z
ROBOTIQ_2F85_MAX_APERTURE_M = 0.085


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Robotiq 2F-140 grasp poses to Robotiq 2F-85 poses."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input grasp file (.json from demo_workspace_grasp.py, or Isaac .yml/.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <input_stem>_robotiq_2f85.<same suffix>",
    )
    parser.add_argument(
        "--source-depth",
        type=float,
        default=ROBOTIQ_2F140_DEPTH_M,
        help="2F-140 depth/contact offset in meters (default: 0.1950)",
    )
    parser.add_argument(
        "--target-depth",
        type=float,
        default=ROBOTIQ_2F85_DEPTH_M,
        help="2F-85 depth/contact offset in meters (default: 0.1303)",
    )
    parser.add_argument(
        "--max-aperture",
        type=float,
        default=ROBOTIQ_2F85_MAX_APERTURE_M,
        help="2F-85 maximum inner opening in meters, used by --filter-by-aperture",
    )
    parser.add_argument(
        "--filter-by-aperture",
        action="store_true",
        help=(
            "Drop grasps whose local object width estimate exceeds --max-aperture. "
            "Requires --object-points or a JSON input with source_csv."
        ),
    )
    parser.add_argument(
        "--object-points",
        type=Path,
        default=None,
        help="Optional object_points_*.csv used to estimate grasp width for filtering.",
    )
    parser.add_argument(
        "--aperture-margin",
        type=float,
        default=0.005,
        help="Extra clearance, in meters, added to the estimated local object width.",
    )
    parser.add_argument(
        "--write-paired-yaml",
        action="store_true",
        help="When converting JSON, also write a paired Isaac YAML grasp file.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_robotiq_2f85{input_path.suffix}")


def load_json_grasps(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)

    grasps = np.asarray(payload["grasp_poses"], dtype=np.float64)
    confidences = np.asarray(payload["grasp_conf"], dtype=np.float64)
    validate_grasps(grasps, confidences, path)
    return grasps, confidences, payload


def load_isaac_yaml_grasps(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import trimesh.transformations as tra
    import yaml

    with path.open() as f:
        payload = yaml.safe_load(f)

    grasp_items = list(payload.get("grasps", {}).items())
    grasps: list[np.ndarray] = []
    confidences: list[float] = []

    for _, grasp in grasp_items:
        transform = tra.quaternion_matrix(
            [grasp["orientation"]["w"], *grasp["orientation"]["xyz"]]
        )
        transform[:3, 3] = np.asarray(grasp["position"], dtype=np.float64)
        grasps.append(transform)
        confidences.append(float(grasp["confidence"]))

    grasps_np = np.asarray(grasps, dtype=np.float64)
    confidences_np = np.asarray(confidences, dtype=np.float64)
    validate_grasps(grasps_np, confidences_np, path)
    return grasps_np, confidences_np, payload


def validate_grasps(grasps: np.ndarray, confidences: np.ndarray, path: Path) -> None:
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"{path} does not contain an Nx4x4 grasp_poses array")
    if len(grasps) != len(confidences):
        raise ValueError(
            f"{path} has {len(grasps)} grasps but {len(confidences)} confidences"
        )


def convert_depth(grasps: np.ndarray, source_depth: float, target_depth: float) -> np.ndarray:
    converted = np.array(grasps, copy=True)
    local_shift = np.array([0.0, 0.0, source_depth - target_depth], dtype=np.float64)
    converted[:, :3, 3] += converted[:, :3, :3] @ local_shift
    converted[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return converted


def load_object_points_csv(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if not points:
        raise ValueError(f"No object points found in {path}")
    return np.asarray(points, dtype=np.float64)


def estimate_local_widths(
    grasps: np.ndarray,
    object_points: np.ndarray,
    target_depth: float,
    aperture_margin: float,
) -> np.ndarray:
    widths = np.full(len(grasps), np.inf, dtype=np.float64)
    if len(object_points) == 0:
        return widths

    depth_window = 0.035
    lateral_window = 0.05

    for idx, grasp in enumerate(grasps):
        rotation = grasp[:3, :3]
        translation = grasp[:3, 3]
        local_points = (object_points - translation) @ rotation

        in_closing_region = (
            (local_points[:, 2] >= target_depth - depth_window)
            & (local_points[:, 2] <= target_depth + depth_window)
            & (np.abs(local_points[:, 1]) <= lateral_window)
        )
        region_points = local_points[in_closing_region]
        if len(region_points) < 3:
            continue

        widths[idx] = float(np.ptp(region_points[:, 0]) + aperture_margin)

    return widths


def resolve_object_points_path(
    args: argparse.Namespace, payload: dict[str, Any], input_path: Path
) -> Path | None:
    if args.object_points is not None:
        return args.object_points

    source_csv = payload.get("source_csv")
    if source_csv is None:
        return None

    source_path = Path(source_csv)
    if source_path.is_file():
        return source_path

    workspace_dir = Path(payload.get("workspace_dir", input_path.parent.parent))
    source_parts = source_path.parts
    if "segments" in source_parts:
        segment_idx = source_parts.index("segments")
        segment_relpath = Path(*source_parts[segment_idx:])
        candidate = workspace_dir / segment_relpath
        if candidate.is_file():
            return candidate

    return None


def save_json_output(
    path: Path,
    payload: dict[str, Any],
    grasps: np.ndarray,
    confidences: np.ndarray,
    conversion: dict[str, Any],
) -> None:
    output_payload = dict(payload)
    output_payload["grasp_poses"] = grasps.tolist()
    output_payload["grasp_conf"] = confidences.tolist()
    output_payload["gripper_name"] = "robotiq_2f85"
    output_payload["conversion"] = conversion
    with path.open("w") as f:
        json.dump(output_payload, f, indent=2)


def main() -> None:
    import trimesh.transformations as tra
    import yaml

    from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format

    args = parse_args()

    input_path = args.input.resolve()
    output_path = (args.output or default_output_path(input_path)).resolve()

    if input_path.suffix.lower() == ".json":
        grasps, confidences, payload = load_json_grasps(input_path)
        input_format = "json"
    elif input_path.suffix.lower() in {".yml", ".yaml"}:
        grasps, confidences, payload = load_isaac_yaml_grasps(input_path)
        input_format = "isaac_yaml"
    else:
        raise ValueError(f"Unsupported grasp file suffix: {input_path.suffix}")

    converted_grasps = convert_depth(
        grasps, source_depth=args.source_depth, target_depth=args.target_depth
    )

    keep_mask = np.ones(len(converted_grasps), dtype=bool)
    width_estimates: np.ndarray | None = None
    object_points_path = None
    if args.filter_by_aperture:
        object_points_path = resolve_object_points_path(args, payload, input_path)
        if object_points_path is None:
            raise ValueError(
                "--filter-by-aperture requires --object-points, or a JSON input "
                "whose source_csv exists on disk"
            )
        object_points = load_object_points_csv(object_points_path)
        width_estimates = estimate_local_widths(
            converted_grasps,
            object_points,
            target_depth=args.target_depth,
            aperture_margin=args.aperture_margin,
        )
        keep_mask = width_estimates <= args.max_aperture

    converted_grasps = converted_grasps[keep_mask]
    confidences = confidences[keep_mask]

    conversion = {
        "from_gripper": "robotiq_2f140",
        "to_gripper": "robotiq_2f85",
        "source_depth_m": args.source_depth,
        "target_depth_m": args.target_depth,
        "local_approach_shift_m": args.source_depth - args.target_depth,
        "max_aperture_m": args.max_aperture,
        "aperture_filter_enabled": bool(args.filter_by_aperture),
        "num_input_grasps": int(len(grasps)),
        "num_output_grasps": int(len(converted_grasps)),
    }
    if object_points_path is not None:
        conversion["object_points"] = str(object_points_path)
    if width_estimates is not None:
        conversion["dropped_by_aperture"] = int(np.count_nonzero(~keep_mask))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        save_json_output(output_path, payload, converted_grasps, confidences, conversion)
        if args.write_paired_yaml:
            yaml_path = output_path.with_suffix(".yml")
            save_to_isaac_grasp_format(converted_grasps, confidences, str(yaml_path))
            print(f"Wrote paired Isaac YAML: {yaml_path}")
    elif output_path.suffix.lower() in {".yml", ".yaml"}:
        save_to_isaac_grasp_format(converted_grasps, confidences, str(output_path))
    else:
        raise ValueError(f"Unsupported output suffix: {output_path.suffix}")

    print(
        f"Converted {len(grasps)} grasps from Robotiq 2F-140 to "
        f"{len(converted_grasps)} Robotiq 2F-85 grasps."
    )
    print(f"Applied local +Z shift: {args.source_depth - args.target_depth:.4f} m")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
