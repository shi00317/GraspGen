#!/usr/bin/env python3
"""Capture a segmented workspace and generate object grasp poses.

This combines ``camera/capture_workspace.py`` and
``kinova_gen3/demo_workspace_grasp.py`` into one capture-to-grasp pipeline.

Example:
    python kinova_gen3/capture_and_grasp.py \
        --calibration_file camera/data/calibration/calibration.json \
        --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml \
        --object bottle --merge_cameras --return_topk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# When this file is run directly, Python adds ``kinova_gen3/`` rather than the
# repository root to sys.path.  Add the root so sibling packages such as
# ``camera`` and ``grasp_gen`` can be imported.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from camera.workspace import capture_workspace

if __package__:
    from .demo_workspace_grasp import (
        absolute_path_without_resolving,
        generate_workspace_grasps,
        validate_gripper_config,
    )
else:
    from demo_workspace_grasp import (
        absolute_path_without_resolving,
        generate_workspace_grasps,
        validate_gripper_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a workspace, segment an object, and generate grasps"
    )

    capture = parser.add_argument_group("workspace capture")
    capture.add_argument("--calibration_file", required=True)
    capture.add_argument("--output_dir", default="data")
    capture.add_argument("--object", required=True, help="Object prompt for SAM3")
    capture.add_argument("--width", type=int, default=1280)
    capture.add_argument("--height", type=int, default=720)
    capture.add_argument("--hardware_sync", action="store_true")
    capture.add_argument("--master_serial", default=None)
    capture.add_argument("--max_timestamp_diff_ms", type=float, default=33.0)
    capture.add_argument("--pcd_voxel_size", type=float, default=0.005)
    capture.add_argument("--max_range_m", type=float, default=1.3)
    capture.add_argument("--warmup_frames", type=int, default=30)
    capture.add_argument("--sam3_device", default=None, choices=["cuda", "cpu"])
    capture.add_argument("--sam3_score_threshold", type=float, default=0.0)
    capture.add_argument("--sam3_top_k", type=int, default=0)

    grasp = parser.add_argument_group("grasp generation")
    grasp.add_argument("--gripper_config", required=True)
    grasp.add_argument("--camera", default=None, help="Use only this camera serial")
    grasp.add_argument("--merge_cameras", action="store_true")
    grasp.add_argument("--instance", type=int, default=1)
    grasp.add_argument("--grasp_threshold", type=float, default=-1.0)
    grasp.add_argument("--num_grasps", type=int, default=200)
    grasp.add_argument("--return_topk", action="store_true")
    grasp.add_argument("--topk_num_grasps", type=int, default=-1)
    grasp.add_argument("--no_outlier_removal", action="store_true")
    grasp.add_argument("--grasp_voxel_size", type=float, default=0.002)
    grasp.add_argument("--grasp_output_dir", default=None)
    grasp.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """Run capture, segmentation, and grasp generation in sequence."""
    calibration_file = Path(args.calibration_file).expanduser().resolve()
    # Do not resolve this path: Hugging Face snapshot configs are symlinks and
    # their checkpoint filenames are relative to the snapshot directory.
    gripper_config = absolute_path_without_resolving(args.gripper_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    grasp_output_dir: Optional[Path] = (
        Path(args.grasp_output_dir).expanduser().resolve()
        if args.grasp_output_dir
        else None
    )
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    if not args.object.strip():
        raise ValueError("--object must be a non-empty segmentation prompt")

    # Validate all model files before initializing or warming up the cameras.
    validate_gripper_config(gripper_config)

    print("\n=== Stage 1/2: Capture and segment workspace ===")
    workspace_dir = capture_workspace(
        calibration_file=str(calibration_file),
        output_dir=str(output_dir),
        width=args.width,
        height=args.height,
        hardware_sync=args.hardware_sync,
        master_serial=args.master_serial,
        max_timestamp_diff_ms=args.max_timestamp_diff_ms,
        pcd_voxel_size=args.pcd_voxel_size,
        max_range_m=args.max_range_m,
        warmup_frames=args.warmup_frames,
        segment_prompts=[args.object.strip()],
        sam3_device=args.sam3_device,
        sam3_score_threshold=args.sam3_score_threshold,
        sam3_top_k=args.sam3_top_k,
    )

    print(f"\nCaptured workspace: {workspace_dir}")
    print("\n=== Stage 2/2: Generate grasp poses ===")
    outputs = generate_workspace_grasps(
        workspace_dir=workspace_dir,
        gripper_config=gripper_config,
        object_name=args.object.strip(),
        camera_serial=args.camera,
        merge_cameras=args.merge_cameras,
        instance_index=args.instance,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        return_topk=args.return_topk,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=not args.no_outlier_removal,
        voxel_size=args.grasp_voxel_size,
        output_dir=grasp_output_dir,
        visualize=not args.no_visualization,
    )

    if outputs:
        print(f"\nPipeline complete: saved {len(outputs)} grasp result set(s).")
    else:
        print("\nPipeline complete, but no valid grasps were generated.")
    return outputs


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
