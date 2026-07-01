#!/usr/bin/env python3
"""Capture a segmented workspace, generate grasp poses, and optionally execute one.

This combines ``kinova_gen3.camera.workspace``,
``scripts/demo_workspace_grasp.py``, and ``kinova_gen3.robot.execute`` into
one capture-to-grasp pipeline.

Example:
    python scripts/capture_and_grasp.py \
        --calibration_file data/calibration/calibration.json \
        --gripper_config /models/checkpoints/graspgen_robotiq_2f_140.yml \
        --object bottle --merge_cameras --return_topk --execute

Grasps are generated with the Robotiq 2F-140 model convention and converted
to 2F-85 poses before execution (see ``--no_gripper_conversion`` to skip).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if sys.version_info.major == 3 and sys.version_info.minor >= 10:   
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    setattr(collections,"MutableSequence", collections.abc.MutableSequence)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_ROBOT_T_W_R_FILE = _REPO_ROOT / "config" / "robot_T_w_r.json"
for path in (_REPO_ROOT, _SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from demo_workspace_grasp import (
    absolute_path_without_resolving,
    generate_workspace_grasps,
    validate_gripper_config,
)
from kinova_gen3.camera.workspace import capture_workspace
from kinova_gen3.robot.execute import (
    convert_robotiq_2f140_grasps_to_2f85,
    execute_world_grasp,
    load_grasps_from_json,
    load_robot_base_transform,
)
from kinova_gen3.utilities.convert_robotiq_2f140_to_2f85 import (
    ROBOTIQ_2F140_DEPTH_M,
    ROBOTIQ_2F85_DEPTH_M,
)
from kinova_gen3.robot.utilities import DeviceConnection
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient


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
    grasp.add_argument(
        "--query_gripper",
        action="store_true",
        help=(
            "Draw the live robot gripper frame and left/right fingertip locations "
            "in the grasp visualization (requires robot connection and T_w_r)"
        ),
    )

    robot = parser.add_argument_group("robot execution")
    robot.add_argument(
        "--execute",
        action="store_true",
        help="Move the Kinova arm to one generated grasp and close the gripper",
    )
    robot.add_argument(
        "--grasp_index",
        type=int,
        default=0,
        help="Index into the saved grasp list (0 = highest confidence when using --return_topk)",
    )
    robot.add_argument(
        "--grasp_set",
        type=int,
        default=0,
        help="Which saved grasp result set to use when multiple cameras are processed",
    )
    robot.add_argument(
        "--robot_T_w_r",
        default=str(DEFAULT_ROBOT_T_W_R_FILE),
        help="JSON file with a 4x4 T_w_r transform from robot base to world frame",
    )
    robot.add_argument(
        "--pre_grasp_offset",
        type=float,
        default=0.10,
        help="Pre-grasp retreat along approach axis in meters",
    )
    robot.add_argument("--speed", type=float, default=0.15, help="Cartesian motion speed")
    robot.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the selected grasp in robot base frame without moving",
    )
    robot.add_argument(
        "--no_gripper_conversion",
        action="store_true",
        help=(
            "Skip converting 2F-140 model grasps to Robotiq 2F-85 poses "
            "(conversion is on by default for --execute)"
        ),
    )
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Robot IP address")
    parser.add_argument("-u", "--username", type=str, default="admin", help="Robot login username")
    parser.add_argument("-p", "--password", type=str, default="admin", help="Robot login password")
    return parser.parse_args()


def execute_selected_grasp(
    args: argparse.Namespace,
    outputs: list[tuple[Path, Path]],
) -> bool:
    """Execute one grasp from the pipeline output on the Kinova robot."""
    if not outputs:
        print("No grasp outputs available to execute.")
        return False
    if args.grasp_set < 0 or args.grasp_set >= len(outputs):
        raise ValueError(
            f"--grasp_set {args.grasp_set} out of range for {len(outputs)} result set(s)"
        )

    _, json_path = outputs[args.grasp_set]
    grasps, confidences = load_grasps_from_json(str(json_path))
    if not args.no_gripper_conversion:
        shift_m = ROBOTIQ_2F140_DEPTH_M - ROBOTIQ_2F85_DEPTH_M
        grasps = convert_robotiq_2f140_grasps_to_2f85(grasps)
        print(
            f"Converted grasps from Robotiq 2F-140 model to 2F-85 "
            f"(local +Z shift {shift_m:.4f} m)"
        )
    if len(grasps) == 0:
        print(f"No grasps found in {json_path}")
        return False
    if args.grasp_index < 0 or args.grasp_index >= len(grasps):
        raise ValueError(
            f"--grasp_index {args.grasp_index} out of range for {len(grasps)} grasp(s)"
        )

    T_w_g = grasps[args.grasp_index]
    confidence = float(confidences[args.grasp_index])
    T_w_r = load_robot_base_transform(args.robot_T_w_r)

    print(f"\n=== Stage 3/3: Execute grasp {args.grasp_index} (conf={confidence:.4f}) ===")
    print(f"Grasp file: {json_path}")

    with DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        return execute_world_grasp(
            base,
            T_w_g,
            T_w_r,
            pre_grasp_offset_m=args.pre_grasp_offset,
            speed=args.speed,
            dry_run=args.dry_run,
        )


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

    total_stages = 3 if args.execute else 2

    print(f"\n=== Stage 1/{total_stages}: Capture and segment workspace ===")
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
        robot_T_w_r_file=args.robot_T_w_r,
    )

    print(f"\nCaptured workspace: {workspace_dir}")
    print(f"\n=== Stage 2/{total_stages}: Generate grasp poses ===")
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
        highlight_index=args.grasp_index,
        highlight_set=args.grasp_set,
        robot_T_w_r_file=args.robot_T_w_r,
        query_gripper=args.query_gripper,
        robot_ip=args.ip,
        robot_username=args.username,
        robot_password=args.password,
    )

    if outputs:
        print(f"\nSaved {len(outputs)} grasp result set(s).")
    else:
        print("\nNo valid grasps were generated.")
        return outputs

    if args.execute:
        success = execute_selected_grasp(args, outputs)
        if success:
            print("\nPipeline complete: grasp executed successfully.")
        else:
            print("\nPipeline complete, but grasp execution failed.")
    else:
        print("\nPipeline complete.")
    return outputs


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
