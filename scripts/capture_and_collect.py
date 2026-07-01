#!/usr/bin/env python3
"""Capture a segmented workspace and record manual grasp contact points.

This script reuses the workspace capture from ``capture_and_grasp.py``, then
puts the Kinova arm in Cartesian admittance mode so the operator can hand-guide
the robot to a grasp pose and close the gripper. After confirmation, it records
the closest object point-cloud locations to the live left/right fingertip
positions.

Example:
    python scripts/capture_and_collect.py \\
        --calibration_file data/calibration/calibration.json \\
        --object bottle --merge_cameras
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections

    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    setattr(collections, "MutableSequence", collections.abc.MutableSequence)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_ROBOT_T_W_R_FILE = _REPO_ROOT / "config" / "robot_T_w_r.json"
for path in (_REPO_ROOT, _SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from kinova_gen3.camera.object_points import (  # noqa: E402
    iter_segmented_objects,
    load_capture_metadata,
    merge_object_point_clouds,
)
from kinova_gen3.camera.workspace import capture_workspace  # noqa: E402
from kinova_gen3.robot.execute import (  # noqa: E402
    load_robot_base_transform,
    set_gripper,
)
from kinova_gen3.robot.gripper_kinematics import (  # noqa: E402
    GripperTipPositions,
    format_tip_positions,
    query_live_fingertip_positions,
)
from kinova_gen3.robot.recordDemo import is_admittance_mode_active  # noqa: E402
from kinova_gen3.robot.utilities import DeviceConnection  # noqa: E402
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient  # noqa: E402
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient  # noqa: E402
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import (  # noqa: E402
    ControlConfigClient,
)
from kortex_api.autogen.messages import Base_pb2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a workspace, hand-guide the robot in admittance mode, "
            "and record fingertip contact points on the segmented object"
        )
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

    object_sel = parser.add_argument_group("object selection")
    object_sel.add_argument("--camera", default=None, help="Use only this camera serial")
    object_sel.add_argument("--merge_cameras", action="store_true")
    object_sel.add_argument("--instance", type=int, default=1)

    collect = parser.add_argument_group("contact collection")
    collect.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of manual grasp demonstrations to record in one session",
    )
    collect.add_argument(
        "--collection_output_dir",
        default=None,
        help="Directory for collection JSON (defaults to the workspace capture folder)",
    )
    collect.add_argument(
        "--gripper_name",
        default="robotiq_2f85",
        help="Fingertip geometry model used for live tip positions",
    )
    collect.add_argument(
        "--open_gripper_at_start",
        action="store_true",
        help="Open the gripper before entering admittance mode",
    )
    collect.add_argument(
        "--skip_admittance_enable",
        action="store_true",
        help=(
            "Do not call SetAdmittance; enable admittance manually on the bracelet "
            "or web app before hand-guiding"
        ),
    )
    collect.add_argument(
        "--max_contact_distance_m",
        type=float,
        default=0.05,
        help="Warn when the nearest object point is farther than this from a fingertip",
    )

    robot = parser.add_argument_group("robot")
    robot.add_argument(
        "--robot_T_w_r",
        default=str(DEFAULT_ROBOT_T_W_R_FILE),
        help="JSON file with a 4x4 T_w_r transform from robot base to world frame",
    )
    parser.add_argument("--ip", type=str, default="192.168.1.10", help="Robot IP address")
    parser.add_argument("-u", "--username", type=str, default="admin", help="Robot login username")
    parser.add_argument("-p", "--password", type=str, default="admin", help="Robot login password")
    return parser.parse_args()


def _admittance_mode_value(name: str, fallback: int) -> int:
    return int(getattr(Base_pb2, name, fallback))


def enable_cartesian_admittance(base: BaseClient) -> bool:
    """Put the arm in Cartesian admittance mode for hand-guiding."""
    try:
        admittance = Base_pb2.Admittance()
        admittance.admittance_mode = _admittance_mode_value("CARTESIAN", 1)
        base.SetAdmittance(admittance)
        print("Cartesian admittance mode enabled.")
        return True
    except Exception as exc:  # noqa: BLE001 - surface robot API failure
        print(f"Failed to enable Cartesian admittance mode: {exc}")
        return False


def disable_admittance(base: BaseClient) -> bool:
    """Disable admittance mode and return to normal control."""
    try:
        admittance = Base_pb2.Admittance()
        admittance.admittance_mode = _admittance_mode_value("DISABLED", 4)
        base.SetAdmittance(admittance)
        print("Admittance mode disabled.")
        return True
    except Exception as exc:  # noqa: BLE001 - surface robot API failure
        print(f"Failed to disable admittance mode: {exc}")
        return False


def load_object_point_cloud(workspace_dir: Path, args: argparse.Namespace):
    metadata = load_capture_metadata(workspace_dir)
    objects = list(
        iter_segmented_objects(
            workspace_dir,
            metadata,
            object_name=args.object.strip(),
            camera_serial=args.camera,
            instance_index=args.instance,
        )
    )
    if args.merge_cameras:
        return merge_object_point_clouds(objects), metadata
    if len(objects) != 1:
        serials = ", ".join(obj.camera_serial for obj in objects)
        raise ValueError(
            f"Multiple segmented object clouds matched ({serials}). "
            "Pass --camera to select one or use --merge_cameras."
        )
    return objects[0], metadata


def closest_points_on_cloud(
    object_points: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest object points, squared distances, and point indices."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(object_points, dtype=np.float64))
    distances, indices = tree.query(np.asarray(query_points, dtype=np.float64))
    nearest = object_points[indices]
    return nearest, distances, indices


def build_contact_record(
    tips: GripperTipPositions,
    object_points: np.ndarray,
    *,
    max_contact_distance_m: float,
) -> dict:
    fingertips = np.vstack([tips.left_world, tips.right_world])
    nearest_points, distances, indices = closest_points_on_cloud(object_points, fingertips)

    left_contact = nearest_points[0]
    right_contact = nearest_points[1]
    left_distance = float(distances[0])
    right_distance = float(distances[1])

    print("\nClosest object points to fingertips (world frame, meters):")
    print(
        f"  left  fingertip: [{tips.left_world[0]:.4f}, {tips.left_world[1]:.4f}, "
        f"{tips.left_world[2]:.4f}]"
    )
    print(
        f"  left  contact  : [{left_contact[0]:.4f}, {left_contact[1]:.4f}, "
        f"{left_contact[2]:.4f}]  (distance {left_distance:.4f} m, idx {int(indices[0])})"
    )
    print(
        f"  right fingertip: [{tips.right_world[0]:.4f}, {tips.right_world[1]:.4f}, "
        f"{tips.right_world[2]:.4f}]"
    )
    print(
        f"  right contact  : [{right_contact[0]:.4f}, {right_contact[1]:.4f}, "
        f"{right_contact[2]:.4f}]  (distance {right_distance:.4f} m, idx {int(indices[1])})"
    )

    for label, distance in (("left", left_distance), ("right", right_distance)):
        if distance > max_contact_distance_m:
            print(
                f"Warning: {label} fingertip is {distance:.4f} m from the nearest "
                f"object point (threshold {max_contact_distance_m:.4f} m)."
            )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "grip_value": float(tips.grip_value),
        "left_fingertip_world": tips.left_world.tolist(),
        "right_fingertip_world": tips.right_world.tolist(),
        "left_contact_on_object": left_contact.tolist(),
        "right_contact_on_object": right_contact.tolist(),
        "left_contact_distance_m": left_distance,
        "right_contact_distance_m": right_distance,
        "left_contact_object_index": int(indices[0]),
        "right_contact_object_index": int(indices[1]),
        "T_w_e": tips.T_w_e.tolist(),
        "T_r_e": tips.T_r_e.tolist(),
    }


def wait_for_user_confirmation(sample_index: int, total_samples: int) -> bool:
    prompt = (
        f"\nSample {sample_index}/{total_samples}: hand-guide the arm to the grasp pose, "
        "close the gripper, then press Enter to record contacts"
    )
    if total_samples > 1:
        prompt += " (or type 'q' to stop)"
    prompt += "..."
    response = input(prompt + "\n> ").strip().lower()
    return response not in {"q", "quit", "exit"}


def collect_manual_contacts(
    args: argparse.Namespace,
    workspace_dir: Path,
    object_cloud,
    metadata: dict,
) -> Path:
    T_w_r = load_robot_base_transform(args.robot_T_w_r)
    output_dir = (
        Path(args.collection_output_dir).expanduser().resolve()
        if args.collection_output_dir
        else workspace_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    print(f"\n=== Stage 2/2: Manual admittance collection ({args.num_samples} sample(s)) ===")
    print(
        "After admittance mode is active, hand-guide the robot to the target pose, "
        "close the gripper, then confirm each sample to record fingertip contacts."
    )

    with DeviceConnection.createTcpConnection(args) as tcp_router:
        with DeviceConnection.createUdpConnection(args) as udp_router:
            base = BaseClient(tcp_router)
            base_cyclic = BaseCyclicClient(udp_router)
            control_config = ControlConfigClient(tcp_router)

            if args.open_gripper_at_start:
                print("Opening gripper...")
                set_gripper(base, 0.0)

            if not args.skip_admittance_enable:
                if not enable_cartesian_admittance(base):
                    raise RuntimeError("Could not enable Cartesian admittance mode.")
            elif not is_admittance_mode_active(control_config):
                print(
                    "Warning: admittance mode is not active yet. Enable it on the "
                    "bracelet or web app before hand-guiding."
                )

            try:
                for sample_idx in range(1, args.num_samples + 1):
                    if not wait_for_user_confirmation(sample_idx, args.num_samples):
                        print("Stopping collection early.")
                        break

                    tips = query_live_fingertip_positions(
                        T_w_r,
                        base=base,
                        base_cyclic=base_cyclic,
                        gripper_name=args.gripper_name,
                    )
                    print(format_tip_positions(tips))

                    record = build_contact_record(
                        tips,
                        object_cloud.points,
                        max_contact_distance_m=args.max_contact_distance_m,
                    )
                    record["sample_index"] = sample_idx
                    records.append(record)
                    print(f"Recorded sample {sample_idx}.")
            finally:
                if not args.skip_admittance_enable:
                    disable_admittance(base)

    if not records:
        raise RuntimeError("No contact samples were recorded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"collection_{object_cloud.prompt}_inst{object_cloud.instance_index:02d}_{timestamp}"
    output_path = output_dir / f"{stem}.json"
    payload = {
        "workspace_dir": str(workspace_dir.resolve()),
        "calibration_file": metadata.get("calibration_file"),
        "object": object_cloud.prompt,
        "camera_serial": object_cloud.camera_serial,
        "instance_index": object_cloud.instance_index,
        "source_csv": str(object_cloud.source_csv),
        "num_object_points": int(len(object_cloud.points)),
        "gripper_name": args.gripper_name,
        "robot_T_w_r": args.robot_T_w_r,
        "samples": records,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {len(records)} contact sample(s) to: {output_path}")
    return output_path


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Path]:
    calibration_file = Path(args.calibration_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    if not args.object.strip():
        raise ValueError("--object must be a non-empty segmentation prompt")
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

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
        robot_T_w_r_file=args.robot_T_w_r,
    )
    print(f"\nCaptured workspace: {workspace_dir}")

    object_cloud, metadata = load_object_point_cloud(workspace_dir, args)
    print(
        f"Loaded object point cloud: {len(object_cloud.points)} points "
        f"({object_cloud.camera_serial}, instance {object_cloud.instance_index})"
    )

    collection_path = collect_manual_contacts(args, workspace_dir, object_cloud, metadata)
    print("\nPipeline complete.")
    return workspace_dir, collection_path


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
