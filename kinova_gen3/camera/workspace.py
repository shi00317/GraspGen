"""Synchronized multi-camera workspace capture."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from .realsense import RealsenseCamera
from .visualization import merge_rgbd_frames_to_world


def load_calibration(calibration_file: str) -> Dict[str, np.ndarray]:
    """Load world-to-camera transforms from a calibration JSON file."""
    with open(calibration_file) as f:
        data = json.load(f)
    return {serial: np.array(transform) for serial, transform in data.items()}


def load_transform_4x4(transform_file: str, key: str = "T_w_r") -> np.ndarray:
    """Load a homogeneous 4x4 transform from JSON."""
    with open(transform_file) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if key in data:
            data = data[key]
        elif "transform" in data:
            data = data["transform"]

    transform = np.array(data, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected {transform_file} to contain a 4x4 transform, got {transform.shape}"
        )
    return transform


def configure_hardware_sync(
    serials: list[str],
    master_serial: Optional[str] = None,
) -> rs.context:
    """
    Configure RealSense hardware sync before starting camera pipelines.

    One camera is set as master (mode 1), all others as slave (mode 2).
    Requires physical sync cabling between cameras.
    """
    ctx = rs.context()
    if not serials:
        return ctx

    if master_serial is None:
        master_serial = serials[0]
    elif master_serial not in serials:
        raise ValueError(f"Master serial {master_serial} is not in calibration")

    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        if serial not in serials:
            continue

        mode = 1 if serial == master_serial else 2
        dev.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, mode)
        role = "master" if mode == 1 else "slave"
        print(f"Camera {serial}: hardware sync = {role}")

    return ctx


def _camera_init_order(
    serials: list[str],
    hardware_sync: bool,
    master_serial: Optional[str],
) -> list[str]:
    """Start the sync master first when hardware sync is enabled."""
    if not hardware_sync or len(serials) <= 1:
        return serials

    master = master_serial or serials[0]
    return [master] + [serial for serial in serials if serial != master]


def capture_synced_rgbd_frames(
    cameras: Dict[str, RealsenseCamera],
    max_timestamp_diff_ms: float = 33.0,
    timeout_sec: float = 5.0,
) -> Tuple[Dict[str, o3d.geometry.RGBDImage], Dict[str, float]]:
    """
    Capture one RGB-D frame per camera with closest timestamp alignment.

    Returns:
        RGB-D frames and depth timestamps (ms) keyed by camera serial.
    """
    serials = list(cameras.keys())
    if not serials:
        raise RuntimeError("No cameras available for capture")

    latest: Dict[str, Tuple[o3d.geometry.RGBDImage, float]] = {}
    deadline = time.time() + timeout_sec
    best_frames: Optional[Dict[str, o3d.geometry.RGBDImage]] = None
    best_timestamps: Optional[Dict[str, float]] = None
    best_diff_ms = float("inf")

    while time.time() < deadline:
        for serial, camera in cameras.items():
            result = camera.get_frame_with_timestamp()
            if result is not None:
                latest[serial] = result

        if len(latest) < len(serials):
            continue

        timestamps = {serial: latest[serial][1] for serial in serials}
        diff_ms = max(timestamps.values()) - min(timestamps.values())

        if diff_ms <= max_timestamp_diff_ms:
            return (
                {serial: latest[serial][0] for serial in serials},
                timestamps,
            )

        if diff_ms < best_diff_ms:
            best_diff_ms = diff_ms
            best_frames = {serial: latest[serial][0] for serial in serials}
            best_timestamps = timestamps

    if best_frames is not None and best_timestamps is not None:
        print(
            f"Warning: using best-effort sync "
            f"(timestamp delta={best_diff_ms:.1f} ms > {max_timestamp_diff_ms} ms)"
        )
        return best_frames, best_timestamps

    raise RuntimeError("Failed to capture synchronized frames from all cameras")


def save_workspace_capture(
    output_folder: Path,
    rgb_images: Dict[str, np.ndarray],
    workspace_pcd: o3d.geometry.PointCloud,
    metadata: dict,
    rgbd_frames: Optional[Dict[str, o3d.geometry.RGBDImage]] = None,
    cameras: Optional[Dict[str, RealsenseCamera]] = None,
    T_w_c_dict: Optional[Dict[str, np.ndarray]] = None,
    segment_prompts: Optional[Sequence[str]] = None,
    max_range_m: Optional[float] = None,
    sam3_device: Optional[str] = None,
    sam3_score_threshold: float = 0.0,
    sam3_top_k: int = 0,
) -> Path:
    """Save RGB images, merged workspace PCD, optional segmentation outputs, and metadata."""
    output_folder.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_folder / "rgb"
    rgb_dir.mkdir(exist_ok=True)

    saved_rgb = {}
    for serial, rgb in rgb_images.items():
        rgb_path = rgb_dir / f"{serial}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        saved_rgb[serial] = str(rgb_path.relative_to(output_folder))

    pcd_path = output_folder / "workspace.pcd"
    o3d.io.write_point_cloud(str(pcd_path), workspace_pcd)

    segmentation_metadata = None
    if segment_prompts:
        if rgbd_frames is None or cameras is None or T_w_c_dict is None:
            raise ValueError(
                "segment_prompts requires rgbd_frames, cameras, and T_w_c_dict"
            )
        from .segmentation import segment_workspace_capture

        print(f"\nSegmenting object(s): {list(segment_prompts)}")
        segmentation_metadata = segment_workspace_capture(
            rgb_images=rgb_images,
            rgbd_frames=rgbd_frames,
            cameras=cameras,
            T_w_c_dict=T_w_c_dict,
            prompts=segment_prompts,
            output_folder=output_folder,
            max_range_m=max_range_m,
            device=sam3_device,
            score_threshold=sam3_score_threshold,
            top_k=sam3_top_k,
        )

    metadata = {
        **metadata,
        "rgb_images": saved_rgb,
        "workspace_pcd": pcd_path.name,
        "num_points": len(workspace_pcd.points),
    }
    if segmentation_metadata is not None:
        metadata["segmentation"] = segmentation_metadata
    metadata_path = output_folder / "capture.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {len(saved_rgb)} RGB image(s) to: {rgb_dir}")
    print(f"Saved workspace point cloud to: {pcd_path} ({len(workspace_pcd.points)} points)")
    print(f"Capture metadata saved to: {metadata_path}")
    return output_folder


def capture_workspace(
    calibration_file: str,
    output_dir: str = "data",
    width: int = 1280,
    height: int = 720,
    hardware_sync: bool = False,
    master_serial: Optional[str] = None,
    max_timestamp_diff_ms: float = 33.0,
    pcd_voxel_size: float = 0.005,
    max_range_m: Optional[float] = 2.0,
    warmup_frames: int = 30,
    segment_prompts: Optional[Sequence[str]] = None,
    sam3_device: Optional[str] = None,
    sam3_score_threshold: float = 0.0,
    sam3_top_k: int = 0,
    robot_T_w_r_file: Optional[str] = None,
) -> Path:
    """
    Capture synchronized RGB images and a merged workspace point cloud.

    Uses extrinsics from ``calibration_file`` to transform each camera cloud
    into the shared world frame (ChArUco board frame from calibration).

    Args:
        calibration_file: Path to calibration.json with T_w_c per camera serial
        output_dir: Parent directory for timestamped workspace capture folder
        width: Camera color/depth width
        height: Camera color/depth height
        hardware_sync: Configure RealSense master/slave sync before capture
        master_serial: Master camera serial (defaults to first calibrated camera)
        max_timestamp_diff_ms: Max allowed timestamp spread for software alignment
        pcd_voxel_size: Voxel size for downsampling the merged workspace cloud
        max_range_m: Drop depth points beyond this range (meters) in each camera frame
        warmup_frames: Frames to discard before capture for auto-exposure settling
        segment_prompts: SAM3 text prompts for per-camera object segmentation
        sam3_device: SAM3 inference device ("cuda" or "cpu")
        sam3_score_threshold: Drop SAM3 instances below this score
        sam3_top_k: Keep only the top K SAM3 masks per prompt (0 = all)
        robot_T_w_r_file: Optional JSON file with ``T_w_r``. ``T_w_r`` maps
            robot-base coordinates into the ChArUco/world frame, so generated
            world-frame grasps can be converted with ``inv(T_w_r) @ T_w_g``.

    Returns:
        Path to the output folder containing RGB images, workspace.pcd, and
        optional per-camera segmentation outputs under segments/
    """
    calibration_path = Path(calibration_file)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    T_w_c_dict = load_calibration(str(calibration_path))
    serials = list(T_w_c_dict.keys())
    print(f"Loaded calibration for {len(serials)} camera(s): {serials}")

    robot_transform_metadata = None
    if robot_T_w_r_file is not None:
        robot_transform_path = Path(robot_T_w_r_file)
        if not robot_transform_path.is_file():
            raise FileNotFoundError(
                f"Robot base transform file not found: {robot_transform_path}"
            )
        T_w_r = load_transform_4x4(str(robot_transform_path), key="T_w_r")
        T_r_w = np.linalg.inv(T_w_r)
        robot_transform_metadata = {
            "source_file": str(robot_transform_path.resolve()),
            "frame_convention": (
                "T_w_r maps robot-base coordinates into the ChArUco/world frame; "
                "convert generated world-frame grasps with T_r_g = inv(T_w_r) @ T_w_g."
            ),
            "T_w_r": T_w_r.tolist(),
            "T_r_w": T_r_w.tolist(),
        }
        print(f"Loaded robot base transform: {robot_transform_path}")

    rs_context = None
    if hardware_sync and len(serials) > 1:
        rs_context = configure_hardware_sync(serials, master_serial=master_serial)

    init_order = _camera_init_order(serials, hardware_sync, master_serial)
    cameras: Dict[str, RealsenseCamera] = {}
    for serial in init_order:
        try:
            cameras[serial] = RealsenseCamera(
                serial_number=serial,
                width=width,
                height=height,
                rs_context=rs_context,
            )
            print(f"Camera {serial} initialized")
        except Exception as e:
            print(f"Failed to initialize camera {serial}: {e}")
            for camera in cameras.values():
                camera.stop()
            raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(output_dir) / f"workspace_{timestamp}"

    try:
        print(f"\nWarming up cameras ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            for camera in cameras.values():
                camera.get_frame()

        print("\nPress Enter to capture workspace RGB images and point cloud...")
        input()

        rgbd_frames, frame_timestamps = capture_synced_rgbd_frames(
            cameras,
            max_timestamp_diff_ms=max_timestamp_diff_ms,
        )

        workspace_pcd, rgb_images = merge_rgbd_frames_to_world(
            rgbd_frames,
            cameras,
            T_w_c_dict,
            voxel_size=pcd_voxel_size,
            max_range_m=max_range_m,
        )

        if len(rgb_images) != len(serials):
            missing = set(serials) - set(rgb_images.keys())
            raise RuntimeError(f"Missing RGB frames from camera(s): {missing}")

        if len(workspace_pcd.points) == 0:
            raise RuntimeError("Merged workspace point cloud is empty")

        timestamp_spread_ms = max(frame_timestamps.values()) - min(
            frame_timestamps.values()
        )
        metadata = {
            "calibration_file": str(calibration_path.resolve()),
            "camera_serials": serials,
            "frame_timestamps_ms": frame_timestamps,
            "timestamp_spread_ms": timestamp_spread_ms,
            "hardware_sync": hardware_sync,
            "master_serial": master_serial or (serials[0] if hardware_sync else None),
            "pcd_voxel_size": pcd_voxel_size,
            "max_range_m": max_range_m,
        }
        if robot_transform_metadata is not None:
            metadata["robot_base_transform"] = robot_transform_metadata

        return save_workspace_capture(
            output_folder,
            rgb_images,
            workspace_pcd,
            metadata,
            rgbd_frames=rgbd_frames,
            cameras=cameras,
            T_w_c_dict=T_w_c_dict,
            segment_prompts=segment_prompts,
            max_range_m=max_range_m,
            sam3_device=sam3_device,
            sam3_score_threshold=sam3_score_threshold,
            sam3_top_k=sam3_top_k,
        )

    finally:
        print("\nStopping cameras...")
        for camera in cameras.values():
            camera.stop()
