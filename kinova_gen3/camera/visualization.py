"""Rerun visualization for calibration results."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .icp import (
    capture_camera_point_clouds,
    refine_extrinsics_with_icp,
)
from .realsense import RealsenseCamera


ROBOT_T_W_R = np.array(
    [
        [-1, 0, 0, 0.578],
        [0, 1, 0, 0],
        [0, 0, -1, -0.15],
        [0, 0, 0, 1],
    ]
)

CAMERA_FRUSTUM_COLORS = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
]


def _set_rerun_frame_time(frame_idx: int) -> None:
    """Set Rerun timeline in a way compatible with different SDK versions."""
    import rerun as rr

    try:
        rr.set_time("frame", sequence=int(frame_idx))
    except (TypeError, AttributeError):
        try:
            rr.set_time_sequence("frame", int(frame_idx))
        except AttributeError:
            pass


def _camera_frustum_strips(
    T_w_c: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    scale: float = 0.2,
) -> List[np.ndarray]:
    """Build camera frustum line strips in world coordinates."""
    w, h = intrinsics.width, intrinsics.height
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    corners_cam = np.array(
        [
            [0, 0, 0],
            [(0 - cx) / fx * scale, (0 - cy) / fy * scale, scale],
            [(w - cx) / fx * scale, (0 - cy) / fy * scale, scale],
            [(w - cx) / fx * scale, (h - cy) / fy * scale, scale],
            [(0 - cx) / fx * scale, (h - cy) / fy * scale, scale],
        ]
    )
    corners_world = (T_w_c @ np.hstack([corners_cam, np.ones((5, 1))]).T).T[:, :3]

    return [
        corners_world[[0, 1]],
        corners_world[[0, 2]],
        corners_world[[0, 3]],
        corners_world[[0, 4]],
        corners_world[[1, 2, 3, 4, 1]],
    ]


def _axis_strips(
    origin: np.ndarray, rotation: np.ndarray, length: float
) -> Dict[str, np.ndarray]:
    """Build RGB axis line strips from an origin and rotation matrix."""
    return {
        "x": np.array([origin, origin + rotation[:, 0] * length]),
        "y": np.array([origin, origin + rotation[:, 1] * length]),
        "z": np.array([origin, origin + rotation[:, 2] * length]),
    }


def log_static_calibration_scene(
    T_w_c_dict: Dict[str, np.ndarray],
    cameras: Dict[str, RealsenseCamera],
    robot_T_w_r: np.ndarray = ROBOT_T_W_R,
) -> None:
    """Log static coordinate frames and camera frustums to Rerun."""
    import rerun as rr

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    world_axes = _axis_strips(np.zeros(3), np.eye(3), 0.15)
    for axis_name, color in zip(
        ("x", "y", "z"),
        ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]),
    ):
        rr.log(
            f"world/frames/board/{axis_name}",
            rr.LineStrips3D([world_axes[axis_name]], colors=[color], radii=0.004),
            static=True,
        )

    robot_T_r_w = np.linalg.inv(robot_T_w_r)
    robot_origin = robot_T_r_w[:3, 3]
    robot_rot = robot_T_r_w[:3, :3]
    robot_axes = _axis_strips(robot_origin, robot_rot, 0.1)
    for axis_name, color in zip(
        ("x", "y", "z"),
        ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]),
    ):
        rr.log(
            f"world/frames/robot/{axis_name}",
            rr.LineStrips3D([robot_axes[axis_name]], colors=[color], radii=0.003),
            static=True,
        )

    for idx, (serial, T_w_c) in enumerate(T_w_c_dict.items()):
        if serial not in cameras:
            continue

        color = CAMERA_FRUSTUM_COLORS[idx % len(CAMERA_FRUSTUM_COLORS)]
        quat_xyzw = R.from_matrix(T_w_c[:3, :3]).as_quat()
        rr.log(
            f"world/cameras/{serial}",
            rr.Transform3D(
                translation=T_w_c[:3, 3],
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            ),
            static=True,
        )

        cam_axes = _axis_strips(T_w_c[:3, 3], T_w_c[:3, :3], 0.1)
        for axis_name, axis_color in zip(
            ("x", "y", "z"),
            ([255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]),
        ):
            rr.log(
                f"world/cameras/{serial}/axes/{axis_name}",
                rr.LineStrips3D([cam_axes[axis_name]], colors=[axis_color], radii=0.002),
                static=True,
            )

        frustum_strips = _camera_frustum_strips(T_w_c, cameras[serial].intrinsics)
        rr.log(
            f"world/cameras/{serial}/frustum",
            rr.LineStrips3D(frustum_strips, colors=[color] * len(frustum_strips), radii=0.002),
            static=True,
        )


def _filter_pcd_by_range(
    pcd: o3d.geometry.PointCloud, max_range_m: float
) -> o3d.geometry.PointCloud:
    """Drop points farther than max_range_m from the cloud origin (camera frame)."""
    if max_range_m <= 0 or len(pcd.points) == 0:
        return pcd

    points = np.asarray(pcd.points)
    keep = np.linalg.norm(points, axis=1) <= max_range_m
    if not np.any(keep):
        return o3d.geometry.PointCloud()

    return pcd.select_by_index(np.flatnonzero(keep))


def merge_rgbd_frames_to_world(
    rgbd_frames: Dict[str, o3d.geometry.RGBDImage],
    cameras: Dict[str, RealsenseCamera],
    T_w_c_dict: Dict[str, np.ndarray],
    voxel_size: float = 0.005,
    max_range_m: Optional[float] = 2.0,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
    """Merge per-camera RGB-D frames into one world-frame point cloud."""
    combined_pcd = o3d.geometry.PointCloud()
    images: Dict[str, np.ndarray] = {}

    for serial, rgbd in rgbd_frames.items():
        if serial not in T_w_c_dict or serial not in cameras:
            continue

        images[serial] = np.asarray(rgbd.color)
        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            cameras[serial].intrinsics,
        )
        if max_range_m is not None and max_range_m > 0:
            pcd_cam = _filter_pcd_by_range(pcd_cam, max_range_m)
        pcd_world = o3d.geometry.PointCloud(pcd_cam)
        pcd_world.transform(T_w_c_dict[serial])
        if voxel_size > 0:
            pcd_world = pcd_world.voxel_down_sample(voxel_size)
        combined_pcd += pcd_world

    return combined_pcd, images


def capture_merged_pointcloud_frame(
    cameras: Dict[str, RealsenseCamera],
    T_w_c_dict: Dict[str, np.ndarray],
    voxel_size: float = 0.005,
    max_range_m: Optional[float] = 2.0,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, np.ndarray]]:
    """Capture and merge one multi-camera point cloud frame plus RGB images."""
    rgbd_frames: Dict[str, o3d.geometry.RGBDImage] = {}

    for serial, camera in cameras.items():
        if serial not in T_w_c_dict:
            continue

        rgbd = camera.get_frame()
        if rgbd is None:
            continue

        rgbd_frames[serial] = rgbd

    return merge_rgbd_frames_to_world(
        rgbd_frames, cameras, T_w_c_dict, voxel_size=voxel_size, max_range_m=max_range_m
    )


def pointcloud_to_rerun(
    pcd: o3d.geometry.PointCloud, max_points: int = 80000
):
    """Convert an Open3D point cloud to a Rerun Points3D archetype."""
    import rerun as rr

    points = np.asarray(pcd.points)
    if len(points) == 0:
        return rr.Points3D([], radii=0.003)

    colors = np.asarray(pcd.colors)
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.7)
    colors_rgba = np.clip(colors, 0.0, 1.0)
    colors_rgba = (colors_rgba * 255).astype(np.uint8)
    if colors_rgba.shape[1] == 3:
        colors_rgba = np.hstack(
            [colors_rgba, np.full((len(colors_rgba), 1), 255, dtype=np.uint8)]
        )

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors_rgba = colors_rgba[idx]

    return rr.Points3D(points, colors=colors_rgba, radii=0.003)


def record_calibration_rerun_video(
    calibration_data: dict,
    output_folder: Path,
    width: int = 1280,
    height: int = 720,
    refine_with_icp: bool = False,
    icp_voxel_sizes: Tuple[float, ...] = (0.02, 0.01, 0.005),
    icp_max_correspondence_distance: float = 0.02,
    video_duration_sec: float = 5.0,
    video_fps: int = 10,
    pcd_voxel_size: float = 0.005,
    max_points_per_frame: int = 80000,
) -> None:
    """
    Record a multi-camera calibration visualization video and save as Rerun .rrd.

    Captures live RGB-D frames, merges them into a world-frame point cloud per
    timeline step, and logs camera images plus static scene geometry to Rerun.
    """
    try:
        import rerun as rr
    except ImportError:
        print("Rerun not available. Install with: pip install rerun-sdk")
        return

    T_w_c_dict = {
        serial: (
            np.array(transform) if not isinstance(transform, np.ndarray) else transform
        )
        for serial, transform in calibration_data.items()
    }

    print(f"Loaded calibration for {len(T_w_c_dict)} camera(s)")

    cameras: Dict[str, RealsenseCamera] = {}
    for serial in T_w_c_dict.keys():
        try:
            cameras[serial] = RealsenseCamera(
                serial_number=serial, width=width, height=height
            )
            print(f"Camera {serial} initialized")
        except Exception as e:
            print(f"Failed to initialize camera {serial}: {e}")

    if not cameras:
        print("Error: No cameras could be initialized")
        return

    num_frames = max(1, int(video_duration_sec * video_fps))
    frame_interval = 1.0 / video_fps
    rrd_path = output_folder / "calibration_video.rrd"

    print("\n" + "=" * 60)
    print(f"Press Enter to record {num_frames} frames at {video_fps} FPS")
    print(f"Rerun recording will be saved to: {rrd_path}")
    print("=" * 60)

    try:
        input("\nPress Enter to start recording...")

        if refine_with_icp and len(cameras) >= 2:
            print("\nCapturing point clouds for ICP refinement...")
            pcds_cam = capture_camera_point_clouds(cameras)
            if len(pcds_cam) >= 2:
                refined, icp_metrics = refine_extrinsics_with_icp(
                    pcds_cam,
                    T_w_c_dict,
                    voxel_sizes=icp_voxel_sizes,
                    max_correspondence_distance=icp_max_correspondence_distance,
                )
                T_w_c_dict = refined

                calibration_path = output_folder / "calibration.json"
                with open(calibration_path, "w") as f:
                    json.dump(
                        {s: T.tolist() for s, T in T_w_c_dict.items()},
                        f,
                        indent=4,
                    )
                icp_path = output_folder / "icp_refinement.json"
                with open(icp_path, "w") as f:
                    json.dump(icp_metrics, f, indent=4)
                print(f"Updated calibration saved to: {calibration_path}")

        rr.init("calibration_video", spawn=False)
        rr.save(str(rrd_path))
        log_static_calibration_scene(T_w_c_dict, cameras)

        print(f"\nRecording {num_frames} frames...")
        last_combined_pcd = o3d.geometry.PointCloud()

        for frame_idx in range(num_frames):
            loop_start = time.time()
            _set_rerun_frame_time(frame_idx)

            combined_pcd, images = capture_merged_pointcloud_frame(
                cameras,
                T_w_c_dict,
                voxel_size=pcd_voxel_size,
            )

            if len(combined_pcd.points) > 0:
                last_combined_pcd = combined_pcd
                rr.log(
                    "world/scene/points",
                    pointcloud_to_rerun(combined_pcd, max_points=max_points_per_frame),
                )

            for serial, rgb in images.items():
                rr.log(f"cameras/{serial}", rr.Image(rgb))

            if frame_idx % max(1, num_frames // 10) == 0 or frame_idx == num_frames - 1:
                print(
                    f"  Frame {frame_idx + 1}/{num_frames}: "
                    f"{len(combined_pcd.points)} points"
                )

            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0 and frame_idx < num_frames - 1:
                time.sleep(sleep_time)

        if len(last_combined_pcd.points) > 0:
            ply_path = output_folder / "pointcloud.ply"
            o3d.io.write_point_cloud(str(ply_path), last_combined_pcd)
            print(f"Last-frame point cloud saved to: {ply_path}")

        print(f"\nRerun recording saved to: {rrd_path}")
        print(f"View with: rerun {rrd_path}")

    finally:
        print("\nStopping cameras...")
        for camera in cameras.values():
            camera.stop()
        print("Done")
