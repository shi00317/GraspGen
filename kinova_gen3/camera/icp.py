"""Point cloud ICP refinement for extrinsic calibration."""

from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .realsense import RealsenseCamera


def capture_camera_point_clouds(
    cameras: Dict[str, RealsenseCamera],
) -> Dict[str, o3d.geometry.PointCloud]:
    """Capture one depth-colored point cloud per camera in the camera frame."""
    pcds_cam: Dict[str, o3d.geometry.PointCloud] = {}

    for serial, camera in cameras.items():
        rgbd = camera.get_frame()
        if rgbd is None:
            print(f"Failed to capture from camera {serial}")
            continue

        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            camera.intrinsics,
        )
        pcds_cam[serial] = pcd_cam
        print(f"  Camera {serial}: {len(pcd_cam.points)} points")

    return pcds_cam


def preprocess_pcd_for_icp(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Downsample and denoise a point cloud for ICP registration."""
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    if voxel_size > 0:
        pcd_clean = pcd_clean.voxel_down_sample(voxel_size)
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    return pcd_clean


def run_multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_sizes: Tuple[float, ...],
    max_correspondence_distance: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """Run coarse-to-fine point-to-plane ICP."""
    current_transform = init_transform.copy()
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = current_transform
    result.fitness = 0.0
    result.inlier_rmse = 0.0

    for voxel_size in voxel_sizes:
        source_down = preprocess_pcd_for_icp(source, voxel_size)
        target_down = preprocess_pcd_for_icp(target, voxel_size)

        if len(source_down.points) == 0 or len(target_down.points) == 0:
            continue

        max_dist = max(max_correspondence_distance, voxel_size * 2.0)
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_dist,
            current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        current_transform = result.transformation

    return result


def refine_extrinsics_with_icp(
    pcds_cam: Dict[str, o3d.geometry.PointCloud],
    T_w_c_dict: Dict[str, np.ndarray],
    ref_serial: Optional[str] = None,
    voxel_sizes: Tuple[float, ...] = (0.02, 0.01, 0.005),
    max_correspondence_distance: float = 0.02,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Refine camera extrinsics by aligning each camera's point cloud to a reference.

    The reference camera extrinsic is kept fixed. For every other camera, multi-scale
    ICP finds a world-frame correction that best overlaps its cloud with the
    reference cloud.

    Args:
        pcds_cam: Point clouds in each camera's optical frame
        T_w_c_dict: Initial world-to-camera transforms from ChArUco calibration
        ref_serial: Reference camera serial (defaults to first entry)
        voxel_sizes: Voxel sizes for coarse-to-fine ICP
        max_correspondence_distance: Maximum pairing distance in meters

    Returns:
        Refined T_w_c transforms and per-camera ICP metrics
    """
    serials = [s for s in T_w_c_dict if s in pcds_cam]
    if len(serials) < 2:
        print("ICP refinement skipped: need at least 2 cameras with point clouds")
        return T_w_c_dict, {}

    if ref_serial is None:
        ref_serial = serials[0]
    elif ref_serial not in serials:
        raise ValueError(f"Reference camera {ref_serial} has no point cloud")

    print(f"\n{'='*60}")
    print(f"ICP refinement (reference camera: {ref_serial})")
    print(f"{'='*60}")

    pcd_ref_world = o3d.geometry.PointCloud(pcds_cam[ref_serial])
    pcd_ref_world.transform(T_w_c_dict[ref_serial])
    refined: Dict[str, np.ndarray] = {ref_serial: T_w_c_dict[ref_serial].copy()}
    metrics: Dict[str, dict] = {
        ref_serial: {"role": "reference", "fitness": 1.0, "rmse": 0.0}
    }

    for serial in serials:
        if serial == ref_serial:
            continue

        pcd_src_world = o3d.geometry.PointCloud(pcds_cam[serial])
        pcd_src_world.transform(T_w_c_dict[serial])
        result = run_multiscale_icp(
            pcd_src_world,
            pcd_ref_world,
            np.eye(4),
            voxel_sizes,
            max_correspondence_distance,
        )

        T_delta = result.transformation
        T_w_c_refined = T_delta @ T_w_c_dict[serial]
        refined[serial] = T_w_c_refined

        delta_rot = R.from_matrix(T_delta[:3, :3])
        delta_trans = np.linalg.norm(T_delta[:3, 3])
        delta_angle = np.linalg.norm(delta_rot.as_rotvec()) * 180 / np.pi

        metrics[serial] = {
            "role": "refined",
            "fitness": float(result.fitness),
            "rmse": float(result.inlier_rmse),
            "delta_translation_m": float(delta_trans),
            "delta_rotation_deg": float(delta_angle),
            "T_delta": T_delta.tolist(),
        }

        print(
            f"Camera {serial}: fitness={result.fitness:.4f}, "
            f"rmse={result.inlier_rmse*1000:.2f}mm, "
            f"delta=({delta_trans*1000:.2f}mm, {delta_angle:.3f}deg)"
        )

    return refined, metrics


def print_extrinsic_delta(
    serial: str, T_before: np.ndarray, T_after: np.ndarray
) -> None:
    """Print translation/rotation change between two extrinsic estimates."""
    delta = T_after @ np.linalg.inv(T_before)
    delta_trans = np.linalg.norm(delta[:3, 3])
    delta_angle = (
        np.linalg.norm(R.from_matrix(delta[:3, :3]).as_rotvec()) * 180 / np.pi
    )
    print(
        f"  Camera {serial}: correction "
        f"({delta_trans*1000:.2f}mm, {delta_angle:.3f}deg)"
    )
