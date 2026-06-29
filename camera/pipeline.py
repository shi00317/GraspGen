"""End-to-end extrinsic calibration pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2

from .discovery import get_connected_cameras
from .icp import (
    capture_camera_point_clouds,
    print_extrinsic_delta,
    refine_extrinsics_with_icp,
)
from .multi_calibrate import calibrate_all_cameras_simultaneously
from .realsense import RealsenseCamera
from .visualization import record_calibration_rerun_video


def run_calibration(
    output_dir: str = "data",
    num_samples: int = 10,
    width: int = 1280,
    height: int = 720,
    squares_x: int = 5,
    squares_y: int = 7,
    square_length: float = 0.034,
    marker_length: float = 0.026,
    dict_id: int = cv2.aruco.DICT_6X6_250,
    visualize_pcd: bool = True,
    refine_with_icp: bool = True,
    icp_voxel_sizes: Tuple[float, ...] = (0.02, 0.01, 0.005),
    icp_max_correspondence_distance: float = 0.02,
    video_duration_sec: float = 5.0,
    video_fps: int = 10,
    pcd_voxel_size: float = 0.005,
    max_points_per_frame: int = 80000,
):
    """
    Run fixed-board extrinsic calibration for all connected cameras.

    Args:
        output_dir: Directory to save calibration results
        num_samples: Number of observations to collect per camera
        width: Camera width resolution in pixels
        height: Camera height resolution in pixels
        squares_x: Number of chessboard squares in X direction
        squares_y: Number of chessboard squares in Y direction
        square_length: Length of chessboard square in meters
        marker_length: Length of ArUco marker in meters
        dict_id: ArUco dictionary identifier
        visualize_pcd: Record multi-camera point cloud video as Rerun .rrd
        refine_with_icp: Refine ChArUco extrinsics with multi-camera point cloud ICP
        icp_voxel_sizes: Voxel sizes for coarse-to-fine ICP
        icp_max_correspondence_distance: Max ICP correspondence distance in meters
        video_duration_sec: Duration of the Rerun visualization recording
        video_fps: Capture rate for the Rerun visualization recording
        pcd_voxel_size: Voxel size for downsampling logged point clouds
        max_points_per_frame: Max points logged to Rerun per frame
    """
    # Create timestamped output folder under parent directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(output_dir) / f"calibration_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = str(output_folder / "calibration.json")

    # Detect cameras
    print("Detecting connected RealSense cameras...")
    serial_numbers = get_connected_cameras()

    if not serial_numbers:
        print("Error: No RealSense cameras detected")
        return

    print(f"Found {len(serial_numbers)} camera(s): {serial_numbers}")

    # Initialize cameras
    cameras = []
    for sn in serial_numbers:
        try:
            cam = RealsenseCamera(serial_number=sn, width=width, height=height)
            cameras.append(cam)
            print(f"Camera {sn} initialized")
        except Exception as e:
            print(f"Failed to initialize camera {sn}: {e}")
            for c in cameras:
                c.stop()
            return

    calibration_results = {}
    charuco_results = {}
    icp_metrics = {}

    try:
        # Calibrate all cameras simultaneously
        results = calibrate_all_cameras_simultaneously(
            cameras,
            num_samples,
            squares_x,
            squares_y,
            square_length,
            marker_length,
            dict_id,
        )

        charuco_results = {serial: T_w_c.copy() for serial, T_w_c in results.items()}

        if refine_with_icp and len(results) >= 2:
            print("\nCapturing point clouds for ICP refinement...")
            camera_dict = {cam.serial_number: cam for cam in cameras}
            pcds_cam = capture_camera_point_clouds(camera_dict)

            if len(pcds_cam) >= 2:
                refined, icp_metrics = refine_extrinsics_with_icp(
                    pcds_cam,
                    results,
                    voxel_sizes=icp_voxel_sizes,
                    max_correspondence_distance=icp_max_correspondence_distance,
                )

                print("\nICP correction vs ChArUco:")
                for serial in refined:
                    if serial in charuco_results:
                        print_extrinsic_delta(
                            serial, charuco_results[serial], refined[serial]
                        )

                results = refined
            else:
                print("ICP refinement skipped: could not capture enough point clouds")

        # Convert numpy arrays to lists for JSON serialization
        calibration_results = {
            serial: T_w_c.tolist() for serial, T_w_c in results.items()
        }

    finally:
        # Cleanup
        for cam in cameras:
            cam.stop()
        print("\nAll cameras stopped")

    # Save results
    if calibration_results:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(calibration_results, f, indent=4)

        if icp_metrics:
            icp_path = output_folder / "icp_refinement.json"
            with open(icp_path, "w") as f:
                json.dump(icp_metrics, f, indent=4)
            print(f"ICP metrics saved to: {icp_path}")

        print(f"\n{'='*60}")
        print("Calibration complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"Calibrated cameras: {list(calibration_results.keys())}")
        print("\nThe calibration transforms points from camera frame to world frame,")
        print("where world origin is at the ChArUco board corner.")

        # Record Rerun visualization video if enabled
        if visualize_pcd and calibration_results:
            record_calibration_rerun_video(
                calibration_results,
                output_folder=output_folder,
                width=width,
                height=height,
                refine_with_icp=False,
                video_duration_sec=video_duration_sec,
                video_fps=video_fps,
                pcd_voxel_size=pcd_voxel_size,
                max_points_per_frame=max_points_per_frame,
            )

    else:
        print("\nNo cameras were successfully calibrated")
