"""Multi-camera ChArUco extrinsic calibration."""

import time
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .calibrate import CameraCalibrator
from .realsense import RealsenseCamera


def calibrate_all_cameras_simultaneously(
    cameras: List[RealsenseCamera],
    num_samples: int = 10,
    squares_x: int = 5,
    squares_y: int = 7,
    square_length: float = 0.034,
    marker_length: float = 0.026,
    dict_id: int = cv2.aruco.DICT_6X6_250,
) -> dict:
    """
    Calibrate all cameras simultaneously with fixed board on desk.

    All camera feeds are displayed at once, and observations are captured
    from all cameras simultaneously with a single key press.

    Args:
        cameras: List of initialized RealSense cameras
        num_samples: Number of observations to collect for averaging
        squares_x: Number of chessboard squares in X direction
        squares_y: Number of chessboard squares in Y direction
        square_length: Length of chessboard square in meters
        marker_length: Length of ArUco marker in meters
        dict_id: ArUco dictionary identifier

    Returns:
        Dictionary mapping serial numbers to 4x4 world-to-camera transforms (T_w_c)
    """
    print(f"\n{'='*60}")
    print(f"Calibrating {len(cameras)} cameras simultaneously")
    print(f"{'='*60}")
    print(f"Will collect {num_samples} observations of the fixed board")
    print("Keep the board stationary on the desk")
    print("Press Enter to START automatic capture from ALL cameras, 'q' to abort")

    # Create calibrator for each camera
    camera_data = {}
    for camera in cameras:
        calibrator = CameraCalibrator(
            camera,
            squares_x=squares_x,
            squares_y=squares_y,
            square_length=square_length,
            marker_length=marker_length,
            dict_id=dict_id,
        )

        intrinsics = camera.intrinsics
        print(f"Camera {camera.serial_number} intrinsics: {intrinsics.intrinsic_matrix}")
        camera_matrix = np.array(
            [
                [
                    intrinsics.intrinsic_matrix[0, 0],
                    0,
                    intrinsics.intrinsic_matrix[0, 2],
                ],
                [
                    0,
                    intrinsics.intrinsic_matrix[1, 1],
                    intrinsics.intrinsic_matrix[1, 2],
                ],
                [0, 0, 1],
            ]
        )

        camera_data[camera.serial_number] = {
            "camera": camera,
            "calibrator": calibrator,
            "camera_matrix": camera_matrix,
            "dist_coeffs": camera.dist_coeffs,
            "T_c_b_samples": [],
            "sample_count": 0,
        }

        print(f"Camera {camera.serial_number} distortion coeffs: {camera.dist_coeffs}")

    sample_count = 0
    capturing = False

    while sample_count < num_samples:
        # Get frames and detections from all cameras
        all_detections = {}
        display_images = []

        for serial, data in camera_data.items():
            rgbd = data["camera"].get_frame()
            if rgbd is None:
                continue

            # Convert to BGR for OpenCV
            color_array = np.asarray(rgbd.color)
            image_bgr = cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)

            # Detect board
            detection = data["calibrator"].detect_board(image_bgr)

            # Display
            display_img = image_bgr.copy()
            if detection is not None:
                corners, ids = detection
                cv2.aruco.drawDetectedCornersCharuco(display_img, corners, ids)

                # Warn if too few corners detected
                num_corners = len(corners)
                color = (
                    (0, 255, 0) if num_corners >= 20 else (0, 165, 255)
                )  # Green if good, orange if low
                status_text = (
                    f"Detected {num_corners} corners - Ready"
                    if num_corners >= 20
                    else f"Detected {num_corners} corners - Warning: Low"
                )
                cv2.putText(
                    display_img,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    display_img,
                    "No board detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                display_img,
                f"Samples: {sample_count}/{num_samples}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            msg = "Capturing..." if capturing else "Press Enter to Start"
            cv2.putText(
                display_img,
                msg,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if capturing else (0, 0, 255),
                2,
            )

            cv2.putText(
                display_img,
                f"Camera: {serial}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            all_detections[serial] = (detection, image_bgr)
            display_images.append(display_img)

        # Combine all camera feeds into a single window
        if display_images:
            num_cameras = len(display_images)

            if num_cameras == 1:
                combined = display_images[0]
            elif num_cameras == 2:
                # Stack horizontally for 2 cameras
                combined = np.hstack(display_images)
            elif num_cameras <= 4:
                # 2x2 grid for 3-4 cameras
                if num_cameras == 3:
                    # Add a black placeholder for the 4th position
                    h, w = display_images[0].shape[:2]
                    display_images.append(np.zeros((h, w, 3), dtype=np.uint8))
                top_row = np.hstack(display_images[:2])
                bottom_row = np.hstack(display_images[2:4])
                combined = np.vstack([top_row, bottom_row])
            else:
                # Grid layout for more cameras (2 cameras per row)
                rows = []
                for i in range(0, num_cameras, 2):
                    if i + 1 < num_cameras:
                        row = np.hstack([display_images[i], display_images[i + 1]])
                    else:
                        # Odd number of cameras, add black placeholder
                        h, w = display_images[i].shape[:2]
                        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                        row = np.hstack([display_images[i], placeholder])
                    rows.append(row)
                combined = np.vstack(rows)

            # Scale to fit screen (use 90% of screen size to leave margin)
            h, w = combined.shape[:2]

            # Try to get actual screen resolution
            try:
                import tkinter as tk

                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                # Use 90% of screen dimensions to leave margin for window decorations
                max_width = int(screen_width * 0.9)
                max_height = int(screen_height * 0.9)
            except Exception:
                # Fallback to common resolution if tkinter fails
                max_width = 1600
                max_height = 900

            scale_h = max_height / h if h > max_height else 1.0
            scale_w = max_width / w if w > max_width else 1.0
            scale = min(scale_h, scale_w)

            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                combined = cv2.resize(
                    combined, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

            cv2.imshow("Multi-Camera Calibration", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nCalibration aborted")
            cv2.destroyAllWindows()
            return {}

        if key == 13 and not capturing:
            capturing = True
            print("\nStarting automatic capture...")

        if capturing:
            # Check if all cameras detected the board
            cameras_with_detection = [
                serial
                for serial, (detection, _) in all_detections.items()
                if detection is not None
            ]

            if not cameras_with_detection:
                print("No cameras detected the board. Skipping frame.")
                continue

            if len(cameras_with_detection) < len(cameras):
                missing = set(camera_data.keys()) - set(cameras_with_detection)
                print(
                    f"Warning: Cameras {missing} did not detect board. Capturing from {cameras_with_detection} only."
                )

            # Capture from all cameras that detected the board
            for serial in cameras_with_detection:
                detection, image_bgr = all_detections[serial]
                corners, ids = detection
                data = camera_data[serial]

                # Compute board pose (camera to board)
                obj_points = data["calibrator"].charuco_board.getChessboardCorners()[
                    ids.flatten()
                ]

                # Use SOLVEPNP_IPPE which is more robust for planar targets (like ChArUco)
                # It handles the planar ambiguity better than ITERATIVE
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners,
                    data["camera_matrix"],
                    data["dist_coeffs"],
                    flags=cv2.SOLVEPNP_IPPE,
                )

                if not success:
                    print(f"Failed to compute board pose for camera {serial}")
                    continue

                # Skipped solvePnPRefineLM as IPPE is usually accurate enough for planar
                # and LM can sometimes cause drift if intrinsics aren't perfect
                # rvec, tvec = cv2.solvePnPRefineLM(...)

                # Build T_c_b (transforms points from board frame to camera frame)
                rmat, _ = cv2.Rodrigues(rvec)
                T_c_b = np.eye(4)
                T_c_b[:3, :3] = rmat
                T_c_b[:3, 3] = tvec.flatten()

                # Compute reprojection error for this sample
                proj_corners, _ = cv2.projectPoints(
                    obj_points, rvec, tvec, data["camera_matrix"], data["dist_coeffs"]
                )
                reproj_error = np.sqrt(np.mean((corners - proj_corners.squeeze()) ** 2))

                data["T_c_b_samples"].append(T_c_b)

                print(
                    f"Camera {serial}: distance={np.linalg.norm(tvec):.3f}m, "
                    f"reproj_error={reproj_error:.2f}px, corners={len(corners)}"
                )

            sample_count += 1
            print(f"Captured sample {sample_count}/{num_samples}")

            # Small delay to avoid identical samples
            time.sleep(0.1)

    cv2.destroyAllWindows()

    # Compute final transforms for each camera
    results = {}

    for serial, data in camera_data.items():
        T_c_b_samples = data["T_c_b_samples"]

        if len(T_c_b_samples) < 3:
            print(
                f"\nCamera {serial}: Only {len(T_c_b_samples)} samples. Need at least 3. Skipping."
            )
            continue

        # Compute T_w_c for each observation: T_w_c = inv(T_c_b)
        # T_c_b transforms from board frame to camera frame
        # T_w_c transforms from camera frame to world frame (board is world)
        T_w_c_samples = [np.linalg.inv(T_c_b) for T_c_b in T_c_b_samples]

        # Average translations (simple mean is correct for translations)
        avg_translation = np.mean([T[:3, 3] for T in T_w_c_samples], axis=0)

        # Average rotations properly using quaternions (not naive matrix averaging)
        # Convert rotation matrices to quaternions
        rotations = [R.from_matrix(T[:3, :3]) for T in T_w_c_samples]

        # Average quaternions using the method from Markley et al. 2007
        # "Averaging Quaternions"
        quaternions = np.array([rot.as_quat() for rot in rotations])  # [x, y, z, w]

        # Ensure all quaternions are in the same hemisphere
        # (q and -q represent the same rotation)
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]

        # Simple mean works when quaternions are close (which they should be)
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)  # Normalize

        # Convert back to rotation matrix
        avg_rotation = R.from_quat(avg_quat).as_matrix()

        T_w_c_final = np.eye(4)
        T_w_c_final[:3, :3] = avg_rotation
        T_w_c_final[:3, 3] = avg_translation

        # Compute calibration consistency metrics
        distances = [np.linalg.norm(T[:3, 3] - avg_translation) for T in T_w_c_samples]
        mean_pos_deviation = np.mean(distances)
        max_pos_deviation = np.max(distances)

        # Compute rotation deviations (angular difference from mean)
        avg_rot = R.from_matrix(avg_rotation)
        rotation_errors = []
        for T in T_w_c_samples:
            rot = R.from_matrix(T[:3, :3])
            # Relative rotation between sample and average
            rel_rot = avg_rot.inv() * rot
            # Convert to angle in degrees
            angle = np.linalg.norm(rel_rot.as_rotvec()) * 180 / np.pi
            rotation_errors.append(angle)

        mean_rot_deviation = np.mean(rotation_errors)
        max_rot_deviation = np.max(rotation_errors)

        print(f"\n{'='*60}")
        print(f"Camera {serial} calibration successful!")
        print(f"{'='*60}")
        print("T_w_c (camera-to-world transform):")
        print(T_w_c_final)
        print("\nCalibration quality:")
        print(f"  Samples collected: {len(T_c_b_samples)}")
        print(f"  Mean position deviation: {mean_pos_deviation*1000:.2f} mm")
        print(f"  Max position deviation: {max_pos_deviation*1000:.2f} mm")
        print(f"  Mean rotation deviation: {mean_rot_deviation:.3f} degrees")
        print(f"  Max rotation deviation: {max_rot_deviation:.3f} degrees")

        results[serial] = T_w_c_final

    if results:
        print(f"\n{'='*60}")
        print("All cameras calibrated!")
        print(f"{'='*60}")
        print(
            "Note: World frame origin is at board corner (see ChArUco board orientation)"
        )

    return results
