"""ChArUco board detection for camera extrinsic calibration."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .realsense import RealsenseCamera


class CameraCalibrator:
    """Detect a ChArUco board and estimate camera-to-board pose."""

    def __init__(
        self,
        camera: RealsenseCamera,
        squares_x: int = 5,
        squares_y: int = 7,
        square_length: float = 0.04,
        marker_length: float = 0.02,
        dict_id: int = cv2.aruco.DICT_6X6_250,
    ):
        self.camera = camera
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.aruco_dict,
        )
        self.detector = cv2.aruco.CharucoDetector(self.charuco_board)

    def detect_board(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect ChArUco corners in a BGR image."""
        charuco_corners, charuco_ids, _, _ = self.detector.detectBoard(image)

        if charuco_corners is None or len(charuco_corners) < 4:
            return None

        return charuco_corners, charuco_ids
