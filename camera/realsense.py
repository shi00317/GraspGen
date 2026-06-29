"""RealSense camera interface."""

from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def _format_device_busy_error(error: RuntimeError, serial_number: str) -> RuntimeError:
    """Return a clearer error when a RealSense device is already in use."""
    message = str(error).lower()
    if "errno=16" not in message and "resource busy" not in message and "busy" not in message:
        return error

    return RuntimeError(
        f"RealSense camera {serial_number} is already in use.\n"
        "Close RealSense Viewer and stop other camera scripts first:\n"
        "  pgrep -af 'capture_workspace|extrinsic_calibrate'\n"
        "If a previous capture is waiting at 'Press Enter', finish or Ctrl+C it."
    )


class RealsenseCamera:
    """Interface for Realsense camera operations."""

    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        rs_context: Optional[rs.context] = None,
    ):
        if serial_number is None:
            raise ValueError("Serial number must be provided for Realsense camera.")

        self.pipeline = rs.pipeline(rs_context) if rs_context is not None else rs.pipeline()
        self.config = rs.config()

        self.serial_number = serial_number
        self.fps = fps
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            self.pipeline.start(self.config)
        except RuntimeError as e:
            raise _format_device_busy_error(e, serial_number) from e

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        profile = self.pipeline.get_active_profile()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(
            width,
            height,
            color_intrinsics.fx,
            color_intrinsics.fy,
            color_intrinsics.ppx,
            color_intrinsics.ppy,
        )

        self.dist_coeffs = np.array(color_intrinsics.coeffs)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.inv_depth_scale = 1.0 / self.depth_scale

    def get_frame(self) -> Optional[o3d.geometry.RGBDImage]:
        """Capture and return the current frame as an Open3D RGBD image."""
        result = self.get_frame_with_timestamp()
        return result[0] if result is not None else None

    def get_frame_with_timestamp(
        self,
    ) -> Optional[Tuple[o3d.geometry.RGBDImage, float]]:
        """Capture a frame and return the RGBD image plus depth timestamp (ms)."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            depth_array = np.asanyarray(depth_frame.get_data())
            color_array = np.asanyarray(color_frame.get_data())
            color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)

            color_o3d = o3d.geometry.Image(color_array)
            depth_o3d = o3d.geometry.Image(depth_array)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=self.inv_depth_scale,
                convert_rgb_to_intensity=False,
            )
            return rgbd, float(depth_frame.get_timestamp())
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def stop(self) -> None:
        """Stop the camera stream."""
        self.pipeline.stop()
