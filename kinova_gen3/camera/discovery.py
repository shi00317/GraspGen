"""RealSense device discovery."""

from typing import List

import pyrealsense2 as rs


def get_connected_cameras() -> List[str]:
    """Return serial numbers for all connected RealSense cameras."""
    ctx = rs.context()
    return [
        dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()
    ]
