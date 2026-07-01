#!/usr/bin/env python3
"""Compute gripper fingertip positions from Kortex TCP pose and finger opening.

The Kinova Kortex API exposes a single tool pose and scalar gripper opening,
not per-finger Cartesian coordinates. This module maps those readings into
left/right fingertip locations using the GraspGen gripper-frame convention
(+Z approach, fingers move along ±X).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kinova_gen3.utilities.convert_robotiq_2f140_to_2f85 import (
    ROBOTIQ_2F140_DEPTH_M,
    ROBOTIQ_2F85_MAX_APERTURE_M,
)

# Pad-center depth along +Z in the GraspGen gripper frame (from gripper YAML contact_points).
ROBOTIQ_2F85_FINGERTIP_DEPTH_M = 0.052324


@dataclass(frozen=True)
class GripperFingerGeometry:
    """Fingertip model parameters in the GraspGen gripper frame."""

    tip_depth_m: float
    max_aperture_m: float


GRIPPER_FINGER_GEOMETRIES: dict[str, GripperFingerGeometry] = {
    "robotiq_2f85": GripperFingerGeometry(
        tip_depth_m=ROBOTIQ_2F85_FINGERTIP_DEPTH_M,
        max_aperture_m=ROBOTIQ_2F85_MAX_APERTURE_M,
    ),
    "robotiq_2f140": GripperFingerGeometry(
        tip_depth_m=ROBOTIQ_2F140_DEPTH_M,
        max_aperture_m=0.12,
    ),
}


@dataclass
class GripperTipPositions:
    """Fingertip locations expressed in multiple frames."""

    left_gripper: np.ndarray
    right_gripper: np.ndarray
    left_robot: np.ndarray
    right_robot: np.ndarray
    left_world: np.ndarray
    right_world: np.ndarray
    grip_value: float
    T_r_e: np.ndarray
    T_w_e: np.ndarray


def resolve_gripper_geometry(gripper_name: str) -> GripperFingerGeometry:
    key = gripper_name.lower().replace("-", "_")
    aliases = {
        "robotiq_2f_85": "robotiq_2f85",
        "robotiq_2f85": "robotiq_2f85",
        "robotiq_2f_140": "robotiq_2f140",
        "robotiq_2f140": "robotiq_2f140",
    }
    resolved = aliases.get(key, key)
    if resolved not in GRIPPER_FINGER_GEOMETRIES:
        available = ", ".join(sorted(GRIPPER_FINGER_GEOMETRIES))
        raise ValueError(
            f"Unsupported gripper '{gripper_name}'. Available models: {available}"
        )
    return GRIPPER_FINGER_GEOMETRIES[resolved]


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to a 3D point."""
    homogeneous = np.asarray(point, dtype=np.float64).reshape(3)
    return (np.asarray(T, dtype=np.float64) @ np.append(homogeneous, 1.0))[:3]


def fingertips_in_gripper_frame(
    grip_value: float,
    *,
    gripper_name: str = "robotiq_2f85",
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right fingertip positions in the GraspGen gripper frame.

    Args:
        grip_value: Normalized opening in [0, 1] where 0 is fully open and
            1 is fully closed (Kinova ``GetMeasuredGripperMovement`` convention).
        gripper_name: Registered fingertip geometry name.
    """
    geometry = resolve_gripper_geometry(gripper_name)
    grip_value = float(np.clip(grip_value, 0.0, 1.0))
    half_aperture = (1.0 - grip_value) * (geometry.max_aperture_m / 2.0)
    left = np.array([half_aperture, 0.0, geometry.tip_depth_m], dtype=np.float64)
    right = np.array([-half_aperture, 0.0, geometry.tip_depth_m], dtype=np.float64)
    return left, right


def compute_fingertip_positions(
    T_r_e: np.ndarray,
    T_w_r: np.ndarray,
    grip_value: float,
    *,
    gripper_name: str = "robotiq_2f85",
    T_tool_gripper: Optional[np.ndarray] = None,
) -> GripperTipPositions:
    """Compute fingertip positions from a robot-base tool pose and gripper opening."""
    T_r_e = np.asarray(T_r_e, dtype=np.float64)
    T_w_r = np.asarray(T_w_r, dtype=np.float64)
    T_tool_gripper = (
        np.eye(4, dtype=np.float64)
        if T_tool_gripper is None
        else np.asarray(T_tool_gripper, dtype=np.float64)
    )
    T_r_g = T_r_e @ T_tool_gripper
    T_w_e = T_w_r @ T_r_e
    T_w_g = T_w_r @ T_r_g

    left_g, right_g = fingertips_in_gripper_frame(
        grip_value, gripper_name=gripper_name
    )
    return GripperTipPositions(
        left_gripper=left_g,
        right_gripper=right_g,
        left_robot=transform_point(T_r_g, left_g),
        right_robot=transform_point(T_r_g, right_g),
        left_world=transform_point(T_w_g, left_g),
        right_world=transform_point(T_w_g, right_g),
        grip_value=float(grip_value),
        T_r_e=T_r_e,
        T_w_e=T_w_e,
    )


def query_live_fingertip_positions(
    T_w_r: np.ndarray,
    *,
    base: Optional[BaseClient] = None,
    base_cyclic: Optional[BaseCyclicClient] = None,
    ip: str = "192.168.1.10",
    username: str = "admin",
    password: str = "admin",
    gripper_name: str = "robotiq_2f85",
    T_tool_gripper: Optional[np.ndarray] = None,
) -> GripperTipPositions:
    """Query the robot and return fingertip positions in the calibration world frame."""
    from kinova_gen3.robot.execute import get_current_pose
    from kinova_gen3.robot.recordDemo import get_gripper_state
    from kinova_gen3.robot.utilities import DeviceConnection

    if base is not None and base_cyclic is not None:
        T_r_e = get_current_pose(base_cyclic)
        grip_value = get_gripper_state(base)
        return compute_fingertip_positions(
            T_r_e,
            T_w_r,
            grip_value,
            gripper_name=gripper_name,
            T_tool_gripper=T_tool_gripper,
        )

    conn_args = argparse.Namespace(ip=ip, username=username, password=password)
    with DeviceConnection.createTcpConnection(conn_args) as tcp_router:
        with DeviceConnection.createUdpConnection(conn_args) as udp_router:
            live_base = base or BaseClient(tcp_router)
            live_cyclic = base_cyclic or BaseCyclicClient(udp_router)
            T_r_e = get_current_pose(live_cyclic)
            grip_value = get_gripper_state(live_base)
            return compute_fingertip_positions(
                T_r_e,
                T_w_r,
                grip_value,
                gripper_name=gripper_name,
                T_tool_gripper=T_tool_gripper,
            )


def format_tip_positions(tips: GripperTipPositions) -> str:
    """Human-readable summary of fingertip coordinates."""
    return (
        "Gripper fingertips (world frame, meters):\n"
        f"  left : [{tips.left_world[0]:.4f}, {tips.left_world[1]:.4f}, {tips.left_world[2]:.4f}]\n"
        f"  right: [{tips.right_world[0]:.4f}, {tips.right_world[1]:.4f}, {tips.right_world[2]:.4f}]\n"
        f"  grip opening value: {tips.grip_value:.3f} (0=open, 1=closed)"
    )
