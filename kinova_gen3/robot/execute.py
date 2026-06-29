#!/usr/bin/env python3

"""
Execute robot poses and gripper commands using the Kortex API.
This module provides functionality to command the robot to reach target poses
and control the gripper for deployment of learned policies.
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional

# Add path to Kortex API examples
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api_python/examples"))

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

from . import utilities


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (in degrees).
    Uses ZYX convention matching the robot's convention.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        (theta_x, theta_y, theta_z) in degrees
    """
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        theta_x = np.arctan2(R[2, 1], R[2, 2])
        theta_y = np.arctan2(-R[2, 0], sy)
        theta_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        theta_x = np.arctan2(-R[1, 2], R[1, 1])
        theta_y = np.arctan2(-R[2, 0], sy)
        theta_z = 0
    
    # Convert to degrees
    return np.degrees(theta_x), np.degrees(theta_y), np.degrees(theta_z)


def transformation_matrix_to_pose(T):
    """
    Convert 4x4 transformation matrix to pose (x, y, z, theta_x, theta_y, theta_z).
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Dictionary with keys: x, y, z (meters), theta_x, theta_y, theta_z (degrees)
    """
    # Extract translation
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    # Extract rotation matrix
    R = T[:3, :3]
    
    # Convert to Euler angles
    theta_x, theta_y, theta_z = rotation_matrix_to_euler(R)
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'theta_x': theta_x,
        'theta_y': theta_y,
        'theta_z': theta_z
    }


def get_current_pose(base_cyclic: BaseCyclicClient):
    """
    Get the current end-effector pose as a 4x4 transformation matrix.
    
    Args:
        base_cyclic: BaseCyclicClient instance
        
    Returns:
        4x4 transformation matrix representing current pose
    """
    from .recordDemo import pose_to_transformation_matrix
    
    feedback = base_cyclic.RefreshFeedback()
    pose = feedback.base
    T_w_e = pose_to_transformation_matrix(pose)
    
    return T_w_e


def move_to_pose(
    base: BaseClient,
    target_pose: dict,
    speed: float = 0.2,
    blend_radius: float = 0.0
) -> bool:
    """
    Command the robot to move to a target Cartesian pose.
    
    Args:
        base: BaseClient instance
        target_pose: Dictionary with x, y, z (m), theta_x, theta_y, theta_z (degrees)
        speed: Movement speed (0.0 to 1.0), default 0.2
        blend_radius: Blending radius in meters for smooth trajectories, default 0.0
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create action with reach_pose for single Cartesian movement
        action = Base_pb2.Action()
        action.name = "move_to_pose"
        action.application_data = ""
        
        # Set target pose
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pose['x']
        cartesian_pose.y = target_pose['y']
        cartesian_pose.z = target_pose['z']
        cartesian_pose.theta_x = target_pose['theta_x']
        cartesian_pose.theta_y = target_pose['theta_y']
        cartesian_pose.theta_z = target_pose['theta_z']
        
        # Set speed constraint
        action.reach_pose.constraint.speed.translation = speed
        action.reach_pose.constraint.speed.orientation = speed * 15.0  # degrees/s
        
        # Execute action
        base.ExecuteAction(action)
        
        # Wait for action to complete
        time.sleep(0.05)  # Brief wait for action to start
        
        return True
        
    except Exception as e:
        print(f"Error moving to pose: {e}")
        return False


def move_to_sequence_pose(
    base: BaseClient,
    target_poses: List[dict],
    speed: float = 0.2,
    blend_radius: float = 0.0,
    timeout: float = 20.0
) -> bool:
    """
    Command the robot to move through a sequence of target Cartesian poses.
    
    Args:
        base: BaseClient instance
        target_poses: List of dictionaries with x, y, z (m), theta_x, theta_y, theta_z (degrees)
        speed: (Not used in this implementation, kept for consistency)
        blend_radius: Default blending radius in meters if not specified in pose dict
        timeout: Timeout in seconds for the trajectory execution
        
    Returns:
        True if successful, False otherwise
    """
    import threading

    def check_for_end_or_abort(e):
        """
        Return a closure checking for ACTION_END or ACTION_ABORT.
        """
        def check(notification, e=e):
            print("EVENT : " + \
                  Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END or \
               notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    try:
        # Set servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)
        
        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False
        
        for i, pose in enumerate(target_poses):
            waypoint = waypoints.waypoints.add()
            waypoint.name = f"waypoint_{i}"
            
            cartesian_pose = waypoint.cartesian_waypoint
            cartesian_pose.pose.x = pose['x']
            cartesian_pose.pose.y = pose['y']
            cartesian_pose.pose.z = pose['z']
            cartesian_pose.pose.theta_x = pose['theta_x']
            cartesian_pose.pose.theta_y = pose['theta_y']
            cartesian_pose.pose.theta_z = pose['theta_z']
            
            # Use specific blending if available, else default
            cartesian_pose.blending_radius = pose.get('blending_radius', blend_radius)
            cartesian_pose.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            
        # Verify validity of waypoints
        result = base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) == 0:
            e = threading.Event()
            notification_handle = base.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )

            print("Moving cartesian trajectory...")
            base.ExecuteWaypointTrajectory(waypoints)

            print("Waiting for trajectory to finish ...")
            finished = e.wait(timeout)
            base.Unsubscribe(notification_handle)

            if finished:
                print("Cartesian trajectory completed")
                return True
            else:
                print("Timeout on action notification wait")
                return False
        else:
            print("Error found in trajectory") 
            result.trajectory_error_report.PrintDebugString()
            return False
            
    except Exception as e:
        print(f"Error executing sequence pose: {e}")
        return False




def move_to_pose_matrix(
    base: BaseClient,
    T_target: np.ndarray,
    speed: float = 0.2,
    blend_radius: float = 0.0
) -> bool:
    """
    Command the robot to move to a target pose specified as a transformation matrix.
    
    Args:
        base: BaseClient instance
        T_target: 4x4 transformation matrix
        speed: Movement speed (0.0 to 1.0)
        blend_radius: Blending radius in meters
        
    Returns:
        True if successful, False otherwise
    """
    pose_dict = transformation_matrix_to_pose(T_target)
    return move_to_pose(base, pose_dict, speed, blend_radius)


def set_gripper(base: BaseClient, gripper_value: float) -> bool:
    """
    Set gripper position.
    
    Args:
        base: BaseClient instance
        gripper_value: Gripper position [0.0, 1.0] where 0 is open and 1 is closed
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Clamp value to [0, 1]
        # gripper_value = np.clip(gripper_value, 0.0, 1.0)
        
        # Create gripper command
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = gripper_value
        
        # Send command
        base.SendGripperCommand(gripper_command)
        
        # Wait for gripper to reach position
        time.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"Error setting gripper: {e}")
        return False


def control_gripper_from_command(base: BaseClient, grip_command: float) -> bool:
    """
    Control gripper based on model output command.
    
    Args:
        base: BaseClient instance
        grip_command: -1 for close, 1 for open (as per model output format)
        
    Returns:
        True if successful, False otherwise
    """
    # if grip_command < 0:
    #     # Close gripper
    #     return set_gripper(base, 1.0)
    # else:
    #     # Open gripper
    return set_gripper(base, grip_command)


def execute_action_sequence(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    start_action: np.ndarray,
    raw_action: np.ndarray,
    grips: float,
    current_T_r_e: Optional[np.ndarray] = None,
    speed: float = 0.2,
    blend_radius: float = 0.01,
    execute_all: bool = False
) -> bool:
    """
    Execute a sequence of predicted actions.
    
    Args:
        base: BaseClient instance
        base_cyclic: BaseCyclicClient instance
        actions: [Pred_horizon, 4, 4] relative transformation matrices
        grips: [Pred_horizon, 1] gripper commands (-1 close, 1 open)
        current_T_w_e: Current end-effector pose, if None will be read from robot
        speed: Movement speed (0.0 to 1.0)
        blend_radius: Blending radius for smooth trajectories
        execute_all: If True, execute all predicted actions; if False, execute only first action
        
    Returns:
        True if successful, False otherwise
    """
    try:
        
        steps = 1
        for i in range(1, steps + 1):
            alpha = i / steps
            interp_action = start_action + (raw_action - start_action) * alpha
            
            keys = ['x', 'y', 'z', 'theta_x', 'theta_y', 'theta_z']
            pose_dict = dict(zip(keys, interp_action))
            print(f"Moving to ", f"[{pose_dict['x']:.3f}, {pose_dict['y']:.3f}, {pose_dict['z']:.3f}] m")

            # Execute movement
            success = move_to_pose(base, pose_dict, speed, blend_radius)
            if not success:
                print(f"Failed to execute action")
                return False
            
            # Execute gripper command
            set_gripper_success = control_gripper_from_command(base, grips)
            
            if not set_gripper_success:
                print(f"Warning: Failed to set gripper for action")
    
        return True
        
    except Exception as e:
        print(f"Error executing action sequence: {e}")
        return False


def load_robot_base_transform(path: Optional[str] = None) -> np.ndarray:
    """Load T_w_r (robot base expressed in the calibration world frame).

    Grasp poses from workspace capture are in the ChArUco board world frame.
    ``T_w_r`` maps points from the robot base frame into that world frame.
    """
    if path is None:
        from kinova_gen3.camera.visualization import ROBOT_T_W_R

        return np.array(ROBOT_T_W_R, dtype=np.float64)

    import json

    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "T_w_r" in data:
            data = data["T_w_r"]
        elif "transform" in data:
            data = data["transform"]
    return np.array(data, dtype=np.float64)


def convert_robotiq_2f140_grasps_to_2f85(
    grasps: np.ndarray,
    *,
    source_depth: Optional[float] = None,
    target_depth: Optional[float] = None,
) -> np.ndarray:
    """Shift 2F-140 GraspGen poses for a physical Robotiq 2F-85 gripper."""
    from kinova_gen3.utilities.convert_robotiq_2f140_to_2f85 import (
        ROBOTIQ_2F140_DEPTH_M,
        ROBOTIQ_2F85_DEPTH_M,
        convert_depth,
    )

    return convert_depth(
        grasps,
        source_depth=source_depth or ROBOTIQ_2F140_DEPTH_M,
        target_depth=target_depth or ROBOTIQ_2F85_DEPTH_M,
    )


def load_grasps_from_json(json_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load grasp poses and confidences from a pipeline JSON result file."""
    import json

    with open(json_path) as f:
        payload = json.load(f)
    grasps = np.array(payload["grasp_poses"], dtype=np.float64)
    confidences = np.array(payload["grasp_conf"], dtype=np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasp_poses shape (N, 4, 4), got {grasps.shape}")
    return grasps, confidences


def world_grasp_to_robot_pose(
    T_w_g: np.ndarray,
    T_w_r: np.ndarray,
) -> dict:
    """Convert a world-frame grasp pose to a Kinova base-frame pose dict."""
    T_r_w = np.linalg.inv(T_w_r)
    T_r_g = T_r_w @ T_w_g
    return transformation_matrix_to_pose(T_r_g)


def execute_world_grasp(
    base: BaseClient,
    T_w_g: np.ndarray,
    T_w_r: np.ndarray,
    *,
    pre_grasp_offset_m: float = 0.10,
    speed: float = 0.15,
    gripper_open: float = 0.0,
    gripper_close: float = 1.0,
    timeout: float = 30.0,
    dry_run: bool = False,
) -> bool:
    """Move to a generated grasp in the robot base frame and close the gripper.

    The grasp pose is defined in the calibration world frame (ChArUco board).
    A pre-grasp waypoint is placed ``pre_grasp_offset_m`` back along the
    gripper approach axis (+Z in the GraspGen convention).
    """
    T_r_w = np.linalg.inv(T_w_r)
    T_r_g = T_r_w @ T_w_g

    T_g_pre = np.eye(4)
    T_g_pre[2, 3] = -pre_grasp_offset_m
    T_r_pre = T_r_g @ T_g_pre

    pre_pose = transformation_matrix_to_pose(T_r_pre)
    grasp_pose = transformation_matrix_to_pose(T_r_g)

    print(
        f"Grasp in robot base frame: "
        f"pos=[{grasp_pose['x']:.3f}, {grasp_pose['y']:.3f}, {grasp_pose['z']:.3f}] m, "
        f"rpy=[{grasp_pose['theta_x']:.1f}, {grasp_pose['theta_y']:.1f}, "
        f"{grasp_pose['theta_z']:.1f}] deg"
    )

    if dry_run:
        print("Dry run: skipping robot motion.")
        return True

    print("Opening gripper...")
    if not set_gripper(base, gripper_open):
        return False

    print(
        f"Moving to pre-grasp ({pre_grasp_offset_m:.2f} m offset) "
        "then grasp pose..."
    )
    if not move_to_sequence_pose(
        base,
        [pre_pose, grasp_pose],
        speed=speed,
        timeout=timeout,
    ):
        return False

    print("Closing gripper...")
    return set_gripper(base, gripper_close)


def execute_single_action(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    action: np.ndarray,
    grip: float,
    current_T_w_e: Optional[np.ndarray] = None,
    speed: float = 0.2
) -> bool:
    """
    Execute a single action (useful for receding horizon control).
    
    Args:
        base: BaseClient instance
        base_cyclic: BaseCyclicClient instance
        action: [4, 4] relative transformation matrix
        grip: Gripper command (-1 close, 1 open)
        current_T_w_e: Current end-effector pose
        speed: Movement speed
        
    Returns:
        True if successful, False otherwise
    """
    actions = action[np.newaxis, :, :]  # Add batch dimension
    grips = np.array([[grip]])
    
    return execute_action_sequence(
        base, base_cyclic, actions, grips, 
        current_T_w_e, speed, 
        blend_radius=0.0, 
        execute_all=False
    )


def main():
    """
    Example usage of the execution functions.
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Execute robot poses and actions")
    parser.add_argument("--test", action="store_true", help="Run test movements")
    args = utilities.parseConnectionArguments(parser)
    
    # Connect to robot
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        
        with utilities.DeviceConnection.createUdpConnection(args) as router_realtime:
            base_cyclic = BaseCyclicClient(router_realtime)
            
            if args.test:
                print("\nTesting execution functions...")
                
                # Get current pose
                print("\n1. Getting current pose...")
                current_pose = get_current_pose(base_cyclic)
                print(f"Current pose:\n{current_pose}")
                
                # Test gripper
                print("\n2. Testing gripper...")
                print("Opening gripper...")
                set_gripper(base, 0.0)
                time.sleep(1)
                
                print("Closing gripper...")
                set_gripper(base, 1.0)
                time.sleep(1)
                
                print("Opening gripper...")
                set_gripper(base, 0.0)
                
                # Test small relative movement
                print("\n3. Testing relative movement...")
                delta_T = np.eye(4)
                delta_T[2, 3] = 0.05  # Move 5cm up in Z
                
                target_T = current_pose @ delta_T
                print("Moving 5cm up...")
                move_to_pose_matrix(base, target_T, speed=0.1)
                
                time.sleep(1)
                
                print("Returning to original position...")
                move_to_pose_matrix(base, current_pose, speed=0.1)
                
                print("\nTest complete!")
            else:
                print("Use --test flag to run test movements")
                print("Or import this module to use execution functions in your code")


if __name__ == "__main__":
    main()

