#!/usr/bin/env python3

"""
Recording demonstrations in admittance mode for deploy.py
This module provides functionality to record robot demonstrations when 
admittance mode (Cartesian or Joint) is active.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Optional
from kinova_gen3.robot.utilities import DeviceConnection

if sys.version_info.major == 3 and sys.version_info.minor >= 10:   
    import collections
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
    setattr(collections,"MutableSequence", collections.abc.MutableSequence)

# Add path to Kortex API examples
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api_python/examples"))

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, ControlConfig_pb2

def euler_to_rotation_matrix(theta_x, theta_y, theta_z):
    """
    Convert Euler angles (in degrees) to rotation matrix.
    
    Args:
        theta_x: Rotation around X axis in degrees
        theta_y: Rotation around Y axis in degrees
        theta_z: Rotation around Z axis in degrees
        
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    rx = np.radians(theta_x)
    ry = np.radians(theta_y)
    rz = np.radians(theta_z)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def pose_to_transformation_matrix(pose):
    """
    Convert robot pose (x, y, z, theta_x, theta_y, theta_z) to 4x4 transformation matrix.
    
    Args:
        pose: BaseFeedback object with tool_pose_x, tool_pose_y, tool_pose_z (meters) 
              and tool_pose_theta_x, tool_pose_theta_y, tool_pose_theta_z (degrees)
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    
    # Translation
    T[0, 3] = pose.tool_pose_x
    T[1, 3] = pose.tool_pose_y
    T[2, 3] = pose.tool_pose_z
    
    # Rotation
    T[:3, :3] = euler_to_rotation_matrix(pose.tool_pose_theta_x, pose.tool_pose_theta_y, pose.tool_pose_theta_z)
    
    return T


def is_admittance_mode_active(control_config: ControlConfigClient) -> bool:
    """
    Check if the robot is currently in admittance mode (Cartesian, Joint, or Null Space).
    
    Args:
        control_config: ControlConfigClient instance
        
    Returns:
        True if in any admittance mode, False otherwise
    """
    try:
        control_mode_info = control_config.GetControlMode()
        
        admittance_modes = [
            ControlConfig_pb2.CARTESIAN_ADMITTANCE,
            ControlConfig_pb2.JOINT_ADMITTANCE,
            ControlConfig_pb2.NULL_SPACE_ADMITTANCE
        ]
        
        is_admittance = control_mode_info.control_mode in admittance_modes
        
        if is_admittance:
            mode_names = {
                ControlConfig_pb2.CARTESIAN_ADMITTANCE: "Cartesian Admittance",
                ControlConfig_pb2.JOINT_ADMITTANCE: "Joint Admittance",
                ControlConfig_pb2.NULL_SPACE_ADMITTANCE: "Null Space Admittance"
            }
            print(f"Admittance mode active: {mode_names[control_mode_info.control_mode]}")
        
        return is_admittance
        
    except Exception as e:
        print(f"Error checking control mode: {e}")
        return False


def get_gripper_state(base: BaseClient) -> float:
    """
    Get the current gripper state (position).
    
    Args:
        base: BaseClient instance
        
    Returns:
        Gripper position [0.0, 1.0] where 0 is open and 1 is closed.
        Returns 0.0 if no gripper is detected or error occurs.
    """
    try:
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
        
        if len(gripper_measure.finger) > 0:
            # Return position normalized to [0, 1]
            return gripper_measure.finger[0].value
        else:
            return 0.0
            
    except Exception as e:
        print(f"Warning: Could not get gripper state: {e}")
        return 0.0


def record_demonstration_step(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    control_config: ControlConfigClient
) -> Optional[Dict]:
    """
    Record a single step of robot state when in admittance mode.
    
    Args:
        base: BaseClient instance
        base_cyclic: BaseCyclicClient instance for real-time feedback
        control_config: ControlConfigClient instance
        
    Returns:
        Dictionary with 'T_w_e' (4x4 matrix), 'grip' (float), and optionally 'pcd' (Nx3 array).
        Returns None if not in admittance mode or error occurs.
    """
    # Check if in admittance mode
    if not is_admittance_mode_active(control_config):
        return None
    
    try:
        # Get real-time feedback
        feedback = base_cyclic.RefreshFeedback()
        
        # Extract end-effector pose
        pose = feedback.base
        
        # Convert to transformation matrix
        T_w_e = pose_to_transformation_matrix(pose)
        
        # Get gripper state
        grip = get_gripper_state(base)
        
        # Prepare step data
        step_data = {
            'T_w_e': T_w_e,
            'grip': grip,
            'pcd': None  # Point cloud data would be captured separately if vision sensor is available
        }
        
        return step_data
        
    except Exception as e:
        print(f"Error recording demonstration step: {e}")
        return None


def record_demonstration(
    base: BaseClient,
    base_cyclic: BaseCyclicClient,
    control_config: ControlConfigClient,
    duration: float = 30.0,
    frequency: float = 10.0
) -> Dict[str, List]:
    """
    Record a full demonstration in admittance mode.
    
    This function continuously records robot state while the robot is in admittance mode.
    Recording automatically starts when admittance mode is detected and stops when
    the mode changes or duration expires.
    
    Args:
        base: BaseClient instance
        base_cyclic: BaseCyclicClient instance
        control_config: ControlConfigClient instance
        duration: Maximum recording duration in seconds (default: 30.0)
        frequency: Recording frequency in Hz (default: 10.0)
        
    Returns:
        Dictionary with keys:
            - 'pcds': List of point clouds (currently None placeholders)
            - 'T_w_es': List of 4x4 transformation matrices
            - 'grips': List of gripper states [0.0, 1.0]
    """
    demo_data = {
        'pcds': [],
        'T_w_es': [],
        'grips': []
    }
    
    sample_time = 1.0 / frequency
    start_time = time.time()
    recording_started = False
    
    print(f"\nWaiting for admittance mode to be activated...")
    print(f"Will record for up to {duration} seconds at {frequency} Hz")
    print("Press Ctrl+C to stop recording early\n")
    
    try:
        while (time.time() - start_time) < duration:
            # Record step
            step_data = record_demonstration_step(base, base_cyclic, control_config)
            
            if step_data is not None:
                if not recording_started:
                    print("Recording started!")
                    recording_started = True
                    start_time = time.time()  # Reset start time when recording actually begins
                
                # Append data
                demo_data['pcds'].append(step_data['pcd'])
                demo_data['T_w_es'].append(step_data['T_w_e'])
                demo_data['grips'].append(step_data['grip'])
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"\rRecorded {len(demo_data['T_w_es'])} samples in {elapsed:.1f}s", end='')
                
            elif recording_started:
                # Admittance mode was active but now stopped
                print("\n\nAdmittance mode deactivated. Recording stopped.")
                break
            
            # Sleep to maintain frequency
            time.sleep(sample_time)
            
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
    
    if len(demo_data['T_w_es']) > 0:
        print(f"\n\nRecording complete!")
        print(f"Total samples: {len(demo_data['T_w_es'])}")
        print(f"Duration: {(time.time() - start_time):.2f} seconds")
    else:
        print("\n\nNo data recorded. Make sure admittance mode was activated.")
    
    return demo_data


def record_multiple_demonstrations(
    args,
    num_demos: int = 2,
    duration_per_demo: float = 30.0,
    frequency: float = 10.0
) -> List[Dict[str, List]]:
    """
    Record multiple demonstrations in sequence.
    
    Args:
        args: Connection arguments (from utilities.parseConnectionArguments)
        num_demos: Number of demonstrations to record
        duration_per_demo: Maximum duration for each demo in seconds
        frequency: Recording frequency in Hz
        
    Returns:
        List of demonstration dictionaries, each with 'pcds', 'T_w_es', and 'grips' keys
    """
    all_demos = []
    
    print(f"\n{'='*60}")
    print(f"Recording {num_demos} demonstrations")
    print(f"{'='*60}\n")
    
    with DeviceConnection.createTcpConnection(args) as router:
        # Create required services
        base = BaseClient(router)
        control_config = ControlConfigClient(router)
        
        with DeviceConnection.createUdpConnection(args) as router_realtime:
            base_cyclic = BaseCyclicClient(router_realtime)
            
            for i in range(num_demos):
                print(f"\n{'='*60}")
                print(f"Demonstration {i+1}/{num_demos}")
                print(f"{'='*60}")
                
                demo = record_demonstration(
                    base,
                    base_cyclic,
                    control_config,
                    duration=duration_per_demo,
                    frequency=frequency
                )
                
                all_demos.append(demo)
                
                if i < num_demos - 1:
                    print("\n\nPrepare for next demonstration...")
                    print("Press Enter when ready...")
                    input()
    
    return all_demos


def main():
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Record robot demonstrations in admittance mode")
    parser.add_argument("--num_demos", type=int, default=1, help="Number of demonstrations to record")
    parser.add_argument("--duration", type=float, default=30.0, help="Maximum duration per demo (seconds)")
    parser.add_argument("--frequency", type=float, default=10.0, help="Recording frequency (Hz)")
    args = DeviceConnection.parseConnectionArguments(parser)
    
    # Record demonstrations
    demos = record_multiple_demonstrations(
        args,
        num_demos=args.num_demos,
        duration_per_demo=args.duration,
        frequency=args.frequency
    )
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("Recording Summary")
    print(f"{'='*60}")
    # print(demos)
    for i, demo in enumerate(demos):
        print(f"Demo {i+1}: {len(demo['T_w_es'])} samples")
    
    # Optionally save to file
    # import pickle
    # with open('demonstrations.pkl', 'wb') as f:
    #     pickle.dump(demos, f)
    # print("\nDemonstrations saved to 'demonstrations.pkl'")
    
    return demos


if __name__ == "__main__":
    main()
