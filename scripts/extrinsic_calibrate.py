#!/usr/bin/env python3
"""CLI entry point for multi-camera extrinsic calibration."""

from pathlib import Path
from typing import Tuple

import cv2
from hydra_zen import make_config, store, zen

from kinova_gen3.camera.pipeline import run_calibration

Config = make_config(
    output_dir="data",
    num_samples=10,
    width=1280,
    height=720,
    squares_x=5,
    squares_y=7,
    square_length=0.034,
    marker_length=0.026,
    dict_id=cv2.aruco.DICT_6X6_250,
    visualize_pcd=True,
    refine_with_icp=True,
    icp_voxel_sizes=(0.02, 0.01, 0.005),
    icp_max_correspondence_distance=0.02,
    video_duration_sec=5.0,
    video_fps=10,
    pcd_voxel_size=0.005,
    max_points_per_frame=80000,
)

store(Config, name="base")


def task(
    output_dir: str,
    num_samples: int,
    width: int,
    height: int,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_id: int,
    visualize_pcd: bool,
    refine_with_icp: bool,
    icp_voxel_sizes: Tuple[float, ...],
    icp_max_correspondence_distance: float,
    video_duration_sec: float,
    video_fps: int,
    pcd_voxel_size: float,
    max_points_per_frame: int,
):
    """Run calibration with Hydra-managed configuration."""
    run_calibration(
        output_dir=str(Path.cwd() / output_dir),
        num_samples=num_samples,
        width=width,
        height=height,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_id=dict_id,
        visualize_pcd=visualize_pcd,
        refine_with_icp=refine_with_icp,
        icp_voxel_sizes=icp_voxel_sizes,
        icp_max_correspondence_distance=icp_max_correspondence_distance,
        video_duration_sec=video_duration_sec,
        video_fps=video_fps,
        pcd_voxel_size=pcd_voxel_size,
        max_points_per_frame=max_points_per_frame,
    )


def main() -> None:
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    store.add_to_hydra_store()
    zen(task).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )


if __name__ == "__main__":
    main()
