"""SAM3 object segmentation and per-pixel workspace point export."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    import torch
    from PIL import Image


PIXEL_POINTS_FIELDNAMES = [
    "image_topic",
    "cloud_topic",
    "u",
    "v",
    "cloud_point_index",
    "x",
    "y",
    "z",
    "r",
    "g",
    "b",
]


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_")
    return name or "object"


def default_output_dir(image_path: Path, prompt: str) -> Path:
    return image_path.parent / "segments" / f"{image_path.stem}_{safe_name(prompt)}"


def tensor_to_numpy(value) -> np.ndarray:
    import torch

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def normalize_masks(masks, image_size: Tuple[int, int]) -> np.ndarray:
    mask_array = tensor_to_numpy(masks)
    mask_array = np.squeeze(mask_array)

    if mask_array.size == 0:
        return np.zeros((0, image_size[1], image_size[0]), dtype=bool)
    if mask_array.ndim == 2:
        mask_array = mask_array[np.newaxis, :, :]
    if mask_array.ndim != 3:
        raise RuntimeError(f"Unexpected SAM3 mask shape: {mask_array.shape}")

    return mask_array > 0


def select_masks(
    masks: np.ndarray,
    boxes,
    scores,
    score_threshold: float,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_array = tensor_to_numpy(scores).reshape(-1)
    box_array = (
        tensor_to_numpy(boxes).reshape((-1, 4)) if len(score_array) else np.zeros((0, 4))
    )

    keep = np.where(score_array >= score_threshold)[0]
    keep = keep[np.argsort(score_array[keep])[::-1]]
    if top_k > 0:
        keep = keep[:top_k]

    return masks[keep], box_array[keep], score_array[keep]


def camera_topics(cam_index: int) -> Tuple[str, str]:
    image_topic = f"/sync/cam_{cam_index}/rgb/image_rect_color"
    cloud_topic = f"/sync/cam_{cam_index}/depth_registered/points"
    return image_topic, cloud_topic


def build_pixel_points(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    T_w_c: np.ndarray,
    cam_index: int,
    max_range_m: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Build one row per valid depth pixel with world-frame coordinates."""
    if rgb.shape[:2] != depth_m.shape:
        raise RuntimeError(
            f"RGB shape {rgb.shape[:2]} does not match depth shape {depth_m.shape}"
        )

    height, width = depth_m.shape
    image_topic, cloud_topic = camera_topics(cam_index)
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    rotation = T_w_c[:3, :3]
    translation = T_w_c[:3, 3]

    rows: List[Dict[str, object]] = []
    for v in range(height):
        for u in range(width):
            z = float(depth_m[v, u])
            if not np.isfinite(z) or z <= 0:
                continue

            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            if max_range_m is not None and max_range_m > 0:
                if np.linalg.norm((x_cam, y_cam, z)) > max_range_m:
                    continue

            point_cam = np.array([x_cam, y_cam, z], dtype=np.float64)
            point_world = rotation @ point_cam + translation
            red, green, blue = rgb[v, u]

            rows.append(
                {
                    "image_topic": image_topic,
                    "cloud_topic": cloud_topic,
                    "u": u,
                    "v": v,
                    "cloud_point_index": v * width + u,
                    "x": f"{point_world[0]:.9g}",
                    "y": f"{point_world[1]:.9g}",
                    "z": f"{point_world[2]:.9g}",
                    "r": int(red),
                    "g": int(green),
                    "b": int(blue),
                }
            )

    return rows


def export_object_points_per_mask(
    pixel_points: List[Dict[str, object]],
    output_dir: Path,
    masks: np.ndarray,
) -> List[Tuple[Path, int]]:
    if len(masks) == 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    exports: List[Tuple[Path, int]] = []
    for index, mask in enumerate(masks, start=1):
        object_points_csv = output_dir / f"object_points_{index:02d}.csv"
        rows_written = 0
        with object_points_csv.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=PIXEL_POINTS_FIELDNAMES)
            writer.writeheader()
            for row in pixel_points:
                u = int(row["u"])
                v = int(row["v"])
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                    writer.writerow(row)
                    rows_written += 1
        exports.append((object_points_csv, rows_written))

    return exports


def save_masked_depth_maps(
    depth_m: np.ndarray,
    masks: np.ndarray,
    output_dir: Path,
) -> List[Path]:
    """Save per-instance depth maps (uint16 millimeters, zero outside mask)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    for index, mask in enumerate(masks, start=1):
        masked_depth = np.where(mask, depth_mm, 0).astype(np.uint16)
        depth_path = output_dir / f"depth_{index:02d}.png"
        cv2.imwrite(str(depth_path), masked_depth)
        saved.append(depth_path)

    return saved


def save_mask_images(masks: np.ndarray, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for index, mask in enumerate(masks, start=1):
        mask_path = output_dir / f"mask_{index:02d}.png"
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
        saved.append(mask_path)
    return saved


class Sam3Segmenter:
    """Lazy-loaded SAM3 text-prompt segmenter."""

    def __init__(self, device: Optional[str] = None):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return
        import torch
        from PIL import Image  # noqa: F401
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(device=self.device)
        self._processor = Sam3Processor(model, device=self.device)

    def segment(
        self,
        image: "Image.Image",
        prompt: str,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        self._ensure_loaded()
        processor = self._processor
        assert processor is not None

        autocast_enabled = self.device == "cuda"
        with torch.autocast(
            device_type=self.device, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = normalize_masks(output["masks"], image.size)
        return select_masks(
            masks, output["boxes"], output["scores"], score_threshold, top_k
        )


def segment_camera_view(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    T_w_c: np.ndarray,
    serial: str,
    cam_index: int,
    prompt: str,
    output_dir: Path,
    segmenter: Sam3Segmenter,
    max_range_m: Optional[float] = None,
    score_threshold: float = 0.0,
    top_k: int = 0,
    save_masks: bool = True,
) -> Dict[str, object]:
    """Segment one camera view and export object point CSVs plus depth maps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_points = build_pixel_points(
        rgb,
        depth_m,
        intrinsics,
        T_w_c,
        cam_index,
        max_range_m=max_range_m,
    )

    from PIL import Image

    image = Image.fromarray(rgb)
    masks, boxes, scores = segmenter.segment(
        image,
        prompt,
        score_threshold=score_threshold,
        top_k=top_k,
    )

    object_exports = export_object_points_per_mask(pixel_points, output_dir, masks)
    depth_exports = save_masked_depth_maps(depth_m, masks, output_dir)
    mask_exports = save_mask_images(masks, output_dir) if save_masks else []

    return {
        "serial": serial,
        "prompt": prompt,
        "pixel_point_rows": len(pixel_points),
        "num_instances": len(masks),
        "scores": [float(score) for score in scores],
        "boxes": boxes.tolist(),
        "object_points": [
            {"path": str(path), "rows": rows} for path, rows in object_exports
        ],
        "depth_maps": [str(path) for path in depth_exports],
        "masks": [str(path) for path in mask_exports],
    }


def segment_workspace_capture(
    rgb_images: Dict[str, np.ndarray],
    rgbd_frames: Dict[str, o3d.geometry.RGBDImage],
    cameras: Dict[str, object],
    T_w_c_dict: Dict[str, np.ndarray],
    prompts: Sequence[str],
    output_folder: Path,
    max_range_m: Optional[float] = None,
    device: Optional[str] = None,
    score_threshold: float = 0.0,
    top_k: int = 0,
) -> Dict[str, object]:
    """Run SAM3 segmentation for each camera view and object prompt."""
    if not prompts:
        return {"enabled": False, "cameras": {}}

    segmenter = Sam3Segmenter(device=device)
    segments_root = output_folder / "segments"
    camera_results: Dict[str, Dict[str, object]] = {}
    serial_order = {serial: index + 1 for index, serial in enumerate(T_w_c_dict)}

    for serial, rgb in rgb_images.items():
        if serial not in rgbd_frames or serial not in cameras or serial not in T_w_c_dict:
            continue

        depth_m = np.asarray(rgbd_frames[serial].depth).astype(np.float64)
        camera = cameras[serial]
        prompt_results: Dict[str, object] = {}

        for prompt in prompts:
            prompt_dir = segments_root / serial / safe_name(prompt)
            result = segment_camera_view(
                rgb=rgb,
                depth_m=depth_m,
                intrinsics=camera.intrinsics,
                T_w_c=T_w_c_dict[serial],
                serial=serial,
                cam_index=serial_order[serial],
                prompt=prompt,
                output_dir=prompt_dir,
                segmenter=segmenter,
                max_range_m=max_range_m,
                score_threshold=score_threshold,
                top_k=top_k,
            )
            prompt_results[prompt] = result
            print(
                f"Camera {serial}, prompt {prompt!r}: "
                f"{result['num_instances']} instance(s), "
                f"{result['pixel_point_rows']} valid depth pixels"
            )
            for export in result["object_points"]:
                print(f"  saved {export['rows']} rows: {export['path']}")

        camera_results[serial] = prompt_results

    return {
        "enabled": True,
        "prompts": list(prompts),
        "device": segmenter.device,
        "score_threshold": score_threshold,
        "top_k": top_k,
        "cameras": camera_results,
    }
