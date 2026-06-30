import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

# Explicit colors for GT / Prediction comparison.
# Colors are written in RGB order to follow the original visualization convention.
GT_COLOR = (0, 255, 0)       # Green
PRED_COLOR = (255, 0, 0)     # Red


def _to_numpy(array):
    """Convert torch.Tensor or array-like object to numpy."""
    if array is None:
        return None
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "cpu") and hasattr(array, "numpy"):
        return array.cpu().numpy()
    return np.asarray(array)


def _draw_camera_boxes(
    canvas: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes],
    labels: Optional[np.ndarray],
    transform: Optional[np.ndarray],
    classes: Optional[List[str]],
    color: Tuple[int, int, int],
    thickness: float,
) -> np.ndarray:
    """Draw 3D bounding boxes projected onto camera image."""
    if bboxes is None or len(bboxes) == 0:
        return canvas

    if transform is None:
        raise ValueError("Camera visualization requires lidar2image transform.")

    labels = _to_numpy(labels)
    corners = _to_numpy(bboxes.corners)
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )

    transform = copy.deepcopy(transform).reshape(4, 4)
    coords = coords @ transform.T
    coords = coords.reshape(-1, 8, 4)

    # Keep boxes whose all corners are in front of the camera.
    valid_indices = np.all(coords[..., 2] > 0, axis=1)
    coords = coords[valid_indices]
    if labels is not None:
        labels = labels[valid_indices]

    # Draw far boxes first.
    depth_order = np.argsort(-np.min(coords[..., 2], axis=1))
    coords = coords[depth_order]
    if labels is not None:
        labels = labels[depth_order]

    coords = coords.reshape(-1, 4)
    coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]
    coords = coords[..., :2].reshape(-1, 8, 2)

    edges = [
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    ]

    for index in range(coords.shape[0]):
        for start, end in edges:
            cv2.line(
                canvas,
                coords[index, start].astype(np.int32),
                coords[index, end].astype(np.int32),
                color,
                int(thickness),
                cv2.LINE_AA,
            )

    return canvas


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    # Original single-box API. Kept for backward compatibility.
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    # New GT / Prediction API.
    gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
    gt_labels: Optional[np.ndarray] = None,
    pred_bboxes: Optional[LiDARInstance3DBoxes] = None,
    pred_labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    gt_color: Tuple[int, int, int] = GT_COLOR,
    pred_color: Tuple[int, int, int] = PRED_COLOR,
    thickness: float = 4,
) -> None:
    """Visualize 3D boxes on camera image.

    Supports both the original API:
        visualize_camera(..., bboxes=..., labels=...)

    and the GT/Prediction comparison API:
        visualize_camera(..., gt_bboxes=..., pred_bboxes=...)
    """
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Backward-compatible path.
    if bboxes is not None:
        draw_color = color
        if draw_color is None:
            # Use the first label's object palette when possible.
            if labels is not None and classes is not None and len(labels) > 0:
                name = classes[int(_to_numpy(labels)[0])]
                draw_color = OBJECT_PALETTE.get(name, PRED_COLOR)
            else:
                draw_color = PRED_COLOR

        canvas = _draw_camera_boxes(
            canvas,
            bboxes=bboxes,
            labels=labels,
            transform=transform,
            classes=classes,
            color=draw_color,
            thickness=thickness,
        )

    # New GT / Prediction comparison path.
    canvas = _draw_camera_boxes(
        canvas,
        bboxes=gt_bboxes,
        labels=gt_labels,
        transform=transform,
        classes=classes,
        color=gt_color,
        thickness=thickness,
    )
    canvas = _draw_camera_boxes(
        canvas,
        bboxes=pred_bboxes,
        labels=pred_labels,
        transform=transform,
        classes=classes,
        color=pred_color,
        thickness=thickness,
    )

    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def _draw_lidar_boxes(
    *,
    bboxes: Optional[LiDARInstance3DBoxes],
    labels: Optional[np.ndarray],
    classes: Optional[List[str]],
    color: Tuple[int, int, int],
    thickness: float,
) -> None:
    """Draw 3D bounding boxes on LiDAR BEV plot."""
    if bboxes is None or len(bboxes) == 0:
        return

    corners = _to_numpy(bboxes.corners)
    coords = corners[:, [0, 3, 7, 4, 0], :2]

    for index in range(coords.shape[0]):
        plt.plot(
            coords[index, :, 0],
            coords[index, :, 1],
            linewidth=thickness,
            color=np.array(color) / 255,
        )


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    # Original single-box API. Kept for backward compatibility.
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    # New GT / Prediction API.
    gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
    gt_labels: Optional[np.ndarray] = None,
    pred_bboxes: Optional[LiDARInstance3DBoxes] = None,
    pred_labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    gt_color: Tuple[int, int, int] = GT_COLOR,
    pred_color: Tuple[int, int, int] = PRED_COLOR,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    """Visualize LiDAR BEV with optional GT / Prediction comparison."""
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    # Backward-compatible path.
    if bboxes is not None:
        draw_color = color
        if draw_color is None:
            if labels is not None and classes is not None and len(labels) > 0:
                name = classes[int(_to_numpy(labels)[0])]
                draw_color = OBJECT_PALETTE.get(name, PRED_COLOR)
            else:
                draw_color = PRED_COLOR

        _draw_lidar_boxes(
            bboxes=bboxes,
            labels=labels,
            classes=classes,
            color=draw_color,
            thickness=thickness,
        )

    # New GT / Prediction comparison path.
    _draw_lidar_boxes(
        bboxes=gt_bboxes,
        labels=gt_labels,
        classes=classes,
        color=gt_color,
        thickness=thickness,
    )
    _draw_lidar_boxes(
        bboxes=pred_bboxes,
        labels=pred_labels,
        classes=classes,
        color=pred_color,
        thickness=thickness,
    )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool_, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
