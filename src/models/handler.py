"""Handle all the model interaction."""

import shutil
from pathlib import Path
from typing import Any, TypeAlias

import cv2 as cv
import numpy as np
from loguru import logger
from ultralytics.models import YOLO

from src.models.embedding import EmbeddingDict, match_boxes_embedding

# =============================================================================
# Type Aliases
# =============================================================================

Points: TypeAlias = list[tuple[int, int]]
BoxWithConf: TypeAlias = tuple[Points, float]
BoxesPerCategory: TypeAlias = dict[str, list[BoxWithConf]]

# =============================================================================
# Model Loading & Saving
# =============================================================================


def is_pytorch_model(model_name: str | Path) -> bool:
    """Check if the model is a PyTorch model based on file extension."""
    model_path = str(model_name)
    return model_path.endswith(".pt") or model_path.endswith(".pth")


def _prepare_save_directory(save_path: Path) -> None:
    """Create save directory if it doesn't exist."""
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    elif not save_path.is_dir():
        logger.error(f"Save path {save_path} exists but is not a directory")
        raise ValueError(f"Save path {save_path} exists but is not a directory")


def _move_model_to_destination(model_path: Path, dest_path: Path) -> None:
    """Move model file to destination, handling existing files."""
    if dest_path.exists():
        if dest_path.is_dir():
            shutil.rmtree(dest_path)
        else:
            dest_path.unlink()

    if not dest_path.parent.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(model_path), str(dest_path))
    logger.info(f"Model moved to: {dest_path}")


def save_model(save_dir: str | Path, model_name: str | Path) -> None:
    """Save model to the specified directory."""
    try:
        save_path = Path(save_dir)
        _prepare_save_directory(save_path)

        model_path = Path(model_name)
        if not model_path.exists():
            return

        dest_path = save_path / model_path.name
        needs_move = (
            not dest_path.exists() or dest_path.resolve() != model_path.resolve()
        )

        if needs_move:
            _move_model_to_destination(model_path, dest_path)

    except Exception as e:
        logger.error(f"Failed to save model to {save_dir}: {e}")
        raise


def _log_model_info(
    model_name: str | Path,
    task: str | None,
    device: str | None,
    save_dir: str | Path | None,
) -> None:
    """Log model configuration information."""
    device_info = device if device else "Neural Engine (CoreML)"
    save_status = "✅" if save_dir else "❌"

    logger.info("Starting YOLO object detection with:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Task: {task}")
    logger.info(f"  Device: {device_info}")
    logger.info(f"  Save directory: {save_dir} - {save_status}")


def get_model(
    model_name: str | Path = "yolo26n.pt",
    *,
    task: str | None = None,
    verbose: bool = True,
    device: str = "mps",
    save_dir: str | Path | None = None,
) -> tuple[YOLO, str | None]:
    """Load a YOLO model and configure it for the specified device.

    Args:
        model_name: Path to the YOLO model file (.pt, .mlpackage, .onnx).
        task: Task type for the model (e.g., 'detect', 'segment', 'obb').
        verbose: Whether to print verbose output during model loading.
        device: Device for inference ('cpu', 'cuda', 'mps').
        save_dir: Directory to save/copy the model. If None, model stays in place.

    Returns:
        Tuple of (loaded YOLO model, device for inference or None for CoreML).
    """
    model = YOLO(model=model_name, task=task, verbose=verbose)

    if save_dir is not None:
        save_model(save_dir=save_dir, model_name=model_name)

    _log_model_info(model_name, task, device, save_dir)

    if is_pytorch_model(model_name):
        model.to(device)
        logger.info(f"PyTorch model loaded on device: {device}")
        return model, device

    logger.info("CoreML model loaded (Neural Engine will be used automatically)")
    return model, None


# =============================================================================
# Drawing Functions
# =============================================================================


def draw_label(
    frame: np.ndarray,
    label: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a label with background on the frame."""
    x, y = position
    (text_width, text_height), _ = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    y_label = max(y, text_height + 10)

    # Draw background rectangle
    cv.rectangle(
        frame,
        (x, y_label - text_height - 6),
        (x + text_width + 4, y_label + 2),
        color,
        -1,
    )

    # Draw text
    cv.putText(
        frame,
        label,
        (x + 2, y_label - 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv.LINE_AA,
    )


def _draw_obb_box(
    matrix: cv.typing.MatLike,
    points: Points,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw an oriented bounding box (polygon with 4 corners)."""
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv.polylines(matrix, [pts], isClosed=True, color=color, thickness=2)
    draw_label(matrix, label, points[0], color)


def _draw_aabb_box(
    matrix: cv.typing.MatLike,
    points: Points,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw an axis-aligned bounding box (rectangle)."""
    p1, p2 = points[0], points[1]
    cv.rectangle(matrix, p1, p2, color, 2)
    draw_label(matrix, label, p1, color)


def draw_boxes(
    matrix: cv.typing.MatLike,
    boxes: BoxesPerCategory | dict[str, list[tuple[Points, float, str | None]]],
    *,
    is_obb: bool = False,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw all bounding boxes on the image.

    Args:
        matrix: The image to draw on.
        boxes: Dictionary mapping category names to list of
            ``(points, confidence)`` or ``(points, confidence, label)``.
        is_obb: Whether the boxes are oriented bounding boxes.
        color: BGR color tuple for the boxes.
    """
    draw_fn = _draw_obb_box if is_obb else _draw_aabb_box

    for category, category_boxes in boxes.items():
        for entry in category_boxes:
            points, confidence = entry[0], entry[1]
            matched_label: str | None = entry[2] if len(entry) > 2 else None
            display = matched_label if matched_label else category
            label = f"{display}: {confidence:.2f}"
            draw_fn(matrix, points, label, color)


# =============================================================================
# IoU Calculation
# =============================================================================


def _compute_obb_iou(box1: Points, box2: Points) -> float:
    """Calculate IoU for oriented bounding boxes."""
    pts1 = np.array(box1, dtype=np.float32)
    pts2 = np.array(box2, dtype=np.float32)

    ret, intersection_pts = cv.rotatedRectangleIntersection(
        cv.minAreaRect(pts1), cv.minAreaRect(pts2)
    )

    if ret == cv.INTERSECT_NONE or intersection_pts is None:
        return 0.0

    intersection_area = cv.contourArea(intersection_pts)
    area_1 = cv.contourArea(pts1)
    area_2 = cv.contourArea(pts2)
    union = area_1 + area_2 - intersection_area

    return intersection_area / union if union > 0 else 0.0


def _compute_aabb_iou(box1: Points, box2: Points) -> float:
    """Calculate IoU for axis-aligned bounding boxes."""
    (x1_1, y1_1), (x2_1, y2_1) = box1[0], box1[1]
    (x1_2, y1_2), (x2_2, y2_2) = box2[0], box2[1]

    # Intersection coordinates
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)

    # No intersection
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area_1 + area_2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou(box1: Points, box2: Points, *, is_obb: bool = False) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box points.
        box2: Second box points.
        is_obb: Whether the boxes are oriented bounding boxes.

    Returns:
        IoU score between 0.0 and 1.0.
    """
    if is_obb:
        return _compute_obb_iou(box1, box2)
    return _compute_aabb_iou(box1, box2)


# =============================================================================
# Non-Maximum Suppression
# =============================================================================


def _apply_nms_to_category(
    boxes: list[BoxWithConf],
    iou_threshold: float,
    *,
    is_obb: bool,
) -> list[BoxWithConf]:
    """Apply NMS to a single category of boxes."""
    sorted_boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    kept_boxes: list[BoxWithConf] = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        kept_boxes.append(best)

        sorted_boxes = [
            box
            for box in sorted_boxes
            if compute_iou(best[0], box[0], is_obb=is_obb) < iou_threshold
        ]

    return kept_boxes


def apply_nms(
    boxes: BoxesPerCategory,
    *,
    iou_threshold: float = 0.7,
    is_obb: bool = False,
) -> BoxesPerCategory:
    """Apply Non-Maximum Suppression to remove overlapping boxes.

    Args:
        boxes: Dictionary mapping categories to list of (points, confidence).
        iou_threshold: IoU threshold above which boxes are suppressed.
        is_obb: Whether the boxes are oriented bounding boxes.

    Returns:
        Filtered boxes after NMS.
    """
    return {
        category: _apply_nms_to_category(category_boxes, iou_threshold, is_obb=is_obb)
        for category, category_boxes in boxes.items()
    }


# =============================================================================
# Detection Extraction
# =============================================================================


def _validate_reduction_coefs(
    reduction_coefs: tuple[int | float, int | float],
) -> tuple[int, int] | None:
    """Validate and convert reduction coefficients."""
    if len(reduction_coefs) != 2:
        logger.warning("Reduction coefficients must have exactly 2 values")
        return None

    if any(coef < 1 for coef in reduction_coefs):
        logger.warning("Reduction coefficients must be >= 1")
        return None

    return (int(reduction_coefs[0]), int(reduction_coefs[1]))


def _resize_for_inference(
    matrix: np.ndarray,
    scale: tuple[int, int],
) -> np.ndarray:
    """Resize image for model inference."""
    height, width = matrix.shape[:2]
    new_size = (width // scale[0], height // scale[1])
    return cv.resize(matrix, new_size, interpolation=cv.INTER_LINEAR)


def _run_inference(
    model: YOLO,
    image: np.ndarray,
    device: str | None,
    *,
    verbose: bool,
) -> list[Any]:
    """Run model inference on image."""
    if device is not None:
        return model(image, verbose=verbose, device=device)
    return model(image, verbose=verbose)


def _extract_obb_points(
    detections: Any,
    idx: int,
    scale: tuple[int, int],
) -> Points:
    """Extract scaled corner points from OBB detection."""
    corners = detections.xyxyxyxy[idx].cpu().numpy()
    scale_x, scale_y = scale
    return [
        (int(scale_x * corners[j, 0]), int(scale_y * corners[j, 1])) for j in range(4)
    ]


def _extract_aabb_points(
    detections: Any,
    idx: int,
    scale: tuple[int, int],
) -> Points:
    """Extract scaled corner points from axis-aligned detection."""
    xyxy = detections.xyxy[idx].cpu().numpy().astype(np.int32)
    x1, y1, x2, y2 = xyxy
    scale_x, scale_y = scale
    return [
        (int(scale_x * x1), int(scale_y * y1)),
        (int(scale_x * x2), int(scale_y * y2)),
    ]


def _extract_detections(
    detections: Any,
    names: dict[int, str],
    scale: tuple[int, int],
    confidence_threshold: float,
    *,
    is_obb: bool,
) -> BoxesPerCategory:
    """Extract and filter detections from model output."""
    confs = detections.conf.cpu().numpy()
    classes = detections.cls.cpu().numpy().astype(np.int32)

    extract_fn = _extract_obb_points if is_obb else _extract_aabb_points
    boxes_per_category: BoxesPerCategory = {}

    for idx in range(len(detections)):
        conf = float(confs[idx])
        if conf < confidence_threshold:
            continue

        class_name = names[classes[idx]]
        points = extract_fn(detections, idx, scale)

        if class_name not in boxes_per_category:
            boxes_per_category[class_name] = []
        boxes_per_category[class_name].append((points, conf))

    return boxes_per_category


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def process_image(
    model: YOLO,
    matrix: cv.typing.MatLike,
    *,
    task: str = "detection",
    device: str | None = "mps",
    embedding: EmbeddingDict | None = None,
    verbose: bool = False,
    reduction_coefs: tuple[int | float, int | float] = (2, 2),
    confidence_threshold: float = 0.6,
) -> np.ndarray:
    """Process an image with the YOLO model and draw detections.

    Args:
        model: The YOLO model instance.
        matrix: The image frame to process.
        task: Task type ('detection' or 'obb').
        device: Device for inference. None for CoreML (auto Neural Engine).
        embedding: Optional embedding dictionary for object matching.
            Build one with :func:`~src.models.embedding.build_embedding_from_directory`
            or :func:`~src.models.embedding.update_embedding`.
        verbose: Whether to print verbose output.
        reduction_coefs: Factors to reduce image size for faster inference.
        confidence_threshold: Minimum confidence for detections.

    Returns:
        The processed image with drawn detections.
    """
    # Validate parameters
    scale = _validate_reduction_coefs(reduction_coefs)
    if scale is None:
        return matrix

    is_obb = task == "obb"

    # Run inference
    resized = _resize_for_inference(matrix, scale)
    results = _run_inference(model, resized, device, verbose=verbose)[0]

    # Get detections based on task type
    detections = results.obb if is_obb else results.boxes

    if detections is None or len(detections) == 0:
        return matrix

    # Extract and process detections
    boxes = _extract_detections(
        detections,
        results.names,
        scale,
        confidence_threshold,
        is_obb=is_obb,
    )

    boxes = apply_nms(boxes, is_obb=is_obb)

    if embedding is not None:
        matched_boxes = match_boxes_embedding(boxes, embedding, matrix)
        draw_boxes(matrix, matched_boxes, is_obb=is_obb)
    else:
        draw_boxes(matrix, boxes, is_obb=is_obb)

    return matrix
