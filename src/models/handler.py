"""Handle all the model interaction."""

from pathlib import Path

import cv2 as cv
import numpy as np
from loguru import logger
from ultralytics.models import YOLO


def get_model(
    model_name: str | Path = "yolo26n.pt",
    *,  # You must use the name of the argument after that
    task: str | None = None,
    verbose: bool = True,
    device: str = "mps",
) -> YOLO:
    """Load a YOLO model and move it to the specified device."""
    model = YOLO(model=model_name, task=task, verbose=verbose)
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    return model


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
    cv.rectangle(
        frame,
        (x, y_label - text_height - 6),
        (x + text_width + 4, y_label + 2),
        color,
        -1,
    )
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


def draw_boxes(
    matrix: cv.typing.MatLike,
    boxes: dict[int, list[tuple[tuple[int, int], tuple[int, int], float]]],
) -> None:
    """Draw all the current boxes at once."""
    for categorie, boxes_cat in boxes.items():
        for boxe in boxes_cat:
            p1, p2, boxe_conf = boxe
            cv.rectangle(matrix, p1, p2, (0, 255, 0), 2)
            label = f"{categorie}: {boxe_conf:.2f}"
            draw_label(matrix, label, p1, (0, 255, 0))


def get_IOU(
    boxe_1: tuple[tuple[int, int], tuple[int, int]],
    boxe_2: tuple[tuple[int, int], tuple[int, int]],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    (x1_1, y1_1), (x2_1, y2_1) = boxe_1
    (x1_2, y1_2), (x2_2, y2_2) = boxe_2
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area_1 + area_2 - intersection
    return intersection / union if union > 0 else 0.0


def clean_boxes(
    boxes: dict[int, list[tuple[tuple[int, int], tuple[int, int], float]]],
    *,
    IOU_threshold: float = 0.7,
) -> dict[int, list[tuple[tuple[int, int], tuple[int, int], float]]]:
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    for categorie, boxes_cat_unsorted in boxes.items():
        boxes_cat = sorted(boxes_cat_unsorted, key=lambda x: x[2], reverse=True)
        i = 0
        while i < len(boxes_cat):
            head = boxes_cat[i][:2]
            j = i + 1
            while j < len(boxes_cat):
                curr = boxes_cat[j][:2]
                iou_score = get_IOU(head, curr)
                if iou_score >= IOU_threshold:
                    boxes_cat.pop(j)
                else:
                    j += 1
            i += 1
        boxes[categorie] = boxes_cat
    return boxes


def process_image(
    model: YOLO,
    matrix: cv.typing.MatLike,
    *,
    previous_boxes: list[tuple[int, int, int, int, int]] | None = None,
    verbose: bool = False,
    reduction_coefs: tuple[int | float, int | float] = (2, 2),
) -> np.ndarray:
    """Process an image with the YOLO model and draw detections."""
    if len(reduction_coefs) != 2 or any(
        reduction_coef < 1 for reduction_coef in reduction_coefs
    ):
        logger.warning("The reduction coef must be > 1")
        return matrix
    else:
        int_reduction_coefs = tuple(int(x) for x in reduction_coefs)

    if previous_boxes:
        previous_boxes = []

    matrix_height, matrix_width, _ = matrix.shape
    resized_matrix = cv.resize(
        matrix,
        (
            matrix_width // int_reduction_coefs[0],
            matrix_height // int_reduction_coefs[1],
        ),
        interpolation=cv.INTER_LINEAR,
    )
    pred = model(resized_matrix, verbose=verbose)
    objects = pred[0]
    names = objects.names

    boxes = objects.boxes
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(np.int32)
        xyxys = boxes.xyxy.cpu().numpy().astype(np.int32)

        scale_x, scale_y = int_reduction_coefs

        boxes_per_categorie: dict[
            int, list[tuple[tuple[int, int], tuple[int, int], float]]
        ] = {}

        for i in range(len(boxes)):
            conf = float(confs[i])
            cls = classes[i]
            class_name = names[cls]

            x1_red, y1_red, x2_red, y2_red = xyxys[i]
            x1 = int(scale_x * x1_red)
            x2 = int(scale_x * x2_red)
            y1 = int(scale_y * y1_red)
            y2 = int(scale_y * y2_red)
            if class_name not in boxes_per_categorie:
                boxes_per_categorie[class_name] = [((x1, y1), (x2, y2), conf)]
            else:
                boxes_per_categorie[class_name].append(((x1, y1), (x2, y2), conf))

        boxes_per_categorie = clean_boxes(boxes_per_categorie)

        draw_boxes(matrix=matrix, boxes=boxes_per_categorie)

    return matrix
