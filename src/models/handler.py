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


def process_image(model: YOLO, matrix: np.ndarray) -> np.ndarray:
    """Process an image with the YOLO model and draw detections."""
    pred = model(matrix, verbose=False)
    objects = pred[0]
    names = objects.names

    if objects.boxes is not None and len(objects.boxes) > 0:
        for box in objects.boxes:
            conf = float(box.conf.detach().cpu().numpy()[0])
            cls = int(box.cls.detach().cpu().numpy()[0])
            class_name = names[cls]

            xyxy = box.xyxy.detach().cpu().numpy().astype(np.int32)
            x1, y1, x2, y2 = xyxy[0]

            cv.rectangle(matrix, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            draw_label(matrix, label, (x1, y1), (0, 255, 0))

    return matrix
