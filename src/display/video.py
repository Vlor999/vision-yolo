"""handle the video display."""

from time import time

import cv2 as cv
from loguru import logger
from ultralytics.models import YOLO

from src.models.handler import process_image


def create_default_window(window_name: str, model: YOLO, delay: int = 20) -> None:
    """Create the default window."""
    no_key = False
    cam = cv.VideoCapture(0)
    cv.namedWindow(window_name)
    while True:
        tic = time()
        ret, frame = cam.read()
        if not ret:
            logger.warning("Frame not captured")
            break
        fliped_frame = cv.flip(frame, 1)
        key = cv.waitKey(delay=delay)
        if key == ord("q"):
            no_key = False
            logger.info(f"Quitting the window: {window_name}")
            break
        if key == ord("+"):
            delay += 1
            no_key = False
        if key == ord("-"):
            delay = max(1, delay - 1)
            no_key = False
        elif key == -1:
            if not no_key:
                logger.debug("No key pressed")
                no_key = True
        else:
            logger.info(f"Key not handled: {chr(key)}")
            no_key = False

        image = process_image(model, fliped_frame)
        tac = time()
        fps = round(1 / (tac - tic), 1)

        fps_text = f"FPS: {fps}"
        cv.putText(
            image,
            fps_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.imshow(winname=window_name, mat=image)
    cam.release()
    cv.destroyWindow(window_name)
