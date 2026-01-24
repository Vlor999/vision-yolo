"""handle the video display."""

from threading import Thread
from time import time
from typing import Any

import cv2 as cv
import numpy as np
from loguru import logger
from ultralytics.models import YOLO

from src.models.handler import process_image


class VideoStream:
    """Threaded video capture class for improved performance."""

    def __init__(self, index: int = 0):
        """Initialize the video stream.

        Args:
            index: Camera device index (default: 0).
        """
        self.cap = cv.VideoCapture(index=index)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self) -> "VideoStream":
        """Start the video stream thread.

        Returns:
            Self for method chaining.
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self) -> None:
        """Update loop for reading frames in a separate thread."""
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.cap.read()

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the most recent frame.

        Returns:
            Tuple of (success, frame) where success is boolean and frame is numpy array.
        """
        return self.ret, self.frame

    def stop(self) -> None:
        """Stop the video stream thread."""
        self.stopped = True

    def get(self, *args: Any, **kwargs: Any) -> float:
        """Get video capture properties.

        Args:
            *args: Property arguments to pass to cv.VideoCapture.get().
            **kwargs: Property keyword arguments.

        Returns:
            The requested property value.
        """
        return self.cap.get(*args, **kwargs)

    def release(self) -> None:
        """Release the video capture and stop the thread."""
        self.stop()
        self.cap.release()


def create_default_window(window_name: str, model: YOLO, delay: int = 20) -> None:
    """Create the default window for video display with object detection.

    Args:
        window_name: Name of the OpenCV window.
        model: YOLO model for object detection.
        delay: Delay between frames in milliseconds (default: 20).
    """
    no_key = False
    cam = VideoStream(index=0).start()
    frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
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
