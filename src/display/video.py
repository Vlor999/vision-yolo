"""handle the video display."""

from collections import deque
from threading import Lock, Thread
from time import sleep, time
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
        self.thread: Thread | None = None
        self.lock = Lock()

    def start(self) -> "VideoStream":
        """Start the video stream thread.

        Returns:
            Self for method chaining.
        """
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self) -> None:
        """Update loop for reading frames in a separate thread."""
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                sleep(0.001)

    def read(self) -> tuple[bool, np.ndarray]:
        """Read the most recent frame (thread-safe).

        Returns:
            Tuple of (success, frame) where success is boolean and frame is numpy array.
        """
        with self.lock:
            return (
                (self.ret, self.frame.copy())
                if self.frame is not None
                else (self.ret, self.frame)
            )

    def stop(self) -> None:
        """Stop the video stream thread."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            logger.info("Thread properly closed")

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
        if self.cap.isOpened():
            self.cap.release()
            logger.info("Camera properly closed")


def add_informations(
    image: np.ndarray,
    metrics: deque[float],
    fps_times: deque[float],
    diff_time: float,
    delay: int,
) -> np.ndarray:
    """Add informations at the top on the panel."""
    metrics.append(diff_time)
    fps_times.append(diff_time)
    avg_time = sum(fps_times) / len(fps_times)
    fps = round(1 / avg_time, 1) if avg_time > 0 else 0.0

    # Calculate metrics statistics
    avg_metric = sum(metrics) / len(metrics) * 1000 if metrics else 0
    min_metric = min(metrics) * 1000 if metrics else 0
    max_metric = max(metrics) * 1000 if metrics else 0

    fps_text = f"FPS: {fps} | Delay: {delay}ms"
    metrics_text = f"Avg: {avg_metric:.1f}ms | Min: {min_metric:.1f}ms | Max: {max_metric:.1f}ms | Frames: {len(metrics)}"

    _, nc, _ = image.shape
    panel = np.zeros(shape=(70, nc, 3), dtype=np.uint8)
    cv.putText(
        panel,
        fps_text,
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        panel,
        metrics_text,
        (10, 55),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 200),
        1,
        cv.LINE_AA,
    )

    image = np.vstack([panel, image])
    return image


def create_default_window(
    window_name: str,
    model: YOLO,
    *,
    delay: int = 20,
    fps_smoothing: int = 10,
) -> None:
    """Create the default window for video display with object detection.

    Args:
        window_name: Name of the OpenCV window.
        model: YOLO model for object detection.
        delay: Delay between frames in milliseconds (default: 20).
        fps_smoothing: Number of frames to average for FPS calculation (default: 10).
    """
    no_key = False
    cam = VideoStream(index=0).start()
    cv.namedWindow(window_name)

    fps_times: deque[float] = deque(maxlen=fps_smoothing)
    metrics: deque[float] = deque(maxlen=30 * 3600)  # 30 images par sec * 3600 = 108000

    while True:
        tic = time()
        ret, frame = cam.read()
        if not ret or frame is None:
            logger.warning("Frame not captured")
            break

        fliped_frame = cv.flip(frame, 1)
        key = cv.waitKey(delay=delay)

        if key == ord("q"):
            logger.info(f"Quitting the window: {window_name}")
            break
        elif key == ord("+"):
            delay += 1
            no_key = False
        elif key == ord("-"):
            delay = max(1, delay - 1)
            no_key = False
        elif key == -1:
            if not no_key:
                logger.debug("No key pressed")
                no_key = True
        else:
            logger.info(f"Key not handled: {chr(key)}")
            no_key = False

        image = process_image(model, fliped_frame, reduction_coefs=(5, 3))
        tac = time()

        image = add_informations(
            image=image,
            metrics=metrics,
            fps_times=fps_times,
            diff_time=tac - tic,
            delay=delay,
        )

        cv.imshow(winname=window_name, mat=image)

    cam.release()
    cv.destroyWindow(window_name)
