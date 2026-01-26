"""Tests for `display/video.py`."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.display import video


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that create_default_window sets up camera and handles quit key."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, fake_frame)
    mock_flip.return_value = fake_frame
    mock_wait_key.return_value = ord("q")
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_named_window.assert_called_once_with("test_window")
    mock_flip.assert_called_once()
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_key_plus(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '+' increases delay then 'q' quits."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, fake_frame)
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("+"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_named_window.assert_called_once_with("test_window")
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_key_minus(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '-' decreases delay then 'q' quits."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, fake_frame)
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("-"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_named_window.assert_called_once_with("test_window")
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_no_key(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that no key pressed is handled correctly."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, fake_frame)
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [-1, -1, ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_unhandled_key(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that unhandled keys are logged."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, fake_frame)
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("x"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_frame_not_captured(
    mock_capture,
    mock_named_window,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that frame not captured breaks the loop."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    mock_cam.read.return_value = (False, None)

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_named_window.assert_called_once_with("test_window")
    mock_flip.assert_not_called()
    mock_process_image.assert_not_called()
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")
