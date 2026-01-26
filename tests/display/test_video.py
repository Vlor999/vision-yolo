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
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that create_default_window sets up camera and handles quit key."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream.read.return_value = (True, fake_frame)
    mock_stream.get.return_value = 640
    mock_resize.return_value = fake_frame
    mock_flip.return_value = fake_frame
    mock_wait_key.return_value = ord("q")
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_named_window.assert_called_once_with("test_window")
    mock_flip.assert_called_once()
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window_key_plus(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '+' increases delay then 'q' quits."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream.read.return_value = (True, fake_frame)
    mock_stream.get.return_value = 640
    mock_resize.return_value = fake_frame
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("+"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_named_window.assert_called_once_with("test_window")
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window_key_minus(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '-' decreases delay then 'q' quits."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream.read.return_value = (True, fake_frame)
    mock_stream.get.return_value = 640
    mock_resize.return_value = fake_frame
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("-"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_named_window.assert_called_once_with("test_window")
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window_no_key(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that no key pressed is handled correctly."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream.read.return_value = (True, fake_frame)
    mock_stream.get.return_value = 640
    mock_resize.return_value = fake_frame
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [-1, -1, ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window_unhandled_key(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that unhandled keys are logged."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_stream.read.return_value = (True, fake_frame)
    mock_stream.get.return_value = 640
    mock_resize.return_value = fake_frame
    mock_flip.return_value = fake_frame
    mock_wait_key.side_effect = [ord("x"), ord("q")]
    mock_process_image.return_value = fake_frame

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "flip")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video, "VideoStream")
def test_create_default_window_frame_not_captured(
    mock_video_stream_class,
    mock_named_window,
    mock_resize,
    mock_flip,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that frame not captured breaks the loop."""
    mock_stream = MagicMock()
    mock_video_stream_class.return_value.start.return_value = mock_stream
    mock_stream.read.return_value = (False, None)
    mock_stream.get.return_value = 640

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_named_window.assert_called_once_with("test_window")
    mock_flip.assert_not_called()
    mock_process_image.assert_not_called()
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")
