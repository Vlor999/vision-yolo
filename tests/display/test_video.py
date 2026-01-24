"""Tests for `display/video.py`."""

from unittest.mock import MagicMock, patch

from src.display import video


@patch.object(video.cv, "destroyWindow")
@patch.object(video.cv, "imshow")
@patch.object(video.cv, "putText")
@patch.object(video, "process_image")
@patch.object(video.cv, "waitKey")
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window(
    mock_capture,
    mock_named_window,
    mock_resize,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that create_default_window sets up camera and handles quit key."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    mock_cam.get.return_value = 640
    mock_cam.read.return_value = (True, MagicMock())
    mock_wait_key.return_value = ord("q")
    mock_resize.return_value = MagicMock()
    mock_process_image.return_value = MagicMock()

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
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_key_plus(
    mock_capture,
    mock_named_window,
    mock_resize,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '+' increases delay then 'q' quits."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    mock_cam.get.return_value = 640
    mock_cam.read.return_value = (True, MagicMock())
    # First call returns '+', second call returns 'q' to exit
    mock_wait_key.side_effect = [ord("+"), ord("q")]
    mock_resize.return_value = MagicMock()
    mock_process_image.return_value = MagicMock()

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
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_key_minus(
    mock_capture,
    mock_named_window,
    mock_resize,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '-' decreases delay then 'q' quits."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    mock_cam.get.return_value = 640
    mock_cam.read.return_value = (True, MagicMock())
    # First call returns '-', second call returns 'q' to exit
    mock_wait_key.side_effect = [ord("-"), ord("q")]
    mock_resize.return_value = MagicMock()
    mock_process_image.return_value = MagicMock()

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
@patch.object(video.cv, "resize")
@patch.object(video.cv, "namedWindow")
@patch.object(video.cv, "VideoCapture")
def test_create_default_window_key_minus_one(
    mock_capture,
    mock_named_window,
    mock_resize,
    mock_wait_key,
    mock_process_image,
    mock_put_text,
    mock_imshow,
    mock_destroy_window,
) -> None:
    """Test that pressing '-' decreases delay then 'q' quits."""
    mock_cam = MagicMock()
    mock_capture.return_value = mock_cam
    mock_cam.get.return_value = 640
    mock_cam.read.return_value = (True, MagicMock())
    # First call returns '-', second call returns 'q' to exit
    mock_wait_key.side_effect = [-1, ord("q")]
    mock_resize.return_value = MagicMock()
    mock_process_image.return_value = MagicMock()

    mock_model = MagicMock()
    video.create_default_window("test_window", mock_model)

    mock_capture.assert_called_once_with(0)
    mock_named_window.assert_called_once_with("test_window")
    mock_cam.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")
