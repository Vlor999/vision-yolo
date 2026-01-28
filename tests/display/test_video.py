"""Tests for `display/video.py`."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.display import video


def test_video_stream_init() -> None:
    """Test VideoStream initialization."""
    with patch.object(video.cv, "VideoCapture") as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)

        mock_videocapture.assert_called_once_with(index=0)
        assert stream.cap == mock_cap
        assert stream.stopped is False
        assert stream.thread is None


def test_video_stream_start() -> None:
    """Test VideoStream start method."""
    with (
        patch.object(video.cv, "VideoCapture") as mock_videocapture,
        patch.object(video.Thread, "start") as mock_start,
    ):
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)
        result = stream.start()

        mock_start.assert_called_once()
        assert result == stream
        assert stream.thread is not None


def test_video_stream_read() -> None:
    """Test VideoStream read method."""
    with patch.object(video.cv, "VideoCapture") as mock_videocapture:
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, mock_frame)
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)
        ret, frame = stream.read()

        assert ret is True
        assert np.array_equal(frame, mock_frame)


def test_video_stream_stop() -> None:
    """Test VideoStream stop method."""
    with patch.object(video.cv, "VideoCapture") as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)
        stream.thread = MagicMock()
        stream.thread.is_alive.return_value = True
        stream.thread.join = MagicMock()

        stream.stop()

        assert stream.stopped is True
        stream.thread.join.assert_called_once_with(timeout=1.0)


def test_video_stream_get() -> None:
    """Test VideoStream get method."""
    with patch.object(video.cv, "VideoCapture") as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.return_value = 640.0
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)
        result = stream.get(video.cv.CAP_PROP_FRAME_WIDTH)

        mock_cap.get.assert_called_once_with(video.cv.CAP_PROP_FRAME_WIDTH)
        assert result == 640.0


def test_video_stream_release() -> None:
    """Test VideoStream release method."""
    with patch.object(video.cv, "VideoCapture") as mock_videocapture:
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.isOpened.return_value = True
        mock_cap.release = MagicMock()
        mock_videocapture.return_value = mock_cap

        stream = video.VideoStream(index=0)
        stream.thread = MagicMock()
        stream.thread.is_alive.return_value = True
        stream.thread.join = MagicMock()

        stream.release()

        mock_cap.release.assert_called_once()
        stream.thread.join.assert_called_once_with(timeout=1.0)
        assert stream.stopped is True


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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

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
    video.create_default_window(
        "test_window", mock_model, model_name="test_model", save_dir="test_dir"
    )

    mock_video_stream_class.assert_called_once_with(index=0)
    mock_named_window.assert_called_once_with("test_window")
    mock_flip.assert_not_called()
    mock_process_image.assert_not_called()
    mock_stream.release.assert_called_once()
    mock_destroy_window.assert_called_once_with("test_window")


def test_add_informations() -> None:
    """Test add_informations function."""
    from collections import deque

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    metrics = deque([0.01, 0.02, 0.03], maxlen=3)
    fps_times = deque([0.01, 0.02], maxlen=2)
    diff_time = 0.015
    delay = 20

    result = video.add_informations(
        image,
        metrics,
        fps_times,
        diff_time,
        delay,
        model_info="yolov8n",
        device_info="mps",
        additional_info="Saved in: models",
    )

    # Should have added a panel at the top
    assert result.shape[0] == 200  # 100 + 100 panel (increased height)
    assert result.shape[1] == 100
    assert result.shape[2] == 3
