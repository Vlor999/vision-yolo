"""Tests for `models/handler.py`."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.models import handler


@patch.object(handler, "logger")
@patch.object(handler, "YOLO")
def test_get_model(mock_yolo_class, mock_logger) -> None:
    """Test that get_model loads and configures the model correctly."""
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model

    result = handler.get_model(
        model_name="test_model.pt", task="detect", verbose=False, device="cpu"
    )

    mock_yolo_class.assert_called_once_with(
        model="test_model.pt", task="detect", verbose=False
    )
    mock_model.to.assert_called_once_with("cpu")
    mock_logger.info.assert_called_once()
    assert result == mock_model


@patch.object(handler.cv, "putText")
@patch.object(handler.cv, "rectangle")
@patch.object(handler.cv, "getTextSize")
def test_draw_label(mock_get_text_size, mock_rectangle, mock_put_text) -> None:
    """Test that draw_label draws text with background correctly."""
    mock_get_text_size.return_value = ((50, 10), 2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    handler.draw_label(frame, "test", (10, 20), (0, 255, 0))

    mock_get_text_size.assert_called_once()
    mock_rectangle.assert_called_once()
    mock_put_text.assert_called_once()


@patch.object(handler, "draw_label")
@patch.object(handler.cv, "rectangle")
def test_process_image(mock_rectangle, mock_draw_label) -> None:
    """Test that process_image processes detections and draws boxes."""
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.conf.detach.return_value.cpu.return_value.numpy.return_value = [0.95]
    mock_box.cls.detach.return_value.cpu.return_value.numpy.return_value = [0]
    mock_box.xyxy.detach.return_value.cpu.return_value.numpy.return_value.astype.return_value = [
        [10, 20, 100, 200]
    ]

    mock_boxes = MagicMock()
    mock_boxes.__iter__ = lambda self: iter([mock_box])
    mock_boxes.__len__ = lambda self: 1

    mock_result = MagicMock()
    mock_result.names = {0: "person"}
    mock_result.boxes = mock_boxes
    mock_model.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = handler.process_image(mock_model, frame)

    mock_model.assert_called_once()
    mock_rectangle.assert_called_once()
    mock_draw_label.assert_called_once()
    assert result is frame


@patch.object(handler, "draw_label")
@patch.object(handler.cv, "rectangle")
def test_process_image_no_detections(mock_rectangle, mock_draw_label) -> None:
    """Test that process_image handles no detections correctly."""
    mock_model = MagicMock()

    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: 0

    mock_result = MagicMock()
    mock_result.names = {0: "person"}
    mock_result.boxes = mock_boxes
    mock_model.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = handler.process_image(mock_model, frame)

    mock_model.assert_called_once()
    mock_rectangle.assert_not_called()
    mock_draw_label.assert_not_called()
    assert result is frame
