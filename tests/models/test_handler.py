"""Tests for `models/handler.py`."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.models import handler

# =============================================================================
# Model Loading Tests
# =============================================================================


@patch.object(handler, "_log_model_info")
@patch.object(handler, "YOLO")
def test_get_model(mock_yolo_class, mock_log_model_info) -> None:
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
    mock_log_model_info.assert_called_once()
    assert result == (mock_model, "cpu")


def test_is_pytorch_model() -> None:
    """Test is_pytorch_model function."""
    assert handler.is_pytorch_model("model.pt") is True
    assert handler.is_pytorch_model("model.pth") is True
    assert handler.is_pytorch_model("model.mlpackage") is False
    assert handler.is_pytorch_model("model.onnx") is False


@patch.object(handler, "_log_model_info")
@patch.object(handler, "YOLO")
def test_get_model_coreml(mock_yolo_class, mock_log_model_info) -> None:
    """Test get_model with CoreML model."""
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model

    result = handler.get_model(
        model_name="test_model.mlpackage", task="detect", verbose=False, device="cpu"
    )

    mock_yolo_class.assert_called_once_with(
        model="test_model.mlpackage", task="detect", verbose=False
    )
    mock_model.to.assert_not_called()  # CoreML models don't use .to()
    mock_log_model_info.assert_called_once()
    assert result == (mock_model, None)


# =============================================================================
# Drawing Tests
# =============================================================================


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
def test_draw_boxes_aabb(mock_rectangle, mock_draw_label) -> None:
    """Test that draw_boxes draws axis-aligned boxes correctly."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = {
        "person": [([(10, 20), (100, 200)], 0.95)],
        "car": [([(50, 60), (150, 250)], 0.8)],
    }

    handler.draw_boxes(frame, boxes, is_obb=False)

    assert mock_rectangle.call_count == 2
    assert mock_draw_label.call_count == 2


@patch.object(handler, "draw_label")
@patch.object(handler.cv, "polylines")
def test_draw_boxes_obb(mock_polylines, mock_draw_label) -> None:
    """Test that draw_boxes draws oriented boxes correctly."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = {
        "plane": [([(10, 20), (100, 20), (100, 80), (10, 80)], 0.95)],
    }

    handler.draw_boxes(frame, boxes, is_obb=True)

    assert mock_polylines.call_count == 1
    assert mock_draw_label.call_count == 1


# =============================================================================
# IoU Calculation Tests
# =============================================================================


def test_compute_iou_no_overlap() -> None:
    """Test IoU when boxes don't overlap."""
    box1 = [(0, 0), (10, 10)]
    box2 = [(20, 20), (30, 30)]

    result = handler.compute_iou(box1, box2, is_obb=False)

    assert result == 0.0


def test_compute_iou_full_overlap() -> None:
    """Test IoU when boxes are identical."""
    box1 = [(0, 0), (10, 10)]
    box2 = [(0, 0), (10, 10)]

    result = handler.compute_iou(box1, box2, is_obb=False)

    assert result == 1.0


def test_compute_iou_partial_overlap() -> None:
    """Test IoU when boxes partially overlap."""
    box1 = [(0, 0), (10, 10)]
    box2 = [(5, 5), (15, 15)]

    result = handler.compute_iou(box1, box2, is_obb=False)

    # Intersection: 5x5=25, Union: 100+100-25=175
    assert abs(result - 25 / 175) < 0.001


def test_compute_iou_zero_area() -> None:
    """Test IoU when a box has zero area."""
    box1 = [(0, 0), (0, 0)]
    box2 = [(0, 0), (10, 10)]

    result = handler.compute_iou(box1, box2, is_obb=False)

    assert result == 0.0


def test_compute_iou_obb() -> None:
    """Test IoU for oriented bounding boxes."""
    # Two identical squares
    box1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
    box2 = [(0, 0), (10, 0), (10, 10), (0, 10)]

    result = handler.compute_iou(box1, box2, is_obb=True)

    assert result == 1.0


def test_compute_iou_obb_no_overlap() -> None:
    """Test IoU for non-overlapping oriented bounding boxes."""
    box1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
    box2 = [(50, 50), (60, 50), (60, 60), (50, 60)]

    result = handler.compute_iou(box1, box2, is_obb=True)

    assert result == 0.0


# =============================================================================
# NMS Tests
# =============================================================================


def test_apply_nms_removes_duplicates() -> None:
    """Test that apply_nms removes overlapping boxes with lower confidence."""
    boxes = {
        "person": [
            ([(0, 0), (10, 10)], 0.9),
            ([(1, 1), (11, 11)], 0.7),  # Overlaps significantly with first
            ([(100, 100), (110, 110)], 0.8),  # Doesn't overlap
        ]
    }

    result = handler.apply_nms(boxes, iou_threshold=0.5)

    assert len(result["person"]) == 2
    assert result["person"][0][1] == 0.9  # Highest confidence kept
    assert result["person"][1][1] == 0.8  # Non-overlapping kept


def test_apply_nms_keeps_all_non_overlapping() -> None:
    """Test that apply_nms keeps all non-overlapping boxes."""
    boxes = {
        "car": [
            ([(0, 0), (10, 10)], 0.9),
            ([(50, 50), (60, 60)], 0.8),
            ([(100, 100), (110, 110)], 0.7),
        ]
    }

    result = handler.apply_nms(boxes, iou_threshold=0.5)

    assert len(result["car"]) == 3


def test_apply_nms_empty_input() -> None:
    """Test apply_nms with empty input."""
    boxes = {}

    result = handler.apply_nms(boxes, iou_threshold=0.5)

    assert result == {}


def test_apply_nms_single_box() -> None:
    """Test apply_nms with single box."""
    boxes = {"person": [([(0, 0), (10, 10)], 0.9)]}

    result = handler.apply_nms(boxes, iou_threshold=0.5)

    assert len(result["person"]) == 1
    assert result["person"][0][1] == 0.9


# =============================================================================
# Process Image Tests
# =============================================================================


@patch.object(handler, "draw_boxes")
@patch.object(handler, "apply_nms")
@patch.object(handler.cv, "resize")
def test_process_image(mock_resize, mock_apply_nms, mock_draw_boxes) -> None:
    """Test that process_image processes detections and draws boxes."""
    mock_model = MagicMock()

    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: 1
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
    mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array(
        [0]
    )
    # Mock xyxy to return proper shape for indexing
    mock_xyxy = MagicMock()
    mock_xyxy.__getitem__ = lambda self, idx: MagicMock(
        cpu=lambda: MagicMock(
            numpy=lambda: MagicMock(astype=lambda dtype: np.array([10, 20, 100, 200]))
        )
    )
    mock_boxes.xyxy = mock_xyxy

    mock_result = MagicMock()
    mock_result.names = {0: "person"}
    mock_result.boxes = mock_boxes
    mock_model.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_resize.return_value = frame
    mock_apply_nms.return_value = {"person": [([(20, 40), (200, 400)], 0.95)]}

    result = handler.process_image(mock_model, frame)

    mock_model.assert_called_once()
    mock_apply_nms.assert_called_once()
    mock_draw_boxes.assert_called_once()
    assert result is frame


@patch.object(handler, "draw_boxes")
@patch.object(handler.cv, "resize")
def test_process_image_no_detections(mock_resize, mock_draw_boxes) -> None:
    """Test that process_image handles no detections correctly."""
    mock_model = MagicMock()

    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: 0

    mock_result = MagicMock()
    mock_result.names = {0: "person"}
    mock_result.boxes = mock_boxes
    mock_model.return_value = [mock_result]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_resize.return_value = frame

    result = handler.process_image(mock_model, frame)

    mock_model.assert_called_once()
    mock_draw_boxes.assert_not_called()
    assert result is frame


@patch.object(handler, "logger")
def test_process_image_invalid_reduction_coefs(mock_logger) -> None:
    """Test that process_image warns on invalid reduction coefficients."""
    mock_model = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = handler.process_image(mock_model, frame, reduction_coefs=(0.5, 2))

    mock_logger.warning.assert_called_once()
    mock_model.assert_not_called()
    assert result is frame


@patch.object(handler, "logger")
def test_process_image_wrong_reduction_coefs_length(mock_logger) -> None:
    """Test that process_image warns on wrong reduction coefficients length."""
    mock_model = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = handler.process_image(mock_model, frame, reduction_coefs=(2,))

    mock_logger.warning.assert_called_once()
    mock_model.assert_not_called()
    assert result is frame


@patch.object(handler, "_log_model_info")
@patch("shutil.move")
@patch("os.path.exists")
def test_get_model_with_save_dir(mock_exists, mock_move, mock_log_model_info) -> None:
    """Test get_model with save_dir parameter."""
    mock_exists.return_value = True

    with (
        patch.object(handler, "YOLO") as mock_yolo_class,
        patch("builtins.open", create=True),
    ):
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        result = handler.get_model(
            model_name="test_model.pt",
            task="detect",
            verbose=False,
            device="cpu",
            save_dir="./saved_models",
        )

        mock_yolo_class.assert_called_once_with(
            model="test_model.pt", task="detect", verbose=False
        )
        mock_model.to.assert_called_once_with("cpu")
        # mock_move should be called if the conditions are met
        if mock_move.called:
            mock_move.assert_called_once()
        mock_log_model_info.assert_called()
        assert result == (mock_model, "cpu")
