"""Tests for `main.py`."""

from unittest.mock import patch

from src import main


def test_parse_args_defaults() -> None:
    """Test that parse_args returns default values when no arguments provided."""
    with patch("sys.argv", ["main.py"]):
        args = main.parse_args()
        assert args.model_name == "./models/yolo26n.pt"
        assert args.device == "mps"
        assert args.save_dir == "models"
        assert args.task == "detection"
        assert args.window_name == "YOLO - Object Detection"


def test_parse_args_custom_values() -> None:
    """Test that parse_args returns custom values when provided."""
    with patch(
        "sys.argv",
        [
            "main.py",
            "--model-name",
            "custom_model.pt",
            "--device",
            "cpu",
            "--save-dir",
            "custom_dir",
            "--task",
            "obb",
            "--window-name",
            "Custom Window",
        ],
    ):
        args = main.parse_args()
        assert args.model_name == "custom_model.pt"
        assert args.device == "cpu"
        assert args.save_dir == "custom_dir"
        assert args.task == "obb"
        assert args.window_name == "Custom Window"


@patch.object(main, "create_default_window")
@patch.object(main, "get_model")
@patch.object(main, "parse_args")
def test_main(mock_parse_args, mock_get_model, mock_create_window) -> None:
    """Test that main loads the model and creates a window."""
    mock_parse_args.return_value = type(
        "MockArgs",
        (),
        {
            "model_name": "./models/yolo26n.pt",
            "device": "mps",
            "save_dir": "models",
            "task": "detection",
            "window_name": "YOLO - Object Detection",
        },
    )()
    mock_get_model.return_value = ("fake_model", "mps")

    main.main()

    mock_get_model.assert_called_once_with(
        model_name="./models/yolo26n.pt",
        task="detection",
        device="mps",
        save_dir="models",
    )
    mock_create_window.assert_called_once_with(
        "YOLO - Object Detection",
        "fake_model",
        device="mps",
        model_name="./models/yolo26n.pt",
        save_dir="models",
        task="detection",
    )
