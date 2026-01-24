"""Tests for `main.py`."""

from unittest.mock import patch

from src import main


@patch.object(main, "create_default_window")
@patch.object(main, "get_model")
def test_main(mock_get_model, mock_create_window) -> None:
    """Test that main loads the model and creates a window."""
    mock_get_model.return_value = "fake_model"

    main.main()

    mock_get_model.assert_called_once_with(
        model_name="./models/yolo26n.pt", task=None, device="mps"
    )
    mock_create_window.assert_called_once_with("test", "fake_model")
