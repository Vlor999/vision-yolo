"""Main function that call the models and dosplay the video."""

from src.display.video import create_default_window
from src.models.handler import get_model


def main() -> None:
    """Load the YOLO model and start the video detection window.

    This function initializes the YOLO model with the specified device,
    then creates a video window for real-time object detection.
    """
    device = "mps"
    model = get_model(model_name="./models/yolo26n.pt", task=None, device=device)
    create_default_window("test", model)


if __name__ == "__main__":
    main()
