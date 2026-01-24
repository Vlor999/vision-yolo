"""Main function that call the models and dosplay the video."""

from src.display.video import create_default_window
from src.models.handler import get_model


def main() -> None:
    """Main function that load the model and run it."""
    device = "mps"
    model = get_model(model_name="./models/yolo26n.pt", task=None, device=device)
    create_default_window("test", model)


if __name__ == "__main__":
    main()
