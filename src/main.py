"""Main function that call the models and dosplay the video."""

from argparse import ArgumentParser, Namespace

from src.display.video import create_default_window
from src.models.handler import get_model


def parse_args() -> Namespace:
    """Parse command line arguments for YOLO object detection.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = ArgumentParser(
        prog="Vision-yolo",
        description="Launch the YOLO model for real-time object detection",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="./models/yolo26n.pt",
        help="Path to YOLO model (.pt for PyTorch, .mlpackage for CoreML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for inference: 'mps', 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save the model (default: models)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=["detection", "obb", "segmentation", "pose", "classification"],
        help="Task you want to do",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="YOLO - Object Detection",
        help="The name of the window",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Load the YOLO model and start the video detection window."""
    args = parse_args()

    model, device = get_model(
        model_name=args.model_name,
        task=args.task,
        device=args.device,
        save_dir=args.save_dir,
    )

    create_default_window(
        args.window_name,
        model,
        device=device,
        model_name=args.model_name,
        save_dir=args.save_dir,
        task=args.task,
    )


if __name__ == "__main__":
    main()
