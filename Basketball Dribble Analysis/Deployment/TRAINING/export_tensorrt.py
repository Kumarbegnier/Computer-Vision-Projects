import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO checkpoint to TensorRT engine")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GB")
    parser.add_argument("--half", action="store_true", help="Enable FP16")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Please install ultralytics: pip install ultralytics") from exc

    args = parse_args()
    model = YOLO(args.weights)
    model.export(
        format="engine",
        imgsz=args.imgsz,
        workspace=args.workspace,
        half=args.half,
    )


if __name__ == "__main__":
    main()
