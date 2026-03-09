import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model for basketball detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO checkpoint")
    parser.add_argument("--data", default="data.yaml", help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="runs/train", help="Output project dir")
    parser.add_argument("--name", default="basketball_finetune", help="Run name")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Please install ultralytics: pip install ultralytics") from exc

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
