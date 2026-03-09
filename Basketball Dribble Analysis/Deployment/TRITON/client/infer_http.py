import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Triton HTTP inference client")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="basketball_yolo_trt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    try:
        import tritonclient.http as httpclient  # type: ignore
        from tritonclient.utils import np_to_triton_dtype  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install triton client: pip install tritonclient[http]") from exc

    args = parse_args()
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    frame = cv2.resize(frame, (args.size, args.size))
    batched = np.expand_dims(frame, axis=0).astype(np.uint8)

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)
    inp = httpclient.InferInput("images", batched.shape, np_to_triton_dtype(batched.dtype))
    inp.set_data_from_numpy(batched)
    out = httpclient.InferRequestedOutput("detections")
    response = client.infer(model_name=args.model, inputs=[inp], outputs=[out])
    detections = response.as_numpy("detections")
    print(detections)


if __name__ == "__main__":
    main()
