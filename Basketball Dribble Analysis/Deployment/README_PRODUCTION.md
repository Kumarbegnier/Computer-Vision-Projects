# Production Pipeline

This project now supports the following runtime stack:
- Fine-tuned YOLO detector
- ByteTrack / BoT-SORT / DeepSORT tracker options
- Kalman smoothing + custom bounce logic
- TensorRT export path
- Triton serving path
- DeepStream 8 pipeline templates

## 1) Fine-tune YOLO

```bash
cd "Basketball Dribble Analysis/Deployment/TRAINING"
python train_finetune.py --model yolov8n.pt --data data.yaml --epochs 100 --imgsz 640
```

Use your own dataset layout under `dataset_yolo/images/{train,val,test}` and labels in YOLO format.

## 2) Export TensorRT Engine

```bash
cd "Basketball Dribble Analysis/Deployment/TRAINING"
python export_tensorrt.py --weights runs/train/basketball_finetune/weights/best.pt --imgsz 640 --half
```

Copy engine to:
- `Deployment/TRITON/model_repository/basketball_yolo_trt/1/model.plan`

## 3) Run Triton

```bash
tritonserver --model-repository "Basketball Dribble Analysis/Deployment/TRITON/model_repository"
```

## 4) Run Inference (Project CLI)

Classic detector:
```bash
python "Basketball Dribble Analysis/Code/Code.py" --detector-backend classic --tracker-backend kalman
```

YOLO + ByteTrack:
```bash
python "Basketball Dribble Analysis/Code/Code.py" --detector-backend yolo --tracker-backend bytetrack --yolo-model path/to/best.pt
```

YOLO + BoT-SORT:
```bash
python "Basketball Dribble Analysis/Code/Code.py" --detector-backend yolo --tracker-backend botsort --yolo-model path/to/best.pt
```

Triton detector:
```bash
python "Basketball Dribble Analysis/Code/Code.py" --detector-backend triton --tracker-backend kalman --triton-url localhost:8000 --triton-model-name basketball_yolo_trt
```

## 5) API / Streamlit knobs

`api.py` and `streamlit_app.py` expose:
- `detector_backend`: `classic | yolo | triton`
- `tracker_backend`: `kalman | bytetrack | botsort | deepsort`
- `yolo_model`
- `triton_url`
- `triton_model_name`

## 6) DeepStream 8

Template files:
- `Deployment/DEEPSTREAM/deepstream_app_config.txt`
- `Deployment/DEEPSTREAM/config_infer_primary_yolo.txt`

Update model paths (`onnx-file`, `model-engine-file`, `labelfile-path`) before running.

## Notes

- Missing optional dependencies gracefully fall back to `classic + kalman`.
- For full stack, install:
  - `ultralytics`
  - `deep-sort-realtime`
  - `tritonclient[http]`
  - `supervision` (optional for external ByteTrack path)
