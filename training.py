from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")

    results = model.train(
        data='./dataset',
        epochs=50,
        imgsz=224,
        batch=16,
        workers=2,
        device=0,
    )