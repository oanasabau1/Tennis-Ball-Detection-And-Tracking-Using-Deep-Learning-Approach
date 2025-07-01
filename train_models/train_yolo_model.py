from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("D:/tennis_thesis/models/yolov5m.pt")
    # model = YOLO("D:/tennis_thesis/models/yolov8m.pt")
    results = model.train(
        data="D:/tennis_thesis/combined_tennis_ball_dataset/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        patience=25,
        device="cuda",
        project="runs/train",
        name="tennis_ball_yolov5m",
        val=True,
        save=True,
        verbose=True
    )

