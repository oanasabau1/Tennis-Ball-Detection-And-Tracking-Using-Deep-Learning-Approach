from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("models/yolov8m.pt")

    results = model.train(
        data="D:\\tennis_thesis\\new_tennis_ball_dataset\\data.yaml",
        epochs=2,
        imgsz=640,
        device='cuda',
        name="tennis_ball_detection_yolov8m.pt",
        val=True,
        save=True,
        verbose=True,
    )
