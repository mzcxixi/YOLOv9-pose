from ultralytics import YOLO

model = YOLO('runs/pose/train2/weights/best.pt')
model.predict(source='valid/images', save=True)