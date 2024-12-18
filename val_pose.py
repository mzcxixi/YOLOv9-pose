from ultralytics import YOLO

model = YOLO('runs/pose/train2/weights/best.pt')
model.val( batch=16, imgsz=640)
