from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v9-pose/yolov9-pose.yaml')
model.train(cfg='ultralytics/cfg/default_detection.yaml',data='ultralytics/cfg/datasets/apple_pose.yaml', epochs=700, batch=8, imgsz=640)
