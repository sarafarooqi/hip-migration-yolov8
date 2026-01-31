from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")

r = results[0]
print("Image shape:", r.orig_shape)
print("Detections:", len(r.boxes))
