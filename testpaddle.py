from ultralytics import YOLO
model = YOLO("model/best-paddlekeypoint.pt")
result = model("C:/Users/chen/yolo/runs/pose/train2/weights/test6.jpg", show=True)
print(result[0].keypoints.xy)